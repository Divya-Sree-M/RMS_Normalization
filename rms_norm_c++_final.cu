#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <fstream>

#define BLOCK_SIZE 256

template <typename T>
struct CustomTensor {
    T* device_data;
    int size;

    CustomTensor(int num_elements) : size(num_elements) {
        cudaMalloc(&device_data, size * sizeof(T));
    }
    ~CustomTensor() {
        cudaFree(device_data);
    }
    void copy_to_gpu(const std::vector<T>& host_data) {
        cudaMemcpy(device_data, host_data.data(), size * sizeof(T), cudaMemcpyHostToDevice);
    }
    void copy_to_cpu(std::vector<T>& host_data) {
        cudaMemcpy(host_data.data(), device_data, size * sizeof(T), cudaMemcpyDeviceToHost);
    }
};

__global__ void rmsnorm_fp32_kernel_vectorized(const float* __restrict__ X, float* __restrict__ Y, int N, float epsilon) {
    constexpr int max_warps = BLOCK_SIZE / 32;
    __shared__ float s_warp_sums[max_warps];
    __shared__ float s_inv_rms;
    int tx = threadIdx.x;
    int row_offset = blockIdx.x * N;
    const float4* x_f4 = (const float4*)(X + row_offset);
    float4* y_f4 = (float4*)(Y + row_offset);
    int f4_elements_per_row = N / 4; 

    float thread_sum_sq = 0.0f;
    for (int i = tx; i < f4_elements_per_row; i += blockDim.x) {
        float4 packed_x = x_f4[i]; 
        thread_sum_sq += packed_x.x * packed_x.x;
        thread_sum_sq += packed_x.y * packed_x.y;
        thread_sum_sq += packed_x.z * packed_x.z;
        thread_sum_sq += packed_x.w * packed_x.w;
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    int lane = tx % warpSize; 
    int wid = tx / warpSize;
    if (lane == 0) {
        s_warp_sums[wid] = thread_sum_sq;
    }
    __syncthreads(); 
    if (tx == 0) {
        float total_sum_sq = 0.0f;
        for (int i = 0; i < max_warps; ++i) {
            total_sum_sq += s_warp_sums[i];
        }
        s_inv_rms = rsqrtf((total_sum_sq / N) + epsilon);
    }
    __syncthreads(); 
    float scale = s_inv_rms;
    for (int i = tx; i < f4_elements_per_row; i += blockDim.x) {
        float4 packed_x = x_f4[i];
        float4 packed_y;

        packed_y.x = packed_x.x * scale;
        packed_y.y = packed_x.y * scale;
        packed_y.z = packed_x.z * scale;
        packed_y.w = packed_x.w * scale;
        y_f4[i] = packed_y;
    }
}

__global__ void rmsnorm_fp16_kernel(const __half* __restrict__ X, __half* __restrict__ Y, int N, float epsilon) {
    constexpr int max_warps = BLOCK_SIZE / 32;
    __shared__ float s_warp_sums[max_warps];
    __shared__ float s_inv_rms;

    int tx = threadIdx.x;
    int row_offset = blockIdx.x * N;
    const __half* x_row = X + row_offset;
    __half* y_row = Y + row_offset;
    const float4* x_f4 = (const float4*)x_row;
    int f4_elements_per_row = N / 8; 
    float thread_sum_sq = 0.0f;
    for (int i = tx; i < f4_elements_per_row; i += blockDim.x) {
        float4 packed_x = x_f4[i]; 
        
        const __half* local_x = (const __half*)&packed_x;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float val = __half2float(local_x[j]);
            thread_sum_sq += val * val;
        }
    }
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        thread_sum_sq += __shfl_down_sync(0xffffffff, thread_sum_sq, offset);
    }

    int lane = tx % warpSize; 
    int wid = tx / warpSize;
    if (lane == 0) {
        s_warp_sums[wid] = thread_sum_sq;
    }
    __syncthreads();
    if (tx == 0) {
        float total_sum_sq = 0.0f;
        for (int i = 0; i < max_warps; ++i) {
            total_sum_sq += s_warp_sums[i];
        }
        s_inv_rms = rsqrtf((total_sum_sq / N) + epsilon);
    }
    __syncthreads();
    float scale = s_inv_rms;
    float4* y_f4 = (float4*)y_row;

    for (int i = tx; i < f4_elements_per_row; i += blockDim.x) {
        float4 packed_x = x_f4[i];
        const __half* local_x = (const __half*)&packed_x;
        float4 packed_y;
        __half* local_y = (__half*)&packed_y;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float original_val = __half2float(local_x[j]);
            local_y[j] = __float2half(original_val * scale);
        }

        y_f4[i] = packed_y; 
    }
}

void cpu_rmsnorm_reference(const std::vector<float>& X, std::vector<float>& Y, int num_rows, int row_size, float epsilon) {
    for (int r = 0; r < num_rows; ++r) {
        double sum_sq = 0.0; 
        int row_offset = r * row_size;
        for (int c = 0; c < row_size; ++c) {
            double val = static_cast<double>(X[row_offset + c]);
            sum_sq += val * val;
        }
        float scale = 1.0f / std::sqrt(static_cast<float>(sum_sq / row_size) + epsilon);
        for (int c = 0; c < row_size; ++c) {
            Y[row_offset + c] = X[row_offset + c] * scale;
        }
    }
}

int main() {
    const int num_rows = 2048; 
    const int row_size = 2048; 
    const int N = num_rows * row_size; 
    const float epsilon = 1e-5f;

    std::cout << std::setprecision(7) << std::fixed;
    std::cout << "REFACTORED FUSED RMSNORM BENCHMARK (SM_86 Optimized)\n";
    std::cout << "==============================================================\n\n";

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    float ms = 0.0f;

    std::vector<float> h_X(N);
    for(int i = 0; i < N; ++i) h_X[i] = 1.123456f + (i % 5) * 0.1f;
    std::vector<float> h_C_reference(N, 0.0f);
    
    cpu_rmsnorm_reference(h_X, h_C_reference, num_rows, row_size, epsilon);
    double total_flops = 4.0 * N;
    {
        std::vector<float> h_Y_gpu(N, 0.0f);
        CustomTensor<float> tensor_X(N), tensor_Y(N);
        tensor_X.copy_to_gpu(h_X);
        rmsnorm_fp32_kernel_vectorized<<<num_rows, BLOCK_SIZE>>>(tensor_X.device_data, tensor_Y.device_data, row_size, epsilon);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        rmsnorm_fp32_kernel_vectorized<<<num_rows, BLOCK_SIZE>>>(tensor_X.device_data, tensor_Y.device_data, row_size, epsilon);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);
        tensor_Y.copy_to_cpu(h_Y_gpu);

        float max_err = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_err = std::max(max_err, std::abs(h_Y_gpu[i] - h_C_reference[i]));
        }
        double bytes_processed = 2.0 * N * sizeof(float);
        double bandwidth_gb_s = (bytes_processed / (ms / 1000.0)) / 1e9;
        double tflops = (total_flops / (ms / 1000.0)) / 1e12;
        std::cout << "--- Precision Config 1: [FP32 Single-Precision] ---\n";
        std::cout << "  Calculated Fidelity (Max Abs Error vs CPU): " << max_err << "\n";
        std::cout << "  Latency Benchmark Output Timing       : " << ms << " ms\n";
        std::cout << "  Sustained Hardware VRAM Throughput    : " << bandwidth_gb_s << " GB/s\n";
        std::cout << "  Compute Throughput Performance        : " << tflops << " TFLOPS\n\n";
        std::cout << "  First 5 Output Values (Fidelity Check):\n  [";
        for(int i = 0; i < 5; ++i) {
            std::cout << h_Y_gpu[i] << (i < 4 ? ", " : "");
        }
        std::cout << "]\n\n";
    }

    {
        std::vector<__half> h_X_f16(N);
        std::vector<__half> h_Y_gpu_f16(N);
        for(int i = 0; i < N; ++i) h_X_f16[i] = __float2half(h_X[i]);
        CustomTensor<__half> tensor_X(N), tensor_Y(N);
        tensor_X.copy_to_gpu(h_X_f16);
        rmsnorm_fp16_kernel<<<num_rows, BLOCK_SIZE>>>(tensor_X.device_data, tensor_Y.device_data, row_size, epsilon);
        cudaDeviceSynchronize();
        cudaEventRecord(start);
        rmsnorm_fp16_kernel<<<num_rows, BLOCK_SIZE>>>(tensor_X.device_data, tensor_Y.device_data, row_size, epsilon);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ms, start, stop);

        tensor_Y.copy_to_cpu(h_Y_gpu_f16);
        float max_err = 0.0f;
        for (int i = 0; i < N; ++i) {
            max_err = std::max(max_err, std::abs(__half2float(h_Y_gpu_f16[i]) - h_C_reference[i]));
        }
        double bytes_processed = 2.0 * N * sizeof(__half);
        double bandwidth_gb_s = (bytes_processed / (ms / 1000.0)) / 1e9;
        double tflops = (total_flops / (ms / 1000.0)) / 1e12;

        std::cout << "--- Precision Config 2: [FP16 Half-Precision] ---\n";
        std::cout << "  Calculated Fidelity (Max Abs Error vs CPU): " << max_err << "\n";
        std::cout << "  Latency Benchmark Output Timing       : " << ms << " ms\n";
        std::cout << "  Sustained Hardware VRAM Throughput    : " << bandwidth_gb_s << " GB/s\n";
        std::cout << "  Compute Throughput Performance        : " << tflops << " TFLOPS\n\n";
        std::cout << "  First 5 Output Values (Fidelity Check):\n  [";
        for(int i = 0; i < 5; ++i) {
            std::cout << __half2float(h_Y_gpu_f16[i]) << (i < 4 ? ", " : "");
        }
        std::cout << "]\n\n";
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
