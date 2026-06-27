#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip> // For accurate precision decimal printing
#include <cuda_runtime.h>
#include <cuda_fp16.h>

template <typename T>
struct CustomTensor {
    T* device_data;
    int size;
    CustomTensor(int num_elements) : size(num_elements) { cudaMalloc(&device_data, size * sizeof(T)); }
    ~CustomTensor() { cudaFree(device_data); }
    void copy_to_gpu(const std::vector<T>& host_data) { cudaMemcpy(device_data, host_data.data(), size * sizeof(T), cudaMemcpyHostToDevice); }
    void copy_to_cpu(std::vector<T>& host_data) { cudaMemcpy(host_data.data(), device_data, size * sizeof(T), cudaMemcpyDeviceToHost); }
};

__global__ void vectorized_op_fp32(const float* A, const float* B, float* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx < N) {
        float4 a_val = reinterpret_cast<const float4*>(A)[idx / 4];
        float4 b_val = reinterpret_cast<const float4*>(B)[idx / 4];
        float4 c_val;
        c_val.x = a_val.x * b_val.x;
        c_val.y = a_val.y * b_val.y;
        c_val.z = a_val.z * b_val.z;
        c_val.w = a_val.w * b_val.w;
        reinterpret_cast<float4*>(C)[idx / 4] = c_val;
    }
}

__global__ void vectorized_op_fp16(const __half* A, const __half* B, __half* C, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < N) {
        half2 a_val = reinterpret_cast<const half2*>(A)[idx / 2];
        half2 b_val = reinterpret_cast<const half2*>(B)[idx / 2];
        half2 c_val = __hmul2(a_val, b_val); 
        reinterpret_cast<half2*>(C)[idx / 2] = c_val;
    }
}

int main() {
    const int N = 1 << 20; 
    
    // Inputs matching the Python test configuration
    const float val_A = 2.5015f;
    const float val_B = 4.003f;
    const double ground_truth = (double)val_A * (double)val_B;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    int threads = 256;

    std::cout << std::fixed << std::setprecision(7); // Force 7 decimal points
    std::cout << "=== CUSTOM CUDA BENCHMARK & PRECISION RESULTS ===\n";

    // -------------------------------------------------------------------------
    // RUNNING CONFIG 1: FLOAT32 CONFIGURATION
    // -------------------------------------------------------------------------
    std::vector<float> h_A_f32(N, val_A);
    std::vector<float> h_B_f32(N, val_B);
    std::vector<float> h_C_f32(N, 0.0f);

    CustomTensor<float> tensor_A_f32(N);
    CustomTensor<float> tensor_B_f32(N);
    CustomTensor<float> tensor_C_f32(N);
    tensor_A_f32.copy_to_gpu(h_A_f32);
    tensor_B_f32.copy_to_gpu(h_B_f32);

    int blocks_f32 = (N / 4 + threads - 1) / threads;
    vectorized_op_fp32<<<blocks_f32, threads>>>(tensor_A_f32.device_data, tensor_B_f32.device_data, tensor_C_f32.device_data, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    vectorized_op_fp32<<<blocks_f32, threads>>>(tensor_A_f32.device_data, tensor_B_f32.device_data, tensor_C_f32.device_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    double bytes_f32 = (double)N * sizeof(float) * 3.0;
    double throughput_f32 = (bytes_f32 / 1e9) / (milliseconds / 1000.0);

    tensor_C_f32.copy_to_cpu(h_C_f32);
    double f32_error = std::abs((double)h_C_f32[0] - ground_truth);

    std::cout << "FP32 Value: " << h_C_f32[0] 
              << " | Error: " << f32_error
              << " | Time: " << milliseconds << " ms"
              << " | Throughput: " << throughput_f32 << " GB/s\n";

    // -------------------------------------------------------------------------
    // RUNNING CONFIG 2: FLOAT16 CONFIGURATION
    // -------------------------------------------------------------------------
    std::vector<__half> h_A_f16(N, __float2half(val_A));
    std::vector<__half> h_B_f16(N, __float2half(val_B));
    std::vector<__half> h_C_f16(N, __float2half(0.0f));

    CustomTensor<__half> tensor_A_f16(N);
    CustomTensor<__half> tensor_B_f16(N);
    CustomTensor<__half> tensor_C_f16(N);
    tensor_A_f16.copy_to_gpu(h_A_f16);
    tensor_B_f16.copy_to_gpu(h_B_f16);

    int blocks_f16 = (N / 2 + threads - 1) / threads;
    vectorized_op_fp16<<<blocks_f16, threads>>>(tensor_A_f16.device_data, tensor_B_f16.device_data, tensor_C_f16.device_data, N);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    vectorized_op_fp16<<<blocks_f16, threads>>>(tensor_A_f16.device_data, tensor_B_f16.device_data, tensor_C_f16.device_data, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    double bytes_f16 = (double)N * sizeof(__half) * 3.0;
    double throughput_f16 = (bytes_f16 / 1e9) / (milliseconds / 1000.0);

    tensor_C_f16.copy_to_cpu(h_C_f16);
    double f16_calculated_val = __half2float(h_C_f16[0]);
    double f16_error = std::abs(f16_calculated_val - ground_truth);

    std::cout << "FP16 Value: " << f16_calculated_val 
              << " | Error: " << f16_error
              << " | Time: " << milliseconds << " ms"
              << " | Throughput: " << throughput_f16 << " GB/s\n";

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
