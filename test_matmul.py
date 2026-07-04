import torch
import time
from torch.utils.cpp_extension import load_inline

# -----------------------------------------------------------------------------
# 1. Define the CUDA C++ Code as a Python String
# -----------------------------------------------------------------------------
cuda_source = """
#include <torch/extension.h>
#include <cuda_runtime.h>

#define TILE_DIM 32

__global__ void matmul_tiled_kernel(
    const float* __restrict__ A, 
    const float* __restrict__ B, 
    float* __restrict__ C, 
    int M, int N, int K
) {
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int row = blockIdx.y * TILE_DIM + threadIdx.y;
    int col = blockIdx.x * TILE_DIM + threadIdx.x;

    float c_value = 0.0f;

    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load Matrix A tile
        if (row < M && (t * TILE_DIM + threadIdx.x) < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_DIM + threadIdx.x];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // Load Matrix B tile
        if ((t * TILE_DIM + threadIdx.y) < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[(t * TILE_DIM + threadIdx.y) * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads();

        #pragma unroll 32
        for (int i = 0; i < TILE_DIM; ++i) {
            c_value += s_A[threadIdx.y][i] * s_B[i][threadIdx.x];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = c_value;
    }
}

// C++ Bridge function that PyTorch calls
torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B) {
    auto M = A.size(0);
    auto K = A.size(1);
    auto N = B.size(1);

    auto C = torch::empty({M, N}, A.options());

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matmul_tiled_kernel<<<numBlocks, threadsPerBlock>>>(
        A.data_ptr<float>(), 
        B.data_ptr<float>(), 
        C.data_ptr<float>(), 
        M, N, K
    );

    return C;
}
"""

cpp_source = """
torch::Tensor matmul_cuda_forward(torch::Tensor A, torch::Tensor B);
"""

# -----------------------------------------------------------------------------
# 2. Compile the CUDA Code on the fly
# -----------------------------------------------------------------------------
print("Compiling custom CUDA MatMul kernel... (This may take a minute first time)")
custom_matmul = load_inline(
    name="custom_matmul",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["matmul_cuda_forward"],
    verbose=False
)
print("Compilation successful!\n")

# -----------------------------------------------------------------------------
# 3. Setup Matrix Dimensions (Standard LLM Hidden Dimensions)
# -----------------------------------------------------------------------------
M, K, N = 1024, 4096, 4096  # Change these to evaluate scale
torch.manual_seed(42)

# Generate identical data pointers on the GPU
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

# -----------------------------------------------------------------------------
# 4. Check Numeric Fidelity
# -----------------------------------------------------------------------------
print("--- Checking Numeric Fidelity ---")
# Run Ground Truth (PyTorch)
Y_pytorch = torch.matmul(A, B)

# Run Custom Kernel
Y_custom = custom_matmul.matmul_cuda_forward(A, B)

# Assert correctness within floating point accuracy
is_correct = torch.allclose(Y_pytorch, Y_custom, rtol=1e-4, atol=1e-4)
print(f"Fidelity Match Status: {is_correct}")

if not is_correct:
    max_diff = torch.max(torch.abs(Y_pytorch - Y_custom))
    print(f"CRITICAL ERROR: Max difference is {max_diff.item()}")
else:
    print("Pass: Custom kernel matches PyTorch outputs accurately.")

# -----------------------------------------------------------------------------
# 5. Profile Throughput (Performance Benchmark)
# -----------------------------------------------------------------------------
print("\n--- Benchmarking Throughput ---")
NUM_WARMUP = 20
NUM_ITERS = 100

# Compute Floating Point Operations (FLOPs) for a standard matrix multiplication
# Formula: 2 * M * N * K (each element involves 1 multiplication and 1 addition)
total_flops = 2.0 * M * N * K

# Warmup both engines to discard initial GPU initialization overheads
for _ in range(NUM_WARMUP):
    _ = torch.matmul(A, B)
    _ = custom_matmul.matmul_cuda_forward(A, B)
torch.cuda.synchronize()

# Benchmark PyTorch Native (cuBLAS under the hood)
start_evt = torch.cuda.Event(enable_timing=True)
end_evt = torch.cuda.Event(enable_timing=True)

start_evt.record()
for _ in range(NUM_ITERS):
    _ = torch.matmul(A, B)
end_evt.record()
torch.cuda.synchronize()
pytorch_time = start_evt.elapsed_time(end_evt)/ NUM_ITERS / 1000.0  # seconds per iter
pytorch_tflops = (total_flops / pytorch_time) / 1e12

# Benchmark Your Tiled CUDA Kernel
start_evt.record()
for _ in range(NUM_ITERS):
    _ = custom_matmul.matmul_cuda_forward(A, B)
end_evt.record()
torch.cuda.synchronize()
custom_time = start_evt.elapsed_time(end_evt) / NUM_ITERS / 1000.0  # seconds per iter
custom_tflops = (total_flops / custom_time) / 1e12

# Output Metrics
print(f"Matrix Size: A({M}x{K}) * B({K}x{N})")
print(f"PyTorch (cuBLAS) Execution Time: {pytorch_time*1000:.3f} ms | Throughput: {pytorch_tflops:.2f} TFLOPS")
print(f"Your Tiled CUDA Execution Time : {custom_time*1000:.3f} ms | Throughput: {custom_tflops:.2f} TFLOPS")
efficiency = (pytorch_time / custom_time) * 100
print(f"Performance relative to hardware peak/cuBLAS: {efficiency:.2f}%")
