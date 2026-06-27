# High-Performance Fused RMSNorm & Element-wise CUDA Kernels

A custom C++/CUDA implementation of Root Mean Square Normalization (RMSNorm) optimized for Ampere architectures (SM_86), featuring kernel fusion to eliminate memory-bound bottlenecks. This repository includes performance comparison benchmarks against native PyTorch (`torch.nn.functional.rms_norm`) evaluated across massive workloads exceeding **4 million tokens**.

## Architectural Overview

Standard deep learning frameworks execute layers sequentially, forcing intermediate tensor allocations back into VRAM. For an element-wise or reduction operation like RMSNorm, this creates massive memory bandwidth bottlenecks.

This project implements **Kernel Fusion**:
* **Fused Reduction**: The mean-square calculation, reciprocal square root, and element-wise scaling are fused into a single monolithic GPU kernel.
* **Stream & Event Architecture**: Memory copies (H2D/D2H) and kernel dispatches are managed via explicit **CUDA Streams** (`cudaStream_t`) to overlap data transport with compute. Performance profiling is driven synchronously via **CUDA Events** (`cudaEvent_t`) on the device queue to guarantee nanosecond-precision timing without host-side blocking penalties.

---

## Benchmark Results (SM_86 Optimized)

Evaluated on a massive workload payload (> 4,000,000 tokens), comparing the custom refactored fused CUDA implementation directly against PyTorch's native implementation.

### FP32 Single-Precision Performance

| Implementation | Latency (ms) | VRAM Throughput (GB/s) | Compute Performance (TFLOPS) |
| :--- | :--- | :--- | :--- |
| **PyTorch Native** | 0.2313 ms | 145.05 GB/s | 0.0725 TFLOPS |
| **Custom Fused CUDA** | **0.1259 ms** | **266.40 GB/s** | **0.1332 TFLOPS** |
| *Performance Gain* | *~1.83x Faster* | *+83.6% Bandwidth Efficiency* | *~1.84x Compute Utilization* |

### FP16 Half-Precision Performance

| Implementation | Latency (ms) | VRAM Throughput (GB/s) | Compute Performance (TFLOPS) |
| :--- | :--- | :--- | :--- |
| **PyTorch Native** | 0.0768 ms | 218.45 GB/s | 0.2185 TFLOPS |
| **Custom Fused CUDA** | **0.0542 ms** | **309.13 GB/s** | **0.3091 TFLOPS** |
| *Performance Gain* | *~1.41x Faster* | *+41.5% Bandwidth Efficiency* | *~1.41x Compute Utilization* |

### Numerical Fidelity & Precision Analysis
The custom CUDA implementation maintains strict mathematical alignment with standard CPU/PyTorch references, remaining well within acceptable IEEE-754 precision limits:
* **FP32 Absolute Error vs CPU**: 0.0000001 
* **FP16 Absolute Error vs CPU**: 0.0007259 (Expected precision loss due to 10-bit mantissa serialization).

---

## File Structure

* **`rms_norm.cu`**: The main C++/CUDA implementation containing the fused global reduction kernels, thread-block reductions, static shared memory setups, and the benchmarking loop.
* **`elementwise_ops.cu`**: Optimized parallel implementations for element-wise tensor additions, multiplications, and activation functions.
* **`rms_norm_final.py`**: The PyTorch profiling script utilizing `torch.cuda.Event` wrappers to capture native performance across identical token dimensions.

---

## Advanced Kernel Configurations

The execution launch parameters are tuned specifically to maximize streaming multiprocessor (SM) occupancy:

```cpp
// Kernel launch using explicit stream handles and dynamic shared memory if needed
rmsNormKernel<<<grid, block, 0, compute_stream>>>(d_input, d_output, d_weights, epsilon, elements);
```
* **Shared Memory Usage**: Minimizes off-chip VRAM transactional overhead by caching reduction tiles inside low-latency, on-chip block registers.
* **Asynchronous Streams**: Decouples computing states to hide execution overhead underneath host-to-device memory pipelines.

---

## Compilation & Execution

### Prerequisites
* NVIDIA CUDA Toolkit (v11.0 or higher recommended)
* GCC/G++ Compiler supporting C++17
* PyTorch environment with CUDA extension capabilities

### Compiling the CUDA Binary
```bash
nvcc -O3 -arch=sm_86 -std=c++17 rms_norm.cu -o rms_norm_benchmark
```

### Running the Profiles
To run the custom compiled C++ benchmark binary:
```bash
./rms_norm_benchmark
```

To run the comparative PyTorch profile:
```bash
python3 rms_norm_final.py
```
