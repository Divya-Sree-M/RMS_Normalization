import torch
import ctypes
import numpy as np

def verify_and_profile(vector_length: int):
    # 1. Load your compiled C++ shared library
    try:
        cpp_lib = ctypes.CDLL("./libvecadd.so")
    except OSError:
        print("Error: Could not find 'libvecadd.so'. Compile it with nvcc.")
        return

    # Set the return type explicitly to float for the C++ function
    cpp_lib.runCudaVectorAdd.restype = ctypes.c_float

    # 2. Generate random base data using PyTorch
    torch.manual_seed(42)
    tensor_A = torch.rand(vector_length, dtype=torch.float32)
    tensor_B = torch.rand(vector_length, dtype=torch.float32)
    
    # 3. Establish the ground-truth golden reference in PyTorch
    pytorch_result = tensor_A + tensor_B
    cpp_result = torch.zeros(vector_length, dtype=torch.float32)

    # 4. Extract raw physical memory address pointers
    ptr_A = ctypes.cast(tensor_A.numpy().ctypes.data, ctypes.POINTER(ctypes.c_float))
    ptr_B = ctypes.cast(tensor_B.numpy().ctypes.data, ctypes.POINTER(ctypes.c_float))
    ptr_C = ctypes.cast(cpp_result.numpy().ctypes.data, ctypes.POINTER(ctypes.c_float))

    print(f"Profiling Custom CUDA vs PyTorch Reference (Vector Length: {vector_length})...")

    # 5. Execute your C++ Pipeline and retrieve raw kernel time
    kernel_ms = cpp_lib.runCudaVectorAdd(
        ctypes.c_int(vector_length),
        ptr_A, ptr_B, ptr_C
    )

    # 6. Verify Numerical Fidelity
    fidelity_match = torch.allclose(cpp_result, pytorch_result, rtol=1e-5, atol=1e-5)

    # 7. Calculate Memory Throughput
    # Your kernel executes 3 memory operations per vector index: Read A, Read B, Write C
    # 1 float = 4 bytes
    total_bytes = vector_length * 4 * 3 
    kernel_seconds = kernel_ms / 1000.0
    throughput_gb_s = (total_bytes / 1e9) / kernel_seconds

    # --- PRINT COMBINED PROFILE RESULS ---
    print("\n" + "="*30)
    print("      BENCHMARK ANALYSIS      ")
    print("="*30)
    print(f"Numerical Fidelity Match : {fidelity_match}")
    print(f"Kernel Execution Time    : {kernel_ms:.4f} ms ({kernel_ms * 1000.0:.2f} µs)")
    print(f"Calculated Throughput    : {throughput_gb_s:.2f} GB/s")
    print("="*30)

if __name__ == "__main__":
    # Test with a massive size to saturate physical hardware cache and memory channels
    # 16,777,216 elements requires roughly 201 MB of high-speed memory streaming
    verify_and_profile(vector_length=16777216)
