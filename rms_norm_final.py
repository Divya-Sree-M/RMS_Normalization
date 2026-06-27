import torch
import time
import numpy as np

num_rows, row_size = 2048, 2048
N = num_rows * row_size
eps = 1e-5

print("=== PYTORCH NATIVE THROUGHPUT & PERFORMANCE PROFILER ===")

configs = [
    ("FP32", torch.float32, 4), 
    ("FP16", torch.float16, 2)
]
total_flops = 4.0 * N

for name, dtype, item_size in configs:
    base_pattern = np.array([1.123456 + (i * 0.1) for i in range(5)], dtype=np.float32)
    h_X_flat = np.resize(base_pattern, N).astype(np.float32 if dtype == torch.float32 else np.float16)
    x = torch.from_numpy(h_X_flat).reshape(num_rows, row_size).to("cuda")
    rms_layer = torch.nn.RMSNorm(row_size, eps=eps, elementwise_affine=False, dtype=dtype, device="cuda")

    _ = rms_layer(x)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    y_pytorch = rms_layer(x)
    end_event.record()
    torch.cuda.synchronize()
    ms = start_event.elapsed_time(end_event)
    
    bytes_processed = 2.0 * N * item_size
    bandwidth = (bytes_processed / (ms / 1000.0)) / 1e9
    tflops = (total_flops / (ms / 1000.0)) / 1e12
    print(f"\n{name} Configuration Results:")
    print(f"  PyTorch Latency   : {ms:.4f} ms")
    print(f"  PyTorch Bandwidth : {bandwidth:.2f} GB/s")
    print(f"  PyTorch Compute   : {tflops:.4f} TFLOPS")
    sample_vals = y_pytorch[0, :5].detach().cpu().float().tolist()
    print(f"  First 5 Output Values (Fidelity Check):\n  {sample_vals}\n")

