import torch
import time

# Scale N up slightly to minimize initialization noise if you'd like
N = 1 << 20 

# Use numbers that introduce trailing floating-point precision differences
val_A = 2.5015
val_B = 4.003
# The mathematical absolute reference (FP64 / Double Precision)
ground_truth = float(val_A * val_B)

print("=== PYTORCH BENCHMARK & PRECISION RESULTS ===")

# -------------------------------------------------------------------------
# CONFIG 1: FLOAT32 PERFORMANCE & PRECISION CHECK
# -------------------------------------------------------------------------
a_f32 = torch.full((N,), val_A, device="cuda", dtype=torch.float32)
b_f32 = torch.full((N,), val_B, device="cuda", dtype=torch.float32)

torch.cuda.synchronize()
start_f32 = time.perf_counter()
c_f32_out = a_f32 * b_f32
torch.cuda.synchronize()
time_f32 = (time.perf_counter() - start_f32) * 1000 

bytes_f32 = N * 4 * 3
throughput_f32 = (bytes_f32 / 1e9) / (time_f32 / 1000.0)

# Measure precision variation from true double value
f32_calculated_val = c_f32_out[0].item()
f32_error = abs(f32_calculated_val - ground_truth)

print(f"FP32 Value: {f32_calculated_val:.7f} | Error: {f32_error:.7f} | Time: {time_f32:.4f} ms | Throughput: {throughput_f32:.2f} GB/s")

# -------------------------------------------------------------------------
# CONFIG 2: FLOAT16 PERFORMANCE & PRECISION CHECK
# -------------------------------------------------------------------------
a_f16 = torch.full((N,), val_A, device="cuda", dtype=torch.float16)
b_f16 = torch.full((N,), val_B, device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
start_f16 = time.perf_counter()
c_f16_out = a_f16 * b_f16
torch.cuda.synchronize()
time_f16 = (time.perf_counter() - start_f16) * 1000

bytes_f16 = N * 2 * 3
throughput_f16 = (bytes_f16 / 1e9) / (time_f16 / 1000.0)

# Measure precision variation from true double value
f16_calculated_val = c_f16_out[0].item()
f16_error = abs(f16_calculated_val - ground_truth)

print(f"FP16 Value: {f16_calculated_val:.7f} | Error: {f16_error:.7f} | Time: {time_f16:.4f} ms | Throughput: {throughput_f16:.2f} GB/s")
