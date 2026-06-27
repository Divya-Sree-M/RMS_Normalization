import torch

# Initialize with the exact same mathematical values
val_a = 3.14159265
val_b = 2.71828182

precisions = {
    "FP32": torch.float32,
    "FP16": torch.float16,
    "BF16": torch.bfloat16
}

print("=== PYTORCH HIGH-PRECISION STRESS TEST ===")
for name, dtype in precisions.items():
    a = torch.tensor([val_a], device="cuda", dtype=dtype)
    b = torch.tensor([val_b], device="cuda", dtype=dtype)
    
    # Calculate the exact matching formula: sqrt(A * B) + 0.12345
    c = torch.sqrt(a * b) + 0.12345
    
    # Print out multiple trailing decimals explicitly
    print(f"{name} PyTorch Value: {c.item():.7f}")
