import torch
import jax

print("=== PyTorch ===")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    x = torch.randn(1024, 1024, device="cuda")
    y = x @ x
    print("torch matmul:", y.shape)

print("\n=== JAX ===")
print("jax:", jax.__version__)
print("devices:", jax.devices())
