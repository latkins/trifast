import triton
import torch
import triton.testing
from pairformer_kernel.torch import triangle_attention


def get_tensors(n=128, d=32, h=4, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = torch.randn(h, n, n, d).to(device, dtype=dtype)
    k = torch.randn(h, n, n, d).to(device, dtype=dtype)
    v = torch.randn(h, n, n, d).to(device, dtype=dtype)
    bias = torch.randn(h, n, n).to(device, dtype=dtype)
    mask = torch.randn(n, n).to(device) > 0
    return q, k, v, bias, mask


def profile(n, d=32, h=2, device="cuda", dtype=torch.float32):
    q, k, v, bias, mask = get_tensors(n, d, h, device, dtype)

    y = triangle_attention(q, k, v, bias, mask)
    print(y.shape)




#for n in [16, 32, 64, 128, 256, 512]:
profile(n=32, h=4, d=32)
