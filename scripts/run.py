import torch
from pairformer_kernel.torch import triangle_attention
from pairformer_kernel.equiv import attention_reference


def get_tensors(n=128, d=32, h=4, device="cuda", dtype=torch.bfloat16):
    torch.manual_seed(0)
    q = (
        torch.empty((h, n, n, d), device=device, dtype=dtype)
        .normal_(mean=0.0, std=1.0)
        .requires_grad_()
    )
    k = (
        torch.empty((h, n, n, d), device=device, dtype=dtype)
        .normal_(mean=0.0, std=1.0)
        .requires_grad_()
    )
    v = (
        torch.empty((h, n, n, d), device=device, dtype=dtype)
        .normal_(mean=0.0, std=1.0)
        .requires_grad_()
    )
    bias = (
        torch.empty((h, n, n), device=device, dtype=dtype)
        .normal_(mean=0.0, std=1.0)
        .requires_grad_()
    )

    mask = torch.zeros(n, n).to(device) > 0
    mask = torch.randint(0, 2, (n, n), device=device, dtype=dtype)
    return q, k, v, bias, mask


def profile(n, d=32, h=2, device="cuda", dtype=torch.float32):
    q, k, v, bias, mask = get_tensors(n, d, h, device, dtype)
    dout = torch.randn_like(q)

    ref_out = attention_reference(q, k, v, bias, mask)
    ref_out.backward(dout)

    # copy grads from ref, set back to None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri_out = triangle_attention(q, k, v, bias, mask)
    tri_out.backward(dout)
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    atol = 1e-2
    rtol = 0.01

    print(f"out: {(tri_out - ref_out).abs().max():.3f}")
    print(f"dv: {(tri_dv - ref_dv).abs().max():.3f}")
    print(f"dk: {(tri_dk - ref_dk).abs().max():.3f}")
    print(f"dq: {(tri_dq - ref_dq).abs().max():.3f}")
    print(f"db: {(tri_db - ref_db).abs().max():.3f}")

    d = (ref_out - tri_out).abs()

    torch.set_printoptions(precision=2)

    # Show more rows and columns (adjust numbers as needed)

# for n in [16, 32, 64, 128, 256, 512]:
for _ in range(50):
    profile(n=362, h=4, d=32)
