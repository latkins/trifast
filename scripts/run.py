import torch
from trifast.torch import triangle_attention
from trifast.equiv import attention_reference, triangle_attention_simple
from trifast.utils import gen_tensors, enable_tf32, disable_tf32


def profile(n, d, h, scale=1.0, device=torch.device("cuda"), dtype=torch.float32):
    q, k, v, bias, mask = gen_tensors(n, d, h, use_mask=False, device=device, dtype=dtype, std=scale, batch=3)

    ref_out = disable_tf32(attention_reference)(
        q.to(torch.float64),
        k.to(torch.float64),
        v.to(torch.float64),
        bias.to(torch.float64),
        mask,
    )
    ref_out.sum().backward()

    # copy grads from ref, set back to None
    ref_dq, q.grad = q.grad.clone(), None
    ref_dk, k.grad = k.grad.clone(), None
    ref_dv, v.grad = v.grad.clone(), None
    ref_db, bias.grad = bias.grad.clone(), None

    tri_out = triangle_attention(q, k, v, bias, mask)
    tri_out.sum().backward()
    tri_dq, q.grad = q.grad.clone(), None
    tri_dk, k.grad = k.grad.clone(), None
    tri_dv, v.grad = v.grad.clone(), None
    tri_db, bias.grad = bias.grad.clone(), None

    torch_out = enable_tf32(attention_reference)(q, k, v, bias, mask)
    torch_out.sum().backward()
    torch_dq, q.grad = q.grad.clone(), None
    torch_dk, k.grad = k.grad.clone(), None
    torch_dv, v.grad = v.grad.clone(), None
    torch_db, bias.grad = bias.grad.clone(), None

    # print(
    #     f"out tri: {(tri_out - ref_out).abs().max():.3f}, dv: {(tri_dv - ref_dv).abs().max():.3f}, dk: {(tri_dk - ref_dk).abs().max():.3f}, dq: {(tri_dq - ref_dq).abs().max():.3f}, db: {(tri_db - ref_db).abs().max():.3f}"
    # )
    # print(
    #     f"out tor: {(torch_out - ref_out).abs().max():.3f}, dv: {(torch_dv - ref_dv).abs().max():.3f}, dk: {(torch_dk - ref_dk).abs().max():.3f}, dq: {(torch_dq - ref_dq).abs().max():.3f}, db: {(torch_db - ref_db).abs().max():.3f}"
    # )
    ratios = {
        "out": (
            (tri_out - ref_out).abs().max() / (torch_out - ref_out).abs().max()
        ).detach(),
        "dv": (tri_dv - ref_dv).abs().max() / (torch_dv - ref_dv).abs().max(),
        "dk": (tri_dk - ref_dk).abs().max() / (torch_dk - ref_dk).abs().max(),
        "dq": (tri_dq - ref_dq).abs().max() / (torch_dq - ref_dq).abs().max(),
        "db": (tri_db - ref_db).abs().max() / (torch_db - ref_db).abs().max(),
    }
    # print(
    #     f"out rat: {ratios['out']:.3f}, dv: {ratios['dv']:.3f}, dk: {ratios['dk']:.3f}, dq: {ratios['dq']:.3f}, db: {ratios['db']:.3f}"
    # )
    # print("======")
    return ratios

    d_dq = (tri_dq - ref_dq).abs()
    d_dk = (tri_dk - ref_dk).abs()
    d_dv = (tri_dv - ref_dv).abs()
    d_db = (tri_db - ref_db).abs()

    torch.set_printoptions(precision=2)

import logging

# logging.basicConfig(
#     level=logging.DEBUG,  # Set root logger to DEBUG to allow all levels
#     format='%(name)s - %(levelname)s - %(message)s'
# )

def run(n, d, h, scale=1.0, device=torch.device("cuda"), dtype=torch.float32):
    q, k, v, bias, mask = gen_tensors(n, d, h, use_mask=True, device=device, dtype=dtype, std=scale)

    tri_out = triangle_attention(q, k, v, bias, mask)
    tri_out.sum().backward()

#
for n in [32, 64, 128, 256, 512, 768, 1024, 2048, 4096]:
    for dtype in [torch.float32, torch.bfloat16]:
        for h in [1, 4]:
            for d in [16, 32]:
                print(n, dtype, h, d)
                run(n, d, h, dtype=dtype)

# from collections import defaultdict
#
# all_rats = defaultdict(list)

# profile(n=32, h=4, d=16, dtype=torch.bfloat16)
# for _ in range(10000):
#     # profile(n=499, h=4, d=32, scale=4.0, dtype=torch.bfloat16)
#
#     rats = profile(n=32, h=4, d=32, scale=2, dtype=torch.float32)
#
#     for k, v in rats.items():
#         all_rats[k].append(v)
#
# for k, v in all_rats.items():
#     print(k, max(v), min(v), sum(v) / len(v))


from einops import rearrange
class FakeModule(torch.nn.Module):
    def __init__(self, h: int, d: int):
        super().__init__()

        self.h = h

        self.q = torch.nn.Linear(d, h * d)
        self.k = torch.nn.Linear(d, h * d)
        self.v = torch.nn.Linear(d, h * d)
        self.b = torch.nn.Linear(d, h)

    def forward(self, x, mask):
        q = rearrange(self.q(x), "... (h d) -> ... h d", h=self.h)
        k = rearrange(self.k(x), "... (h d) -> ... h d", h=self.h)
        v = rearrange(self.v(x), "... (h d) -> ... h d", h=self.h)
        b = self.b(x)

        return triangle_attention(q, k, v, b, mask)


# def test_weight_updates(n: int=64, h: int=1, d: int=32, do_mask: bool=False, bs: int=2, dtype: torch.dtype=torch.float32):
#
#     x = torch.randn((bs, h, n, n, d), device="cuda", dtype=dtype, requires_grad=True)
#
#     model = FakeModule(h, d).to("cuda", dtype)
#     opt = torch.optim.Adam(model.parameters(), lr=1e-3)
#
#     mask = torch.randint(0, 2, (bs, n, n), device="cuda", dtype=torch.bool) if do_mask else torch.zeros((bs, n, n), device="cuda", dtype=torch.bool)
#
#     orig_params = {k: v.clone() for k, v in model.named_parameters()}
#
#     for _ in range(5):
#         out = model(x, mask)
#
#         lbl = torch.randn_like(out) * 4
#
#         opt.zero_grad()
#         loss = torch.nn.functional.mse_loss(out, lbl)
#         loss.backward()
#         opt.step()
#
#     updated_params = dict(model.named_parameters())
#
#     for k in orig_params.keys():
#         assert not torch.all(orig_params[k] == updated_params[k]), f"Parameter {k} did not update."
#
# test_weight_updates()
