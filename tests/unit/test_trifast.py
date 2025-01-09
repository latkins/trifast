import torch
import pytest
from einops import rearrange
from trifast.torch import triangle_attention
from trifast.equiv import attention_reference
from trifast.utils import gen_tensors, clone_and_clear_grad, disable_tf32, enable_tf32

from tests.utils import (
    set_seed,
    compare_directions,
    compare_relative_direction,
    dot,
    compare_values,
)

set_seed(1337)


dtype_eps = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-4,
}


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("std", [1.0, 2.0])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
        (32, 1, 32),
        (64, 1, 64),
        (16, 4, 128),
        *[(n, 4, 32) for n in range(17, 200, 1)],
        (191, 4, 32),
    ],
)
def test_values(
    n: int, h: int, d: int, mask: bool, bs: int, dtype: torch.dtype, std: float
):
    device = torch.device("cuda")
    q, k, v, b, m = gen_tensors(
        n, d, h, use_mask=mask, device=device, dtype=torch.float32, batch=bs, std=std
    )
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()

    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_kernel.sum().backward()
    dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    compare_values(o_kernel, o_pt, o_ref, "o failed", eps=dtype_eps[dtype])
    compare_values(dq_kernel, dq_pt, dq_ref, "dq failed", eps=dtype_eps[dtype])
    compare_values(dk_kernel, dk_pt, dk_ref, "dk failed", eps=dtype_eps[dtype])
    compare_values(dv_kernel, dv_pt, dv_ref, "dv failed", eps=dtype_eps[dtype])
    compare_values(db_kernel, db_pt, db_ref, "db failed", eps=dtype_eps[dtype])
    torch.cuda.synchronize()


def compare_dot(kernel_output, pytorch_output, ref_output, msg="", threshold=0.05):
    # threshold is how much worse tri can be than the pytorch version.

    # magnitude of the ref vector
    ref_magnitude = dot(ref_output, ref_output)

    # dot product of tri and pt, normed by ref magnitude
    # These are 1.0 if perfect.
    kernel_score = dot(kernel_output, ref_output) / ref_magnitude
    pt_score = dot(pytorch_output, ref_output) / ref_magnitude

    # If kernel is better than pt, that is fine (hence the negative threshold)
    error = kernel_score - pt_score

    assert (
        error >= (-1 * threshold)
    ), f"{msg} dot product mismatch: {error:.3f} tri: {kernel_score:.3f}, pt: {pt_score:.3f}"


@pytest.mark.parametrize("bs", [1, 2])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("std", [1.0, 2.0])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
        (32, 1, 32),
        (64, 1, 64),
        (16, 4, 128),
        *[(n, 4, 32) for n in range(17, 200, 1)],
    ],
)
def test_vectors(
    n: int, h: int, d: int, mask: bool, bs: int, dtype: torch.dtype, std: float
):
    device = torch.device("cuda")

    q, k, v, b, m = gen_tensors(
        n, d, h, mask, device, dtype=torch.float32, std=std, batch=bs
    )
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()

    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_kernel = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_kernel.sum().backward()
    dq_kernel, dk_kernel, dv_kernel, db_kernel = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    compare_relative_direction(o_kernel, o_pt, o_ref, "Output")
    compare_relative_direction(dq_kernel, dq_pt, dq_ref, "dQ")
    compare_relative_direction(dk_kernel, dk_pt, dk_ref, "dK")
    compare_relative_direction(dv_kernel, dv_pt, dv_ref, "dV")
    compare_relative_direction(db_kernel, db_pt, db_ref, "dB")

    compare_directions(o_kernel, o_pt, o_ref, "Output")
    compare_directions(dq_kernel, dq_pt, dq_ref, "dQ")
    compare_directions(dk_kernel, dk_pt, dk_ref, "dK")
    compare_directions(dv_kernel, dv_pt, dv_ref, "dV")
    compare_directions(db_kernel, db_pt, db_ref, "dB")

    compare_dot(o_kernel, o_pt, o_ref, "Output", threshold=0.01)
    compare_dot(dq_kernel, dq_pt, dq_ref, "dQ", threshold=0.01)
    compare_dot(dk_kernel, dk_pt, dk_ref, "dK", threshold=0.01)
    compare_dot(dv_kernel, dv_pt, dv_ref, "dV", threshold=0.01)
    compare_dot(db_kernel, db_pt, db_ref, "dB", threshold=0.01)

    torch.cuda.synchronize()


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


@pytest.mark.parametrize(
    ("n, h, d, do_mask, bs, dtype"),
    [
        (16, 1, 16, False, 1, torch.float16),
        (32, 1, 32, False, 1, torch.bfloat16),
        (64, 1, 64, False, 4, torch.float32),
    ],
)
def test_weight_updates(
    n: int, h: int, d: int, do_mask: bool, bs: int, dtype: torch.dtype
):
    x = torch.randn((bs, n, n, d), device="cuda", dtype=dtype, requires_grad=True)

    model = FakeModule(h, d).to("cuda", dtype)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    mask = (
        torch.randint(0, 2, (bs, n, n), device="cuda", dtype=torch.bool)
        if do_mask
        else torch.zeros((bs, n, n), device="cuda", dtype=torch.bool)
    )

    orig_params = {k: v.clone() for k, v in model.named_parameters()}

    for _ in range(5):
        out = model(x, mask)

        lbl = torch.randn_like(out) * 4

        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(out, lbl)
        loss.backward()
        opt.step()

    updated_params = dict(model.named_parameters())

    for k in orig_params.keys():
        assert not torch.all(
            orig_params[k] == updated_params[k]
        ), f"Parameter {k} did not update."
