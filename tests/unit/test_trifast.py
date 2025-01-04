import torch
import pytest
from trifast.torch import triangle_attention
from trifast.equiv import attention_reference, disable_tf32, enable_tf32
from trifast.utils import gen_tensors, clone_and_clear_grad

from tests.utils import set_seed

set_seed(1337)


def compare_values(tri, pt, ref, msg="", eps=1e-4):
    a = (tri.float() - ref.float()).abs().max().item()
    b = (pt.float() - ref.float()).abs().max().item() + eps

    # This factor of 3 is pretty arbitrary.
    assert a <= 3 * b, f"{msg} value mismatch, tri: {a:.3e}, pt: {b:.3e}"


dtype_eps = {
    torch.float16: 1e-3,
    torch.bfloat16: 1e-3,
    torch.float32: 1e-4,
}


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize(
    ("n, h, d"),
    [
        (16, 1, 16),
        (32, 1, 32),
        (64, 1, 64),
        (16, 4, 128),
        *[(n, 4, 32) for n in range(17, 200, 3)],
    ],
)
def test_values(n: int, h: int, d: int, mask: bool, dtype: torch.dtype):
    device = torch.device("cuda")
    q, k, v, b, m = gen_tensors(n, d, h, mask, device, dtype=torch.float32)
    torch.cuda.synchronize()

    o_ref = disable_tf32(attention_reference)(q, k, v, b, m)
    o_ref.sum().backward()

    dq_ref, dk_ref, dv_ref, db_ref = clone_and_clear_grad(q, k, v, b)

    o_tri = triangle_attention(q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m)
    o_tri.sum().backward()
    dq_tri, dk_tri, dv_tri, db_tri = clone_and_clear_grad(q, k, v, b)

    o_pt = enable_tf32(attention_reference)(
        q.to(dtype), k.to(dtype), v.to(dtype), b.to(dtype), m
    )
    o_pt.sum().backward()
    dq_pt, dk_pt, dv_pt, db_pt = clone_and_clear_grad(q, k, v, b)

    compare_values(o_tri, o_pt, o_ref, "o failed", eps=dtype_eps[dtype])
    compare_values(dq_tri, dq_pt, dq_ref, "dq failed", eps=dtype_eps[dtype])
    compare_values(dk_tri, dk_pt, dk_ref, "dk failed", eps=dtype_eps[dtype])
    compare_values(dv_tri, dv_pt, dv_ref, "dv failed", eps=dtype_eps[dtype])
    compare_values(db_tri, db_pt, db_ref, "db failed", eps=dtype_eps[dtype])

    torch.cuda.synchronize()


def dot(a, b):
    return torch.dot(a.float().flatten(), b.float().flatten())


def cosine_similarity(a: torch.Tensor, b: torch.Tensor, eps=1e-8) -> float:
    """
    Flatten a and b, compute the dot product over
    the product of their magnitudes, and return the scalar.
    """
    a_flat = a.flatten()
    b_flat = b.flatten()
    dot = torch.dot(a_flat, b_flat)
    norm_a = torch.norm(a_flat)
    norm_b = torch.norm(b_flat)
    return (dot / (norm_a * norm_b + eps)).item()


def compare_directions(tensor_kernel, tensor_pt, tensor_ref, msg="", threshold=0.99):
    """
    Compare directions (via dot product / cosine similarity).
    Assert that both kernel and pt are above a threshold similarity to ref.
    """
    cs_kernel_ref = cosine_similarity(tensor_kernel.float(), tensor_ref.float())
    cs_pt_ref = cosine_similarity(tensor_pt.float(), tensor_ref.float())

    assert (
        cs_kernel_ref > threshold
    ), f"{msg} kernel->ref direction mismatch: {cs_kernel_ref:.3f}"
    assert cs_pt_ref > threshold, f"{msg} pt->ref direction mismatch: {cs_pt_ref:.3f}"


def compare_relative_direction(
    tensor_kernel, tensor_pt, tensor_ref, msg="", ratio=0.99
):
    """
    Make sure kernel->ref direction alignment is close to pt->ref direction alignment.
    i.e. cos_sim(kernel, ref) >= ratio * cos_sim(pt, ref).
    """
    cs_kernel_ref = cosine_similarity(tensor_kernel.float(), tensor_ref.float())
    cs_pt_ref = cosine_similarity(tensor_pt.float(), tensor_ref.float())

    assert cs_kernel_ref >= ratio * cs_pt_ref, (
        f"{msg} kernel->ref relative direction mismatch. "
        f"Got cos_sim(kernel, ref)={cs_kernel_ref:.4f} vs. ratio * cos_sim(pt, ref)={ratio*cs_pt_ref:.4f}"
    )


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
        *[(n, 4, 32) for n in range(17, 200, 3)],
        (512, 2, 32),
    ],
)
def test_vectors(n: int, h: int, d: int, mask: bool, dtype: torch.dtype, std: float):
    device = torch.device("cuda")

    q, k, v, b, m = gen_tensors(n, d, h, mask, device, dtype=torch.float32, std=std)
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
