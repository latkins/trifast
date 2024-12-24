import torch
import pytest
from hypothesis import given, strategies as st, settings
from pairformer_kernel.torch import triangle_attention
from pairformer_kernel.equiv import attention_reference


# Simplified power of two strategy with smaller range
def powers_of_two(min_exponent=0, max_exponent=6):
    return st.integers(min_exponent, max_exponent).map(lambda x: 2**x)


# More constrained shape strategy
shape_strategy = st.tuples(
    st.integers(min_value=1, max_value=4),  # h
    powers_of_two(min_exponent=4, max_exponent=6),  # n
    powers_of_two(min_exponent=4, max_exponent=5),  # d
)

# Simplified dtype options
dtype_strategy = st.sampled_from([torch.float32, torch.float16, torch.bfloat16])


def random_tensor(shape, dtype, device="cuda"):
    """Generate random noise tensor with appropriate scaling."""

    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=dtype)

    if dtype == torch.float16 or dtype == torch.bfloat16:
        # Smaller values to avoid overflow
        scale = 0.5
    else:
        scale = 1.0
    return torch.empty(shape, device=device, dtype=dtype).normal_(mean=0.0, std=scale).requires_grad_()


@settings(deadline=None)
@given(shape=shape_strategy, dtype=dtype_strategy)
def test_triangle_attention_equivalence_forward(shape, dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    h, n, d = shape
    device = "cuda"

    # Generate random tensors directly
    q = random_tensor((h, n, n, d), dtype, device)
    k = random_tensor((h, n, n, d), dtype, device)
    v = random_tensor((h, n, n, d), dtype, device)
    bias = random_tensor((h, n, n), dtype, device)
    mask = random_tensor((n, n), dtype=torch.bool, device=device)

    atol = 1e-2
    rtol = 0.01

    torch.cuda.synchronize()
    with torch.no_grad():
        out1 = attention_reference(q, k, v, bias, mask, upcast=False)
        out2 = triangle_attention(q, k, v, bias, mask)

        diff = torch.abs(out1 - out2)

    torch.cuda.synchronize()

    torch.testing.assert_close(
        out1,
        out2,
        rtol=rtol,
        atol=atol,
        msg=f"""Failed:
        out1.shape: {out1.shape}, out2.shape: {out2.shape}, dtype: {dtype},
        max_diff: {diff.max().item():.3f},
        """,
    )


@settings(deadline=None)
@given(shape=shape_strategy, dtype=dtype_strategy)
def test_triangle_attention_backward_equivalence(shape, dtype) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    h, n, d = shape
    device = "cuda"

    # Generate random tensors directly
    q = random_tensor((h, n, n, d), dtype, device)
    k = random_tensor((h, n, n, d), dtype, device)
    v = random_tensor((h, n, n, d), dtype, device)
    bias = random_tensor((h, n, n), dtype, device)
    mask = random_tensor((n, n), dtype=torch.bool, device=device)

    mask = torch.zeros_like(mask)

    dout = torch.randn_like(q)

    atol = 1e-2
    rtol = 0.01

    torch.cuda.synchronize()

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

    torch.cuda.synchronize()

    torch.testing.assert_close(
        tri_dq,
        ref_dq,
        rtol=rtol,
        atol=atol,
        msg=f"""Failed:
            dtype: {dtype},
            max_diff: {(tri_dq - ref_dq).max().item():.3f},
        """,
    )

    torch.testing.assert_close(
        tri_dk,
        ref_dk,
        rtol=rtol,
        atol=atol,
        msg=f"""Failed:
            dtype: {dtype},
            max_diff: {(tri_dk - ref_dk).max().item():.3f},
        """,
    )

    torch.testing.assert_close(
        tri_dv,
        ref_dv,
        rtol=rtol,
        atol=atol,
        msg=f"""Failed:
            dtype: {dtype},
            max_diff: {(tri_dv - ref_dv).max().item():.3f},
        """,
    )

    torch.testing.assert_close(
        tri_db,
        ref_db,
        rtol=rtol,
        atol=atol,
        msg=f"""Failed:
                dtype: {dtype},
                max_diff: {(tri_db - ref_db).max().item():.3f},
            """,
    )
