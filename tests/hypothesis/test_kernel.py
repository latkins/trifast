import torch
import pytest
from hypothesis import given, strategies as st, settings
from trifast.torch import triangle_attention
from trifast.equiv import attention_reference


def powers_of_two(min_exponent=0, max_exponent=6):
    return st.integers(min_exponent, max_exponent).map(lambda x: 2**x)

def mixed_size_strategy(min_n, max_n, min_exponent, max_exponent):
    return st.one_of(
        powers_of_two(min_exponent=min_exponent, max_exponent=max_exponent),
        st.integers(min_value=min_n, max_value=max_n),
    )

shape_strategy = st.tuples(
    st.integers(min_value=1, max_value=4),  # h
    mixed_size_strategy(min_exponent=4, max_exponent=8, min_n=4, max_n=200),  # n
    powers_of_two(min_exponent=4, max_exponent=5),  # d
)

# Simplified dtype options
dtype_strategy = st.sampled_from([torch.float32, torch.float16, torch.bfloat16])


def random_tensor(shape, dtype, device="cuda"):
    """Generate random noise tensor with appropriate scaling."""

    scale = 1.0
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, device=device, dtype=dtype)

    # if dtype == torch.float16 or dtype == torch.bfloat16:
    #     # Smaller values to avoid overflow
    #     scale = 0.5
    return (
        torch.empty(shape, device=device, dtype=dtype)
        .normal_(mean=0.0, std=scale)
        .requires_grad_()
    )


def assert_close(a, b, msg, tol):
    diff = (a - b).abs()
    assert diff.max().item() < tol, f"{msg} max_diff: {diff.max().item():.3f}"


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

    tol = 5e-2

    torch.cuda.synchronize()
    with torch.no_grad():
        out1 = attention_reference(q, k, v, bias, mask, upcast=False)
        out2 = triangle_attention(q, k, v, bias, mask)

    torch.cuda.synchronize()

    assert_close(out1, out2, f"Failed: shape: {out1.shape}, dtype: {dtype}", tol)


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

    tol = 1e-2 if dtype == torch.float16 else 5e-2
    # atol = 1e-2
    # rtol = 0.01

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

    assert_close(
        tri_dq,
        ref_dq,
        f"""Failed:
            dtype: {dtype},
        """,
        tol,
    )

    assert_close(
        tri_dk,
        ref_dk,
        f"""Failed:
            dtype: {dtype},
        """,
        tol,
    )

    assert_close(
        tri_dv,
        ref_dv,
        f"""Failed:
            dtype: {dtype},
        """,
        tol,
    )

    assert_close(
        tri_db,
        ref_db,
        f"""Failed:
                dtype: {dtype},
            """,
        tol,
    )
