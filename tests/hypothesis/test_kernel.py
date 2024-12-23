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
        scale = 0.1
    else:
        scale = 1.0
    return torch.randn(shape, device=device, dtype=dtype) * scale


@settings(deadline=None)
@given(shape=shape_strategy, dtype=dtype_strategy)
def test_triangle_attention_equivalence(shape, dtype) -> None:
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
        out1 = attention_reference(q, k, v, bias, mask)
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
