from functools import wraps
import torch


def gen_tensors(
    n: int,
    d: int,
    h: int,
    use_mask: bool,
    device: torch.device,
    dtype: torch.dtype,
    batch: int = 1,
    std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    k = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    v = torch.normal(
        0, std, (batch, h, n, n, d), device=device, dtype=dtype, requires_grad=True
    )
    b = torch.normal(
        0, std, (batch, h, n, n), device=device, dtype=dtype, requires_grad=True
    )
    m = (
        torch.randint(0, 2, (batch, n, n), device=device, dtype=torch.bool)
        if use_mask
        else torch.zeros((batch, n, n), device=device, dtype=torch.bool)
    )

    return q, k, v, b, m


def clone_and_clear_grad(*tensors):
    """
    Clone gradients of tensors and clear them.
    Returns a tuple of cloned gradients.
    """
    grads = tuple(t.grad.clone() if t.grad is not None else None for t in tensors)
    for t in tensors:
        t.grad = None
    return grads


def disable_tf32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        cuda, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            False,
            False,
        )
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                cuda,
                cudnn,
            )

    return wrapped


def enable_tf32(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        cuda, cudnn = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            True,
            True,
        )
        try:
            return fn(*args, **kwargs)
        finally:
            torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
                cuda,
                cudnn,
            )

    return wrapped
