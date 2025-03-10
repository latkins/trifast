import triton
import torch
from typing import List, Tuple
from einops import rearrange

import triton.testing

from trifast.triton import (
    _fwd,
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)

@torch.library.custom_op("trifast::triangle_attention", mutates_args=(), device_types=["cuda"])
def _(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    b: torch.Tensor,
    mask: torch.Tensor
) -> List[torch.Tensor]:
    """
    Forward pass of triangle attention.
    
    Args:
        q (torch.Tensor): Query tensor of shape (b, h, n, n, d).
        k (torch.Tensor): Key tensor of shape (b, h, n, n, d).
        v (torch.Tensor): Value tensor of shape (b, h, n, n, d).
        b (torch.Tensor): Bias tensor of shape (b, h, n, n).
        mask (torch.Tensor): Mask tensor of shape (b, n, n), dtype=torch.bool.

    Returns:
        o(torch.Tensor): Output tensor of shape (b, h, n, n, d).
        l(torch.Tensor): Auxiliary result (only for backward path)
    """
    sm_scale = q.shape[-1] ** -0.5
    bs, h, _, n, dim = q.shape

    q = rearrange(q, "b h ... -> (b h) ...").contiguous()
    k = rearrange(k, "b h ... -> (b h) ...").contiguous()
    v = rearrange(v, "b h ... -> (b h) ...").contiguous()
    b = rearrange(b, "b h ... -> (b h) ...").contiguous()
    mask = mask.contiguous()

    bh = q.shape[0]

    def grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)

    o = torch.zeros_like(q)
    l = torch.zeros((bh, n, n), device=q.device, dtype=torch.float32)

    # fmt: off
    _fwd[grid](
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        l, l.stride(0), l.stride(1), l.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        neg_inf=torch.finfo(q.dtype).min,
        sm_scale=sm_scale, N=n, H=h, DIM=dim,
    )
    
    o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    l = rearrange(l, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    return o, l


@torch.library.register_fake("trifast::triangle_attention")
def _(
        q: torch.Tensor,    
        k: torch.Tensor,    
        v: torch.Tensor,    
        b: torch.Tensor,    
        mask: torch.Tensor, 
) -> List[torch.Tensor]:
    bs, h, _, n, dim = q.shape
    return (
        torch.empty_like(q),
        torch.empty((bs, h, n, n), device=q.device, dtype=torch.float32)
    )

@torch.library.custom_op("trifast::triangle_attention_bwd", 
                         mutates_args=(), device_types=["cuda"])
def _(
    do: torch.Tensor,   
    o: torch.Tensor,    
    l: torch.Tensor,    
    q: torch.Tensor,    
    k: torch.Tensor,    
    v: torch.Tensor,    
    b: torch.Tensor,    
    mask: torch.Tensor,                       
) -> List[torch.Tensor]:
    """
    Backward pass of triangle attention.
    
    Args:
        do: (torch.Tensor): gradient result of forward operation, shape (b, h, n, n, d).
        o: (torch.Tensor): result of forward operation, shape (b, h, n, n, d).
        l: (torch.Tensor): result of forward operation, shape (b, h, n, n, d).
        q (torch.Tensor): Query tensor of shape (b, h, n, n, d).
        k (torch.Tensor): Key tensor of shape (b, h, n, n, d).
        v (torch.Tensor): Value tensor of shape (b, h, n, n, d).
        b (torch.Tensor): Bias tensor of shape (b, h, n, n).
        mask (torch.Tensor): Mask tensor of shape (b, n, n), dtype=torch.bool.

    Returns:
        List[torch.Tensor]: Output tensor of shape (b, h, n, n, d).
    """
    bs, h, _, n, dim = q.shape
    sm_scale = dim ** -0.5

    # Rearrange tensors to merge batch and head dimensions for Triton kernel compatibility
    q = rearrange(q, "b h ... -> (b h) ...")
    k = rearrange(k, "b h ... -> (b h) ...")
    v = rearrange(v, "b h ... -> (b h) ...")
    b = rearrange(b, "b h ... -> (b h) ...")
    o = rearrange(o.contiguous(), "b h ... -> (b h) ...")
    l = rearrange(l.contiguous(), "b h ... -> (b h) ...")
    
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    db = torch.zeros_like(b)

    bh = q.shape[0]  # Batch size combined with head dimension

    d = torch.zeros((bh, n, n), dtype=q.dtype, device=q.device)

    def q_grid(x):
        return (triton.cdiv(n, x["BLOCK_J"]), n, bh)
    
    # fmt: off
    # Backward pass for query gradients
    # NOTE: This also calculates delta for kv/b!
    _bwd_q[q_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dq, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
        )    
    # fmt: on

    def kv_grid(x):
        return (triton.cdiv(n, x["BLOCK_K"]), n, bh)

    # Backward pass for key and value gradients
    # fmt: off
    _bwd_kv[kv_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
        dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
    )
    # fmt: on
    
    def b_grid(x):
        return (
            triton.cdiv(n, x["BLOCK_J"]),
            triton.cdiv(n, x["BLOCK_K"]),
            bh,
        )
    
    # Backward pass for bias gradients
    # fmt: off
    _bwd_b[b_grid](
        d, d.stride(0), d.stride(1), d.stride(2),
        q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        b, b.stride(0), b.stride(1), b.stride(2),
        l, l.stride(0), l.stride(1), l.stride(2),
        mask, mask.stride(0), mask.stride(1), mask.stride(2),
        do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
        db, db.stride(0), db.stride(1), db.stride(2),
        sm_scale=sm_scale,
        neg_inf=torch.finfo(q.dtype).min,
        H=h, N=n, DIM=dim,
    )
    # fmt: on

    # Rearrange gradients back to original shape
    dq = rearrange(dq, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dk = rearrange(dk, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    dv = rearrange(dv, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
    db = rearrange(db, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

    # Return gradients for query, key, value, and bias
    return dq, dk, dv, db

@torch.library.register_fake("trifast::triangle_attention_bwd")
def _(
        do: torch.Tensor,
        o: torch.Tensor,
        l: torch.Tensor,
        q: torch.Tensor,   
        k: torch.Tensor,   
        v: torch.Tensor,   
        b: torch.Tensor,   
        mask: torch.Tensor,
) -> List[torch.Tensor]:
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty_like(b),
    )

# Define setup_context() function for saving intermediates during forward pass
def setup_context(ctx, inputs: tuple[torch.Tensor], output: Tuple[torch.Tensor]):
    q, k, v, b, mask = inputs
    o,l = output
    ctx.save_for_backward(q, k, v, b, mask, o, l)
    ctx.mask = mask

# Define the backward function for autograd registration
def triangle_attention_bwd(ctx, grad_output):
    """
    Backward pass for triangle attention.

    Args:
      ctx: The context object containing saved variables from setup_context().
      grad_output (torch.Tensor): Gradient of the loss w.r.t. the output tensor.

    Returns:
      Gradients w.r.t. inputs in the same order as provided in forward.
      Returns None for non-differentiable inputs like `mask`.
    """
    q, k, v, b, mask, o, l = ctx.saved_tensors
    do, _ = grad_output
    # mask = ctx.mask
    out = torch.ops.trifast.triangle_attention_bwd(do, o, l, q, k, v, b, mask)
    return *out, None


# Register autograd support for triangle_attention's forward pass
# TODO: bwd_bwd support ?
torch.library.register_autograd(
    "trifast::triangle_attention",
    triangle_attention_bwd,
    setup_context=setup_context
)

def triangle_attention(
    q: torch.Tensor,   
    k: torch.Tensor,   
    v: torch.Tensor,   
    b: torch.Tensor,   
    mask: torch.Tensor 
) -> torch.Tensor:     
    out, _ = torch.ops.trifast.triangle_attention(q, k, v, b, mask)
    return out
