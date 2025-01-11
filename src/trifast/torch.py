import triton
import torch
from jaxtyping import Bool, Float
from einops import rearrange

import triton.testing

from trifast.triton import (
    _fwd,
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)


class _triangle_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Float[torch.Tensor, "b h n n d"],
        k: Float[torch.Tensor, "b h n n d"],
        v: Float[torch.Tensor, "b h n n d"],
        b: Float[torch.Tensor, "b h n n"],
        mask: Bool[torch.Tensor, "b n n"],
    ) -> Float[torch.Tensor, "b h n n d"]:
        sm_scale = q.shape[-1] ** -0.5

        bs, h, _, n, dim = q.shape

        # TODO: Should also allow flattening arbitrary batch dims.
        q = rearrange(q, "b h ... -> (b h) ...").contiguous()
        k = rearrange(k, "b h ... -> (b h) ...").contiguous()
        v = rearrange(v, "b h ... -> (b h) ...").contiguous()
        b = rearrange(b, "b h ... -> (b h) ...").contiguous()

        # e.g. batch x head
        bh = q.shape[0]

        grid = lambda x: (triton.cdiv(n, x["BLOCK_J"]), n, bh)

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


        q = rearrange(q, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        k = rearrange(k, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        v = rearrange(v, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        b = rearrange(b, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        l = rearrange(l, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        o = rearrange(o, "(b h) ... -> b h ...", h=h, b=bs).contiguous()

        ctx.save_for_backward(q, k, v, b, mask, o, l)
        ctx.grid = grid
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(
        ctx, *grad_output: Float[torch.Tensor, "b h n n d"]
    ) -> tuple[
        Float[torch.Tensor, "b h n n d"],  # dq
        Float[torch.Tensor, "b h n n d"],  # dk
        Float[torch.Tensor, "b h n n d"],  # dv
        Float[torch.Tensor, "b h n n"],  # db
        Bool[torch.Tensor, "b n n"],  # dmask
    ]:
        # There is only one gradient.
        do = grad_output[0]
        q, k, v, b, mask, o, l = ctx.saved_tensors

        bs, h, _, n, dim = q.shape

        # TODO: Should also allow flattening arbitrary batch dims.
        q = rearrange(q, "b h ... -> (b h) ...")
        k = rearrange(k, "b h ... -> (b h) ...")
        v = rearrange(v, "b h ... -> (b h) ...")
        b = rearrange(b, "b h ... -> (b h) ...")
        o = rearrange(o, "b h ... -> (b h) ...")
        l = rearrange(l, "b h ... -> (b h) ...")
        do = rearrange(do, "b h ... -> (b h) ...")

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        db = torch.zeros_like(b)
        dmask = torch.zeros_like(mask)  # Don't need grads, but torch expects a tensor

        bh = q.shape[0]

        d = torch.zeros((bh, n, n), dtype=q.dtype, device=q.device)

        q_grid = lambda x: (triton.cdiv(n, x["BLOCK_J"]), n, bh)
        # fmt: off
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
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
        )
        # fmt: on

        # Do the actual backward pass.
        grid = lambda x: (triton.cdiv(n, x["BLOCK_K"]), n, bh)
        # fmt: off
        _bwd_kv[grid](
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
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
        )
        # fmt: on

        b_grid = lambda x: (
            triton.cdiv(n, x["BLOCK_J"]),
            triton.cdiv(n, x["BLOCK_K"]),
            bh,
        )

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
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
        )
        # fmt: on

        dq = rearrange(dq, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        dk = rearrange(dk, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        dv = rearrange(dv, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        db = rearrange(db, "(b h) ... -> b h ...", h=h, b=bs).contiguous()
        return dq, dk, dv, db, dmask


triangle_attention = _triangle_attention.apply
