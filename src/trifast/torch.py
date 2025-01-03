import triton
import torch
from jaxtyping import Bool, Float

import triton.testing

from trifast.triton import (
    _fwd,
    _bwd_preprocess,
    _bwd_kv,
    _bwd_q,
    _bwd_b,
)


class _triangle_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: Float[torch.Tensor, "... h n n d"],
        k: Float[torch.Tensor, "... h n n d"],
        v: Float[torch.Tensor, "... h n n d"],
        b: Float[torch.Tensor, "... n n h"],
        mask: Bool[torch.Tensor, "... n n"],
    ) -> Float[torch.Tensor, "... h n n d"]:
        sm_scale = q.shape[-1] ** -0.5

        # TODO: logic to flatten batch/head dims.
        h, m, n, dim = q.shape

        grid = lambda x: (triton.cdiv(n, x["BLOCK_J"]), n, h)

        o = torch.zeros_like(q)
        l = torch.zeros((h, n, n), device=q.device, dtype=torch.float32)

        # fmt: off
        kernel = _fwd[grid](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            l, l.stride(0), l.stride(1), l.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b, b.stride(0), b.stride(1), b.stride(2),
            mask, mask.stride(0), mask.stride(1),
            neg_inf=torch.finfo(q.dtype).min,
            sm_scale=sm_scale, N=n, H=h, DIM=dim,
            # BLOCK_J=params["BLOCK_J"], BLOCK_K=params["BLOCK_K"],
            # num_warps=params["num_warps"], num_stages=params["num_stages"],
        )
        ctx.save_for_backward(q, k, v, b, mask, o, l)
        ctx.grid = grid
        ctx.sm_scale = sm_scale

        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, b, mask, o, l = ctx.saved_tensors

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        db = torch.zeros_like(b)
        dmask = torch.zeros_like(mask)

        h, m, n, dim = q.shape

        # e.g. [h,n,n]
        # we need d_{hij} = (o_{hij} \dot do_{hij}) to simplify gradient computation, so pre-compute
        d = torch.empty_like(l)
        PRE_BLOCK_J = 16
        pre_grid = (triton.cdiv(n, PRE_BLOCK_J), n, h)

        # fmt: off
        _bwd_preprocess[pre_grid](
            o, o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            d, d.stride(0), d.stride(1), d.stride(2),
            N=n, H=h, DIM=dim, BLOCK_J=PRE_BLOCK_J
        )

        # Do the actual backward pass.
        grid = lambda x: (triton.cdiv(n, x["BLOCK_K"]), n, h)
        # fmt: off
        _bwd_kv[grid](
            d, d.stride(0), d.stride(1), d.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b, b.stride(0), b.stride(1), b.stride(2),
            l, l.stride(0), l.stride(1), l.stride(2),
            mask, mask.stride(0), mask.stride(1),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dk, dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
            dv, dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
            # BLOCK_K=params["BLOCK_K"], BLOCK_J=params["BLOCK_J"],
            # num_warps=params["num_warps"], num_stages=params["num_stages"],
        )

        q_grid = lambda x: (triton.cdiv(n, x["BLOCK_J"]), n, h)
        # fmt: off
        _bwd_q[q_grid](
            d, d.stride(0), d.stride(1), d.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b, b.stride(0), b.stride(1), b.stride(2),
            l, l.stride(0), l.stride(1), l.stride(2),
            mask, mask.stride(0), mask.stride(1),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            dq, dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
            # BLOCK_J=params["BLOCK_J"], BLOCK_K=params["BLOCK_K"],
            # num_warps=params["num_warps"], num_stages=params["num_stages"],
        )

        b_grid = lambda x: (triton.cdiv(n, x['BLOCK_J']), triton.cdiv(n, x['BLOCK_K']), h)

        _bwd_b[b_grid](
            d, d.stride(0), d.stride(1), d.stride(2),
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v, v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            b, b.stride(0), b.stride(1), b.stride(2),
            l, l.stride(0), l.stride(1), l.stride(2),
            mask, mask.stride(0), mask.stride(1),
            do, do.stride(0), do.stride(1), do.stride(2), do.stride(3),
            db, db.stride(0), db.stride(1), db.stride(2),
            sm_scale=ctx.sm_scale,
            neg_inf=torch.finfo(q.dtype).min,
            H=h, N=n, DIM=dim,
            # BLOCK_J=params['BLOCK_J'], BLOCK_K=params['BLOCK_K'],
            # num_warps=params['num_warps'], num_stages=params['num_stages'],
        )


        return dq, dk, dv, db, dmask


triangle_attention = _triangle_attention.apply
