import triton
import torch
from jaxtyping import Bool, Float
import triton.testing

from pairformer_kernel.kernel import triangle_attention_fwd_kernel


class _triangle_attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Float[torch.Tensor, "... h n n d"],
        key: Float[torch.Tensor, "... h n n d"],
        value: Float[torch.Tensor, "... h n n d"],
        bias: Float[torch.Tensor, "... n n h"],
        mask: Bool[torch.Tensor, "... n n"],
    ) -> Float[torch.Tensor, "... h n n d"]:
        sm_scale = query.shape[-1] ** -0.5

        # TODO: logic to flatten batch/head dims.
        h, n, _, d = query.shape

        grid = lambda args: (triton.cdiv(n, args["BLOCK_J"]), n, h)

        out = torch.zeros_like(query)
        # TODO should lse be float32 for stability?
        # lse: LogSumExp
        lse = torch.zeros((h, n, n), device=query.device, dtype=torch.float32)

        # fmt: off
        kernel = triangle_attention_fwd_kernel[grid](
            out, out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            lse, lse.stride(0), lse.stride(1), lse.stride(2),
            query, query.stride(0), query.stride(1), query.stride(2), query.stride(3),
            key, key.stride(0), key.stride(1), key.stride(2), key.stride(3),
            value, value.stride(0), value.stride(1), value.stride(2), value.stride(3),
            bias, bias.stride(0), bias.stride(1), bias.stride(2),
            mask, mask.stride(0), mask.stride(1),
            neg_inf=torch.finfo(query.dtype).min,
            sm_scale=sm_scale, N=n, DIM=d
        )

        ctx.save_for_backward(query, key, value, bias, out, lse)
        ctx.grid = grid
        ctx.sm_scale = sm_scale

        return out

    @staticmethod
    def backward(ctx, do):
        q, k, v, b, o, lse = ctx.saved_tensors

        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        db = torch.empty_like(b)

        # need to recompute

        return None, None, None, None


triangle_attention = _triangle_attention.apply
