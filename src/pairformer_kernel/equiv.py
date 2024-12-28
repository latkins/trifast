import torch
from jaxtyping import Bool, Float

import math
from einops import einsum, rearrange

from deepspeed.ops.deepspeed4science import DS4Sci_EvoformerAttention


def neg_inf(dtype) -> float:
    return torch.finfo(dtype).min


def triangle_attention_simple(
    q: Float[torch.Tensor, "... h n n d"],
    k: Float[torch.Tensor, "... h n n d"],
    v: Float[torch.Tensor, "... h n n d"],
    bias: Float[torch.Tensor, "... h n n"],
    mask: Bool[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "... h n n d"]:
    q = q * (q.shape[-1] ** -0.5)

    qk_dot = einsum(q, k, "... h i j d, ... h i k d -> ... h i j k")

    #                                                                                 i  j k
    qk_dot_bias = qk_dot + rearrange(bias, "... h n m -> ... h () n m")
    # This modifies qk_dot_bias in place.
    qk_dot_bias.masked_fill_(
        #                                              h  i j  k
        rearrange(mask, "... n m -> ... () n () m"),
        neg_inf(q.dtype),
    )
    a_ijk = torch.softmax(qk_dot_bias, dim=-1)

    o_ij = einsum(a_ijk, v, "... h i j k, ... h i k d -> ... h i j d")

    return o_ij


def attention_reference(
    q: Float[torch.Tensor, "... h n n d"],
    k: Float[torch.Tensor, "... h n n d"],
    v: Float[torch.Tensor, "... h n n d"],
    bias: Float[torch.Tensor, "... h n n"],
    mask: Bool[torch.Tensor, "... n n"],
    upcast: bool = True,
) -> Float[torch.Tensor, "... h n n d"]:
    if upcast:
        q, k, v, bias = [t.to(torch.float32) for t in (q, k, v, bias)]

    mask_bias = neg_inf(q.dtype) * (mask).to(q.dtype)
    sm_scale = q.shape[-1] ** -0.5

    q = rearrange(q, "... h i j d -> ... () i h j d")
    k = rearrange(k, "... h i j d -> ... () i h d j")
    v = rearrange(v, "... h i j d -> ... () i h j d")

    bias = rearrange(bias, "... h i j -> ... () () h i j")
    # mask_bias should be ... i 1 1 k e.g. broadcast over
    mask_bias = rearrange(mask_bias, "... i j -> ... i () () j")

    a = torch.matmul(q, k) * sm_scale  # ... i h j k
    a += mask_bias
    a += bias

    a = torch.softmax(a, dim=-1, dtype=torch.float32 if upcast else q.dtype)
    a_v = torch.matmul(a, v)

    o = rearrange(a_v, "... () i h j d -> ... h i j d")

    return o


def triangle_self_attention_ds4s(
    q: Float[torch.Tensor, "... h n n d"],
    k: Float[torch.Tensor, "... h n n d"],
    v: Float[torch.Tensor, "... h n n d"],
    bias: Float[torch.Tensor, "... h n n"],
    mask: Bool[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "... h n n d"]:
    mask_bias = neg_inf(q.dtype) * (mask).to(q.dtype)

    q = rearrange(q, "... h i j d -> ... () i j h d")
    k = rearrange(k, "... h i j d -> ... () i j h d")
    v = rearrange(v, "... h i j d -> ... () i j h d")

    mask_bias = rearrange(mask_bias, "... i j -> ... () i () () j")
    bias = rearrange(bias, "... h i j -> ... () () h i j")
    biases = [mask_bias, bias]

    orig_dtype = q.dtype
    if orig_dtype not in [torch.bfloat16, torch.float16]:
        o = DS4Sci_EvoformerAttention(
            q.to(dtype=torch.bfloat16),
            k.to(dtype=torch.bfloat16),
            v.to(dtype=torch.bfloat16),
            [b.to(dtype=torch.bfloat16) for b in biases],
        )
        o = o.to(dtype=orig_dtype)
    else:
        o = DS4Sci_EvoformerAttention(q, k, v, biases)

    o = rearrange(o, "... () i j h d -> ... h i j d")

    return o


def triangle_attention_block(
    query: Float[torch.Tensor, "... h n n d"],
    key: Float[torch.Tensor, "... h n n d"],
    value: Float[torch.Tensor, "... h n n d"],
    bias: Float[torch.Tensor, "... h n n"],
    mask: Bool[torch.Tensor, "... n n"],
) -> Float[torch.Tensor, "... h n n d"]:
    b_max, n, n, d = query.shape
    dtype = query.dtype

    sm_scale = query.shape[-1] ** -0.5
    query = query * sm_scale
    device = query.device

    chunk = 32

    output = torch.zeros((b_max, n, n, d), dtype=dtype, device=device)
    L = torch.zeros((b_max, n, n), dtype=dtype, device=device)

    for b in range(b_max):
        for i in range(0, n, chunk):
            i_end = min(i + chunk, n)

            for j in range(0, n, chunk):
                j_end = min(j + chunk, n)
                scores_max = torch.full(
                    (i_end - i, j_end - j),
                    float("-inf"),
                    device=device,
                    dtype=dtype,
                )
                scores_sum = torch.zeros(
                    (i_end - i, j_end - j),
                    device=device,
                    dtype=dtype,
                )
                out_block = torch.zeros(
                    (i_end - i, j_end - j, d),
                    device=device,
                    dtype=dtype,
                )

                q_block = query[b, i:i_end, j:j_end]
                for k in range(0, n, chunk):
                    k_end = min(k + chunk, n)

                    k_block = key[b, i:i_end, k:k_end]  # [Bi, Bk]
                    v_block = value[b, i:i_end, k:k_end]  # [Bi, Bk]
                    b_block = bias[b, j:j_end, k:k_end]  # [Bj, Bk]

                    scores = (
                        einsum(q_block, k_block, "i j d, i k d -> i j k").to(dtype)
                        + b_block[None, :, :]
                    )  # [Bi, Bj, Bk]

                    block_max = torch.max(scores, dim=-1)[0]
                    scores_new = torch.maximum(scores_max, block_max)

                    exp_scores = torch.exp(scores - scores_new[..., None])

                    scores_sum = torch.exp(
                        scores_max - scores_new
                    ) * scores_sum + exp_scores.sum(dim=-1)

                    out_block = torch.exp(scores_max - scores_new)[
                        ..., None
                    ] * out_block + einsum(
                        exp_scores, v_block, "i j k, i k d -> i j d"
                    ).to(dtype)

                    scores_max = scores_new

                out_block = out_block / scores_sum[:, :, None]
                output[b, i:i_end, j:j_end] = out_block
                logsumexp = scores_max + torch.log(scores_sum)
                L[b, i:i_end, j:j_end] = logsumexp
    return output


def blocked_triangular_attention(
    query: torch.Tensor,  # shape [h, n, n, d]
    key: torch.Tensor,  # shape [h, n, n, d]
    value: torch.Tensor,  # shape [h, n, n, d]
    bias: torch.Tensor,  # shape [h, n, n]
    mask: torch.Tensor,  # shape [h, n, n]
    chunk: int = 32,
) -> torch.Tensor:
    """
    Computes blocked triangular attention with batch/head dimensions:
    For each head h:
        a_ijk = softmax_k(1/sqrt(c) * q_ij^T k_ik + b_jk)
        o_ij = sum_k(a_ijk * v_ik)
    """
    batch, n = query.shape[0], query.shape[1]
    d = query.shape[-1]
    device = query.device
    dtype = query.dtype
    scale = 1.0 / math.sqrt(d)

    # Pre-scale q
    query = query * scale

    # Initialize output
    output = torch.zeros((batch, n, n, d), device=device)
    L = torch.zeros((batch, n, n), dtype=dtype, device=device)

    # Process each batch/head independently
    for b in range(batch):
        for i in range(0, n, chunk):
            i_end = min(i + chunk, n)

            for j in range(0, n, chunk):
                j_end = min(j + chunk, n)

                scores_max = torch.full(
                    (i_end - i, j_end - j),
                    float("-inf"),
                    device=device,
                    dtype=dtype,
                )
                scores_sum = torch.zeros(
                    (i_end - i, j_end - j),
                    device=device,
                    dtype=dtype,
                )
                out_block = torch.zeros(
                    (i_end - i, j_end - j, d),
                    device=device,
                    dtype=dtype,
                )

                q_block = query[b, i:i_end, j:j_end]  # [Bi, Bj, d]
                for k in range(0, n, chunk):
                    k_end = min(k + chunk, n)

                    k_block = key[b, i:i_end, k:k_end]  # [Bi, Bk, d]
                    v_block = value[b, i:i_end, k:k_end]  # [Bi, Bk, d]
                    b_block = bias[b, j:j_end, k:k_end]  # [Bj, Bk]

                    # if (b == 0) and (i == 0) and (j == 0):
                    #     print("k_sum", k_block.sum())

                    scores = (
                        einsum(q_block, k_block, "i j d, i k d -> i j k").to(dtype)
                        + b_block[None]
                    )  # [Bi, Bj, Bk]

                    block_max = scores.max(dim=-1)[0]
                    scores_new = torch.maximum(scores_max, block_max)

                    exp_scores = torch.exp(scores - scores_new[..., None])

                    scores_sum = torch.exp(
                        scores_max - scores_new
                    ) * scores_sum + exp_scores.sum(dim=-1)

                    out_block = torch.exp(scores_max - scores_new)[
                        ..., None
                    ] * out_block + einsum(
                        exp_scores, v_block, "i j k, i k d -> i j d"
                    ).to(dtype)

                    scores_max = scores_new

                out_block = out_block / scores_sum[..., None]
                output[b, i:i_end, j:j_end] = out_block
                logsumexp = scores_max + torch.log(scores_sum)
                L[b, i:i_end, j:j_end] = logsumexp

    return output
