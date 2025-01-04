import torch
from jaxtyping import Bool, Float

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
) -> Float[torch.Tensor, "... h n n d"]:
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

    a = torch.softmax(a, dim=-1)
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
