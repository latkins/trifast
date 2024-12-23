import sys
import triton
import torch
from jaxtyping import Bool, Float
import triton.testing
import triton.language as tl
from pairformer_kernel.autotune_helpers import config_pruner, estimate_performance


# We don't want to do a lot of autotune if we are in a test.
is_testing = "pytest" in sys.modules

configs = [
    triton.Config({"BLOCK_J": BJ, "BLOCK_K": BK}, num_stages=s, num_warps=w)
    for BJ in [16, 32, 64, 128]
    for BK in [16, 32, 64, 128]
    for s in ([1, 2, 3, 4])
    for w in [2, 4, 8]
]


short_configs = [
    triton.Config({"BLOCK_J": 16, "BLOCK_K": 16}, num_stages=2, num_warps=1),
]


pruner = {
    "perf_model": estimate_performance,
    "top_n": 3,
    "early_config_prune": config_pruner,
}

c = short_configs if is_testing else configs

# fmt: off
@triton.autotune(configs=c,
                 prune_configs_by=None if is_testing else pruner,
                 key=["DIM", "N"])
@triton.jit
def triangle_attention_fwd_kernel(
    o_ptr, stride_oh, stride_oi, stride_oj, stride_od,
    lse_ptr, stride_lseh, stride_lsei, stride_lsej,
    q_ptr, stride_qh, stride_qi, stride_qj, stride_qd,
    k_ptr, stride_kh, stride_ki, stride_kj, stride_kd,
    v_ptr, stride_vh, stride_vi, stride_vj, stride_vd,
    b_ptr, stride_bh, stride_bi, stride_bj,
    m_ptr, stride_mi, stride_mj,
    sm_scale,
    neg_inf,
    N: tl.constexpr, DIM: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(2)  # Parallelize along h
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_j = tl.program_id(0)  # Parallelize over chunks of j

    # Compute offsets for the current block
    h_offset = pid_h
    i_offset = pid_i
    j_offset = pid_j * BLOCK_J # The max idx of the chunk of j.
    k_offset = 0  # K always starts at 0

    q_block_ptr = tl.make_block_ptr(
        q_ptr + (h_offset * stride_qh) + (i_offset * stride_qi),
        shape=(N, DIM),
        strides=(stride_qj, stride_qd),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0),
    )
    k_block_ptr = tl.make_block_ptr(
        k_ptr + (h_offset * stride_kh) + (i_offset * stride_ki),
        shape=(DIM, N),
        strides=(stride_kd, stride_kj),
        offsets=(0, k_offset),
        block_shape=(DIM, BLOCK_K),
        order=(0, 1),
    )

    v_block_ptr = tl.make_block_ptr(
        v_ptr + (h_offset * stride_vh) + (i_offset * stride_vi),
        shape=(N, DIM),
        strides=(stride_vj, stride_vd),
        offsets=(k_offset, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0),
    )
    o_block_ptr = tl.make_block_ptr(
        o_ptr + (h_offset * stride_oh) + (i_offset * stride_oi),
        shape=(N, DIM),
        strides=(stride_oj, stride_od),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0),
    )

    b_block_ptr = tl.make_block_ptr(
        b_ptr + (h_offset * stride_bh),
        shape=(N, N),
        strides=(stride_bi, stride_bj),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, BLOCK_K),
        order=(0, 1),
    )
    m_block_ptr = tl.make_block_ptr(
        m_ptr,
        shape=(N, N),
        strides=(stride_mi, stride_mj),
        offsets=(i_offset, 0),
        block_shape=(1, BLOCK_K),
        order=(0, 1),
    )

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    q_block = tl.load(q_block_ptr)  # [j,d]
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

    for k_offset in range(0, N, BLOCK_K):
        k_block = tl.load(k_block_ptr)  # [k,d]
        b_block = tl.load(b_block_ptr)  # [j,k]

        m_block = tl.load(m_block_ptr) # [1,k]

        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, k_block, scores)  # [j,k]
        scores *= 1.44269504 # 1.0 / ln(2), [j,k]

        # we want to make scores -inf at mask locations
        scores = tl.where(m_block, neg_inf, scores)  # [j,k]

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        scores = scores - block_max[:, None]  # [j,k]
        exp_scores = tl.math.exp2(scores)  # [j,k]

        summed_exp_scores = tl.sum(exp_scores, 1)  # [j]
        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + summed_exp_scores  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_block_ptr)  # [k,d]
        exp_scores = exp_scores.to(v_block.type.element_ty)  # [j,k]
        acc = tl.dot(exp_scores, v_block, acc)  # [j,d]

        scores_max = block_max

        # Advance to next block along the k dimension.
        k_block_ptr = tl.advance(k_block_ptr, (0, BLOCK_K))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))
        m_block_ptr = tl.advance(m_block_ptr, (0, BLOCK_K))

    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(o_ptr.type.element_ty)
    tl.store(o_block_ptr, final_output)
    scores_max += tl.math.log2(sm_denom)

    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + (h_offset * stride_lseh) + (i_offset * stride_lsei),
        shape=(N,),
        strides=(stride_lsej,),
        offsets=(j_offset,),
        block_shape=(BLOCK_J,),
        order=(0,),
    )

    tl.store(lse_block_ptr, scores_max)


# fmt: off
@triton.jit
def triangle_attention_bwd_kernel():
    pass
