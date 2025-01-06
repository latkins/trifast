import torch
import math
import triton
from triton.runtime import driver
import triton.testing
import triton.language as tl
from trifast.autotune_helpers import get_tflops
from triton.testing import get_dram_gbps


def prune(configs, named_args, **kwargs):
    device = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability()
    dtsize = named_args["q_ptr"].element_size()
    dtype = named_args["q_ptr"].dtype

    # Stolen from xformers: make sure we have enough smem
    pruned_configs = []
    for config in configs:
        kw = config.kwargs
        BLOCK_J, BLOCK_K, num_stages = (
            kw["BLOCK_J"],
            kw["BLOCK_K"],
            config.num_stages,
        )

        max_shared_memory = driver.active.utils.get_device_properties(device)[
            "max_shared_mem"
        ]
        # This is an under estimate
        required_shared_memory = (BLOCK_J * BLOCK_K) * num_stages * dtsize
        if required_shared_memory <= max_shared_memory:
            pruned_configs.append(config)
    configs = pruned_configs
    return configs


cfgs = [
    triton.Config(
        {"BLOCK_J": block_j, "BLOCK_K": block_k},
        num_warps=warps,
        num_stages=stages,
    )
    for block_j in [16, 32, 64, 128]
    for block_k in [16, 32, 64, 128]
    for warps in [1, 2, 4, 8]
    for stages in [1, 2, 3, 4, 5]
]


def estimate_fwd(
    num_warps,
    num_stages,
    q_ptr,
    N,
    H,
    DIM,  # dimensions
    BLOCK_J,
    BLOCK_K,
    **kwargs,
):
    device = torch.cuda.current_device()
    dtype = q_ptr.dtype
    dtsize = q_ptr.element_size()

    # Calculate number of CTAs
    num_cta_j = triton.cdiv(N, BLOCK_J)  # parallelize over chunks of j
    num_cta_i = H  # parallelize over heads (i)
    num_ctas = num_cta_j * num_cta_i
    num_k_blocks = triton.cdiv(N, BLOCK_K)

    # If input is smaller than block size
    N = max(N, BLOCK_J)

    # Time to compute
    # For each k block:
    # 1. Q @ K^T for scores: BLOCK_J x DIM @ DIM x BLOCK_K
    # 2. exp2/log2 ops for softmax: ~3 ops per element
    # 3. softmax @ V: BLOCK_J x BLOCK_K @ BLOCK_K x DIM
    ops_per_k_block = (
        2 * BLOCK_J * BLOCK_K * DIM  # Q @ K^T
        + 3 * BLOCK_J * BLOCK_K  # softmax ops
        + 2 * BLOCK_J * BLOCK_K * DIM  # attention @ V
    )
    total_ops = ops_per_k_block * num_k_blocks * H / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # Memory bandwidth calculation
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)

    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )
    l2_bw = dram_bw * 4

    # Calculate memory loads
    # Q is loaded once per head and reused
    load_q_dram = N * DIM * H * dtsize

    # Per k block:
    # - K block: loaded and potentially reused across j blocks
    # - V block: loaded and potentially reused across j blocks
    # - B (bias) block: loaded for each block combination
    # - Mask block: loaded for each k block
    load_k_dram = N * DIM * H * dtsize * (1 + 0.2 * (num_cta_j - 1))
    load_k_l2 = N * DIM * H * dtsize * 0.8 * (num_cta_j - 1)

    load_v_dram = N * DIM * H * dtsize * (1 + 0.2 * (num_cta_j - 1))
    load_v_l2 = N * DIM * H * dtsize * 0.8 * (num_cta_j - 1)

    load_b_dram = N * N * H * dtsize  # Assuming full bias matrix
    load_mask_dram = N * N * dtsize  # Assuming full mask matrix

    # Total memory traffic
    total_dram = (
        load_q_dram + load_k_dram + load_v_dram + load_b_dram + load_mask_dram
    ) / (1024 * 1024)  # MB
    total_l2 = (load_k_l2 + load_v_l2) / (1024 * 1024)  # MB

    # Loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # Store output and LSE
    store_bw = dram_bw * 0.6
    store_o_dram = N * DIM * H * dtsize / (1024 * 1024)  # MB
    store_lse_dram = N * H * dtsize / (1024 * 1024)  # MB
    store_ms = (store_o_dram + store_lse_dram) / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms

    return total_time_ms


# fmt: off
@triton.heuristics(values={'CLOSEST_N': lambda args: 2 ** int(math.ceil(math.log2(args['N'])))})
@triton.autotune(configs=cfgs, key=["H", "DIM", "CLOSEST_N"], prune_configs_by={
    "early_config_prune": prune,
    "perf_model": estimate_fwd,
    "top_k": 10
})
@triton.jit
def _fwd(
    o_ptr, stride_oh, stride_om, stride_on, stride_od,
    lse_ptr, stride_lseh, stride_lsem, stride_lsen,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    mask_ptr, stride_maskm, stride_maskn,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)  # Parallelize over chunks of j
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_h = tl.program_id(2)  # Parallelize along h

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2)

    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0 # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_kt_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_kt_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn) # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]

    base_lse_ptr = lse_ptr + (start_h * stride_lseh) + (start_i * stride_lsem)
    lse_ptrs = base_lse_ptr + (j_idxs * stride_lsen) # [j]

    base_mask_ptr = mask_ptr
    mask_ptrs= base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn) # [k]

    base_o_ptr = o_ptr + (start_h * stride_oh) + (start_i * stride_om)
    o_ptrs = base_o_ptr + (j_idxs[:, None] * stride_on) + (d_idxs[None, :] * stride_od) # [j,d]

    scores_max = tl.full([BLOCK_J], value=-float("inf"), dtype=tl.float32)
    sm_denom = tl.full([BLOCK_J], value=0, dtype=tl.float32)
    acc = tl.full([BLOCK_J, DIM], value=0, dtype=tl.float32)

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None])  # [j,d]
    q_block = q_block * tl.full([1], value=sm_scale, dtype=q_block.type.element_ty)

    for start_k in range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N


        kt_block = tl.load(kt_ptrs, mask_k[None, :])  # [d,k]
        b_block = tl.load(b_ptrs,  mask_j[:, None] | mask_k[None, :])  # [j,k]
        m_block = tl.load(mask_ptrs, mask_k) # [k]

        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores, input_precision="ieee")  # [j,k]
        scores *= inv_ln2 # 1.0 / ln(2), [j,k]

        # we want to make scores -inf at mask locations
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        # Iterative softmax
        block_max = tl.maximum(scores_max, tl.max(scores, 1))  # [j]
        scores = scores - block_max[:, None]  # [j,k]
        exp_scores = tl.math.exp2(scores)  # [j,k]

        summed_exp_scores = tl.sum(exp_scores, 1)  # [j]
        exp_scale = tl.math.exp2(scores_max - block_max)  # [j]

        sm_denom = sm_denom * exp_scale + summed_exp_scores  # [j]

        acc = acc * exp_scale[:, None]  # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None])  # [k,d]
        exp_scores = exp_scores.to(input_dtype)  # [j,k]

        acc = tl.dot(exp_scores, v_block, acc, input_precision="ieee")  # [j,d]

        scores_max = block_max

        # Advance to next block along the k dimension.
        kt_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn


    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(input_dtype)
    tl.store(o_ptrs, final_output, mask=mask_j[:, None])

    lse = (scores_max * ln2) + tl.log(sm_denom)

    tl.store(lse_ptrs, lse, mask=mask_j)


bwd_pre_cfgs = [
    triton.Config(
        {"BLOCK_J": block_j},
        num_warps=warps,
        num_stages=stages,
    )
    for block_j in [16, 32, 64, 128]
    for warps in [1, 2, 4, 8]
    for stages in [1, 2, 3, 4, 5]
]

def estimate_bwd_preprocess(
    num_warps,
    num_stages,
    o_ptr,  # tensors
    N, H, DIM,  # dimensions
    BLOCK_J,
    **kwargs,
):

    device = torch.cuda.current_device()
    dtype = o_ptr.dtype
    dtsize = o_ptr.element_size()

    # Calculate number of CTAs
    num_cta_j = triton.cdiv(N, BLOCK_J)  # parallelize over chunks of j
    num_cta_i = N  # parallelize along i
    num_cta_h = H  # parallelize along h
    num_ctas = num_cta_j * num_cta_i * num_cta_h

    # If input is smaller than block size
    N = max(N, BLOCK_J)

    # Time to compute
    # Each block computes BLOCK_J dot products of length DIM
    ops_per_block = 2 * BLOCK_J * DIM  # multiply-add for dot products
    total_ops = ops_per_block * num_cta_j * N * H / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # Memory bandwidth calculation
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)

    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )
    l2_bw = dram_bw * 4

    # Calculate memory loads
    # Each CTA loads a BLOCK_J x DIM block from both o and do
    load_o_dram = N * N * H * DIM * dtsize * (1 + 0.2 * (num_cta_j - 1))
    load_o_l2 = N * N * H * DIM * dtsize * 0.8 * (num_cta_j - 1)

    load_do_dram = N * N * H * DIM * dtsize * (1 + 0.2 * (num_cta_j - 1))
    load_do_l2 = N * N * H * DIM * dtsize * 0.8 * (num_cta_j - 1)

    # Total memory traffic
    total_dram = (load_o_dram + load_do_dram) / (1024 * 1024)  # MB
    total_l2 = (load_o_l2 + load_do_l2) / (1024 * 1024)  # MB

    # Loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # Store results
    store_bw = dram_bw * 0.6
    store_d_dram = N * N * H * dtsize / (1024 * 1024)  # MB (dot product results)
    store_ms = store_d_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms

    return total_time_ms

#fmt: off
@triton.heuristics(values={'CLOSEST_N': lambda args: 2 ** int(math.ceil(math.log2(args['N'])))})
@triton.autotune(configs=bwd_pre_cfgs, key=["H", "DIM", "CLOSEST_N"], prune_configs_by={
    "perf_model": estimate_bwd_preprocess,
    "top_k": 10
})
@triton.jit
def _bwd_preprocess(o_ptr, stride_oh, stride_oi, stride_oj, stride_od,
                    do_ptr, stride_doh, stride_doi, stride_doj, stride_dod,
                    d_ptr, stride_dh, stride_di, stride_dj,
                    N, H, DIM: tl.constexpr,
                    CLOSEST_N: tl.constexpr,
                    BLOCK_J: tl.constexpr,
                    ):

    pid_h = tl.program_id(2)  # Parallelize along h
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_j = tl.program_id(0)  # Parallelize over chunks of j

    h_offset = pid_h
    i_offset = pid_i
    j_offset = pid_j * BLOCK_J # The max idx of the chunk of j.

    o_block_ptr = tl.make_block_ptr(
        o_ptr + (h_offset * stride_oh) + (i_offset * stride_oi),
        shape=(N, DIM),
        strides=(stride_oj, stride_od),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0),
    )

    do_block_ptr = tl.make_block_ptr(
        do_ptr + (h_offset * stride_doh) + (i_offset * stride_doi),
        shape=(N, DIM),
        strides=(stride_doj, stride_dod),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0),
    )

    d_block_ptr = tl.make_block_ptr(
        d_ptr + (h_offset * stride_dh) + (i_offset * stride_di),
        shape=(N,),
        strides=(stride_dj,),
        offsets=(j_offset,),
        block_shape=(BLOCK_J,),
        order=(0,),
    )


    o_block = tl.load(o_block_ptr, boundary_check=(0,))
    do_block = tl.load(do_block_ptr, boundary_check=(0,))

    vals = tl.sum(do_block * o_block, axis=1)

    tl.store(d_block_ptr, vals.to(tl.float32), boundary_check=(0,))

def estimate_bwd_kv(
    num_warps,
    num_stages,
    q_ptr,  # tensors
    N, H, DIM,  # dimensions
    BLOCK_J, BLOCK_K,
    **kwargs,
):

    device = torch.cuda.current_device()
    dtype = q_ptr.dtype
    dtsize = q_ptr.element_size()

    # Calculate number of CTAs
    num_cta_k = triton.cdiv(N, BLOCK_K)  # parallelize over chunks of k
    num_cta_i = N  # parallelize along i (sequence dimension)
    num_cta_h = H  # parallelize along h (head dimension)
    num_ctas = num_cta_k * num_cta_i * num_cta_h

    # Number of j-dimension blocks we'll process
    num_j_blocks = triton.cdiv(N, BLOCK_J)

    # If input is smaller than block size
    N = max(N, max(BLOCK_J, BLOCK_K))

    # Time to compute
    # For each j block:
    # 1. Q @ K^T for scores: BLOCK_J x DIM @ DIM x BLOCK_K
    # 2. exp2 ops for softmax: BLOCK_J x BLOCK_K
    # 3. DO @ V^T for dV: BLOCK_J x DIM @ DIM x BLOCK_K
    # 4. DO @ V for dS: BLOCK_J x DIM @ DIM x BLOCK_K
    # 5. dS @ Q for dK: BLOCK_J x BLOCK_K @ BLOCK_J x DIM
    ops_per_j_block = (
        2 * BLOCK_J * BLOCK_K * DIM +     # Q @ K^T
        BLOCK_J * BLOCK_K +               # exp2 ops
        2 * BLOCK_J * BLOCK_K * DIM +     # DO @ V^T
        2 * BLOCK_J * BLOCK_K * DIM +     # DO @ V
        2 * BLOCK_J * BLOCK_K * DIM       # dS @ Q
    )
    total_ops = ops_per_j_block * num_j_blocks * num_cta_k * H / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # Memory bandwidth calculation
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)

    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )
    l2_bw = dram_bw * 4

    # Calculate memory loads
    # Initial loads per CTA:
    load_k_dram = N * DIM * H * dtsize  # K loaded once per CTA
    load_v_dram = N * DIM * H * dtsize  # V loaded once per CTA

    # Per j-block loads:
    # Q block loaded and potentially reused
    load_q_dram = N * N * H * DIM * dtsize * (1 + 0.2 * (num_j_blocks - 1))
    load_q_l2 = N * N * H * DIM * dtsize * 0.8 * (num_j_blocks - 1)

    # Bias blocks
    load_b_dram = N * N * H * dtsize

    # Row max and mask
    load_l_dram = N * H * dtsize
    load_m_dram = N * N * dtsize

    # DO blocks
    load_do_dram = N * N * H * DIM * dtsize * (1 + 0.2 * (num_j_blocks - 1))
    load_do_l2 = N * N * H * DIM * dtsize * 0.8 * (num_j_blocks - 1)

    # Delta blocks
    load_d_dram = N * H * dtsize

    # Total memory traffic
    total_dram = (
        load_k_dram + load_v_dram + load_q_dram + load_b_dram +
        load_l_dram + load_m_dram + load_do_dram + load_d_dram
    ) / (1024 * 1024)  # MB
    total_l2 = (load_q_l2 + load_do_l2) / (1024 * 1024)  # MB

    # Loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # Store results (dk and dv)
    store_bw = dram_bw * 0.6
    store_dkv_dram = 2 * N * DIM * H * dtsize / (1024 * 1024)  # MB (both dk and dv)
    store_ms = store_dkv_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms

    return total_time_ms

# fmt: off
@triton.heuristics(values={'CLOSEST_N': lambda args: 2 ** int(math.ceil(math.log2(args['N'])))})
@triton.autotune(configs=cfgs, key=["H", "DIM", "CLOSEST_N"], prune_configs_by={
    "early_config_prune": prune,
    "perf_model": estimate_bwd_kv,
    "top_k": 10
})
@triton.jit
def _bwd_kv(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    # program id
    pid_k = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    start_h = pid_h
    start_i = pid_i
    start_j = 0 # we iterate over j, so each pid starts at 0
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J)
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    kt_ptrs = base_k_ptr + (d_idxs[:, None]) * stride_kd + (k_idxs[None, :] * stride_kn)   # [d,k]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr + (d_idxs[:, None] * stride_vd) + (k_idxs[None,:] * stride_vn)  # [d,k]

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]

    base_mask_ptr = m_ptr
    mask_ptrs= base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn) # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
    dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd) # [k,d]

    base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
    dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd) # [k,d]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N

    # load k/v once per pid
    vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
    kt_block = tl.load(kt_ptrs, mask_k[None, :]) # [d,k]
    kt_block = kt_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [k,d]
    m_block = tl.load(mask_ptrs, mask_k) # [k]

    # accumulate over j for dk/dv
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        mask_j = (j_idxs + start_j) < N

        q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
        b_block = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]).to(tl.float32) # [j,k]

        scores = tl.dot(q_block, kt_block, b_block, input_precision="ieee") # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores= tl.where(m_block[None, :], neg_inf, scores)

        row_max = tl.load(l_ptrs, mask=mask_j) # [j]
        sm_value = tl.math.exp2((scores  - row_max[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
        dv_block += tl.dot(tl.trans(sm_value).to(input_dtype), do, input_precision="ieee") # [k,d]

        delta = tl.load(d_ptrs, mask_j) # [j]

        dsm_value = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dsm_value = tl.dot(do, vt_block, dsm_value, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        dscores = dscores.to(input_dtype) # [j,k]

        dk_block += tl.dot(tl.trans(dscores), q_block, input_precision="ieee") # [k,d]

        # increment pointers
        q_ptrs += BLOCK_J * stride_qn
        d_ptrs += BLOCK_J * stride_dn
        b_ptrs += BLOCK_J * stride_bm
        l_ptrs += BLOCK_J * stride_ln
        do_ptrs += BLOCK_J * stride_don
        mask_ptrs += BLOCK_J * stride_mn


    dk_block *= sm_scale
    tl.store(dk_ptrs, dk_block.to(input_dtype), mask_k[:, None])
    tl.store(dv_ptrs, dv_block.to(input_dtype), mask_k[:, None])

def estimate_bwd_q(
    num_warps,
    num_stages,
    q_ptr,
    N, H, DIM,  # dimensions
    CLOSEST_N: tl.constexpr,
    BLOCK_J, BLOCK_K,
    **kwargs,
):
    device = torch.cuda.current_device()
    dtype = q_ptr.dtype
    dtsize = q_ptr.element_size()

    # Calculate number of CTAs
    num_cta_j = triton.cdiv(N, BLOCK_J)  # parallelize over chunks of j
    num_cta_i = N  # parallelize along i (sequence dimension)
    num_cta_h = H  # parallelize along h (head dimension)
    num_ctas = num_cta_j * num_cta_i * num_cta_h

    # Number of k-dimension blocks we'll process
    num_k_blocks = triton.cdiv(N, BLOCK_K)

    # If input is smaller than block size
    N = max(N, max(BLOCK_J, BLOCK_K))

    # Time to compute
    # Initial loads and per-block operations:
    # For each k block:
    # 1. Q @ K^T for scores: BLOCK_J x DIM @ DIM x BLOCK_K
    # 2. exp2 ops for softmax: BLOCK_J x BLOCK_K
    # 3. DO @ V for dsm_value: BLOCK_J x DIM @ DIM x BLOCK_K
    # 4. dscores @ K for dq accumulation: BLOCK_J x BLOCK_K @ BLOCK_K x DIM
    ops_per_k_block = (
        2 * BLOCK_J * BLOCK_K * DIM +     # Q @ K^T
        BLOCK_J * BLOCK_K +               # exp2 ops
        2 * BLOCK_J * BLOCK_K * DIM +     # DO @ V
        2 * BLOCK_J * BLOCK_K * DIM       # dscores @ K
    )
    total_ops = ops_per_k_block * num_k_blocks * num_cta_j * H / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # Memory bandwidth calculation
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)

    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )
    l2_bw = dram_bw * 4

    # Calculate memory loads
    # Initial loads per CTA:
    load_q_dram = BLOCK_J * DIM * dtsize  # Q block loaded once
    load_l_dram = BLOCK_J * dtsize        # denominator
    load_d_dram = BLOCK_J * dtsize        # delta
    load_do_dram = BLOCK_J * DIM * dtsize # DO block

    # Per k-block loads:
    # K blocks loaded and potentially reused
    load_k_dram = N * DIM * H * dtsize * (1 + 0.2 * (num_k_blocks - 1))
    load_k_l2 = N * DIM * H * dtsize * 0.8 * (num_k_blocks - 1)

    # V blocks (transposed)
    load_v_dram = N * DIM * H * dtsize * (1 + 0.2 * (num_k_blocks - 1))
    load_v_l2 = N * DIM * H * dtsize * 0.8 * (num_k_blocks - 1)

    # Bias blocks
    load_b_dram = N * N * H * dtsize

    # Mask blocks
    load_mask_dram = N * N * dtsize

    # Total memory traffic
    total_dram = (
        load_q_dram + load_l_dram + load_d_dram + load_do_dram +
        load_k_dram + load_v_dram + load_b_dram + load_mask_dram
    ) / (1024 * 1024)  # MB
    total_l2 = (load_k_l2 + load_v_l2) / (1024 * 1024)  # MB

    # Loading time in ms
    load_ms = total_dram / dram_bw + total_l2 / l2_bw

    # Store results (dq)
    store_bw = dram_bw * 0.6
    store_dq_dram = BLOCK_J * DIM * dtsize / (1024 * 1024)  # MB
    store_ms = store_dq_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms

    return total_time_ms

# fmt: off
@triton.heuristics(values={'CLOSEST_N': lambda args: 2 ** int(math.ceil(math.log2(args['N'])))})
@triton.autotune(configs=cfgs, key=["H", "DIM", "CLOSEST_N"], prune_configs_by={
    "early_config_prune": prune,
    "perf_model": estimate_bwd_q,
    "top_k": 10
})
@triton.jit
def _bwd_q(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    mask_ptr, stride_maskm, stride_maskn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dq_ptr, stride_dqh, stride_dqm, stride_dqn, stride_dqd,
    sm_scale,
    neg_inf,
    N, H, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr,  BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    pid_j = tl.program_id(0)
    pid_i = tl.program_id(1)
    pid_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2)

    start_h = pid_h
    start_i = pid_i
    start_j = pid_j * BLOCK_J
    start_k = 0 # we iterate over k, so each pid starts at 0

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K)
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :] * stride_kd) # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    vt_ptrs = base_v_ptr  + (d_idxs[:, None] * stride_vd) + (k_idxs[None, :] * stride_vn) # [d,k]

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]

    base_mask_ptr = mask_ptr
    mask_ptrs= base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn) # [k]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    base_dq_ptr = dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
    dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs[None, :] * stride_dqd) # [j,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    mask_j = j_idxs < N

    q_block = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
    sm_denom = tl.load(l_ptrs, mask_j) # [j]
    delta = tl.load(d_ptrs, mask_j) # [j]
    do_block = tl.load(do_ptrs, mask_j[:, None]) # [j,d]

    dq_block = tl.zeros([BLOCK_J, DIM], dtype=tl.float32)

    # iterte over k for dq = \sum_{k} ds_{jk} k_{k}
    for start_k in range(0, N, BLOCK_K):
        start_k = tl.multiple_of(start_k, BLOCK_K)
        mask_k = (k_idxs + start_k) < N

        b_block = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]).to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k) # [k]
        k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
        k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [j,d]

        scores = tl.dot(q_block, tl.trans(k_block), b_block, input_precision="ieee") # [j,k]
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        sm_value = tl.math.exp2((scores  - sm_denom[:, None]) * inv_ln2) # [j,k]

        vt_block = tl.load(vt_ptrs, mask_k[None, :]) # [d,k]
        dsm_value = tl.dot(do_block, vt_block, input_precision="ieee") # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        dscores = dscores.to(input_dtype) # [j,k]

        dq_block += tl.dot(dscores, k_block, input_precision="ieee")

        k_ptrs += BLOCK_K * stride_kn
        vt_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])

def estimate_bwd_b(
    num_warps,
    num_stages,
    q_ptr,
    N, H, DIM,  # dimensions
    CLOSEST_N: tl.constexpr,
    BLOCK_J, BLOCK_K,
    **kwargs,
):
    device = torch.cuda.current_device()
    dtype = q_ptr.dtype
    dtsize = q_ptr.element_size()
    BLOCK_I = 1  # hardcoded in kernel

    # Calculate number of CTAs
    num_cta_j = triton.cdiv(N, BLOCK_J)  # parallelize over chunks of j
    num_cta_k = triton.cdiv(N, BLOCK_K)  # parallelize over chunks of k
    num_cta_h = H  # parallelize along h (head dimension)
    num_ctas = num_cta_j * num_cta_k * num_cta_h

    # Number of i-dimension blocks we'll process
    num_i_blocks = triton.cdiv(N, BLOCK_I)

    # If input is smaller than block size
    N = max(N, max(BLOCK_J, BLOCK_K))

    # Time to compute
    # For each i block:
    # 1. Q @ K^T for scores: BLOCK_J x DIM @ DIM x BLOCK_K
    # 2. exp2 ops for softmax: BLOCK_J x BLOCK_K
    # 3. DO @ V^T for dsm_value: BLOCK_J x DIM @ DIM x BLOCK_K
    # 4. Element-wise ops for final gradient
    ops_per_i_block = (
        2 * BLOCK_J * BLOCK_K * DIM +     # Q @ K^T
        BLOCK_J * BLOCK_K +               # exp2 ops
        2 * BLOCK_J * BLOCK_K * DIM +     # DO @ V^T
        2 * BLOCK_J * BLOCK_K            # Element-wise ops
    )
    total_ops = ops_per_i_block * num_i_blocks * num_cta_j * num_cta_k * H / (1024 * 1024 * 1024)  # GOPS
    tput = get_tflops(device, num_ctas, num_warps, dtype)
    compute_ms = total_ops / tput

    # Memory bandwidth calculation
    num_sm = driver.active.utils.get_device_properties(device)["multiprocessor_count"]
    active_cta_ratio = min(1, num_ctas / num_sm)
    active_cta_ratio_bw1 = min(1, num_ctas / 32)
    active_cta_ratio_bw2 = max(min(1, (num_ctas - 32) / (108 - 32)), 0)

    dram_bw = get_dram_gbps(device) * (
        active_cta_ratio_bw1 * 0.95 + active_cta_ratio_bw2 * 0.05
    )
    l2_bw = dram_bw * 4

    # Calculate memory loads
    # Per i-block loads:
    load_q_dram = N * DIM * H * dtsize   # Q blocks
    load_k_dram = N * DIM * H * dtsize   # K blocks
    load_v_dram = N * DIM * H * dtsize   # V blocks

    # Bias blocks loaded once per iteration
    load_b_dram = N * N * H * dtsize

    # L and delta loaded for each block
    load_l_dram = N * H * dtsize
    load_d_dram = N * H * dtsize

    # DO blocks
    load_do_dram = N * DIM * H * dtsize

    # Mask blocks
    load_m_dram = N * N * dtsize

    # Total memory traffic
    total_dram = (
        load_q_dram + load_k_dram + load_v_dram + load_b_dram +
        load_l_dram + load_d_dram + load_do_dram + load_m_dram
    ) / (1024 * 1024)  # MB

    # Loading time in ms
    load_ms = total_dram / dram_bw

    # Store results (db)
    store_bw = dram_bw * 0.6
    store_db_dram = N * N * H * dtsize / (1024 * 1024)  # MB (full bias gradient)
    store_ms = store_db_dram / store_bw

    total_time_ms = max(compute_ms, load_ms) + store_ms


    return total_time_ms

# fmt: off
@triton.heuristics(values={'CLOSEST_N': lambda args: 2 ** int(math.ceil(math.log2(args['N'])))})
@triton.autotune(configs=cfgs, key=["H", "DIM", "CLOSEST_N"], prune_configs_by={
    "early_config_prune": prune,
    "perf_model": estimate_bwd_b,
    "top_k": 10
})
@triton.jit
def _bwd_b(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    m_ptr, stride_mm, stride_mn,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    db_ptr, stride_dbh, stride_dbm, stride_dbn,
    sm_scale,
    neg_inf,
    H, N, DIM: tl.constexpr,
    CLOSEST_N: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty
    BLOCK_I: tl.constexpr = 1

    # program id
    pid_j = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),

    start_h = pid_h
    start_i = 0 # we iterate over j, so each pid starts at 0
    start_j = pid_j * BLOCK_J
    start_k = pid_k * BLOCK_K

    # Indices of blocks.
    k_idxs = tl.arange(0, BLOCK_K) + start_k
    j_idxs = tl.arange(0, BLOCK_J) + start_j
    d_idxs = tl.arange(0, DIM)

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd  # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]

    base_l_ptr = l_ptr + (start_h * stride_lh)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]

    base_mask_ptr = m_ptr
    mask_ptrs= base_mask_ptr + (k_idxs * stride_mn) # [k]

    base_do_ptr = do_ptr + (start_h * stride_doh)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]

    base_db_ptr = db_ptr + (pid_h * stride_dbh)
    db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)

    base_d_ptr = d_ptr + (start_h * stride_dh)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N
    mask_j = j_idxs < N

    db_block = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)

    # loop over i
    for start_i in range(0, N, BLOCK_I):
        start_i = tl.multiple_of(start_i, BLOCK_I)
        q_block = tl.load(q_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        q_block *= tl.full([1], value=sm_scale, dtype=input_dtype)
        k_block = tl.load(k_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]

        b_block = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :], cache_modifier=".cg").to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k, cache_modifier=".cg") # [k]

        scores = tl.dot(q_block, tl.trans(k_block), b_block, input_precision="ieee") # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores= tl.where(m_block[None, :], neg_inf, scores)

        sm_denom = tl.load(l_ptrs, mask=mask_j, cache_modifier=".cg") # [j]
        sm_score = tl.math.exp2((scores  - sm_denom[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None], cache_modifier=".cg") # [j,d]
        delta = tl.load(d_ptrs, mask_j, cache_modifier=".cg") # [j]

        v_block = tl.load(v_ptrs, mask_k[:, None], cache_modifier=".cg") # [k,d]
        dsm_value = tl.dot(do, tl.trans(v_block), input_precision="ieee") # [j,k]

        dscores = sm_score * (dsm_value - delta[:, None]) # [j,k]

        db_block += dscores

        # increment pointers
        q_ptrs += stride_qm * BLOCK_I
        k_ptrs += stride_km * BLOCK_I
        v_ptrs += stride_vm * BLOCK_I
        l_ptrs += stride_lm * BLOCK_I
        mask_ptrs += stride_mm * BLOCK_I
        d_ptrs += stride_dm * BLOCK_I
        do_ptrs += stride_dom * BLOCK_I

    tl.store(db_ptrs, db_block.to(input_dtype), mask=mask_j[:, None] & mask_k[None, :])
