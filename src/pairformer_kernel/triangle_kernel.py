import sys
import triton
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
c = short_configs


# fmt: off
@triton.autotune(configs=c,
                 prune_configs_by=None if is_testing else pruner,
                 key=["DIM", "N"])
@triton.jit
def _fwd(
    o_ptr, stride_oh, stride_oi, stride_oj, stride_od,
    lse_ptr, stride_lseh, stride_lsei, stride_lsej,
    q_ptr, stride_qh, stride_qi, stride_qj, stride_qd,
    k_ptr, stride_kh, stride_ki, stride_kj, stride_kd,
    v_ptr, stride_vh, stride_vi, stride_vj, stride_vd,
    b_ptr, stride_bh, stride_bi, stride_bj,
    mask_ptr, stride_maski, stride_maskj,
    sm_scale,
    neg_inf,
    N: tl.constexpr, DIM: tl.constexpr,
    BLOCK_J: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(2)  # Parallelize along h
    pid_i = tl.program_id(1)  # Parallelize along i
    pid_j = tl.program_id(0)  # Parallelize over chunks of j

    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2),

    # Compute offsets for the current block
    h_offset = pid_h
    i_offset = pid_i
    j_offset = pid_j * BLOCK_J
    k_offset = 0  # K always starts at 0

    q_block_ptr = tl.make_block_ptr(
        q_ptr + (h_offset * stride_qh) + (i_offset * stride_qi),
        shape=(N, DIM),
        strides=(stride_qj, stride_qd),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0),
    )
    kt_block_ptr = tl.make_block_ptr(
        k_ptr + (h_offset * stride_kh) + (i_offset * stride_ki),
        shape=(DIM, N),
        strides=(stride_kd, stride_kj),
        offsets=(0, k_offset),
        block_shape=(DIM, BLOCK_K),
        order=(0, 1),
    )
    b_block_ptr = tl.make_block_ptr(
        b_ptr + (h_offset * stride_bh),
        shape=(N, N),
        strides=(stride_bi, stride_bj),
        offsets=(j_offset, 0),
        block_shape=(BLOCK_J, BLOCK_K),
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

    mask_block_ptr = tl.make_block_ptr(
        mask_ptr,
        shape=(N, N),
        strides=(stride_maski, stride_maskj),
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
        kt_block = tl.load(kt_block_ptr)  # [k,d]
        b_block = tl.load(b_block_ptr)  # [j,k]

        m_block = tl.load(mask_block_ptr) # [1,k]

        scores = b_block.to(tl.float32)
        scores = tl.dot(q_block, kt_block, scores)  # [j,k]
        scores *= inv_ln2 # 1.0 / ln(2), [j,k]

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
        kt_block_ptr = tl.advance(kt_block_ptr, (0, BLOCK_K))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))
        mask_block_ptr = tl.advance(mask_block_ptr, (0, BLOCK_K))

    normalize = acc / sm_denom[:, None]
    final_output = normalize.to(o_ptr.type.element_ty)
    tl.store(o_block_ptr, final_output)

    # lse = scores_max + tl.math.log2(sm_denom)
    lse = (scores_max * ln2) + tl.log(sm_denom)

    lse_block_ptr = tl.make_block_ptr(
        lse_ptr + (h_offset * stride_lseh) + (i_offset * stride_lsei),
        shape=(N,),
        strides=(stride_lsej,),
        offsets=(j_offset,),
        block_shape=(BLOCK_J,),
        order=(0,),
    )

    tl.store(lse_block_ptr, lse)

#fmt: off
@triton.jit
def _bwd_preprocess(o_ptr, stride_oh, stride_oi, stride_oj, stride_od,
                    do_ptr, stride_doh, stride_doi, stride_doj, stride_dod,
                    d_ptr, stride_dh, stride_di, stride_dj,
                    N: tl.constexpr, DIM: tl.constexpr,
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
        shape=(N, ),
        strides=(stride_dj, ),
        offsets=(j_offset, ),
        block_shape=(BLOCK_J, ),
        order=(0, ),
    )


    block_o = tl.load(o_block_ptr)
    block_do = tl.load(do_block_ptr)

    vals = tl.sum(block_do * block_o, axis=1)

    tl.store(d_block_ptr, vals.to(tl.float32))


# fmt: off
@triton.jit
def _bwd_kvb_kernel(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
    dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
    db_ptr, stride_dbh, stride_dbm, stride_dbn,
    sm_scale,
    H, M, N,
    BLOCK_J: tl.constexpr, DIM: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty


    # program id
    start_k = tl.program_id(0)
    off_i = tl.program_id(1)
    off_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),
    ln2: tl.constexpr = 0.6931471824645996 # = ln(2),
    off_k = start_k * BLOCK_K
    off_j = 0

    # init blocks
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + (off_i * stride_qm) + (off_h * stride_qh),
        shape=(N, DIM),
        strides=(stride_qn, stride_qd),
        offsets=(off_j, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + (off_i * stride_km) + (off_h * stride_kh),
        shape=(N, DIM),
        strides=(stride_kn, stride_kd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + (off_i * stride_vm) + (off_h * stride_vh),
        shape=(N, DIM),
        strides=(stride_vn, stride_vd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + (off_i * stride_dom) + (off_h * stride_doh),
        shape=(N, DIM),
        strides=(stride_don, stride_dod),
        offsets=(off_j, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0)
    )
    dk_block_ptr = tl.make_block_ptr(
        base=dk_ptr + (off_i * stride_dkm) + (off_h * stride_dkh),
        shape=(N, DIM),
        strides=(stride_dkn, stride_dkd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    dv_block_ptr = tl.make_block_ptr(
        base=dv_ptr + (off_i * stride_dvm) + (off_h * stride_dvh),
        shape=(N, DIM),
        strides=(stride_dvn, stride_dvd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + (off_h * stride_bh),
        shape=(N, N),
        strides=(stride_bm, stride_bn),
        offsets=(off_j, off_k),
        block_shape=(BLOCK_J, BLOCK_K),
        order=(1, 0)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + (off_i * stride_dm) + (off_h * stride_dh),
        shape=(N,),
        strides=(stride_dn,),
        offsets=(off_j,),
        block_shape=(BLOCK_J,),
        order=(0,)
    )
    lse_block_ptr = tl.make_block_ptr(
        base=l_ptr + (off_i * stride_lm) + (off_h * stride_lh),
        shape=(N,),
        strides=(stride_ln,),
        offsets=(off_j,),
        block_shape=(BLOCK_J,),
        order=(0,)
    )


    # load k/v once per pid
    v = tl.load(v_block_ptr) # [k,d]
    k = tl.load(k_block_ptr) # [k,d]
    k = k * tl.full([1], value=sm_scale, dtype=input_dtype) # [k,d]

    # accumulate over j for dk/dv
    dk = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # need this style of pointer for atomic add
    db_ptrs = db_ptr + (off_h * stride_dbh)
    offs_db_j = tl.arange(0, BLOCK_J)
    offs_db_k = tl.arange(0, BLOCK_K) + (start_k * BLOCK_K)
    db_ptrs = db_ptrs + (offs_db_j[:, None] * stride_dbm + offs_db_k[None, :] * stride_dbn)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)

        q = tl.load(q_block_ptr) # [j,d]
        b = tl.load(b_block_ptr).to(tl.float32) # [j,k]

        s = tl.dot(q, tl.trans(k), b) # [j,k]
        l = tl.load(lse_block_ptr) # [j]
        p = tl.math.exp2((s  - l[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_block_ptr) # [j,d]
        dv += tl.dot(tl.trans(p).to(input_dtype), do) # [k,d]

        delta = tl.load(d_block_ptr) # [j]

        dp = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v), dp) # [j,k]

        ds = p * (dp - delta[:, None]) # [j,k]

        # This is likely very slow?
        tl.atomic_add(db_ptrs + (start_j * stride_dbm),  ds.to(tl.float32))

        ds = ds.to(input_dtype) # [j,k]
        dk += tl.dot(tl.trans(ds), q) # [k,d]

        # increment pointers
        q_block_ptr = tl.advance(q_block_ptr, (BLOCK_J, 0))
        do_block_ptr = tl.advance(do_block_ptr, (BLOCK_J, 0))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_J, 0))
        lse_block_ptr = tl.advance(lse_block_ptr, (BLOCK_J,))
        d_block_ptr = tl.advance(d_block_ptr, (BLOCK_J,))


    dk *= sm_scale
    tl.store(dk_block_ptr, dk.to(input_dtype))
    tl.store(dv_block_ptr, dv.to(input_dtype))


@triton.jit
def _bwd_q_kernel(
    d_ptr, stride_dh, stride_dm, stride_dn,
    q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
    k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
    v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
    b_ptr, stride_bh, stride_bm, stride_bn,
    l_ptr, stride_lh, stride_lm, stride_ln,
    do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
    dq_ptr, stride_dqh, stride_dqm, stride_dqn, stride_dqd,
    sm_scale,
    H, M, N,
    BLOCK_J: tl.constexpr, DIM: tl.constexpr, BLOCK_K: tl.constexpr,
):
    input_dtype = q_ptr.dtype.element_ty

    start_j = tl.program_id(0)
    off_i = tl.program_id(1)
    off_h = tl.program_id(2)
    inv_ln2: tl.constexpr = 1.44269504 # = 1.0 / ln(2),

    off_k = 0
    off_j = start_j * BLOCK_J

    # init blocks
    q_block_ptr = tl.make_block_ptr(
        base=q_ptr + (off_i * stride_qm) + (off_h * stride_qh),
        shape=(N, DIM),
        strides=(stride_qn, stride_qd),
        offsets=(off_j, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0)
    )
    k_block_ptr = tl.make_block_ptr(
        base=k_ptr + (off_i * stride_km) + (off_h * stride_kh),
        shape=(N, DIM),
        strides=(stride_kn, stride_kd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    v_block_ptr = tl.make_block_ptr(
        base=v_ptr + (off_i * stride_vm) + (off_h * stride_vh),
        shape=(N, DIM),
        strides=(stride_vn, stride_vd),
        offsets=(off_k, 0),
        block_shape=(BLOCK_K, DIM),
        order=(1, 0)
    )
    do_block_ptr = tl.make_block_ptr(
        base=do_ptr + (off_i * stride_dom) + (off_h * stride_doh),
        shape=(N, DIM),
        strides=(stride_don, stride_dod),
        offsets=(off_j, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + (off_h * stride_bh),
        shape=(N, N),
        strides=(stride_bm, stride_bn),
        offsets=(off_j, off_k),
        block_shape=(BLOCK_J, BLOCK_K),
        order=(1, 0)
    )
    d_block_ptr = tl.make_block_ptr(
        base=d_ptr + (off_i * stride_dm) + (off_h * stride_dh),
        shape=(N,),
        strides=(stride_dn,),
        offsets=(off_j,),
        block_shape=(BLOCK_J,),
        order=(0,)
    )
    lse_block_ptr = tl.make_block_ptr(
        base=l_ptr + (off_i * stride_lm) + (off_h * stride_lh),
        shape=(N,),
        strides=(stride_ln,),
        offsets=(off_j,),
        block_shape=(BLOCK_J,),
        order=(0,)
    )
    dq_block_ptr = tl.make_block_ptr(
        base=dq_ptr + (off_i * stride_dqm) + (off_h * stride_dqh),
        shape=(N, DIM),
        strides=(stride_dqn, stride_dqd),
        offsets=(off_j, 0),
        block_shape=(BLOCK_J, DIM),
        order=(1, 0)
    )

    q = tl.load(q_block_ptr) # [j,d]
    l = tl.load(lse_block_ptr) # [j]
    delta = tl.load(d_block_ptr) # [j]
    do = tl.load(do_block_ptr) # [j,d]

    dq = tl.zeros([BLOCK_J, DIM], dtype=tl.float32)

    # iterte over k for dq = \sum_{k} ds_{jk} k_{k}
    for k_start in range(0, N, BLOCK_K):
        k = tl.load(k_block_ptr) # [k,d]
        k = k * tl.full([1], value=sm_scale, dtype=input_dtype) # [j,d]
        v = tl.load(v_block_ptr) # [k,d]
        b = tl.load(b_block_ptr).to(tl.float32) # [j,k]

        s = tl.dot(q, tl.trans(k), b) # [j,k]

        p = tl.math.exp2((s  - l[:, None]) * inv_ln2) # [j,k]

        dp = tl.dot(do, tl.trans(v)) # [j,k]

        ds = p * (dp - delta[:, None]) # [j,k]
        ds = ds.to(input_dtype) # [j,k]

        dq += tl.dot(ds, k)

        k_block_ptr = tl.advance(k_block_ptr, (BLOCK_K, 0))
        v_block_ptr = tl.advance(v_block_ptr, (BLOCK_K, 0))
        b_block_ptr = tl.advance(b_block_ptr, (0, BLOCK_K))

    tl.store(dq_block_ptr, dq.to(input_dtype))
