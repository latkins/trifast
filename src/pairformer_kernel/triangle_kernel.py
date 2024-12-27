import triton
import triton.testing
import triton.language as tl


# We don't want to do a lot of autotune if we are in a test.


# fmt: off
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
    N: tl.constexpr, DIM: tl.constexpr,
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
        scores = tl.dot(q_block, kt_block, scores)  # [j,k]
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

        acc = tl.dot(exp_scores, v_block, acc)  # [j,d]

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

# fmt: off
@triton.jit
def _bwd_kvb_kernel(
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
    db_ptr, stride_dbh, stride_dbm, stride_dbn,
    sm_scale,
    neg_inf,
    H, M, N,
    BLOCK_J: tl.constexpr, DIM: tl.constexpr, BLOCK_K: tl.constexpr,
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
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd  # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]

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

    base_db_ptr = db_ptr + (pid_h * stride_dbh)
    db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    mask_k = k_idxs < N

    # load k/v once per pid
    v_block = tl.load(v_ptrs, mask_k[:, None]) # [k,d]
    k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
    k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [k,d]
    m_block = tl.load(mask_ptrs, mask_k) # [k]

    # accumulate over j for dk/dv
    dk_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)
    dv_block = tl.zeros([BLOCK_K, DIM], dtype=tl.float32)

    # loop over a column
    for start_j in range(0, N, BLOCK_J):
        start_j = tl.multiple_of(start_j, BLOCK_J)
        mask_j = (j_idxs + start_j) < N

        q = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
        b = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]).to(tl.float32) # [j,k]

        scores = tl.dot(q, tl.trans(k_block), b) # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
        scores= tl.where(m_block[None, :], neg_inf, scores)

        row_max = tl.load(l_ptrs, mask=mask_j) # [j]
        p = tl.math.exp2((scores  - row_max[:, None]) * inv_ln2) # [j,k]

        do = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
        dv_block += tl.dot(tl.trans(p).to(input_dtype), do) # [k,d]

        delta = tl.load(d_ptrs, mask_j) # [j]

        dp = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
        dp = tl.dot(do, tl.trans(v_block), dp) # [j,k]

        ds = p * (dp - delta[:, None]) # [j,k]

        # This is likely very slow?
        # mask ds here
        tl.atomic_add(db_ptrs + (start_j * stride_dbm),  ds.to(tl.float32))

        ds = ds.to(input_dtype) # [j,k]
        # don't sum of masked values.

        dk_block += tl.dot(tl.trans(ds), q) # [k,d]

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


# fmt: off
@triton.jit
def _bwd_q_kernel(
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
    H, M, N,
    BLOCK_J: tl.constexpr, DIM: tl.constexpr, BLOCK_K: tl.constexpr,
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
    d_idxs = tl.arange(0, DIM)[None, :] # [1, d]

    # Set up ptrs to blocks.
    base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
    q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs * stride_qd) # [j,d]

    base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
    k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs) * stride_kd # [k,d]

    base_b_ptr = b_ptr + (start_h * stride_bh)
    b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs * stride_bn) # [j,k]

    base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
    v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs * stride_vd) # [k,d]

    base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
    l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]

    base_mask_ptr = mask_ptr
    mask_ptrs= base_mask_ptr + (start_i * stride_maskm) + (k_idxs * stride_maskn) # [k]

    base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
    d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]

    base_dq_ptr = dq_ptr + (start_h * stride_dqh) + (start_i * stride_dqm)
    dq_ptrs = base_dq_ptr + (j_idxs[:, None] * stride_dqn) + (d_idxs * stride_dqd) # [j,d]

    base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
    do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs * stride_dod) # [j,d]

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

        k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
        k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [j,d]
        v_block = tl.load(v_ptrs, mask_k[:, None]) # [k,d]
        b_block = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]).to(tl.float32) # [j,k]
        m_block = tl.load(mask_ptrs, mask_k) # [k]

        scores = tl.dot(q_block, tl.trans(k_block), b_block) # [j,k]
        scores = tl.where(m_block[None, :], neg_inf, scores)  # [j,k]
        scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)

        sm_value = tl.math.exp2((scores  - sm_denom[:, None]) * inv_ln2) # [j,k]

        dsm_value = tl.dot(do_block, tl.trans(v_block)) # [j,k]

        dscores = sm_value * (dsm_value - delta[:, None]) # [j,k]
        dscores = dscores.to(input_dtype) # [j,k]

        dq_block += tl.dot(dscores, k_block)

        k_ptrs += BLOCK_K * stride_kn
        v_ptrs += BLOCK_K * stride_vn
        b_ptrs += BLOCK_K * stride_bn
        mask_ptrs += BLOCK_K * stride_maskn

    tl.store(dq_ptrs, dq_block.to(input_dtype), mask=mask_j[:, None])

# fmt: off
# @triton.jit
# def _bwd_b_kernel(
#     d_ptr, stride_dh, stride_dm, stride_dn,
#     q_ptr, stride_qh, stride_qm, stride_qn, stride_qd,
#     k_ptr, stride_kh, stride_km, stride_kn, stride_kd,
#     v_ptr, stride_vh, stride_vm, stride_vn, stride_vd,
#     b_ptr, stride_bh, stride_bm, stride_bn,
#     l_ptr, stride_lh, stride_lm, stride_ln,
#     m_ptr, stride_mm, stride_mn,
#     do_ptr, stride_doh, stride_dom, stride_don, stride_dod,
#     dk_ptr, stride_dkh, stride_dkm, stride_dkn, stride_dkd,
#     dv_ptr, stride_dvh, stride_dvm, stride_dvn, stride_dvd,
#     db_ptr, stride_dbh, stride_dbm, stride_dbn,
#     sm_scale,
#     neg_inf,
#     H, M, N,
#     BLOCK_J: tl.constexpr, DIM: tl.constexpr, BLOCK_K: tl.constexpr,
# ):
#     input_dtype = q_ptr.dtype.element_ty
#
#     # program id
#     pid_k = tl.program_id(0)
#     pid_i = tl.program_id(1)
#     pid_h = tl.program_id(2)
#
#     inv_ln2: tl.constexpr = 1.4426950408889634 # = 1.0 / ln(2),
#
#     start_h = pid_h
#     start_i = pid_i
#     start_j = 0 # we iterate over j, so each pid starts at 0
#     start_k = pid_k * BLOCK_K
#
#     # Indices of blocks.
#     k_idxs = tl.arange(0, BLOCK_K) + start_k
#     j_idxs = tl.arange(0, BLOCK_J)
#     d_idxs = tl.arange(0, DIM)
#
#     # Set up ptrs to blocks.
#     base_q_ptr = q_ptr + (start_h * stride_qh) + (start_i * stride_qm)
#     q_ptrs = base_q_ptr + (j_idxs[:, None] * stride_qn) + (d_idxs[None, :] * stride_qd) # [j,d]
#
#     base_k_ptr = k_ptr + (start_h * stride_kh) + (start_i * stride_km)
#     k_ptrs = base_k_ptr + (k_idxs[:, None] * stride_kn) + (d_idxs[None, :]) * stride_kd  # [k,d]
#
#     base_b_ptr = b_ptr + (start_h * stride_bh)
#     b_ptrs = base_b_ptr + (j_idxs[:, None] * stride_bm) + (k_idxs[None, :] * stride_bn) # [j,k]
#
#     base_v_ptr = v_ptr + (start_h * stride_vh) + (start_i * stride_vm)
#     v_ptrs = base_v_ptr + (k_idxs[:, None] * stride_vn) + (d_idxs[None, :] * stride_vd) # [k,d]
#
#     base_l_ptr = l_ptr + (start_h * stride_lh) + (start_i * stride_lm)
#     l_ptrs = base_l_ptr + (j_idxs * stride_ln) # [j]
#
#     base_mask_ptr = m_ptr
#     mask_ptrs= base_mask_ptr + (start_i * stride_mm) + (k_idxs * stride_mn) # [k]
#
#     base_do_ptr = do_ptr + (start_h * stride_doh) + (start_i * stride_dom)
#     do_ptrs = base_do_ptr + (j_idxs[:, None] * stride_don) + (d_idxs[None, :] * stride_dod) # [j,d]
#
#     base_dk_ptr = dk_ptr + (start_h * stride_dkh) + (start_i * stride_dkm)
#     dk_ptrs = base_dk_ptr + (k_idxs[:, None] * stride_dkn) + (d_idxs[None, :] * stride_dkd) # [k,d]
#
#     base_dv_ptr = dv_ptr + (start_h * stride_dvh) + (start_i * stride_dvm)
#     dv_ptrs = base_dv_ptr + (k_idxs[:, None] * stride_dvn) + (d_idxs[None, :] * stride_dvd) # [k,d]
#
#     base_db_ptr = db_ptr + (pid_h * stride_dbh)
#     db_ptrs = base_db_ptr + (j_idxs[:, None] * stride_dbm + k_idxs[None, :] * stride_dbn)
#
#     base_d_ptr = d_ptr + (start_h * stride_dh) + (start_i * stride_dm)
#     d_ptrs = base_d_ptr + (j_idxs * stride_dn)  # [j]
#
#     mask_k = k_idxs < N
#
#     # load k/v once per pid
#     v_block = tl.load(v_ptrs, mask_k[:, None]) # [k,d]
#     k_block = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
#     k_block = k_block * tl.full([1], value=sm_scale, dtype=input_dtype) # [k,d]
#     m_block = tl.load(mask_ptrs, mask_k) # [k]
#
#     # s_ijk = q_ij \dot k_ik + b_jk
#     # l_ij
#     # d_ij
#     # do_ij
#     # so best if we load an i/j block and step over k? would requires two loads per step.
#     # if we can only block over one dim tho we should block over i?
#     # db: [j,k] -> we need to sum over i
#
#     # if we step over k (e.g. work on an i/j block)
#     # load q_ij
#     # load b_jk
#     # load l_ij
#     # load d_ij
#     # load do_ij
#     # init db_jk
#     #for start_k in range(0, N, BLOCK_K):
#     #    load k_ik
#     #    load b_jk
#     #    compute etc
#     #    write db_ij
#
#     # if we step over i (e.g. work on a j/k block)
#     # load b_jk
#     # for start_i in range(0, N, BLOCK_I):
#         # load q_ij, k_ik, l_ij, d_ij, do_ij, m_ik (?)
#         # compute etc
#         # sum to db
#     # write db
#
#
#     b_block = tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]) # [j,k]
#     db_block = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
#     mask_j = (j_idxs + start_j) < N
#
#     # loop over a column
#     for start_i in range(0, N):
#         q = tl.load(q_ptrs, mask_j[:, None]) # [j,d]
#         k = tl.load(k_ptrs, mask_k[:, None]) # [k,d]
#         b= tl.load(b_ptrs, mask_j[:, None] | mask_k[None, :]).to(tl.float32) # [j,k]
#         l = tl.load(l_ptrs, mask_j)
#         m_block = tl.load(mask_ptrs, mask_k) # [k]
#         d_ij = tl.load(d_ptrs, mask_j)
#         do = tl.load(do_ptrs, mask_j[:, None])
#
#         scores = tl.dot(q, tl.trans(k_block), b) # [j,k]
#         scores = tl.where(mask_j[:, None] & mask_k[None, :], scores, neg_inf)
#         scores= tl.where(m_block[None, :], neg_inf, scores)
#
#         row_max = tl.load(l_ptrs, mask=mask_j) # [j]
#         p = tl.math.exp2((scores  - row_max[:, None]) * inv_ln2) # [j,k]
#
#         do = tl.load(do_ptrs, mask_j[:, None]) # [j,d]
#
#         delta = tl.load(d_ptrs, mask_j) # [j]
#
#         dp = tl.zeros([BLOCK_J, BLOCK_K], dtype=tl.float32)
#         dp = tl.dot(do, tl.trans(v_block), dp) # [j,k]
#
#         ds = p * (dp - delta[:, None]) # [j,k]
#
#         # This is likely very slow?
#         # mask ds here
#         db_block += ds.to(tl.float32))
#
#         # increment pointers
#         q_ptrs += BLOCK_J * stride_qn
#         d_ptrs += BLOCK_J * stride_dn
#         b_ptrs += BLOCK_J * stride_bm
#         l_ptrs += BLOCK_J * stride_ln
#         do_ptrs += BLOCK_J * stride_don
#         mask_ptrs += BLOCK_J * stride_mn
