#!/usr/bin/env python3
# ============================================================================
# glm_fp8_contract.py -- VECTORIZED, GPU-SCALE mirror of OUR GLM-5.2-FP8 contract
# ----------------------------------------------------------------------------
# WHAT THIS IS
#   tools/glm_fp8_ref.py is the GOLDEN, bit-exact-but-SLOW (pure-python, scalar)
#   model of OUR FP8 arithmetic contract -- the exact thing src/glm_matmul_fp8.v
#   + src/fp8_e4m3.vh compute.  This module is the SAME arithmetic, VECTORIZED so
#   it can run the full GLM-5.2-FP8 (753 B) model on a GPU.  It is BIT-IDENTICAL
#   to glm_fp8_ref (proven in the self-test: same bf16 output, bit-for-bit), not
#   merely "close": the contract's BFP within-block accumulation is EXACT, so a
#   vectorized integer implementation CAN reproduce it to the last bit.
#
# THE VECTORIZATION (the GPU win)
#   The expensive part of glm_fp8_ref is the inner triple loop that does, per
#   (i,j), a scalar fp8_mul + fp32_to_fixed for every k.  We replace it with an
#   EXACT INTEGER MATMUL.  The key algebraic fact (see fp8_e4m3.vh fp8_mul):
#       an E4M3 value decodes to  v = (-1)^s * M * 2^P   with INTEGER significand
#         M (normal: {1,mmm}=8..15, subnormal: mmm=1..7)  and integer P (=PB-9).
#   So the EXACT fixed-point product (weight 2^-ACC_FRAC, ACC_FRAC=18) of two
#   E4M3 codes a,b is
#       fixed(a*b) = sign_a*sign_b * (Ma*Mb) * 2^(PaB+PbB)
#                  = ( sign_a*Ma*2^PaB ) * ( sign_b*Mb*2^PbB )   <-- INTEGER product
#   i.e. fp32_to_fixed(fp8_mul(a,b)) == Av * Wv where
#       Av = (-1)^s_a * (Ma << PaB)   (a SIGNED INTEGER, |.| <= 15<<14 = 245760)
#       Wv = (-1)^s_w * (Mb << PbB).
#   Therefore the EXACT within-K-block accumulation acc[i][j][blk] (an integer of
#   weight 2^-18) is just the segmented INTEGER MATMUL  (Av[M,K] @ Wv[K,N]) summed
#   per 128-wide K-block.  That is a single int64 GEMM per K-block -> BLAS / GPU
#   tensor cores, exact (no rounding inside the block, exactly as the RTL).
#
#   Everything else (the dynamic per-token POW2 a_shift activation quant, the
#   E4M3 RNE+saturate encode, the per-block fixed->fp32 rounding, the cross-block
#   fp32 dequant fold with RNE+FTZ fp32 mul/add, the final 2^-a_shift undo and
#   bf16 narrow) is reproduced bit-for-bit by VECTORIZED integer/bit-twiddling
#   kernels (one elementwise pass over the M*N output), each a faithful mirror of
#   the corresponding `function automatic` in glm_fp.vh / glm_matmul_fp8.v /
#   fp8_e4m3.vh -- the SAME ones glm_fp8_ref mirrors scalar-ly.
#
# BACKENDS
#   * torch  (if importable) : the kernels run on torch int64 tensors -> CUDA/MPS.
#   * numpy  (else, if importable) : the kernels run on numpy int64 arrays.
#   * pure python (else) : an independent scalar fallback that uses the SAME
#       Av*Wv integer-decomposition algorithm (NOT glm_fp8_ref's per-element
#       fp8_mul), so even the no-numpy path is a genuine cross-check of the
#       decomposition against the golden.
#   The torch/numpy kernels are written ONCE, parameterized by the array module
#   `xp`, using only operators (& | ^ >> << + - *), comparisons, and xp.where --
#   which are semantically identical on numpy arrays and torch tensors -- so the
#   numpy proof here transfers to the torch path bit-for-bit.
#
# PUBLIC API  (mirrors glm_fp8_ref)
#   fp8_e4m3_encode(x)  : python float -> E4M3 byte   (RNE + saturate to +/-448)
#   fp8_e4m3_decode(b)  : E4M3 byte -> python float    (exact)
#   block_fp8_gemm(A_bf16, W_fp8, w_scale_inv, blk=128, a_shift=None, backend=None)
#                       : C[M,N] bf16 codes (+ a_shift list), bit-exact to
#                         glm_fp8_ref.block_fp8_gemm.  A_bf16, W_fp8, w_scale_inv
#                         may be python lists OR numpy/torch int arrays.
#
# SELF-TEST:  python3 tools/glm_fp8_contract.py   -> "ALL <N> TESTS PASSED", exit 0
#   (bit-exact vs tools/glm_fp8_ref.py on a synthetic FP8-tile suite, for EVERY
#    available backend, plus argmax-preserving vs a float64 reference.)
# ============================================================================
import sys, os, struct, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_fp8_ref as ref          # the GOLDEN scalar contract (bit-exact mirror)

# ---- optional vectorized backends -----------------------------------------
try:
    import numpy as _np
    _HAS_NP = True
except Exception:
    _np = None
    _HAS_NP = False
try:
    import torch as _torch
    _HAS_TORCH = True
except Exception:
    _torch = None
    _HAS_TORCH = False

ACC_FRAC = ref.ACC_FRAC            # 18 -- keep in lockstep with the golden / RTL
M32 = 0xFFFFFFFF

# ============================================================================
# Scalar public codecs (thin pass-through to the golden mirror)
# ============================================================================
def fp8_e4m3_encode(x):
    """float -> E4M3 byte (RNE + saturate to +/-448).  Bit-exact to the RTL."""
    return ref.fp8_e4m3_encode(x)

def fp8_e4m3_decode(b):
    """E4M3 byte -> python float (exact)."""
    return ref.fp8_e4m3_decode(b)

# ============================================================================
# VECTORIZED bit-twiddling kernels.  Each takes the array module `xp`
# (numpy or torch) and operates elementwise on int64 arrays holding the
# relevant bit pattern.  Every kernel is a faithful mirror of the same-named
# scalar function in glm_fp8_ref (which itself mirrors the RTL).
# ============================================================================
def _zeros_like(xp, a):
    return a - a                                   # int64 zeros, same shape/backend

# ---- bf16 <-> fp32 ----------------------------------------------------------
def _bf16_to_fp32(xp, b):
    return (b & 0xFFFF) << 16

def _fp32_to_bf16(xp, f):
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    m = f & 0x7FFFFF
    inf_nan = (e == 0xFF)
    cond = (((m & 0x7FFF) != 0) | (((m >> 16) & 1) != 0))
    round_up = xp.where((((m >> 15) & 1) == 1) & cond, 1, 0)
    top_plus = (((e << 7) | (m >> 16)) + round_up) & 0x7FFF
    new_e = (top_plus >> 7) & 0xFF
    normal = xp.where(new_e == 0xFF, (s << 15) | (0xFF << 7), (s << 15) | top_plus)
    nan_res = (s << 15) | (0xFF << 7) | (1 << 6)
    inf_res = (s << 15) | (0xFF << 7)
    res = xp.where(inf_nan, xp.where(m == 0, inf_res, nan_res), normal)
    return res

# ---- fp32 scale by 2^k  (k an int array, broadcastable) --------------------
def _fp32_scale_pow2(xp, f, k):
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    m = f & 0x7FFFFF
    ne = e + k
    res = (s << 31) | ((ne & 0xFF) << 23) | m
    res = xp.where(ne >= 255, (s << 31) | (0xFF << 23), res)
    res = xp.where(ne <= 0, (s << 31), res)
    res = xp.where(e == 0, (s << 31), res)              # zero / FTZ
    res = xp.where(e == 0xFF, f & M32, res)             # inf/nan unchanged (top)
    return res

# ---- fp32 multiply (RNE, FTZ) ----------------------------------------------
def _fp32_mul(xp, a, b):
    sa = (a >> 31) & 1; sb = (b >> 31) & 1; sr = sa ^ sb
    ea = (a >> 23) & 0xFF; eb = (b >> 23) & 0xFF
    maf = a & 0x7FFFFF; mbf = b & 0x7FFFFF
    nan = ((ea == 0xFF) & (maf != 0)) | ((eb == 0xFF) & (mbf != 0))
    inf = ((ea == 0xFF) & (maf == 0)) | ((eb == 0xFF) & (mbf == 0))
    zero = (ea == 0) | (eb == 0)
    ma = (1 << 23) | maf
    mb = (1 << 23) | mbf
    prod = ma * mb                                      # <= 2^48, fits int64
    exp0 = ea + eb - 127
    top = ((prod >> 47) & 1) == 1
    mant_t = (prod >> 24) & 0xFFFFFF
    guard_t = (prod >> 23) & 1
    sticky_t = ((prod & 0x7FFFFF) != 0)
    mant_n = (prod >> 23) & 0xFFFFFF
    guard_n = (prod >> 22) & 1
    sticky_n = ((prod & 0x3FFFFF) != 0)
    mant = xp.where(top, mant_t, mant_n)
    guard = xp.where(top, guard_t, guard_n)
    sticky = xp.where(top, xp.where(sticky_t, 1, 0), xp.where(sticky_n, 1, 0))
    exp_s = xp.where(top, exp0 + 1, exp0)
    round_up = guard & (sticky | (mant & 1))
    mant_r = mant + round_up
    carry = ((mant_r >> 24) & 1) == 1
    mant_r = xp.where(carry, mant_r >> 1, mant_r)
    exp_s = xp.where(carry, exp_s + 1, exp_s)
    normal = xp.where(exp_s >= 255, (sr << 31) | (0xFF << 23),
              xp.where(exp_s <= 0, (sr << 31),
                       (sr << 31) | ((exp_s & 0xFF) << 23) | (mant_r & 0x7FFFFF)))
    res = normal
    res = xp.where(zero, (sr << 31), res)
    res = xp.where(inf, xp.where(zero, 0x7FC00000, (sr << 31) | (0xFF << 23)), res)
    res = xp.where(nan, 0x7FC00000, res)
    return res

# ---- fp32 add (RNE, FTZ) ----------------------------------------------------
def _fp32_add(xp, a, b):
    sa = (a >> 31) & 1; sb = (b >> 31) & 1
    ea = (a >> 23) & 0xFF; eb = (b >> 23) & 0xFF
    maf = a & 0x7FFFFF; mbf = b & 0x7FFFFF
    nan_a = (ea == 0xFF) & (maf != 0); nan_b = (eb == 0xFF) & (mbf != 0)
    inf_a = (ea == 0xFF) & (maf == 0); inf_b = (eb == 0xFF) & (mbf == 0)
    zero_a = (ea == 0); zero_b = (eb == 0)
    ma = (1 << 23) | maf
    mb = (1 << 23) | mbf
    a_ge = ea >= eb
    exp_big = xp.where(a_ge, ea, eb)
    s_big = xp.where(a_ge, sa, sb)
    s_small = xp.where(a_ge, sb, sa)
    m_big = xp.where(a_ge, ma, mb)
    m_small = xp.where(a_ge, mb, ma)
    shamt = xp.where(a_ge, ea - eb, eb - ea)
    m_big_e = m_big << 3
    full = m_small << 3
    shc = xp.where(shamt > 60, 60, shamt)               # clamp; ge27 path overrides
    ms = full >> shc
    lost_mask = (1 << shc) - 1
    sticky_small = xp.where((shamt > 3) & ((full & lost_mask) != 0), 1, 0)
    m_small_e = xp.where(shamt >= 27, xp.where(m_small != 0, 1, 0), ms | sticky_small)
    same_sign = (s_big == s_small)
    big_ge = m_big_e >= m_small_e
    summ = xp.where(same_sign, m_big_e + m_small_e,
                    xp.where(big_ge, m_big_e - m_small_e, m_small_e - m_big_e))
    res_sign = xp.where(same_sign, s_big, xp.where(big_ge, s_big, s_small))
    cancel = (summ == 0)
    exp_s = exp_big
    mag = summ
    carry = ((mag >> 27) & 1) == 1
    exp_s = xp.where(carry, exp_s + 1, exp_s)
    mag = xp.where(carry, mag >> 1, mag)
    for _ in range(27):                                 # bounded leading-zero norm
        cond = (~carry) & (((mag >> 26) & 1) == 0)
        mag = xp.where(cond, mag << 1, mag)
        exp_s = xp.where(cond, exp_s - 1, exp_s)
    mant = (mag >> 3) & 0xFFFFFF
    guard = (mag >> 2) & 1
    round_bit = (mag >> 1) & 1
    sticky = mag & 1
    round_up = guard & (round_bit | sticky | (mant & 1))
    mant_r = mant + round_up
    c2 = ((mant_r >> 24) & 1) == 1
    mant_r = xp.where(c2, mant_r >> 1, mant_r)
    exp_s = xp.where(c2, exp_s + 1, exp_s)
    normal = xp.where(exp_s >= 255, (res_sign << 31) | (0xFF << 23),
              xp.where(exp_s <= 0, (res_sign << 31),
                       (res_sign << 31) | ((exp_s & 0xFF) << 23) | (mant_r & 0x7FFFFF)))
    res = xp.where(cancel, _zeros_like(xp, a), normal)
    # special-value overrides (priority: nan > inf > zero), applied last = highest
    res = xp.where(zero_a & zero_b, ((sa & sb) << 31), res)
    res = xp.where(zero_a & (~zero_b), b & M32, res)
    res = xp.where(zero_b & (~zero_a), a & M32, res)
    res = xp.where(inf_a & inf_b, xp.where(sa == sb, (sa << 31) | (0xFF << 23), 0x7FC00000), res)
    res = xp.where(inf_a & (~inf_b), (sa << 31) | (0xFF << 23), res)
    res = xp.where(inf_b & (~inf_a), (sb << 31) | (0xFF << 23), res)
    res = xp.where(nan_a | nan_b, 0x7FC00000, res)
    return res

# ---- E4M3 encode (fp32 -> E4M3), RNE + saturate ----------------------------
def _fp32_to_fp8e4m3(xp, f):
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    m = f & 0x7FFFFF
    is_nan = (e == 0xFF) & (m != 0)
    is_inf = (e == 0xFF) & (m == 0)
    is_zero = (e == 0)
    E = e - 127
    sig = (1 << 23) | m
    # ---- normal path (E >= -6) ----
    efield0 = E + 7
    g = (m >> 19) & 1
    st = ((m & 0x7FFFF) != 0)
    rup = g & xp.where(st | (((m >> 20) & 1) == 1), 1, 0)
    msum = ((m >> 20) & 0x7) + rup
    msum_carry = ((msum & 0x8) != 0)
    efield = xp.where(msum_carry, efield0 + 1, efield0)
    mant3 = xp.where(msum_carry, 0, msum & 0x7)
    overflow = (efield > 15) | ((efield == 15) & (mant3 == 7))
    normal_res = xp.where(overflow, (s << 7) | 0x7E, (s << 7) | ((efield & 0xF) << 3) | mant3)
    # ---- subnormal path (-10 <= E <= -7) ----
    qi_7 = (sig >> 21) & 0x7; g_7 = (sig >> 20) & 1; st_7 = ((sig & 0xFFFFF) != 0)
    qi_8 = (sig >> 22) & 0x3; g_8 = (sig >> 21) & 1; st_8 = ((sig & 0x1FFFFF) != 0)
    qi_9 = (sig >> 23) & 0x1; g_9 = (sig >> 22) & 1; st_9 = ((sig & 0x3FFFFF) != 0)
    qi_a = _zeros_like(xp, sig);  g_a = (sig >> 23) & 1; st_a = ((sig & 0x7FFFFF) != 0)
    q_int = xp.where(E == -7, qi_7, xp.where(E == -8, qi_8, xp.where(E == -9, qi_9, qi_a)))
    g_sub = xp.where(E == -7, g_7, xp.where(E == -8, g_8, xp.where(E == -9, g_9, g_a)))
    st_sub = xp.where(E == -7, st_7, xp.where(E == -8, st_8, xp.where(E == -9, st_9, st_a)))
    rup_sub = g_sub & xp.where(st_sub | ((q_int & 1) == 1), 1, 0)
    q = q_int + rup_sub
    sub_res = xp.where(q == 0, (s << 7), xp.where(q == 8, (s << 7) | 0x08, (s << 7) | (q & 0x7)))
    # ---- combine by E range ----
    res = xp.where(E >= -6, normal_res,
           xp.where(E <= -11, (s << 7), sub_res))
    res = xp.where(is_zero, (s << 7), res)
    res = xp.where(is_inf, (s << 7) | 0x7E, res)
    res = xp.where(is_nan, (s << 7) | 0x7F, res)
    return res

# ---- E4M3 code -> SIGNED INTEGER significand Av/Wv (the decomposition) -------
def _e4m3_to_signed_sig(xp, x):
    """E4M3 byte array -> signed integer significand v s.t. the EXACT fixed-point
       (weight 2^-18) product of two codes a,b equals va*vb.  Zero code -> 0."""
    s = (x >> 7) & 1
    e = (x >> 3) & 0xF
    m = x & 0x7
    Ma = xp.where(e == 0, m, (0x8 | m))
    PaB = xp.where(e == 0, _zeros_like(xp, x), e - 1)
    mag = Ma << PaB
    return xp.where(s == 1, -mag, mag)

# ---- fixed-point accumulator (weight 2^-18, signed int) -> fp32 (one RNE) ----
def _fixed_to_fp32(xp, x):
    neg = (x < 0)
    mag = xp.where(neg, -x, x)
    s = xp.where(neg, 1, 0)
    is_zero = (mag == 0)
    # p = floor(log2(mag)) for mag>0  (exact highest-set-bit, bounded binary search)
    p = _zeros_like(xp, x)
    n = mag
    for sh in (32, 16, 8, 4, 2, 1):
        c = (n >> sh) > 0
        p = xp.where(c, p + sh, p)
        n = xp.where(c, n >> sh, n)
    e_b = (p - ACC_FRAC) + 127
    hi = (p >= 23)
    sh_hi = xp.where(hi, p - 23, 0)
    shifted = mag >> sh_hi
    mant_hi = shifted & 0xFFFFFF
    sh_g = xp.where(p >= 24, p - 24, 0)
    guard_hi = xp.where(p >= 24, (mag >> sh_g) & 1, 0)
    sticky_mask = (1 << sh_g) - 1
    sticky_hi = xp.where((p >= 25) & ((mag & sticky_mask) != 0), 1, 0)
    sh_lo = xp.where(hi, 0, 23 - p)
    mant_lo = (mag << sh_lo) & 0xFFFFFF
    mant = xp.where(hi, mant_hi, mant_lo)
    guard = xp.where(hi, guard_hi, 0)
    sticky = xp.where(hi, sticky_hi, 0)
    round_up = guard & (sticky | (mant & 1))
    mant_r = mant + round_up
    carry = ((mant_r >> 24) & 1) == 1
    mant_r = xp.where(carry, mant_r >> 1, mant_r)
    e_b = xp.where(carry, e_b + 1, e_b)
    res = xp.where(e_b >= 255, (s << 31) | (0xFF << 23),
           xp.where(e_b <= 0, (s << 31),
                    (s << 31) | ((e_b & 0xFF) << 23) | (mant_r & 0x7FFFFF)))
    res = xp.where(is_zero, _zeros_like(xp, x), res)
    return res

# ============================================================================
# Backend plumbing: convert python-list inputs <-> backend int64 arrays
# ============================================================================
def _as_i64(xp, data):
    if xp is _np:
        return _np.asarray(data, dtype=_np.int64)
    if xp is _torch:
        return _torch.as_tensor(data, dtype=_torch.int64)
    raise RuntimeError("no array backend")

def _to_list(xp, arr):
    if xp is _np:
        return arr.astype(_np.int64).tolist()
    if xp is _torch:
        return arr.to(_torch.int64).tolist()
    raise RuntimeError("no array backend")

# ============================================================================
# Vectorized a_shift derivation (per-token POW2 activation scale)
#   amax via the backend (compare bf16->fp32 magnitudes by their bit pattern,
#   which for finite floats is monotone in value), then the EXACT pow2 exponent
#   floor(log2(448/amax)) per row -- matching glm_fp8_ref.derive_a_shift.
# ============================================================================
def _derive_a_shift_vec(xp, A):
    M = A.shape[0]
    f = _bf16_to_fp32(xp, A)                  # (M,K) fp32 bits
    magbits = f & 0x7FFFFFFF                  # clear sign; monotone in |value|
    amax_bits = magbits.max(axis=1) if xp is _np else magbits.amax(dim=1)
    out = []
    amax_list = _to_list(xp, amax_bits)
    for ub in amax_list:
        amax = ref.u322f(ub & 0xFFFFFFFF)
        out.append(0 if amax == 0.0 else int(math.floor(math.log2(448.0 / amax))))
    return out

# ============================================================================
# THE VECTORIZED [128,128] BLOCK-SCALED DYNAMIC-ACT FP8 GEMM
# ============================================================================
def _block_fp8_gemm_xp(xp, A_bf16, W_fp8, w_scale_inv, blk, a_shift):
    A = _as_i64(xp, A_bf16)                    # (M,K) bf16 codes
    W = _as_i64(xp, W_fp8)                     # (K,N) E4M3 codes
    WS = _as_i64(xp, w_scale_inv)              # (n_ob,n_kb) bf16 codes
    M, K = int(A.shape[0]), int(A.shape[1])
    N = int(W.shape[1])
    n_kblk = (K + blk - 1) // blk

    if a_shift is None:
        a_shift = _derive_a_shift_vec(xp, A)
    ash = _as_i64(xp, a_shift).reshape(M, 1)   # (M,1)

    # --- dynamic FP8 quantization of activations (per-token pow2 scale) ---
    a_f = _bf16_to_fp32(xp, A)                 # (M,K) fp32 bits
    a_fs = _fp32_scale_pow2(xp, a_f, ash)      # * 2^ash  (exponent add)
    a_q = _fp32_to_fp8e4m3(xp, a_fs)           # (M,K) E4M3 codes

    # --- EXACT within-block accumulation == segmented INTEGER MATMUL ---
    Av = _e4m3_to_signed_sig(xp, a_q)          # (M,K) signed int significands
    Wv = _e4m3_to_signed_sig(xp, W)            # (K,N) signed int significands
    WS_fp32 = _bf16_to_fp32(xp, WS)            # (n_ob,n_kb) fp32 bits

    # column j's out-block index, as an int array (N,)
    if xp is _np:
        col_ob = (_np.arange(N, dtype=_np.int64) // blk)
    else:
        col_ob = (_torch.arange(N, dtype=_torch.int64) // blk)

    s = None                                   # running fp32-bits accumulator (M,N)
    for bj in range(n_kblk):
        k0 = bj * blk
        k1 = min(k0 + blk, K)
        acc = Av[:, k0:k1] @ Wv[k0:k1, :]      # (M,N) EXACT int64 (no rounding)
        blk_f = _fixed_to_fp32(xp, acc)        # ONE RNE rounding per block
        # per-column weight scale for this K-block: WS_fp32[col_ob, bj] -> (N,)
        wf_row = WS_fp32[col_ob, bj]           # (N,)
        wf = wf_row.reshape(1, N) + _zeros_like(xp, blk_f)   # broadcast to (M,N)
        prod = _fp32_mul(xp, blk_f, wf)
        s = prod if s is None else _fp32_add(xp, s, prod)
    if s is None:                              # K==0 degenerate
        s = _zeros_like(xp, _as_i64(xp, [[0] * N for _ in range(M)]))

    # --- undo per-token 2^-a_shift (exact), narrow to bf16 ---
    deq = _fp32_scale_pow2(xp, s, -ash)
    C = _fp32_to_bf16(xp, deq)
    return _to_list(xp, C), list(a_shift)

# ============================================================================
# PURE-PYTHON fallback (no numpy/torch): the SAME Av*Wv integer-decomposition
# algorithm, scalar.  Uses glm_fp8_ref's canonical scalar fp32 primitives for
# the fold (so only the decomposition itself is "new") -> an independent
# cross-check of the decomposition against the golden even with no array libs.
# ============================================================================
def _e4m3_to_signed_sig_scalar(x):
    s = (x >> 7) & 1
    e = (x >> 3) & 0xF
    m = x & 0x7
    if e == 0:
        Ma, PaB = m, 0
    else:
        Ma, PaB = (0x8 | m), e - 1
    mag = Ma << PaB
    return -mag if s else mag

def _block_fp8_gemm_py(A_bf16, W_fp8, w_scale_inv, blk, a_shift):
    M = len(A_bf16)
    K = len(W_fp8) if W_fp8 else 0
    N = len(W_fp8[0]) if K else 0
    n_kblk = (K + blk - 1) // blk
    if a_shift is None:
        a_shift = [ref.derive_a_shift(A_bf16[i]) for i in range(M)]
    C = [[0] * N for _ in range(M)]
    Wv = [[_e4m3_to_signed_sig_scalar(W_fp8[k][j]) for j in range(N)] for k in range(K)]
    for i in range(M):
        ash = a_shift[i]
        Av = [0] * K
        for k in range(K):
            a_fs = ref.fp32_scale_pow2(ref.bf16_to_fp32(A_bf16[i][k]), ash)
            Av[k] = _e4m3_to_signed_sig_scalar(ref.fp32_to_fp8e4m3(a_fs))
        for j in range(N):
            acc = [0] * n_kblk
            for k in range(K):
                acc[k // blk] += Av[k] * Wv[k][j]
            out_blk = j // blk
            s = 0
            for bj in range(n_kblk):
                blk_f = ref.fixed_to_fp32(acc[bj])
                wf = ref.bf16_to_fp32(w_scale_inv[out_blk][bj])
                s = ref.fp32_add(s, ref.fp32_mul(blk_f, wf))
            C[i][j] = ref.fp32_to_bf16(ref.fp32_scale_pow2(s, -ash))
    return C, list(a_shift)

# ============================================================================
# PUBLIC GEMM dispatcher
# ============================================================================
def _pick_backend(backend):
    if backend in ("torch", "numpy", "python"):
        return backend
    if backend is not None:
        raise ValueError(f"unknown backend {backend!r}")
    if _HAS_TORCH:
        return "torch"
    if _HAS_NP:
        return "numpy"
    return "python"

def block_fp8_gemm(A_bf16, W_fp8, w_scale_inv, blk=128, a_shift=None, backend=None):
    """C[M,N] = A[M,K] @ W[K,N] in the GLM-5.2-FP8 contract, BIT-EXACT to
    glm_fp8_ref.block_fp8_gemm (and to src/glm_matmul_fp8.v).

      A_bf16     : [M][K] bf16 codes (16-bit) -- activations.
      W_fp8      : [K][N] E4M3 codes (8-bit)  -- pre-quantized weights, contraction
                   orientation W[k][n] (n = output channel).
      w_scale_inv: [n_out_blk][n_k_blk] bf16 codes -- one block scale per
                   128(out)x128(K) block; column n uses w_scale_inv[n//blk][bj].
      a_shift    : optional [M] per-token POW2 act scales (signed int); None ->
                   derived per row (largest pow2 keeping the row inside E4M3 range).
      backend    : 'torch' | 'numpy' | 'python' | None (auto: torch>numpy>python).

    Returns (C, a_shift): C is [M][N] bf16 codes.  Inputs may be python lists or
    numpy/torch int arrays (they are coerced)."""
    bk = _pick_backend(backend)
    if a_shift is not None:
        a_shift = [int(v) for v in (a_shift.tolist() if hasattr(a_shift, "tolist") else a_shift)]
    if bk == "torch":
        return _block_fp8_gemm_xp(_torch, A_bf16, W_fp8, w_scale_inv, blk, a_shift)
    if bk == "numpy":
        return _block_fp8_gemm_xp(_np, A_bf16, W_fp8, w_scale_inv, blk, a_shift)
    # python fallback needs nested lists
    def _ll(x):
        return x.tolist() if hasattr(x, "tolist") else x
    return _block_fp8_gemm_py(_ll(A_bf16), _ll(W_fp8), _ll(w_scale_inv), blk, a_shift)

# ============================================================================
# ============================  SELF-TEST  ===================================
# ============================================================================
def _lcg(seed):
    x = seed & 0xFFFFFFFF
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield x

def _rand_bf16(g, lo_exp=-4, hi_exp=4):
    """A bf16 code with a controlled magnitude range (sign+exp+mant from the LCG)."""
    s = next(g) & 1
    e = 127 + (lo_exp + (next(g) % (hi_exp - lo_exp + 1)))
    m = next(g) & 0x7F
    return ((s << 15) | ((e & 0xFF) << 7) | m) & 0xFFFF

def _rand_fp8(g):
    c = next(g) & 0xFF
    if (c & 0x7F) == 0x7F:                  # avoid the single NaN code (outside contract)
        c ^= 0x01
    return c

def _make_tile(seed, M, K, N, exp_lo, exp_hi, scale_lo, scale_hi):
    g = _lcg(seed)
    A = [[_rand_bf16(g, exp_lo, exp_hi) for _ in range(K)] for _ in range(M)]
    W = [[_rand_fp8(g) for _ in range(N)] for _ in range(K)]
    n_ob = (N + 128 - 1) // 128
    n_kb = (K + 128 - 1) // 128
    WS = [[ref.fp32_to_bf16(ref.f2u32(scale_lo + (scale_hi - scale_lo) * ((next(g) % 1000) / 1000.0)))
           for _ in range(n_kb)] for _ in range(n_ob)]
    return A, W, WS

def _tile_suite():
    """A representative suite mirroring the bit-accuracy generator pattern:
       generic ranges, near-saturation, tiny/subnormal, all-zero row, single
       K-block, lopsided two-block scales, multi-out-block (N>128)."""
    S = []
    S.append(("generic-small",  _make_tile(0xA1, 3, 4,   2, -3, 3,  0.5, 2.0)))
    S.append(("generic-mid",    _make_tile(0xB2, 5, 64,  8, -2, 4,  0.01, 0.05)))
    S.append(("one-kblock-128", _make_tile(0xC3, 4, 128, 6, -4, 5,  0.02, 0.2)))
    S.append(("two-kblock-200", _make_tile(0xD4, 4, 200, 5, -3, 4,  0.005, 0.03)))
    S.append(("near-sat-448",   _make_tile(0xE5, 3, 32,  4,  6, 8,  0.4, 0.5)))
    S.append(("tiny-subnormal", _make_tile(0xF6, 3, 48,  4, -12, -8, 0.001, 0.002)))
    S.append(("multi-outblk",   _make_tile(0x17, 4, 96, 160, -2, 3,  0.01, 0.04)))
    S.append(("wide-K-300",     _make_tile(0x28, 6, 300, 4, -3, 5,  0.008, 0.06)))
    # all-zero activation row (degenerate a_shift / argmax tie)
    name, (A, W, WS) = "all-zero-row", _make_tile(0x39, 3, 40, 5, -2, 3, 0.02, 0.1)
    A[0] = [0] * len(A[0])
    S.append((name, (A, W, WS)))
    # lopsided two-block scales: force 2^-7 vs 2^3 on the two K-blocks
    name, (A, W, WS) = "lopsided-scales", _make_tile(0x4A, 4, 160, 6, -3, 4, 0.01, 0.01)
    for ob in range(len(WS)):
        WS[ob][0] = ref.fp32_to_bf16(ref.f2u32(2.0 ** -7))
        if len(WS[ob]) > 1:
            WS[ob][1] = ref.fp32_to_bf16(ref.f2u32(2.0 ** 3))
    S.append((name, (A, W, WS)))
    return S

def _f64_ref_argmax(A, W, WS, a_shift, blk=128):
    """A near-exact float64 reference: decode E4M3 operands to float, apply the
       SAME per-token pow2 act-quant + block scale, accumulate in float64.  Used
       only for the argmax (next-token) cross-check, not for bit-exactness."""
    M = len(A); K = len(W); N = len(W[0]) if K else 0
    out = []
    for i in range(M):
        ash = a_shift[i]
        row = []
        Aq = []
        for k in range(K):
            af = ref.u322f(ref.bf16_to_fp32(A[i][k])) * (2.0 ** ash)
            aq = ref.fp8_e4m3_decode(ref.fp8_e4m3_encode(af))   # E4M3-rounded
            Aq.append(aq)
        for j in range(N):
            ob = j // blk
            acc = 0.0
            for bj in range((K + blk - 1) // blk):
                k0, k1 = bj * blk, min(bj * blk + blk, K)
                wf = ref.u322f(ref.bf16_to_fp32(WS[ob][bj]))
                seg = 0.0
                for k in range(k0, k1):
                    seg += Aq[k] * ref.fp8_e4m3_decode(W[k][j])
                acc += wf * seg
            row.append(acc * (2.0 ** -ash))
        out.append(row)
    return out

def _argmax(row):
    bi, bv = 0, None
    for j, v in enumerate(row):
        if bv is None or v > bv:
            bv, bi = v, j
    return bi

def _selftest():
    backends = ["python"]
    if _HAS_NP:
        backends.append("numpy")
    if _HAS_TORCH:
        backends.append("torch")
    print(f"backends available: {backends}  (numpy={_HAS_NP} torch={_HAS_TORCH})")

    suite = _tile_suite()
    n_pass = 0
    n_fail = 0

    # ---- (T1) E4M3 encode/decode round-trip over all 256 codes (per backend) ----
    for bk in backends:
        rt_ok = True
        if bk == "python":
            for code in range(256):
                e = (code >> 3) & 0xF; m = code & 0x7
                if (e == 0xF and m == 0x7):
                    continue                       # NaN canonical slot
                f = ref.fp8e4m3_to_fp32(code)
                if ref.fp32_to_fp8e4m3(f) != code:
                    rt_ok = False; break
        else:
            xp = _np if bk == "numpy" else _torch
            codes = _as_i64(xp, list(range(256)))
            f = _zeros_like(xp, codes)             # build fp32 of each code via ref (golden) then re-encode
            # decode via golden (scalar) to fp32 bits, then vectorized re-encode
            fbits = _as_i64(xp, [ref.fp8e4m3_to_fp32(c) for c in range(256)])
            re = _to_list(xp, _fp32_to_fp8e4m3(xp, fbits))
            for code in range(256):
                e = (code >> 3) & 0xF; m = code & 0x7
                if (e == 0xF and m == 0x7):
                    continue
                if re[code] != code:
                    rt_ok = False; break
        print(f"  [T1:{bk}] E4M3 256-code round-trip: {'PASS' if rt_ok else 'FAIL'}")
        n_pass += rt_ok; n_fail += (not rt_ok)

    # ---- (T2) vectorized fp32 primitive kernels vs the golden scalar (np/torch) ----
    for bk in backends:
        if bk == "python":
            continue
        xp = _np if bk == "numpy" else _torch
        g = _lcg(0x5151)
        # random fp32 patterns incl. zeros/normals across exponents
        vals = []
        for _ in range(4000):
            s = next(g) & 1
            e = next(g) % 256
            if e in (0, 0xFF):                      # keep to finite normals for op cross-check
                e = 1 + (next(g) % 254)
            m = next(g) & 0x7FFFFF
            vals.append((s << 31) | (e << 23) | m)
        a = _as_i64(xp, vals)
        b = _as_i64(xp, list(reversed(vals)))
        mul_v = _to_list(xp, _fp32_mul(xp, a, b))
        add_v = _to_list(xp, _fp32_add(xp, a, b))
        bf_v = _to_list(xp, _fp32_to_bf16(xp, a))
        ks = _as_i64(xp, [((i % 21) - 10) for i in range(len(vals))])
        sc_v = _to_list(xp, _fp32_scale_pow2(xp, a, ks))
        # fixed_to_fp32 over signed integers spanning the accumulator range
        ints = [((next(g) << 20) | next(g)) * (1 if next(g) & 1 else -1) for _ in range(2000)]
        fx_v = _to_list(xp, _fixed_to_fp32(xp, _as_i64(xp, ints)))
        ok = True
        for i in range(len(vals)):
            if mul_v[i] != ref.fp32_mul(vals[i], vals[len(vals) - 1 - i]): ok = False; break
            if add_v[i] != ref.fp32_add(vals[i], vals[len(vals) - 1 - i]): ok = False; break
            if bf_v[i] != ref.fp32_to_bf16(vals[i]): ok = False; break
            if sc_v[i] != ref.fp32_scale_pow2(vals[i], ((i % 21) - 10)): ok = False; break
        if ok:
            for i in range(len(ints)):
                if fx_v[i] != ref.fixed_to_fp32(ints[i]): ok = False; break
        print(f"  [T2:{bk}] vectorized fp32 mul/add/scale/bf16/fixed vs golden: {'PASS' if ok else 'FAIL'}")
        n_pass += ok; n_fail += (not ok)

    # ---- (T3) block_fp8_gemm BIT-EXACT vs glm_fp8_ref, every backend, every tile ----
    n_tiles = 0
    for bk in backends:
        bk_ok = True
        for name, (A, W, WS) in suite:
            gold, ash_g = ref.block_fp8_gemm(A, W, WS, blk=128)
            got, ash = block_fp8_gemm(A, W, WS, blk=128, backend=bk)
            n_tiles += 1
            if ash != ash_g or got != gold:
                bk_ok = False
                # locate first mismatch
                for i in range(len(gold)):
                    for j in range(len(gold[0])):
                        if got[i][j] != gold[i][j]:
                            print(f"    [{bk}/{name}] MISMATCH C[{i}][{j}] "
                                  f"{got[i][j]:#06x} != {gold[i][j]:#06x} (a_shift {ash}/{ash_g})")
                            break
                    else:
                        continue
                    break
            tag = "PASS" if (ash == ash_g and got == gold) else "FAIL"
            ok = (ash == ash_g and got == gold)
            n_pass += ok; n_fail += (not ok)
        print(f"  [T3:{bk}] block_fp8_gemm vs golden over {len(suite)} tiles: "
              f"{'ALL PASS' if bk_ok else 'FAIL'}")

    # ---- (T4) argmax-preserving vs a float64 near-exact reference -------------
    am_ok = True
    am_rows = 0
    for name, (A, W, WS) in suite:
        gold, ash = ref.block_fp8_gemm(A, W, WS, blk=128)
        f64 = _f64_ref_argmax(A, W, WS, ash, blk=128)
        for i in range(len(gold)):
            if all(c == 0 for c in A[i]):           # all-zero row -> degenerate tie, skip
                continue
            am_rows += 1
            g_row = [ref.u322f(ref.bf16_to_fp32(c)) for c in gold[i]]
            if _argmax(g_row) != _argmax(f64[i]):
                am_ok = False
                print(f"    [argmax/{name}] row {i}: contract={_argmax(g_row)} f64={_argmax(f64[i])}")
    print(f"  [T4] argmax(contract bf16) == argmax(float64 ref) over {am_rows} rows: "
          f"{'PASS' if am_ok else 'FAIL'}")
    n_pass += am_ok; n_fail += (not am_ok)

    total = n_pass + n_fail
    print(f"bit-exact GEMM tile checks: {n_tiles} (across {len(backends)} backend(s))")
    if n_fail == 0:
        print(f"ALL {total} TESTS PASSED")
        return 0
    print(f"FAILED {n_fail}/{total} TESTS")
    return 1


if __name__ == "__main__":
    sys.exit(_selftest())
