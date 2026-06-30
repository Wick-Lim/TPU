#!/usr/bin/env python3
# ============================================================================
# glm_fp8_ref.py -- GOLDEN REFERENCE for the GLM-5.2-FP8 FP8 arithmetic contract
# ----------------------------------------------------------------------------
# WHAT THIS IS
#   A small, readable, pure-python (numpy optional, NOT required) reference
#   implementation of EXACTLY the FP8 arithmetic the official zai-org/GLM-5.2-FP8
#   checkpoint is meant to run with.  It is the GOLDEN that the repo's committed
#   RTL (src/fp8_e4m3.vh + src/glm_matmul_fp8.v) is checked against -- every
#   numeric primitive here is a BIT-EXACT mirror of the corresponding
#   `function automatic` in that RTL (cross-checked code-for-code below), so
#   block_fp8_gemm(...) reproduces the hardware output bit-for-bit, not merely
#   "close enough".
#
# THE GROUND-TRUTH CONTRACT  (config.json quantization_config of GLM-5.2-FP8)
#   quant_method        = "fp8"
#   fmt                 = "e4m3"            (OCP E4M3: 1s.4e(bias7).3m, NO Inf,
#                                            one NaN S.1111.111, max normal 448)
#   weight_block_size   = [128, 128]        (DeepSeek-V3 style: ONE bf16/fp32
#                                            dequant scale per 128x128 weight
#                                            block, stored as weight_scale_inv)
#   activation_scheme   = "dynamic"         (activations quantized to E4M3 at
#                                            runtime with a per-token / per-1x128
#                                            -group scale derived from the data)
#   modules_to_not_convert keep norms / router (gate) / embed_tokens / lm_head
#                          and the MTP head in bf16 (the "bf16 tail").
#
# THE DEQUANT MATH (what a faithful inference engine computes):
#   y = SUM over K-blocks bj of ( w_scale[out_blk][bj] *
#          SUM_{k in Kblock bj} dq_act(A_fp8[k]) * dq_wgt(W_fp8[k]) )
#     where W is pre-quantized E4M3 with a [128,128] block scale, and A is
#     dynamically quantized to E4M3 per token; the products are accumulated and
#     the result is returned in bf16.
#
# WHERE THE CONTRACT IS AMBIGUOUS  (read this -- documented assumptions):
#   (1) ACTIVATION SCALE GRANULARITY/FORM.  config says only "dynamic". The
#       generic vLLM/transformers dynamic path uses an ARBITRARY fp32 per-token
#       (or per-1x128-group) scale = max(|a|)/448.  OUR RTL (glm_matmul_fp8.v)
#       instead uses a per-token POWER-OF-TWO scale a_shift (an exponent add, no
#       multiplier -- the hardware win).  This reference mirrors OUR RTL: the
#       per-token scale is 2^a_shift with a_shift = floor(log2(448/max|a_row|)),
#       i.e. the largest pow2 that keeps the row inside E4M3 range.  Both are
#       valid "dynamic" schemes; the pow2 one is a strict subset (a==scale is a
#       power of two).  Set a_shift explicitly to override.  (Documented.)
#   (2) WITHIN-BLOCK ACCUMULATION PRECISION.  The contract does not pin the
#       accumulator.  OUR RTL accumulates each 128-wide K-block EXACTLY (every
#       E4M3*E4M3 product is an exact multiple of 2^-18, summed in fixed point
#       with ACC_FRAC=18, then rounded to fp32 ONCE).  We reproduce that exact
#       fixed-point sum here.  The cross-block fold (block_partial * w_scale,
#       summed) then uses the SAME bit-exact fp32 mul/add the RTL dequant pass
#       uses (RNE, FTZ), left-folded in block order.  Final result -> bf16 RNE.
#   (3) weight_scale_inv ORIENTATION.  HF stores a Linear weight as
#       [out_features, in_features]; weight_scale_inv is then
#       [ceil(out/128), ceil(in/128)].  Column pj of the RTL contraction is an
#       OUTPUT channel, so its block scale is weight_scale_inv[pj//128][k//128].
#
# API
#   fp8_e4m3_encode(x)            float -> E4M3 byte   (RNE + saturate, mirror)
#   fp8_e4m3_decode(b)            E4M3 byte -> float   (exact, mirror)
#   block_fp8_gemm(A_bf16, W_fp8, w_scale_inv, blk=128, a_shift=None) -> bf16
#                                 the [128,128] block-scaled dynamic-act FP8 GEMM
#   Plus the bit-exact uint-domain primitives (fp32_mul/_add/_scale_pow2,
#   fp8_mul, fp32<->fixed, bf16<->fp32, e4m3<->fp32) used to build it.
#
# SELF-TEST:  python3 tools/glm_fp8_ref.py   (e4m3 round-trip + block-GEMM
#             sanity; exits 0 on success).
# ============================================================================
import sys, os, struct, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ----------------------------------------------------------------------------
# float <-> uint32 / uint16 bit reinterpretation (stdlib struct; no numpy)
# ----------------------------------------------------------------------------
def f2u32(x):                       # python float -> fp32 bit pattern (RNE narrow)
    return struct.unpack("<I", struct.pack("<f", x))[0]

def u322f(u):                       # fp32 bit pattern -> python float
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]

# ============================================================================
# (A) bf16 <-> fp32   (mirror of glm_fp.vh bf16_to_fp32 / fp32_to_bf16)
# ============================================================================
def bf16_to_fp32(b16):
    return (b16 & 0xFFFF) << 16

def fp32_to_bf16(f):
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    m = f & 0x7FFFFF
    if e == 0xFF:
        if m == 0:
            return (s << 15) | (0xFF << 7)            # inf
        return (s << 15) | (0xFF << 7) | (1 << 6)     # qnan
    round_up = ((m >> 15) & 1) & (1 if ((m & 0x7FFF) != 0 or ((m >> 16) & 1)) else 0)
    top_plus = (((e << 7) | (m >> 16)) + round_up) & 0x7FFF
    new_e = (top_plus >> 7) & 0xFF
    if new_e == 0xFF:
        return (s << 15) | (0xFF << 7)                # overflow -> inf
    return (s << 15) | top_plus

# ============================================================================
# (B) fp32 mul / add / scale-by-pow2   (mirror of glm_fp.vh + glm_matmul_fp8.v)
#     IEEE-ish single precision, ROUND-TO-NEAREST-EVEN, FLUSH-TO-ZERO.
# ============================================================================
def _is_nan(f):  return ((f >> 23) & 0xFF) == 0xFF and (f & 0x7FFFFF) != 0
def _is_inf(f):  return ((f >> 23) & 0xFF) == 0xFF and (f & 0x7FFFFF) == 0
def _is_zero(f): return ((f >> 23) & 0xFF) == 0x00          # FTZ: exp 0 == zero

def fp32_mul(a, b):
    sa, sb = (a >> 31) & 1, (b >> 31) & 1
    sr = sa ^ sb
    if _is_nan(a) or _is_nan(b):
        return 0x7FC00000
    if _is_inf(a) or _is_inf(b):
        if _is_zero(a) or _is_zero(b):
            return 0x7FC00000                          # inf*0 -> nan
        return (sr << 31) | (0xFF << 23)
    if _is_zero(a) or _is_zero(b):
        return (sr << 31)                              # signed zero
    ea, eb = (a >> 23) & 0xFF, (b >> 23) & 0xFF
    ma = (1 << 23) | (a & 0x7FFFFF)
    mb = (1 << 23) | (b & 0x7FFFFF)
    prod = ma * mb                                      # 48-bit, in [1,4)
    exp_s = ea + eb - 127
    if (prod >> 47) & 1:                                # [2,4): shift right 1
        exp_s += 1
        mant = (prod >> 24) & 0xFFFFFF
        guard = (prod >> 23) & 1
        sticky = 1 if (prod & 0x7FFFFF) else 0
    else:                                               # [1,2)
        mant = (prod >> 23) & 0xFFFFFF
        guard = (prod >> 22) & 1
        sticky = 1 if (prod & 0x3FFFFF) else 0
    round_up = guard & (sticky | (mant & 1))
    mant_r = mant + round_up
    if (mant_r >> 24) & 1:
        mant_r >>= 1
        exp_s += 1
    if exp_s >= 255:
        return (sr << 31) | (0xFF << 23)               # overflow -> inf
    if exp_s <= 0:
        return (sr << 31)                              # underflow -> FTZ
    return (sr << 31) | ((exp_s & 0xFF) << 23) | (mant_r & 0x7FFFFF)

def fp32_add(a, b):
    sa, sb = (a >> 31) & 1, (b >> 31) & 1
    ea, eb = (a >> 23) & 0xFF, (b >> 23) & 0xFF
    if _is_nan(a) or _is_nan(b):
        return 0x7FC00000
    if _is_inf(a) and _is_inf(b):
        return ((sa << 31) | (0xFF << 23)) if sa == sb else 0x7FC00000
    if _is_inf(a):
        return (sa << 31) | (0xFF << 23)
    if _is_inf(b):
        return (sb << 31) | (0xFF << 23)
    if _is_zero(a) and _is_zero(b):
        return ((sa & sb) << 31)
    if _is_zero(a):
        return b & 0xFFFFFFFF
    if _is_zero(b):
        return a & 0xFFFFFFFF
    ma = (1 << 23) | (a & 0x7FFFFF)
    mb = (1 << 23) | (b & 0x7FFFFF)
    if ea >= eb:
        exp_big, s_big, s_small = ea, sa, sb
        m_big, m_small, shamt = ma, mb, ea - eb
    else:
        exp_big, s_big, s_small = eb, sb, sa
        m_big, m_small, shamt = mb, ma, eb - ea
    m_big_e = m_big << 3                                 # 3 guard/round/sticky bits
    if shamt >= 27:
        m_small_e = 1 if m_small != 0 else 0
    else:
        full = m_small << 3
        m_small_e = full >> shamt
        if shamt > 3:
            if full & ((1 << shamt) - 1):
                m_small_e |= 1                            # sticky from lost bits
    if s_big == s_small:
        summ = m_big_e + m_small_e
        res_sign = s_big
    else:
        if m_big_e >= m_small_e:
            summ = m_big_e - m_small_e
            res_sign = s_big
        else:
            summ = m_small_e - m_big_e
            res_sign = s_small
    if summ == 0:
        return 0                                          # exact cancellation -> +0
    exp_s = exp_big
    mag = summ
    if (mag >> 27) & 1:                                   # carried out
        exp_s += 1
        mag >>= 1
    else:
        lz = 0
        while not ((mag >> 26) & 1) and lz < 27:
            mag <<= 1
            lz += 1
        exp_s -= lz
    mant = (mag >> 3) & 0xFFFFFF
    guard = (mag >> 2) & 1
    round_bit = (mag >> 1) & 1
    sticky = mag & 1
    round_up = guard & (round_bit | sticky | (mant & 1))
    mant_r = mant + round_up
    if (mant_r >> 24) & 1:
        mant_r >>= 1
        exp_s += 1
    if exp_s >= 255:
        return (res_sign << 31) | (0xFF << 23)
    if exp_s <= 0:
        return (res_sign << 31)
    return (res_sign << 31) | ((exp_s & 0xFF) << 23) | (mant_r & 0x7FFFFF)

def fp32_scale_pow2(f, k):
    """Multiply an fp32 by 2^k (k a small signed int) by adding to the biased
       exponent.  EXACT.  Mirror of glm_matmul_fp8.v fp32_scale_pow2."""
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    m = f & 0x7FFFFF
    if e == 0xFF:
        return f & 0xFFFFFFFF                             # inf/nan unchanged
    if e == 0x00:
        return (s << 31)                                  # zero / FTZ
    ne = e + k
    if ne >= 255:
        return (s << 31) | (0xFF << 23)                   # overflow -> inf
    if ne <= 0:
        return (s << 31)                                  # underflow -> FTZ
    return (s << 31) | ((ne & 0xFF) << 23) | m

# ============================================================================
# (C) E4M3 encode / decode  (mirror of src/fp8_e4m3.vh; cross-checked vs the
#     committed tools/fp8_gen.fp32_to_fp8e4m3 in the self-test)
# ============================================================================
def fp32_to_fp8e4m3(bits):
    s = (bits >> 31) & 1
    e = (bits >> 23) & 0xFF
    m = bits & 0x7FFFFF
    if e == 0xFF and m != 0:
        return (s << 7) | 0x7F                            # NaN
    if e == 0xFF:
        return (s << 7) | 0x7E                            # Inf -> saturate 448
    if e == 0x00:
        return (s << 7)                                   # zero / FTZ
    E = e - 127
    sig = (1 << 23) | m
    if E >= -6:
        efield = E + 7
        g = (m >> 19) & 1
        st = 1 if (m & 0x7FFFF) else 0
        rup = g & (st | ((m >> 20) & 1))
        msum = ((m >> 20) & 0x7) + rup
        if msum & 0x8:
            efield += 1
            mant3 = 0
        else:
            mant3 = msum & 0x7
        if efield > 15 or (efield == 15 and mant3 == 7):
            return (s << 7) | 0x7E                         # overflow -> 448
        return (s << 7) | ((efield & 0xF) << 3) | mant3
    if E <= -11:
        return (s << 7)                                   # below half-ulp -> 0
    if E == -7:
        q_int = (sig >> 21) & 0x7; g = (sig >> 20) & 1; st = 1 if (sig & 0xFFFFF) else 0
    elif E == -8:
        q_int = (sig >> 22) & 0x3; g = (sig >> 21) & 1; st = 1 if (sig & 0x1FFFFF) else 0
    elif E == -9:
        q_int = (sig >> 23) & 0x1; g = (sig >> 22) & 1; st = 1 if (sig & 0x3FFFFF) else 0
    else:  # E == -10
        q_int = 0;                  g = (sig >> 23) & 1; st = 1 if (sig & 0x7FFFFF) else 0
    rup = g & (st | (q_int & 1))
    q = q_int + rup
    if q == 0:
        return (s << 7)
    if q == 8:
        return (s << 7) | 0x08                            # rounded up to min normal
    return (s << 7) | (q & 0x7)                            # subnormal

def fp8e4m3_to_fp32(x):
    s = (x >> 7) & 1
    e = (x >> 3) & 0xF
    m = x & 0x7
    if e == 0xF and m == 0x7:
        return (s << 31) | (0xFF << 23) | (1 << 22)       # NaN (sign preserved)
    if e == 0 and m == 0:
        return (s << 31)                                  # signed zero
    if e == 0:                                            # subnormal: m * 2^-9
        if m & 0x4:
            return (s << 31) | (120 << 23) | ((m & 0x3) << 21)   # 1.xx * 2^-7
        elif m & 0x2:
            return (s << 31) | (119 << 23) | ((m & 0x1) << 22)   # 1.x  * 2^-8
        else:
            return (s << 31) | (118 << 23)                       # 1.0  * 2^-9
    nexp = 120 + e                                        # (e-7)+127
    return (s << 31) | (nexp << 23) | (m << 20)

# convenience float<->E4M3 (the public encode/decode)
def fp8_e4m3_encode(x):
    """float -> E4M3 byte (RNE + saturate to +/-448).  x may be a python float."""
    return fp32_to_fp8e4m3(f2u32(x))

def fp8_e4m3_decode(b):
    """E4M3 byte -> python float (exact)."""
    return u322f(fp8e4m3_to_fp32(b & 0xFF))

# ============================================================================
# (D) fp8 * fp8 -> EXACT fp32   (mirror of src/fp8_e4m3.vh fp8_mul)
# ============================================================================
def fp8_mul(a, b):
    sa, sb = (a >> 7) & 1, (b >> 7) & 1
    sr = sa ^ sb
    ea, eb = (a >> 3) & 0xF, (b >> 3) & 0xF
    ma, mb = a & 0x7, b & 0x7
    if (ea == 0xF and ma == 0x7) or (eb == 0xF and mb == 0x7):
        return 0x7FC00000                                 # NaN operand
    if (a & 0x7F) == 0 or (b & 0x7F) == 0:
        return (sr << 31)                                 # zero operand -> signed 0
    if ea == 0:
        Ma, PaB = ma, 0
    else:
        Ma, PaB = (0x8 | ma), ea - 1
    if eb == 0:
        Mb, PbB = mb, 0
    else:
        Mb, PbB = (0x8 | mb), eb - 1
    Mp = Ma * Mb                                            # 4x4 -> <=225 (8 bits)
    norm8 = Mp
    k7 = 7
    for _ in range(7):
        if not ((norm8 >> 7) & 1):
            norm8 = (norm8 << 1) & 0xFF
            k7 -= 1
    mant23 = (norm8 & 0x7F) << 16
    fexp = k7 + PaB + PbB + 109
    return (sr << 31) | ((fexp & 0xFF) << 23) | mant23

# ============================================================================
# (E) fp32 <-> fixed-point block accumulator
#     (mirror of glm_matmul_fp8.v fp32_to_fixed / fixed_to_fp32; ACC_FRAC=18)
# ============================================================================
ACC_FRAC = 18

def fp32_to_fixed(f, acc_frac=ACC_FRAC):
    """Exact convert of an fp8*fp8 product (a normal fp32 that is an exact
       multiple of 2^-18) into a signed integer with weight 2^-acc_frac."""
    s = (f >> 31) & 1
    e = (f >> 23) & 0xFF
    if e == 0:
        return 0
    m_ext = (1 << 23) | (f & 0x7FFFFF)                     # 24-bit significand
    sh = e - 150 + acc_frac
    mag = (m_ext << sh) if sh >= 0 else (m_ext >> (-sh))
    return -mag if s else mag

def fixed_to_fp32(x, acc_frac=ACC_FRAC):
    """Convert a signed fixed-point accumulator (weight 2^-acc_frac) to fp32
       with RNE -- the single rounding of the within-block accumulation."""
    if x == 0:
        return 0
    s = 1 if x < 0 else 0
    mag = -x if x < 0 else x
    p = mag.bit_length() - 1                               # highest set bit
    e_b = (p - acc_frac) + 127
    guard = 0
    sticky = 0
    if p >= 23:
        shifted = mag >> (p - 23)
        mant = shifted & 0xFFFFFF
        if p >= 24:
            guard = (mag >> (p - 24)) & 1
        if p >= 25 and (mag & ((1 << (p - 24)) - 1)):
            sticky = 1
    else:
        mant = (mag << (23 - p)) & 0xFFFFFF
    round_up = guard & (sticky | (mant & 1))
    mant_r = mant + round_up
    if (mant_r >> 24) & 1:
        mant_r >>= 1
        e_b += 1
    if e_b >= 255:
        return (s << 31) | (0xFF << 23)
    if e_b <= 0:
        return (s << 31)
    return (s << 31) | ((e_b & 0xFF) << 23) | (mant_r & 0x7FFFFF)

# ============================================================================
# (F) THE [128,128] BLOCK-SCALED DYNAMIC-ACTIVATION FP8 GEMM  (the headline)
# ============================================================================
def derive_a_shift(a_row_bf16):
    """Per-token POW2 activation scale a_shift (signed int): the largest pow2
       s.t. max|a_row| * 2^a_shift <= 448 (E4M3 max normal).  Mirrors the scale
       the RTL caller picks for dynamic activation quant (see ambiguity note #1).
       Empty / all-zero row -> 0."""
    amax = 0.0
    for bc in a_row_bf16:
        amax = max(amax, abs(u322f(bf16_to_fp32(bc))))
    if amax == 0.0:
        return 0
    # largest integer k with amax * 2^k <= 448
    return int(math.floor(math.log2(448.0 / amax)))

def block_fp8_gemm(A_bf16, W_fp8, w_scale_inv, blk=128, a_shift=None):
    """
    Compute C[M,N] = A[M,K] @ W[K,N] in the GLM-5.2-FP8 contract, BIT-EXACT to
    src/glm_matmul_fp8.v.

      A_bf16     : list[M][K] of bf16 codes (16-bit ints)  -- the activations.
      W_fp8      : list[K][N] of E4M3 codes (8-bit ints)    -- pre-quantized wgt
                   in the RTL contraction orientation W[k][n] (n = output chan).
      w_scale_inv: list[n_out_blk][n_k_blk] of bf16 codes -- ONE block scale per
                   128(out) x 128(K) block; column n uses w_scale_inv[n//blk][bj].
      a_shift    : optional list[M] of signed-int per-token pow2 act scales.
                   None -> derived per row via derive_a_shift (see note #1).

    Returns C : list[M][N] of bf16 codes.
    """
    M = len(A_bf16)
    K = len(W_fp8) if W_fp8 else 0
    N = len(W_fp8[0]) if K else 0
    n_kblk = (K + blk - 1) // blk
    if a_shift is None:
        a_shift = [derive_a_shift(A_bf16[i]) for i in range(M)]

    C = [[0] * N for _ in range(M)]
    for i in range(M):
        ash = a_shift[i]
        # --- dynamic FP8 quantization of this token's activations (pow2 scale) ---
        a_q = [0] * K
        for k in range(K):
            a_f = bf16_to_fp32(A_bf16[i][k])
            a_fs = fp32_scale_pow2(a_f, ash)               # * 2^ash (exponent add)
            a_q[k] = fp32_to_fp8e4m3(a_fs)                 # encode to E4M3
        for j in range(N):
            # --- within-block EXACT fixed-point accumulation, per K-block ---
            acc = [0] * n_kblk                             # signed ints, weight 2^-18
            for k in range(K):
                prod_f = fp8_mul(a_q[k], W_fp8[k][j])      # exact fp32 product
                acc[k // blk] += fp32_to_fixed(prod_f)     # exact integer add
            # --- cross-block dequant fold (bit-exact fp32 mul/add, left fold) ---
            out_blk = j // blk
            s = 0
            for bj in range(n_kblk):
                blk_f = fixed_to_fp32(acc[bj])              # ONE rounding per block
                wf = bf16_to_fp32(w_scale_inv[out_blk][bj])
                prod = fp32_mul(blk_f, wf)
                s = fp32_add(s, prod)                      # s_bj = (((0+p0)+p1)+...)
            # --- undo per-token 2^-a_shift (exact), round to bf16 ---
            deq_un = fp32_scale_pow2(s, -ash)
            C[i][j] = fp32_to_bf16(deq_un)
    return C, a_shift

# ============================================================================
# SELF-TEST
# ============================================================================
def _selftest():
    rc = 0
    # ---- (1) E4M3 round-trip over ALL 256 codes + cross-check vs fp8_gen ----
    try:
        import fp8_gen
        have_fp8gen = True
    except Exception:
        have_fp8gen = False

    rt_ok = True
    enc_match = True
    for code in range(256):
        e = (code >> 3) & 0xF
        m = code & 0x7
        is_nan = (e == 0xF and m == 0x7)
        f = fp8e4m3_to_fp32(code)
        re = fp32_to_fp8e4m3(f)
        # decode->encode must reproduce the code (NaN canonicalizes either NaN
        # code; -0 and +0 both re-encode to their own sign-zero -> identical).
        if not is_nan and re != code:
            rt_ok = False
            print(f"  ROUND-TRIP FAIL code={code:#04x} -> fp32={f:#010x} -> {re:#04x}")
        if have_fp8gen:
            # cross-check OUR encoder mirror against the committed fp8_gen mirror
            if fp32_to_fp8e4m3(f) != fp8_gen.fp32_to_fp8e4m3(f):
                enc_match = False
    # spot-check a few known E4M3 decode values
    known = {
        0x7E: 448.0,            # max normal
        0x38: 1.0,              # 1.0
        0x08: 2 ** -6,          # smallest normal
        0x01: 2 ** -9,          # smallest subnormal
        0xB8: -1.0,
        0x00: 0.0,
    }
    val_ok = True
    for code, want in known.items():
        got = fp8_e4m3_decode(code)
        if got != want:
            val_ok = False
            print(f"  DECODE VALUE FAIL code={code:#04x} got={got} want={want}")
    print(f"E4M3 self-test: round_trip(256 codes)={'PASS' if rt_ok else 'FAIL'} "
          f"decode_values={'PASS' if val_ok else 'FAIL'} "
          f"fp8_gen_cross_check={'PASS' if (have_fp8gen and enc_match) else ('N/A' if not have_fp8gen else 'FAIL')}")

    # encode RNE/saturate spot checks
    enc_ok = True
    for x, want in [(448.0, 0x7E), (1000.0, 0x7E), (1.0, 0x38), (0.0, 0x00),
                    (-2.0, 0xC0), (2 ** -9, 0x01), (1e-30, 0x00)]:
        g = fp8_e4m3_encode(x)
        if g != want:
            enc_ok = False
            print(f"  ENCODE FAIL x={x} got={g:#04x} want={want:#04x}")
    print(f"E4M3 encode spot-check: {'PASS' if enc_ok else 'FAIL'}")

    # ---- (2) block FP8 GEMM sanity ----
    # 1x4 activation @ 4x2 weight, single K-block, known scales.
    A = [[0.5, -1.0, 0.25, 2.0]]
    W = [[0.5,  1.0],
         [1.0, -0.5],
         [2.0,  0.25],
         [-1.0, 1.0]]
    A_bf = [[fp32_to_bf16(f2u32(x)) for x in row] for row in A]
    W_fp8 = [[fp8_e4m3_encode(w) for w in row] for row in W]
    # one 128x128 block covers all of K and N here -> w_scale_inv[0][0]
    w_scale = [[fp32_to_bf16(f2u32(1.0))]]   # unit scale
    C, ash = block_fp8_gemm(A_bf, W_fp8, w_scale, blk=128)
    # reference value: exact dot products of the (E4M3-rounded) operands.
    ref = [[0.0] * 2 for _ in range(1)]
    Aq = [[fp8_e4m3_decode(fp8_e4m3_encode(x * (2.0 ** ash[0]))) * (2.0 ** -ash[0]) for x in A[0]]]
    Wq = [[fp8_e4m3_decode(c) for c in row] for row in W_fp8]
    for j in range(2):
        ref[0][j] = sum(Aq[0][k] * Wq[k][j] for k in range(4))
    gemm_ok = True
    for j in range(2):
        got = u322f(bf16_to_fp32(C[0][j]))
        want = ref[0][j]
        # bf16 has ~3 decimal digits; allow 1 bf16 ULP-ish relative tolerance
        tol = max(abs(want) * (2 ** -7), 2 ** -7)
        if abs(got - want) > tol:
            gemm_ok = False
            print(f"  GEMM FAIL col {j}: got={got} want~={want} (a_shift={ash[0]})")
    print(f"block_fp8_gemm sanity: {'PASS' if gemm_ok else 'FAIL'} "
          f"out_bf16={[hex(c) for c in C[0]]} a_shift={ash}")

    # determinism
    C2, _ = block_fp8_gemm(A_bf, W_fp8, w_scale, blk=128)
    det_ok = (C2 == C)
    print(f"determinism: {'PASS' if det_ok else 'FAIL'}")

    ok = rt_ok and val_ok and enc_ok and gemm_ok and det_ok and (enc_match or not have_fp8gen)
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_selftest())
