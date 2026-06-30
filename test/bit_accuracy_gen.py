#!/usr/bin/env python3
# ============================================================================
# bit_accuracy_gen.py -- tile-suite generator for test/bit_accuracy_tb.v
# ----------------------------------------------------------------------------
# WHAT THIS IS
#   Builds a SUITE of representative glm_matmul_fp8 tiles and, for each, computes
#   TWO golden outputs with tools/glm_fp8_ref.py primitives:
#
#     C_bfp   : the EXACT-BFP result -- what our committed RTL (src/glm_matmul_fp8.v)
#               must produce BIT-FOR-BIT.  Uses ref.block_fp8_gemm (the
#               fixed-point/ACC_FRAC=18 within-block accumulator the RTL uses).
#
#     C_fp32  : the REAL-CONTRACT result -- what a realistic FP8 inference engine
#               (Hopper / DeepSeek-V3 / vLLM tensor-core fp32 accumulator) computes:
#               the SAME e4m3 quant + [128,128] block dequant, but the within-block
#               sum is accumulated in fp32 with a rounding at EVERY add (NOT exact).
#
#   The two differ ONLY in the within-block accumulator (exact BFP vs rounding
#   fp32) -- this is exactly the honest "does our accumulator choice matter?"
#   question.  The binding metric is the ARGMAX over the PE_N output columns of
#   each row (the next-token decision a real model makes): the expected argmax
#   handed to the TB is computed from C_fp32 (the real engine), and the TB asserts
#   the RTL (== C_bfp) picks the SAME column.
#
#   A near-exact float64 ground truth (C_ref) is also computed for the checker so
#   the report can show BFP is at least as accurate as the fp32-acc engine.
#
#   Tile geometry (must match the TB params): PE_M=2, PE_N=4, KMAX=256, BLK=128
#   -> up to 2 K-blocks, exercising the multi-block weight-scale fold + edges.
#
#   Files written into <out_dir> (one packed hex word per line):
#     a_col.hex   : NTILE*KMAX lines, 16*PE_M-bit bf16 activation col A[*][k]
#     w_row.hex   : NTILE*KMAX lines,  8*PE_N-bit E4M3 weight row    W[k][*]
#     ashift.hex  : NTILE lines,       8*PE_M-bit signed per-row pow2 act scale
#     wscale.hex  : NTILE lines,      16*PE_N*NB-bit bf16 block scales
#     klen.hex    : NTILE lines,      k_len (K beats) for the tile
#     expect.hex  : NTILE lines,      16*PE_M*PE_N-bit packed C_bfp (RTL must match)
#     argmax.hex  : NTILE lines,       8*PE_M-bit per-row expected argmax (from C_fp32)
#     ntile.txt   : the tile count
#   plus meta.json : full per-tile outputs (C_bfp / C_fp32 / C_ref) for the checker.
#
# Usage:  python3 test/bit_accuracy_gen.py <out_dir>
# ============================================================================
import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))
import glm_fp8_ref as ref

PE_M, PE_N, KMAX, BLK = 2, 4, 256, 128
NB = (KMAX + BLK - 1) // BLK
MAXTILE = 32          # must match bit_accuracy_tb.v memory sizing (pad to silence
                      # $readmemh range warnings; the TB only reads NTILE tiles)


# ----------------------------------------------------------------------------
# tiny deterministic LCG (no numpy) -- per-tile seeded for reproducibility
# ----------------------------------------------------------------------------
def lcg(seed):
    x = seed & 0xFFFFFFFF
    while True:
        x = (1103515245 * x + 12345) & 0x7FFFFFFF
        yield x


def bf16(x):
    return ref.fp32_to_bf16(ref.f2u32(x))


def fp8(x):
    c = ref.fp8_e4m3_encode(x)
    return (c ^ 0x01) if (c & 0x7F) == 0x7F else c   # never the NaN code


# ----------------------------------------------------------------------------
# Unified FP8 block GEMM, parameterized by the WITHIN-BLOCK ACCUMULATOR:
#   mode="bfp"  : exact fixed-point (ACC_FRAC=18) sum, one rounding to fp32 per
#                 block -- EXACTLY what our committed RTL does (mirrors
#                 ref.block_fp8_gemm; cross-checked against it in build_suite).
#   mode="fp32" : a ROUNDING fp32 accumulator (one fp32_add per term) -- the
#                 realistic Hopper / DeepSeek-V3 / vLLM tensor-core engine.
# Everything else (e4m3 quant, [128,128] block dequant fold, undo 2^-a_shift) is
# identical, so the two modes isolate ONLY the accumulator choice.
# Returns (C_bf16[M][N], C_fp32bits[M][N]) -- the bf16 output AND the pre-narrow
# fp32 dequant result, so the checker can see the accumulator gap before bf16
# erases it.
# ----------------------------------------------------------------------------
def block_fp8_gemm_mode(A_bf16, W_fp8, w_scale_inv, blk, a_shift, mode):
    M = len(A_bf16)
    K = len(W_fp8)
    N = len(W_fp8[0]) if K else 0
    n_kblk = (K + blk - 1) // blk
    C_bf = [[0] * N for _ in range(M)]
    C_f = [[0] * N for _ in range(M)]
    for i in range(M):
        ash = a_shift[i]
        a_q = [ref.fp32_to_fp8e4m3(ref.fp32_scale_pow2(ref.bf16_to_fp32(A_bf16[i][k]), ash))
               for k in range(K)]
        for j in range(N):
            out_blk = j // blk
            s = 0
            for bj in range(n_kblk):
                if mode == "bfp":
                    fx = 0                                       # exact fixed-point
                    for k in range(bj * blk, min((bj + 1) * blk, K)):
                        fx += ref.fp32_to_fixed(ref.fp8_mul(a_q[k], W_fp8[k][j]))
                    blkf = ref.fixed_to_fp32(fx)                 # ONE rounding/block
                else:  # "fp32"
                    blkf = 0
                    for k in range(bj * blk, min((bj + 1) * blk, K)):
                        blkf = ref.fp32_add(blkf, ref.fp8_mul(a_q[k], W_fp8[k][j]))
                wf = ref.bf16_to_fp32(w_scale_inv[out_blk][bj])
                s = ref.fp32_add(s, ref.fp32_mul(blkf, wf))
            deq = ref.fp32_scale_pow2(s, -ash)
            C_f[i][j] = deq
            C_bf[i][j] = ref.fp32_to_bf16(deq)
    return C_bf, C_f


# ----------------------------------------------------------------------------
# NEAR-EXACT float64 ground truth: accumulate the exact fp8*fp8 products and the
# exact (bf16-decoded) block scales in float64 (exact for these magnitudes); NO
# per-add rounding, NO bf16 narrowing -- the real-number value of the tile.
# Returns python floats (used only by the checker to contextualize error).
# ----------------------------------------------------------------------------
def block_fp8_gemm_exact(A_bf16, W_fp8, w_scale_inv, blk, a_shift):
    M = len(A_bf16)
    K = len(W_fp8)
    N = len(W_fp8[0]) if K else 0
    n_kblk = (K + blk - 1) // blk
    C = [[0.0] * N for _ in range(M)]
    for i in range(M):
        ash = a_shift[i]
        a_q = [ref.fp32_to_fp8e4m3(ref.fp32_scale_pow2(ref.bf16_to_fp32(A_bf16[i][k]), ash))
               for k in range(K)]
        for j in range(N):
            out_blk = j // blk
            s = 0.0
            for bj in range(n_kblk):
                blkacc = 0.0
                for k in range(bj * blk, min((bj + 1) * blk, K)):
                    blkacc += ref.u322f(ref.fp8_mul(a_q[k], W_fp8[k][j]))   # exact
                wf = ref.u322f(ref.bf16_to_fp32(w_scale_inv[out_blk][bj]))
                s += blkacc * wf
            C[i][j] = s * (2.0 ** (-ash))
    return C


# ----------------------------------------------------------------------------
# tile builders: each returns (A[PE_M][k_len] bf16, W[k_len][PE_N] e4m3,
#                              w_scale_inv[1][NB] bf16, k_len, label)
# ----------------------------------------------------------------------------
def _scales(seed, vals=None):
    if vals is None:
        vals = [0.0125, 0.05]            # distinct per-K-block scales
    return [[bf16(vals[bj]) for bj in range(NB)]]


def tile_random(seed, k_len=KMAX, lo=-4.0, hi=4.0, scales=None):
    g = lcg(seed)
    span = hi - lo
    A = [[bf16(lo + (next(g) % 100000) / 100000.0 * span) for _ in range(k_len)]
         for _ in range(PE_M)]
    W = [[fp8((-1.0 if (next(g) & 1) else 1.0) * ((next(g) % 900) / 100.0)) for _ in range(PE_N)]
         for _ in range(k_len)]
    return A, W, scales or _scales(seed), k_len, f"random(seed={seed:#x},K={k_len})"


def tile_saturation(seed):
    # activations and weights near the E4M3 max (448) -> exercise saturate + big sums
    g = lcg(seed)
    A = [[bf16((-1.0 if (next(g) & 1) else 1.0) * (300.0 + (next(g) % 200))) for _ in range(KMAX)]
         for _ in range(PE_M)]
    W = [[fp8((-1.0 if (next(g) & 1) else 1.0) * (200.0 + (next(g) % 250))) for _ in range(PE_N)]
         for _ in range(KMAX)]
    return A, W, _scales(seed, [0.002, 0.004]), KMAX, "saturation(near 448)"


def tile_tiny(seed):
    # subnormal / tiny range -> exercise the small-exponent FTZ + pow2 act scale
    g = lcg(seed)
    A = [[bf16((-1.0 if (next(g) & 1) else 1.0) * (1e-3 + (next(g) % 1000) * 1e-6)) for _ in range(KMAX)]
         for _ in range(PE_M)]
    W = [[fp8((-1.0 if (next(g) & 1) else 1.0) * (2.0 ** -(6 + (next(g) % 4)))) for _ in range(PE_N)]
         for _ in range(KMAX)]
    return A, W, _scales(seed, [0.25, 0.5]), KMAX, "tiny(subnormal-ish)"


def tile_zero_row(seed):
    # row 0 all zeros (a_shift=0 path, output 0), row 1 normal -> tie-break + mix
    A0, W, ws, _, _ = tile_random(seed)
    for k in range(KMAX):
        A0[0][k] = bf16(0.0)
    return A0, W, ws, KMAX, "zero_row(row0=0)"


def tile_single_block(seed):
    # k_len=128 -> exactly ONE K-block (the K<=BLK path; bank1 stays 0)
    return tile_random(seed, k_len=BLK, scales=_scales(seed, [0.03, 0.0]))


def tile_lopsided_blocks(seed):
    # two K-blocks with VERY different block scales -> one block dominates the fold
    return tile_random(seed, k_len=KMAX, scales=_scales(seed, [2.0 ** -7, 2.0 ** 3]))


def tile_clear_argmax(seed):
    # weights crafted so column `win` is the decisive logit by a wide margin
    g = lcg(seed)
    A = [[bf16(0.5 + (next(g) % 100) / 200.0) for _ in range(KMAX)] for _ in range(PE_M)]
    W = [[0] * PE_N for _ in range(KMAX)]
    win = [2, 1]                      # winning column per row (rows share W, so encode via magnitude)
    for k in range(KMAX):
        for pj in range(PE_N):
            base = 1.0 + 0.5 * pj      # column 3 strongest, monotone -> clear argmax=3
            W[k][pj] = fp8(base)
    return A, W, _scales(seed, [0.01, 0.01]), KMAX, "clear_argmax"


def build_suite():
    specs = [
        tile_random(0xBADC0DE),                 # baseline generic
        tile_random(0x1234567),
        tile_random(0x0FEDCBA, lo=-1.0, hi=1.0),
        tile_random(0x55AA55A, lo=-16.0, hi=16.0),
        tile_saturation(0xA11CE),
        tile_tiny(0xB0B),
        tile_zero_row(0xC0FFEE),
        tile_single_block(0xD15EA5E),
        tile_single_block(0x5EED01),
        tile_lopsided_blocks(0xF00D),
        tile_lopsided_blocks(0x9ABCDEF),
        tile_clear_argmax(0x2468),
        tile_random(0x13579BD, lo=-8.0, hi=8.0),
        tile_random(0xACE0F),
    ]
    suite = []
    for (A, W, wsi, k_len, label) in specs:
        a_shift = [ref.derive_a_shift(A[i]) for i in range(PE_M)]
        # exact-BFP (RTL) and realistic fp32-acc engine, both bf16 + pre-narrow fp32
        C_bfp, F_bfp = block_fp8_gemm_mode(A, W, wsi, BLK, a_shift, "bfp")
        C_fp32, F_fp32 = block_fp8_gemm_mode(A, W, wsi, BLK, a_shift, "fp32")
        C_ref = block_fp8_gemm_exact(A, W, wsi, BLK, a_shift)
        # CROSS-CHECK: our bfp-mode reimplementation must equal the validated
        # committed golden tools/glm_fp8_ref.block_fp8_gemm, bit-for-bit.
        C_gold, _ = ref.block_fp8_gemm(A, W, wsi, blk=BLK, a_shift=a_shift)
        if C_gold != C_bfp:
            raise AssertionError(f"bfp-mode != golden block_fp8_gemm for {label}")
        suite.append(dict(A=A, W=W, wsi=wsi, k_len=k_len, label=label,
                          a_shift=a_shift, C_bfp=C_bfp, C_fp32=C_fp32, C_ref=C_ref,
                          F_bfp=F_bfp, F_fp32=F_fp32))
    return suite


# ----------------------------------------------------------------------------
# argmax helpers (decode bf16 codes to real value; first-occurrence tie-break --
# MUST match the TB's keybf() compare exactly).
# ----------------------------------------------------------------------------
def argmax_row_bf16(codes):
    best, bi = None, 0
    for j, c in enumerate(codes):
        v = ref.u322f(ref.bf16_to_fp32(c))
        if best is None or v > best:
            best, bi = v, j
    return bi


def gen(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    suite = build_suite()
    NTILE = len(suite)

    a_lines, w_lines = [], []
    ash_lines, ws_lines, kl_lines, exp_lines, am_lines = [], [], [], [], []

    for t in suite:
        A, W, wsi, k_len = t["A"], t["W"], t["wsi"], t["k_len"]
        a_shift, C_bfp, C_fp32 = t["a_shift"], t["C_bfp"], t["C_fp32"]

        # a_col / w_row: KMAX lines each (pad beats >= k_len with 0; TB streams k_len)
        for k in range(KMAX):
            aw = 0
            for pi in range(PE_M):
                code = (A[pi][k] & 0xFFFF) if k < k_len else 0
                aw |= code << (16 * pi)
            a_lines.append(f"{aw:0{(16*PE_M)//4}x}")
            ww = 0
            for pj in range(PE_N):
                code = (W[k][pj] & 0xFF) if k < k_len else 0
                ww |= code << (8 * pj)
            w_lines.append(f"{ww:0{(8*PE_N)//4}x}")

        ash = 0
        for pi in range(PE_M):
            ash |= (a_shift[pi] & 0xFF) << (8 * pi)
        ash_lines.append(f"{ash:0{(8*PE_M)//4}x}")

        ws = 0
        for bj in range(NB):
            for pj in range(PE_N):
                ws |= (wsi[0][bj] & 0xFFFF) << (16 * (bj * PE_N + pj))
        ws_lines.append(f"{ws:0{(16*PE_N*NB)//4}x}")

        kl_lines.append(f"{k_len:03x}")

        ce = 0
        for pi in range(PE_M):
            for pj in range(PE_N):
                ce |= (C_bfp[pi][pj] & 0xFFFF) << (16 * (pi * PE_N + pj))
        exp_lines.append(f"{ce:0{(16*PE_M*PE_N)//4}x}")

        # expected argmax (the REAL-ENGINE decision) = argmax of C_fp32 per row
        am = 0
        for pi in range(PE_M):
            am |= (argmax_row_bf16(C_fp32[pi]) & 0xFF) << (8 * pi)
        am_lines.append(f"{am:0{(8*PE_M)//4}x}")

    def wfile(fn, lines, pad_to=None, pad="0"):
        # pad unused tile slots with zero words so the TB's MAXTILE-sized memories
        # load without $readmemh "not enough words" range warnings.
        if pad_to is not None and len(lines) < pad_to:
            lines = lines + [pad] * (pad_to - len(lines))
        with open(os.path.join(out_dir, fn), "w") as f:
            f.write("\n".join(lines) + "\n")

    wfile("a_col.hex", a_lines, MAXTILE * KMAX)
    wfile("w_row.hex", w_lines, MAXTILE * KMAX)
    wfile("ashift.hex", ash_lines, MAXTILE)
    wfile("wscale.hex", ws_lines, MAXTILE)
    wfile("klen.hex", kl_lines, MAXTILE)
    wfile("expect.hex", exp_lines, MAXTILE)
    wfile("argmax.hex", am_lines, MAXTILE)
    wfile("ntile.txt", [str(NTILE)])

    # full per-tile outputs for the python checker (decoded floats for error stats)
    meta = []
    for t in suite:
        meta.append(dict(
            label=t["label"], k_len=t["k_len"], a_shift=t["a_shift"],
            C_bfp=t["C_bfp"], C_fp32=t["C_fp32"], C_ref=t["C_ref"],
            F_bfp=t["F_bfp"], F_fp32=t["F_fp32"]))
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(dict(PE_M=PE_M, PE_N=PE_N, KMAX=KMAX, BLK=BLK, NB=NB, NTILE=NTILE,
                       tiles=meta), f)

    print(f"bit_accuracy_gen: wrote {NTILE} tiles to {out_dir} "
          f"(PE_M={PE_M} PE_N={PE_N} KMAX={KMAX} BLK={BLK} NB={NB})")
    for i, t in enumerate(suite):
        print(f"  tile {i:2d}: {t['label']:<28s} a_shift={t['a_shift']}")
    return NTILE


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/bitacc"
    gen(out)
