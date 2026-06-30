#!/usr/bin/env python3
# ============================================================================
# fp8_ctxpack.py  --  OFFLINE CONTEXT-MODELED ENCODER for weight_decomp2.v
#                                  (IMPROVEMENT_PLAN.md ULTRA_PERF #3, beats P2.1)
# ----------------------------------------------------------------------------
# The committed canonical-Huffman coder (weight_decomp / fp8_huff.py) is ORDER-0:
# one static table for the whole stream -> 1.34x (5.97 b/sym) on representative
# FP8 weights.  The residual redundancy is in the CONTEXT: trained FP8-E4M3
# weights, scanned in memory order inside a [128,128] block, have strong
# MAGNITUDE LOCALITY (neighbouring weights share an exponent range) and
# sign/exponent correlation.  An ORDER-1 model -- pick one of a few Huffman
# tables by a cheap function of the PREVIOUS decoded byte -- captures that and
# reaches ~1.5-1.7x, with on-chip cost = a few small extra tables.
#
# THIS is the matching OFFLINE half of the on-chip decompressor weight_decomp2:
#   * identical CONTEXT FUNCTION  ctxfn(prev_byte)  (HW: 2 comparators)
#   * NCTX canonical-Huffman tables, one per context class
#   * encode each byte with the table selected by the context, then update the
#     context from that byte; append EOB under the final context (self-delimiting)
#
# It reuses the COMMITTED length-limited package-merge + canonical-code layout
# (tools/fp8_huff.py) per context, and the COMMITTED bit-exact E4M3 quantizer
# mirror (tools/fp8_gen.py) for the representative-weight generator -- neither
# file is modified.  Pure byte manipulation; FP8 values are never numerically
# interpreted by the coder.
#
# CLI:  fp8_ctxpack.py gen <vec_file>      build the weight_decomp2_tb vector
# ============================================================================
import sys, os, struct, random, math
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_huff import package_merge, canonical, MAXLEN, EOB_SYM   # reuse, do not edit
from fp8_huff import compress as compress_order0                 # order-0 baseline
from fp8_gen import fp32_to_fp8e4m3, f2bits                      # reuse E4M3 mirror

# ----------------------------------------------------------------------------
# CONTEXT MODEL  --  NCTX classes selected by the previous decoded byte.
#   magnitude m = prev_byte[6:0]  (drop the sign bit -- sign is ~uncorrelated;
#   the EXPONENT/magnitude is what carries the locality).  Two threshold
#   compares + a zero test => 4 classes.  This is exactly what weight_decomp2
#   computes in HW (ctxfn): cheap, no multiply.
# ----------------------------------------------------------------------------
NCTX     = 4
THRESH1  = 0x18      # m < 24  -> exponent field 0..2  (tiny)
THRESH2  = 0x40      # m < 64  -> exponent field 3..7  (small/mid)
INIT_CTX = 0         # context for the very first symbol


def ctxfn(prev_byte):
    m = prev_byte & 0x7F
    if   m == 0:        return 0       # exact zero
    elif m < THRESH1:   return 1       # tiny magnitude
    elif m < THRESH2:   return 2       # small/mid magnitude
    else:               return 3       # large magnitude


def build_code_hist(hist):
    """Canonical length-limited Huffman from a {symbol:count} histogram.
    Returns (codes{sym:(val,len)}, order[syms], count[len])."""
    syms = sorted(hist.keys())
    weights = [hist[s] for s in syms]
    if len(syms) == 1:
        lengths = {syms[0]: 1}
    else:
        Ls = package_merge(weights, MAXLEN)
        lengths = {syms[i]: Ls[i] for i in range(len(syms))}
    return canonical(lengths, syms)


def compress(byte_block):
    """Context-modeled compress.  Returns (comp_bytes, tables) where
    tables[c] = (order[c], count[c]) for each context class (None if unused)."""
    block = list(byte_block)
    # 1) per-context histograms over the symbols emitted IN that context
    hists = [Counter() for _ in range(NCTX)]
    ctx = INIT_CTX
    for b in block:
        hists[ctx][b] += 1
        ctx = ctxfn(b)
    term_ctx = ctx
    hists[term_ctx][EOB_SYM] += 1            # EOB terminates under final context

    # 2) build one canonical table per (used) context
    codes_per = [None] * NCTX
    tables    = [None] * NCTX
    for c in range(NCTX):
        if len(hists[c]) == 0:
            continue                         # never consulted by the decoder
        codes, order, count = build_code_hist(hists[c])
        codes_per[c] = codes
        tables[c]    = (order, count)

    # 3) emit code bits MSB-first, context-selected, EOB under final context
    bits = []
    ctx = INIT_CTX
    for b in block:
        val, ln = codes_per[ctx][b]
        for i in range(ln - 1, -1, -1):
            bits.append((val >> i) & 1)
        ctx = ctxfn(b)
    val, ln = codes_per[term_ctx][EOB_SYM]
    for i in range(ln - 1, -1, -1):
        bits.append((val >> i) & 1)
    while len(bits) % 8 != 0:
        bits.append(0)
    comp = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        comp.append(byte)
    return bytes(comp), tables


# ---------------------------------------------------------------------------
# REPRESENTATIVE TRAINED-WEIGHT GENERATOR (with magnitude LOCALITY)
#   Real trained weights vary smoothly in memory order: we model the per-weight
#   log-magnitude as an AR(1) random walk (so neighbouring exponents correlate),
#   with a ~random sign, then quantize fp32 -> E4M3 with the committed bit-exact
#   mirror.  The resulting FP8 byte stream has the order-1 structure a context
#   model exploits.  Pure i.i.d. samples (rho=0) reduce gracefully to order-0.
# ---------------------------------------------------------------------------
def correlated_block(n, scale, rho, sigma, seed):
    rng = random.Random(seed)
    mean_lm = math.log(scale)
    lm = mean_lm
    out = []
    for _ in range(n):
        lm = mean_lm + rho * (lm - mean_lm) + sigma * rng.gauss(0.0, 1.0)
        mag = math.exp(lm)
        sgn = -1.0 if rng.random() < 0.5 else 1.0
        out.append(f2bits(sgn * mag))
    return out


def iid_block(n, scale, seed):
    rng = random.Random(seed)
    return [f2bits(rng.gauss(0.0, scale)) for _ in range(n)]


def gen_cases():
    """list of (rep_flag, [fp32 bit patterns]).  rep_flag=1 = trained-weight
    blocks that count toward the headline ratio."""
    cases = []
    # --- representative trained-weight blocks: smooth magnitude locality ---
    cases.append((1, correlated_block(2048, 0.020, 0.985, 0.28, 11)))  # tight, very smooth
    cases.append((1, correlated_block(1536, 0.050, 0.975, 0.35, 12)))  # typical std
    cases.append((1, correlated_block(1024, 0.100, 0.970, 0.45, 13)))  # broader, still local
    cases.append((1, correlated_block(2048, 0.035, 0.980, 0.30, 14)))  # another tensor row
    # --- edge / worst cases ---
    cases.append((0, [0] * 256))                                      # all-zero
    cases.append((0, [f2bits(1.0)] * 512))                            # all-same
    cases.append((0, iid_block(1024, 0.05, 21)))                      # i.i.d (rho=0 limit)
    wc = []                                                           # max-entropy block
    rng = random.Random(0xBADBEEF)
    for _ in range(2048):
        s = rng.getrandbits(1); e = rng.randint(121, 134); man = rng.getrandbits(23)
        wc.append((s << 31) | (e << 23) | man)
    cases.append((0, wc))
    cases.append((0, [f2bits(0.0123)]))                               # single sample
    return cases


def emit_counts_line(count):
    return " ".join(str(count[l] if l < len(count) else 0) for l in range(1, MAXLEN + 1))


def do_gen(path):
    cases = gen_cases()
    rep_o = rep_c = rep_c0 = tot_o = tot_c = 0
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for k, (rep, vals) in enumerate(cases):
            fp8 = bytes(fp32_to_fp8e4m3(v) for v in vals)
            comp, tables = compress(fp8)
            comp0, _, _ = compress_order0(fp8)            # order-0 baseline, same block
            f.write(f"{rep} {len(vals)} {len(comp)} {NCTX}\n")
            for c in range(NCTX):
                if tables[c] is None:
                    f.write("0\n")
                    f.write(" ".join("0" for _ in range(MAXLEN)) + "\n")
                    f.write("\n")
                else:
                    order, count = tables[c]
                    f.write(f"{len(order)}\n")
                    f.write(emit_counts_line(count) + "\n")
                    f.write(" ".join(str(s) for s in order) + "\n")
            f.write(" ".join(str(b) for b in comp) + "\n")
            f.write(" ".join(str(v) for v in vals) + "\n")
            r  = len(fp8) / len(comp)  if comp  else 0.0
            r0 = len(fp8) / len(comp0) if comp0 else 0.0
            tot_o += len(fp8); tot_c += len(comp)
            if rep:
                rep_o += len(fp8); rep_c += len(comp); rep_c0 += len(comp0)
            sys.stderr.write(
                f"case {k} ({'rep' if rep else 'edge'}): {len(fp8):5d} -> ctx {len(comp):5d} "
                f"({r:4.2f}x) | order0 {len(comp0):5d} ({r0:4.2f}x)\n")
    sys.stderr.write(
        f"REPRESENTATIVE: {rep_o} -> ctx {rep_c} = {rep_o/rep_c:.3f}x   "
        f"order0 {rep_c0} = {rep_o/rep_c0:.3f}x   "
        f"gain {rep_c0/rep_c:.3f}x\n")
    sys.stderr.write(f"ALL: {tot_o} -> {tot_c}  ratio {tot_o/tot_c:.3f}x\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gen":
        do_gen(sys.argv[2])
    else:
        sys.stderr.write("usage: fp8_ctxpack.py gen <vec_file>\n")
        sys.exit(1)
