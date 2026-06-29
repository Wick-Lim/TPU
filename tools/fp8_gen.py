#!/usr/bin/env python3
# ============================================================================
# fp8_gen.py  --  OFFLINE VECTOR BUILDER for weight_decomp_tb.v
# ----------------------------------------------------------------------------
#   gen <vec_file>
#     Build the round-trip test vector for the FP8 weight decompressor:
#       1. sample fp32 WEIGHTS from trained-weight-like distributions
#          (near-zero-heavy Gaussian / Laplacian) + edge cases,
#       2. quantize each fp32 to E4M3 with a BIT-EXACT MIRROR of the repo's
#          fp8_e4m3.vh `fp32_to_fp8e4m3` (RNE + saturation, subnormals),
#       3. canonical-Huffman compress the resulting FP8 BYTE block with the
#          committed encoder (tools/fp8_huff.compress),
#       4. emit per case: the canonical count table + symbol order, the
#          compressed bytes, AND the raw fp32 bit patterns.
#
#   The TB re-derives the golden FP8 byte stream by running the ACTUAL RTL
#   `fp32_to_fp8e4m3` on those fp32 patterns and ASSERTS the decoded DUT output
#   equals it -- which simultaneously proves (a) lossless round-trip and (b)
#   that the python E4M3 mirror used to build the compressed image is identical
#   to the repo's hardware encode (any divergence -> $fatal in the TB).
#
# Imports the committed encoder unchanged (no modification to fp8_huff.py).
# ============================================================================
import sys, os, struct, random, math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fp8_huff import compress, MAXLEN          # reuse the encoder, do not edit it


# ---------------------------------------------------------------------------
# Bit-exact mirror of fp8_e4m3.vh : fp32_to_fp8e4m3  (RNE + saturate to 448)
# ---------------------------------------------------------------------------
def fp32_to_fp8e4m3(bits):
    s  = (bits >> 31) & 1
    e  = (bits >> 23) & 0xFF
    m  = bits & 0x7FFFFF
    if e == 0xFF and m != 0:
        return (s << 7) | 0x7F                 # NaN  (S.1111.111)
    if e == 0xFF:
        return (s << 7) | 0x7E                 # Inf -> saturate 448
    if e == 0x00:
        return (s << 7)                        # zero / FTZ fp32 subnormal
    E   = e - 127
    sig = (1 << 23) | m                        # 24-bit significand 1.m
    if E >= -6:
        # ---------------- NORMAL E4M3 ----------------
        efield = E + 7                          # >= 1
        g   = (m >> 19) & 1
        st  = 1 if (m & 0x7FFFF) else 0         # m[18:0]
        rup = g & (st | ((m >> 20) & 1))        # m[20]
        msum = ((m >> 20) & 0x7) + rup          # 4-bit add
        if msum & 0x8:                          # mantissa carried out
            efield += 1
            mant3 = 0
        else:
            mant3 = msum & 0x7
        if efield > 15 or (efield == 15 and mant3 == 7):
            return (s << 7) | 0x7E              # overflow / NaN-slot -> 448
        return (s << 7) | ((efield & 0xF) << 3) | mant3
    if E <= -11:
        return (s << 7)                        # below half-ulp -> zero
    # ---------------- SUBNORMAL E4M3 (E in [-10..-7]) ----------------
    if E == -7:
        q_int = (sig >> 21) & 0x7;  g = (sig >> 20) & 1;  st = 1 if (sig & 0xFFFFF)  else 0
    elif E == -8:
        q_int = (sig >> 22) & 0x3;  g = (sig >> 21) & 1;  st = 1 if (sig & 0x1FFFFF) else 0
    elif E == -9:
        q_int = (sig >> 23) & 0x1;  g = (sig >> 22) & 1;  st = 1 if (sig & 0x3FFFFF) else 0
    else:  # E == -10
        q_int = 0;                  g = (sig >> 23) & 1;  st = 1 if (sig & 0x7FFFFF) else 0
    rup = g & (st | (q_int & 1))                # RNE
    q   = q_int + rup                           # 0..8
    if q == 0:
        return (s << 7)                        # -> signed zero
    if q == 8:
        return (s << 7) | 0x08                 # rounded up to smallest normal
    return (s << 7) | (q & 0x7)                # subnormal


def f2bits(x):
    return struct.unpack("<I", struct.pack("<f", x))[0]


def laplacian(rng, scale):
    u = rng.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1.0 - 2.0 * abs(u))


def gen_cases():
    """Return list of (rep_flag, [fp32 bit patterns]).  rep_flag=1 marks the
    representative trained-weight blocks that count toward the headline ratio."""
    rng = random.Random(0xC0FFEE)
    cases = []
    # --- representative trained-weight-like blocks (near-zero-heavy) ---
    cases.append((1, [f2bits(rng.gauss(0.0, 0.02)) for _ in range(2048)]))  # tight Gaussian
    cases.append((1, [f2bits(rng.gauss(0.0, 0.05)) for _ in range(1024)]))  # typical weight std
    cases.append((1, [f2bits(laplacian(rng, 0.10)) for _ in range(1536)]))  # Laplacian tail
    cases.append((1, [f2bits(rng.gauss(0.0, 0.25)) for _ in range(1024)]))  # broader spread
    # --- edge / worst cases ---
    cases.append((0, [0] * 256))                                            # all-zero
    cases.append((0, [f2bits(1.0)] * 512))                                  # all-same (-> 0x38)
    wc = []                                                                  # max-entropy block
    for _ in range(2048):
        s = rng.getrandbits(1)
        e = rng.randint(121, 134)        # fp32 biased exp -> E4M3 normal e in 1..14
        man = rng.getrandbits(23)
        wc.append((s << 31) | (e << 23) | man)
    cases.append((0, wc))
    cases.append((0, [f2bits(0.0123)]))                                     # single sample
    return cases


def do_gen(path):
    cases = gen_cases()
    rep_o = rep_c = tot_o = tot_c = 0
    with open(path, "w") as f:
        f.write(f"{len(cases)}\n")
        for k, (rep, vals) in enumerate(cases):
            fp8 = bytes(fp32_to_fp8e4m3(v) for v in vals)
            comp, order, count = compress(fp8)
            counts = [count[l] if l < len(count) else 0 for l in range(1, MAXLEN + 1)]
            f.write(f"{rep} {len(vals)} {len(order)} {len(comp)}\n")
            f.write(" ".join(str(c) for c in counts) + "\n")
            f.write(" ".join(str(o) for o in order) + "\n")
            f.write(" ".join(str(b) for b in comp) + "\n")
            f.write(" ".join(str(v) for v in vals) + "\n")
            r = len(fp8) / len(comp) if comp else 0.0
            tot_o += len(fp8); tot_c += len(comp)
            if rep:
                rep_o += len(fp8); rep_c += len(comp)
            sys.stderr.write(
                f"case {k} ({'rep' if rep else 'edge'}): {len(fp8):5d} -> {len(comp):5d} "
                f"bytes  ratio {r:4.2f}x  ({8*len(comp)/max(1,len(fp8)):.2f} b/sym)\n")
    sys.stderr.write(
        f"REPRESENTATIVE: {rep_o} -> {rep_c}  ratio {rep_o/rep_c:.2f}x   "
        f"ALL: {tot_o} -> {tot_c}  ratio {tot_o/tot_c:.2f}x\n")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gen":
        do_gen(sys.argv[2])
    else:
        sys.stderr.write("usage: fp8_gen.py gen <vec_file>\n")
        sys.exit(1)
