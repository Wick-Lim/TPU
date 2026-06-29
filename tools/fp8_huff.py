#!/usr/bin/env python3
# ============================================================================
# fp8_huff.py  --  OFFLINE ENCODER for weight_decomp.v (GLM-5.2 FP8 P2.1)
# ----------------------------------------------------------------------------
# The matching compressor for the on-chip canonical-Huffman decompressor
# `weight_decomp`.  It is the OFFLINE half that produces the compressed Flash
# image; the RTL TB calls it to compress a block then streams the bytes back
# through the DUT and checks bit-exact round-trip + measures the ratio.
#
# Pipeline (all on the 257-symbol alphabet 0..255 = FP8 bytes, 256 = EOB):
#   1. histogram the FP8 byte block (+1 EOB)
#   2. LENGTH-LIMITED Huffman code lengths via PACKAGE-MERGE (max length MAXLEN)
#   3. CANONICAL code assignment (codes of a length = a contiguous range,
#      symbols in (length, symbol) order) -- the exact layout weight_decomp
#      decodes with count_table[len] + symbol_table[i].
#   4. pack the code bits MSB-first into bytes (bit 7 of byte 0 = first bit),
#      append the EOB code, zero-pad the last byte.
#
# It emits a numeric vector file consumed by test/weight_decomp_tb.v.
# Pure byte manipulation -- the FP8 values are never numerically interpreted.
# ============================================================================
import sys, random
from collections import Counter

MAXLEN  = 15
EOB_SYM = 256


def package_merge(weights, L):
    """Length-limited prefix-code lengths (Larmore-Hirschberg package-merge).
    weights: list of positive ints (one per symbol). Returns list of lengths."""
    n = len(weights)
    if n == 1:
        return [1]
    base = sorted(range(n), key=lambda i: (weights[i], i))   # original coins
    orig = [(weights[i], [i]) for i in base]
    packages = []
    for _ in range(L):
        merged = []
        for k in range(0, len(packages) - 1, 2):
            w = packages[k][0] + packages[k + 1][0]
            mem = packages[k][1] + packages[k + 1][1]
            merged.append((w, mem))
        packages = sorted(merged + orig, key=lambda x: x[0])
    chosen = packages[: 2 * n - 2]
    cnt = Counter()
    for _, mem in chosen:
        for i in mem:
            cnt[i] += 1
    return [cnt[i] for i in range(n)]


def canonical(lengths, syms):
    """Given per-symbol code lengths, assign CANONICAL codes.
    Returns (codes{sym:(val,len)}, order[list of syms], count[len])."""
    maxl = max(lengths.values())
    count = [0] * (maxl + 1)
    for s in syms:
        count[lengths[s]] += 1
    # canonical first-code per length
    firstcode = [0] * (maxl + 2)
    code = 0
    for l in range(1, maxl + 1):
        firstcode[l] = code
        code = (code + count[l]) << 1
    order = sorted(syms, key=lambda s: (lengths[s], s))
    nextc = list(firstcode)
    codes = {}
    for s in order:
        l = lengths[s]
        codes[s] = (nextc[l], l)
        nextc[l] += 1
    return codes, order, count


def build_code(byte_block):
    """Histogram -> length-limited Huffman -> canonical code for a block."""
    hist = Counter(byte_block)
    hist[EOB_SYM] += 1                      # one EOB terminator
    syms = sorted(hist.keys())
    weights = [hist[s] for s in syms]
    if len(syms) == 1:                       # degenerate: single symbol
        lengths = {syms[0]: 1}
    else:
        Ls = package_merge(weights, MAXLEN)
        lengths = {syms[i]: Ls[i] for i in range(len(syms))}
    codes, order, count = canonical(lengths, syms)
    return codes, order, count


def compress(byte_block):
    """Return (comp_bytes, order, count) for an FP8 byte block."""
    codes, order, count = build_code(byte_block)
    bits = []
    for b in list(byte_block) + [EOB_SYM]:
        val, ln = codes[b]
        for i in range(ln - 1, -1, -1):      # MSB-first
            bits.append((val >> i) & 1)
    while len(bits) % 8 != 0:                # zero-pad final byte
        bits.append(0)
    comp = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | bits[i + j]
        comp.append(byte)
    return bytes(comp), order, count


# ---- block generators (near-zero-heavy, FP8-byte-like distributions) -------
def laplacian_fp8_block(nbytes, scale, seed):
    """A near-zero-heavy FP8-E4M3 *byte* block.  We do NOT do FP arithmetic:
    we just bias the SIGN-MAGNITUDE byte distribution toward small-exponent
    codes (near zero), which is what trained FP8 weights look like."""
    rng = random.Random(seed)
    out = bytearray()
    for _ in range(nbytes):
        # geometric-ish magnitude: small exponents dominate
        mag = int(abs(rng.gauss(0, scale)))
        if mag > 127:
            mag = 127
        sign = rng.getrandbits(1) << 7
        out.append(sign | (mag & 0x7F))
    return bytes(out)


def emit_vector(cases, fh):
    fh.write(f"{len(cases)}\n")
    for orig in cases:
        comp, order, count = compress(orig)
        counts = [count[l] if l < len(count) else 0 for l in range(1, MAXLEN + 1)]
        fh.write(f"{MAXLEN} {len(order)} {len(orig)} {len(comp)}\n")
        fh.write(" ".join(str(c) for c in counts) + "\n")
        fh.write(" ".join(str(s) for s in order) + "\n")
        fh.write(" ".join(str(b) for b in comp) + "\n")
        fh.write(" ".join(str(b) for b in orig) + "\n")


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else "vec.txt"
    cases = [
        laplacian_fp8_block(1024, 6,  1),    # typical trained-weight skew
        laplacian_fp8_block(2048, 3,  2),    # very near-zero-heavy (high ratio)
        laplacian_fp8_block(777,  16, 3),    # broader spread (lower ratio)
        bytes([0x00] * 256),                  # all-zero (max ratio, run-heavy)
        bytes(range(256)) * 2,                # uniform 0..255 (worst case ~8b)
        bytes([0x3C]),                        # single byte block (edge)
    ]
    tot_o = tot_c = 0
    with open(out_path, "w") as fh:
        emit_vector(cases, fh)
    # report ratios to stderr
    for k, orig in enumerate(cases):
        comp, _, _ = compress(orig)
        tot_o += len(orig)
        tot_c += len(comp)
        r = len(orig) / len(comp) if comp else 0
        sys.stderr.write(
            f"case {k}: {len(orig):5d} -> {len(comp):5d} bytes  "
            f"ratio {r:4.2f}x  ({8*len(comp)/max(1,len(orig)):.2f} bits/sym)\n")
    sys.stderr.write(
        f"TOTAL: {tot_o} -> {tot_c} bytes  ratio {tot_o/tot_c:4.2f}x\n")
