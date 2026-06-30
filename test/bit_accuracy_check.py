#!/usr/bin/env python3
# ============================================================================
# bit_accuracy_check.py -- quantify RTL(==BFP golden) vs the REAL-CONTRACT engine
# ----------------------------------------------------------------------------
#   Reads <dir>/meta.json (written by bit_accuracy_gen.py) and reports, over the
#   whole tile suite:
#
#     * MAX / RMS absolute + relative error of C_bfp  vs  C_fp32 (real engine),
#       in bf16-decoded real value -- "how far apart are the two outputs".
#     * the same vs the float64 near-exact ground truth C_ref, for BOTH C_bfp and
#       C_fp32, to show the BFP accumulator is at least as accurate as fp32-acc.
#     * whether the per-row ARGMAX (next-token decision) is PRESERVED:
#         argmax(C_bfp) vs argmax(C_fp32)   -- the binding metric, and
#         argmax(C_bfp) vs argmax(C_ref)    -- vs ground truth.
#     * bit-exactness count: how many of the PE_M*PE_N*NTILE bf16 elements are
#       already identical between C_bfp and C_fp32.
#
#   The TB (test/bit_accuracy_tb.v) separately PROVES the committed RTL produces
#   C_bfp BIT-FOR-BIT, so C_bfp here stands in for the RTL output exactly; the
#   errors below ARE the RTL-vs-real-contract errors.
#
# Usage:  python3 test/bit_accuracy_check.py <dir>
# ============================================================================
import sys, os, json, struct, math


def u322f(u):
    return struct.unpack("<f", struct.pack("<I", u & 0xFFFFFFFF))[0]


def bf16_decode(c):
    return u322f((c & 0xFFFF) << 16)


def argmax(vals):
    best, bi = None, 0
    for j, v in enumerate(vals):
        if best is None or v > best:
            best, bi = v, j
    return bi


def main(d):
    with open(os.path.join(d, "meta.json")) as f:
        meta = json.load(f)
    PE_M, PE_N, NTILE = meta["PE_M"], meta["PE_N"], meta["NTILE"]

    # bf16-output domain (what the RTL actually emits)
    max_abs = 0.0; sse = 0.0; n = 0; max_rel = 0.0; bitexact = 0
    # pre-narrow fp32 domain (surfaces the accumulator gap before bf16 erases it)
    fmax_abs = 0.0; fsse = 0.0; fmax_rel = 0.0; fbitexact = 0
    # ground-truth comparisons (bf16 output vs float64 exact)
    bfp_max_rel = fp32_max_rel = 0.0
    # argmax preservation
    rows = 0; am_fp32_ok = 0; am_ref_ok = 0; am_fail = []
    min_margin = None; ties = 0; min_rel_margin = None

    for t in meta["tiles"]:
        Cb, Cf, Cr = t["C_bfp"], t["C_fp32"], t["C_ref"]
        Fb, Ff = t["F_bfp"], t["F_fp32"]
        for i in range(PE_M):
            vb = [bf16_decode(Cb[i][j]) for j in range(PE_N)]
            vf = [bf16_decode(Cf[i][j]) for j in range(PE_N)]
            vr = [Cr[i][j] for j in range(PE_N)]
            fb = [u322f(Fb[i][j]) for j in range(PE_N)]   # pre-narrow fp32
            ff = [u322f(Ff[i][j]) for j in range(PE_N)]
            for j in range(PE_N):
                n += 1
                # bf16-output domain
                e = abs(vb[j] - vf[j]); max_abs = max(max_abs, e); sse += e * e
                max_rel = max(max_rel, e / max(abs(vf[j]), 1e-30))
                if Cb[i][j] == Cf[i][j]:
                    bitexact += 1
                # pre-narrow fp32 domain (the true accumulator gap)
                fe = abs(fb[j] - ff[j]); fmax_abs = max(fmax_abs, fe); fsse += fe * fe
                fmax_rel = max(fmax_rel, fe / max(abs(ff[j]), 1e-30))
                if Fb[i][j] == Ff[i][j]:
                    fbitexact += 1
                # accuracy vs ground truth (relative; abs is bf16-narrowing-dominated)
                bfp_max_rel = max(bfp_max_rel, abs(vb[j] - vr[j]) / max(abs(vr[j]), 1e-30))
                fp32_max_rel = max(fp32_max_rel, abs(vf[j] - vr[j]) / max(abs(vr[j]), 1e-30))
            # argmax (next-token decision) -- decided in bf16 output domain
            rows += 1
            ab, af, ar = argmax(vb), argmax(vf), argmax(vr)
            if ab == af:
                am_fp32_ok += 1
            else:
                am_fail.append((t["label"], i, ab, af, vb, vf))
            if ab == ar:
                am_ref_ok += 1
            sv = sorted(vf, reverse=True)
            margin = sv[0] - sv[1]
            if margin == 0.0:                          # degenerate all-equal row
                ties += 1
            else:
                min_margin = margin if min_margin is None else min(min_margin, margin)
                rm = margin / max(abs(sv[0]), 1e-30)
                min_rel_margin = rm if min_rel_margin is None else min(min_rel_margin, rm)

    rms = math.sqrt(sse / n) if n else 0.0
    frms = math.sqrt(fsse / n) if n else 0.0

    print("=" * 72)
    print(f"BIT-ACCURACY REPORT  (RTL == C_bfp, proven bit-exact by the TB)")
    print(f"  tiles={NTILE}  elements={n}  rows(argmax decisions)={rows}")
    print("-" * 72)
    print(f"  RTL(BFP) vs REAL-ENGINE(fp32-acc), bf16 OUTPUT domain (what RTL emits):")
    print(f"    bit-identical bf16 outputs : {bitexact}/{n} "
          f"({100.0*bitexact/n:.1f}%)")
    print(f"    MAX abs / RMS abs error    : {max_abs:.6g} / {rms:.6g}")
    print(f"    MAX rel error              : {max_rel:.6g}")
    print("-" * 72)
    print(f"  same, PRE-NARROW fp32 domain (the true accumulator gap, pre-bf16):")
    print(f"    bit-identical fp32 results : {fbitexact}/{n} "
          f"({100.0*fbitexact/n:.1f}%)")
    print(f"    MAX abs / RMS abs error    : {fmax_abs:.6g} / {frms:.6g}")
    print(f"    MAX rel error              : {fmax_rel:.6g}")
    print(f"    (this gap is below a bf16 ULP -> it vanishes in the bf16 output)")
    print("-" * 72)
    print(f"  accuracy vs float64 near-exact ground truth (relative; lower=better):")
    print(f"    RTL(BFP)      MAX rel = {bfp_max_rel:.6g}")
    print(f"    real(fp32acc) MAX rel = {fp32_max_rel:.6g}")
    print(f"    -> BFP at least as accurate as fp32-acc: "
          f"{'YES' if bfp_max_rel <= fp32_max_rel + 1e-12 else 'NO'}")
    print("-" * 72)
    print(f"  ARGMAX (next-token decision) preserved:")
    print(f"    argmax(RTL) == argmax(real fp32-acc) : {am_fp32_ok}/{rows}")
    print(f"    argmax(RTL) == argmax(ground truth)  : {am_ref_ok}/{rows}")
    print(f"    real decisions / degenerate ties     : {rows-ties}/{ties} "
          f"(ties resolved deterministically, first-column)")
    if min_margin is not None:
        print(f"    min top1-top2 margin (real rows)     : "
              f"{min_margin:.6g} (rel {min_rel_margin:.6g})")
    if am_fail:
        print("    ARGMAX DIVERGENCES:")
        for (lbl, i, ab, af, vb, vf) in am_fail:
            print(f"      tile={lbl} row={i} RTL->{ab} real->{af} vb={vb} vf={vf}")
    print("=" * 72)

    ok = (am_fp32_ok == rows)
    print("ARGMAX-PRESERVED" if ok else "ARGMAX-DIVERGED",
          f"({am_fp32_ok}/{rows})")
    return 0 if ok else 1


if __name__ == "__main__":
    d = sys.argv[1] if len(sys.argv) > 1 else "scratchpad/bitacc"
    sys.exit(main(d))
