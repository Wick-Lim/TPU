#!/usr/bin/env python3
# flash_layout.py -- OFFLINE Flash expert-placement packer + measurement harness.
#                    (docs/IMPROVEMENT_PLAN.md P1.2 -- feeds src/flash_xbar.v)
#
# WHY THIS EXISTS
#   src/flash_xbar.v stripes Flash reads to N_CH channels (dies) by ADDRESS:
#   channel = addr[BANK_LSB +: log2(N_CH)].  It only reaches ~N_CH x single-die
#   bandwidth IF the reads a token issues SPREAD across the N_CH channels.  In a
#   GLM-5.2 MoE layer a token fetches its top-8 expert weights from Flash AT ONCE;
#   if several of those 8 experts live on the SAME channel, that channel is a
#   HOTSPOT and must serialize m reads while the others sit idle -> the token's
#   effective Flash parallelism collapses from N_CH toward N_CH/m.
#
#   The expert->channel mapping is an OFFLINE choice: expert e's PLACEMENT = which
#   channel (which addr bits) holds its weight block.  This tool PACKS experts onto
#   channels so co-activated experts land on DIFFERENT channels (and channels stay
#   load-balanced), then MEASURES the per-fetch channel balance vs naive baselines.
#
# THE FETCH UNIT (what must be parallel)
#   A layer's top-8 are fetched together, so the co-activation graph is the
#   per-LAYER top-8 set.  top-8 of layer L only ever co-activates with other
#   experts of layer L, so the co-activation graph is BLOCK-DIAGONAL by layer:
#   we pack each layer's 256 experts onto N_CH channels INDEPENDENTLY.  The result
#   is still a global assignment layout[L*256 + e] -> channel.  With top-8 over
#   N_CH=8 channels the IDEAL is 1 expert/channel per fetch == 8x == 100% of peak.
#
# Usage:  python3 tools/flash_layout.py            # build trace, pack, print table
#         python3 tools/flash_layout.py --nch 16   # sweep a different channel count
#         python3 tools/flash_layout.py --dump-map tools/flash_map.hex
import sys, random
from collections import defaultdict

# ---- reuse the SAME calibrated GLM router as tools/route_trace.py -------------
# (route_trace runs a print sweep at import time; silence it -- we only want its
#  pure generator functions make_popularity / sample_topk and its L,E,K constants.)
sys.path.insert(0, __file__.rsplit('/', 1)[0])
import os, io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    import route_trace as rt      # make_popularity / sample_topk / L,E,K

L, E, K = rt.L, rt.E, rt.K        # 75 layers, 256 experts/layer, top-8
SIGMA, RHO, SEED = 0.6, 0.30, 7   # moderate central calibration (same as --dump)


# =============================================================================
# (1) build per-(token,layer) top-8 selection sets  ==  the fetch units
#     sel[t][l] = list of 8 expert ids (layer-local 0..E-1)
# =============================================================================
def build_selection(tokens=rt.TOKENS, sigma=SIGMA, rho=RHO, seed=SEED):
    rng = random.Random(seed)
    pops = [rt.make_popularity(E, sigma, seed * 1000 + l) for l in range(L)]
    sel = [[None] * L for _ in range(tokens)]
    for t in range(tokens):
        for l in range(L):
            fresh = rt.sample_topk(pops[l], K, rng)
            if t > 0 and rho > 0:
                prev = sel[t - 1][l]
                keep = int(round(K * rho))
                merged = prev[:keep] + [x for x in fresh if x not in prev[:keep]]
                sel[t][l] = merged[:K]
            else:
                sel[t][l] = fresh
    return sel


def coactivation(sel):
    """Per-layer expert frequency and pairwise co-activation counts.
       freq[l][e]      = #tokens where e is in layer l's top-8
       co[l][(ei,ej)]  = #tokens where ei AND ej are both in layer l's top-8 (ei<ej)
    """
    freq = [defaultdict(int) for _ in range(L)]
    co   = [defaultdict(int) for _ in range(L)]
    for t in range(len(sel)):
        for l in range(L):
            s = sorted(sel[t][l])
            for i, ei in enumerate(s):
                freq[l][ei] += 1
                for ej in s[i + 1:]:
                    co[l][(ei, ej)] += 1
    return freq, co


# =============================================================================
# (2) PACKERS : expert -> channel.  Returns layout[l] = {e: channel}.
# =============================================================================
def pack_optimized(freq, co, n_ch):
    """Balanced greedy least-conflict per layer.
       Place experts most-frequent-first; each goes to the channel that (a) is
       under the per-channel capacity cap and (b) least conflicts with experts
       already there (sum of co-activation weight to that channel), tie-break by
       least load.  This is the light graph-coloring-ish pass from the brief."""
    cap = (E + n_ch - 1) // n_ch          # 256/8 = 32 experts/channel cap
    layout = []
    for l in range(L):
        # frequent experts first -- they dominate the realized hotspots
        order = sorted(range(E), key=lambda e: -freq[l][e])
        # adjacency: neighbor co-weights per expert
        adj = defaultdict(dict)
        for (ei, ej), w in co[l].items():
            adj[ei][ej] = w
            adj[ej][ei] = w
        assign = {}
        load = [0] * n_ch
        for e in order:
            best_ch, best_key = 0, None
            for ch in range(n_ch):
                if load[ch] >= cap:
                    continue
                conflict = 0
                nbrs = adj[e]
                for ne, w in nbrs.items():
                    if assign.get(ne) == ch:
                        conflict += w
                key = (conflict, load[ch], ch)   # least conflict, then least load
                if best_key is None or key < best_key:
                    best_key, best_ch = key, ch
            assign[e] = best_ch
            load[best_ch] += 1
        layout.append(assign)
    return layout


def pack_roundrobin(n_ch):
    """NAIVE baseline #1: channel = expert_id % n_ch (the addr-LSB striping that
       flash_xbar does by default with no offline placement)."""
    return [{e: e % n_ch for e in range(E)} for _ in range(L)]


def pack_random(n_ch, seed=12345):
    """NAIVE baseline #2: uniformly random placement, balanced by construction
       (shuffle a round-robin pattern so loads stay ~equal)."""
    layout = []
    for l in range(L):
        rng = random.Random(seed + l)
        slots = [e % n_ch for e in range(E)]
        rng.shuffle(slots)
        layout.append({e: slots[e] for e in range(E)})
    return layout


# =============================================================================
# (3) MEASURE : per fetch (token,layer top-8) -> channel balance + parallelism
# =============================================================================
def measure(sel, layout, n_ch):
    n_fetch = 0
    sum_eff = 0.0          # sum of effective parallelism (K/max-per-channel)
    sum_max = 0            # sum of max-per-channel
    worst_max = 0
    sum_busy_time = 0      # sum of max-per-channel == total serial "rounds"
    sum_reads = 0          # total reads (== n_fetch*K)
    hist_max = defaultdict(int)
    for t in range(len(sel)):
        for l in range(L):
            chans = [0] * n_ch
            for e in sel[t][l]:
                chans[layout[l][e]] += 1
            m = max(chans)
            k = len(sel[t][l])
            n_fetch += 1
            sum_max += m
            sum_eff += k / m
            sum_busy_time += m
            sum_reads += k
            worst_max = max(worst_max, m)
            hist_max[m] += 1
            if m > worst_max:
                worst_max = m
    avg_max  = sum_max / n_fetch
    avg_eff  = sum_eff / n_fetch
    # %peak BW (time-honest): peak finishes K reads in K/n_ch rounds; layout
    # finishes in max-per-channel rounds.  BW fraction = (K/n_ch)/max, averaged
    # over fetches as total_ideal_rounds / total_actual_rounds.
    ideal_rounds  = sum_reads / n_ch
    actual_rounds = sum_busy_time
    pct_peak = 100.0 * ideal_rounds / actual_rounds
    return dict(avg_max=avg_max, worst_max=worst_max, avg_eff=avg_eff,
                pct_peak=pct_peak, hist=dict(hist_max), n_fetch=n_fetch)


def load_balance(layout, freq, n_ch):
    """Static channel load imbalance, weighted by access frequency (how lopsided
       is total Flash traffic per channel)."""
    tot = [0] * n_ch
    for l in range(L):
        for e, ch in layout[l].items():
            tot[ch] += freq[l][e]
    s = sum(tot)
    return max(tot) / (s / n_ch)   # peak/mean (1.0 == perfectly balanced)


# =============================================================================
# driver
# =============================================================================
def main():
    n_ch = 8
    dump_map = None
    if '--nch' in sys.argv:
        n_ch = int(sys.argv[sys.argv.index('--nch') + 1])
    if '--dump-map' in sys.argv:
        dump_map = sys.argv[sys.argv.index('--dump-map') + 1]
    assert (n_ch & (n_ch - 1)) == 0, "N_CH must be a power of two (flash_xbar addr striping)"

    print(f"# flash_layout -- GLM-5.2 MoE expert placement for {n_ch}-channel Flash fabric")
    print(f"#   router calib sigma={SIGMA} rho={RHO} seed={SEED} | L={L} E={E} top-K={K} | "
          f"tokens={rt.TOKENS}")
    print(f"#   fetch unit = one layer's top-{K}; ideal = 1 expert/channel = "
          f"{min(K, n_ch)}x = 100% of peak\n")

    sel = build_selection()
    freq, co = coactivation(sel)

    layouts = {
        'OPTIMIZED (greedy least-conflict)': pack_optimized(freq, co, n_ch),
        'naive round-robin (e % N_CH)'      : pack_roundrobin(n_ch),
        'naive random (balanced)'           : pack_random(n_ch),
    }

    print(f"{'layout':<36}{'avgMax':>8}{'wstMax':>8}{'effPar':>8}{'%peakBW':>9}"
          f"{'loadImb':>9}")
    print('-' * 78)
    results = {}
    for name, lay in layouts.items():
        r = measure(sel, lay, n_ch)
        imb = load_balance(lay, freq, n_ch)
        results[name] = (r, imb)
        print(f"{name:<36}{r['avg_max']:>8.3f}{r['worst_max']:>8d}"
              f"{r['avg_eff']:>8.3f}{r['pct_peak']:>8.1f}%{imb:>9.3f}")
    print('-' * 78)
    print(f"  avgMax  = avg max-experts-on-any-one-channel per top-{K} fetch "
          f"(lower=better, ideal {max(1, K // n_ch)})")
    print(f"  effPar  = avg effective parallelism = K / maxPerChannel "
          f"(higher=better, ideal {min(K, n_ch)})")
    print(f"  %peakBW = fraction of {n_ch}-channel peak Flash BW realized "
          f"(time-honest: ideal/actual rounds)")
    print(f"  loadImb = static per-channel traffic peak/mean (1.0=perfectly balanced)\n")

    # per-fetch max-per-channel histograms (how often is there a hotspot)
    print(f"per-fetch max-per-channel distribution (% of the {results[list(layouts)[0]][0]['n_fetch']} fetches):")
    allm = sorted({m for (r, _) in results.values() for m in r['hist']})
    print(f"{'maxPerCh':<36}" + "".join(f"{m:>7}" for m in allm))
    for name, (r, _) in results.items():
        n = r['n_fetch']
        row = "".join(f"{100.0 * r['hist'].get(m, 0) / n:>6.1f}%" for m in allm)
        print(f"{name:<36}{row}")
    print()

    # honest verdict
    opt = results['OPTIMIZED (greedy least-conflict)'][0]
    rr  = results['naive round-robin (e % N_CH)'][0]
    rnd = results['naive random (balanced)'][0]
    print("VERDICT")
    print(f"  Optimized layout lifts flash_xbar to {opt['pct_peak']:.1f}% of {n_ch}x peak "
          f"({opt['avg_eff']:.2f}x effective), vs round-robin {rr['pct_peak']:.1f}% "
          f"({rr['avg_eff']:.2f}x) and random {rnd['pct_peak']:.1f}% ({rnd['avg_eff']:.2f}x).")
    gain = opt['pct_peak'] / rnd['pct_peak'] - 1.0
    print(f"  => offline placement buys ~{100 * gain:+.1f}% Flash BW over a "
          f"frequency-blind (random/round-robin) layout.")
    if n_ch <= K:
        print(f"  HONEST CEILING: with top-{K} >= {n_ch} channels EVERY fetch needs "
              f">={K // n_ch + (1 if K % n_ch else 0)} reads on some channel, so 100% peak "
              f"is reachable only if all top-{K} miss each other's channel -- popularity")
        print(f"  skew + the pigeonhole ({K} experts, {n_ch} channels) cap the realistic gain; "
              f"the win is mainly KILLING the heavy {max(allm)}-on-one-channel tail.")

    if dump_map:
        lay = layouts['OPTIMIZED (greedy least-conflict)']
        with open(dump_map, 'w') as f:
            for l in range(L):
                for e in range(E):
                    f.write(f"{lay[l][e]:x}\n")   # channel for global id l*E+e
        print(f"\nwrote optimized expert->channel map to {dump_map} "
              f"({L * E} entries, channel = layout[l*{E}+e])")


if __name__ == '__main__':
    main()
