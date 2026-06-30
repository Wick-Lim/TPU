#!/usr/bin/env python3
# ULTRA_PERF#1 amortization measurement for batched_moe.
# Reuses route_trace.py's CALIBRATED GLM router (256 experts x 75 layers, top-8,
# load-balanced with mild popularity skew) to measure, per MoE layer, the
# EXPERT-FETCH count for a batch of B tokens = |union of their top-8 sets|, vs the
# naive per-token B*8.  Reports the curve, per-token footprint |union|/B, the
# amortization factor (B*8)/|union|, the analytic E[distinct]=E*(1-(1-K/E)^B), the
# aggregate tok/s estimate, and the knee.
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import route_trace as rt

E, K, L = rt.E, rt.K, rt.L            # 256, 8, 75
EXPERT_MB = rt.EXPERT_MB              # 37.7 MB FP8 per expert

# Calibrated central router (matches route_trace 'moderate (central)').
SIGMA, RHO, SEED = 0.6, 0.30, 7
import random

def per_layer_selections(sigma, seed, n_tokens):
    """Rebuild route_trace's per-(token,layer) top-K selection `sel`."""
    rng = random.Random(seed)
    pops = [rt.make_popularity(E, sigma, seed*1000 + l) for l in range(L)]
    sel = [[None]*L for _ in range(n_tokens)]
    for t in range(n_tokens):
        for l in range(L):
            fresh = rt.sample_topk(pops[l], K, rng)
            if t > 0 and RHO > 0:
                prev = sel[t-1][l]; keep = int(round(K*RHO))
                merged = prev[:keep] + [x for x in fresh if x not in prev[:keep]]
                sel[t][l] = merged[:K]
            else:
                sel[t][l] = fresh
    return sel

def measure(B_list):
    N = max(B_list)*8 + 32                      # enough tokens to form many batches
    sel = per_layer_selections(SIGMA, SEED, N)
    out = {}
    for B in B_list:
        # average |union| over all disjoint B-token batches x all 75 layers
        tot_union = 0; tot_naive = 0; n = 0
        b0 = 0
        while b0 + B <= N:
            for l in range(L):
                u = set()
                for t in range(b0, b0+B):
                    u.update(sel[t][l])
                tot_union += len(u)
                tot_naive += B*K
                n += 1
            b0 += B
        out[B] = (tot_union/n, tot_naive/n)     # (mean|union|, mean B*K)
    return out

def analytic(B):
    return E*(1.0 - (1.0 - K/E)**B)

if __name__ == "__main__":
    B_list = [1, 8, 32, 64, 256]
    res = measure(B_list)

    # Flash bandwidth model (single channel HBM-class for the expert weights).
    FLASH_GBps = 1600.0                          # aggregate expert-weight read BW (GB/s)
    h = 0.0                                       # cold cache: footprint = full fetch
    # tok/s ~= Flash_BW / [ (1-h) * footprint_bytes_per_token ]
    # footprint_per_token = (|union|/B) experts/layer * L layers * EXPERT_MB
    def toks(foot_experts_per_tok_per_layer):
        bytes_per_tok = foot_experts_per_tok_per_layer * L * EXPERT_MB * 1e6
        return FLASH_GBps*1e9 / ((1.0-h) * bytes_per_tok)

    print(f"GLM-5.2: E={E} experts/layer, K={K} top-k, L={L} MoE layers, "
          f"{EXPERT_MB} MB/expert FP8")
    print(f"router: calibrated central (sigma={SIGMA}, rho={RHO}); Flash BW={FLASH_GBps} GB/s\n")
    print(f"{'B':>4} | {'|union|':>9} {'B*K':>6} | {'/tok':>6} {'analytic':>9} | "
          f"{'amort=(B*K)/|U|':>16} | {'tok/s/user':>11} {'agg tok/s':>11}")
    print("-"*96)
    base_foot = None
    for B in B_list:
        u, naive = res[B]
        foot_per_tok = u / B                      # experts/layer/token
        amort = naive / u                         # (B*K)/|union|
        an = analytic(B)
        ts_user = toks(foot_per_tok)              # per-user tok/s (footprint shrinks)
        ts_agg  = ts_user * B                     # B users share the union -> aggregate
        if B == 1: base_foot = foot_per_tok
        print(f"{B:>4} | {u:>9.1f} {naive:>6.0f} | {foot_per_tok:>6.3f} {an:>9.1f} | "
              f"{amort:>16.3f} | {ts_user:>11.2f} {ts_agg:>11.1f}")

    # knee: where marginal per-user footprint reduction per doubling drops below 10%.
    print("\nknee analysis (per-user footprint = |union|/B, lower is better):")
    prev = None
    knee = None
    for B in [1,2,4,8,16,32,64,128,256]:
        r = measure([B])[B]
        f = r[0]/B
        if prev is not None:
            gain = (prev - f)/prev*100
            tag = ""
            if knee is None and gain < 10.0:
                knee = B; tag = "  <-- knee (per-user gain < 10%/doubling)"
            print(f"  B={B:>3}: foot/tok={f:6.3f} experts/layer  "
                  f"(reduction vs B/2: {gain:5.1f}%){tag}")
        else:
            print(f"  B={B:>3}: foot/tok={f:6.3f} experts/layer  (baseline)")
        prev = f
    print(f"\nSaturation: E[distinct] -> {E} (all experts) as B grows; "
          f"analytic |union| at B=256 = {analytic(256):.1f}/{E} "
          f"({100*analytic(256)/E:.1f}% of pool).")
    if knee:
        print(f"KNEE at B~={knee}: beyond this the union is already most of the "
              f"pool, so per-user amortization saturates (aggregate tok/s still "
              f"rises, but sub-linearly).")
