#!/usr/bin/env python3
# GLM-5.2-scale MoE routing-trace generator + LRU expert-cache hit-rate sweep.
# Calibrated (NOT captured): models a TRAINED MoE router -- load-balanced (mild popularity
# skew) with weak temporal locality, per-layer-independent. GLM-5.2: 256 experts/layer,
# top-8, 75 MoE layers. Expert weight = 37.7 MB (FP8). Cache id space = layer*256+idx (19200).
#
# Usage:
#   python3 tools/route_trace.py           # print hit-rate-vs-cache sweep (decode + batched)
#   python3 tools/route_trace.py --dump    # write tools/glm_trace.hex + python LRU reference;
#                                          # then iverilog tools/glm_cache_confirm_tb.v +
#                                          # src/expert_cache_ctrl.v confirms the RTL matches.
import random, math, sys

L, E, K = 75, 256, 8            # MoE layers, experts/layer, top-k
EXPERT_MB = 37.7               # FP8 weight per expert
TOKENS = 160                   # decoded tokens in the trace

def make_popularity(E, sigma, seed):
    # log-normal-ish popularity per layer: w_i ∝ exp(sigma*g_i), g~N(0,1).
    # sigma=0 -> uniform; larger sigma -> more skew. cap to realistic load imbalance.
    rng = random.Random(seed)
    w = [math.exp(sigma * rng.gauss(0, 1)) for _ in range(E)]
    s = sum(w); return [x / s for x in w]

def sample_topk(pop, K, rng):
    # K distinct experts sampled without replacement, weighted by popularity.
    idx = list(range(len(pop))); w = list(pop); out = []
    for _ in range(K):
        t = rng.random() * sum(w); c = 0.0
        for j in range(len(idx)):
            c += w[j]
            if c >= t:
                out.append(idx[j]); del idx[j]; del w[j]; break
    return out

def gen_trace(sigma, rho, pattern, seed=1):
    # sigma: popularity skew. rho: temporal locality (carry fraction of prev token's picks).
    # pattern: 'decode' (token-major: per token sweep all L layers) or 'batchB' (layer-major
    #          across a batch of B tokens -> within-layer reuse).
    rng = random.Random(seed)
    pops = [make_popularity(E, sigma, seed * 1000 + l) for l in range(L)]
    # per-layer selection per token
    sel = [[None] * L for _ in range(TOKENS)]
    for t in range(TOKENS):
        for l in range(L):
            fresh = sample_topk(pops[l], K, rng)
            if t > 0 and rho > 0:
                prev = sel[t - 1][l]
                keep = int(round(K * rho))
                merged = prev[:keep] + [x for x in fresh if x not in prev[:keep]]
                sel[t][l] = merged[:K]
            else:
                sel[t][l] = fresh
    # flatten to access order
    trace = []
    if pattern == 'decode':
        for t in range(TOKENS):
            for l in range(L):
                for e in sel[t][l]:
                    trace.append(l * E + e)
    elif pattern.startswith('batch'):
        B = int(pattern[5:])
        for tb in range(0, TOKENS, B):
            for l in range(L):                      # layer-major across the batch
                for t in range(tb, min(tb + B, TOKENS)):
                    for e in sel[t][l]:
                        trace.append(l * E + e)
    return trace

def lru_hitrate(trace, slots):
    # exact LRU (same policy as expert_cache_ctrl, proven bit-exact). O(1)/access via dict order.
    from collections import OrderedDict
    cache = OrderedDict(); hit = 0
    for x in trace:
        if x in cache:
            hit += 1; cache.move_to_end(x)
        else:
            cache[x] = True
            if len(cache) > slots: cache.popitem(last=False)
    return hit, len(trace)

# --dump: write a trace + python LRU reference for the RTL confirmation TB, then exit
if '--dump' in sys.argv:
    tr = gen_trace(0.6, 0.30, 'decode', seed=7)[:18000]   # moderate calib, ~30 tokens
    with open('tools/glm_trace.hex', 'w') as f:
        for x in tr: f.write(f"{x:04x}\n")
    for slots in (600, 900):
        h, n = lru_hitrate(tr, slots)
        print(f"PYREF slots={slots} hits={h} miss={n-h} hit_rate={100*h/n:.2f}%")
    print(f"wrote tools/glm_trace.hex ({len(tr)} accesses)")
    sys.exit(0)

# per-token routed footprint
FOOT = L * K                                   # 600 experts touched per token
print(f"GLM-scale: L={L} E={E} K={K} -> per-token footprint = {FOOT} experts = {FOOT*EXPERT_MB/1024:.1f} GB")
print(f"trace tokens={TOKENS}, total experts pool = {L*E} (={L*E*EXPERT_MB/1024:.0f} GB FP8)\n")

SLOT_SIZES = [150, 300, 600, 900, 1200, 1800, 2700]   # -> HBM cache GB
CALIB = [("uniform-ish (pessimistic)", 0.3, 0.20),
         ("moderate (central)",        0.6, 0.30),
         ("skewed+local (optimistic)", 1.0, 0.45)]

for pattern in ['decode', 'batch8', 'batch32']:
    print(f"================ access pattern: {pattern} ================")
    print(f"{'slots':>6}{'HBM_GB':>8} | " + " | ".join(f"{n.split()[0]:>11}" for n,_,_ in CALIB))
    for slots in SLOT_SIZES:
        gb = slots * EXPERT_MB / 1024
        rates = []
        for _, sig, rho in CALIB:
            tr = gen_trace(sig, rho, pattern)
            h, n = lru_hitrate(tr, slots)
            rates.append(100.0 * h / n)
        print(f"{slots:>6}{gb:>8.1f} | " + " | ".join(f"{r:>10.1f}%" for r in rates))
    print()
