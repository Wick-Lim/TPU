# ULTRA_PERF — Ranked Ultra-High-Performance Opportunity Report
### GLM-5.2-FP8 single-module accelerator (FP8 die + 64 GB DDR5 + 1 TB Flash)

**Scope.** Opportunities *beyond* the already-built+measured levers (flash_xbar 7.99× latency-hide,
weight_decomp 1.34×, MTP K=2 +23%, clk_en_ctrl 74% idle-gate, flash_layout +40% balance, the −87.6% BFP
accumulator, fmax fixes, predictor-prefetch [measured no-op]). Numbers marked **[EST]** are model estimates,
not measured. Compute is bit-exact to real GLM-5.2-FP8 (docs/BIT_ACCURACY.md); levers that change outputs are
flagged **NOT bit-exact**.

**The one equation.** The workload is Flash-bandwidth-bound:
`tok/s ≈ Flash_BW / [(1−h)·footprint] · K`. Only three classes of idea move this wall:
**(i) move fewer expert bytes** (sparsity / dedup / stronger decomp), **(ii) compute at the data** (near-Flash),
**(iii) raise speculative K** (better drafts + batched verify). Everything else is incremental — die-side
fmax/area work does **not** move the wall (the die is already ~75% idle behind Flash).

---

## 1. Headline table — TOP opportunities (ranked by impact × feasibility)

| # | Opportunity | Mechanism (1-line) | Quantified impact **[EST]** | Where | Effort | Ceiling? |
|---|-------------|--------------------|------------------------------|-------|--------|----------|
| 1 | **Expert-grouped layer-synchronous batched MoE** | Fetch the per-layer expert *union* once from Flash, reuse across B token-rows | Aggregate **6–8×**: ~36–50 tok/s @B≈256 vs ~6 single-user | system-arch | high | ✅ |
| 2 | **PE_M batch-widening of FP8 wrappers** | swiglu/router/mla/mtp hardcode PE_M=1 → widen to 8–32; one weight fetch → B rows | Silicon enabler for #1; 0 extra dequant muls, 0 extra weight BW | **rtl-here** | med | (enabler) |
| 3 | **Stronger weight decompressor (context-rANS)** | Spend idle die on context-modeled rANS in Flash→DDR5 refill path | 1.34×→**1.5–1.7×** fewer Flash bytes → ~16→18–20 tok/s, ~3.8→3.0 J/tok | **rtl-here** | high | ✅ |
| 4 | **Resident dense draft model → high-K spec decode** | ~1–3B DDR5-resident draft proposes K=4–8; target verifies in one pass | K_eff 1.7→**3–5** → ~2–3× single-user, **bit-exact** | rtl-here* | high | ✅ |
| 5 | **Batched single-pass verification** | Make the main forward process {base + D draft} positions *together* | The gate for ANY spec Flash gain (today's ÷K on Flash = **0**) | **rtl-here** | high | ✅ |
| 6 | **Contextual activation sparsity in SwiGLU** | Low-rank predictor → fetch only active W_up cols / W_down rows | ~1.5–3× fewer routed bytes (22→8–14 GB); **NOT bit-exact** | **rtl-here** | high | ✅ |
| 7 | **Dynamic top-k expert pruning (k_eff<8)** | Threshold-mask tiny renormalized gates in the router FSM | ~1.3–1.6× fewer routed bytes; cheapest big lever; **NOT bit-exact** | **rtl-here** | low | ✅ |
| 8 | **Exact router-driven prefetch + K-token union** | Run cheap router GEMV for K spec tokens → exact union prefetch | Demand-stall 81%→~99% + ~1.5–2× byte-dedup; **bit-exact** | **rtl-here** | med | ✅ |
| 9 | **Unmask + compress the 28 GB hot-weight DDR5 read** | Point weight_decomp at hot path; amortize K×; +DDR5 channels | The *next* wall: ~21 GB/56→42 ms → ~24 tok/s hot-bound floor | **rtl-here** | med | ✅ |
| 10 | **IndexShare (DSA index once / 4 layers)** | Cache index-list; skip dsa_indexer on 3 of 4 layers (model-faithful) | At 1M ctx: index-read 10→2.5 GB/tok → keeps long-ctx Flash-bound | **rtl-here** | med | ✅ |
| 11 | **Parallel/pipelined DSA indexer** | Replace in-order 1-MAC dot with 128-lane reduction tree | At 1M ctx: ~0.05→~6 tok/s (kills O(S)·7-cyc drain); **bit-exact** | **rtl-here** | high | ✅ |
| 12 | **MLA weight absorption (attend in 512-dim latent)** | Fold W_uk into q, W_uv into W_o; drop per-key up-projection | Removes ~3.2e5 per-key GEMMs/tok; **bit-exact** (matmul reassoc) | **rtl-here** | high | ✅ |
| 13 | **Near-Flash / computational-storage expert compute** | Move FP8 MACs to NAND; stream 12 KB act down, 12 KB result up | ~1000× fewer bus bytes/expert → **10×+** ceiling. **No J/tok win** | out-of-scope | high | ✅ |
| 14 | **Batch × MTP multiply** | Run all B streams' MTP drafts in the same grouped pass | ~1.7× *on top of* batch → ~60–85 tok/s aggregate @B≈256 | **rtl-here** | low | (compose) |
| 15 | **Paged multi-sequence KV cache** | vLLM-style block table for B independent contexts | Enabler: without it B>1 *distinct users* impossible | **rtl-here** | high | (enabler) |

\* #4 RTL substrate (g_kn verifier) exists; the draft *weights* are a training task.

---

## 2. CEILING-CHANGERS vs INCREMENTAL/CONDITIONAL

### 2a. CEILING-CHANGERS — flip Flash-bound → compute-bound, or 10×-class

These touch `(1−h)·footprint`, `K`, or the bus itself.

- **Near-Flash compute (#13)** — the single biggest lever: ~1000× fewer bus bytes/expert lifts the ceiling
  to the NAND internal-sense limit (**10×+ tok/s [EST]**). Caveats: custom CSD/PIM silicon (**out-of-scope**
  for this repo), and it does **not** cut J/token (the sense energy is the cost). The compute core is reusable
  verified RTL.
- **Aggregate batching (#1/#2/#3-knee)** — the datacenter regime. Reframes batching from the doc's "~1.5×"
  (a B=32, LRU-hit-rate artifact) to a **6–8× aggregate** lever via expert-union reuse. New knee at **B≈256**
  (all 256 experts active: `E[distinct]=256·(1−0.96875^B)`), new ceiling = the compute roofline
  **~50 tok/s aggregate @100 GB/s Flash**, reached near B≈355. Per-user latency floors at ~0.14 tok/s → an
  **offline/throughput product, not chat**.
- **Stronger weight decomp (#3)** — only thing that uses the 75%-idle die to cut the *actual* wall.
  1.34→1.5–1.7× is a direct multiplier on **both** single-user tok/s **and** J/token (Flash bytes ≈ 80% of
  per-token energy). **rtl-here**, faithful, stacks multiplicatively.
- **Higher speculative K (#4 + #5 + #8)** — the faithful single-user lever. Today the ÷K on Flash is **0**
  (glm_model_fp8 is strictly one-position; the g_kn counter is harness-fed). #5 batched-verify is the *gate*;
  #4 a resident dense draft raises α (K_eff 1.7→3–5); #8 exact-router-union dedups the K-token expert set.
  **All bit-exact** (target verifies every token). Honest cap: the MoE union penalty (#22 below) keeps
  K_eff_flash well below the dense-model ×K.
- **Activation sparsity (#6) + dynamic top-k (#7)** — shrink `footprint` directly (~1.5–3× and ~1.3–1.6×).
  Both **NOT bit-exact** (quality knobs). #7 is the cheapest big lever (a comparator+mask in the router).
- **Hot-weight DDR5 second wall (#9)** — not today's wall, but bites once the first 3–4 Flash levers land;
  decomp+K-amortize+channels keep the post-Flash-fix regime from re-stalling at a ~24 tok/s DDR5 floor.
- **Long-context faithful set (#10 IndexShare, #11 parallel indexer, #12 MLA absorption)** — at 1M ctx the
  O(S) indexer and per-key up-projection, *not* Flash, become the wall (in-order indexer ≈ 0.05 tok/s). These
  restore the Flash-bound ~6 tok/s at extreme context. All **model-faithful / bit-exact**.

### 2b. INCREMENTAL / CONDITIONAL — smaller or regime-specific

- **Batch × MTP (#14)** — free ~1.7× *compose* on top of batching; multiplicative, not a new ceiling.
- **Enablers (#2 PE_M widen, #15 paged KV, continuous-batch scheduler)** — required to *realize* #1, but
  add no ceiling by themselves. Scheduler defends the peak (a half-full batch pays full union for half the
  tokens → half aggregate).
- **Attention hot-path batch reuse + DDR5 realloc** — defensive; prevents attention/hot-weight becoming the
  secondary bottleneck at high B; frees ~15–25 GB DDR5 for KV.
- **Pipelined KV gather / FP8 latent-KV / widen attention engines** — long-ctx *latency/footprint*, ~1% of
  bytes; not ceiling movers.
- **PE-array scaling (wider PE_N), fmax tail fixes, output-stationary SRAM, pipeline MLA softmax** — **~0 on
  single-user decode** (compute already hidden 4× under Flash). Value is **prefill/TTFT** (linear in array
  size) and energy/voltage headroom.
- **Pipeline draft into Flash shadow / reuse accepted KV / deeper layer-pipeline** — latency-only on the
  already-idle die; **0% on the bandwidth ceiling**.

### 2c. HONEST NEGATIVES (do not re-propose)

- **Multiple compute dies sharing one DDR5+Flash** — **0** within the module (shared bus). Linear only as
  N *separate* modules (N× cost). The bottleneck is the bus, not die count.
- **Deeper prefetch / layer-pipeline** — already bandwidth-bound (81% demand-stall removed); residual is the
  irreducible DDR5 read floor (~5% utilization, not throughput).
- **Predictor-prefetch** — measured no-op; hit-rate is entropy-capped by fine-grained routing.
- **Hierarchical block-max pruned indexing** — could cut indexer 5–20× at 1M but is **off the faithful path**
  (changes outputs); escape hatch only.

---

## 3. "What to build" — the single biggest RTL-here lever

### Build A — Expert-grouped batched MoE (#1+#2+#15) — the aggregate ceiling
The only thing that needs to be *invented*; the math die is ready.

1. **PE_M batch-widen the wrappers (#2, the keystone).** glm_matmul_fp8 already supports PE_M>1 (8·PE_M
   a_shift port, per-row accumulator banks, PE_M·PE_N dequant walk; the scarce 24×24 dequant muls stay pinned
   at NB **regardless of PE_M**). swiglu_expert_fp8 / moe_router_fp8 / mla_attn_fp8 / mtp_head_fp8 hardcode
   PE_M=1. Set `PE_M=TILE` (8–32): widen `xbuf→[TILE][HIDDEN]`, `hbuf→[TILE][INTER]`, carry TILE per-row
   a_shift/hsh (dynamic-quant exponent is already per-vector), stream the **same** w_col to all rows.
2. **Grouped dispatcher.** Per MoE layer: (a) route all B tokens, histogram top-8 picks into a per-expert
   token-list (reuse scatter_gather.v + topk_select); (b) for each *distinct* active expert, gather its rows
   into a PE_M tile, fetch the ~37 MB expert from Flash/DDR5 **once**, run the grouped GEMM, scatter back;
   (c) advance all B tokens to L+1 in lockstep. Union ≤256 experts ≤9.5 GB → resident in 64 GB DDR5; the
   union is the **only** Flash traffic, shared across all B rows.
3. **Paged KV (#15).** Extend kv_cache_pager from single-ring to a block table `(seq_id, logical_pos)→page`
   + per-seq append_count + shared pool + per-seq DSA gather. MLA latent KV ~1 KB/tok/layer → B=256 at few-K
   ctx fits the DDR5 freed by the shrunken (one-layer-union) expert cache.

**Payoff [EST]:** per-token routed footprint `75·256·(1−0.96875^B)·37MB/B` → B=256 ≈ 2.8 GB/tok (7.9×) →
~36 tok/s aggregate @100 GB/s; ~50 tok/s compute-roofline cap near B≈355; ×1.7 more with MTP (#14).

### Build B — Faithful high-K speculation (#5+#4+#8) — the single-user ceiling
1. **Batched verify (#5).** Restructure glm_model_fp8 / spec_decode_top to forward {base + D draft} positions
   in one main pass; per-layer expert streaming serves all; commit the accepted prefix (g_kn already computes
   longest-accepted = greedy = bit-exact).
2. **Resident dense draft (#4).** Hold a ~1–3B DDR5-resident draft (attention + shared expert + heads only →
   **zero Flash**) proposing K=4–8; or Medusa heads (no chain decay). Output stays bit-exact (target verifies).
3. **Exact union prefetch (#8).** Run the cheap router GEMV for all K spec tokens during draft compute →
   exact top-8 union → prefetch before needed. Hides Flash latency (81%→~99%) and dedups the K-token set.
4. **Geometry rule (#22):** on a Flash-bound MoE, prefer **deep-narrow chains** over wide trees — each branch
   drags in (1−r) divergent experts you may reject; a naive 4-wide tree at r=0.35 → K_eff_flash≈0.85
   (a **regression**). Depth keeps K_eff_flash>1.

---

## 4. Honest roofline — the three regimes

| Regime | Product | Bound by | tok/s (today → with levers) **[EST]** | Levers that apply |
|--------|---------|----------|----------------------------------------|-------------------|
| **Single-user** | USB-C box, interactive chat | Flash BW (16 GB routed/tok @100 GB/s) | ~3–12 → **~25–40** | decomp #3, sparsity #6, top-k #7, draft-K #4+5+8, hot-weight #9 |
| **Aggregate-serving** | datacenter batch, offline | Flash union then compute roofline | ~6 (B=1) → **~36–50** (×1.7 MTP → 60–85) | batched MoE #1, PE_M #2, paged KV #15, B≈256 knee, scheduler |
| **Compute-bound** | hypothetical full-resident HBM | FP8 roofline (80 GFLOP/tok ÷ ~4 TFLOP/s ≈ 20 ms) | **~40–50** ceiling | only if experts free (near-Flash #13 or HBM); array-scale for prefill |

**Reading it.** Single-user stacking the faithful RTL levers (flash_xbar N× + decomp 1.34→~1.6× +
activation-sparsity ~2× + draft-K~4 + hot-weight decomp) projects **~25–40 tok/s @100 GB/s**, approaching the
~50 tok/s compute ceiling — at which point the answer is a bigger/cheaper compute die (compute is **not** the
BOM cost). Aggregate batching reaches the **same** ~50 tok/s ceiling but as throughput (per-user latency
floors at ~0.14 tok/s — offline only). The **only** way to raise the ceiling *itself* above ~50 is near-Flash
compute (#13, custom silicon). Cache cleverness (predictor) and extra dies are provably capped.

---

## 5. Dedup map (overlapping ideas collapsed)

- **PE_M widening** appeared 3× (wrapper-widen / expert-path-widen / attention-widen) → one keystone **#2**,
  with attention/hot-path as a defensive sub-case.
- **Draft model / raise-α** appeared 4× (on-die draft, resident draft, native multi-head, raise-α) →
  consolidated into **#4** (resident dense/Medusa draft, bit-exact).
- **Batched verify** appeared in both SPECULATIVE and CROSS-CUTTING → **#5** (the gate), with the
  geometry-rule (deep-narrow #22) and union-aware scheduling (#8) as its design constraints.
- **Exact router prefetch + K-token union** appeared in SPECULATIVE and CROSS-CUTTING → **#8**.
- **B≈256 knee / continuous-batch scheduler** are framing+defense of **#1**, not separate ceilings.
- **fmax tails / array-scale / softmax-pipeline / SRAM-stationary** all collapse to one note: **prefill &
  energy only, ~0 on single-user decode** (§2b).

---

*Estimates [EST] are first-order model projections (master eq + roofline), not silicon measurements. Faithful
levers preserve GLM-5.2-FP8 bit-exactness; #6/#7 (and FP8 latent-KV, hierarchical indexing) are quality knobs
and must be validated against the accuracy contract before shipping.*
