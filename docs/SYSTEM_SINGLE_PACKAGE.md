# Single-Package GLM-5.2-FP8 Inference System — design note

> **Scope.** A system design for running the *published* `zai-org/GLM-5.2-FP8`
> checkpoint on **one low-power package** — a custom FP8 compute die integrated in-package
> with **64 GB HBM** and **1 TB Flash** — instead of a multi-chip HBM cluster. It targets
> "the real 753B model runs, at interactive-ish speed, at low power," not datacenter-scale
> real-time serving.
>
> Numbers tagged **[EST]** are system-level estimates (market-/physics-derived), not measured
> RTL results. The compute datapath this wraps is the verified RTL in this repo (see
> [`ACCEL_GLM52.md`](ACCEL_GLM52.md) and the `*_fp8` units); the memory/streaming system here
> is **designed, not built**.

---

## 1. Goal

One package that runs GLM-5.2-FP8 (753B params, ~40B active/token, 1M context) by **storing
the whole model in cheap on-package Flash and streaming the per-token working set through
fast HBM into an FP8 compute die** — exploiting MoE sparsity (8/256 experts/layer) so only a
small fraction of the 753 GB is touched per token. Optimize for **low power / low heat**
(near-memory integration) over peak throughput.

## 2. The problem

| | Size [EST] | Consequence |
|---|---|---|
| Weights (FP8, 1 B/param) | **~753 GB** (725 GB are cold routed experts) | No chip holds it on-die or in HBM |
| Latent-KV cache @ 1M ctx | **~94 GB** (MLA; an MHA cache would be 5.36 TB) | Also too big for SRAM/HBM alone |
| Compute / token | **~80 GFLOP** (~40B active × 2) | *Small* — a modest die does it in ~80 ms |

The model is **memory-bandwidth-bound, not compute-bound**: per token you must *read* the
active weights (~22 GB of routed experts + ~28 GB hot), and that read time dwarfs the math.
So the design problem is a **memory hierarchy + streaming** problem.

## 3. Architecture

```
                 ┌──────────────────────── ONE PACKAGE ────────────────────────┐
                 │                                                              │
   token ──▶     │   ┌───────────────┐  weight-pull   ┌────────────────────┐    │
                 │   │  FP8 COMPUTE  │◀──(w_req/w_col)─│  64 GB HBM (~TB/s)  │    │
                 │   │     DIE       │   bf16 acts     │  • hot weights ~28GB│    │
   logits ◀──    │   │ MLA·MoE·SwiGLU│──▶ bf16 out     │  • KV working window│    │
                 │   │ + MTP + bf16  │                 │  • EXPERT CACHE ~34GB│   │
                 │   │   tail        │                 └─────────┬──────────┘    │
                 │   └───────────────┘                  miss ▲   │ refill         │
                 │                                            │   ▼                │
                 │                              ┌──────────────────────────────┐  │
                 │                              │   1 TB FLASH (~10s GB/s,      │  │
                 │                              │   wide in-package bus)        │  │
                 │                              │  • full 725 GB cold experts   │  │
                 │                              │  • KV overflow (cold pages)    │ │
                 │                              └──────────────────────────────┘  │
                 └──────────────────────────────────────────────────────────────┘
```

Three components, three roles:
- **FP8 compute die** — the verified RTL (MLA attention, MoE, SwiGLU, RoPE, RMSNorm, LM head,
  MTP) with FP8 E4M3 weight matmuls + bf16 tail. Pulls weights via a streaming interface.
- **64 GB HBM** — the *fast working memory*: everything reused every token (hot weights), the
  KV working window, and the **routed-expert cache**.
- **1 TB Flash** — the *cheap bulk store*: the entire 753 GB FP8 model + KV overflow.

## 4. Memory tiering & map

| Tier | Size | Bandwidth [EST] | Contents | Reused every token? |
|---|---|---|---|---|
| On-die SRAM | MBs | ~10s TB/s | activations, GEMM tiles, DSA index scratch, double-buffers | — |
| **HBM** | **64 GB** | **~1–3 TB/s** | **hot weights ~28 GB** (attention all layers, shared expert, dense FFN, router, embed/LM-head, norms) + KV working window + **expert cache ~34 GB** | hot: yes |
| **Flash** | **1 TB** | **~10s GB/s** | **full 725 GB cold routed-expert pool** + KV cold pages | no (streamed on demand) |

The split that makes it work: **non-routed params (~28 GB) are a *fixed* set used every
token → resident in HBM.** The **725 GB routed experts are a *data-dependent* set (8/256 per
layer, chosen at runtime) → live in Flash, streamed/cached on demand.**

## 5. Per-token dataflow

1. **Embed** token (bf16, HBM) → residual `x`.
2. For each of 78 layers:
   a. RMSNorm(x) (bf16, HBM).
   b. **MLA attention** — weight projections (W_dq..W_o) pulled FP8 from **HBM (hot)**; q·K
      score + softmax + weighted-V in bf16; KV append + DSA-gather of 2048 rows from the **KV
      window (HBM)** / overflow (Flash).
   c. **FFN** — dense layers (first 3): SwiGLU from HBM. MoE layers (75): **router** picks
      top-8 experts → for each, **check the HBM expert cache → hit: read HBM; miss: stream the
      ~37 MB expert from Flash into HBM, evict LRU** → SwiGLU; + shared expert (HBM).
   d. Residual adds (bf16).
3. Final RMSNorm + **LM-head GEMV** (bf16, HBM) → next-token logits → argmax/sample.
4. **Prefetch**: while layer L computes, DMA layer L+1's likely experts Flash→HBM (double-buffer).

Hot reads (~28 GB) come from HBM (~10–30 ms total). The **routed-expert reads (~22 GB) are
the bottleneck** — HBM-cache hits are fast, misses hit Flash.

## 6. The bottleneck — routed-expert streaming

Per token the MoE layers need **75 × 8 = 600 expert blocks** (~37 MB each) = **~22 GB [EST]**,
scattered data-dependently across the 725 GB pool. Speed is set by:

```
  t_token ≈ max( t_compute≈80ms , t_hot_HBM≈20ms , t_routed )
  t_routed ≈ (miss_rate × 22 GB) / Flash_BW   +   (hit × 22 GB) / HBM_BW
```

With a 34 GB expert cache (≈900 of 19,200 expert-instances) and expert-popularity skew +
batch reuse, the **miss rate** — not raw compute — governs throughput.

## 7. Performance model [EST]

### 7.1 Measured cache hit rate (calibrated GLM-scale trace, RTL-confirmed)

The make-or-break unknown — the expert-cache hit rate — was simulated at GLM scale (256
experts × 75 layers, top-8) with a routing trace calibrated to a *trained* MoE router
(load-balanced → mild popularity skew, weak temporal locality), fed through the **real
`expert_cache_ctrl` RTL** (hit/miss bit-exact vs a python LRU model). Tools:
`tools/route_trace.py`, `tools/glm_cache_confirm_tb.v`.

**batch=1 (interactive decode) hit rate vs HBM cache size:**

| HBM cache | slots | uniform-ish (realistic) | skewed (optimistic) |
|---|---|---|---|
| 5.5 GB | 150 | **0 %** | **0 %** |
| 22 GB | 600 | 26 % | 51 % |
| **34 GB** (the 64 GB-HBM config) | 900 | **27 %** | 53 % |
| 66 GB | 1800 | 31 % | 58 % |

Two non-obvious findings the sim revealed:
- **Hard threshold at ~22 GB (= one token's 600-expert footprint).** Below it the hit rate is
  **0 %** — in batch=1 the decoder sweeps all 75 layers per token, so an expert is evicted long
  before the *next* token revisits its layer unless the cache holds a full token's footprint.
  **The 64 GB HBM (34 GB cache) sits just past this knee.**
- **Trained routers are load-balanced → less cacheable than a naive Zipf.** The realistic
  ("uniform-ish") hit rate at 34 GB is **~27 %**, not the ~67 % a synthetic skewed trace
  suggested.

**Batching is the real lever, not cache size.** With layer-major batched access (experts reused
within a layer across the batch) the hit rate is ~28–50 % (batch 8) to ~47–66 % (batch 32)
**even at a 5.5 GB cache** — cache size becomes nearly irrelevant.

### 7.2 Combined throughput model (batching + prefetch)

With **prefetch** the Flash *latency* is hidden behind compute (§8: 99 % of stall removed when
the compute window ≥ flash latency), so the machine runs at the Flash **bandwidth** wall. The
master equation (per-token routed footprint = 600 experts; FP8 = 22 GB, INT4 = 11 GB):

> **aggregate tokens/s ≈ Flash_BW / [ (1 − h) × footprint ]**  ·  (then × K for speculative/MTP)

where **h** is the *batched* cache hit rate measured through the real RTL (§7.1, §8):
batch 1 = 26.5 %, batch 8 = 29.7 %, batch 32 = 50.5 %. Each lever moves one term:

| Lever | What it changes | Measured effect |
|---|---|---|
| **Prefetch** | latency-bound → **bandwidth-bound** | required to reach the wall; 99 % stall cut |
| **Batching** | raises **h** → lowers (1 − h) | h 27 %→50 % (batch 1→32) ⇒ only **~1.5×** here |
| **INT4** (re-quant) | halves the **footprint** | **~2×** |
| **Speculative/MTP** | **÷K** weight passes (K tokens/pass) | **~×K** |
| **Flash bandwidth** (hardware) | raises **Flash_BW** | linear |

Aggregate tokens/s (Flash 50 / 100 GB/s, prefetch on):

| Config | h | (1−h)×footprint | @50 GB/s | @100 GB/s |
|---|---|---|---|---|
| FP8 batch 1 (single-user) | 27 % | 16 GB | ~3 | ~6 |
| FP8 batch 32 | 50 % | 11 GB | ~5 | ~9 |
| INT4 batch 32 | 50 % | 5.5 GB | ~9 | ~18 |
| INT4 batch 32 + MTP ×2 | — | — | **~18** | **~37** |

**Honest nuance — batching is not a free Nx.** In this Flash-bandwidth-bound regime batching
only helps through the hit rate, and trained-router entropy caps the reuse: batch 32 gives
**~1.5× aggregate**, not the order-of-magnitude that *compute*-bound batched serving enjoys.
And the aggregate is split across the B streams, so **per-user rate = aggregate ÷ B** — batching
trades single-user latency for aggregate throughput. (Larger B keeps raising h toward "fetch each
expert once per batch," but per-user latency keeps dropping.)

**Bottom line:** **~3–6 tokens/s single-user FP8** (prefetch hides latency; the cache helps only
modestly because trained routing is balanced). The real multipliers are **INT4 (×2)**,
**speculative/MTP (×K)**, and raw **Flash bandwidth** — *not* batching, whose aggregate gain is
modest here. So an INT4 + MTP×2 single package at ~100 GB/s in-package Flash lands around
**~20–40 aggregate tokens/s**. Interactive, not datacenter-real-time. Compute and the single die
are *not* the limit (the die idles on Flash); the wall is moving ~10–16 GB of routed-expert
weights per token across the in-package Flash bus.

## 8. MoE expert-cache subsystem (the heart of it)

Because the active expert set is **data-dependent and changes every token**, routed experts
can't be statically placed — it's a **caching + scheduling** problem:

- **Cache** (HBM, ~34 GB): LRU/LFU of expert blocks; exploits expert-popularity skew.
- **Batching**: many tokens/sequences route to overlapping experts → load once, reuse across
  the batch (biggest throughput lever; costs latency). **RTL-measured** through the committed
  `expert_cache_ctrl` at 34 GB cache: batch 1 / 8 / 32 → **26.5 % / 29.7 % / 50.5 %** hit rate
  (same router picks, only access order changes — isolating batching as the lever).
- **Prefetch/predict**: speculate next experts (the next layer's router is cheap and runs ahead)
  and DMA into HBM during the current layer's compute → hide Flash *latency* (not bandwidth).
  Built + measured as **`src/expert_cache_pf.v`** (a prefetch hint port + demand-priority
  background Flash fetch + a `demand_stall_cycles` counter; demand path bit-exact to
  `expert_cache_ctrl` with prefetch off). On the decode trace under a compute-window model, the
  demand stall is cut **59 % (compute window TC=12)** to **99 % (TC ≥ flash latency)** — the
  fetch is fully overlapped with compute when the window is long enough.
- **Layout**: store co-activated experts contiguously / aligned for sequential Flash reads
  (bandwidth- not IOPS-bound, since each expert is a ~37 MB contiguous block).
- **Speculative / MTP decoding**: GLM-5.2 ships an MTP head (built here as `mtp_head`) — verify
  K tokens per weight-load pass → cut weight traffic ~K×.

## 9. Hardware ceiling vs software leverage

| | Sets it | Knobs |
|---|---|---|
| **Hardware ceiling** | raw Flash/HBM bandwidth, in-package bus width, **energy/bit**, compute rate | more channels, wider bus, more HBM, faster die |
| **Software leverage** | how much you *actually* move + when | batching, expert cache policy, prefetch, **quantization**, speculative/MTP, storage layout, scheduling/overlap |

Software can't beat the bandwidth/energy ceiling but **gets you close to it and cuts demand** —
exactly how today's stacks (vLLM, llama.cpp/KTransformers MoE offload, DeepSpeed ZeRO-Infinity,
FlexGen) run 600B+ MoE models on a single GPU + RAM/SSD.

## 10. Power / heat

The dominant dynamic energy is **moving ~22 GB/token of weights**. In-package near-memory
(short, wide buses) cuts the energy-per-bit of that movement by **~10–100× vs an off-package
PCIe SSD**, which is the lever for the low-power/low-heat goal. FP8's 4×4-mantissa multiply
(measured: `glm_matmul_fp8` uses 18× 7-bit multipliers, vs fp32's 24×24) also keeps the
compute die's dynamic power and DSP/area down — the FP8 datapath is *already* the low-power
choice on the compute side.

## 11. Cost — memory BOM [EST, 2025–26, volatile]

| Chip | $/GB | Qty | Cost |
|---|---|---|---|
| HBM3E | ~$15–25 (AI-shortage-driven) | 64 GB (≈ 2× 36 GB or 48 GB stacks in practice) | **~$1,000–1,600** |
| NAND Flash | ~$0.05–0.10 | 1 TB | **~$50–100** |
| **Memory chips total** | | | **≈ $1,100–1,700 (HBM is >90%)** |

Key point: **HBM dominates cost; the Flash that holds the entire 753 GB model is nearly free
(~$50–100).** *Not included:* advanced in-package integration (CoWoS / 3D stacking — $thousands/
package, capacity-constrained) and the custom compute-die NRE + die cost. For context, an H100's
80 GB HBM is ~$2k of its BOM.

## 12. Mapping to the committed RTL

**What this repo already provides (the compute die):**
- The full GLM-5.2-FP8 operator datapath, fp64/faithful-fp8 verified: `fp8_e4m3`,
  `glm_matmul_fp8`, `swiglu_expert_fp8`, `mla_attn_fp8`, `moe_router_fp8`,
  `glm_decoder_block_fp8`, and the capstone **`glm_model_fp8`** (full forward pass, next-token
  argmax matches the fp8 golden).
- **Streaming weight-pull interfaces** (`w_req`/`w_col` + per-[128,128]-block bf16 scales) on
  every unit — the weight *source is abstracted*, so HBM/Flash/host can drive them.
- The **`mtp_head`** for speculative decoding.
- A small-scale DMA append/gather streaming datapath (`tpu_soc`/`axi_master_dma`/
  `scatter_gather`/`cdc_async_fifo`) exercising the control logic.

**What this design adds (not built — the system layer):**
- HBM + Flash controllers / PHYs and the in-package interconnect.
- The **MoE expert-cache controller** (tag/LRU, miss → Flash DMA → refill/evict, prefetch).
- The KV-cache pager (append + DSA-gather of the 2048-row window; overflow to Flash).
- The runtime/scheduler (batching, prefetch, speculative-decode loop) — largely software.

## 13. Open questions / honest limits

- **Expert-cache hit rate** — now estimated (§7.1) on a *calibrated* GLM-scale trace through
  the real `expert_cache_ctrl` RTL: ~27 % at batch=1 / 34 GB cache, with a hard 0 % floor below
  ~22 GB, and batching as the dominant lever. Still **calibrated, not captured** — the actual
  numbers need a *real* GLM-5.2 routing trace (can't run 753B here); the trained-router balance
  assumption could be off in either direction.
- **In-package Flash bandwidth** (~10s GB/s) is assumed; real wide-NAND integration BW must be
  validated — NAND read physics caps it well below HBM.
- **64 GB HBM is comfortable for FP8** (hot 28 GB + ~34 GB cache); **48 GB is tight**, **INT4
  makes 32–48 GB roomy** (hot ~14 GB) — but INT4 means re-quantizing away from the published
  FP8 checkpoint.
- **Advanced-packaging cost/yield** for compute + 64 GB HBM + 1 TB Flash in one package is the
  real economic risk, separate from the chip BOM above.
- This is **interactive, not datacenter-real-time**; high tokens/s/user at scale still wants
  multi-chip HBM (bandwidth), which this design deliberately trades away for cost/power.
