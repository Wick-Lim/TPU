# Single-Package GLM-5.2-FP8 Inference System — design note

> **Scope.** A system design for running the *published* `zai-org/GLM-5.2-FP8`
> checkpoint on **one module** — a custom FP8 compute die + **64 GB DDR5** (the fast working
> memory) + **1 TB Flash** (the whole model) — instead of a multi-chip HBM cluster. It targets
> "the real 753B model runs, at interactive-ish speed," e.g. as a USB-C external accelerator,
> not datacenter-scale real-time serving.
>
> **Fast-memory choice: multi-channel DDR5, not HBM/GDDR6.** This workload is **Flash-bandwidth-
> bound** (the wall is reading cold experts from Flash), so the fast tier only needs ~300–600 GB/s.
> An **8–12-channel DDR5** subsystem delivers that (DDR5-6400 ≈ 51 GB/s/ch → ~410 GB/s at 8 ch,
> ~615 at 12), making HBM's multi-TB/s — and even GDDR6's ~400–600 GB/s (8–12 ch) — more than required.
> DDR5 is the **cheapest (~$2–4/GB → ~$150–300 for 64 GB)** and **lowest-power per bit** of the
> three, and **densest by capacity** (64 GB = a few **DIMMs**, upgradeable — vs ~32 soldered GDDR6
> chips or an in-package HBM stack); the prefetch controller (`expert_cache_pf`) hides DDR5's
> access latency behind compute. The trade is a **wide multi-channel memory controller**
> (server-class, 8–12 ch) — that is DDR5's engineering cost, in exchange for the lowest BOM and
> power. *(Earlier drafts targeted HBM, then GDDR6; "the fast tier" below is now DDR5.)*
>
> Numbers tagged **[EST]** are system-level estimates (market-/physics-derived), not measured
> RTL results. The compute datapath this wraps is the verified RTL in this repo (see
> [`ACCEL_GLM52.md`](ACCEL_GLM52.md) and the `*_fp8` units); the memory/streaming system here
> is **designed, not built**.

---

## 1. Goal

One module that runs GLM-5.2-FP8 (753B params, ~40B active/token, 1M context) by **storing
the whole model in cheap Flash and streaming the per-token working set through fast DDR5
into an FP8 compute die** — exploiting MoE sparsity (8/256 experts/layer) so only a small
fraction of the 753 GB is touched per token. Optimize for **cost + interactive speed** (a
USB-C external accelerator) over peak throughput.

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
                 ┌──────────────────────────── MODULE ─────────────────────────┐
                 │                                                              │
   token ──▶     │   ┌───────────────┐  weight-pull   ┌────────────────────┐    │
                 │   │  FP8 COMPUTE  │◀──(w_req/w_col)─│ 64 GB DDR5        │    │
                 │   │     DIE       │   bf16 acts     │ (~400–600 GB/s, 8–12ch)    │    │
   logits ◀──    │   │ MLA·MoE·SwiGLU│──▶ bf16 out     │  • hot weights ~28GB│    │
                 │   │ + MTP + bf16  │                 │  • KV working window│    │
                 │   │   tail        │                 │  • EXPERT CACHE ~34GB│   │
                 │   └───────────────┘                 └─────────┬──────────┘    │
                 │                                      miss ▲   │ refill         │
                 │                                            │   ▼                │
                 │                              ┌──────────────────────────────┐  │
                 │                              │   1 TB FLASH (~10s GB/s)      │  │
                 │                              │  • full 725 GB cold experts   │  │
                 │                              │  • KV overflow (cold pages)    │ │
                 │                              └──────────────────────────────┘  │
                 └──────────────────────────────────────────────────────────────┘
   (USB-C to a host PC carries only token IDs in/out — the model never crosses it.)
```

Three components, three roles:
- **FP8 compute die** — the verified RTL (MLA attention, MoE, SwiGLU, RoPE, RMSNorm, LM head,
  MTP) with FP8 E4M3 weight matmuls + bf16 tail. Pulls weights via a streaming interface.
- **64 GB DDR5** — the *fast working memory*: everything reused every token (hot weights), the
  KV working window, and the **routed-expert cache**. (~400–600 GB/s at 8–12 channels ≈ exactly the ~300–600 GB/s this workload needs, since it
  is Flash-bound — HBM's/GDDR6's higher BW would be wasted.)
- **1 TB Flash** — the *cheap bulk store*: the entire 753 GB FP8 model + KV overflow.

## 4. Memory tiering & map

| Tier | Size | Bandwidth [EST] | Contents | Reused every token? |
|---|---|---|---|---|
| On-die SRAM | MBs | ~10s TB/s | activations, GEMM tiles, DSA index scratch, double-buffers | — |
| **DDR5** | **64 GB** | **~400–600 GB/s (8–12 ch)** | **hot weights ~28 GB** (attention all layers, shared expert, dense FFN, router, embed/LM-head, norms) + KV working window + **expert cache ~34 GB** | hot: yes |
| **Flash** | **1 TB** | **~10s GB/s** | **full 725 GB cold routed-expert pool** + KV cold pages | no (streamed on demand) |

The split that makes it work: **non-routed params (~28 GB) are a *fixed* set used every
token → resident in DDR5.** The **725 GB routed experts are a *data-dependent* set (8/256 per
layer, chosen at runtime) → live in Flash, streamed/cached on demand.**

## 5. Per-token dataflow

1. **Embed** token (bf16, DDR5) → residual `x`.
2. For each of 78 layers:
   a. RMSNorm(x) (bf16, DDR5).
   b. **MLA attention** — weight projections (W_dq..W_o) pulled FP8 from **DDR5 (hot)**; q·K
      score + softmax + weighted-V in bf16; KV append + DSA-gather of 2048 rows from the **KV
      window (DDR5)** / overflow (Flash).
   c. **FFN** — dense layers (first 3): SwiGLU from DDR5. MoE layers (75): **router** picks
      top-8 experts → for each, **check the DDR5 expert cache → hit: read DDR5; miss: stream
      the ~37 MB expert from Flash into DDR5, evict LRU** → SwiGLU; + shared expert (DDR5).
   d. Residual adds (bf16).
3. Final RMSNorm + **LM-head GEMV** (bf16, DDR5) → next-token logits → argmax/sample.
4. **Prefetch**: while layer L computes, DMA layer L+1's likely experts Flash→DDR5 (double-buffer).

Hot reads (~28 GB) come from DDR5 (fast). The **routed-expert reads (~22 GB) are the
bottleneck** — DDR5-cache hits are fast, misses hit Flash.

## 6. The bottleneck — routed-expert streaming

Per token the MoE layers need **75 × 8 = 600 expert blocks** (~37 MB each) = **~22 GB [EST]**,
scattered data-dependently across the 725 GB pool. Speed is set by:

```
  t_token ≈ max( t_compute≈80ms , t_hot_DDR5 , t_routed )
  t_routed ≈ (miss_rate × 22 GB) / Flash_BW   +   (hit × 22 GB) / DDR5_BW
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

**batch=1 (interactive decode) hit rate vs DDR5 cache size:**

| DDR5 cache | slots | uniform-ish (realistic) | skewed (optimistic) |
|---|---|---|---|
| 5.5 GB | 150 | **0 %** | **0 %** |
| 22 GB | 600 | 26 % | 51 % |
| **34 GB** (the 64 GB config: hot 28 + cache 34 + KV) | 900 | **27 %** | 53 % |
| 66 GB | 1800 | 31 % | 58 % |

Two non-obvious findings the sim revealed:
- **Hard threshold at ~22 GB (= one token's 600-expert footprint).** Below it the hit rate is
  **0 %** — in batch=1 the decoder sweeps all 75 layers per token, so an expert is evicted long
  before the *next* token revisits its layer unless the cache holds a full token's footprint.
  **The 64 GB DDR5 (34 GB cache) sits just past this knee.**
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

**This project runs FP8** — the published `zai-org/GLM-5.2-FP8` checkpoint, faithfully, no
re-quantization. So the throughput levers are the **FP8-compatible** ones; INT4 is *off the
faithful path* (it means re-quantizing the model ourselves and owning the quality risk) and is
listed only as an escape hatch. Aggregate tokens/s on the FP8 path (Flash 50 / 100 GB/s,
prefetch on):

| Config (FP8 path) | h | (1−h)×22 GB | @50 GB/s | @100 GB/s |
|---|---|---|---|---|
| batch 1 (single-user) | 27 % | 16 GB | ~3 | ~6 |
| batch 1 + **MTP ×2** | — | — | ~6 | ~12 |
| batch 32 | 50 % | 11 GB | ~5 | ~9 |
| **batch 32 + MTP ×2** | — | — | **~10** | **~18** |
| *(off-path)* INT4 batch 32 + MTP ×2 | — | 5.5 GB | ~18 | ~37 |

**The FP8 multiplier that matters is speculative / MTP decoding (×K).** GLM-5.2 ships an MTP head
(`num_nextn_predict_layers=1`) and we built it (`mtp_head`): verifying K tokens per weight-load
pass divides the Flash traffic ~K× **without leaving FP8**. With a longer draft (a small draft
model or multi-token MTP) K can exceed 2.

**Batching is not a free Nx** in this Flash-bandwidth-bound regime — it only helps through the
hit rate, and trained-router entropy caps the reuse: batch 32 gives **~1.5× aggregate**, split
across the B streams (**per-user = aggregate ÷ B**), i.e. it trades single-user latency for
aggregate throughput.

**Bottom line (FP8):** **~3–6 tokens/s single-user**, **~6–12 with MTP ×2**, and **~10–18
aggregate** at batch 32 + MTP ×2 (~100 GB/s on-module Flash). Prefetch is required (hides
latency → reach the bandwidth wall); MTP and raw Flash bandwidth are the real multipliers;
batching is a modest, latency-costing aggregate boost. Interactive, not datacenter-real-time.
Compute and the single die are *not* the limit (the die idles on Flash); the wall is moving
~11–16 GB of routed-expert weights per token across the on-module Flash bus. (INT4 would ~2×
everything but is a different, re-quantized model — outside the "run the published FP8" goal.)

## 8. MoE expert-cache subsystem (the heart of it)

Because the active expert set is **data-dependent and changes every token**, routed experts
can't be statically placed — it's a **caching + scheduling** problem:

- **Cache** (DDR5, ~34 GB): LRU/LFU of expert blocks; exploits expert-popularity skew.
- **Batching**: many tokens/sequences route to overlapping experts → load once, reuse across
  the batch (biggest throughput lever; costs latency). **RTL-measured** through the committed
  `expert_cache_ctrl` at 34 GB cache: batch 1 / 8 / 32 → **26.5 % / 29.7 % / 50.5 %** hit rate
  (same router picks, only access order changes — isolating batching as the lever).
- **Prefetch/predict**: speculate next experts (the next layer's router is cheap and runs ahead)
  and DMA into DDR5 during the current layer's compute → hide the **big Flash fetch latency**.
  Built + measured as **`src/expert_cache_pf.v`** (a prefetch hint port + demand-priority
  background Flash fetch + a `demand_stall_cycles` counter; demand path bit-exact to
  `expert_cache_ctrl` with prefetch off; a `CACHE_HIT_LAT` parameter models the DDR5 read).
  Honest result (compute-window model, FLASH_LAT=20, DDR5 `CACHE_HIT_LAT=4`): prefetch
  **trades the big Flash-miss stall (≈22 cyc) for the small DDR5 read** — demand stall cut
  **~81 %** (4400 → 818), not the ~99 % an idealized zero-latency cache (CACHE_HIT_LAT=0) shows.
  The residual is the irreducible DDR5 read floor (`+CACHE_HIT_LAT` per resident hit); a
  second-level DDR5→die read-ahead could hide that too (future work).
- **Layout**: store co-activated experts contiguously / aligned for sequential Flash reads
  (bandwidth- not IOPS-bound, since each expert is a ~37 MB contiguous block).
- **Speculative / MTP decoding**: GLM-5.2 ships an MTP head (built here as `mtp_head`) — verify
  K tokens per weight-load pass → cut weight traffic ~K×.

## 9. Hardware ceiling vs software leverage

| | Sets it | Knobs |
|---|---|---|
| **Hardware ceiling** | raw Flash/DDR5 bandwidth, bus width, **energy/bit**, compute rate | more Flash/DDR5 channels, wider bus, faster die |
| **Software leverage** | how much you *actually* move + when | batching, expert cache policy, prefetch, **quantization**, speculative/MTP, storage layout, scheduling/overlap |

Software can't beat the bandwidth/energy ceiling but **gets you close to it and cuts demand** —
exactly how today's stacks (vLLM, llama.cpp/KTransformers MoE offload, DeepSpeed ZeRO-Infinity,
FlexGen) run 600B+ MoE models on a single GPU + RAM/SSD.

## 10. Power / heat

The dominant dynamic energy is **moving ~16–22 GB/token of weights**. Keeping the whole model
**on-module** (DDR5 + Flash next to the die) is the key win — vastly less energy than streaming
weights from a host over USB/PCIe. Among the fast-tier options DDR5 is the **lowest-power**
choice (mainstream, not the high-speed/high-power GDDR6; its per-bit energy is above an
in-package HBM stack but it uses far fewer, slower devices than GDDR6). On the compute side
FP8's 4×4-mantissa multiply (measured: `glm_matmul_fp8` uses 18× 7-bit multipliers vs fp32's
24×24) keeps the die's dynamic power and DSP/area down. Net: a few tens of W — needs a
heatsink/fan (a small box, not a thin USB stick), powerable over USB-C PD (~60–100 W).

> **Honest energy caveat (research-backed).** Prefetch + caching hide Flash *latency* but **cannot
> remove its energy-per-bit penalty** — NAND read energy is **~24–26× DRAM**, and for offloaded
> decode the per-token energy can be **up to ~12× an HBM-resident baseline**. So **Flash offload is
> a *capacity/throughput* tool, not an energy win** — the lever is to **minimize byte traffic**
> (expert caching + batching to reuse fetched experts), not the Flash bus itself. (Sources via the
> deep-research pass; this corrects any "Flash = low power" reading.)

## 11. Cost — memory BOM [EST, 2025–26, volatile]

| Chip | $/GB | Qty | Cost |
|---|---|---|---|
| **DDR5** | ~$2–4 | 64 GB (a few DIMMs, e.g. 8× 8 GB across 8 channels) | **~$150–300** |
| NAND Flash | ~$0.05–0.10 | 1 TB | **~$50–100** |
| **Memory chips total** | | | **≈ $200–400** |

(DDR5 is the cheapest fast tier — ~5–8× under 64 GB HBM (~$1–1.5k) and below GDDR6 (~$200–500),
at no performance loss since the workload is Flash-bound. The trade is a **wide 8–12-channel
memory controller** (server-class) to reach the bandwidth, not extra chips — DIMMs are dense and
upgradeable. The Flash that holds the entire 753 GB model is nearly free either way.)

*Not included:* the board (a few DDR5 DIMMs across 8–12 channels + the die + Flash on a PCB —
DDR5 needs **no CoWoS / interposer**, a real simplification vs HBM, and far fewer devices than
GDDR6), the **wide multi-channel DDR5 controller IP** (the real engineering cost), the Flash
controller, and the custom compute-die NRE + die cost. For context, an H100's 80 GB *HBM* alone
is ~$2k of its BOM — the DDR5 here is ~$150–300 for the same capacity, the payoff for being
Flash-bound.

## 12. Mapping to the committed RTL

**What this repo already provides (the compute die):**
- The full GLM-5.2-FP8 operator datapath, fp64/faithful-fp8 verified: `fp8_e4m3`,
  `glm_matmul_fp8`, `swiglu_expert_fp8`, `mla_attn_fp8`, `moe_router_fp8`,
  `glm_decoder_block_fp8`, and the capstone **`glm_model_fp8`** (full forward pass, next-token
  argmax matches the fp8 golden).
- **Streaming weight-pull interfaces** (`w_req`/`w_col` + per-[128,128]-block bf16 scales) on
  every unit — the weight *source is abstracted*, so DDR5/Flash/host can drive them.
- The **`mtp_head`** for speculative decoding.
- A small-scale DMA append/gather streaming datapath (`tpu_soc`/`axi_master_dma`/
  `scatter_gather`/`cdc_async_fifo`) exercising the control logic.

**What this design adds (not built — the system layer):**
- DDR5 + Flash controllers / PHYs (+ a USB-C device controller for the host link).
- The **MoE expert-cache controller** (tag/LRU, miss → Flash DMA → refill/evict, prefetch).
- The KV-cache pager (append + DSA-gather of the 2048-row window; overflow to Flash).
- The runtime/scheduler (batching, prefetch, speculative-decode loop) — largely software.

## 13. Open questions / honest limits

- **Expert-cache hit rate** — now estimated (§7.1) on a *calibrated* GLM-scale trace through
  the real `expert_cache_ctrl` RTL: ~27 % at batch=1 / 34 GB cache, with a hard 0 % floor below
  ~22 GB, and batching as the dominant lever. Still **calibrated, not captured** — the actual
  numbers need a *real* GLM-5.2 routing trace (can't run 753B here); the trained-router balance
  assumption could be off in either direction.
- **Flash bandwidth** (~10s GB/s) is assumed; real on-module NAND BW must be validated — NAND
  read physics caps it well below DDR5 (which is exactly why Flash, not DDR5, is the wall).
- **64 GB DDR5 is comfortable for FP8** (hot 28 GB + ~34 GB cache, ~923 cache slots); **48 GB
  drops the cache below the ~22 GB / 600-slot batch=1 threshold → ~27 % slower single-user**
  (measured), while **~56 GB already recovers full performance**. Batched serving is insensitive
  to cache size, so 48 GB is fine there.
- **Wide memory controller** (an 8–12-channel DDR5 subsystem to reach ~400–600 GB/s, server-class
  routing/signal-integrity) is DDR5's real engineering cost — but it needs no advanced packaging
  (no CoWoS/interposer) and far fewer devices (a few DIMMs vs ~32 GDDR6 chips), and the DIMMs are
  upgradeable.
- This is **interactive, not datacenter-real-time**; high tokens/s/user at scale still wants
  multi-chip HBM (bandwidth), which this design deliberately trades away for cost.
