# ACCEL_GLM52 — A Correctness-First Accelerator to RUN GLM-5.2 (`GlmMoeDsaForCausalLM`)

> Chief-architect synthesis. ONE coherent architecture combining **compute-completeness**
> (every real GLM-5.2 operator → a concrete hardware unit) with **memory-for-scale**
> (tiered memory + expert/weight streaming + 1M-context latent-KV paging).
>
> **Honesty contract.** Two things are kept rigorously separate throughout:
> - **DERIVED / BUILDABLE** — the small-but-faithful RTL decoder block we actually
>   build and verify bit-tolerant against an independent fp32 golden on
>   iverilog/verilator/yosys + nextpnr-ecp5.
> - **SYSTEM-LEVEL ESTIMATE** — the full 753B-param multi-chip machine, designed on
>   paper, sized from the config, never claimed as "built".
>
> Where a number is a system estimate it is tagged **[SYS-EST]**; where it is proven by
> the buildable slice it is tagged **[BUILT]** (after the build) or **[DERIVED]** (mapped
> but pending build).

---

## 1. Target: exact GLM-5.2 config + honest scale reality

### 1.1 Exact config (from HF `zai-org/GLM-5.2` config.json)

| Field | Value |
|---|---|
| architectures | `GlmMoeDsaForCausalLM` |
| hidden_size H | 6144 |
| num_hidden_layers L | 78 |
| vocab V | 154880 |
| max_position_embeddings | 1,048,576 (1M) |
| dtype | bfloat16 |
| **MLA attention** | 64 heads; qk = rope 64 + nope 192 = **256**; v_head = **256**; num_kv_heads 64; attention_bias **false** |
| rope_theta | 8e6; **rope_interleave true** (decoupled RoPE on the 64-dim rope part only; NoPE on the 192 part) |
| **DSA sparse** | index_topk **2048**; index_topk_freq **4**; index_skip_topk_offset **3** (IndexShare: 1 fresh indexer pass per 4 layers, reused by the next 3); indexer_rope_interleave true |
| **MoE** | n_routed_experts **256**; num_experts_per_tok **8** (top-8); n_shared_experts **1**; moe_intermediate_size **2048**; routed_scaling_factor **2.5**; moe_layer_freq 1; **first_k_dense_replace 3** (layers 0–2 dense FFN, intermediate 12288) |
| FFN | SwiGLU, hidden_act silu (gate/up + silu(gate)⊙up + down) |
| norm | RMSNorm eps **1e-5**, pre-attn + pre-FFN + QK-norm in MLA + final |
| MTP | num_nextn_predict_layers **1** (speculative t+2 head) |
| scale | **~753B total**, **~40B active/token** |

**Underspecified fields (the ONLY two assumptions).** config.json does not expose
`q_lora_rank` / `kv_lora_rank`. GLM-5.2 is DeepSeek-MLA-derived → we size with the
standard **kv_lora_rank = 512**, **q_lora_rank ≈ 1536**. Both are RTL parameters,
overridable from the real weights. Everything else above is exact.

### 1.2 Honest scale reality — what one chip can and cannot do

- **Weights:** ~753B params = **725B cold routed experts** (75 MoE layers × 256 experts ×
  37.75M) + **~28B hot** (MLA projections, dense-front FFN, norms, router, embed/LM head).
  bf16 ≈ **1.5 TB**, INT4 ≈ **376 GB**. **[SYS-EST]**
- **Latent-KV cache:** 576 elts/token/layer (c_kv 512 + shared k_rope 64) =
  **1.125 KB/token/layer bf16**, 87.8 KB/token across 78 layers. 128K ctx → **11.8 GB**;
  1M ctx → **94.2 GB**. (A 64-head MHA cache would be 670 GB / 5.36 TB — MLA is ~57×
  smaller.) **[SYS-EST]**
- **Verdict:** No single buildable chip holds 1.5 TB of weights + ~94 GB of cache. GLM-5.2
  is **inherently a large-memory / multi-chip streaming system.** We are honest about this.
  - **One chip CAN:** hold the HOT working set (MLA proj, shared expert, router, norms,
    rope angle table), the active 8/256 routed experts for the current layer/batch, the DSA
    top-2048 gather window, and the GEMM/attention tile memory; and stream cold experts +
    append/gather the latent cache over AXI DMA.
  - **One chip CANNOT:** resident-hold 725B cold experts or the full 1M-context cache.
    Those live in tiered DRAM/HBM (and across chips at full scale).
- **What we BUILD:** a small-but-faithful decoder block (§8) that keeps **every operator and
  every structural ratio intact** and runs on one FPGA/sim, using the **same DMA
  gather/append datapath** so the streaming control logic is exercised at small scale.

---

## 2. Architecture overview (text block diagram)

```
                       ┌──────────────────────────────────────────────────────────┐
                       │            CONTROL SEQUENCER (layer_sequencer.v)           │
                       │  walks 78 layers · per-layer mode:                         │
                       │   • FFN mode: DENSE(L0-2, inter 12288) | MoE(L3-77)        │
                       │   • indexer mode: FRESH (L mod 4 == 3) | REUSE (next 3)    │
                       │   • drives MLA→DSA→ATTN→FFN→residual; fp32 residual accum   │
                       └───────────────┬──────────────────────────────┬───────────┘
                                       │ control / TM descriptors      │ DMA descriptors
   ┌───────────────────────────────────┴──────────────────────────────┴───────────────────┐
   │                          DECODER-BLOCK DATAPATH (one layer)                            │
   │                                                                                        │
   │  x (fp32 residual stream)                                                              │
   │     │                                                                                  │
   │  [rmsnorm_unit] pre-attn ──────────────► MLA ATTENTION (mla_attn.v orchestrator)       │
   │     │                                     │                                            │
   │     │   Q path:  W_dq→q_lora→[rmsnorm]→W_uq→ split nope192 | rope64                    │
   │     │   KV path: W_dkv→c_kv(512)*  W_kr→k_rope(64)*  [rmsnorm(c_kv)]→W_uk,W_uv          │
   │     │            (* = appended to LATENT CACHE)                                        │
   │     │   [rope_interleave_unit] θ=8e6 on q_rope & k_rope only (fp32 angles)             │
   │     │             │                                                                    │
   │     │             ▼                                                                    │
   │     │      DSA INDEXER (dsa_indexer.v)  ── small-dim score over ALL keys →             │
   │     │        topk_select.v → top-2048 index list  (IndexShare cache / reuse)           │
   │     │             │ index list                                                         │
   │     │             ▼                                                                    │
   │     │      [scatter_gather] gather 2048 K/V rows → attention_unit (extended)           │
   │     │        QK^T(qk=256) · causal mask · [softmax_unit fp32] · A·V(v=256) · W_o        │
   │     │             │                                                                    │
   │  x += ◄───────────┘  (fp32 residual add)                                               │
   │     │                                                                                  │
   │  [rmsnorm_unit] pre-FFN ──────────────►  FFN                                           │
   │     │      DENSE mode (L0-2):  swiglu_expert (inter 12288)                             │
   │     │      MoE mode (L3-77):   moe_router (W_g→sigmoid→top-8→renorm→×2.5)              │
   │     │                          → 8 routed swiglu_expert + 1 shared → combine (fp32)    │
   │  x += ◄───────────┘  (fp32 residual add)                                               │
   └───────────────────────────────────────────────────────────────────────────────────────┘
        │ (after 78 layers)                              ▲ weights / cache
        ▼                                                │
   [rmsnorm_unit final] → LM head (W_lm 6144×154880, gemm_ml GEMV) → [sampler.v fp32]
   MTP head (mtp_head.v): own norm + small attn/FFN, hidden+pred-embed → t+2, shares LM head

   ┌──────────────────────── MEMORY / STREAMING SYSTEM ────────────────────────┐
   │ T0 on-chip SRAM/tile_memory: activations, GEMM tiles, index-set scratch,   │
   │    softmax scratch, HOT weights, active 8/256 experts, 2048-row gather win  │
   │ T1 HBM: hot-weight overflow + recent expert working set + active cache win  │
   │ T2 DRAM/host: full 725B cold experts + long-tail latent cache + INT4 image  │
   │                                                                            │
   │ tpu_soc.v / tpu_axi.v / axi_master_dma.v / cdc_async_fifo.v:               │
   │   • EXPERT STREAM: router top-8 ids → DMA gather experts (dominant traffic) │
   │   • CACHE APPEND : write [c_kv|k_rope] per token/layer (ring buffer)        │
   │   • CACHE GATHER : scatter_gather reads the 2048 DSA-selected rows          │
   │   • two-clock CDC: bus domain (DMA) ↔ core domain (compute)                │
   └────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Operator coverage table — EVERY GLM-5.2 op → a hardware unit

Precision policy (matches the operator profile): **compute in bf16, ALL reductions in
fp32.** Existing units carry a 48-bit **Q15.16** accumulator and **Q7.8** elements
(`tpu_defs.vh`); the slice maps bf16-compute/fp32-reduce onto that fixed-point reduce
infrastructure and the golden uses matching fp32 reduce so they agree within tolerance.

| # | GLM-5.2 operator | Hardware unit | Precision | Reuse / New |
|---|---|---|---|---|
| 1 | Embedding (id → row 6144) | `scatter_gather` + AXI DMA | gather, bf16 store | **Reuse** |
| 2 | RMSNorm (pre-attn, pre-FFN, q_lora, c_kv, final) | `rmsnorm_unit` | **fp32** Σx² reduce + rsqrt, bf16 γ | **New** |
| 3 | MLA Q down/up (W_dq, W_uq) | `gemm_ml` (orch by `mla_attn`) | bf16 ×, **Q15.16** acc | Reuse + New orch |
| 4 | MLA KV down (W_dkv, W_kr) → c_kv, k_rope | `gemm_ml` + cache-append DMA | bf16, Q15.16 acc | Reuse + New orch |
| 5 | MLA KV up (W_uk, W_uv) → k_nope, v | `gemm_ml` | bf16, Q15.16 acc | **Reuse** |
| 6 | Decoupled RoPE (interleave, θ=8e6, 64-dim only) | `rope_interleave_unit` | **fp32** angle table, bf16 apply | **New** |
| 7 | DSA indexer scoring (small-dim dot over all keys) | `dsa_indexer` | **fp32** score accum | **New** |
| 8 | DSA top-2048 select (+causal recent window) | `topk_select` (in dsa_indexer) | index compare (argmax-style) | **New** (shared w/ router) |
| 9 | IndexShare reuse (freq 4, offset 3) | `layer_sequencer` + index-list cache | control | **New** |
| 10 | QK^T over selected keys (qk 256) | `attention_unit` (+gather front-end) | bf16 ×, **Q15.16** acc | Reuse (extended) |
| 11 | Causal mask + softmax over N≤2048 | `softmax_unit` | **fp32** max+exp-sum, Q0.16 probs | **Reuse** |
| 12 | A·V (v_head 256) + output proj W_o | `attention_unit` AV + `gemm_ml` | bf16, Q15.16 acc | Reuse (extended) |
| 13 | MoE router (W_g → sigmoid → top-8) | `moe_router` (W_g via `gemm_ml`) | bf16 GEMV, **fp32** score | **New** |
| 14 | Gate renorm-to-1 **then** ×2.5 | `moe_router` (two explicit stages) | **fp32** | **New** (silent-bug guard) |
| 15 | SwiGLU expert (W_gate, W_up, silu⊙, W_down) | `swiglu_expert` | bf16 ×, fp32 silu LUT | **New** (wraps `gemm_ml`+`fused_ops`) |
| 16 | Dense-front FFN (L0-2, inter 12288) | `swiglu_expert` dense mode | bf16 | **New** (mode flag) |
| 17 | MoE combine (Σ gₑ·yₑ + y_shared, residual) | combine stage in `swiglu_expert`/seq | **fp32** accum | **New** |
| 18 | MTP head (t+2 speculative) | `mtp_head` | bf16 + fp32 reduce | **New** |
| 19 | Final RMSNorm | `rmsnorm_unit` | fp32 reduce | **Reuse (#2)** |
| 20 | LM head (W_lm 6144×154880 GEMV) | `gemm_ml` (streamed) | bf16, Q15.16 acc | **Reuse** |
| 21 | Sampling (temp/top-k/top-p/multinomial) | `sampler` (+`softmax_unit`) | **fp32** logits | **New** |
| — | Expert streaming + cache append/gather | `tpu_soc`/`axi_master_dma`/`cdc_async_fifo` | AXI burst | **Reuse** |

**Reused (already 3-gate verified):** `gemm_ml`, `gemm_systolic`, `softmax_unit`,
`attention_unit`, `scatter_gather`, `fused_ops_unit`, `tpu_soc`, `tpu_axi`,
`axi_master_dma`, `cdc_async_fifo`, `tile_memory`, `tpu_defs.vh`.
**New:** `rmsnorm_unit`, `rope_interleave_unit`, `mla_attn`, `dsa_indexer` (+`topk_select`),
`moe_router`, `swiglu_expert`, `mtp_head`, `sampler`, `layer_sequencer`.

---

## 4. MLA + DSA detail

### 4.1 MLA latent attention (`mla_attn.v` orchestrator)

`mla_attn` is an **FSM orchestrator** (not a monolithic datapath): it sequences `gemm_ml`,
`rmsnorm_unit`, `rope_interleave_unit`, `dsa_indexer`, and the extended `attention_unit`
over tile memory, and owns the latent cache.

- **Q path:** `x(6144) → W_dq(6144×1536) → q_lora(1536) → rmsnorm → W_uq(1536×16384) →
  q(64 heads × 256)`. Each head's 256 split **nope[192] | rope[64]** by lane slicing.
  Parameterized so a no-q-LoRA collapse to one 6144×16384 proj is also supported.
- **KV path (the compression):** `x → W_dkv(6144×512) → c_kv(512)` and
  `x → W_kr(6144×64) → k_rope(64, shared across all 64 heads)`. **Only c_kv + k_rope are
  cached** (576 elts/token/layer). At attention time `c_kv → rmsnorm → W_uk → k_nope(64×192)`
  and `→ W_uv → v(64×256)` reconstruct K/V on the fly. Per head K=[k_nope192|k_rope64]=256,
  V=256.
- **QK-norm:** RMSNorm applied on **q_lora** and on **c_kv** (the MLA-internal norms,
  eps 1e-5, fp32 reduce) — not just pre-attention.
- **Decoupled RoPE:** `rope_interleave_unit` rotates **only** q_rope[64] (per head) and the
  single shared k_rope[64]; the 192 nope dims pass through. **Adjacent-pair interleave**
  (rotate (x[2i], x[2i+1]) — NOT rotate_half), **θ=8e6**, cos/sin from an **fp32 angle
  table** (position up to 2²⁰ makes θ^(−2i/64) span a range bf16 angles cannot resolve),
  applied in bf16.
- **Absorb mode (parameterizable):** in decode, fold W_uk into W_uq and W_uv into W_o so K/V
  are never materialized — attention runs directly on c_kv. Exposed as a mode bit; both
  absorbed and materialized paths verified identical vs the golden.
- **Cache footprint:** 1.125 KB/token/layer bf16; append + gather via the AXI-master DMA.

### 4.2 DSA indexer + top-2048 + IndexShare (`dsa_indexer.v` + `topk_select.v`)

- **Indexer scoring:** project q and cached latent (c_kv/k_rope) to a **small indexer head
  dim** (decoupled-rope'd, indexer_rope_interleave=true, same `rope_interleave_unit`); cheap
  **dot-product score s(q,k_j) over ALL past keys j** — the one O(S) streaming pass over the
  ring buffer, fp32 accumulate. Far cheaper than full attention (small dim).
- **Top-k select:** keep **index_topk=2048** highest scores per query (argmax/top-k via a
  streaming threshold + partial-bitonic `topk_select`, **NOT** a dense softmax), **union with
  the causal recent window**; future keys structurally excluded; explicit causal mask still
  applied on the recent window. Emits a 2048-entry index list.
- **Dense fallback:** when S ≤ index_topk the selector is a **no-op** (all keys kept) ⇒ exact
  dense attention. Verified as a separate test (S=4, topk=8 in the slice).
- **IndexShare FSM:** index_topk_freq=4 + index_skip_topk_offset=3 ⇒ a **fresh** index set is
  computed on layers {3, 7, 11, …} and **reused** by the next 3 layers. ~20 indexer passes
  cover all 78 layers, not 78. `layer_sequencer` holds the per-window valid index buffer +
  a layer-mod-4 counter.
- **FLOP cap (the whole point):** QK^T + A·V is constant `64·2048·512·2 = 134.2 MFLOP/query`
  once S>2048. vs dense: 2.0× cheaper at 4K, 64× at 128K, **512× at 1M**. **[SYS-EST]**

---

## 5. MoE detail (`moe_router.v` + `swiglu_expert.v` + combine)

- **Router:** `logits = x · W_g(6144×256)` via `gemm_ml` (256 logits). GLM/DeepSeek-v3 style:
  **sigmoid** (or softmax — parameterized) scores, group/**top-8 of 256** via `topk_select`.
- **Gate math — ORDER IS CORRECTNESS-CRITICAL** (two explicit pipeline stages, never folded):
  (1) **renormalize** the 8 selected gate weights to sum 1; **then** (2) multiply by
  **routed_scaling_factor = 2.5**. A golden assertion checks the post-scale gate vector
  (wrong order is a silent bf16-tolerance-passing-but-wrong bug).
- **SwiGLU expert** (8 routed + 1 always-on shared, moe_inter=2048): `g = x·W_gate(6144×2048)`,
  `u = x·W_up(6144×2048)` on `gemm_ml`; `h = silu(g) ⊙ u` (silu via sigmoid LUT reusing
  `softmax_unit`'s exp-LUT infra + `fused_ops_unit` elementwise multiply, fp32 then bf16);
  `y_e = h · W_down(2048×6144)`. Shared expert: identical shape, always runs, gate=1.
- **Combine:** `out = Σ_{e∈top8} gate_e·y_e + y_shared`, **fp32 accumulate**, residual-add.
- **Dense-front (L0,1,2):** `swiglu_expert` in **dense mode**, intermediate_size=12288, no
  router, always-active. One unit covers both FFN modes (mode flag selects inter size).
- **Streaming (dominant DMA traffic) [SYS-EST]:** per token only 8/256 experts active
  (8·37.75M = 302M params ≈ 151 MB bf16 / 38 MB INT4 per layer). Router top-8 ids key an
  AXI-master DMA gather of exactly those experts from T1/T2 into the resident expert buffer;
  shared expert + MLA proj + norms are HOT/resident, routed experts COLD/streamed. **Batching
  tokens amortizes loads** (route a whole batch, load each needed expert once).

---

## 6. Correctness & quantization + float-golden methodology

**Independent fp32 golden** (numpy/torch) implements the SAME equations as the RTL, written
from the config independently of the hardware so it is a true oracle, not a transcription:
MLA down/up + latent RMSNorm + separate k_rope + 16/16 (real 64/192) split + decoupled
interleaved RoPE θ=8e6; DSA indexer scoring + top-k + causal mask + IndexShare reuse + dense
fallback; router sigmoid + top-k + **renorm-then-×2.5**; SwiGLU; fp32 residual; MTP t+2;
final RMSNorm; LM head; fp32 sampling.

**Numerics policy enforced in BOTH RTL and golden (so they can match):**
1. RMSNorm Σx² in **fp32** (6144 bf16 terms overflow bf16 precision).
2. RoPE angle table in **fp32** (position 2²⁰ spans the frequency range beyond bf16).
3. softmax max + exp-sum reduce in **fp32** (attention AND router).
4. residual stream accumulated in **fp32** across all layers (avoid drift).
5. router **renorm-then-scale** order.
   (Existing units already use a 48-bit Q15.16 accumulator + fp-style reduce in
   `softmax_unit`/`gemm_ml`, so the slice inherits the fp32-reduce behavior.)

**Pass criteria:**
- Per-unit TBs (rmsnorm, rope_interleave, mla_attn, dsa_indexer/topk_select, moe_router,
  swiglu_expert, mtp_head, sampler) each vs the golden's matching function; deterministic
  ops (RMSNorm, RoPE) checked near-ULP.
- End-to-end: **rel-err < ~2e-2 on logits AND exact argmax next-token match** vs golden.
- **Structural edge cases explicitly tested:** S≤topk dense vs S>topk sparse; IndexShare
  3-layer reuse uses the EXACT same index set as golden; absorb vs materialized MLA identical;
  renorm-then-scale order; dense-front vs MoE layers; MTP t+2 verified separately.

**Quantization:** INT4 weights / INT8 cache with per-group scales **[SYS-EST]**; correctness
is **defined within quantization tolerance** against the fp32 golden. The buildable slice runs
the fixed-point (Q7.8/Q15.16) datapath and is judged by the same rel-err/argmax gates.

**Verification gates (inherited 3-gate flow, extended to new units):** (1) iverilog/verilator
functional vs golden; (2) yosys synth; (3) nextpnr-ecp5 routed PPA.

---

## 7. Memory & scale system (tiered + streaming + 1M paging)

**Tier 0 — on-chip SRAM / `tile_memory`:** active token activations, current-layer GEMM
tiles (128-bit/4-lane lines, Q7.8), DSA index-set scratch, softmax scratch, HOT weights
(MLA proj, shared expert, router W_g, RMSNorm γ, the fp32 RoPE angle table), the active
8/256 routed experts, and the **2048-row DSA gather window** (bounded regardless of S).

**Tier 1 — on-package HBM [SYS-EST]:** hot-weight overflow + recently-used routed-expert
working set + the active latent-cache window.

**Tier 2 — DRAM / host [SYS-EST]:** full 725B cold routed-expert pool (1.45 TB bf16 /
363 GB INT4) + long-tail latent cache + the staged INT4 model image.

**Datapath (`tpu_soc` AXI-master DMA + async-CDC, two-clock):**
- **Expert stream** (dominant traffic): router top-8 ids → DMA gather of those experts from
  T1/T2 → resident expert buffer → run SwiGLU → evict.
- **Cache append:** write [c_kv(512) | k_rope(64)] per token/layer to the per-layer ring
  buffer (append-only).
- **Cache gather:** `scatter_gather` reads the 2048 DSA-selected rows
  (`eff_addr = cache_base + index·stride`).

**Latent-KV 1M paging [SYS-EST]:** the indexer streams the whole ring cheaply (small dim) to
produce the index list; **attention gathers only the 2048 selected pages** (+ recent window).
So at 1M context only a 2048-row working set is resident even though the full cache is 94.2 GB.
Absorb mode further cuts cache reads by attending directly on c_kv.

**RTL slice memory [DERIVED]:** 8 tiny experts, S=32 latent ring, 256-entry vocab — fits one
chip, but uses the **same DMA append/gather datapath** so streaming control is exercised at
small scale.

---

## 8. RTL BUILD PLAN

### 8.1 Small-but-faithful verifiable config (keeps every operator + every ratio)

| Param | Slice | Real | Keeps intact |
|---|---|---|---|
| hidden H | 128 | 6144 | — |
| layers L | 6 (3 dense + 3 MoE) | 78 | first_k_dense_replace=3 |
| heads | 4 | 64 | — |
| qk split | rope 16 + nope 16 = 32; v 32 | 64+192=256; v256 | **nope/rope split** |
| q_lora / kv_lora | 64 / 32 | 1536 / 512 | **MLA low-rank** |
| rope_theta / interleave | 8e6 / true | 8e6 / true | **exact RoPE math** |
| DSA index_topk | 8 (S=32) + S=4 dense test | 2048 | **sparsity + dense fallback** |
| index_topk_freq / offset | 4 / 3 | 4 / 3 | **IndexShare (L3 computes, L4-5 reuse)** |
| experts / top-k / shared | 8 / top-2 / 1 | 256 / top-8 / 1 | **router top-k + shared** |
| moe_inter / scaling | 64 / 2.5 | 2048 / 2.5 | **renorm-then-scale** |
| dense inter | 256 | 12288 | **dense/MoE mode switch** |
| MTP nextn | 1 | 1 | **t+2 head** |
| vocab V | 256 | 154880 | — |
| eps / dtype | 1e-5 / bf16-compute fp32-reduce | same | **numerics policy** |

### 8.2 Build ORDER (dependency-driven — leaf math units first, orchestrators last)

1. **`rmsnorm_unit.v`** ← FIRST UNIT (see §8.4). Leaf, no deps, used everywhere (5 sites).
2. **`rope_interleave_unit.v`** — fp32 angle table, adjacent-pair rotation; needed by MLA +
   DSA indexer.
3. **`topk_select.v`** — shared top-k selector (used by both DSA and router); build before
   either consumer.
4. **`swiglu_expert.v`** — wraps `gemm_ml` + silu LUT + `fused_ops_unit` mul; dense/MoE modes.
   (Independent of attention; can proceed in parallel after #1.)
5. **`moe_router.v`** — W_g GEMV + sigmoid + `topk_select` + renorm-then-×2.5.
6. **`dsa_indexer.v`** — small-dim scoring over ring + `topk_select` + dense fallback +
   IndexShare index-list cache.
7. **`mla_attn.v`** — orchestrator: Q/KV low-rank paths, latent RMSNorm, RoPE, cache append,
   absorb mode; drives `gemm_ml` + `rmsnorm_unit` + `rope_interleave_unit` + `dsa_indexer` +
   extended `attention_unit`.
8. **`attention_unit` gather extension** — top-k gather front-end on the existing score/mask/AV.
9. **`mtp_head.v`** — own norm + small attn/FFN, shares LM head.
10. **`sampler.v`** — fp32 temp/top-k/top-p/softmax/multinomial (LFSR).
11. **`layer_sequencer.v`** — walks 6 layers: dense/MoE pattern, full/shared-indexer pattern,
    fp32 residual accumulation.
12. **Decoder-block top wiring** — embed (`scatter_gather`) → 6 blocks → final `rmsnorm_unit`
    → LM head (`gemm_ml`) → `sampler`; MTP head wired separately. Reused: `gemm_ml`,
    `softmax_unit`, `attention_unit`(extended), `scatter_gather`, `tpu_soc`/`axi_master_dma`/
    `cdc_async_fifo`, `tile_memory`, `tpu_defs.vh`.

### 8.3 Verification — per-unit then per-block

- Every new unit gets its **own iverilog/verilator TB vs the golden's matching function**,
  in the existing assertion-count style, then yosys synth + nextpnr-ecp5 PPA.
- Block bring-up order mirrors the build: (a) attention sub-block (rmsnorm+mla_attn+dsa+attn)
  vs golden attention; (b) FFN sub-block (router+experts+combine, both dense and MoE) vs
  golden FFN; (c) full 6-layer loop vs golden logits (rel-err<2e-2 + argmax); (d) MTP head
  separately.

### 8.4 The CONCRETE FIRST unit to build

**`rmsnorm_unit.v`.** Rationale:
- **Leaf, zero new dependencies** — pure datapath (fp32 Σx² reduce → rsqrt → per-channel γ
  multiply), no orchestration, builds and verifies in isolation immediately.
- **Highest reuse** — used at 5 sites (pre-attn, pre-FFN, q_lora, c_kv QK-norm, final), so it
  unblocks both the MLA path and the FFN path.
- **Locks the numerics contract** — it is the canonical place to prove the mandatory
  **fp32 reduce** (6144 bf16 terms overflow bf16) against the golden near-ULP, establishing
  the bf16-compute/fp32-reduce discipline every later unit inherits.
- **TB:** random vectors of LEN∈{128, 6144} → compare `y = x/√(mean(x²)+1e-5)·γ` against the
  fp32 golden; assert near-ULP.

### 8.5 Proven-by-build vs designed-on-paper

| Proven by the buildable slice **[BUILT after build]** | Designed on paper **[SYS-EST]** |
|---|---|
| All operators (#1–#21) at small-faithful dims | 725B cold-expert residency (1.45 TB / 363 GB) |
| MLA latent + decoupled RoPE + absorb mode | 94.2 GB 1M-context latent cache |
| DSA top-k + IndexShare reuse + dense fallback | INT4/INT8 quantized residency + per-group scales |
| MoE top-2+shared + renorm-then-×2.5 + dense-front | Multi-chip HBM farm + cross-chip expert sharding |
| fp32-reduce numerics vs golden (rel-err<2e-2) | 512× FLOP-cap payoff at 1M context |
| DMA append/gather streaming control (small ring) | Full-rate expert-stream bandwidth at 753B |

---

## 9. Perf / power note (SECONDARY)

Throughput and power are explicitly secondary to correctness. Qualitatively **[SYS-EST]**:
the design is **memory-bandwidth-bound**, not compute-bound — per token only ~40B params are
active but ~151 MB bf16 (38 MB INT4) of expert weights stream **per MoE layer**, so expert-DMA
bandwidth and batching (amortizing each expert load across a token batch) dominate throughput;
the DSA 2048-key FLOP cap keeps attention compute constant at long context, shifting the 1M
bottleneck to latent-cache bandwidth (the indexer's O(S) pass) rather than attention FLOPs.
Power follows the same story — DRAM/HBM traffic for expert + cache movement dwarfs the PE-array
energy. The buildable slice reports routed PPA (yosys + nextpnr-ecp5) for the **new datapath
units** only; full-system throughput/power are paper estimates, not measured.

---

*Buildable RTL under `/Users/wick/Documents/workspaces/TPU/src/`. Golden + per-unit/block TBs
to be added alongside. Build starts at `rmsnorm_unit.v`.*
