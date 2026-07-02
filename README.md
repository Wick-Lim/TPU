# TPU — a GLM-5.2-FP8 inference accelerator in Verilog

A synthesizable Verilog accelerator whose single deliverable goal is to **run one
real model well: [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)** —
the published FP8 checkpoint of GLM-5.2 (`GlmMoeDsaForCausalLM`). The full GLM-5.2
operator datapath is built in **native FP8 E4M3** and verified at a small-but-faithful slice —
the complete forward pass produces the correct next token — and it is wrapped by the
single-module memory/streaming system (multi-channel DDR5 + Flash expert cache, weight loader,
boot loader, multi-clock CDC) that runs the real 753B model, with the memory-system controllers
bounded-model-checked (`make formal`).

Underneath sits a classic **5-stage scalar TPU core** (the original *TPU v2.0*, see
[`SPEC.md`](SPEC.md)) used as the control/integration substrate; this README leads
with the GLM-5.2 accelerator, which is the active work.

**Branches:** `prototype` (frozen at `fee8501`) = the **research prototype** — the full FP8
datapath + memory system + ultra-perf batching stack, bit-exact and mechanism-proven at a
small-but-faithful slice. `main` = the **product** track, taking it from verified-prototype-RTL to
a shippable accelerator that runs the real checkpoint reliably. See
[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md).

The normative documents:
- **[`docs/PRODUCT_ROADMAP.md`](docs/PRODUCT_ROADMAP.md)** — the product (not research) development direction: the #1 gate (real-checkpoint full-model fidelity), then robustness/vendor-IP/physical/software/manufacturing phases, and the FPGA-card-vs-ASIC product fork.
- **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — the GLM-5.2 accelerator architecture: exact config, MLA + DSA + MoE detail, the fp64-golden methodology, the memory/streaming system, and the RTL build order.
- **[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md)** — a single-module system design to run the real 753B GLM-5.2-FP8 (FP8 compute die + 64 GB DDR5 + 1 TB Flash, e.g. a USB-C external accelerator): memory tiering, MoE expert caching/streaming, the bottleneck/perf/cost model, and how the committed RTL is the compute die.
- **[`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md)** — the perf/power improvement roadmap targeting the Flash bottleneck, with each lever's measured result (what helped, what was a measured no-op).
- **[`docs/PPA_FP8.md`](docs/PPA_FP8.md)** — area/timing (cells + ltp) characterization of the optimized FP8 datapath + controllers, the fmax-limiting paths, and the −87.6% accumulator confirmation at scale.
- **[`docs/FORMAL.md`](docs/FORMAL.md)** — bounded model checking (yosys-smtbmc + z3) of the memory-system controllers: properties proven, bounds, and the honest coverage.
- **[`SPEC.md`](SPEC.md)** / **[`docs/ISA.md`](docs/ISA.md)** — the underlying scalar TPU core microarchitecture + ISA.

---

## The target: `zai-org/GLM-5.2-FP8`

GLM-5.2 is a 753B-param MoE model (≈40B active/token). The published checkpoint is
FP8, and its `config.json` *quantization_config* is what drives the hardware:

| Field | Value | Hardware consequence |
|---|---|---|
| `quant_method` / `fmt` | **fp8 / e4m3** | a 4-bit-exponent / 3-bit-mantissa float multiply (a 4×4 mantissa multiply) |
| `weight_block_size` | **[128, 128]** | one bf16 dequant scale per 128×128 weight block (block-scaled accumulation) |
| `activation_scheme` | **dynamic** | activations quantized to E4M3 at runtime (per-token pow2 scale, derived on-chip) |
| `modules_to_not_convert` | norms / router / embed / lm_head | those stay **bf16** — so our "bf16 tail" matches the checkpoint, it is not an approximation |

Architecture (the slice preserves every ratio): hidden 6144, 78 layers
(`first_k_dense_replace=3`), 64 heads (`head_dim=192`), **MLA** latent attention
(`qk_nope 192 + qk_rope 64`, `v 256`, `kv_lora 512`, `q_lora 1536`), **MoE** 256
experts top-8 + 1 shared (`moe_intermediate 2048`), dense `intermediate 12288`,
**DSA** sparse attention (`index_topk 2048`), vocab 154880, 1M context,
`rope_theta 8e6` interleaved, RMSNorm `eps 1e-5`, MTP (`num_nextn_predict_layers 1`).

> The real-config MLA low-rank sizes (`q_lora 1536`, `kv_lora 512`) follow the
> DeepSeek-MLA standard used throughout [`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md);
> they are RTL parameters and are to be confirmed against the checkpoint
> safetensors shapes when the full-config parameter file (`configs/full_glm52.vh`)
> lands. The committed RTL slice uses `q_lora 64 / kv_lora 32` (every ratio preserved).

---

## Status

### bf16 reference datapath — COMPLETE (the dev scaffold + golden)

Every GLM-5.2 operator, built and verified against an independent fp64 X-aware golden,
committed to `main`. The **full model forward pass runs and predicts the correct next
token**.

| Layer | Units |
|---|---|
| FP primitives | `glm_fp.vh`, `glm_fp_pipe.v` (pipelined add/mul/mac/rsqrt/exp) |
| Ops | `rmsnorm_unit`, `topk_select`, `glm_matmul(_pipe)`, `glm_act`, `rope_interleave_unit` (lean cos/sin-LUT rebuild, 10.5× smaller), `glm_softmax` |
| Attention | `dsa_indexer` (DSA IndexShare), `mla_attn` (full MLA) |
| FFN / MoE | `swiglu_expert`, `moe_router` |
| Integration | `glm_decoder_block` (one full layer, 10/10), **`glm_model` (full forward pass, 5/5, next-token argmax matches golden)** |
| Output | `sampler`, `mtp_head` (t+2 speculative) |

### FP8 E4M3 datapath — COMPLETE

Every **weight** matmul is native FP8 E4M3 (4×4 mantissa multiply → block accumulate →
per-[128,128]-block scale → bf16), with norms/softmax/rope/residual and the activation×activation
attention matmuls in bf16 (matching the checkpoint's `modules_to_not_convert`). The **full FP8
forward pass runs and predicts the correct next token**.

| Unit | What is FP8 | Verification |
|---|---|---|
| `fp8_e4m3.vh` | E4M3 decode / encode-RNE+saturate / 4×4 mantissa multiply | **exhaustive** — ALL 66069 (256 decodes + all 256×256 multiply pairs vs fp64) |
| `glm_matmul_fp8.v` | block-scaled FP8 GEMM ([128,128], dynamic act); **BFP fixed-point accumulator** (bit-exact at ACC_FRAC=18, −87.6% cells vs fp32-accumulate) | 224 tests, worst err/tol 0.43 |
| `swiglu_expert_fp8.v` | gate/up/down GEMMs FP8, bf16 silu tail | 1024 tests, dense + MoE |
| `mla_attn_fp8.v` | 7 weight projections FP8, bf16 attention/rope/norm/softmax/dsa | 7 tests, worst rel-err 0.39% |
| `moe_router_fp8.v` | gate GEMV FP8, bf16 sigmoid/topk/renorm | 185 tests, top-K indices exact |
| `glm_decoder_block_fp8.v` | one full FP8 decoder layer | 9 tests, dense + MoE |
| **`glm_model_fp8.v`** | **full FP8 forward pass** | **3 tests, next-token argmax matches the fp8 golden (4/31/20)** |
| `mtp_head_fp8.v` | FP8 multi-token-prediction (t+2) head | 6 tests |

### Single-module system (the real-753B memory/streaming hardware) — BUILT

The RTL that runs the real model from Flash through a fast tier into the FP8 die (see
[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md), [`docs/PPA_FP8.md`](docs/PPA_FP8.md),
[`docs/FORMAL.md`](docs/FORMAL.md)). 64 GB multi-channel DDR5 + 1 TB Flash, e.g. a USB-C box.

| Unit | Role | Verification |
|---|---|---|
| `expert_cache_pf.v` | DDR5 routed-expert cache: LRU + freq policy + prefetch + hit-latency | 623 tests; **BMC-proven** (hit→slot, no-dup, uniqueness, liveness) |
| `expert_predictor.v` | confidence-thresholded expert prefetch predictor | 20 tests (GLM trace) |
| `kv_cache_pager.v` | MLA latent-KV ring + DSA-gather + Flash overflow | 73 tests; **BMC-proven** (in-bounds, gather correct) |
| `ddr5_xbar.v` | N-channel banked DDR5 read fabric (~N× aggregate BW) | 3073 tests (7.93× @8ch); **BMC-proven** (no-spurious/no-overflow/tag) |
| `flash_xbar.v` | N-channel banked **Flash** read fabric (deep outstanding queue hides NAND latency) | 2049 tests (7.99× latency-hide + N× bank); **BMC-proven** |
| `weight_loader.v` | checkpoint FP8 + block-scale → matmul pull DMA | 240 tests (loader-fed == direct-fed, bit-exact) |
| `boot_loader.v` | power-up Flash→DDR5 model-load sequencer | 9240 tests; **BMC-proven** (done-gate) |
| `spec_decode_seq/_top.v` | MTP speculative-decode loop (**K>1 multi-token draft**) | 621 / 1379 / 19 tests (spec == greedy, K=1/2/3); **BMC-proven** (monotonic) |
| **`glm_fp8_soc.v`** | top: compute + cache + pager + Flash arbiter | 3 tests (token == standalone) |
| **`glm_fp8_system.v`** | production top: compute + **ddr5_xbar + weight_loader** in the datapath | 3 tests (token == standalone) |
| **`glm_fp8_system_cdc.v`** | 2-clock wrapper (host/USB ↔ compute via `cdc_async_fifo`) | 31 tests (token == standalone across async clocks) |

Verified by `make unittests` (every per-unit TB) + `make formal` (6 controllers, z3 BMC).
`make all` = `test hazard unittests lint synth formal` (the full CI surface); `make synth-glm`
(the whole-chip GLM structural gate), `make bitacc`, `make cache-study`, and `make formal-ind`
are additional targets run separately.

### Performance / power levers — measured (see [`docs/IMPROVEMENT_PLAN.md`](docs/IMPROVEMENT_PLAN.md))

The workload is **Flash-bandwidth-bound** (`tokens/s ≈ Flash_BW / [(1−h)·footprint] · K`), so the
levers target Flash, not compute. Built + measured:

| Lever | What | Measured |
|---|---|---|
| `flash_xbar.v` | parallel Flash channels + deep outstanding queue | **7.99× latency-hide + N× banking** |
| `tools/flash_layout.py` | offline expert→channel placement (kill hotspots) | **39 % → 55 % of 8× peak BW (~+40 %)** |
| `weight_decomp.v` | on-chip lossless FP8 decompress (fewer Flash bytes) | **1.34×** bit-exact |
| `spec_decode_seq.v` K>1 | multi-token speculative draft | **K=2 ≈ +23 %** (spec == greedy) |
| `clk_en_ctrl.v` | gate the ~75 %-idle die | **74 % of idle dynamic power gated** |
| `expert_prefetch_top.v` | predictor-driven prefetch | **measured NO-OP** at real cache size (honest — popular experts already resident) |

Stacking the built levers projects **~3 → ~30+ tokens/s single-user** and **~9 → ~3 J/token**
[EST] — the gains come from Flash bandwidth + fewer bytes, *not* cache cleverness. The compute
die's own optimizations (the −87.6 %-cell BFP accumulator, the fmax fixes, the BMC) improve
area/power/timing/correctness but don't move tok/s, because the die is only ~20–25 % utilized
(Flash-starved). All system tok/s / J numbers are **[EST]** (market/physics, not P&R).

**Out of scope** (vendor IP / physical): the DDR5/Flash/USB-C PHYs (TB-stubbed), the tokenizer
(software), and full-scale synthesis + place-and-route + tapeout.

---

## Tang Nano 20K (Gowin GW2A-18) — why FP8

> **This is an intermediate physical sanity-test, not the deliverable.** The Tang Nano
> physically cannot run GLM-5.2 (it fits only ~1 time-multiplexed small FP8 GEMM); it just
> proves the FP8 arithmetic maps to real silicon fabric on a $30 board. The actual target is a
> custom ASIC / large FPGA running the full 753B model with the DDR5+Flash system — the fit
> below must **not** drive design decisions (that would cage the design to one tiny GEMM).

Measured fit on the GW2A-18 (budget: ~20,736 LUT4, ~15,552 FF, ~48 DSP, 46×18Kb BSRAM,
+ 8 MB on-board PSRAM):

- **fp32 does NOT fit** — `mla_attn` alone ≈ **396 DSP-equiv** (8× the device; each fp32
  multiply maps to ~4 DSP — a 24×24 mantissa multiply).
- **FP8 frees the scarce DSP** — its 4×4 mantissa multiply sips DSP and spends the
  plentiful LUT instead (`glm_matmul_fp8` uses 18× 7-bit multipliers, ~0 DSP). The LUT cost
  is then the **fp32 accumulators**, which scale with the PE-array size. Measured (4-LUT map):

  | `glm_matmul_fp8` config | LUT4 | FF | GW2A-18 |
  |---|---|---|---|
  | PE 4×4, K=256 (default) | ~294,000 | ~90,000 | ❌ ~14× over |
  | **PE 1×1, K=128 (time-multiplexed)** | **~10,100** | **~2,900** | ✅ **fits (49 % LUT, ~0 DSP)** |

- So a **single time-multiplexed FP8 GEMM fits the Tang Nano 20K** (~half the LUTs) — enough
  to prove the FP8 datapath on real silicon. A *full decoder block* (many GEMMs + the bf16
  tail) does not fit; that needs heavy sequencing over one small GEMM or a larger FPGA.

A *real flashable bitstream* additionally needs the open-source Gowin P&R
(`nextpnr-himbaechel`/apicula) installed and the physical board; `synth_gowin` here
gives the resource-fit answer without it.

---

## Slice configuration

The RTL is built at a small-but-faithful slice that keeps every GLM-5.2 operator and
ratio: MODEL_DIM=128, 6 layers (3 dense + 3 MoE), 4 heads, MLA nope16/rope16/v32,
q_lora64/kv_lora32, 8-expert top-2 + 1 shared, INTER_MOE=64, INTER_DENSE=256, VOCAB=256,
S_MAX=8. Running the real 753B model adds the hundreds-of-GB memory/streaming system
documented in [`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md) and array scaling.

---

## Underlying scalar TPU core (substrate)

A classic **5-stage scalar pipeline** (IF→ID→EX→MEM→WB) with full data-hazard
forwarding, load-use stalls, and a variable-length busy stall that hands multi-cycle
tensor instructions to dedicated compute units operating on real tiles in an on-chip
tile memory. Includes parameterized tensor units (`gemm_systolic`, `conv2d_unit`,
`softmax_unit`, `attention_unit`), an AXI4-Lite slave wrapper (`tpu_axi.v`), and a Yosys
`synth_ecp5` PPA flow. Full detail in [`SPEC.md`](SPEC.md), [`docs/ISA.md`](docs/ISA.md),
[`docs/PPA.md`](docs/PPA.md).

---

## Toolchain

```sh
# Icarus Verilog (sim), Verilator (lint), Yosys (synth + synth_gowin fit)
brew install icarus-verilog verilator yosys
```

iverilog/vvp 13.0, verilator 5.048, yosys 0.66. Every GLM-5.2 unit is verified against
an independent fp64 / faithful-fp8 X-aware golden; on success a TB prints
`ALL N TESTS PASSED`, on any mismatch it prints the failing case and `$fatal`s.

## Build / test

```sh
make unittests   # build+run every per-unit TB, including the GLM-5.2 + FP8 + system units
make formal      # bounded model checking (yosys-smtbmc + z3) of the 6 memory-system controllers
make cache-study # GLM-trace cache hit-rate / batching / prefetch / decompress / layout measurements
make lint        # verilator --lint-only -Wall on the design
make synth       # yosys elaborate/synth gate on the scalar TPU top (no error, no latch)
make synth-glm   # yosys whole-chip structural gate on the GLM product top glm_fp8_system_cdc
make all         # test + hazard + unittests + lint + synth + synth-glm + formal (the full CI surface)
```

Per-GLM-unit canonical compile (list sources explicitly — zsh does not word-split):

```sh
mkdir -p build
# full bf16 forward-pass capstone:
iverilog -g2012 -Wall -I src -o build/glm_model_sim \
    test/glm_model_tb.v src/glm_model.v src/glm_decoder_block.v src/rmsnorm_unit.v \
    src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v \
    src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v \
    src/moe_router.v src/sampler.v src/glm_fp_pipe.v
vvp build/glm_model_sim          # -> ALL 5 TESTS PASSED (next-token argmax matches golden)

# FP8 E4M3 primitives (exhaustive):
iverilog -g2012 -Wall -I src -o build/fp8 test/fp8_e4m3_tb.v && vvp build/fp8   # ALL 66069 TESTS PASSED

# GW2A-18 (Tang Nano 20K) resource fit for an FP8 unit:
yosys -p "read_verilog -sv -I src src/glm_matmul_fp8.v src/glm_fp_pipe.v; \
          synth_gowin -top glm_matmul_fp8; stat"
```
