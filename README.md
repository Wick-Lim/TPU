# TPU — a GLM-5.2-FP8 inference accelerator in Verilog

A synthesizable Verilog accelerator whose single deliverable goal is to **run one
real model well: [`zai-org/GLM-5.2-FP8`](https://huggingface.co/zai-org/GLM-5.2-FP8)** —
the published FP8 checkpoint of GLM-5.2 (`GlmMoeDsaForCausalLM`). The full GLM-5.2
operator datapath is built and verified at a small-but-faithful slice, the complete
forward pass produces correct next tokens, and the datapath is being re-targeted to
**native FP8 E4M3** to match the published checkpoint and to fit real silicon.

Underneath sits a classic **5-stage scalar TPU core** (the original *TPU v2.0*, see
[`SPEC.md`](SPEC.md)) used as the control/integration substrate; this README leads
with the GLM-5.2 accelerator, which is the active work.

The normative documents:
- **[`docs/ACCEL_GLM52.md`](docs/ACCEL_GLM52.md)** — the GLM-5.2 accelerator architecture: exact config, MLA + DSA + MoE detail, the fp64-golden methodology, the memory/streaming system, and the RTL build order.
- **[`docs/SYSTEM_SINGLE_PACKAGE.md`](docs/SYSTEM_SINGLE_PACKAGE.md)** — a single-package system design to run the real 753B GLM-5.2-FP8 (FP8 compute die + 64 GB HBM + 1 TB Flash): memory tiering, MoE expert caching/streaming, the bottleneck/perf/cost model, and how the committed RTL is the compute die.
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
(`qk_nope 192 + qk_rope 64`, `v 256`, `kv_lora 512`, `q_lora 2048`), **MoE** 256
experts top-8 + 1 shared (`moe_intermediate 2048`), dense `intermediate 12288`,
**DSA** sparse attention (`index_topk 2048`), vocab 154880, 1M context,
`rope_theta 8e6` interleaved, RMSNorm `eps 1e-5`, MTP (`num_nextn_predict_layers 1`).

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

### FP8 E4M3 datapath — IN PROGRESS

Re-targeting every **weight** matmul to native FP8 E4M3 (4×4 mantissa multiply → fp32
accumulate → per-[128,128]-block scale → bf16), keeping norms/softmax/rope/residual and
the activation×activation attention matmuls in bf16.

| Unit | What is FP8 | Verification |
|---|---|---|
| `fp8_e4m3.vh` | E4M3 decode / encode-RNE+saturate / 4×4 mantissa multiply | **exhaustive** — ALL 66069 (256 decodes + all 256×256 multiply pairs vs fp64) |
| `glm_matmul_fp8.v` | block-scaled FP8 GEMM ([128,128], dynamic act) | 224 tests, worst err/tol 0.43 |
| `swiglu_expert_fp8.v` | gate/up/down GEMMs FP8, bf16 silu tail | 1024 tests, dense + MoE |
| `mla_attn_fp8.v` | 7 weight projections FP8, bf16 attention/rope/norm/softmax/dsa | 7 tests, worst rel-err 0.39% |
| `moe_router_fp8.v` | gate GEMV FP8, bf16 sigmoid/topk/renorm | 185 tests, top-K indices exact |
| `glm_decoder_block_fp8.v` | assembles the FP8 leaf units | *building* |
| `glm_model_fp8.v` | full FP8 forward pass | *next* |

---

## Tang Nano 20K (Gowin GW2A-18) — why FP8

Measured (Yosys `synth_gowin`) fit on the GW2A-18 (budget: ~20,736 LUT4, ~15,552 FF,
~48 DSP, 46×18Kb BSRAM, + 8 MB on-board PSRAM):

- **fp32 does NOT fit** — `mla_attn` alone ≈ **396 DSP-equiv** (8× the device), because
  each fp32 multiply maps to ~4 DSP (a 24×24 mantissa multiply).
- **FP8 frees the scarce DSP** — its 4×4 mantissa multiply sips DSP and spends the
  plentiful LUT instead (the opposite resource profile). On a DSP-constrained device
  this is the enabling change; the precise decoder-block LUT fit is being measured with
  the FP8 synth.

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
make unittests   # build+run every per-unit TB, including the GLM-5.2 + FP8 units
make lint        # verilator --lint-only -Wall on the design
make synth       # yosys elaborate/synth gate (no error, no latch)
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
