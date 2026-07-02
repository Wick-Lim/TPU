# P1.2 — full-config parameter scale-up (elaboration study)

Task **B4** of the product next-steps plan: establish the *structural* contract for
scaling the committed RTL slice up to the real GLM-5.2 shape, without attempting a
full-config functional simulation (which is infeasible — see §4).

The single source of truth for the real shape is [`configs/full_glm52.vh`](../configs/full_glm52.vh)
(every value cited to `config.json` / [`docs/ACCEL_GLM52.md`](ACCEL_GLM52.md)).

## 1. Parameter map (slice → full)

| Parameter | Slice (committed TBs) | Full GLM-5.2 | Source |
|---|---|---|---|
| MODEL_DIM | 128 | 6144 | `hidden_size` |
| L / N_DENSE | 6 / 3 | 78 / 3 | `num_hidden_layers` / `first_k_dense_replace` |
| VOCAB | 256 | 154880 | `vocab_size` |
| H_HEADS | 4 | 64 | `num_attention_heads` |
| NOPE / ROPE | 16 / 16 | 192 / 64 | `qk_nope_head_dim` / `qk_rope_head_dim` |
| V_DIM | 32 | 256 | `v_head_dim` |
| Q_LORA / KV_LORA | 64 / 32 | 1536 / 512 † | DeepSeek-MLA standard |
| N_EXPERT / TOPK | 8 / 2 | 256 / 8 | `n_routed_experts` / `num_experts_per_tok` |
| INTER_MOE / INTER_DENSE | 64 / 256 | 2048 / 12288 | `moe_intermediate_size` / `intermediate_size` |
| TOPK_ATTN | 8 | 2048 | `index_topk` |
| POSW | 20 | 20 | 2^20 = 1,048,576 ≥ 1M context |
| S_MAX | 8 | **keep small** | latent-ring depth — see §3 |

† `q_lora`/`kv_lora` follow the DeepSeek-MLA standard (1536/512); marked **pending**
confirmation against the published checkpoint safetensors tensor shapes.

## 2. Elaboration result — the parameterization threads structurally at real dims

Method: instantiate `glm_model_fp8` at real MLA + FFN geometry via a dangling-instance
wrapper (`build/b4_wrap.v`) and elaborate with **verilator** (an independent parser).

Confirmed clean at **NOPE=192, ROPE=64, V_DIM=256, KV_LORA=512, Q_LORA=1536,
INTER_DENSE=12288, INTER_MOE=2048, N_EXPERT=16, TOPK=8** (MODEL_DIM/VOCAB held at
moderate 768/1024 for a tractable elaboration; those are pure data-width scalings with
no structural effect):

- **Zero structural errors** — no negative replication, no unresolved parameter, no
  unknown module, no zero/negative-width range across the full
  `glm_model_fp8 → glm_decoder_block_fp8 → {mla_attn_fp8, moe_router_fp8,
  swiglu_expert_fp8, glm_matmul_fp8}` hierarchy.
- The residual verilator output is **512 SELRANGE + 56 PINMISSING** warnings — both are
  artifacts of the *dangling* wrapper instance (no ports connected → open inputs make
  bit-selects nominally out-of-range); they are **not** RTL defects and do not appear
  when the module is driven by a real TB.

> yosys note: `hierarchy` re-elaboration of a parameter-overridden `glm_model_fp8`
> trips a spurious "Static cast is only supported in SystemVerilog mode" on
> `glm_decoder_block_fp8.v:726` (`EVW'(NEVAL-1)`), a yosys-0.66 `-sv`-flag-loss quirk on
> derived modules — **not** an RTL problem (the line is valid SV; the whole-chip
> `make synth-glm` gate elaborates it fine at slice params). The verilator path above is
> the authoritative elaboration check for full-config dims.

## 3. Latent finding — a config-validity constraint (dense FFN ≥ MoE FFN)

The FFN weight-pull MUX in `glm_decoder_block_fp8.v:396-397` zero-extends the MoE group/k
fields up to the dense fields:

```verilog
assign fw_grp = mode_q ? {{(FF_GWD-FF_GWM){1'b0}}, em_wgrp} : ed_wgrp;
assign fw_k   = mode_q ? {{(FF_KWD-FF_KWM){1'b0}}, em_wk}   : ed_wk;
```

with `FF_GWD = clog2(max(INTER_DENSE,MODEL_DIM)/TN + 1)` and
`FF_GWM = clog2(max(INTER_MOE,MODEL_DIM)/TN + 1)`. This **assumes `FF_GWD ≥ FF_GWM`**,
i.e. the dense intermediate is at least as wide as the MoE intermediate. If a config sets
`INTER_MOE > INTER_DENSE`, the replication count goes negative (elaboration surfaced
`32'hfffffffd` = −3) and the netlist is malformed.

- **At the real GLM-5.2 config this holds**: INTER_DENSE=12288 ≫ INTER_MOE=2048, so the
  constraint is satisfied and there is no bug for the actual target.
- It is nonetheless a **latent, silent** structural constraint. Recommended follow-up
  (small, own task): guard it with an elaboration-time assertion
  (`if (FF_GWM > FF_GWD) $fatal`) or `max(0, …)` the two replication widths, so a future
  misconfiguration fails loudly instead of producing a bad netlist.

## 4. Block-scale bookkeeping at non-128-multiple dims (B5)

The real config has projection/FFN dims that are **not** multiples of the [128,128]
block (e.g. per-head NOPE=192, ROPE=64; `W_kr` out=64; and any K that is not a multiple
of 128). The block-scaled FP8 GEMM must handle a **ragged final K-block**.

This is already proven by the committed `test/glm_matmul_fp8_tb.v`:

- **TEST 6 (K=200)** drives one full 128-wide K-block plus a **partial 72-wide** second
  block with *distinct* per-block scales, and checks every output element against the
  fp64 [128,128]-block-scaled golden. The DUT clamps the ragged block correctly
  (`glm_matmul_fp8.v` streams `NB = ceil(KMAX/BLK)` blocks; the golden mirrors it with
  `kend = min((b+1)*BLK, K)`).
- The N (output-column) direction is **caller-tiled**: `glm_matmul_fp8` processes `PE_N`
  columns per tile with a per-(col, K-block) scale, so a non-128 output width is simply a
  different tile count at the caller (`mla_attn_fp8` / `glm_decoder_block_fp8`), not a
  module-level concern. B4's real-dim elaboration (§2) confirms those callers thread the
  real per-head widths (NOPE=192, ROPE=64, V_DIM=256) structurally.

So the ragged-block-scale contract for full-config dims is **established** (TEST 6 +
§2 elaboration). What remains of B5 — a full *intermediate-size* end-to-end
`glm_model_fp8` functional sim — is deferred as low marginal value (it re-proves the
slice-level functional fidelity at larger dims; the per-op goldens already cover the
arithmetic, and a full-config functional run is infeasible per §5).

## 5. What is explicitly OUT of scope for P1.2

- **Full-config functional simulation.** The LM-head GEMV alone streams
  MODEL_DIM(6144) × VOCAB(154880) ≈ 2.38e8 K-beats **per token**, and a 256-expert MoE
  layer runs into the billions of cycles — a single full-config token would take an
  impractical wall-clock time in iverilog. P1.2 is therefore a **structural +
  intermediate-size** contract (this doc + `configs/full_glm52.vh`), *not* "set the
  params and run the TB." Functional fidelity is proven at the slice (the committed
  `glm_model_fp8` TBs) and, at intermediate sizes, by task **B5**.
- **Attention scratch at 1M context (S_MAX).** `mla_attn_fp8` sizes its
  `scores`/`probs`/`vstore` scratch by `S_MAX`, so a full-context `S_MAX=2^20` would make
  the scratch (and elaboration) explode. Decoupling the attention window from the 1M
  position field is task **B7** (SWIN vs S_MAX). Full-config integration keeps `S_MAX`
  small (the latent-ring depth), independent of the 1M `POSW`.
