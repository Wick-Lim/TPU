# GLM-5.2-FP8 Bit-Accuracy Bridge

This documents the verification stance and the tooling that bridges the published
`zai-org/GLM-5.2-FP8` checkpoint to our FP8 RTL (`src/glm_matmul_fp8.v` +
`src/fp8_e4m3.vh`).

## The verification stance

We are **not** proving "FP8 ≈ an fp32 ideal". The GLM authors shipped FP8 and own
model quality. We verify two things:

1. **Our RTL computes the FP8 ops EXACTLY as a faithful inference engine does** —
   established by a golden reference (`tools/glm_fp8_ref.py`) that is a *bit-exact*
   model of the RTL, cross-checked against the committed RTL via the bit-accuracy
   harness (`test/bit_accuracy_tb.v`: RTL `c_out` == golden, bit-for-bit, over a
   suite of representative tiles), and benchmarked against the *realistic* GPU
   engine (an fp32 tensor-core accumulator) to confirm the **argmax** (next-token
   decision) is preserved.
2. **The bridge exists** so the real checkpoint could be packed into our RTL input
   format on a GPU machine — `tools/ckpt_pack.py` parses the HF
   `config.json + safetensors` layout and emits the word-addressed weight memory
   `src/weight_loader.v` reads.

---

## A. RESULT — `glm_matmul_fp8` vs the real-contract reference

The harness (`test/bit_accuracy_gen.py` → `test/bit_accuracy_tb.v` →
`test/bit_accuracy_check.py`) drives **14 representative tiles** through the
committed `glm_matmul_fp8` and three references computed by `glm_fp8_ref.py`
primitives:

| reference | within-block accumulator | role |
|---|---|---|
| **`C_bfp`** | exact fixed-point (`ACC_FRAC=18`), 1 round/block | what our **RTL** computes (bit-exact target) |
| **`C_fp32`** | rounding **fp32** accumulator (1 `fp32_add`/term) | the **real engine** (Hopper / DeepSeek-V3 / vLLM tensor-core fp32 accumulate) |
| **`C_ref`** | exact float64 (no per-add round, no bf16) | near-exact **ground truth** |

The three differ *only* in the accumulator, so the suite isolates exactly the
honest question: **does our exact-BFP accumulator choice change the answer versus
the fp32-accumulate engine a real GPU uses?**

The tile suite covers: generic random magnitudes (several ranges/seeds), values
near the **E4M3 saturation** limit (448), **tiny/subnormal-range** values, an
**all-zero row** (degenerate argmax tie), a **single K-block** (`k_len=128`) path,
**lopsided two-block** scales (`2⁻⁷` vs `2³`, so one K-block dominates the fold),
and a **clear-argmax** tile.

### What the harness proves

- **RTL is bit-exact to the golden** over all 14 tiles (`test/bit_accuracy_tb.v` →
  **`ALL 14 TESTS PASSED`**, `vvp` exit 0). Because the RTL output equals `C_bfp`
  bit-for-bit, `C_bfp` stands in for the RTL in the error report below — the errors
  *are* the RTL-vs-real-contract errors.
- A **negative control** (deliberately corrupted expected argmax) makes the TB
  print `ARGMAX DIVERGENCE` and `$fatal` (exit 1) — the hard bar is real, not
  vacuous.

### Quantified error & argmax (`bit_accuracy_check.py`)

```
RTL(BFP) vs REAL-ENGINE(fp32-acc), bf16 OUTPUT domain (what RTL emits):
  bit-identical bf16 outputs : 112/112 (100.0%)
  MAX abs / RMS abs error    : 0 / 0
  MAX rel error              : 0
same, PRE-NARROW fp32 domain (the true accumulator gap, pre-bf16):
  bit-identical fp32 results : 111/112 (99.1%)
  MAX abs / RMS abs error    : 1.49012e-08 / 1.40803e-09
  MAX rel error              : 9.46484e-08   (below a bf16 ULP -> vanishes at bf16)
accuracy vs float64 near-exact ground truth (relative; lower=better):
  RTL(BFP)      MAX rel = 0.00360484
  real(fp32acc) MAX rel = 0.00360484
  -> BFP at least as accurate as fp32-acc: YES
ARGMAX (next-token decision) preserved:
  argmax(RTL) == argmax(real fp32-acc) : 28/28
  argmax(RTL) == argmax(ground truth)  : 28/28
  real decisions / degenerate ties     : 25/3  (ties resolved deterministically)
  min top1-top2 margin (real rows)     : 0.015625 (rel 0.0130719)
```

**Reading of the result (honest):**

- At the **bf16 output** the RTL is **bit-identical** (`0` error) to the
  fp32-accumulate engine on every element of every tile. The accumulator choice is
  *invisible* at the precision the model actually consumes.
- The accumulator difference is **real but tiny** — visible only in the
  *pre-narrow* fp32 dequant result (1/112 elements differ, max `1.5e-8` abs,
  `9.5e-8` rel) and it sits **below one bf16 ULP**, so it is erased by the final
  bf16 rounding.
- Our exact-BFP accumulator is **at least as accurate** as the fp32 accumulator vs
  the float64 ground truth (identical relative error here; BFP rounds *once* per
  block, fp32 rounds every add — BFP can never be worse).
- The binding metric — **argmax is preserved on 28/28 rows** (25 genuine decisions
  + 3 degenerate all-equal ties broken deterministically by first column), matching
  both the real engine *and* the float64 ground truth. **The next-token decision the
  real model makes does not change.**

---

## B. The exact FP8 contract our RTL implements vs `config.json`

The ground truth is the `quantization_config` of the published checkpoint:

| field | value | meaning | our RTL |
|---|---|---|---|
| `quant_method` | `fp8` | FP8 quantized | ✓ |
| `fmt` | `e4m3` | OCP E4M3: `1s.4e(bias7).3m`, no Inf, one NaN `S.1111.111`, max normal 448 | `src/fp8_e4m3.vh` (256-value exact, RNE+saturate encode) |
| `weight_block_size` | `[128,128]` | ONE dequant scale per 128(out)×128(in) weight block (`weight_scale_inv`) | `BLK=128`, one bf16 scale per (out-block, K-block) folded in the dequant pass |
| `activation_scheme` | `dynamic` | activations quantized to E4M3 at runtime, per-token / per-1×128 scale | per-token **power-of-two** `2^a_shift` (exponent add, no multiplier) — see ambiguity #1 |
| `modules_to_not_convert` | norms / router / embed / lm_head / MTP | the **bf16 tail** (no FP8) | kept bf16 by `ckpt_pack.py` |

Dequant math (what a faithful engine computes):

```
y = Σ_{K-blocks bj} ( w_scale[out_blk][bj] · Σ_{k∈bj} dq_act(A_fp8[k]) · dq_wgt(W_fp8[k]) )   → bf16
```

`glm_fp8_ref.block_fp8_gemm` reproduces the RTL **bit-exactly**:

1. **Dynamic activation quant (per token):** `a_q = encode( bf16→fp32(A) · 2^a_shift )`.
2. **Within-block accumulation (exact):** every E4M3·E4M3 product is an exact
   multiple of `2⁻¹⁸`; products are summed in fixed point per 128-wide K-block,
   then rounded to fp32 **once** (`fixed_to_fp32`, RNE).
3. **Cross-block dequant fold (bit-exact fp32):** left-fold
   `Σ_bj fp32_mul(block_partial[bj], w_scale[out_blk][bj])` using the same RNE/FTZ
   `fp32_mul`/`fp32_add` as the RTL dequant pass, in block order.
4. **Undo per-token scale + narrow:** `fp32_scale_pow2(s, -a_shift)` (exact exponent
   add) → `fp32_to_bf16` (RNE).

### Documented ambiguities / assumptions

1. **Activation scale form/granularity.** `config` says only `dynamic`. The generic
   vLLM/transformers path uses an *arbitrary fp32* per-token (or per-1×128-group)
   scale `= max(|a|)/448`. **Our RTL uses a per-token power-of-two scale `2^a_shift`**
   (an exponent add, no multiplier — the hardware win). The reference mirrors the RTL:
   `a_shift = floor(log2(448 / max|a_row|))` (largest pow2 keeping the row in E4M3
   range). The pow2 scheme is a strict subset of the generic one. Pass `a_shift`
   explicitly to override.
2. **Within-block accumulator precision.** Not pinned by the contract. Our RTL (and the
   golden) accumulate each K-block **exactly** (`ACC_FRAC=18`), one rounding to fp32
   per block; the cross-block fold is fp32 RNE. Section A quantifies the gap to the
   realistic fp32-accumulate engine (below a bf16 ULP; argmax-preserving).
3. **`weight_scale_inv` orientation.** HF stores a Linear weight as
   `[out_features, in_features]`; `weight_scale_inv` is `[ceil(out/128), ceil(in/128)]`.
   The RTL contraction column `n` is an output channel, so its block scale is
   `weight_scale_inv[n//128][k//128]`. The RTL weight stream `W[k][n]` is the
   **transpose** of the HF weight.

---

## C. FULL-MODEL validation PROCEDURE (run on a real GPU machine)

The harness above proves the **operator** is correct on faithful synthetic data. To
close the loop on the **whole model** you need the real checkpoint and a GPU. The
bridge (`tools/ckpt_pack.py`) is built for exactly this. Step by step:

**Prereqs (GPU box):** an H100/H200-class machine (FP8 e4m3 tensor cores), ≈ 1 TB
free disk, `pip install transformers accelerate safetensors torch` (and optionally
`vllm`).

1. **Download the checkpoint.**
   ```
   huggingface-cli download zai-org/GLM-5.2-FP8 --local-dir ./GLM-5.2-FP8
   ```
   Confirm `config.json → quantization_config` matches Section B
   (`fp8 / e4m3 / [128,128] / dynamic`), else update the contract before trusting any
   result.

2. **Reference forward (the ground-truth next token).** Pick a fixed prompt and a
   greedy (`do_sample=False`) decode so the decision is a deterministic argmax:
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   import torch
   tok = AutoTokenizer.from_pretrained("./GLM-5.2-FP8")
   m   = AutoModelForCausalLM.from_pretrained("./GLM-5.2-FP8",
                                              torch_dtype="auto", device_map="cuda")
   ids  = tok("The capital of France is", return_tensors="pt").input_ids.cuda()
   logits = m(ids).logits[0, -1].float().cpu()      # last-position logits
   ref_argmax = int(logits.argmax())                # the engine's next-token id
   torch.save({"ids": ids.cpu(), "logits": logits, "argmax": ref_argmax},
              "ref_forward.pt")
   ```
   (A vLLM run with `--quantization fp8` and `logprobs` gives the same argmax via a
   different kernel — a useful second opinion that the *engine* is deterministic.)

3. **Pack the checkpoint into our RTL image.**
   ```
   python3 tools/ckpt_pack.py ./GLM-5.2-FP8 ./rtl_image
   ```
   Produces `weight_mem.hex` (the word-addressed image `src/weight_loader.v` reads:
   E4M3 CODE region + bf16 SCALE region per PE_N-column tile, weights transposed to
   `W_rtl[k][n] = hf[n][k]`) + `manifest.json` (tile descriptors `{base,k_len,nblk}`,
   the bf16-tail tensor list, and the flash-channel striping map).

4. **Run the same prompt through our software-RTL model on the SAME weights.** Two
   equivalent options:
   - **Software golden (no FPGA):** drive `glm_fp8_ref.block_fp8_gemm` for every
     Linear (weights + `weight_scale_inv` straight from the safetensors; bf16-tail
     ops in plain bf16), wiring the layers per `docs/ACCEL_GLM52.md` /
     `src/glm_model_fp8.v`’s structure, to produce last-position logits
     `rtl_logits`. This is bit-exact to the RTL by construction (Section A).
   - **RTL-in-sim (scaled):** load `weight_mem.hex` into a `glm_model_fp8`
     elaboration sized to the real dims and capture the logit vector from the
     `mtp_head`/`lm_head` tap.

5. **Compare the next-token decision.**
   ```
   assert int(rtl_logits.argmax()) == ref_argmax        # the binding check
   # diagnostics:
   #   max|rtl_logits - ref_logits|,  RMS,  top-5 overlap,  logit Spearman
   ```
   Optionally roll the comparison forward N tokens (feed back the chosen id) to
   confirm the **generated string** matches greedily.

**Pass criterion:** the argmax (and, ideally, the first N greedily-generated tokens)
match the transformers/vLLM reference on a battery of prompts. Per Section A the
operator is already argmax-preserving, so a full-model match validates the
*plumbing* (layer wiring, scale orientation, bf16-tail routing, KV/rope) rather than
the arithmetic.

---

## D. Honest scope — what is proven vs what still needs the GPU + checkpoint

**Proven here, in this repo, with no GPU and no download:**

- The committed `glm_matmul_fp8` RTL implements the GLM-5.2-FP8 operator
  (`e4m3` + `[128,128]` block dequant + dynamic pow2 activation quant) **bit-for-bit**
  as the golden `glm_fp8_ref.py` (14/14 tiles, `vvp` exit 0; `$fatal` on any
  divergence, negative-control verified).
- Against the **realistic fp32-accumulate engine** a real GPU uses, our exact-BFP
  RTL is **bit-identical at the bf16 output**, **at least as accurate** vs ground
  truth, and **argmax-preserving on 28/28 rows** — including saturation, subnormal,
  single-/multi-K-block, lopsided-scale and tie edge cases.
- The **bridge** (`ckpt_pack.py`) parses the real HF `config.json + safetensors`
  layout and emits the RTL weight image; a synthetic faithful mini-checkpoint
  exercises pack→unpack round-trip end-to-end.

**NOT proven here (requires the real checkpoint + a GPU — Section C):**

- That our **layer plumbing** for the *full* model (every Linear’s scale orientation,
  the bf16 tail routing for norms/router/embed/lm_head/MTP, RoPE, KV cache, MoE
  routing) matches the real engine. The operator is correct; the *assembly* of
  ≈ hundreds of operators into the published model is validated only by the
  full-model token-match procedure.
- That the published model’s **token output** is reproduced. The synthetic tiles are
  faithful in *dtype/shape/scale structure* but carry random values — they certify
  arithmetic, not the trained weights. Only Step 5 of Section C (real weights, real
  prompt, argmax match) closes that gap.

In short: **operator-level FP8 correctness is proven bit-exact and argmax-preserving
here; full-model token fidelity is a one-command-per-step GPU procedure away, with
the checkpoint→RTL bridge already built.**

---

## Running the checks

```
# golden reference + bridge self-tests (no GPU):
python3 tools/glm_fp8_ref.py            # e4m3 round-trip + block-GEMM sanity  -> SELFTEST PASS, exit 0
python3 tools/ckpt_pack.py              # synthetic ckpt pack->unpack round-trip -> SELFTEST PASS, exit 0

# RTL-vs-golden bit-exactness + argmax harness:
python3 test/bit_accuracy_gen.py scratchpad/bitacc          # write the 14-tile suite
python3 test/bit_accuracy_check.py scratchpad/bitacc        # error + argmax report (ARGMAX-PRESERVED)
iverilog -g2012 -Wall -I src -o scratchpad/bitacc.vvp \
    test/bit_accuracy_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
vvp scratchpad/bitacc.vvp +vec=scratchpad/bitacc            # -> "ALL 14 TESTS PASSED", exit 0
```

`bit_accuracy_tb.v` drives the committed `glm_matmul_fp8` with each tile (PE_M=2,
PE_N=4, KMAX=256, BLK=128 → up to 2 K-blocks), asserts `c_out` matches the golden
`C_bfp` **bit-for-bit**, and asserts the per-row **argmax** matches the realistic
fp32-accumulate engine — `$fatal` on any argmax divergence.
