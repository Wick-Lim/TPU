# MODAL_VALIDATE — running the P1.1 product gate on a GPU (Modal)

> **What this is.** `docs/PRODUCT_ROADMAP.md` P1.1 is the **#1, BLOCKING** product
> gate: the **real** `zai-org/GLM-5.2-FP8` checkpoint must produce the **real
> model's tokens** through **our** FP8 arithmetic contract. `docs/BIT_ACCURACY.md`
> proved this *on synthetic weights, with no GPU*. `tools/modal_validate.py` is a
> self-contained [Modal](https://modal.com) app that extends that proof to the
> **real 753 GB checkpoint** and the **full-model assembly** on a GPU host —
> **the user's** Modal account, **their** auth, **their** GPU cost.

The arithmetic contract itself is **already proven in this repo, no GPU**:
`tools/glm_fp8_contract.py` (vectorized) is bit-identical to the golden
`tools/glm_fp8_ref.py`, which is a bit-exact mirror of the committed RTL
(`src/glm_matmul_fp8.v` + `src/fp8_e4m3.vh`). This doc is about the part that
*needs* a GPU: the real weights and the real model graph.

---

## Prerequisites

1. **A Modal account** and the CLI authenticated once:
   ```
   pip install modal
   modal token new
   ```
2. **A HuggingFace token** with access to the gated `zai-org/GLM-5.2-FP8` repo,
   stored as a Modal secret named `huggingface-secret` exposing `HF_TOKEN`:
   ```
   modal secret create huggingface-secret HF_TOKEN=hf_xxx
   ```
3. **GPU capacity.** The reference + full-model tiers request `gpu="H200:8"`
   (8×H200 ≈ 1128 GB HBM, enough to hold the ~753 GB FP8 weights resident;
   transformers `device_map="auto"` will offload if a tier runs on less). The
   operator tier (`tier1_operator`) needs only a single `H100`.
4. **The weight cache.** A Modal Volume named `glm52-weights` (auto-created)
   caches the ~753 GB download so it is fetched **once** and reused by every run.

---

## Commands

```bash
# (optional) pre-warm the 753 GB checkpoint into the cache volume, once:
modal run tools/modal_validate.py::download_weights

# the full P1.1 gate (reference golden + operator tier + full-model tier):
modal run tools/modal_validate.py

# operator tier only (no full-model load — fastest, single GPU):
modal run tools/modal_validate.py --tier 1

# reference + full-model gate (skip the operator tier):
modal run tools/modal_validate.py --tier 2

# restrict to the first K prompts of the corpus:
modal run tools/modal_validate.py --prompts 3
```

The `@app.local_entrypoint` `main(tier, prompts)` wires the tiers together and
prints the P1.1 result line:
```
P1.1 next-token argmax match: 8/8 = 100.0%  (PASS)
```

---

## What each tier proves

| Tier | Function | GPU | Proves |
|---|---|---|---|
| Reference | `reference()` | `H200:8` | The **golden**: the UNMODIFIED engine's greedy next-token argmax (+ top-8 logit ids) over the prompt corpus. Tries **vLLM** tensor-parallel (`tensor_parallel_size=8`, `quantization="fp8"`) first; falls back to **transformers** `device_map="auto"`. |
| 1 — operator | `tier1_operator()` | `H100` | **`docs/BIT_ACCURACY.md`, synthetic → REAL.** Pulls a sample of REAL FP8 Linear weights (`*.weight` F8_E4M3 + `*.weight_scale_inv`) straight from the cached safetensors, builds realistic bf16 activations, and compares **our contract** (`glm_fp8_contract.block_fp8_gemm`) against a **reference fp32-accumulate FP8 GEMM** (the real-GPU scheme). Reports per weight: bf16-exact rate, max/RMS abs error, max rel error, and **argmax match**. This tier is **solid** — it only reads tensors, so it does not depend on the loader being able to *run* the arch. |
| 2 — full model | `tier2_fullmodel()` | `H200:8` | **The binding gate.** Loads the real model, **monkeypatches every FP8 Linear's `forward`** to route its matmul through **our contract**, and compares the patched model's next-token argmax to the unmodified `reference()` golden over the corpus. A 100% match validates the **plumbing** (layer wiring, scale orientation, bf16-tail routing, KV/RoPE/MoE) on top of the already-proven arithmetic. |

The accumulator isolation is identical to `BIT_ACCURACY.md` §A: tier 1's
reference uses the **same** per-token pow2 `a_shift` and the **same** block
dequant as our contract — the **only** difference is the accumulator (rolling
fp32 add vs our exact BFP, `ACC_FRAC=18`), so the report measures exactly the
accumulator gap on real weights, nothing else.

---

## Honest caveats

- **`GlmMoeDsa` loader support.** The arch is `GlmMoeDsaForCausalLM`
  (`docs/ACCEL_GLM52.md`). The reference + tier-2 loaders pass
  `trust_remote_code=True` so the HF repo's custom modeling file is used. **If**
  vLLM/transformers in the pinned image do not yet support this arch, the
  reference falls back (vLLM → transformers), and if neither can build the graph
  the full-model tier cannot run. **Tier 1 does not depend on this** — it reads
  tensors directly — so the *operator-on-real-weights* result still stands.
  Pin a known-good transformers/vLLM (or vendor the modeling file) if the arch
  is unsupported; this is the one external dependency the gate cannot remove.
- **Tier-2 patch coverage.** The monkeypatch finds Linears that carry both a
  `.weight` (fp8 dtype) and a `.weight_scale_inv`. If the loader fuses or names
  them differently, `patched` will be low — the function reports the count
  honestly. Per-Linear shapes that the contract cannot handle fall back to the
  original forward so the model still runs (the comparison then reflects partial
  coverage, which is reported, not hidden).
- **The 753 GB download.** First run pays a large, slow transfer (gated on your
  HF token). It is cached in the `glm52-weights` volume thereafter. Ensure your
  Modal plan allows a volume of this size.
- **Cost.** An 8×H200 reference + full-model run is **expensive** (on the order
  of single-digit to low-double-digit USD per run depending on load time and
  Modal's H200 rate, plus the one-time multi-TB-egress download). Use
  `--tier 1` (single H100) and `--prompts K` to keep iteration cheap.
- **Determinism.** Greedy decode (`temperature=0`, `max_tokens=1`) makes the
  next-token a deterministic argmax, so the gate is a hard token match — but
  vLLM and transformers can disagree on rare exact ties; `topk_overlap` is the
  softer cross-check helper for those.

---

## What is verified HERE vs what runs on Modal

**Verified here, no GPU / no modal** (`python3 test/modal_validate_test.py` →
`ALL 7 TESTS PASSED`; `python3 tools/modal_validate.py` → self-check PASS):

- The module **imports without modal** (a no-op shim stands in for
  `modal.App`/`Image`/`Volume`/`Secret`), so the pure-python **compare / argmax
  helpers** (`argmax`, `argmax_match_rate`, `topk_overlap`, `error_stats`,
  `summarize_gate`) — the non-GPU logic of the gate — are unit-tested with the
  stdlib only.
- `python3 -c "import ast; ast.parse(...)"` passes (syntax OK).
- With `modal` installed, `modal.App("glm52-fp8-validate")` **builds**: all four
  functions (`download_weights`, `reference`, `tier1_operator`,
  `tier2_fullmodel`) and the `main` local entrypoint register, and the image /
  GPU / volume / secret config is accepted by the Modal API (verified against
  `modal==1.5.1`).

**Runs on Modal (the user's account + cost):** the actual GPU tiers — the 753 GB
download, the reference forward, the real-weight operator comparison, and the
full-model monkeypatched gate. The **contract is proven in-repo**; the GPU run is
the user's to execute and pay for.
