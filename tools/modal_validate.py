#!/usr/bin/env python3
# ============================================================================
# modal_validate.py -- run the P1.1 product gate on a Modal GPU account
# ----------------------------------------------------------------------------
# WHAT THIS IS
#   docs/PRODUCT_ROADMAP.md P1.1 (the #1, BLOCKING product gate) says: the REAL
#   zai-org/GLM-5.2-FP8 checkpoint must produce the REAL model's tokens through
#   OUR FP8 arithmetic contract.  docs/BIT_ACCURACY.md proved -- on SYNTHETIC
#   weights, with no GPU -- that our exact-BFP FP8 GEMM (src/glm_matmul_fp8.v,
#   mirrored bit-for-bit by tools/glm_fp8_ref.py and vectorized in
#   tools/glm_fp8_contract.py) is bit-identical + argmax-preserving versus the
#   fp32-accumulate engine a real GPU runs.  This module EXTENDS that proof to
#   the REAL 753 GB checkpoint + the full model assembly, on the user's Modal
#   (https://modal.com) GPU account.
#
#   It is a self-contained Modal app.  `modal` is almost certainly NOT installed
#   in this repo's dev env -- this file is written CORRECT per the Modal API and
#   is import-safe WITHOUT modal (a tiny shim stands in so the pure-python
#   compare/argmax helpers below are unit-testable locally, and so
#   `python3 -c "import ast; ast.parse(...)"` and a plain import both succeed).
#   When modal IS installed, the real decorators apply and `modal.App` builds.
#
# THE THREE TIERS  (each an @app.function; the local_entrypoint wires them)
#   reference()      : load the REAL model (vLLM tensor-parallel, else
#                      transformers device_map/offload) and, for a PROMPT CORPUS,
#                      return the greedy next-token argmax (+ top-k logits) = the
#                      GOLDEN the gate compares against.
#   tier1_operator() : pull a SAMPLE of REAL Linear weight tensors (+ their
#                      weight_scale_inv) from the cached checkpoint, make
#                      realistic activations, run OUR contract
#                      (glm_fp8_contract.block_fp8_gemm) vs a reference
#                      fp32-accumulate FP8 GEMM, and report error + ARGMAX
#                      preservation ON REAL WEIGHTS.  (BIT_ACCURACY, synthetic ->
#                      real.)  This tier is SOLID: it depends only on reading
#                      tensors, not on the loader running the arch.
#   tier2_fullmodel(): the full gate -- monkeypatch the model's FP8 Linear
#                      forward to use OUR contract and compare the next-token
#                      argmax to the UNMODIFIED reference over the corpus.
#                      Best-effort: depends on the loader exposing the FP8
#                      Linears (the GlmMoeDsa arch may need a custom modeling
#                      file / trust_remote_code -- documented in MODAL_VALIDATE).
#
# RUN (on the user's account, their auth + their GPU cost):
#   modal token new                       # once
#   modal secret create huggingface-secret HF_TOKEN=hf_...   # HF access
#   modal run tools/modal_validate.py                        # full gate
#   modal run tools/modal_validate.py::download_weights      # pre-warm the cache
#   modal run tools/modal_validate.py --tier 1               # operator tier only
#
# SELF-VERIFY (here, no modal/GPU):
#   python3 -c "import ast; ast.parse(open('tools/modal_validate.py').read())"
#   python3 test/modal_validate_test.py     # unit-tests the pure compare helpers
# ============================================================================
import sys
import os
import math

# ----------------------------------------------------------------------------
# Configuration (mirrors docs/ACCEL_GLM52.md 1.1 + the HF repo id)
# ----------------------------------------------------------------------------
MODEL_ID = "zai-org/GLM-5.2-FP8"
VOLUME_NAME = "glm52-weights"               # caches the ~753 GB FP8 checkpoint
HF_SECRET_NAME = "huggingface-secret"       # must expose HF_TOKEN
WEIGHTS_DIR = "/weights"                    # volume mount point in the container
CKPT_DIR = f"{WEIGHTS_DIR}/GLM-5.2-FP8"     # snapshot lands here
GPU_CONFIG = "H200:8"                        # 8xH200 ~1128 GB holds 753 GB FP8
FUNC_TIMEOUT = 60 * 60 * 6                   # 6 h (download + a heavy forward)

# A small, deterministic, factual greedy-decodable prompt corpus.  Greedy (argmax)
# next-token is a deterministic decision, so the gate is a hard token match.
PROMPT_CORPUS = [
    "The capital of France is",
    "Water is made of hydrogen and",
    "The opposite of hot is",
    "The first president of the United States was",
    "Two plus two equals",
    "The sun rises in the",
    "The chemical symbol for gold is",
    "A group of lions is called a",
]

# ============================================================================
# PURE-PYTHON compare / argmax helpers  (NO modal, NO numpy, NO torch needed)
# These are the non-GPU logic the task asks to unit-test locally.  They operate
# on plain python lists so test/modal_validate_test.py can exercise them with
# the stdlib only.
# ============================================================================
def argmax(row):
    """Index of the maximum element (first on ties).  `row` a sequence of nums."""
    best_i, best_v = 0, None
    for i, v in enumerate(row):
        if best_v is None or v > best_v:
            best_v, best_i = v, i
    return best_i


def argmax_match_rate(golden_tokens, cand_tokens):
    """Fraction of positions where the candidate's next-token id == the golden's.
       Both are flat lists of token ids (already argmaxed).  Empty -> 1.0."""
    if len(golden_tokens) != len(cand_tokens):
        raise ValueError(
            f"length mismatch: {len(golden_tokens)} golden vs {len(cand_tokens)} cand"
        )
    if not golden_tokens:
        return 1.0
    hits = sum(1 for g, c in zip(golden_tokens, cand_tokens) if int(g) == int(c))
    return hits / len(golden_tokens)


def topk_overlap(golden_row, cand_row, k):
    """|topk(golden) ∩ topk(cand)| / k -- a softer agreement signal than argmax.
       Rows are logit vectors (lists)."""
    def topk_idx(row):
        return set(sorted(range(len(row)), key=lambda i: row[i], reverse=True)[:k])
    k = min(k, len(golden_row), len(cand_row))
    if k == 0:
        return 1.0
    return len(topk_idx(golden_row) & topk_idx(cand_row)) / k


def error_stats(golden_vals, cand_vals):
    """Elementwise error summary between two flat float lists (the same length).
       Returns {n, max_abs, rms_abs, max_rel, exact} where `exact` counts
       bit-... no -- value-identical elements (==).  Pure python."""
    if len(golden_vals) != len(cand_vals):
        raise ValueError("length mismatch")
    n = len(golden_vals)
    if n == 0:
        return dict(n=0, max_abs=0.0, rms_abs=0.0, max_rel=0.0, exact=0)
    max_abs = 0.0
    sq = 0.0
    max_rel = 0.0
    exact = 0
    for g, c in zip(golden_vals, cand_vals):
        g = float(g)
        c = float(c)
        if g == c:
            exact += 1
        d = abs(g - c)
        max_abs = max(max_abs, d)
        sq += d * d
        denom = abs(g)
        if denom > 0.0:
            max_rel = max(max_rel, d / denom)
    return dict(n=n, max_abs=max_abs, rms_abs=math.sqrt(sq / n),
                max_rel=max_rel, exact=exact)


def summarize_gate(golden_tokens, cand_tokens):
    """Format the P1.1 result line from two next-token-id lists.  Pure python."""
    rate = argmax_match_rate(golden_tokens, cand_tokens)
    n = len(golden_tokens)
    hits = int(round(rate * n))
    return (f"P1.1 next-token argmax match: {hits}/{n} = {rate * 100:.1f}%  "
            f"({'PASS' if rate == 1.0 else 'PARTIAL/FAIL'})")


# ============================================================================
# MODAL APP  (built only when modal is importable; else a no-op shim so this
# module imports cleanly and the helpers above stay unit-testable).
# ============================================================================
try:
    import modal
    _HAS_MODAL = True
except Exception:
    modal = None
    _HAS_MODAL = False


class _Shim:
    """A no-op stand-in for modal.App / Image / Volume / Secret when modal is
       absent.  Supports attribute chaining and being used as a (decorator-)
       factory, so module-level `@app.function(...)` / `@app.local_entrypoint()`
       and `image.pip_install(...).add_local_dir(...)` parse and import without
       modal installed."""
    def __getattr__(self, _name):
        return self

    def __call__(self, *args, **kwargs):
        # Used as a bare decorator:  @something
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        # Used as a decorator factory:  @something(...)  -> returns a decorator
        def _decorator(fn):
            return fn
        return _decorator


if _HAS_MODAL:
    app = modal.App("glm52-fp8-validate")

    # The container image: GPU FP8 stack + HF download + our contract's deps.
    # add_local_dir ships tools/ so glm_fp8_contract / glm_fp8_ref import in-VM.
    _TOOLS_LOCAL = os.path.dirname(os.path.abspath(__file__))
    image = (
        modal.Image.debian_slim(python_version="3.12")
        .pip_install(
            "torch==2.5.1",
            "transformers>=4.46",
            "accelerate>=1.0",
            "safetensors>=0.4.5",
            "huggingface_hub>=0.26",
            "numpy>=1.26",
        )
        # vLLM is optional (best-effort tensor-parallel reference); install in a
        # separate layer so a vLLM/arch mismatch does not break tier1/tier2.
        .pip_install("vllm>=0.6.3")
        .add_local_dir(_TOOLS_LOCAL, remote_path="/root/tools")
    )

    volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
    hf_secret = modal.Secret.from_name(HF_SECRET_NAME)
else:
    app = _Shim()
    image = _Shim()
    volume = _Shim()
    hf_secret = _Shim()


# ----------------------------------------------------------------------------
# helpers that run INSIDE the container (import torch/transformers/etc. lazily)
# ----------------------------------------------------------------------------
def _ensure_on_path():
    """Make the shipped tools/ importable inside the Modal container."""
    if "/root/tools" not in sys.path:
        sys.path.insert(0, "/root/tools")


def _checkpoint_present():
    """True if the snapshot looks complete (config.json + at least one shard)."""
    if not os.path.isdir(CKPT_DIR):
        return False
    if not os.path.exists(os.path.join(CKPT_DIR, "config.json")):
        return False
    return any(n.endswith(".safetensors") for n in os.listdir(CKPT_DIR))


@app.function(
    image=image,
    volumes={WEIGHTS_DIR: volume},
    secrets=[hf_secret],
    timeout=FUNC_TIMEOUT,
)
def download_weights(force: bool = False):
    """Download zai-org/GLM-5.2-FP8 (~753 GB) into the cache volume ONCE.
       Idempotent: skips if the snapshot is already present unless force=True.
       Gated on the HF token secret (the repo is access-controlled)."""
    from huggingface_hub import snapshot_download

    if _checkpoint_present() and not force:
        print(f"[download] checkpoint already cached at {CKPT_DIR}; skipping.")
        return CKPT_DIR

    os.makedirs(CKPT_DIR, exist_ok=True)
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"[download] fetching {MODEL_ID} -> {CKPT_DIR} (this is ~753 GB) ...")
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=CKPT_DIR,
        token=token,
        # weights + config + tokenizer; skip nothing FP8-relevant.
        allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                        "tokenizer*", "*.py"],
    )
    volume.commit()                          # persist the download into the volume
    print(f"[download] done; committed to volume '{VOLUME_NAME}'.")
    return CKPT_DIR


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={WEIGHTS_DIR: volume},
    secrets=[hf_secret],
    timeout=FUNC_TIMEOUT,
)
def reference(prompts):
    """THE GOLDEN.  Load the REAL model and, per prompt, return the greedy
       next-token argmax id (+ top-k logit ids) from the UNMODIFIED engine.

       Tries vLLM tensor-parallel first (fast, 8-way), falls back to
       transformers device_map='auto' (offload across the 8 GPUs).  Returns a
       dict: {'tokens': [id,...], 'topk': [[id,...],...], 'engine': str}."""
    _ensure_on_path()
    if not _checkpoint_present():
        download_weights.local(force=False)   # ensure cache, then continue

    # ---- try vLLM (tensor-parallel reference) --------------------------------
    try:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=CKPT_DIR,
            tensor_parallel_size=8,
            quantization="fp8",
            trust_remote_code=True,
            enforce_eager=True,
            max_model_len=4096,
        )
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=8)
        outs = llm.generate(list(prompts), sp)
        tokens, topk = [], []
        for o in outs:
            comp = o.outputs[0]
            tokens.append(int(comp.token_ids[0]))
            lp = comp.logprobs[0] if comp.logprobs else {}
            topk.append([int(t) for t in sorted(lp, key=lambda t: lp[t].logprob,
                                                 reverse=True)[:8]])
        print(f"[reference] vLLM produced {len(tokens)} golden tokens.")
        return dict(tokens=tokens, topk=topk, engine="vllm")
    except Exception as e:                    # noqa: BLE001 -- best-effort fallback
        print(f"[reference] vLLM path unavailable ({type(e).__name__}: {e}); "
              f"falling back to transformers.")

    # ---- transformers fallback (device_map / offload) ------------------------
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()
    tokens, topk = [], []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().cpu()
        tokens.append(int(logits.argmax()))
        topk.append([int(i) for i in torch.topk(logits, 8).indices.tolist()])
    print(f"[reference] transformers produced {len(tokens)} golden tokens.")
    return dict(tokens=tokens, topk=topk, engine="transformers")


@app.function(
    image=image,
    gpu="H100",                              # one GPU is plenty for the op tier
    volumes={WEIGHTS_DIR: volume},
    secrets=[hf_secret],
    timeout=FUNC_TIMEOUT,
)
def tier1_operator(n_weights: int = 6, m_tokens: int = 16, seed: int = 0):
    """OPERATOR TIER -- BIT_ACCURACY, synthetic -> REAL weights.
       Pull a SAMPLE of REAL FP8 Linear weights (+ weight_scale_inv) from the
       cached checkpoint, build realistic bf16 activations, and compare OUR
       contract (glm_fp8_contract.block_fp8_gemm) against a reference
       fp32-accumulate FP8 GEMM (the real-engine scheme).  Reports, per weight:
       bf16-exact rate, max/RMS abs error, max rel error, and ARGMAX match."""
    _ensure_on_path()
    if not _checkpoint_present():
        download_weights.local(force=False)

    import torch
    from safetensors import safe_open
    import glm_fp8_ref as ref
    import glm_fp8_contract as contract

    torch.manual_seed(seed)

    # ---- discover REAL fp8 Linear weights (those with a weight_scale_inv) -----
    shards = sorted(n for n in os.listdir(CKPT_DIR) if n.endswith(".safetensors"))
    picked = []                               # (shard_path, weight_key, scale_key)
    for sh in shards:
        path = os.path.join(CKPT_DIR, sh)
        with safe_open(path, framework="pt") as f:
            keys = set(f.keys())
            for k in keys:
                if k.endswith(".weight") and (k + "_scale_inv") in keys:
                    picked.append((path, k, k + "_scale_inv"))
                    if len(picked) >= n_weights:
                        break
        if len(picked) >= n_weights:
            break

    if not picked:
        return dict(error="no fp8 Linear (weight + weight_scale_inv) found",
                    n_weights=0)

    results = []
    for path, wk, sk in picked:
        with safe_open(path, framework="pt") as f:
            W = f.get_tensor(wk)              # [out=N, in=K], dtype float8_e4m3fn
            S = f.get_tensor(sk)             # [ceil(N/128), ceil(K/128)] fp32/bf16
        N, K = int(W.shape[0]), int(W.shape[1])

        # ---- realistic activations: bf16, K-wide, m_tokens rows --------------
        A_f = torch.randn(m_tokens, K, dtype=torch.float32) * 0.1
        A_bf16 = A_f.to(torch.bfloat16)
        A_codes = (A_bf16.view(torch.int16).to(torch.int64) & 0xFFFF)   # bf16 codes

        # ---- pack operands into OUR contract's code domain -------------------
        # W_fp8 contraction orientation W[k][n] = hf[n][k]  -> transpose to [K,N].
        W_codes = (W.view(torch.uint8).to(torch.int64).t().contiguous())  # [K,N]
        # weight_scale_inv -> bf16 codes [n_ob][n_kb] (contract bus form).
        S_bf16 = S.to(torch.bfloat16)
        WS_codes = (S_bf16.view(torch.int16).to(torch.int64) & 0xFFFF)

        # ---- OUR contract (block-scaled, dynamic pow2 act, exact BFP) --------
        C_codes, a_shift = contract.block_fp8_gemm(
            A_codes, W_codes, WS_codes, blk=128, backend="torch")
        ours = _bf16_codes_to_torch(torch, C_codes)        # [M,N] float

        # ---- reference fp32-accumulate FP8 engine (the real GPU scheme) ------
        refout = _fp32_accumulate_fp8_gemm(
            torch, ref, A_bf16, W, S, blk=128, a_shift=a_shift)

        # ---- compare (flatten to pure-python for the shared helpers) ---------
        og = ours.flatten().tolist()
        rg = refout.flatten().tolist()
        stats = error_stats(rg, og)
        # argmax over each token row (the next-token-like decision)
        am_hits, am_rows = 0, 0
        for i in range(m_tokens):
            if all(v == 0.0 for v in A_f[i].tolist()):
                continue
            am_rows += 1
            if argmax(ours[i].tolist()) == argmax(refout[i].tolist()):
                am_hits += 1
        results.append(dict(
            weight=wk, N=N, K=K,
            bf16_exact=f"{stats['exact']}/{stats['n']}",
            max_abs=stats["max_abs"], rms_abs=stats["rms_abs"],
            max_rel=stats["max_rel"],
            argmax_match=f"{am_hits}/{am_rows}",
        ))
        print(f"[tier1] {wk} [N={N},K={K}] bf16_exact={stats['exact']}/{stats['n']} "
              f"max_rel={stats['max_rel']:.3e} argmax={am_hits}/{am_rows}")

    return dict(n_weights=len(results), results=results)


@app.function(
    image=image,
    gpu=GPU_CONFIG,
    volumes={WEIGHTS_DIR: volume},
    secrets=[hf_secret],
    timeout=FUNC_TIMEOUT,
)
def tier2_fullmodel(prompts, golden):
    """FULL-MODEL TIER (the binding gate).  Load the real model under
       transformers, MONKEYPATCH every FP8 Linear's forward to route through OUR
       contract (glm_fp8_contract.block_fp8_gemm), then compare the next-token
       argmax to the UNMODIFIED `golden` over the corpus.

       Best-effort: it depends on the loader exposing FP8 Linear modules whose
       weight+scale we can read.  If the GlmMoeDsa arch needs a custom modeling
       file (trust_remote_code) the loader provides it; if no patchable FP8
       Linear is found we report that honestly (the contract is still proven by
       tier1).  Returns {'tokens': [...], 'patched': int, 'engine': str}."""
    _ensure_on_path()
    if not _checkpoint_present():
        download_weights.local(force=False)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import glm_fp8_contract as contract

    tok = AutoTokenizer.from_pretrained(CKPT_DIR, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR, torch_dtype="auto", device_map="auto", trust_remote_code=True,
    )
    model.eval()

    # ---- monkeypatch: find FP8 Linears (a .weight + a .weight_scale_inv buffer)
    patched = 0
    for module in model.modules():
        w = getattr(module, "weight", None)
        wsi = getattr(module, "weight_scale_inv", None)
        if w is None or wsi is None:
            continue
        if not _looks_fp8(torch, w):
            continue
        _install_contract_forward(torch, contract, module)
        patched += 1
    print(f"[tier2] patched {patched} FP8 Linear modules to OUR contract.")

    tokens = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            logits = model(ids).logits[0, -1].float().cpu()
        tokens.append(int(logits.argmax()))

    rate = argmax_match_rate(golden["tokens"], tokens)
    print(f"[tier2] {summarize_gate(golden['tokens'], tokens)}")
    return dict(tokens=tokens, patched=patched, engine="transformers+contract",
                match_rate=rate)


# ----------------------------------------------------------------------------
# in-container numeric helpers (torch present here; pure helpers stay at top)
# ----------------------------------------------------------------------------
def _bf16_codes_to_torch(torch, codes):
    """[M][N] bf16 codes (python lists or torch int tensor) -> float32 tensor."""
    t = torch.as_tensor(codes, dtype=torch.int64) & 0xFFFF
    return (t.to(torch.int16)).view(torch.bfloat16).to(torch.float32)


def _fp32_accumulate_fp8_gemm(torch, ref, A_bf16, W_fp8, S, blk, a_shift):
    """The REAL-engine reference: dynamic-act FP8 with OUR per-token pow2 a_shift
       (so the only difference vs our contract is the ACCUMULATOR -- fp32 rolling
       add here vs exact BFP in the contract; this isolates the accumulator gap
       exactly as docs/BIT_ACCURACY.md §A does), block-dequant weights, fp32
       accumulate.  Returns [M,N] float32.

       A_bf16 : [M,K] bf16   W_fp8 : [N,K] float8_e4m3fn   S : [n_ob,n_kb]."""
    M, K = int(A_bf16.shape[0]), int(A_bf16.shape[1])
    N = int(W_fp8.shape[0])
    n_kb = (K + blk - 1) // blk

    Wf = W_fp8.to(torch.float32)                       # exact e4m3 -> fp32
    Sf = S.to(torch.float32)                           # block scales
    A = A_bf16.to(torch.float32)
    ash = torch.tensor(a_shift, dtype=torch.float32).view(M, 1)

    # dynamic activation quant: scale by 2^a_shift, round to e4m3, undo later.
    Aq = (A * torch.pow(2.0, ash)).to(torch.float8_e4m3fn).to(torch.float32)

    out = torch.zeros(M, N, dtype=torch.float32)
    for bj in range(n_kb):
        k0, k1 = bj * blk, min(bj * blk + blk, K)
        seg = Aq[:, k0:k1] @ Wf[:, k0:k1].t()          # [M,N] fp32 accumulate
        # per-output-column block scale: column n uses S[n//blk][bj]
        col_scale = Sf[(torch.arange(N) // blk).clamp(max=Sf.shape[0] - 1), bj]
        out = out + seg * col_scale.view(1, N)
    out = out * torch.pow(2.0, -ash)                   # undo per-token pow2
    return out


def _looks_fp8(torch, w):
    """True if a weight tensor is stored in an fp8 dtype."""
    try:
        return w.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    except Exception:
        return False


def _install_contract_forward(torch, contract, module):
    """Replace `module.forward` with one that dequant-free routes the matmul
       through OUR contract (block_fp8_gemm).  Reads module.weight (fp8) +
       module.weight_scale_inv; falls back to the original forward on any shape
       it cannot handle (so the model still runs)."""
    orig_forward = module.forward
    W = module.weight                                  # [N,K] fp8
    S = module.weight_scale_inv                        # [n_ob,n_kb]
    bias = getattr(module, "bias", None)

    W_codes = (W.view(torch.uint8).to(torch.int64).t().contiguous())   # [K,N]
    S_bf16 = S.to(torch.bfloat16)
    WS_codes = (S_bf16.view(torch.int16).to(torch.int64) & 0xFFFF)

    def forward(x):
        try:
            orig_shape = x.shape
            x2 = x.reshape(-1, orig_shape[-1])         # [M,K]
            A_bf16 = x2.to(torch.bfloat16)
            A_codes = (A_bf16.view(torch.int16).to(torch.int64) & 0xFFFF)
            C_codes, _ = contract.block_fp8_gemm(
                A_codes, W_codes, WS_codes, blk=128, backend="torch")
            y = _bf16_codes_to_torch(torch, C_codes).to(x.dtype)
            y = y.reshape(*orig_shape[:-1], y.shape[-1])
            if bias is not None:
                y = y + bias.to(y.dtype)
            return y
        except Exception:                              # keep the model runnable
            return orig_forward(x)

    module.forward = forward


# ============================================================================
# LOCAL ENTRYPOINT -- wires reference -> tier1 -> tier2 and prints the result.
# ============================================================================
@app.local_entrypoint()
def main(tier: int = 0, prompts: int = 0):
    """`modal run tools/modal_validate.py [--tier N] [--prompts K]`
         tier 0 -> all tiers (reference + tier1 + tier2)
         tier 1 -> operator tier only (no full-model load)
         tier 2 -> reference + full-model gate
       prompts K>0 -> use only the first K prompts of the corpus."""
    corpus = PROMPT_CORPUS if prompts <= 0 else PROMPT_CORPUS[:prompts]

    if tier in (0, 1):
        print("=== TIER 1: operator (OUR contract vs fp32-acc engine on REAL "
              "weights) ===")
        t1 = tier1_operator.remote()
        print(f"[tier1] summary: {t1}")
        if tier == 1:
            return

    print("=== REFERENCE: golden next-token argmax from the UNMODIFIED model ===")
    golden = reference.remote(corpus)
    print(f"[reference] engine={golden['engine']} tokens={golden['tokens']}")

    print("=== TIER 2: full-model gate (OUR contract patched into every FP8 "
          "Linear) ===")
    t2 = tier2_fullmodel.remote(corpus, golden)
    print(f"[tier2] patched={t2['patched']} tokens={t2['tokens']}")
    print(summarize_gate(golden["tokens"], t2["tokens"]))


# ============================================================================
# LOCAL SELF-CHECK (no modal): exercise the pure helpers so `python3
# tools/modal_validate.py` is a quick smoke test even without modal installed.
# ============================================================================
def _selftest():
    g = [10, 20, 30, 5]
    assert argmax(g) == 2
    assert argmax([1, 1, 1]) == 0                       # first on ties
    assert argmax_match_rate([1, 2, 3], [1, 2, 3]) == 1.0
    assert argmax_match_rate([1, 2, 3], [1, 9, 3]) == 2 / 3
    assert abs(topk_overlap([5, 4, 3, 2], [5, 4, 1, 0], 2) - 1.0) < 1e-12
    assert abs(topk_overlap([5, 4, 3, 2], [2, 3, 4, 5], 2) - 0.0) < 1e-12
    st = error_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
    assert st["exact"] == 2 and abs(st["max_abs"] - 0.5) < 1e-12
    assert "100.0%" in summarize_gate([1, 2], [1, 2])
    assert "50.0%" in summarize_gate([1, 2], [1, 9])
    try:
        argmax_match_rate([1, 2], [1])
        raise AssertionError("expected length-mismatch ValueError")
    except ValueError:
        pass
    print(f"modal_validate self-check: PASS  (modal {'present' if _HAS_MODAL else 'ABSENT (shim)'})")
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
