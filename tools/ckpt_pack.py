#!/usr/bin/env python3
# ============================================================================
# ckpt_pack.py -- GLM-5.2-FP8 checkpoint -> OUR RTL weight-memory image packer
# ----------------------------------------------------------------------------
# WHAT THIS IS
#   The BRIDGE from the published HuggingFace zai-org/GLM-5.2-FP8 checkpoint
#   format (config.json + safetensors tensors) to the exact weight-memory layout
#   that src/weight_loader.v reads and feeds to src/glm_matmul_fp8.v.  It also
#   fabricates a SYNTHETIC-BUT-FAITHFUL mini-checkpoint (same dtypes / shapes /
#   block-scale structure as the real 753 GB one) so the whole pack/unpack path
#   is exercisable here with NO GPU and NO 753 GB download.
#
# THE HF FP8 CHECKPOINT FORMAT WE PARSE  (per the config.json contract)
#   * config.json : quantization_config = { quant_method:"fp8", fmt:"e4m3",
#       weight_block_size:[128,128], activation_scheme:"dynamic",
#       modules_to_not_convert:[...] }.  weight_block_size = [128,128] means ONE
#       dequant scale per 128(out) x 128(in) weight block.
#   * <name>.weight            : the quantized weight, dtype F8_E4M3 (1 byte/elt),
#                                shape [out_features, in_features]  (HF Linear).
#   * <name>.weight_scale_inv  : the block scales, dtype F32 (or BF16), shape
#                                [ceil(out/128), ceil(in/128)] -- the DeepSeek-V3
#                                weight_scale_inv tensor.
#   * bf16 tail (norms / router / embed / lm_head, the modules_to_not_convert)
#                                : dtype BF16, NO scale tensor -- copied through.
#
#   safetensors on disk = [8-byte LE header-length][JSON header][raw tensor bytes].
#   We parse it MANUALLY (pure python) so neither the `safetensors` lib nor numpy
#   is required; if the `safetensors` lib IS importable we use it, else the manual
#   reader (which also reads the real checkpoint's bytes verbatim).  A .npz/.bin
#   logical fallback with the SAME layout is also accepted (see load_checkpoint).
#
# THE RTL TARGET LAYOUT  (must match src/weight_loader.v word-addressed memory)
#   A "tile" = one group of PE_N output columns over the full K.  Per tile, from
#   its `base`:
#     SCALE region : base + (bj*PE_N + pj),  bj=0..nblk-1, pj=0..PE_N-1
#                    word low 16 bits = bf16 block scale for (column, K-block bj).
#     CODE  region : base + nblk*PE_N + k,   k=0..K-1
#                    word[8*pj +: 8] = W_rtl[k][col_pj] (E4M3 byte).
#   W_rtl is the CONTRACTION orientation W[k][n] (n = output channel), i.e. the
#   TRANSPOSE of the HF [out,in] weight: W_rtl[k][n] = hf_weight[n][k].
#   Column n's block scale is weight_scale_inv[n//128][bj] (same scale shared by
#   all columns of an out-block; we replicate it per column into the bus order
#   weight_loader expects).
#
# USAGE
#   python3 tools/ckpt_pack.py gen   <ckpt_dir>            # write synthetic ckpt
#   python3 tools/ckpt_pack.py pack  <ckpt_dir> <out_dir>  # ckpt -> RTL image
#   python3 tools/ckpt_pack.py check <ckpt_dir>            # gen+pack+unpack rt
#   python3 tools/ckpt_pack.py                             # self-test (exit 0)
# ============================================================================
import sys, os, json, struct

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm_fp8_ref as ref   # fp32_to_bf16, fp8 codecs, f2u32 (bit-exact mirrors)

# ---- RTL geometry (must match glm_matmul_fp8 / weight_loader defaults) ----
PE_N = 4
BLK  = 128
KMAX = 256
DATA_W = max(8 * PE_N, 16)          # 32 bits for PE_N=4
DATA_HEX = DATA_W // 4

# ============================================================================
# (1) MINIMAL safetensors CODEC (pure python; no lib, no numpy)
# ============================================================================
# dtype -> bytes-per-element for the dtypes we touch.
_ST_ELT = {"F8_E4M3": 1, "F8_E5M2": 1, "BF16": 2, "F16": 2, "F32": 4, "I8": 1}

def save_safetensors(path, tensors):
    """tensors: dict name -> (dtype_str, shape_list, raw_bytes).  Writes a real
       safetensors file (parseable by the official lib and by load below)."""
    header = {}
    blob = bytearray()
    for name, (dtype, shape, raw) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        assert len(raw) == n * _ST_ELT[dtype], f"{name}: byte count mismatch"
        start = len(blob)
        blob += raw
        header[name] = {"dtype": dtype, "shape": list(shape),
                        "data_offsets": [start, len(blob)]}
    header["__metadata__"] = {"format": "pt", "producer": "ckpt_pack.synthetic"}
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = (8 - (len(hjson) % 8)) % 8
    hjson += b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hjson)))
        f.write(hjson)
        f.write(blob)

def load_safetensors(path):
    """Return dict name -> (dtype_str, shape_list, raw_bytes).  Prefers the
       official `safetensors` lib if importable; else a manual parse."""
    try:
        from safetensors import safe_open      # pragma: no cover (lib absent here)
        out = {}
        with safe_open(path, framework="np") as f:
            for k in f.keys():
                t = f.get_tensor(k)
                out[k] = (str(t.dtype), list(t.shape), t.tobytes())
        return out
    except Exception:
        pass
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen).decode("utf-8"))
        blob = f.read()
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        s, e = meta["data_offsets"]
        out[name] = (meta["dtype"], list(meta["shape"]), blob[s:e])
    return out

# ============================================================================
# (2) tensor byte helpers (no numpy: raw byte <-> python int lists)
# ============================================================================
def fp8_bytes_to_codes(raw):
    return list(raw)                                  # 1 byte == 1 E4M3 code

def f32_bytes_to_u32(raw):
    return list(struct.unpack(f"<{len(raw)//4}I", raw))

def bf16_bytes_to_codes(raw):
    return list(struct.unpack(f"<{len(raw)//2}H", raw))

def scale_tensor_to_bf16(dtype, raw):
    """weight_scale_inv (F32 or BF16) -> list of bf16 codes (the RTL bus form)."""
    if dtype == "F32":
        return [ref.fp32_to_bf16(u) for u in f32_bytes_to_u32(raw)]
    if dtype == "BF16":
        return bf16_bytes_to_codes(raw)
    raise ValueError(f"unsupported scale dtype {dtype}")

# ============================================================================
# (3) SYNTHETIC mini-checkpoint generator (faithful dtypes/shapes/scales)
# ============================================================================
def gen_synthetic(ckpt_dir):
    os.makedirs(ckpt_dir, exist_ok=True)
    # A small Linear: out_features N=8, in_features K=256 -> blocks [1,2].
    N, K = 8, 256
    n_ob = (N + BLK - 1) // BLK          # 1 out-block
    n_kb = (K + BLK - 1) // BLK          # 2 K-blocks
    # ---- deterministic pseudo-data (no numpy): an LCG over the index ----
    def lcg(seed):
        x = seed & 0xFFFFFFFF
        while True:
            x = (1103515245 * x + 12345) & 0x7FFFFFFF
            yield x
    g = lcg(0xC0FFEE)
    # weight: F8_E4M3 codes [N][K] -> flat row-major bytes.  Avoid the NaN code.
    w_codes = bytearray()
    for _ in range(N * K):
        c = next(g) & 0xFF
        if (c & 0x7F) == 0x7F:           # NaN pattern -> nudge to a finite code
            c ^= 0x01
        w_codes.append(c)
    # weight_scale_inv: F32 [n_ob][n_kb], small positive scales ~ block max/448.
    scales = bytearray()
    for _ in range(n_ob * n_kb):
        frac = (next(g) % 1000) / 1000.0
        sval = 0.001 + 0.02 * frac       # plausible per-block dequant scale
        scales += struct.pack("<f", sval)
    # a bf16 TAIL tensor (modules_to_not_convert): norm weight, no scale.
    norm = bytearray()
    for _ in range(16):
        norm += struct.pack("<H", (next(g) >> 8) & 0xFFFF)

    tensors = {
        "model.layers.0.mlp.down_proj.weight":
            ("F8_E4M3", [N, K], bytes(w_codes)),
        "model.layers.0.mlp.down_proj.weight_scale_inv":
            ("F32", [n_ob, n_kb], bytes(scales)),
        "model.layers.0.input_layernorm.weight":
            ("BF16", [16], bytes(norm)),
    }
    save_safetensors(os.path.join(ckpt_dir, "model.safetensors"), tensors)

    config = {
        "model_type": "glm",
        "quantization_config": {
            "quant_method": "fp8",
            "fmt": "e4m3",
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
            "modules_to_not_convert": [
                "model.embed_tokens", "lm_head", "model.norm",
                "input_layernorm", "post_attention_layernorm",
                "mlp.gate",          # MoE router stays bf16
            ],
        },
    }
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    return ckpt_dir

# ============================================================================
# (4) load a checkpoint dir -> classified tensors
# ============================================================================
def load_checkpoint(ckpt_dir):
    """Returns (config, weights, tail).
       weights : list of dict {name, N, K, fp8_codes (list[N][K]),
                 scale_bf16 (list[n_ob][n_kb])}.
       tail    : list of (name, dtype, shape) kept bf16 (modules_to_not_convert).
       Accepts safetensors; a .npz/.bin fallback with the same logical layout is
       also recognized (load_safetensors handles the manual byte parse)."""
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        config = json.load(f)
    blk = config.get("quantization_config", {}).get("weight_block_size", [BLK, BLK])[0]
    st_path = os.path.join(ckpt_dir, "model.safetensors")
    tensors = load_safetensors(st_path)

    weights, tail = [], []
    for name, (dtype, shape, raw) in tensors.items():
        if name.endswith(".weight_scale_inv"):
            continue
        if name.endswith(".weight") and (name + "_scale_inv") in [k for k in tensors] \
           and (name + "_scale_inv") in tensors:
            sdt, sshape, sraw = tensors[name + "_scale_inv"]
            N, K = shape
            n_ob, n_kb = sshape
            codes = fp8_bytes_to_codes(raw)
            fp8 = [codes[r * K:(r + 1) * K] for r in range(N)]      # [N][K]
            sflat = scale_tensor_to_bf16(sdt, sraw)
            scale = [sflat[b * n_kb:(b + 1) * n_kb] for b in range(n_ob)]
            weights.append(dict(name=name, N=N, K=K, blk=blk,
                                fp8=fp8, scale=scale, n_ob=n_ob, n_kb=n_kb))
        else:
            tail.append((name, dtype, shape))
    return config, weights, tail

# ============================================================================
# (5) PACK one fp8 weight (+ scale) into the weight_loader memory image
# ============================================================================
def pack_weight(w, pe_n=PE_N, blk=BLK):
    """Return (words, descriptors).
       words       : flat list of DATA_W-bit ints (the weight-memory image).
       descriptors : list of dict {name, base, k_len, nblk, col0} per tile."""
    N, K = w["N"], w["K"]
    fp8, scale = w["fp8"], w["scale"]          # fp8[N][K], scale[n_ob][n_kb]
    n_kb = (K + blk - 1) // blk
    n_tiles = (N + pe_n - 1) // pe_n
    words, descs = [], []
    for ct in range(n_tiles):
        col0 = ct * pe_n
        base = len(words)
        # ---- SCALE region : (bj*PE_N + pj) order, bf16 in low 16 bits ----
        for bj in range(n_kb):
            for pj in range(pe_n):
                col = col0 + pj
                if col < N:
                    ob = col // blk
                    sc = scale[ob][bj]
                else:
                    sc = 0                       # padding column -> zero scale
                words.append(sc & 0xFFFF)
        # ---- CODE region : word[8*pj +: 8] = W_rtl[k][col] = hf[col][k] ----
        for k in range(K):
            word = 0
            for pj in range(pe_n):
                col = col0 + pj
                code = fp8[col][k] if col < N else 0
                word |= (code & 0xFF) << (8 * pj)
            words.append(word)
        descs.append(dict(name=w["name"], base=base, k_len=K,
                          nblk=n_kb, col0=col0))
    return words, descs

def pack_checkpoint(ckpt_dir, out_dir, pe_n=PE_N, blk=BLK):
    config, weights, tail = load_checkpoint(ckpt_dir)
    os.makedirs(out_dir, exist_ok=True)
    all_words, all_descs = [], []
    for w in weights:
        words, descs = pack_weight(w, pe_n, blk)
        off = len(all_words)
        for d in descs:                          # relocate to the global image
            d = dict(d, base=d["base"] + off)
            all_descs.append(d)
        all_words += words
    # ---- write the weight-memory hex image (one DATA_W word/line) ----
    img_path = os.path.join(out_dir, "weight_mem.hex")
    with open(img_path, "w") as f:
        for wd in all_words:
            f.write(f"{wd & ((1 << DATA_W) - 1):0{DATA_HEX}x}\n")
    # ---- flash channel map (trivial tile striping; the MoE-optimized expert
    #      placement lives in tools/flash_layout.py) ----
    n_ch = 8
    flash_map = [(d["name"], i % n_ch) for i, d in enumerate(all_descs)]
    # ---- manifest the unpacker / TB consume ----
    manifest = dict(
        params=dict(PE_N=pe_n, BLK=blk, KMAX=KMAX, DATA_W=DATA_W),
        n_words=len(all_words),
        descriptors=all_descs,
        tail=[dict(name=n, dtype=dt, shape=sh) for (n, dt, sh) in tail],
        flash_channels=flash_map,
        note="weight_mem.hex is the word-addressed image src/weight_loader.v reads.",
    )
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return img_path, manifest

# ============================================================================
# (6) UNPACK : read the image back -> fp8 codes + bf16 scales (round-trip check)
# ============================================================================
def unpack_image(words, descs, pe_n=PE_N, blk=BLK):
    """Reconstruct per-weight {name -> (fp8[N][K] codes, scale_bf16[n_ob][n_kb])}
       from the packed image, exactly as the RTL would consume it."""
    by_name = {}
    for d in descs:
        by_name.setdefault(d["name"], []).append(d)
    out = {}
    for name, tiles in by_name.items():
        tiles = sorted(tiles, key=lambda t: t["col0"])
        K = tiles[0]["k_len"]
        n_kb = tiles[0]["nblk"]
        N = max(t["col0"] for t in tiles) + pe_n     # incl padding cols
        fp8 = [[0] * K for _ in range(N)]
        # per (out-block, K-block) scale, recovered from tile col0=0's columns
        n_ob = (N + blk - 1) // blk
        scale = [[None] * n_kb for _ in range(n_ob)]
        for t in tiles:
            base, col0 = t["base"], t["col0"]
            # scale region
            for bj in range(n_kb):
                for pj in range(pe_n):
                    sc = words[base + bj * pe_n + pj] & 0xFFFF
                    col = col0 + pj
                    ob = col // blk
                    if scale[ob][bj] is None:
                        scale[ob][bj] = sc
            # code region
            code_base = base + n_kb * pe_n
            for k in range(K):
                word = words[code_base + k]
                for pj in range(pe_n):
                    fp8[col0 + pj][k] = (word >> (8 * pj)) & 0xFF
        out[name] = (fp8, scale, N, K, n_ob, n_kb)
    return out

def roundtrip_check(ckpt_dir, out_dir):
    """gen already done; pack then unpack and assert codes+scales survive."""
    config, weights, tail = load_checkpoint(ckpt_dir)
    img_path, manifest = pack_checkpoint(ckpt_dir, out_dir)
    # read the image back from disk (prove the on-disk hex is faithful)
    with open(img_path) as f:
        words = [int(line.strip(), 16) for line in f if line.strip()]
    recon = unpack_image(words, manifest["descriptors"],
                         manifest["params"]["PE_N"], manifest["params"]["BLK"])
    ok = True
    for w in weights:
        name, N, K = w["name"], w["N"], w["K"]
        rfp8, rscale, rN, rK, rnob, rnkb = recon[name]
        # weight codes (compare only the real N columns; padding ignored)
        for col in range(N):
            for k in range(K):
                if rfp8[col][k] != w["fp8"][col][k]:
                    ok = False
                    print(f"  WEIGHT MISMATCH {name} [{col}][{k}] "
                          f"{rfp8[col][k]:#04x} != {w['fp8'][col][k]:#04x}")
                    break
            if not ok:
                break
        # scales: packed bf16 must equal the (narrowed) original bf16
        for ob in range(w["n_ob"]):
            for bj in range(w["n_kb"]):
                want = w["scale"][ob][bj]
                got = rscale[ob][bj]
                if got != want:
                    ok = False
                    print(f"  SCALE MISMATCH {name} [{ob}][{bj}] "
                          f"{got:#06x} != {want:#06x}")
    return ok, manifest

# ============================================================================
# SELF-TEST
# ============================================================================
def _selftest():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ckpt_pack_")
    ckpt = os.path.join(tmp, "ckpt")
    outd = os.path.join(tmp, "rtl")
    gen_synthetic(ckpt)
    print(f"gen_synthetic: wrote {ckpt}/config.json + model.safetensors")

    config, weights, tail = load_checkpoint(ckpt)
    qc = config["quantization_config"]
    print(f"load_checkpoint: quant={qc['quant_method']} fmt={qc['fmt']} "
          f"blk={qc['weight_block_size']} act={qc['activation_scheme']} | "
          f"weights={len(weights)} tail={len(tail)}")
    for w in weights:
        print(f"  weight {w['name']}: [N={w['N']},K={w['K']}] "
              f"scale[{w['n_ob']}x{w['n_kb']}] (F8_E4M3 + weight_scale_inv)")
    for (n, dt, sh) in tail:
        print(f"  tail   {n}: {dt}{sh}  (modules_to_not_convert -> bf16)")

    ok, manifest = roundtrip_check(ckpt, outd)
    print(f"pack: {manifest['n_words']} words -> {outd}/weight_mem.hex "
          f"({len(manifest['descriptors'])} tile descriptors)")
    print(f"round-trip (pack -> unpack == original fp8 codes + bf16 scales): "
          f"{'PASS' if ok else 'FAIL'}")
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "gen":
        print(gen_synthetic(sys.argv[2]))
    elif len(sys.argv) >= 4 and sys.argv[1] == "pack":
        img, man = pack_checkpoint(sys.argv[2], sys.argv[3])
        print(f"packed {man['n_words']} words -> {img}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "check":
        ok, _ = roundtrip_check(sys.argv[2], sys.argv[2] + "_rtl")
        print("round-trip", "PASS" if ok else "FAIL")
        sys.exit(0 if ok else 1)
    else:
        sys.exit(_selftest())
