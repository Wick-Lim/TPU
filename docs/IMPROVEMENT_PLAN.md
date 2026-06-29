# GLM-5.2-FP8 accelerator — performance / power improvement plan

The governing model (from `docs/SYSTEM_SINGLE_PACKAGE.md` §7):

```
  tokens/s  ≈   Flash_BW  /  [ (1 − h) × footprint ]   ×   K
  J/token   ≈   bytes_moved × energy/bit   (Flash bytes dominate, ~24–26× DRAM/bit)
```

So every lever is one of: **raise Flash_BW**, **raise hit-rate h**, **shrink footprint**, **raise K
(speculative)**, or **cut bytes_moved** (which helps both tok/s and J/token). The compute die is
*not* the bottleneck (~20–25 % utilized) — die-side optimization (the −87.6 % accumulator, fmax,
formal) improved cost/thermals/correctness but does **not** move tok/s or J/token until Flash is
unblocked. This plan targets the real bottleneck.

Baseline (measured h, [EST] BW): Flash 50 GB/s, h=27 %, K=1 → **~3 tok/s single-user**, ~8–10 J/token.

Legend: 🟢 RTL-doable in this repo · 🟡 system/architecture (design + vendor IP) · 🔴 out of RTL scope.

---

## P1 — Flash bandwidth (the linear lever; biggest single win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 1.1 | **`flash_xbar`** — N-channel banked Flash read fabric | 🟢 | Same pattern as `ddr5_xbar`: stripe expert fetches across N Flash dies/channels → ~N× aggregate Flash_BW. A real 1 TB module is many NAND dies; parallel reads are how you reach "10s of GB/s". Build + BMC-verify like ddr5_xbar. **✅ DONE: deep per-channel outstanding queue (QDEPTH) hides NAND latency 7.99× (Little's law) + N_CH banking stacks (~57× combined @8ch×Q8); 2049 tests + BMC K=12.** | **~N× tok/s** (linear). 4ch ≈ 3→12 tok/s |
| 1.2 | **Flash expert layout** — co-activated experts on different channels | 🟡 | Offline placement so a token's 600 experts spread across channels (avoid channel hotspots), mirroring the DDR5 stripe. A loader/packer convention + a placement table. | sustains 1.1's N× (avoids <N× from collisions) |
| 1.3 | **Deeper Flash read pipeline** — more outstanding requests | 🟢 | Raise the fetch queue depth so Flash stays saturated despite per-read latency (BW = outstanding / latency). Wire into `expert_cache_pf` + `flash_xbar`. | recovers the latency-bound gap to peak BW |
| 1.4 | Faster Flash medium (PCIe5 NVMe / more dies) | 🟡 | Vendor/board choice; the controller is vendor IP. Document the BW target the RTL fabric must feed. | linear, but $ + board |

## P2 — Cut bytes moved (raises tok/s AND lowers J/token — the dual win)

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 2.1 | **Expert decompressor** in the Flash→DDR5 path | 🟢 | FP8 expert weights have entropy < 8 bits; store them losslessly-compressed in Flash and decompress on-chip during fetch (e.g. a lightweight zero-RLE / dictionary / Huffman decoder). Cuts Flash bytes/expert ~1.3–1.6× → effective Flash_BW ↑ and Flash energy ↓ by the same factor. Build a `weight_decomp` unit + verify decompressed == original FP8. **✅ DONE: canonical-Huffman `weight_decomp`, bit-exact lossless, measured 1.34× on representative FP8 weights (5.97 bits/sym); 8 tests.** | **~1.34×** measured tok/s + Flash energy |
| 2.2 | **MTP K>1 / better draft** — verify more tokens per weight-load | 🟢 | We have MTP×2 (K=2). Extend `mtp_head_fp8` / `spec_decode_seq` to a deeper draft (multi-token MTP or a small draft head) so K and acceptance α rise → Flash traffic ÷ K_eff without leaving FP8. *(The shipped model has 1 MTP layer; K>1 needs a chained/multi-token draft — more involved, lower-confidence.)* | **×K_eff** (K=2→3 ≈ +50 % tok/s) |
| 2.3 | **Higher cache hit-rate h** — bigger cache + predictor-driven prefetch | 🟢 | Wire `expert_predictor` into `expert_cache_pf` prefetch. **❌ MEASURED NO-OP at GLM cache size (`expert_prefetch_top`):** at SLOTS=900 (>reuse distance) the predictor hints the most-popular expert, which LRU **already keeps resident** → 0 prefetches issued, hit-rate byte-identical to baseline; deep LOOKAHEAD 1/2/3 identical. Only helps in the narrow regime just *below* the reuse-distance knee (SLOTS=550: 0%→4.6%). **Conclusion: hit-rate is capped by fine-grained-routing entropy, not prefetch cleverness — this lever does not move the real config.** | **~0 %** at real cache size (honest) |

## P3 — Hide latency / raise utilization

| # | Item | Type | Plan | Est. impact |
|---|---|---|---|---|
| 3.1 | **Predictor-driven deep prefetch loop** | 🟢 | Use `expert_predictor` confidence to prefetch ahead of the demand cache (L+2..) so Flash stays saturated and the die stalls less; double/triple-buffer experts. | sustains P1/P2 (keeps Flash busy) |
| 3.2 | **Idle-die clock gating** | 🟢 | The die idles ~75 % waiting on Flash; clock-gate the compute lanes during fetch stalls. Pure power win (no tok/s change). | **~10–20 %** lower idle energy |

## P4 — Energy-specific (J/token)

The Flash byte movement is ~80 % of per-token energy. P2 (decompress, MTP, hit-rate) is the main
energy lever — it directly cuts Flash bytes. P3.2 (clock gating) trims idle. Beyond RTL:

| # | Item | Type | Note |
|---|---|---|---|
| 4.1 | HBM instead of DDR5 (if energy ≫ cost) | 🟡 | HBM is the lower-energy-per-bit fast tier; the DDR5 choice trades energy for cost. A build-time option. |
| 4.2 | Computational storage / near-Flash compute | 🔴 | Research; moves compute to the data to avoid moving bytes. Out of RTL scope. |

---

## Projected combined effect (single-user, [EST])

Stacking the 🟢 RTL items on the baseline (~3 tok/s, ~8–10 J/token):

| Step | Lever | tok/s | J/token | status |
|---|---|---|---|---|
| baseline | — | ~3 | ~9 | — |
| + flash_xbar 4ch (1.1) | Flash_BW ×4 | ~12 | ~9 | ✅ built (7.99× latency-hide + N× bank) |
| + expert decompress 1.34× (2.1) | bytes ÷1.34 | ~16 | ~6.7 | ✅ built (measured 1.34×) |
| + MTP K_eff 1.7 (2.2) | ÷ traffic | ~27 | ~4.5 | ⏳ involved (1 MTP layer) |
| ~~+ hit-rate (2.3)~~ | ~~(1−h)↓~~ | — | — | ❌ measured NO-OP at real cache |
| + clock gating (3.2) | idle ↓ | ~27 | ~3.8 | ⏳ enable-side RTL (ICG vendor) |

**Revised target (honest, post-measurement): ~3 → ~16 tok/s built today** (flash_xbar ×4 +
decompress 1.34×), **~27 with MTP K>1**; **~9 → ~4–7 J/token**. The cache-hit-rate lever (2.3) was
*measured* to be a no-op at the real cache size — the popular experts the predictor names are
already resident — so the real gains come from **Flash bandwidth + fewer bytes (decompress, MTP)**,
not cache cleverness. All numbers [EST]; the built RTL items are verified here.

## Execution order (RTL, by impact-per-effort)

1. **`flash_xbar`** (P1.1) — biggest single win, proven pattern (clone ddr5_xbar), BMC-verifiable.
2. **`weight_decomp`** (P2.1) — dual tok/s+energy win, self-contained, verify decode==FP8.
3. **predictor-driven deep prefetch** (P3.1) + hit-rate (P2.3) — wire the built predictor in, measure.
4. **MTP K>1** (P2.2) — extend the speculative loop, verify spec==greedy still exact.
5. **idle clock-gating** (P3.2) — power, low-risk.
6. Document the 🟡 system items (Flash layout, faster medium, HBM option) as build/board choices.
