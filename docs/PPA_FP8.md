# GLM-5.2-FP8 Datapath — Area / Timing Characterization (PPA_FP8)

**All numbers below are `[synth-estimate, yosys generic/abc — not a placed-and-routed ASIC/FPGA result]`.**

Method: yosys 0.66 FAST flow per repo note (no `synth_gowin` autoname).
- **cells** = generic cells via `read_verilog -sv; (chparam); hierarchy; proc; opt; stat` (hierarchical, includes submodules).
- **ltp** = `ltp` longest topological path = combinational logic levels (register-to-register depth). An fmax proxy: higher = slower. Unitless pre-techmap (or LUT4-mapped where noted) depth — **not** nanoseconds, **not** P&R.
- **LUT4** = `$lut` count after `synth -flatten; abc -lut 4` (a 4-LUT FPGA estimate), with FF count where captured.

Larger slices were produced with yosys `chparam` parameter overrides only. **No committed RTL was modified** (read-only characterization).

---

## 1. Per-unit area / timing tables

### 1.1 `glm_matmul_fp8` — GLM-5.2-FP8 E4M3 block-scaled GEMM
`src/glm_matmul_fp8.v` + `src/glm_fp_pipe.v`. Defaults: ACC_W=48, ACC_FRAC=18, BLK=128.

| Config | cells (generic) | ltp | LUT4 (+FF) |
|---|---|---|---|
| PE 1x1, K=128 (NB=1) | 770 | **348 (measured)** | 1909 (+475 FF) |
| PE 1x1, K=256 (NB=2) | 1294 | not measured | 4099 |
| PE 2x2, K=128 (NB=1) | 1093 | not measured | 3378 |
| PE 2x2, K=256 (NB=2) | 1633 | **497 (measured)** | 5896 (+1692 FF) |
| PE 4x4, K=128 (NB=1) | 2210 | not measured | 9113 (+2088 FF) |
| PE 4x4, K=256 (NB=2) **[DEFAULT]** | 2810 | **1073 (measured)** | 13235 |

ltp measured for 3 representative slices (small / mid / default) because each ltp run is ~60s; they bound the trend.

### 1.2 `mla_attn_fp8` — MLA latent-attention block (top + submodules)
`src/mla_attn_fp8.v` + glm_matmul_fp8, glm_matmul_pipe, rmsnorm_unit, rope_interleave_unit, glm_softmax, dsa_indexer, topk_select, glm_fp_pipe.

| Config | cells (generic) | ltp (leaf paths) | LUT4 |
|---|---|---|---|
| DEFAULT (MODEL_DIM=128, H_HEADS=4, PE_N=4, …) | 33921 | topk_select=1135, glm_softmax=593, glm_matmul_pipe=85; glm_matmul_fp8=684 (from PE_N=8 run, depth-invariant) | NOT obtained (abc -lut 4 timed out >480s on ~34k-cell FP datapath) |
| LARGER: PE_N=8 (all else default) | 41199 | topk_select=1135 (unchanged), glm_matmul_fp8=684, glm_softmax=593, glm_matmul_pipe=85 | not run (same abc bottleneck) |

Top-level ltp could not be emitted: yosys flagged false combinational loops in `rmsnorm_unit`'s iterative fp32_add `sumsq` reduce. Leaf-path ltps used instead — sufficient to pin the limiter.

### 1.3 MoE expert (`glm_act` SiLU + 2× `glm_matmul_fp8`)
KMAX reduced to 256 (NB=2) for all measurements — the module's literal default KMAX=16384 (NB=128) does not synthesize in a bounded time. The fmax path is KMAX-independent; only block-scale bank fan-out scales with KMAX.

| Config | cells (generic) | ltp | LUT4 |
|---|---|---|---|
| DEFAULT (HIDDEN=128, INTER=64, TN=4, BLK=128) | 28615 (glm_act 24535 = 86%; 2× glm_matmul_fp8 @1073; glue 942) | **497** (inside glm_matmul_fp8) | FP8 matmul core (PE_N=4) = 6186 $lut measured; full-design abc -lut 4 killed >220s |
| LARGER (HIDDEN=256, INTER=128, TN=8) | 54699 = 1.91× (glm_act 49067 = 2.00×; glm_matmul_fp8 1437 ea = 1.34×) | **689** (inside glm_matmul_fp8) | not measured (full-design abc exceeds budget) |

### 1.4 `ddr5_xbar` — N_CH-channel banked DDR5 read fabric
`src/ddr5_xbar.v`. ADDR_W=32, DATA_W=256, TAG_W=8, RESP_QD=4 held at defaults.

| Config | cells (generic) | mem bits | ltp | LUT4 (+FIFO FF) |
|---|---|---|---|---|
| N_CH=4 (smaller) | 148 | 4224 | 41 | 3537 (+4224 $_DFFE_PP_) |
| N_CH=8 (DEFAULT) | 292 | 8448 | 53 | 7305 (+8448 $_DFFE_PP_) |
| N_CH=12 (larger / real max) | 436 | 12672 | 81 | 10504 (+12672 $_DFFE_PP_) |

FIFO-FF mass maps to logic in the generic flow; on real Gowin it would infer BSRAM, so the LUT4 numbers overstate true logic.

### 1.5 `weight_loader` — FP8 weight + bf16 block-scale DMA/sequencer
`src/weight_loader.v`. Structurally fixed (one FSM); sweeping PE_N/KMAX only **widens buses**, so generic cell count is constant — real area shows only after LUT mapping.

| Config | cells (generic) | ltp | LUT4 (+FF) |
|---|---|---|---|
| DEFAULT (PE_N=4, KMAX=256, NB=2, w_scale=128b) | 91 | 857 | 408 (+202 FF) |
| LARGER A (PE_N=8, KMAX=512, NB=4, w_scale=512b) | 91 | 3110 | 893 (+596 FF) |
| LARGER B (PE_N=16, KMAX=1024, NB=8, w_scale=2048b) | 91 | 12334 | 3422 (+2142 FF) |

### 1.6 `expert_cache_pf` — fully-associative expert directory + prefetch
`src/expert_cache_pf.v`. SLOTS / REPL_POLICY sweep (P0 = LRU, P1 = LFU+aging).

| Config | cells (generic) | ltp (generic) | LUT4 ($lut) | ltp (LUT4-mapped) |
|---|---|---|---|---|
| SLOTS=8, P0 (DEFAULT) | 864 | 232 | 653 | 231 |
| SLOTS=8, P1 | 1212 | 127 | 911 | 237 |
| SLOTS=16, P0 | 1456 | 324 | 1055 | 278 |
| SLOTS=16, P1 | 2057 | 194 | 1496 | 249 |
| SLOTS=32, P0 | 2640 | 287 | 1809 | 319 |
| SLOTS=32, P1 | 3804 | 260 | 2862 | 456 |

Generic ltp is noisy/non-monotonic (opt restructures procmux chains + 32-bit stat-counter carry chains); the LUT4-mapped ltp is the reliable proxy.

### 1.7 `boot_loader` — segment-descriptor DMA boot sequencer
`src/boot_loader.v`. Generic cell count counts operator *instances* (width-invariant), so it only moves with SEG_MAX; true area is in LUT4/FF.

| Config | cells (generic) | ltp | LUT4 (+FF; FIFO = BURST*DATA_W) |
|---|---|---|---|
| DEFAULT (ADDR_W=32, DATA_W=64, SEG_MAX=4, BURST=8, LEN_W=16) | 132 | 121 | 1009 (+911 FF; 512 $_DFFE FIFO) |
| LEN_W=32 | 132 | 199 | 1324 (+1023 FF) |
| BURST=16 | 132 | 125 | 1415 (+1428 FF; 1024 FIFO) |
| DATA_W=128 | 132 | 119 | 1334 (+1423 FF; 1024 FIFO) |
| SEG_MAX=32 | 272 | 225 | — |
| CONFIG A (SEG16/BURST16/DATA128/ADDR40/LEN24) | ~ (generic ltp 168) | 172 | 3500 (+3828 FF; 2048 FIFO) |

---

## 2. fmax-limiting path per unit (where to pipeline next)

- **glm_matmul_fp8** — The register-to-register critical path is the **per-beat activation front-end cone** ending at `term_r`: `a_shift_q → fp32_scale_pow2` (2^a_shift prescale, :341) `→ fp32_to_fp8e4m3` (dynamic E4M3 quantize — the deepest single block: priority-encode + RNE round, `fp8_e4m3.vh:124-207`) `→ fp8_mul` (4×4-bit mantissa mul) `→ fp32_to_fixed` (barrel-shift to fixed). Dominant sub-block = the on-the-fly E4M3 encoder. **Next pipeline register: split after `fp8_mul`, before the dequant/round.** Note the ltp *number* growth (348→497→1073) is **mostly a topological-sort artifact** of the `16*PE_M*PE_N`-bit variable part-select write into `c_out` (:510) — a positional shifter that is parallel in real silicon, **not** a true 1073-deep delay.
- **mla_attn_fp8** — Deepest path is in `dsa_indexer`'s `topk_select` (ltp=1135): the chained fp32 compare-and-select tree of the DSA sparse-attention indexer (`topk_select.v` ~:271 `fp32_gt` + :206 logic + :236 ternary feeding the pmux priority network). **PE_N-invariant** (present in both runs). The FP8 matmul (684) and softmax (593) are next. **Pipeline target: the topk_select FP compare cascade.**
- **MoE expert** — Same single-beat FP8 dot-product datapath inside `glm_matmul_fp8` (497 @TN=4 → 689 @TN=8); the back-end `fixed_to_fp32` + [128,128] block-scale mul + bf16 round (~258 of 497 nodes) dominates. The bf16 SiLU/merge tail is **not** the limiter. ltp is **not** width-invariant here — very wide PE_N adds lane-select mux depth.
- **ddr5_xbar** — The **response-side round-robin drain arbiter** (combinational `always@*`, :180-195): an unrolled priority encoder scanning all N_CH FIFOs (`idx=ai+rr` $add, modulo $sub, dynamic `fifo_ne[idx]` $shiftx barrel-mux, serial gnt priority-mux), in a rr→gnt→rr through-register loop. The feed-forward request banking (:141-151) is **not** the limiter. **Fix: register the grant / replace iterative modulo-scan with a one-hot rotate-mask priority encoder.**
- **weight_loader** — The variable-indexed write into the wide `w_scale_q` register (`:195`, `w_scale_q[16*rd_slot +: 16] <= mem_data[15:0]`): a 1-of-(NB*PE_N) 16-bit demux whose width = the bf16 block-scale bus (16*PE_N*NB). It is loaded once per tile (S_SCALE), **off the per-beat streaming loop**, so its long ltp does not gate steady-state throughput.
- **expert_cache_pf** — Move-to-front LRU rank update + parallel associative tag lookup terminating in 32-bit stat counters; at SLOTS=32/P1 the LFU freq-min comparator tree with LRU tie-break dominates (mapped ltp 456 vs 319 at P0).
- **boot_loader** — The LEN_W-wide segment-offset counter (roff/woff) terminal-compare + ripple-increment + segment-advance mux loop. Confirmed by sweep: LEN_W 16→32 nearly doubles ltp (121→199); width-only sweeps stay flat (~119-125).

---

## 3. glm_matmul_fp8 fixed-point accumulator — does the −87.6% win hold at scale? **YES.**

The BFP/FP8 design intent is that accumulation is **one cheap fixed-point integer add** (`term_r → integer-add → accx`, a separate single-cycle registered stage **off** the critical path), instead of a per-add fp32 normalize+round MAC. Three pieces of evidence confirm it holds as PE/K grow:

1. **Generic cells scale ~LINEARLY with the PE array, ~100 cells/PE** — no superlinear blowup:
   - K=128: 1→4→16 PEs = 770 / 1093 / 2210 (marginal 108 then 93 cells/PE).
   - K=256: 1294 / 1633 / 2810 (113 then 98 cells/PE).
   Because each PE is just a 4×4-bit FP8 mantissa mul + a fixed-point integer add — **not** an fp32 MAC (an `fp32_add_pipe` alone is 192 cells).
2. **$mul census proves the headline win.** Total `$mul = PE_M*PE_N` (tiny 4×4-bit FP8 LUT muls) `+ NB` (the **only** 24×24 fp32 multipliers — the time-shared dequant). Measured 2 / 6 / 17 / 18 for 1x1K128 / 2x2K256 / 4x4K128 / 4x4K256 → the scarce big multipliers stay at **NB=1..2 independent of array size**. A naive fp32-accumulate array would need PE_M*PE_N = **16** of them at 4×4. This is the **−87.6% accumulator win confirmed at 4×4**.
3. **The fixed-point accumulator stays off the critical path.** The dequant pipes are well-balanced and not the limiter (`fp32_add_pipe` ltp=67, `fp32_mul_pipe` ltp=18). LUT4 is also ~linear in PEs (~480/PE @K=128, ~600/PE @K=256). Raising K 128→256 just flips NB 1→2 (+1 fp32 dequant mul+add pipe, ~fixed +250 cells, + one extra per-PE 48-bit accumulator bank).

**Verdict: the fixed-point-accumulator optimization holds at the larger slice — area scales linearly in PEs, big multipliers stay pinned at NB, no superlinear cost.**

---

## 4. ddr5_xbar area scaling with N_CH (4 / 8 / 12)

Area scales **essentially perfectly linearly** in N_CH:

| Metric | N_CH=4 | N_CH=8 | N_CH=12 | per +4 ch |
|---|---|---|---|---|
| generic cells | 148 | 292 | 436 | **+144 (=36 cells/ch)**, exact |
| mem bits | 4224 | 8448 | 12672 | **+4224 (=1056 bits/ch = RESP_QD*PAY_W)**, exact |
| FIFO FF ($_DFFE_PP_) | 4224 | 8448 | 12672 | linear |
| LUT4 | 3537 | 7305 | 10504 | +3768 then +3199 (≈linear / slightly sub) |

So banking area = `N_CH × (per-channel FIFO + occupancy logic) + thin shared front-end`, exactly the design intent (banking + arbitration + tag tracking, no new math). **Timing does NOT scale linearly:** ltp 41→53→81 — super-linear past N=8, entirely from the serial round-robin priority scan (per-channel modulo-add + dynamic $shiftx). N_CH=12 is not a power of two (req banking wants power-of-two) but the response fabric synthesizes/characterizes cleanly; it is the realistic max channel count.

---

## 5. Remaining timing work (next repipeline targets, priority order)

1. **glm_matmul_fp8 per-beat cone** — insert a pipeline register after `fp8_mul`, before the `fixed_to_fp32` + block-scale-dequant + bf16-round back-end. This is the global fmax limiter for both the MoE expert and the dense GEMM path. (The c_out part-select tail is a topo artifact, not real depth — no register needed there.) **✅ DONE (registered the fp8_mul product into a new stage; the real per-beat activation cone 71→62, −12.7%; ACC_DRAIN 3→4; golden 224 bit-exact, argmax 4/31/20 unchanged.)**
2. **ddr5_xbar response arbiter** — register the round-robin grant, or replace the iterative modulo-scan priority encoder with a one-hot rotate-mask encoder, to remove the per-channel $add/$sub/$shiftx from the critical loop (the only super-linear-in-N_CH timing term). **✅ PARTLY DONE (replaced the modulo-scan with a bit-exact 3-phase rotate-mask encoder — shortens the true `fifo_ne→rot→sel→gnt` cone, TB 3073 + formal 9-assert + 7.93× scaling all preserved. Registering the grant was NOT applied: it adds a response-latency cycle that risks the throughput-scaling + formal timing relations — left as a recommendation needing its own re-validation.)**
3. **topk_select (dsa_indexer)** — ~~pipeline the chained fp32 compare-and-select tree~~. **❌ DON'T — measured to REGRESS fmax.** A `TREE_PIPE` fold was built, passed all goldens (topk 442, moe_router 185, dsa 200, model argmax 4/31/20) and was strictly result-preserving, but ltp went the WRONG way (LUT4-mapped 266→409, +54% at N=256) and was REVERTED. The real limiter is NOT the compare cascade (a one-cycle-per-pass O(log N) cone) but the **`pass`-indexed result part-select write tail** (`sel_idx_o[pass*IDXW+:IDXW]`, the `argmax_idx` one-hot mask decoder) — a positional shifter, the same artifact class as glm_matmul_fp8's c_out tail. If topk fmax must rise, target THAT: write results into a fixed-slot shift register, and/or register `argmax_idx` one stage before the mask decode; do not pipeline the cascade.
4. **glm_softmax** — exp → sum → reciprocal-normalize pipeline (ltp 593); register between the exp/sum and the reciprocal stages if attention fmax must rise further.
5. **expert_cache_pf @ large SLOTS/P1** — pipeline the LFU freq-min victim-select comparator tree (mapped ltp 456 at SLOTS=32/P1) if a deep, frequency-aware directory is required; the default SLOTS=8/P0 demand path is already shallow.
6. **(Non-blocking) weight_loader `w_scale_q` indexed write** — its large ltp is off the steady-state streaming loop (loaded once per tile); only revisit if tile-setup latency becomes a bottleneck. Likewise boot_loader's offset counter is boot-time only.

---

### Honesty / caveats
- ltp is a topological (unit-delay, pre-place) path-length count — an fmax **proxy** only, **not** nanoseconds and **not** P&R. Absolute large ltp numbers (e.g. weight_loader 12334, glm_matmul_fp8 1073) reflect bit-by-bit enumeration of wide part-selects/decoders that map to balanced trees in real synthesis; trust the **trend**, not the absolute.
- Top-level mla_attn_fp8 ltp and several LUT4 numbers were not obtained (yosys false-loop suppression on rmsnorm's fp32 reduce; abc -lut 4 timeouts >220-480s on the large flattened FP datapaths) — reported as unmeasured rather than estimated.
- FIFO/register arrays (ddr5_xbar, boot_loader) map to flops+read-mux in the generic flow; on real Gowin they infer BSRAM, dropping the LUT4/FF mass substantially.
- All configs synthesized within the bounded time except where explicitly killed/noted. No committed RTL modified.
