# TPU v2.0 — Architecture Specification

Status: production-design target (supersedes the v1.5 toy core).
Audience: RTL implementers and verification engineers.
Companion document: [`docs/ISA.md`](docs/ISA.md) (instruction encodings, register model, illegal-opcode behavior). The two documents are normative and must stay consistent; this file is the source of truth for microarchitecture and sizes, `ISA.md` for encodings.

---

## 0. Goals and what changes from v1.5

v1.5 is a working 5-stage scalar pipeline whose "tensor" EX units are 32-bit toys (packed scalar dot-product instead of GEMM, 1-D MAC instead of 2-D conv, linear normalization instead of softmax, single-element attention with no softmax). v2.0 keeps the **proven scalar control pipeline and hazard model** but replaces every toy compute unit with a **real algorithm operating on real tensor tiles held in an on-chip tile memory**, with full-width arithmetic and an explicit fixed-point contract.

Production bar addressed:
1. Real algorithms: output-stationary 4×4 systolic GEMM; true 2-D 8×8∗3×3 convolution with line+window buffering; fixed-point exp-based softmax (64-entry exp(−k·0.25) LUT + a divide-free degree-4 Maclaurin residual, and an exact rounded integer-divide reciprocal) over an 8-vector; full scaled-dot-product attention (Q·Kᵀ → softmax → ·V) over seq=4, d=4.
2. Full-width arithmetic, documented Q-formats, explicit saturation/rounding. No silent truncation.
3. Parameterization through `localparam`/`parameter`; single opcode source of truth (`tpu_defs.vh`).
4. Synthesizable: resets on all state, no latches, no comb loops, no non-synthesizable constructs in DUT. Passes `verilator --lint-only -Wall` and yosys elaborate/synth.
5. Real ISA with an immediate path, a load-immediate instruction, a hardwired-zero register (`r0`), illegal-opcode handling.
6. Per-unit self-checking TBs with INDEPENDENT golden models (real-number `$rtoi`/`real` reference, computed differently from the DUT), directed + constrained-random, plus a system integration TB.
7. Docs: this SPEC, the ISA reference, per-module headers, accurate README, CI Makefile.

---

## 1. Datapath and tensor model

### 1.1 Two operand domains

The core has **two distinct, explicitly separated operand domains** (this is the central architectural decision that lets us keep the v1.5 pipeline while adding real tensors):

| Domain | Storage | Width | Addressed by | Used by |
|---|---|---|---|---|
| **Scalar** | Register file `RF`, 16 × 32-bit | 32-bit word | 4-bit reg index | control flow, ALU, address math, LOADI, loads/stores, DMA/gather index math |
| **Tensor** | Tile memory `TM`, 32 × 128-bit lines | one 128-bit *line* = a row of 4 lanes, or a packed sub-vector | 5-bit tile-line index (held in a scalar register or an immediate) | GEMM, CONV2D, SOFTMAX, ATTENTION |

A 4×4 `int8` matrix = 16 bytes = 128 bits = **one TM line group of 1 line** when packed as 16 lanes, or **4 lines** when stored one row (4×`int32`) per line. We standardize on **one TM line = 128 bits = 4 × 32-bit lanes (one matrix row, or 4 vector elements)**. A 4×4 matrix therefore occupies **4 consecutive TM lines**; an 8-vector occupies **2 lines**; an 8×8 feature map occupies **8 lines** (one image row, 8 × int16 = 128 bits, per line).

Rationale: a 32-bit scalar register cannot name a tile, so tensor instructions name tiles by a **base TM line index** (immediate or scalar register holding the index) plus a fixed, parameter-defined extent. This removes the v1.5 fiction of packing a "matrix" into one 32-bit word.

### 1.2 Fixed sizes (small but real)

All sizes are `localparam` in `tpu_defs.vh`; the numbers below are the committed defaults.

| Parameter | Value | Meaning |
|---|---|---|
| `XLEN` | 32 | scalar word width |
| `LANE_W` | 32 | tensor lane width (one element, fixed-point Q-format per op) |
| `LINE_LANES` | 4 | lanes per TM line |
| `LINE_W` | 128 | TM line width = `LANE_W*LINE_LANES` |
| `TM_LINES` | 32 | tile memory depth (lines) |
| `RF_REGS` | 16 | scalar registers (`r0`..`r15`, `r0` hardwired 0) |
| `GEMM_N` | 4 | GEMM tile is `GEMM_N × GEMM_N` (4×4) |
| `CONV_H`/`CONV_W` | 8/8 | conv input feature map 8×8 |
| `CONV_K` | 3 | conv kernel 3×3 |
| `CONV_OH`/`CONV_OW` | 6/6 | conv output (valid padding) = (8-3+1) |
| `SEQ_LEN` | 4 | attention sequence length |
| `D_MODEL` | 4 | attention head dimension |
| `SM_LEN` | 8 | softmax vector length |

These are small enough for near-exhaustive directed + constrained-random verification under iverilog, yet each exercises the *full* algorithm (a 4×4 GEMM is a genuine systolic array; an 8×8∗3×3 conv genuinely needs line/window buffers; seq=4/d=4 attention genuinely needs Q·Kᵀ, a softmax over 4 scores, and a ·V).

### 1.3 Fixed-point formats (single source of truth, no silent truncation)

All tensor data are signed two's-complement fixed-point. The format is **per-operation, documented, and enforced by the unit**, never silently truncated:

| Use | Element format | Accumulator | Notes |
|---|---|---|---|
| GEMM inputs (A, B) | `Q7.8` (16-bit value sign-extended into 32-bit lane) | `Q15.16` 48-bit internal accumulator, then **round-half-up + saturate** to `Q7.8` output | product of two `Q7.8` is `Q14.16`; 4 of them sum without overflow in 48 bits |
| CONV inputs/kernel | `Q7.8` | `Q15.16` 48-bit | 9 taps × `Q14.16`, no overflow in 48 bits |
| SOFTMAX input (logits) | `Q7.8` | exp in `Q15.16`, reciprocal in `Q1.30` | output probabilities in `Q0.16` (range [0,1], 0xFFFF≈1.0) |
| ATTENTION Q,K,V | `Q7.8` | scores `Q15.16`; scaled by `1/sqrt(d)` (`d=4` ⇒ `>>1` exact); softmax `Q0.16`; context `Q7.8` | full pipeline, all widths explicit |

**Rounding policy:** round-half-up (`acc + (1<<(FRAC-1))`) before the right shift that returns to `Q7.8`/output format.
**Saturation policy:** after rounding, clamp to the signed range of the output format; saturation is **flagged** (sticky `sat` status bit per unit, surfaced in the status register — see §4). There is **no truncation that can silently lose magnitude**: every narrowing is round-then-saturate with a visible flag.

This directly kills the v1.5 attention bug (`(query*key)>>8` truncated the 64-bit product to 32 bits): scores are computed in a 48-bit accumulator and only narrowed by an explicit round+saturate with a flag.

---

## 2. Memory map

Flat, word-addressed; the three storages are architecturally distinct address spaces (no aliasing), selected by instruction class.

| Space | Element | Depth | Index width | Reset |
|---|---|---|---|---|
| `RF` scalar registers | 32-bit | 16 | 4 | all 0; `r0` permanently 0 |
| `TM` tile memory | 128-bit line | 32 | 5 | all 0 |
| `DMEM` data memory | 32-bit word | 256 | 8 | all 0 |

- **Scalar LOAD/STORE** move 32-bit words between `RF` and `DMEM` (`DMEM[rA]`), exactly as v1.5 but with an 8-bit address from the immediate or a register (v1.5 only used `[3:0]`).
- **Tile LOAD/STORE (TLOAD/TSTORE)** move one 128-bit line between `TM` and four consecutive `DMEM` words.
- **DMA** copies a run of `DMEM` words `DMEM[dst+i]=DMEM[src+i]`, length from an immediate (1..N), via a small MEM-stage FSM — a real multi-word copy, not v1.5's single word.
- **GATHER/SCATTER** index `DMEM` (`DMEM[base + index*stride]`); v2.0 adds SCATTER (store), removing the v1.5 "gather only" gap.

All three storages: combinational read (so single-cycle scalar EX/MEM still works and so the tensor FSMs can read operands within a cycle), synchronous write, synchronous reset to 0 (no `X`, no non-synthesizable `initial` in the DUT).

---

## 3. Pipeline and hazard model

### 3.1 Scalar pipeline (unchanged in spirit from v1.5)

Five stages **IF → ID → EX → MEM → WB** for all *scalar* instructions (NOP, LOADI, ADD/SUB/AND/OR/XOR/SHIFT, RELU, LOAD/STORE, DMA, GATHER/SCATTER, branches if added). Kept verbatim from v1.5 because it already passes 24/24 with full forwarding and a load-use stall:

- RF: combinational read (3 ports), synchronous write, reset-cleared. `r0` reads as 0 and ignores writes.
- DMEM: combinational read, synchronous write, reset-cleared, dual read port (for DMA source).
- Control decoder produces per-opcode `writes_reg/mem_read/mem_write/uses_tensor`; `write_enable` driven by `writes_reg` (never `opcode!=0`).
- Forwarding from EX, EX/MEM, MEM/WB into the ID operand path; one-cycle load-use stall for LOAD/GATHER/DMA producers. Programmer never hand-spaces dependent scalar ops.
- Illegal opcode → `illegal_opcode` flag + safe NOP (no arch side effects).

### 3.2 Tensor instructions: multi-cycle EX with a busy/stall handshake

A real GEMM/conv/softmax/attention cannot complete in one combinational EX cycle. Tensor ops therefore execute as a **multi-cycle EX** owned by a per-unit FSM:

- A tensor instruction in ID asserts `uses_tensor`. When it reaches EX, the pipeline **stalls IF/ID/EX** (holds the tensor instruction in EX, injects bubbles downstream) until the unit raises `done`. This is the same stall primitive already in v1.5 (freeze upstream, bubble downstream), generalized from a fixed 1-cycle load-use stall to a variable-length `busy` stall.
- Tensor units read their input tiles directly from `TM` (combinational read) and write their result tile(s) back to `TM` through a dedicated `TM` write port; they do **not** write `RF`. Completion (`done`) optionally writes a scalar status word (e.g. saturation flags, argmax) to `rC` so software can branch on it — this reuses the scalar WB path.
- Because tensor ops are self-contained (TM→TM), there is **no operand forwarding hazard** between a tensor op and following scalar ops beyond a structural `TM` read-after-write, which the `busy` stall already serializes (the producing tensor op fully retires before the next tensor op enters EX). A scalar op that reads `TM` indirectly (TLOAD/TSTORE) is serialized the same way.

Latencies (deterministic, `localparam`-derived; cycle counts are the committed RTL targets):

| Op | EX latency (cycles) | Why |
|---|---|---|
| scalar ALU/RELU/LOADI | 1 | combinational EX |
| LOAD/STORE/GATHER/SCATTER | 1 (resolves in MEM) | combinational DMEM |
| DMA(len=L) | L (+2) | one DMEM word/cycle |
| GEMM 4×4 | `2*GEMM_N - 1 + GEMM_N` = 11 | systolic fill+drain (see §5.1) |
| CONV2D 8×8∗3×3 | `(CONV_H+CONV_K) + CONV_OH*CONV_OW + 1` = **48** | 11-cycle TM load stream (8 image rows + 3 kernel rows) + 36 compute (one output pixel/cycle, MAC over the 3×3 window) + 1 done |
| SOFTMAX len=8 | **23** | start→done, inclusive of the start edge.  FSM stages `S_RD0, S_RD1, S_MAX, S_EXP(×8), S_RECIP, S_NORM(×8), S_WR1, S_DONE` (the `softmax_unit.v` header counts 22 exclusive of the start edge; the unit TB measures 23 inclusive — both describe the same waveform) |
| ATTENTION seq=4 | **123** | `SETUP(13) + SEQ_LEN*PER_ROW(4*27) + TAIL(2)` = 13 + 108 + 2 = 123.  The unit serially reuses the len-8 `softmax_unit` once per row (four invocations), so the cost is dominated by four serial softmaxes, not a single fused length-4 softmax |

All are bounded constants, so the system TB has fully predictable timing.  Each latency is asserted bit-exactly by the unit's TB (`conv2d_unit_tb.v`, `softmax_unit_tb.v`, `attention_unit_tb.v`).

> Note (FUSE_*_RELU): a fused tensor op (`FUSE_GEMM_RELU` / `FUSE_CONV_RELU`)
> adds a tile-domain RELU post-op that sweeps the result tile one TM line per
> cycle after the compute drains: +4 lines for GEMM C (4×4), +`ceil(od*od/4)`
> lines for the CONV output (9 for the default 6×6), plus the fixed drain-settle
> window.  This is on top of the base GEMM/CONV2D latency above.

---

## 4. Status / control registers

A small memory-mapped status block (read via a `RDSTATUS` scalar op or auto-written to `rC` on tensor `done`):

| Field | Bits | Meaning |
|---|---|---|
| `sat` | 1 | sticky saturation occurred in last tensor op |
| `unit` | 3 | last tensor unit id |
| `argmax` | 4-bit field carrying a 3-bit value | softmax argmax index (0..7); the high bit [7] is structurally 0 for `SM_LEN=8` (the unit emits `argmax[2:0]`, zero-extended to [7:4]) |
| `illegal` | 1 | last decoded illegal opcode |

All status state has a synchronous reset. `sat` is sticky until cleared by writing the status register (a `CLRSTATUS`/`LOADI`-to-status pattern) — this gives verification a hard, observable signal that saturation policy fired, instead of v1.5's silent overflow.

---

## 5. Per-unit microarchitecture

Each unit: synchronous reset on all state; `start`/`busy`/`done` handshake; reads inputs from `TM`, writes outputs to `TM`; full-width internal accumulators; explicit round+saturate at the output narrowing with a `sat` flag. File name == module name (verilator `DECLFILENAME` clean).

### 5.1 `gemm_systolic.v` — output-stationary 4×4 systolic GEMM (replaces matrix_multiply.v)

- **Algorithm:** C = A·B, A,B,C are 4×4 `Q7.8`. A 4×4 grid of PEs, **output-stationary**: each PE holds one `C[i][j]` 48-bit accumulator. A streams in from the west (skewed by row), B streams in from the north (skewed by column); on each of `2N-1` fill cycles plus `N` steady cycles the PEs MAC `a*b` into their accumulator. After draining, each PE rounds+saturates its `Q15.16` accumulator to `Q7.8` and the 16 results are written back to 4 `TM` lines.
- **Interface:** `start, a_base[4:0], b_base[4:0], c_base[4:0] → busy, done, sat`. Inputs/outputs are TM line indices.
- **Sizes:** 16 PEs, each a `Q7.8 × Q7.8 → Q14.16` multiplier + 48-bit add. Latency 11 cycles (§3.3).
- **Keep/rebuild:** REBUILD (v1.5 packed dot-product is not a GEMM). Reuse the `Q7.8` saturating-round helper.
- **Unit test:** independent golden = a `real`-typed nested-loop matmul in the TB (`real` accumulation, then quantize to `Q7.8`); compare every C element. Directed (identity, all-max → saturation, negatives, zeros) + constrained-random A,B with seeded `$random`. The reference uses `real`, the DUT uses fixed-point integer MACs, so they cannot share a bug.

### 5.2 `conv2d_unit.v` — true 2-D convolution 8×8 ∗ 3×3 (replaces convolution_unit.v)

- **Algorithm:** valid-padding 2-D conv, output 6×6. **Line buffers** (2 rows of 8 `Q7.8`) + a **3×3 window register**; slide the window across the feature map, one output pixel/cycle, MAC the 9 window×kernel products in a `Q15.16` 48-bit accumulator, round+saturate to `Q7.8`. Stride and padding are **real instruction fields** now (ISA `imm` carries `stride∈{1,2}`, `pad∈{0,1}`), eliminating v1.5's hardwired stride=1/pad=0 dead path.
- **Interface:** `start, in_base[4:0], k_base[4:0], out_base[4:0], stride[1:0], pad[1:0] → busy, done, sat`.
- **Sizes:** line buffers 2×8×32b; 3×3 = 9 MACs/cycle; 48-bit acc; out 6×6 (valid, stride1,pad0) = 36 px packed **dense-16** (each pixel a 16-bit `Q7.8` at bits `[16*(p%4) +: 16]`, 4 px/line) into 9 `TM` lines. Since 36 is a multiple of 4, all 9 lines are FULL; a partial final line (unused upper lanes written 0) only occurs for non-default sizes (e.g. stride2/pad0 → 9 px → last line partial).
- **Keep/rebuild:** REBUILD (v1.5 is a 1-D 8-tap MAC).
- **Unit test:** independent golden = direct 4-nested-loop `real` conv in the TB; compare all 36 outputs. Directed (impulse kernel → identity, edge windows, stride2) + random feature maps/kernels. Reference computes spatially in `real`; DUT uses line-buffered fixed-point — independent.

### 5.3 `softmax_unit.v` — fixed-point exp-based softmax, len 8 (replaces VectorALU SOFTMAX)

- **Algorithm:** true softmax `p_i = exp(x_i - max)/Σexp(x_j - max)`.
  1. Pass 1: find `max` over 8 logits (numerical stability).
  2. Pass 2: `e_i = EXP(x_i - max)` computed as a **64-entry `exp(-k·0.25)` LUT** (`Q15.16` outputs; the argument `x_i-max ≤ 0` is split into an integer multiple of `0.25` selecting a LUT entry, times a **divide-free degree-4 Maclaurin polynomial** correcting the sub-quantum residual `exp(-r)`) — NOT linear interpolation between entries. LUT entries `k ≥ 48` (i.e. `exp(≤ -12)`) are 0. Accumulate `Σ e_i` in 48-bit.
  3. Reciprocal `1/Σ` via an **exact rounded integer hardware divide** `recip = (2^46 + (S>>1)) / S` in `Q1.30` — NOT Newton-Raphson and NOT a reciprocal LUT.
  4. Pass 3: `p_i = e_i * recip`, round to `Q0.16`, write 2 `TM` lines.
- **Interface:** `start, x_base[4:0], p_base[4:0] → busy, done, sat, argmax[2:0]`. `argmax` is a **3-bit** index (0..7) of the largest logit.
- **Sizes:** 64-entry exp LUT (`Q15.16`, top 16 entries zero), one 48-bit accumulator, one `Q1.30` exact-divide reciprocal. Latency = 23 (see §3.3).
- **Keep/rebuild:** REBUILD (v1.5 was linear normalization, not exponential).
- **Unit test:** independent golden = `real` `exp()` from the C math library via `$exp`/`real` in the TB (iverilog supports `real` `$exp`-like via `$ln`/`$pow`; use `exp(x)=$pow(2.71828…,x)` or a TB `real` exp helper), normalized in `real`, compared against the DUT to a tolerance (±1–2 LSB of `Q0.16`) to account for LUT interpolation error. Directed (all-equal → uniform 1/8, one-hot → ≈1.0, large spread → saturation behavior) + random logits. The reference uses true floating exp; the DUT uses a LUT — the tolerance bounds the documented approximation error and the two share no code.

### 5.4 `attention_unit.v` — true scaled-dot-product attention, seq=4 d=4 (rebuild)

- **Algorithm:** `Attn(Q,K,V) = softmax(Q·Kᵀ / sqrt(d)) · V`, all `Q7.8`.
  1. Scores `S[i][j] = (Σ_d Q[i][d]*K[j][d])` in 48-bit, scaled by `1/sqrt(4)=1/2` (exact `>>1` after rounding) → `S` is 4×4.
  2. Row-wise softmax (reuse `softmax_unit` over each length-4 row) → attention weights `W` 4×4 in `Q0.16`.
  3. Context `O[i][d] = Σ_j W[i][j]*V[j][d]` in 48-bit, round+saturate to `Q7.8` → `O` 4×4, write 4 `TM` lines.
- **Interface:** `start, q_base, k_base, v_base, o_base (each [4:0]) → busy, done, sat`.
- **Sizes:** reuses GEMM-style MAC for Q·Kᵀ and W·V; reuses `softmax_unit` (invoked once per row → four serial len-8 softmaxes). Latency = 123 = `SETUP(13) + SEQ_LEN*PER_ROW(4*27) + TAIL(2)` (see §3.3).
- **Keep/rebuild:** REBUILD. This is the unit that carried the v1.5 truncation bug; v2.0 has explicit 48-bit scores, an explicit documented `>>1` scale, and a full softmax — the bug class is structurally eliminated.
- **Unit test:** independent golden = full `real` attention in the TB (real matmul, real softmax via real exp, real matmul), quantized at the end; compare all 16 context elements to ±tolerance. Directed (V·identity-weights, one dominant key → that value passes through) + random Q/K/V. Reference is pure `real`; DUT is fixed-point + LUT — independent.

### 5.5 Scalar units (mostly KEEP from v1.5, widened)

- `vector_alu.v` — KEEP & EXTEND: RELU, ADD; **add** SUB/AND/OR/XOR/SHL/SHR for real address/control math (needed by LOADI-driven programs). Pure combinational. Drop the v1.5 fake SOFTMAX (now its own real unit).
- `register_file.v` — KEEP & MODIFY: add `r0` hardwired-zero (read returns 0, write ignored). 3 read ports retained.
- `memory.v` (DMEM) — KEEP & WIDEN address to 8 bits, depth 256; keep dual read port for DMA.
- `tile_memory.v` — NEW: 32×128b, combinational read (≥2 read ports for tensor operands), synchronous write, reset-cleared.
- `dma_controller.v` — REBUILD to a real multi-word copy FSM (`len` from imm), `MEM[dst+i]=MEM[src+i]`.
- `scatter_gather.v` — KEEP gather, ADD scatter (store path) so the unit is symmetric; effective address `base + index*stride`, `stride` from imm.
- `fused_ops_unit.v` — KEEP for scalar fusions (e.g. GEMM-done→RELU on the result tile) but re-express as tile-domain post-ops; combinational over already-computed tiles.
- `instruction_decoder.v` — KEEP, extend field extraction for the new immediate/format bits (§ISA).

---

## 6. Verification strategy

**Independence rule (the v1.5 fix):** no golden model may share arithmetic with the DUT. v1.5's attention golden mirrored the DUT's `>>8` truncation and so never caught the overflow. In v2.0 **every tensor golden is a `real`-typed behavioral model** (floating matmul/conv/exp/softmax) quantized only at the boundary, compared to a documented tolerance for the LUT-approximated ops (softmax/attention) and bit-exactly for the exact ops (GEMM/conv with round+saturate, which the reference also applies *as a separate final quantization step*, not as the accumulation method).

Layers:
1. **Per-unit self-checking TBs** (`test/<unit>_tb.v`): each tensor and scalar unit standalone. Directed corner cases (zero, negative, max→saturation, identity) + constrained-random (seeded `$random`, ≥1000 vectors/unit). Assert `done` timing == spec latency, assert `sat` flag fires exactly when the reference saturates.
2. **Hazard/pipeline TB:** back-to-back scalar dependencies (forwarding), load-use stalls, tensor `busy` stalls (tensor op immediately followed by a dependent TLOAD), illegal opcode, `r0`-write-ignored, LOADI seeding (no hierarchical pokes needed).
3. **System integration TB** (`test/tpu_tb.v`, evolves the current one): a small **program** that uses LOADI to seed state, runs a GEMM then a RELU-on-tile then a SOFTMAX then an ATTENTION, and checks final tiles against an end-to-end `real` reference. Covers every opcode and the full datapath. Keeps `$dumpfile/$dumpvars` → real VCD.
4. **Coverage intent:** every opcode issued; every hazard kind (RAW-EX, RAW-MEM, load-use, tensor-busy); edge cases (overflow/saturation, zero, negative, all-equal softmax, one-hot). Aim near-exhaustive for the small tensor sizes (e.g. exhaustive over small int8 ranges for GEMM with bounded random sampling).

**Toolchain gates (must all pass):**
- `iverilog -g2012 -Wall -I src` compiles design + each TB clean.
- `vvp` runs every TB to `ALL N TESTS PASSED` (`$fatal` on any failure).
- `verilator --lint-only -Wall -Isrc <files>` clean (file names == module names to satisfy `DECLFILENAME`, or scoped `lint_off`).
- `yosys -p "read_verilog -sv -I src <files>; hierarchy -top TPU; proc; opt; stat"` elaborates with no latches / no unresolved hierarchy.

---

## 7. Build order (incremental, always-green)

Build bottom-up so the regression is green at every step (never break the passing baseline before the replacement is verified):
1. `tpu_defs.vh` — add all sizes/Q-formats/new opcodes (single source of truth).
2. `tile_memory.v` + its unit TB.
3. Widen `memory.v` (DMEM 256×32) + `register_file.v` (`r0`=0) + their TBs.
4. `gemm_systolic.v` + unit TB (independent `real` golden).
5. `conv2d_unit.v` + unit TB.
6. `softmax_unit.v` (+ exp LUT) + unit TB.
7. `attention_unit.v` (reuses GEMM MAC + softmax) + unit TB.
8. Extend `vector_alu.v`, `dma_controller.v`, `scatter_gather.v`, `instruction_decoder.v`, control decoder; add LOADI + tensor opcodes + `busy` stall in `tpu_top.v`.
9. System `tpu_tb.v` program-driven integration + hazard TB.
10. Makefile targets `build/lint/synth/test/wave/clean`; README rewrite; per-module headers.

Each tensor unit is independently verifiable before integration, so phase-2 parallelization is safe.

---

## 8. Key decisions (summary)

- Keep the v1.5 scalar 5-stage pipeline + forwarding + load-use stall (proven, 24/24). Generalize its fixed stall into a variable `busy` stall for multi-cycle tensor ops.
- Introduce a **tile memory (TM)** as a second operand domain so tensors aren't crammed into 32-bit registers; tensor ops are TM→TM and name tiles by base line index.
- One committed Q-format per op (`Q7.8` data, 48-bit accumulators, `Q0.16` probabilities), round-half-up + saturate with a sticky `sat` flag. No silent narrowing anywhere.
- Real algorithms at small fixed sizes: 4×4 systolic GEMM, 8×8∗3×3 line-buffered conv, len-8 LUT exp-softmax, seq4/d4 full attention.
- Real ISA: 8-bit opcode, immediate path, `LOADI`, hardwired-zero `r0`, illegal-opcode flag (see `docs/ISA.md`).
- Verification independence enforced: all tensor goldens are `real`-typed behavioral models, never sharing the DUT's fixed-point path.

---

## 9. Module parameterization (tensor units)

The four tensor units are **module-parameterized** over their tile sizes. The
`tpu_defs.vh` `localparam` sizes of §1.2 remain the committed defaults; each unit
takes a module `parameter` that **defaults to that size**, so an unparameterized
instance is **byte-identical in behavior** to the original fixed design and the
per-unit TBs still pass with the same assertion counts. Inside each unit every
index, counter width (`$clog2`-derived), loop bound, operand-bank depth, and
packing index is derived from the parameter — the size-specific mod-4 bit
truncations of the original (`tcnt[1:0]`, `{row[1:0],col[1:0]}`, fixed 3-bit
counters) are replaced by parameter-correct equivalents (true subtraction/`row*N+col`
indices, `$clog2`-sized counters), correct for any in-range size, not just powers
of two.

| Unit | Module parameter(s) | Default | Supported range | Reason for the bound |
|---|---|---|---|---|
| `gemm_systolic.v` | `N` | `GEMM_N` = 4 | `2 ≤ N ≤ LINE_LANES` (= 4) | upper: one row = one TM line of 4 lanes; lower: keep `$clog2(N) ≥ 1` |
| `conv2d_unit.v` | `IMG_H`, `IMG_W`, `K` | 8, 8, 3 | `IMG_W ≤ 8`, `K ≤ 8`, `K ≤ IMG_H` and `K ≤ IMG_W` | one image/kernel row packs dense-16 into one 128-bit line (8 cols); valid output dims ≥ 1 |
| `softmax_unit.v` | `LEN` | `SM_LEN` = 8 | `2 ≤ LEN ≤ 8·LINE_LANES` (= 32) | 1-D reduction over `ceil(LEN/4)` lines; lower: len-1 is degenerate |
| `attention_unit.v` | `SEQ`, `D` | 4, 4 | `D ≤ LINE_LANES` (= 4), `SEQ ≤ SM_LEN` (= 8), `SEQ,D ≥ 1` | one Q/K/V/O row = one TM line (4 lanes); a score row runs through the softmax submodule padded to `SM_PAD = max(SEQ, SM_LEN)` |

**The architectural bound.** All upper bounds trace to **`LINE_LANES = 4`** (§1.2):
the architecture packs one matrix/feature/Q-K-V row into **one 128-bit TM line of 4
lanes**, baked into `tpu_defs.vh` / `tile_memory.v` / `tpu_top.v`. So GEMM `N` and
attention `D` cannot exceed 4, and a conv image/kernel row cannot exceed 8 columns
(4 lanes × 16-bit dense pack). **softmax** is the exception: as a 1-D reduction it
spans `ceil(LEN/4)` TM lines and so scales to `LEN ≤ 32`, but its `argmax` status
port stays a 3-bit architectural field (§4) — exact for `LEN ≤ 8`, low-3-bits
truncated above. Going beyond these bounds (e.g. GEMM `N > 4`) requires a
**multi-line-per-row TM re-architecture** (a TM port / packing change) and is out of
scope for these units (see `docs/ROADMAP.md`).

**2nd-size proofs.** Each unit is verified at a second in-range size against an
independent `real`-typed golden, in addition to its exhaustive default-size TB:
`gemm_systolic_tb` (N=2, N=3), `test/conv2d_param_tb.v` (6×6∗3 → 4×4),
`test/softmax_param_tb.v` (LEN=4 and LEN=16), `test/attention_param_tb.v` (SEQ=2,
D=2). `make unittests` builds and runs all of them.

---

## 10. AXI4-Lite slave wrapper (`src/tpu_axi.v`)

`src/tpu_axi.v` (module `tpu_axi`) wraps the verified `TPU` core (module `TPU`,
unchanged) as a drop-in, SoC-integratable **AXI4-Lite SLAVE** IP. The core is
instantiated and wrapped — **never edited**.

**Clock / reset.** Single clock domain: `ACLK` drives both the AXI slave logic and
the core. `ARESETn` is AXI active-LOW; the core's `rst` is synchronous active-HIGH,
so `core.rst = ~ARESETn`. `ARESETn` is assumed synchronous to `ACLK` (standard AXI
assumption); all wrapper state resets synchronously. This is a **single-clock**
wrapper — there is no CDC (see `docs/ROADMAP.md` for the deferred multi-clock case).

**Issue model (single-step).** The wrapper owns the core's `instruction_in`,
mirroring the testbench's one-instruction-per-cycle `step()`. A host:
1. writes the **`INSTR`** register (the 32-bit instruction word, WSTRB-honoured RMW), then
2. writes **`CTRL.STEP`** (bit0), which drives the `INSTR` word onto
   `instruction_in` for **exactly one `ACLK` cycle** — the cycle the W-channel write
   commits (`BVALID` asserted). On every other cycle `instruction_in` is forced to
   `OP_NOP` (`0x00`).

The host runs any program by issuing INSTR-then-STEP per instruction in sequence and
reading `RESULT`/`STATUS` back after the pipeline latency (5 stages + any tensor
stall). The core's `result_out` and `illegal_opcode` are continuously mirrored into
the read-only `RESULT`/`STATUS` registers; `STATUS.illegal` is sticky and cleared via
`CTRL.CLR_ILL` (bit1).

**Register map** (32-bit registers; byte address = word offset × 4):

| Byte | Word | Name | Access | Meaning |
|---|---|---|---|---|
| `0x00` | 0 | `CTRL`/`STEP` | W / R | W: bit0 `STEP` → issue `INSTR` for one `ACLK`; bit1 `CLR_ILL` → clear sticky illegal. R: bit0 `STEP_DONE`, bit1 `LAST_ILL` (illegal seen on last step) |
| `0x04` | 1 | `INSTR` | RW | the 32-bit instruction word to issue |
| `0x08` | 2 | `RESULT` | RO | mirror of core `result_out` (committed WB word) |
| `0x0C` | 3 | `STATUS` | RO | bit0 sticky `illegal_opcode`, bit1 live `illegal_opcode` (this cycle), bit2 `STEP_DONE`, [31:3] reserved (read 0) |

**Protocol.** 32-bit data; 4-bit `WSTRB` byte strobes honoured on register writes.
Five channels AW/W/B/AR/R with **registered** VALID/READY handshakes (no
combinational `AWREADY↔AWVALID`/`ARREADY↔ARVALID` loops); a single outstanding
transaction (legal for AXI4-Lite). A write commits only when **both** AW and W have
handshaked (in either order). Mapped accesses return `OKAY` (`2'b00`); unmapped word
offsets read 0 / drop writes and return `SLVERR` (`2'b10`).

**Synthesis & test.** Synchronous reset on every state element, no inferred latch, no
combinational loops; passes `verilator --lint-only -Wall` (top `tpu_axi`) and yosys
`check -assert` (top `tpu_axi`). `make axi` builds and runs the BFM testbench
(`test/tpu_axi_tb.v`), which drives a real program through the bus channels and
checks it against independent goldens.

**Scope.** This is a **slave-only** wrapper: it cannot autonomously fetch from
external memory. An AXI **master** path (DMA from external DRAM) is a deferred v3 item
(`docs/ROADMAP.md`).
