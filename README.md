# TPU v2.0 — a small, real, self-checking tensor-processing core (Verilog)

A synthesizable Verilog model of a tensor/AI accelerator: a classic **5-stage
scalar pipeline** (IF → ID → EX → MEM → WB) with full data-hazard forwarding and
load-use stalls, augmented with a **variable-length busy stall** that hands
multi-cycle tensor instructions off to dedicated compute units operating on real
tensor tiles in an on-chip **tile memory**.

This is v2.0: every "tensor" EX unit is a **real algorithm on real tiles** with
full-width fixed-point arithmetic and an explicit, documented Q-format contract —
it supersedes the v1.5 toy core (which packed "matrices" into single 32-bit words).

The normative documents are:
- **[`SPEC.md`](SPEC.md)** — microarchitecture, sizes, fixed-point contract, per-unit design.
- **[`docs/ISA.md`](docs/ISA.md)** — instruction encodings, register model, illegal-opcode behavior.
- **[`src/tpu_defs.vh`](src/tpu_defs.vh)** — the single source of truth for opcodes, sizes, field positions, and the shared rounding/saturation macros.

---

## What the core is

Two architecturally distinct operand domains:

| Domain | Storage | Used by |
|---|---|---|
| **Scalar** | register file `RF` (16 × 32-bit, `r0` hardwired 0) + data memory `DMEM` (256 × 32-bit) | control/address/ALU math, LOADI, loads/stores, DMA, gather/scatter |
| **Tensor** | tile memory `TM` (32 × 128-bit lines = 4 × 32-bit lanes) | GEMM, CONV2D, SOFTMAX, ATTENTION, fused RELU post-ops |

Tensor instructions are **TM → TM**: they name tiles by a 5-bit base line index
(from a register or an immediate), read operands directly out of `TM`
combinationally, run a per-unit FSM, and write result tile(s) back through a
dedicated `TM` write port. On completion they retire a scalar **status word** to
`rC` (saturation / unit id / argmax) so software can branch on it.

### Real algorithms (committed default sizes)

| Unit | Algorithm | Size | Q-format | start→done latency |
|---|---|---|---|---|
| `gemm_systolic.v` | output-stationary systolic GEMM `C=A·B` | 4×4 | `Q7.8` data, `Q15.16` 48-bit accum | **11** |
| `conv2d_unit.v` | true 2-D line/window-buffered convolution | 8×8 ∗ 3×3 (valid → 6×6) | `Q7.8`, `Q15.16` accum | **48** = 11 load + 36 compute + 1 done |
| `softmax_unit.v` | exp-softmax (64-entry `exp(-k·0.25)` LUT + degree-4 Maclaurin residual; exact integer-divide reciprocal in `Q1.30`) | len 8 | `Q7.8` in, `Q0.16` probs | **23** (inclusive of start edge) |
| `attention_unit.v` | scaled-dot-product attention `softmax(Q·Kᵀ/√d)·V` (reuses `softmax_unit` per row) | seq=4, d=4 | `Q7.8`, 48-bit accums | **123** = SETUP 13 + 4×27 + TAIL 2 |
| `fused_ops_unit.v` | tile-domain elementwise RELU post-op (packing-aware: 32-bit lane for GEMM, dense-16 for CONV) | one TM line/cycle | `Q7.8` | combinational; sweeps the full result tile |

All narrowing is **round-half-up then saturate**, with a sticky `sat` flag in the
status register — there is no silent truncation. The shared narrowing macros live
in `tpu_defs.vh` so no two units can diverge on the arithmetic contract.

### Pipeline & hazards

- Combinational RF read (3 ports, `r0=0`); combinational DMEM read (dual-port for DMA); synchronous writes; synchronous reset on all state.
- A control decoder emits per-opcode `writes_reg / mem_read / mem_write / uses_tensor` (never `opcode != 0`).
- Full operand **forwarding** from EX, EX/MEM, MEM/WB into the ID operand path; an EX-stage tensor producer forwards its **status word** (not the raw eff-addr) to a directly-dependent consumer.
- One-cycle **load-use stall** for LOAD/GATHER/DMA producers; the rC-consumer set covers SCATTER **and ATTENTION** (whose `o_base` is read from `rC`).
- Multi-cycle tensor ops freeze IF/ID/EX on a **busy stall**; the shared `TM` port is held by a draining scalar `TLOAD/TSTORE` until it fully retires (the tensor dispatch FSM waits in IDLE), preserving one-live-TM-consumer.
- Illegal opcode → `illegal_opcode` flag + sticky status bit + safe NOP (no architectural side effects).

---

## Parameterized tensor units

The four tensor units are **module-parameterized** over their tile sizes, with the
defaults set to the committed `tpu_defs.vh` sizes so that an unparameterized
instance is **byte-identical in behavior** to the original fixed design (the
per-unit TBs still pass with the same assertion counts). Every internal index,
counter width (`$clog2`-derived), loop bound, and packing index is derived from
the parameter — no hardcoded size literals or mod-4 bit tricks remain.

| Unit | Module parameter(s) | Default | Supported range | 2nd-size proof |
|---|---|---|---|---|
| `gemm_systolic.v` | `N` | `GEMM_N`=4 | `2 ≤ N ≤ 4` | `gemm_systolic_tb` runs N=2/N=3 |
| `conv2d_unit.v` | `IMG_H`, `IMG_W`, `K` | 8/8/3 | `IMG_W,K ≤ 8`; `K ≤ IMG_H,IMG_W` | `test/conv2d_param_tb.v` (6×6∗3 → 4×4) |
| `softmax_unit.v` | `LEN` | `SM_LEN`=8 | `2 ≤ LEN ≤ 32` | `test/softmax_param_tb.v` (LEN=4 and LEN=16) |
| `attention_unit.v` | `SEQ`, `D` | 4/4 | `D ≤ 4`, `SEQ ≤ 8` | `test/attention_param_tb.v` (SEQ=2/D=2) |

> **Supported-range caveat (the architectural bound).** Per-row packing is bounded
> by **`LINE_LANES = 4`**: one matrix/feature/Q-K-V row is packed into **one
> 128-bit TM line of 4 lanes**, baked into `tpu_defs.vh` / `tile_memory.v` /
> `tpu_top.v`. So GEMM `N` and attention `D` cannot exceed 4, and a conv image/kernel
> row cannot exceed 8 columns (4 lanes × 16-bit dense pack). softmax is a 1-D
> reduction that spans `ceil(LEN/4)` lines, so it scales to `LEN ≤ 32`, but its
> `argmax` status port stays a 3-bit architectural field (exact for `LEN ≤ 8`).
> **Arbitrary sizes beyond these bounds need a multi-line-per-row TM
> re-architecture** (a TM port / packing change) and are out of scope for these
> units. The 2nd-size proofs above each check an in-range size against an
> independent `real` golden; `make unittests` builds and runs them.

---

## AXI4-Lite wrapper (`src/tpu_axi.v`)

`src/tpu_axi.v` (module `tpu_axi`) is an **AXI4-Lite SLAVE** wrapper that makes the
verified core a drop-in, SoC-integratable IP. **The core is wrapped, never edited.**

- **Single clock domain.** `ACLK` drives both the AXI slave logic and the core;
  `ARESETn` is active-LOW (`core.rst = ~ARESETn`), assumed synchronous to `ACLK`.
- **Single-step issue model.** The wrapper owns the core's `instruction_in`. A host
  writes the `INSTR` register, then writes `CTRL.STEP`, which drives `INSTR` onto
  `instruction_in` for **exactly one `ACLK` cycle** (the cycle the W-channel write
  commits); every other cycle `instruction_in` is forced to `NOP`. A program runs
  by issuing INSTR-then-STEP per instruction in sequence and reading `RESULT`/`STATUS`
  back after the pipeline latency.
- **Five channels** AW/W/B/AR/R with registered VALID/READY handshakes (no
  combinational handshake loops); single outstanding transaction.

Register map (32-bit registers; byte address = word offset × 4):

| Byte | Name | Access | Meaning |
|---|---|---|---|
| `0x00` | `CTRL`/`STEP` | W | bit0 `STEP` → issue `INSTR` for one cycle; bit1 `CLR_ILL` → clear sticky illegal. R: bit0 `STEP_DONE`, bit1 `LAST_ILL` |
| `0x04` | `INSTR` | RW | the 32-bit instruction word to issue (WSTRB-honoured RMW) |
| `0x08` | `RESULT` | RO | mirror of core `result_out` (committed WB word) |
| `0x0C` | `STATUS` | RO | bit0 sticky `illegal_opcode`, bit1 live `illegal_opcode`, bit2 `STEP_DONE` |

Mapped accesses return `OKAY` (`2'b00`); unmapped offsets read 0 / drop writes and
return `SLVERR` (`2'b10`). `make axi` builds and runs the BFM testbench
(`test/tpu_axi_tb.v`), which drives a real program through the AW/W/B/AR/R channels
and checks it against independent goldens → `ALL N TESTS PASSED`.

---

## PPA flow (`make ppa`, [`docs/PPA.md`](docs/PPA.md))

`make ppa` runs **Yosys `synth_ecp5`** to map each tensor unit (and the full TPU
top) to real Lattice **ECP5** FPGA primitives and reports:

- **Area** — measured per-primitive cell counts (LUT4, TRELLIS_FF, MULT18X18D DSP,
  DP16KD BRAM, CCU2C carry, mux primitives) from the post-synth `stat`.
- **Timing proxy** — `ltp` longest topological **logic-cell depth** for the four
  tensor units (a structural depth metric, **not** a routed delay; the full TPU top
  is area-only to keep the run fast).

Filtered per-unit logs land in `build/ppa/<unit>.log`. **Area is measured; routed
`fmax`/power are NOT** — those need place-and-route (`nextpnr-ecp5`), which is not
installed here. The deepest combinational path is `attention_unit` (ltp 2250) >
`softmax_unit` (1580), which is the actionable signal for the future pipelining
work. See [`docs/PPA.md`](docs/PPA.md) for the full report, methodology, and honesty
notes, and [`docs/ROADMAP.md`](docs/ROADMAP.md) for what is intentionally deferred.

---

## ISA summary

8-bit opcode, two 32-bit instruction formats (R-format and the I-format `LOADI`).
Full table, semantics, and field positions are in **[`docs/ISA.md`](docs/ISA.md)**;
the encodings are defined once in **[`src/tpu_defs.vh`](src/tpu_defs.vh)**.

Classes: scalar/control (`NOP, LOADI, LOAD, STORE, ADD/SUB/AND/OR/XOR/SHL/SHR,
RELU, ADDI, RDSTATUS, CLRSTATUS`), memory-movement (`DMA, GATHER, SCATTER`),
tile transfer (`TLOAD, TSTORE`), tensor compute (`GEMM, CONV2D, SOFTMAX,
ATTENTION`), fused (`FUSE_GEMM_RELU, FUSE_CONV_RELU`). Any unassigned opcode is
illegal.

Programs seed all state through `LOADI` + `STORE`/`TLOAD` — there are **no
hierarchical testbench pokes** for program-visible state.

---

## Toolchain

```sh
# Icarus Verilog (sim), Verilator (lint), Yosys (synth check), GTKWave (optional)
brew install icarus-verilog verilator yosys
brew install gtkwave        # optional, waveform viewer
```

## Build / test / lint / synth / wave (Makefile)

```sh
make build      # compile design + system TB           -> build/tpu_sim
make test       # build, run system integration TB      -> ALL N TESTS PASSED
make hazard     # build, run hazard/pipeline TB          -> ALL N TESTS PASSED
make unittests  # build+run every per-unit TB + the 4 param 2nd-size proofs
                #   (conv2d_param, attention_param, softmax_param) + the AXI BFM TB
make axi        # build+run the AXI4-Lite BFM TB on src/tpu_axi.v -> ALL N TESTS PASSED
make ppa        # yosys synth_ecp5 area (+ ltp logic-depth) per unit -> build/ppa/*.log
make lint       # verilator --lint-only -Wall on the whole design -> clean
make synth      # yosys elaborate/synth gate (no error, no latch)
make wave       # run system TB, leave ./tpu_waveform.vcd (a real VCD)
make clean      # remove build/ and generated *.vcd
```

`make all` runs `test hazard unittests lint synth`. The `axi` and `ppa` targets
are separate gates (`unittests` itself also builds+runs the AXI BFM TB and the
three parameterized 2nd-size proofs).

Each TB is self-checking: on success it prints `ALL N TESTS PASSED`; on any
mismatch it prints the failing case and exits non-zero via `$fatal`.

## Raw commands (canonical compile)

`-o` names the **simulation binary**, not the VCD; the VCD is produced at run
time by `$dumpfile`/`$dumpvars`.

```sh
mkdir -p build
iverilog -g2012 -Wall -I src -o build/tpu_sim \
    test/tpu_tb.v \
    src/tpu_top.v src/instruction_decoder.v src/register_file.v src/memory.v \
    src/tile_memory.v src/vector_alu.v src/dma_controller.v src/scatter_gather.v \
    src/gemm_systolic.v src/conv2d_unit.v src/softmax_unit.v src/attention_unit.v \
    src/fused_ops_unit.v
vvp build/tpu_sim                # runs the sim, writes ./tpu_waveform.vcd

# lint and synth gates
verilator --lint-only -Wall -Isrc --top-module TPU src/*.v
yosys -q -p "read_verilog -sv -I src src/*.v; hierarchy -top TPU -check; proc; opt; check -assert; stat"

# verify a real VCD was written
grep -c '^#' tpu_waveform.vcd    # number of timestamps, must be > 0
gtkwave tpu_waveform.vcd         # optional: inspect the waveform
```

## Verification

- **Per-unit self-checking TBs** (`test/<unit>_tb.v`): each scalar and tensor unit
  standalone, with an **independent `real`-typed golden model** (floating
  matmul/conv/exp/softmax/attention) computed differently from the DUT, plus
  directed corner cases and constrained-random sweeps. The golden never shares
  the DUT's fixed-point path, so the two cannot share a bug.
- **Hazard/pipeline TB** (`test/hazard_tb.v`): back-to-back RAW forwarding,
  load-use stalls, tensor busy stalls, the ATTENTION `o_base` load-use stall,
  illegal-opcode handling, `r0`-write-ignored, LOADI seeding.
- **System integration TB** (`test/tpu_tb.v`): a program-driven end-to-end run
  covering every opcode against an end-to-end `real` reference, leaving a real VCD.

---

## Scope & limitations (honest)

- **Small fixed sizes by design.** 4×4 GEMM, 8×8∗3×3 conv, len-8 softmax, seq=4/d=4
  attention. These are small enough for near-exhaustive directed + random
  verification yet each exercises the *full* algorithm (a genuine systolic array,
  genuine line/window buffers, a genuine LUT exp, a real Q·Kᵀ→softmax→·V). They
  are not tuned for throughput.
- **One committed Q-format per op** (`Q7.8` data, 48-bit accumulators, `Q0.16`
  probabilities). softmax/attention use a LUT-based exp, so their unit TBs compare
  to a small documented tolerance (±1–2 LSB); GEMM/CONV/FUSE are bit-exact.
- **No instruction memory / branches / interrupts.** The core executes the
  instruction presented on `instruction_in` each cycle (the TB acts as a fetch
  unit that freezes on `dbg_pipe_stall`). There is no PC, branch, or exception
  model beyond illegal-opcode → safe NOP.
- **`DMEM` is 256 words.** `DMA` length is encoded in `imm12` but **saturates to
  256** (it does not wrap); copies longer than the memory are meaningless.
- **`ATTENTION` overloads `rC`**: its value supplies the `o_base` at dispatch and
  is then overwritten by the status word on retire — do not assume `rC` survives
  an ATTENTION (documented in `docs/ISA.md` §3).
- **Tile memory has no ECC / no banking.** A single shared 2R+1W port is
  arbitrated so exactly one consumer (a tensor unit, or a scalar `TLOAD/TSTORE`)
  is live at a time.

The v1.5 toy units (`matrix_multiply.v` / `convolution_unit.v`) and their packed-
scalar "tensor" ops have been **removed**; consult `SPEC.md` / `docs/ISA.md` for
the current architecture.
