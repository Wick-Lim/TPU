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
make unittests  # build+run every per-unit TB            -> ALL N TESTS PASSED each
make lint       # verilator --lint-only -Wall on the whole design -> clean
make synth      # yosys elaborate/synth gate (no error, no latch)
make wave       # run system TB, leave ./tpu_waveform.vcd (a real VCD)
make clean      # remove build/ and generated *.vcd
```

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
