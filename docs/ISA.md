# TPU v2.0 — Instruction Set Architecture Reference

Companion to [`../SPEC.md`](../SPEC.md). This document is normative for **encodings, register model, and illegal-opcode behavior**. Sizes/Q-formats/microarchitecture are in `SPEC.md`. All opcodes and field positions live in a single source of truth: `src/tpu_defs.vh`.

---

## 1. Instruction word format

Fixed 32-bit instructions. There are two formats selected by opcode (the decoder extracts all fields unconditionally; each instruction documents which it uses).

### R-format (register operands)
```
 31      24 23   20 19   16 15   12 11                0
+----------+------+------+------+--------------------+
|  opcode  |  rA  |  rB  |  rC  |      imm12         |
|   [8]    | [4]  | [4]  | [4]  |       [12]         |
+----------+------+------+------+--------------------+
```
- `rA`, `rB`: source scalar registers (or tile-base index sources for tensor ops).
- `rC`: destination scalar register (or status destination for tensor ops).
- `imm12`: immediate operand / tile-base index / op modifier (stride, pad, len, value-select), depending on opcode.

### I-format (load-immediate)
```
 31      24 23   20 19                              0
+----------+------+--------------------------------+
|  opcode  |  rC  |            imm20               |
|   [8]    | [4]  |             [20]               |
+----------+------+--------------------------------+
```
- Used **only** by `LOADI`. `rC` = destination; `imm20` = 20-bit sign-extended immediate written to `rC`. This is the seed path: programs set up addresses, tile-base indices, loop counts, and small constants **without** hierarchical testbench pokes (eliminating v1.5's `seed_reg`/`seed_mem` dependence for program setup).

To load a full 32-bit constant, use `LOADI` (low 20 bits, sign-extended) followed by `LUI`/`ORI`-style composition if needed; for the small tile-base indices (5-bit) and addresses (8-bit) used by this core, a single `LOADI` suffices.

---

## 2. Register model

### Scalar register file `RF`
- 16 registers `r0`..`r15`, each 32-bit.
- **`r0` is hardwired to zero** (rationale below). Reads of `r0` always return `0x00000000`; writes to `r0` are silently discarded (write-enable masked when `rC==0`). This gives a guaranteed constant zero for address bases, comparisons, and `NOP`-by-`ADD r0,r0,r0`, and makes "discard result" trivial (`rC=r0`).
  - **Rationale for hardwired-zero (resolving the v1.5 open question):** a constant-zero register is the standard RISC convention; it removes the need for a separate "ignore write" mechanism, simplifies the load-use hazard logic (a producer with `rC==0` can be treated as a non-producer), and gives software a free zero for the new immediate/address math. v1.5 made `r0` a normal register and documented the omission as a limitation; v2.0 closes it.
- 3 combinational read ports (rA, rB, and a third port for the immediate-selected operand used by some ops). 1 synchronous write port (WB). Synchronous reset clears all to 0.

### Tile memory `TM`
- 32 lines × 128 bits (4 × 32-bit lanes). Not directly named by register index; tensor instructions name a **tile base line** via `imm` or via a scalar register holding the index. Tile extents are fixed per op (`SPEC.md` §1.2).

### Data memory `DMEM`
- 256 words × 32-bit. Addressed by an 8-bit address from a scalar register or immediate.

### Status register
- See `SPEC.md` §4: `{sat, unit, argmax, illegal}`. Read with `RDSTATUS`, cleared by writing it. `sat` is sticky.

---

## 3. Opcode map

Opcodes are 8-bit; grouped by class. Every opcode below is defined in `src/tpu_defs.vh`.

| opcode | name | fmt | operation (semantics) | writes |
|---|---|---|---|---|
| `0x00` | `NOP` | R | no operation | none |
| `0x01` | `LOADI` | I | `rC = sign_extend(imm20)` | `rC` |
| `0x02` | `LOAD` | R | `rC = DMEM[ rA + imm12 ]` (8-bit eff addr) | `rC` |
| `0x03` | `STORE` | R | `DMEM[ rA + imm12 ] = rB` | none |
| `0x04` | `ADD` | R | `rC = rA + rB` | `rC` |
| `0x05` | `SUB` | R | `rC = rA - rB` | `rC` |
| `0x06` | `AND` | R | `rC = rA & rB` | `rC` |
| `0x07` | `OR` | R | `rC = rA \| rB` | `rC` |
| `0x08` | `XOR` | R | `rC = rA ^ rB` | `rC` |
| `0x09` | `SHL` | R | `rC = rA << rB[4:0]` | `rC` |
| `0x0A` | `SHR` | R | `rC = rA >> rB[4:0]` (logical) | `rC` |
| `0x0B` | `RELU` | R | `rC = (signed rA < 0) ? 0 : rA` | `rC` |
| `0x0C` | `ADDI` | R | `rC = rA + sign_extend(imm12)` | `rC` |
| `0x0D` | `RDSTATUS` | R | `rC = status_register` | `rC` |
| `0x0E` | `CLRSTATUS` | R | clear sticky `sat`/`illegal` | none |
| `0x10` | `DMA` | R | `for i in 0..len-1: DMEM[rB+i] = DMEM[rA+i]`; `rC = #words copied`. `len = imm12` **saturated to `[0, DMEM_DEPTH=256]`** (a 256-word DMEM bounds a meaningful copy; an over-long encoding clamps to 256, it does not wrap) | `rC`, `DMEM` |
| `0x11` | `GATHER` | R | `rC = DMEM[ rA + (rB << imm12[1:0]) ]` (stride = `1<<imm12[1:0]`) | `rC` |
| `0x12` | `SCATTER` | R | `DMEM[ rA + (rB << imm12[1:0]) ] = ` value in `rC`-source | `DMEM` |
| `0x20` | `TLOAD` | R | `TM[ imm12[4:0] ] = { DMEM[rA+3..rA] }` (load 128-bit line) | `TM` |
| `0x21` | `TSTORE` | R | `{ DMEM[rA+3..rA] } = TM[ imm12[4:0] ]` | `DMEM` |
| `0x30` | `GEMM` | R | `C = A·B`, 4×4 `Q7.8`; A base=`rA[4:0]`, B base=`rB[4:0]`, C base=`imm12[4:0]`; `rC = status` | `TM`, `rC` |
| `0x31` | `CONV2D` | R | 8×8 ∗ 3×3 conv; in base=`rA[4:0]`, kernel base=`rB[4:0]`, out base=`imm12[9:5]`, `stride=imm12[1:0]`, `pad=imm12[3:2]`; `rC = status` | `TM`, `rC` |
| `0x32` | `SOFTMAX` | R | true exp-softmax over len-8 vector at base `rA[4:0]` → probs at base `imm12[4:0]`; `rC = status_word` (the argmax is a field **inside** that word at bits `[7:4]`; there is no separate {status,argmax} tuple) | `TM`, `rC` |
| `0x33` | `ATTENTION` | R | `softmax(Q·Kᵀ/√d)·V`, seq=4 d=4; Q base=`rA[4:0]`, K base=`rB[4:0]`, V base=`imm12[4:0]`, O base=**the VALUE of `rC` read at dispatch**; `rC = status`. **NOTE:** `rC` is overloaded — its register VALUE supplies the O tile base on issue and the SAME architectural register is OVERWRITTEN with the status word on completion, so a program must not assume `rC` is preserved across an ATTENTION | `TM`, `rC` |
| `0x40` | `FUSE_GEMM_RELU` | R | `GEMM` then elementwise RELU on the full C tile (4 lines, in place) | `TM`, `rC` |
| `0x41` | `FUSE_CONV_RELU` | R | `CONV2D` then dense-16 RELU on the **full** output tile (all `ceil(od*od/4)` lines, e.g. 9 for the default 6×6) | `TM`, `rC` |
| others | — | — | **illegal** (see §5) | none |

Notes:
- Tensor ops (`0x30`–`0x41`) take **tile-base line indices**, not data, as operands — a 32-bit register cannot hold a tile. Where an operand is an immediate tile index, the relevant `imm12` sub-field is documented above. Where a base comes from a register, the low 5 bits of that register are the line index.
- `LOADI` is the program seed path; combined with `TLOAD`/`TSTORE` a program loads tiles from `DMEM` into `TM` and back with no testbench hierarchical access.
- `RELU`/fused post-ops use the documented `Q7.8` saturation.

---

## 4. Immediate path (closing the v1.5 gap)

v1.5 had `[11:0]` mostly unused (only ATTENTION used `imm[3:0]` to select a value register). v2.0 gives the immediate a first-class role:
- `LOADI`/`ADDI`: arithmetic immediate (`imm20`/sign-extended `imm12`).
- `LOAD`/`STORE`: address displacement.
- `GATHER`/`SCATTER`: stride exponent (`imm12[1:0]`).
- `DMA`: copy length.
- Tensor ops: tile-base indices and op modifiers (`stride`, `pad`) packed into `imm12`.

This makes every instruction's immediate field meaningful and lets self-contained test programs run from instruction memory alone.

---

## 5. Illegal-opcode behavior

Any opcode not in §3 is **illegal**:
- The combinational decoder raises `illegal_opcode` (and sets the status `illegal` bit) during the instruction's ID stage.
- The instruction executes as a **safe NOP**: no register write, no memory write, no tile write, no status corruption beyond the `illegal` flag.
- This matches and extends v1.5's behavior (which already flagged + NOP'd unknown opcodes), now also covering malformed tensor ops.

Reserved opcode ranges (`0x14`–`0x1F`, `0x22`–`0x2F`, `0x34`–`0x3F`, `0x42`+) are illegal until assigned, so the decoder's `default: illegal` arm is the single point of truth.

---

## 6. Encoding example (a complete program)

A program that loads two 4×4 tiles, multiplies them, RELUs, and stores the result — no hierarchical pokes:

```
LOADI  r1, 0          ; r1 = DMEM base of matrix A
LOADI  r2, 16         ; r2 = DMEM base of matrix B
TLOAD  -, r1, _, imm=0..3   ; TM[0..3] = A (4 lines)   (4 TLOADs)
TLOAD  -, r2, _, imm=4..7   ; TM[4..7] = B
GEMM   rA=0, rB=4, rC=r5, imm=8   ; TM[8..11] = A·B ; r5=status
FUSE_GEMM_RELU ...    ; (or separate RELU pass on the C tile)
TSTORE r3, _, imm=8..11     ; DMEM[...] = C
RDSTATUS r6           ; r6 = status (check sat bit)
```

The system testbench (`test/tpu_tb.v`) drives exactly this style of program and checks the resulting `TM`/`DMEM` against an independent `real`-typed reference (see `SPEC.md` §6).
