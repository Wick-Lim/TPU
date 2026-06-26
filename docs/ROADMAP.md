# TPU Roadmap — what is intentionally NOT done yet (toward v3)

This document is the **honest counterpart** to the README/SPEC: it lists the things
the current design (v2.0 core + parameterization + AXI4-Lite slave + PPA flow)
**deliberately does not do**, and why each was scoped out. None of these are bugs or
oversights — each is a bounded next step with a stated reason for deferral. The aim
is **no overclaiming**: the items below are *not* implemented and *no* result is
claimed for them.

The normative state of what *is* done lives in [`../README.md`](../README.md),
[`../SPEC.md`](../SPEC.md), [`ISA.md`](ISA.md), and the measured PPA in
[`PPA.md`](PPA.md). This file only covers the gaps.

---

## 1. Pipeline the deep combinational tensor blocks

**Status: STARTED — the #1 path (softmax reciprocal divide) is PIPELINED and
re-measured; the remaining tensor paths use the same pattern.**

The PPA characterization ([`PPA.md`](PPA.md) §3–4) measured the longest topological
logic-cell depth (`ltp`, a structural proxy, not a routed delay) of each tensor
unit. The ranking is:

```
attention_unit (2250)  >  softmax_unit (1580)  >  gemm_systolic (262)  >  conv2d_unit (95)
```

**Real place & route confirmed this** ([`PPA.md`](PPA.md) §3.1, nextpnr-ecp5): the
single-cycle **64-bit reciprocal divide** in `softmax_unit` routed at only **~3.4
MHz**. ✅ **It is now pipelined** into a multi-cycle radix-2 restoring sequential
divider (bit-exact quotient), lifting softmax to **~20.9 MHz (6.1×)** — the divide is
gone from the critical path, all values are unchanged, and the cost is latency
(softmax 23→71, attention 87→279, asserted exactly by the TBs). `gemm_systolic` (~28
MHz, the wide MAC) and the attention QK^T/×V reductions are the **same pattern** and
the next targets.

The tensor units are **single deep combinational datapaths** per FSM stage — the
softmax exp/reciprocal approximation chain, the attention QK^T reduction + softmax
reuse + ×V matmul, and the GEMM accumulation/carry chain are all unpipelined. The
actionable finding is to **register these chains into multiple pipeline stages**,
attacking the **softmax reciprocal divide first** (it routes worst, ~3.4 MHz, and
`attention_unit` reuses softmax), then the attention QK^T/×V paths, then the
`gemm_systolic` accumulation tree; `conv2d_unit` (depth 95) is shallow and not a
priority.

**Why deferred.** Adding pipeline stages **changes the cycle-accurate latency** of
each block: results appear N cycles later and the `start`/`busy`/`done` handshake
timing shifts. The current per-unit TBs assert `done` on specific cycles and check
results bit-exactly, so they would all have to be rewritten alongside the RTL. That
is a coordinated RTL + testbench change, out of scope for the characterization-only
PPA work (which changed no RTL, no TB, no Makefile — see [`PPA.md`](PPA.md) §4.2). No
throughput/`fmax` improvement is being claimed; pipelining is a recommendation, not a
result.

---

## 2. AXI MASTER for autonomous DMA from external memory ✅ DONE

**Status: SHIPPED.** `src/axi_master_dma.v` is an AXI4-Lite **master** DMA engine:
a command (`start`/`ext_addr`/`len`/`dir`) makes it drive `AW/W/B` (writes) and
`AR/R` (reads) with registered handshakes, one outstanding transaction, word-aligned
increments, and a sticky `err` on non-OKAY responses, exposing an internal sink
(READ) / source (WRITE) stream. It is verified against an AXI4-Lite slave-memory BFM
(wait states, in/out-of-range) — `make unittests` → `[axi_master_dma] ALL 7 TESTS`.

`src/tpu_soc.v` integrates it: on a host `DMA_CTRL.START` it autonomously **fetches a
program from external memory** over the master port and runs it on the core — no
host single-stepping. See item below for the full two-clock SoC.

---

## 3. Multi-clock CDC ✅ DONE

**Status: SHIPPED.** `src/cdc_async_fifo.v` is a textbook dual-clock async FIFO
(gray-code pointers crossed via 2-FF synchronizers, dual-port RAM, registered
full/empty); verified across asynchronous 7ns/11ns clocks streaming 300 words with
fill-to-full/drain-to-empty and never-violate safety — `[cdc_async_fifo] ALL 309
TESTS`.

`src/tpu_soc.v` (`make soc`) is a genuine **two-clock SoC**: an `ACLK` bus domain
(AXI slave + `axi_master_dma` + FIFO write/read sides) and an independent `CCLK` core
domain (the unchanged `TPU` + an instruction sequencer), bridged by **two** async CDC
FIFOs (instruction `ACLK→CCLK`, result `CCLK→ACLK`) plus single-bit 2-FF
synchronizers for the control pulses — no raw multi-bit crossings. Its TB runs real
programs end-to-end on asynchronous `ACLK=10ns` / `CCLK=13ns` (measured phase drift
0.5–9.5 ns): the host writes a DMA descriptor, the accelerator fetches the program
from external memory, executes it across the clock boundary, and the host reads the
correct `RESULT` back (`12`, `63`) — `[tpu_soc] ALL 21 TESTS`. No latches, lint and
`check -assert` clean for `top=tpu_soc`.

These resolve the former "slave-only / single-clock" limitation of `src/tpu_axi.v`
(which remains as the simple single-clock control wrapper).

---

## 4. Arbitrary-size tiles beyond `LINE_LANES = 4` — multi-line TM ✅ DEMONSTRATED

**Status: the multi-line-row packing is PROVEN with a working unit; folding it into
the existing single-line units is the remaining integration step.**

The shipped tensor units bound tile sizes because each packs **one row into one
128-bit TM line of `LINE_LANES = 4` lanes** (GEMM `N ≤ 4`, attention `D ≤ 4`, conv
rows `≤ 8` columns; softmax already spans `ceil(LEN/4)` lines as a 1-D reduction).
Larger tiles (a 16×16 GEMM, `D = 8` attention) need **multi-line TM rows** — a row
spanning `ceil(N/LINE_LANES)` consecutive lines.

**That packing now exists and is verified.** `src/gemm_ml.v` is an output-stationary
GEMM whose matrix rows span `LINES_PER_ROW = ceil(N/LINE_LANES)` TM lines: row `r`
occupies the lines at `base + r·LINES_PER_ROW`, element `k` at line
`base + r·LINES_PER_ROW + k/LINE_LANES`, lane `k % LINE_LANES`. Its load FSM reads
`LINES_PER_ROW` lines per row and assembles the full row before MAC; writeback packs
each row back across `LINES_PER_ROW` lines. It is proven at **N = 8 (2 lines per
row)** and N = 4 (1 line) against an independent real-domain golden — `make
unittests` → `[gemm_ml] ALL 12838 TESTS` (a deliberately wrong single-line packing
makes N = 8 fail, confirming the multi-line packing is load-bearing). verilator
`-Wall` and yosys `check -assert` clean.

**Remaining integration step.** `gemm_ml` proves the multi-line approach end-to-end as
a self-contained unit. Generalizing the **whole accelerator** to multi-line tiles
(retrofitting `gemm_systolic`/`conv2d_unit`/`attention_unit`, the `tpu_top` dispatch,
and the TM read-port arbitration to the same `LINES_PER_ROW` scheme) is the larger
cross-cutting change; the production single-line units are kept intact and verified,
with `gemm_ml` standing as the validated reference design for that rollout.

---

## 5. ~~Known correctness limitation — attention softmax pad collision~~ ✅ RESOLVED

A read-only adversarial audit of this hardening pass found three real issues. **All
three are now fixed** and are no longer limitations:
- a `conv2d_unit` output-pixel counter under-sized for even/small `K` with `pad=1`
  (now sized `OH + 2·PAD_MAX`, so every supported `K` is safe — proven by the
  even-`K` testbench `conv2d_evenk`);
- a `tpu_axi` STEP that could be silently dropped if it landed during a core
  pipeline stall (the wrapper now **holds** the instruction until the core's front
  end is free and exposes a `STATUS.BUSY` bit; a host polls `BUSY==0` before STEP);
- the attention softmax **pad collision** described below.

**What it was.** `attention_unit` used to run softmax over `SM_PAD = max(SEQ, SM_LEN)`
= 8 lanes, padding lanes `SEQ..SM_PAD-1` with the Q7.8 floor `Q78_MIN = -128.0` and
relying on `exp(pad - rowmax) ≈ 0`. In the extreme corner where **all `SEQ` real
logits in a row themselves saturate to that floor** (every key equally, maximally
anti-aligned), the real logits collided with the pad sentinel, softmax saw `SM_PAD`
identical values → uniform `1/SM_PAD` weights → context scaled by `SEQ/SM_PAD` (a 2×
magnitude loss at the default `SEQ=4`), and `sat` stayed 0 (silent).

**Fix (shipped).** `attention_unit` now instantiates the softmax submodule at
**`LEN = SEQ`** (no `SM_PAD` padding, no `Q78_MIN` sentinel), so it runs over exactly
the `SEQ` real logits and the collision is **impossible by construction**. The
softmax — and hence attention — also becomes strictly shorter: `LAT_TOTAL` drops from
123 to **87** cycles at `SEQ=4` (37 at `SEQ=2`); the attention testbenches' exact-
latency assertions were updated to match. A directed regression (`attention_unit_tb`
D10: every `Q=+max`, `K=−max`, all logits at the floor) confirms the context is now
the full uniform-weight column-mean of `V`, not half of it — it fails on the old code
and passes now. `softmax_unit` and `tpu_top` were not touched; all other tests are
unchanged (attention 4615 → 4639 from the added test).

---

*This roadmap is intentionally conservative: each item is a scoped, reasoned next
step, not a claimed feature. The current shipped state is fully described and
verified in the README, SPEC, ISA, and PPA documents.*
