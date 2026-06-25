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

**Status: NOT done. Recommended for a v3 performance pass.**

The PPA characterization ([`PPA.md`](PPA.md) §3–4) measured the longest topological
logic-cell depth (`ltp`, a structural proxy, not a routed delay) of each tensor
unit. The ranking is:

```
attention_unit (2250)  >  softmax_unit (1580)  >  gemm_systolic (262)  >  conv2d_unit (95)
```

The tensor units are **single deep combinational datapaths** per FSM stage — the
softmax exp/reciprocal approximation chain, the attention QK^T reduction + softmax
reuse + ×V matmul, and the GEMM accumulation/carry chain are all unpipelined. The
PPA report's actionable finding is to **register these chains into multiple pipeline
stages**, attacking `attention_unit` first (deepest, and it serially reuses
`softmax_unit`), then `softmax_unit`, then the `gemm_systolic` accumulation tree;
`conv2d_unit` (depth 95) is already shallow and is not a priority.

**Why deferred.** Adding pipeline stages **changes the cycle-accurate latency** of
each block: results appear N cycles later and the `start`/`busy`/`done` handshake
timing shifts. The current per-unit TBs assert `done` on specific cycles and check
results bit-exactly, so they would all have to be rewritten alongside the RTL. That
is a coordinated RTL + testbench change, out of scope for the characterization-only
PPA work (which changed no RTL, no TB, no Makefile — see [`PPA.md`](PPA.md) §4.2). No
throughput/`fmax` improvement is being claimed; pipelining is a recommendation, not a
result.

---

## 2. AXI MASTER for autonomous DMA from external DRAM

**Status: NOT done. The current wrapper is SLAVE-only.**

`src/tpu_axi.v` is an AXI4-Lite **slave** ([`../SPEC.md`](../SPEC.md) §10): a host on
the bus drives the core one instruction at a time (write `INSTR`, write `CTRL.STEP`,
read back `RESULT`/`STATUS`). The wrapper **cannot initiate bus transactions** — it
has no AW/AR *master* channels and so cannot fetch a program or stream tiles from
external DRAM on its own; every byte of program-visible state is pushed in by the
host.

A v3 step would add an **AXI master** port so the accelerator can autonomously DMA
operand/result tiles between external DRAM and the on-chip `TM`/`DMEM`, turning the
single-step issue model into a descriptor-driven offload engine.

**Why deferred.** A master port is a materially larger block (read/write address
generation, burst handling, outstanding-transaction tracking, and a DMA descriptor
interface into the existing `dma_controller`/`TM` paths), and it needs its own BFM +
goldens. The slave wrapper was the minimal, fully-verifiable step to make the core
SoC-integratable; the master path is the next, separately-scoped increment.

---

## 3. Multi-clock CDC (the wrapper is single-`ACLK`)

**Status: NOT done. The wrapper is a single clock domain.**

`src/tpu_axi.v` runs the AXI slave logic **and** the core on the **same `ACLK`**, with
`ARESETn` assumed synchronous to `ACLK` ([`../SPEC.md`](../SPEC.md) §10). There is no
clock-domain-crossing logic: the bus and the compute core cannot run at independent
frequencies.

A realistic SoC often clocks the AXI fabric and a compute accelerator separately
(e.g. a fast bus and a slower deep-datapath core, especially given the deep
combinational paths in item 1). A v3 step would split the wrapper into a bus clock
domain and a core clock domain with proper CDC (handshake/FIFO synchronizers on the
register interface and the instruction/result paths).

**Why deferred.** CDC adds synchronizer logic and a multi-clock verification burden
(metastability/handshake correctness, dual-clock TB) that the single-clock wrapper
deliberately avoids to keep the issue model simple and the BFM TB deterministic. It
becomes worthwhile precisely once the core is pipelined (item 1) and wants its own
clock.

---

## 4. Arbitrary-size parameterization beyond `LINE_LANES = 4`

**Status: NOT done. Sizes are bounded by the 4-lane TM line.**

The tensor units are parameterized ([`../SPEC.md`](../SPEC.md) §9), but every upper
bound traces to the architectural decision that **one matrix/feature/Q-K-V row packs
into one 128-bit TM line of `LINE_LANES = 4` lanes**:

- GEMM `N ≤ 4`, attention `D ≤ 4` — one row is one TM line.
- conv image/kernel rows `≤ 8` columns — dense-16 pack, 4 lanes × 16-bit.
- softmax scales further (`LEN ≤ 32`) **only** because it is a 1-D reduction over
  `ceil(LEN/4)` lines, and even then its `argmax` status port stays a 3-bit field.

Larger tiles (e.g. a 16×16 GEMM, `D = 8` attention) need **multi-line TM rows**: a
row would span several TM lines, requiring changes to the TM line layout/packing in
`tpu_defs.vh` / `tile_memory.v` / `tpu_top.v`, the per-unit operand-fetch loops, and
the TM read-port arbitration. This is a **TM port / packing re-architecture**, not a
parameter bump.

**Why deferred.** The 4-lane line is baked into the architecture's central
two-operand-domain decision ([`../SPEC.md`](../SPEC.md) §1.1) and into the verified
TM read/write port model. Generalizing to multi-line rows is a cross-cutting
datapath change with its own verification surface; the current parameterization
deliberately stays within the single-line-per-row envelope (proven at a 2nd in-range
size per unit) rather than overreach. Until that re-architecture lands, sizes beyond
the bounds in §9 are **out of scope**, and the units enforce the documented ranges.

---

## 5. Known correctness limitation — attention softmax pad collision

A read-only adversarial audit of this hardening pass found three real issues. Two
were **fixed immediately** and are NOT limitations: a `conv2d_unit` output-pixel
counter that was under-sized for even/small `K` with `pad=1` (now sized
`OH + 2·PAD_MAX`, so every supported `K` is safe), and a `tpu_axi` STEP that could
be silently dropped if it landed during a core pipeline stall (now the wrapper
**holds** the instruction until the core's front end is free and exposes a
`STATUS.BUSY` bit; a host polls `BUSY==0` before the next STEP).

The third is a **pre-existing** (v2.0) corner-case left as a documented bound:

**What.** `attention_unit` runs softmax over `SM_PAD = max(SEQ, SM_LEN)` lanes,
placing the `SEQ` real logits in lanes `0..SEQ-1` and padding the rest with the most
negative Q7.8 value `Q78_MIN = -128.0`, relying on `exp(pad - rowmax) ≈ 0`. In the
EXTREME corner where **all `SEQ` real logits in a row themselves saturate to the
Q7.8 floor** `Q78_MIN` (every key equally, maximally anti-aligned — scaled scores
`< -128.0`, far outside the `|val| ≤ 512` tested range), the real logits collide
with the pad sentinel, softmax sees `SM_PAD` identical values and returns uniform
`1/SM_PAD` weights, so the `SEQ` real weights sum to `SEQ/SM_PAD` instead of `1.0`
and the context magnitude is scaled by `SEQ/SM_PAD` (a 2× loss at the default
`SEQ=4`/`SM_PAD=8`). Because the context accumulator does not overflow, `sat` stays
**0** — a silent magnitude loss in that corner. (Detailed in the `attention_unit.v`
SATURATION POLICY header.)

**Why deferred.** The correct fix is a per-lane **valid mask** in `softmax_unit` so
pad lanes contribute exactly 0 regardless of value (or instantiating softmax at
`LEN=SEQ` with no pad lanes). Both change the softmax cycle-accurate latency and
therefore the `attention_unit` / softmax **golden testbenches** (which assert exact
cycles and `sat==gsat` every vector), so it is a coupled RTL-plus-verification change
rather than a local patch. The normal operating range (the entire `|val| ≤ 512`
tested envelope, and any realistic attention workload) is unaffected; this is a
documented bound on an out-of-range input, not a regression.

---

*This roadmap is intentionally conservative: each item is a scoped, reasoned next
step, not a claimed feature. The current shipped state is fully described and
verified in the README, SPEC, ISA, and PPA documents.*
