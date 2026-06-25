# TPU PPA Report (ECP5)

Power / Performance / Area (PPA) characterization of the Verilog TPU and its four
tensor units, mapped to the **Lattice ECP5** FPGA cell library with the open-source
Yosys synthesis flow.

> **Honesty note.** This report contains **real measured synthesis results** for
> **area** and a **combinational logic-depth** proxy for timing. It does **not**
> contain routed `fmax` or power numbers — those require place-and-route or a power
> analyzer that is **not installed in this environment** (see
> [§5 How to get true routed PPA next](#5-how-to-get-true-routed-ppa-next)). The
> `fmax` figures below are **rough first-order estimates** and are flagged as such.

---

## 1. Methodology

### 1.1 Tool flow

| Stage | Tool | What it produces | Status |
|-------|------|------------------|--------|
| Elaboration + synthesis | **Yosys `synth_ecp5`** (Yosys 0.66) | Logic mapped to real ECP5 primitives (LUT4, CCU2C, MULT18X18D, DP16KD, TRELLIS_FF, PFUMX, L6MUX21) | ✅ measured |
| Area report | Yosys `stat` (post-synth) | Per-primitive cell counts | ✅ measured |
| Timing proxy | Yosys `ltp` | Longest **topological** logic-cell depth on a combinational path | ✅ measured (proxy) |
| Place & route | **nextpnr-ecp5** | Routed `fmax`, real critical path, congestion | ❌ not installed |
| Power | nextpnr / vendor power tool | Dynamic + static power | ❌ not installed |

Each module is synthesized in its **own** Yosys invocation (per the `make ppa`
spec) so a slow run never stalls the others. Per-invocation timeout budget is
600 s. Every invocation **exited 0 with 0 `check` problems** for all five targets.

### 1.2 Area — how the numbers were produced

`synth_ecp5` performs full technology mapping to the ECP5 cell library. The `stat`
block at the end of synthesis reports the exact primitive counts that an ECP5
device would consume:

- **LUT4** — 4-input lookup tables (the basic logic element).
- **TRELLIS_FF** — flip-flops (registers).
- **MULT18X18D** — hard 18×18 DSP multipliers ("DSP" column).
- **DP16KD** — 18 Kb dual-port block RAMs ("BRAM" column).
- **CCU2C** — 2-bit carry/arithmetic units; the count is the **carry-chain length**
  ("carry" column).
- Plus mux primitives **PFUMX** / **L6MUX21** (reported in §2.1) used to build wide
  muxes from LUTs.

These are **real mapped cell counts**, not estimates.

### 1.3 Timing — `ltp` logic depth and the `fmax` estimate

Routed timing needs place-and-route, which is unavailable here. As a proxy we use
Yosys `ltp`, which reports the **longest topological path**: the maximum number of
logic cells chained between any two registers (or I/O) on a purely combinational
path. Deeper `ltp` ⇒ longer worst-case combinational delay ⇒ lower achievable
clock. It is a **structural depth metric, not a delay number** — it counts cells,
not nanoseconds, and ignores routing, fanout, and the fast dedicated carry chain.

**Rough `fmax` estimate (order-of-magnitude only):**

```
fmax_est ≈ 1 / (ltp_depth × t_cell)
```

where `t_cell` is a hand-waved average per-LUT-stage delay. Using a deliberately
conservative `t_cell ≈ 0.30 ns` (a typical ECP5-class LUT+local-route stage), the
estimates in §3 follow. **Limits of this estimate:**

- It treats every cell stage as equal delay; in reality CCU2C carry chains are far
  **faster** per bit than generic LUT logic, so carry-heavy depths are
  pessimistic.
- It ignores **routing delay**, which often dominates on real fabric.
- It ignores **fanout / load**.
- Absolute `t_cell` is a guess; only the **relative** ranking between units is
  trustworthy.

Treat the `fmax` column as a **ranking aid**, not a spec. The honest deliverable
here is the **logic-depth ranking**, which tells you *which block to pipeline
first*.

### 1.4 `ltp` "Detected loop" warnings — not real comb loops

During the `ltp` pass Yosys emits `Detected loop` warnings while traversing CCU2C
carry/accumulator chains. These are **traversal artifacts** of the topological walk
over arithmetic feedback, **not** real combinational loops. Synthesis `check`
reported **0 problems** for every module, confirming the designs are loop-free.

### 1.5 Reproduce

```sh
make ppa
```

This re-runs `synth_ecp5` for every unit, plus `ltp` for the **four tensor units**,
and writes filtered per-unit logs (a few hundred bytes each) to
`build/ppa/<unit>.log`. The **full TPU top is area-only** (no `ltp`): on the
flattened design `ltp` walks ~millions of carry-chain feedback paths (~15 min and a
multi-GB raw log), so the reproducible target reports TPU **area** and runs the
logic-depth proxy only on the four units — which are the actionable pipeline
candidates anyway. Output is filtered to the `stat`/`ltp` summary lines *before*
being written, so logs stay tiny. To run a subset:

```sh
make ppa PPA_UNITS=conv2d_unit
```

All numbers below were extracted from the final post-synth `stat` block and the
`ltp` length in `build/ppa/<unit>.log`.

---

## 2. Area

Area mapped to real ECP5 primitives. "DSP" = MULT18X18D, "BRAM" = DP16KD,
"carry" = CCU2C carry-chain cells.

Numbers below are for the **parameterized** units at their **default** sizes
(GEMM N=4, conv 8×8∗3×3, softmax LEN=8, attention SEQ=4/D=4). DSP counts are
unchanged from the original fixed design (the arithmetic is identical); the small
LUT/FF/carry increase vs. the pre-parameterization design is the parameterization
overhead ($clog2-derived widths + generalized index logic).

| Unit             | LUT4   | FF (TRELLIS_FF) | DSP (MULT18X18D) | BRAM (DP16KD) | Carry (CCU2C) |
|------------------|-------:|----------------:|-----------------:|--------------:|--------------:|
| gemm_systolic    |  4,630 |           1,381 |               48 |             0 |         1,478 |
| conv2d_unit      |  8,678 |           1,352 |               13 |             0 |           101 |
| softmax_unit     |  5,688 |             598 |               10 |             0 |         2,060 |
| attention_unit   |  9,249 |           2,035 |               89 |             0 |         3,178 |
| **TPU (full)**   | **49,675** |      **18,791** |          **161** |         **0** |     **7,544** |

### 2.1 Extra ECP5 mux primitives

`synth_ecp5` also maps wide muxes onto dedicated primitives (counted separately
from LUT4):

| Unit            | L6MUX21 | PFUMX |
|-----------------|--------:|------:|
| gemm_systolic   |       4 |   401 |
| conv2d_unit     |      37 |   841 |
| softmax_unit    |     917 | 1,676 |
| attention_unit  |   1,031 | 2,047 |
| TPU (full)      |   1,860 | 6,798 |

### 2.2 Notes on area

- **BRAM (DP16KD) = 0 for every unit.** `synth_ecp5` mapped **all** on-chip memory
  to **distributed LUT/FF** rather than inferring hard block RAM. This inflates the
  LUT4/FF counts. Inferring DP16KD (e.g. via explicit RAM-style coding or
  `memory_bram` rules) is a future area optimization, especially for the full TPU.
- **DSP usage:** `attention_unit` is the heaviest multiplier user (89
  MULT18X18D), reflecting its QK^T and attention×V matmuls plus reuse of softmax.
  The full TPU uses 161 DSPs.
- The full **TPU** dominates every column (47,744 LUT4, 18,694 FF) — it instantiates
  all four tensor units plus the control/DMA/memory subsystem.

---

## 3. Timing (logic-depth proxy)

`ltp` = longest topological logic-cell depth (timing **proxy**, not a delay).
`fmax_est` uses the rough formula in §1.3 with `t_cell ≈ 0.30 ns` and is an
**order-of-magnitude estimate only** — use the depth ranking, not the MHz value.

| Unit             | ltp depth | fmax_est (≈) | Notes |
|------------------|----------:|-------------:|-------|
| conv2d_unit      |        95 |    ~35 MHz   | Shallowest tensor unit |
| gemm_systolic    |       262 |    ~13 MHz   | Accumulation/carry chain |
| softmax_unit     |     1,580 |     ~2 MHz   | Long exp/reciprocal chains |
| **attention_unit** | **2,250** | **~1.5 MHz** | **Deepest tensor unit** ⚠️ |
| TPU (full)       |  ~5,168 † |    ~0.6 MHz  | Largest overall (top-level paths) |

† The full-TPU `ltp` (~5,168, on the pre-parameterization design) was measured
**once** during characterization. It is
**excluded from `make ppa`** because on the flattened top `ltp` emits ~4.8 M
carry-chain traversal-warning lines and runs ~15 min; the reproducible target
reports TPU **area only** and runs `ltp` on the four tensor units (the actionable
pipeline candidates). Re-measure manually with
`yosys -p "read_verilog -sv -I src <all src>; synth_ecp5 -top TPU; ltp"` if needed.

⚠️ **`attention_unit` has the deepest combinational path among the four tensor
units (ltp 2,250).** The full **TPU** top has the largest `ltp` overall (~5,168),
since its critical path can run through the deepest sub-block plus top-level glue.

The `fmax_est` numbers look very low because `ltp` counts **every** cell on a long
**unpipelined** arithmetic chain (reciprocal/exp, big accumulators) as if it were
one combinational stage. That is exactly the signal that motivates the pipelining
recommendations in §4 — it does **not** mean the silicon would only run at ~1 MHz
once pipelined and routed.

---

## 4. Key finding + recommendations

### 4.1 Key finding

The combinational-depth ranking is:

```
attention_unit (2250)  >  softmax_unit (1580)  >  gemm_systolic (262)  >  conv2d_unit (95)
```

- **`attention_unit` is the prime pipeline candidate.** It is the deepest block
  and it **serially reuses `softmax_unit`** (itself ltp 1,580). The long
  **combinational reciprocal/exp chains** in softmax, together with the **QK^T
  reduction** and attention×V matmul, dominate the path.
- **`softmax_unit` is the second target.** Its depth comes from long
  combinational **exp + reciprocal** approximation chains and the
  max-subtract/normalize sequence.
- **`gemm_systolic`** is moderately deep (262) due to the **accumulation tree /
  carry chain** across the systolic array — a classic pipeline-register target.
- **`conv2d_unit`** is already shallow (95) and is **not** a priority.

### 4.2 Recommendations (v3 perf work — NOT done now)

These are **future** optimizations. They are **explicitly not implemented in this
session** and are recommended for a later **v3 performance pass**:

1. **Pipeline the attention datapath** — register the QK^T reduction, the softmax
   exp/reciprocal chain, and the attention×V matmul into multiple stages.
2. **Pipeline the softmax exp/reciprocal chains** — break the long combinational
   approximation into pipelined stages; this also benefits attention (which reuses
   it).
3. **Pipeline the systolic GEMM accumulation tree** — insert accumulator pipeline
   registers along the carry/reduction path.
4. **Infer DP16KD block RAM** for the larger memories to reclaim LUT/FF area (§2.2).

> ### ⚠️ Pipelining changes cycle-accurate latency — it would break the current TBs
>
> Adding pipeline stages **changes the cycle-accurate latency** of each block:
> results appear N cycles later, and handshake/valid timing shifts. That means the
> existing **testbenches** (which assert results on specific cycles) **would have to
> be updated** alongside any pipelining change. For that reason this work is
> deliberately deferred. **No RTL, no testbench, and no Makefile logic were changed
> to produce this report** — it characterizes the **current on-disk design as-is**.
> No throughput/fmax improvement is being claimed; these are recommendations, not
> results.

---

## 5. How to get true routed PPA next

The numbers above are honest about their limits: **area is real**, but **`fmax`
and power are not routed measurements**. To obtain true PPA, run a place-and-route
or full implementation flow. None of these are installed in this environment.

### Option A — ECP5 routed flow (closest to this report)

Install **oss-cad-suite** (bundles `nextpnr-ecp5`), then feed the Yosys JSON
netlist into nextpnr:

```sh
# 1) install oss-cad-suite (provides nextpnr-ecp5, ecppack, prjtrellis)
#    https://github.com/YosysHQ/oss-cad-suite-build/releases
# 2) emit a JSON netlist from synth_ecp5  (write_json)
# 3) place & route -> real fmax + critical path
nextpnr-ecp5 --json build/ppa/<unit>.json \
             --package CABGA381 --speed 6 \
             --freq 100 --report build/ppa/<unit>.pnr.json
```

`nextpnr-ecp5` reports the **routed max frequency** and the **real critical path**
(with routing delay), replacing the §3 estimates with true timing. `ecppack` then
gives a bitstream for power estimation.

### Option B — Vendor FPGA flow

Use **Lattice Diamond/Radiant** (ECP5) or port to **Vivado** (Xilinx) /
**Quartus** (Intel) for vendor-grade timing closure and power analysis.

### Option C — ASIC flow

Use **OpenLane + Sky130** (open-source RTL-to-GDS) or a commercial ASIC flow for
true silicon PPA (gate-level timing, area in µm², and power).

---

*Generated from `build/ppa/<unit>.log` (Yosys 0.66, `synth_ecp5` + `ltp`).
Reproduce with `make ppa`. Area = measured; fmax = rough estimate pending P&R.*
