# Formal Verification — memory-system controllers (BMC)

Bounded model checking (BMC) of the production memory-system controllers, proving safety
properties with **yosys `write_smt2` + `yosys-smtbmc -s z3`** (SymbiYosys is not installed;
yosys 0.66 + z3 are). Harnesses live in `test/formal/*_fv.v` — each instantiates the **committed,
unmodified** controller, drives its inputs from free formal signals constrained by `assume()` to
the legal protocol, and states the safety properties as `assert()`. Run with `make formal`.

## What is proven (no counterexamples / no bugs found in any controller)

| Controller | Properties proven | Bound K |
|---|---|---|
| `expert_cache_pf` | P1 HIT returns the slot holding the requested expert; P2a no duplicate install; P2b directory uniqueness (no id in two valid slots); **P3 bounded response liveness / no deadlock** (worst-case demand latency = 2·FLAT+1, proven with prefetch enabled + adversarial) | 55 (P1/P2), 40 (P3) |
| `kv_cache_pager` | append/gather indices in-bounds (window never inverts, resident window ≤ ring capacity, read slot < RESIDENT, cold index < append_count); resident gather returns the row appended at that position; cold (Flash-spilled) gather returns the right row | 22–24 |
| `spec_decode_seq` | committed-token count monotonic non-decreasing (never loses a token); main_passes/accepts/rejects monotonic; per-cycle commit ≤ 2 (1 verified + ≤1 bonus) | 40 |
| `ddr5_xbar` | no spurious response (no resp without an accepted req); per-channel response FIFO never overflows (no dropped/lost response); `resp_tag` was always a tag that was issued | 16 |
| `boot_loader` | `done` is a stable level once raised; `done` never asserts early (only after every resident word is written) | 20 |

## Method (the load-bearing tooling note)

yosys 0.66 lowers `assert(...)` into **`$check`** cells, and this `write_smt2` **silently ignores
`$check`** — emitting an SMT model with *no assertions*, so `yosys-smtbmc` reports a meaningless
vacuous `PASSED` (even `assert(0)` "passes"). The fix, used by every harness here, is to insert
**`async2sync; chformal -lower`** before `write_smt2` so the checks become real `$assert`/`$assume`.
Each model was confirmed non-vacuous (`grep -c assert <model>.smt2` > 0, and mutation tests:
weakening/inverting an assertion produces a genuine counterexample). `memory_map` is also needed
on the directory-memory DUTs to avoid a false `write_smt2` logic-loop.

Canonical flow (per harness):
```sh
yosys -p "read_verilog -sv -formal -I src src/<dut>.v test/formal/<dut>_fv.v; \
          prep -top <dut>_fv -flatten; async2sync; chformal -lower; \
          write_smt2 -wires scratchpad/<dut>_fv.smt2"
yosys-smtbmc -s z3 -t <K> scratchpad/<dut>_fv.smt2     # -> ## Status: PASSED
```

## Unbounded proofs by k-induction (`make formal-ind`)

A subset of the safety properties is additionally proven **UNBOUNDED** by temporal **k-induction**
(`yosys-smtbmc -i -s z3`): the base case **and** the induction step both pass, so the asserts hold
on **all reachable states**, not just the first K cycles. The strengthening harnesses are
`test/formal/*_ind_fv.v` (run via the `formal-ind` target / `run_kind`). Naive k-induction fails
the step because the inductive hypothesis admits unreachable states; each harness adds
**strengthening-invariant** asserts that pin the reachable state space until the step closes.

| Controller | Proven UNBOUNDED (k-induction) | K | Stays BOUNDED (BMC only) |
|---|---|---|---|
| `boot_loader` | `done` stable-once-raised + never-early | 8 | — |
| `kv_cache_pager` | append/gather in-bounds + window invariants | 16 | — |
| `spec_decode_seq` | token-accounting equality, per-cycle modular increment bounds, non-decreasing-except-wrap | 2 | strict (non-wrapping) monotonicity (32-bit counter wrap) |
| `ddr5_xbar` | request-path routing safety (one-hot routing / banked select / ready coherence / payload integrity) | 12 | response-FIFO no-overflow, tag-issued |
| `flash_xbar` | **per-channel-queue no-overflow `cnt[c]≤QDEPTH`, `outstanding ≤ N_CH·QDEPTH` (P3), `inflight ≤ outstanding`, no-underflow (P1a/P1b)** | 3 | tag-issued (P2) |

**`flash_xbar` — how the internal counters are reached.** P3 / per-channel no-overflow are *not*
inductive on the harness's black-box shadow counters alone: in a spurious pre-state the global
`outstanding` can sit at the cap while one channel is over-full, so the step admits an issue that
overflows. The fix pins the DUT's **own** per-channel registers `outst[c]`, `cnt[c]` with the
strengthening set **S1** `outst[c]≤QDEPTH` (the acceptance-gate invariant, self-inductive via the
`!ch_full` issue gate), **S2** `cnt[c]≤QDEPTH` (FIFO occupancy, self-inductive via
`mem_resp_ready=!full`), **S3** `cnt[c]≤outst[c]`, and the linkages **L_out** `outstanding=Σ outst[c]`,
**L_in** `inflight=Σ cnt[c]`, **L_die** `die_inflight[c]=outst[c]−cnt[c]` (each inductive by
construction). yosys 0.66 has **no hierarchical-reference support** (`u_dut.outst[0]` parses as a
fresh flat implicit wire, not the register), so the harness declares `(* keep *)` **undriven** probe
wires and the build wires them to the flattened DUT registers with `connect -set \dut_outst0
\u_dut.outst[0] ` — the **trailing space terminates the bracketed escaped id** (otherwise `[0]` is
parsed as a bit-select). The committed RTL is untouched. Verified non-vacuous: probes
`PROBE_OUTST`/`PROBE_FULL`/`PROBE_RESP` each yield a BMC counterexample (cap, FIFO-full, and a firing
response are all reachable, so the bounds are tight), and a mutation (`outst[c]≤QDEPTH−1`) fails both
BMC and induction (the probe is genuinely bound to a register that reaches QDEPTH). Minimal induction
depth is **k=2** (k=1 fails); the target runs k=3 for margin.

**`flash_xbar` P2 stays BOUNDED.** "`resp_valid ⇒ issued[resp_tag]`" holds under BMC (to ≥16) but is
not inductive without a data-invariant over the FIFO **contents** ("every occupied slot holds an
issued tag"). The response FIFO is a 2-D `$mem` whose elements are not exposed as wires, so the
`connect`-binding trick cannot reach them, and yosys 0.66 cannot express the quantified
memory-content invariant from a read-only harness. It therefore remains a bounded (BMC) result.

## Honest coverage

- **Bounded, not unbounded** *(applies to the `make formal` BMC table above; see the k-induction
  table for the properties additionally proven unbounded).* Each property holds for all legal input
  sequences over the first
  K cycles from reset — no k-induction was run, so this is not an unbounded proof. The small
  instances wrap/overflow/evict multiple times within K (e.g. `kv_cache_pager` RESIDENT=4 overflows
  by ~cycle 5; `expert_cache_pf` K=55 ≈ 15 fill/evict/hit transactions), so steady-state behaviour
  is exercised — but a bug needing > K cycles to manifest would not be caught.
- **Parameter scope.** Proven on small tractable instances (e.g. `expert_cache_pf` SLOTS=2/
  N_EXPERT=4, `kv_cache_pager` RESIDENT=4/ROW_BITS=4, `ddr5_xbar` N_CH=2). The RTL is parametric;
  the proof is for the instantiated sizes, with full-width correctness argued by parametricity.
- **One known gap:** `expert_cache_pf` P1/P2 are proven with prefetch **disabled** (prefetch
  installs are silent — no response carries the victim slot — so an external shadow directory can't
  track them, and this yosys build cannot observe DUT internals / `bind` is dropped). P3 liveness
  **is** proven with prefetch enabled. Full prefetch-active directory checking would need internal
  observability (SymbiYosys with working `bind`, or in-RTL assertions).

All numbers are BMC results, not a substitute for the directed/random simulation suites
(`make unittests`) — they are complementary: the TBs check end-to-end function, the BMC proves
the controllers can't violate these invariants for any input within the bound.
