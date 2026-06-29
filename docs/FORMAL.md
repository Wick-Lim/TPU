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

## Honest coverage

- **Bounded, not unbounded.** Each property holds for all legal input sequences over the first
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
