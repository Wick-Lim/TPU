#!/usr/bin/env python3
# ============================================================================
# modal_validate_test.py -- unit tests for the NON-GPU logic of
# tools/modal_validate.py (the pure-python compare / argmax-match helpers).
#
# These are exactly the parts of the P1.1 gate that do NOT need a GPU, modal,
# numpy, or torch -- so they CAN be proven here, in this repo, with the stdlib
# only.  (The GPU tiers themselves run on the user's Modal account; see
# docs/MODAL_VALIDATE.md.)  Importing modal_validate must succeed WITHOUT modal
# installed (the module ships a no-op shim) -- this test asserts that too.
#
# RUN:  python3 test/modal_validate_test.py     -> "ALL <N> TESTS PASSED", exit 0
# ============================================================================
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools"))

# This import must work even though modal is (almost certainly) not installed:
import modal_validate as mv


def _check(cond, msg):
    if not cond:
        raise AssertionError(msg)


def test_import_without_modal():
    # The module imported; if modal is absent it used the shim and `app` is the
    # shim object (still has the decorator-callable attributes used at module
    # scope).  Either way the public helpers are present and callable.
    for name in ("argmax", "argmax_match_rate", "topk_overlap", "error_stats",
                 "summarize_gate"):
        _check(hasattr(mv, name), f"missing helper {name}")
    _check(isinstance(mv.PROMPT_CORPUS, list) and len(mv.PROMPT_CORPUS) >= 1,
           "PROMPT_CORPUS must be a non-empty list")
    return 1


def test_argmax():
    _check(mv.argmax([10, 20, 30, 5]) == 2, "argmax basic")
    _check(mv.argmax([1, 1, 1]) == 0, "argmax ties -> first")
    _check(mv.argmax([-5, -1, -9]) == 1, "argmax negatives")
    _check(mv.argmax([42]) == 0, "argmax singleton")
    return 1


def test_argmax_match_rate():
    _check(mv.argmax_match_rate([1, 2, 3], [1, 2, 3]) == 1.0, "perfect match")
    _check(mv.argmax_match_rate([1, 2, 3], [1, 9, 3]) == 2.0 / 3.0, "one miss")
    _check(mv.argmax_match_rate([], []) == 1.0, "empty -> 1.0")
    _check(mv.argmax_match_rate([5, 5], [9, 9]) == 0.0, "all miss")
    # int coercion: numpy/torch scalars come back as ints; plain ints here
    _check(mv.argmax_match_rate([1, 2], [1.0, 2.0]) == 1.0, "int/float equal")
    raised = False
    try:
        mv.argmax_match_rate([1, 2, 3], [1, 2])
    except ValueError:
        raised = True
    _check(raised, "length mismatch must raise ValueError")
    return 1


def test_topk_overlap():
    _check(abs(mv.topk_overlap([5, 4, 3, 2], [5, 4, 1, 0], 2) - 1.0) < 1e-12,
           "identical top-2")
    _check(abs(mv.topk_overlap([5, 4, 3, 2], [2, 3, 4, 5], 2) - 0.0) < 1e-12,
           "disjoint top-2")
    _check(abs(mv.topk_overlap([5, 4, 3, 2], [5, 1, 4, 0], 2) - 0.5) < 1e-12,
           "half overlap")
    # k larger than vector length clamps without error
    _check(abs(mv.topk_overlap([1, 2], [1, 2], 8) - 1.0) < 1e-12, "k clamps")
    return 1


def test_error_stats():
    st = mv.error_stats([1.0, 2.0, 3.0], [1.0, 2.0, 3.5])
    _check(st["n"] == 3, "n")
    _check(st["exact"] == 2, "exact equal count")
    _check(abs(st["max_abs"] - 0.5) < 1e-12, "max_abs")
    _check(abs(st["max_rel"] - (0.5 / 3.0)) < 1e-12, "max_rel")
    _check(abs(st["rms_abs"] - (0.5 / (3 ** 0.5))) < 1e-12, "rms_abs")
    st0 = mv.error_stats([], [])
    _check(st0["n"] == 0 and st0["max_abs"] == 0.0, "empty stats")
    # zero golden -> not counted in max_rel (avoid div by zero)
    stz = mv.error_stats([0.0, 4.0], [1.0, 4.0])
    _check(stz["max_rel"] == 0.0, "zero golden ignored in rel")
    raised = False
    try:
        mv.error_stats([1.0], [1.0, 2.0])
    except ValueError:
        raised = True
    _check(raised, "error_stats length mismatch raises")
    return 1


def test_summarize_gate():
    s_full = mv.summarize_gate([1, 2, 3, 4], [1, 2, 3, 4])
    _check("4/4" in s_full and "100.0%" in s_full and "PASS" in s_full,
           f"full-match summary: {s_full}")
    s_part = mv.summarize_gate([1, 2, 3, 4], [1, 9, 3, 9])
    _check("2/4" in s_part and "50.0%" in s_part and "PASS" not in s_part,
           f"partial summary: {s_part}")
    return 1


def test_module_selftest():
    # the module's own _selftest must pass (exit code 0)
    _check(mv._selftest() == 0, "module _selftest returned non-zero")
    return 1


def main():
    tests = [
        test_import_without_modal,
        test_argmax,
        test_argmax_match_rate,
        test_topk_overlap,
        test_error_stats,
        test_summarize_gate,
        test_module_selftest,
    ]
    n_pass = 0
    for t in tests:
        n = t()
        n_pass += n
        print(f"  [PASS] {t.__name__}")
    print(f"ALL {n_pass} TESTS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
