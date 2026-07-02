#!/usr/bin/env python3
"""cdc_check.py -- structural CDC sign-off checker for glm_fp8_system_cdc (task C8).

This is a TARGETED structural verifier, not a generic CDC analyzer. It parses
src/glm_fp8_system_cdc.v and asserts that the module's cross-domain interface is
built ENTIRELY from recognized synchronizer structures, and that no raw
multi-bit register is driven directly across the host_clk/core_clk boundary
outside them. It prints a per-crossing GUARDED/UNGUARDED report and exits
non-zero if any required synchronizer is missing or any raw crossing is found.

WHY TARGETED (honest limits): a sound generic Verilog CDC tool must elaborate
the full hierarchy, track every net's driving clock through instances, and
model gray-encoding / pulse-width / reconvergence -- that is a commercial-tool
job. This checker instead encodes the module's DOCUMENTED crossing contract
(the header of glm_fp8_system_cdc.v) and verifies each element is structurally
present and correctly wired, plus a raw-crossing guard. It is a sign-off AID
and regression guard, not a replacement for static CDC analysis. It TRUSTS the
cdc_async_fifo and reset_sync primitives (separately unit-verified) rather than
re-checking their internal gray-pointer syncs.

Crossings verified (per glm_fp8_system_cdc.v header, lines ~50-54):
  host->core : {prompt_tok,start_pos,s_len}  via u_req_fifo (cdc_async_fifo)
  core->host : next_tok                       via u_tok_fifo (cdc_async_fifo)
  core->host : busy (level)                   via busy_s1/busy_s2 2-FF sync
  core->host : done (pulse->toggle)           via done_tgl_c -> done_tgl_h1..h3
  reset      : host_rst/core_rst              via u_host_rst_sync/u_core_rst_sync
"""
import re
import sys


def strip_comments(t):
    t = re.sub(r"/\*.*?\*/", "", t, flags=re.S)
    return re.sub(r"//[^\n]*", "", t)


CHECKS = [
    # (label, src->dst, list of regex fragments that must ALL be present)
    ("prompt_tok/start_pos/s_len", "host->core",
     [r"cdc_async_fifo\s*#?\s*\(.*?\)\s*u_req_fifo\s*\(",
      r"u_req_fifo\s*\(.*?\.wclk\s*\(\s*host_clk\s*\).*?\.rclk\s*\(\s*core_clk\s*\)"]),
    ("next_tok", "core->host",
     [r"cdc_async_fifo\s*#?\s*\(.*?\)\s*u_tok_fifo\s*\(",
      r"u_tok_fifo\s*\(.*?\.wclk\s*\(\s*core_clk\s*\).*?\.rclk\s*\(\s*host_clk\s*\)"]),
    ("busy (level)", "core->host",
     [r"busy_s1\s*<=\s*sys_busy", r"busy_s2\s*<=\s*busy_s1"]),
    ("done (toggle)", "core->host",
     [r"done_tgl_c\s*<=\s*~\s*done_tgl_c", r"done_tgl_h1\s*<=\s*done_tgl_c",
      r"done_tgl_h2\s*<=\s*done_tgl_h1"]),
    ("host reset", "async->host",
     [r"reset_sync\s*#?\s*\(.*?\)\s*u_host_rst_sync\s*\(.*?\.clk\s*\(\s*host_clk\s*\)"]),
    ("core reset", "async->core",
     [r"reset_sync\s*#?\s*\(.*?\)\s*u_core_rst_sync\s*\(.*?\.clk\s*\(\s*core_clk\s*\)"]),
]


def domain_of_always_blocks(body):
    """Map each reg -> the posedge clock of the first always block assigning it."""
    wd = {}
    for m in re.finditer(r"always\s*@\s*\((.*?)\)(.*?)(?=always\s*@|\bendmodule\b)",
                         body, flags=re.S):
        ck = re.search(r"posedge\s+(\w+)", m.group(1))
        if not ck:
            continue
        for a in re.finditer(r"(\w+)\s*(?:\[[^\]]*\])?\s*<=", m.group(2)):
            wd.setdefault(a.group(1), ck.group(1))
    return wd


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "src/glm_fp8_system_cdc.v"
    txt = strip_comments(open(path).read())
    mm = re.search(r"\bmodule\s+(\w+)\b(.*?)\bendmodule\b", txt, flags=re.S)
    if not mm:
        print(f"cdc_check: no module in {path}", file=sys.stderr)
        return 2
    name, body = mm.group(1), mm.group(2)
    print(f"cdc_check: analyzing module {name} ({path})")
    clks = sorted(set(re.findall(r"posedge\s+(\w+)", body)))
    print(f"cdc_check: clock domains = {clks}\n")

    fails = 0
    for label, direction, frags in CHECKS:
        ok = all(re.search(fr, body, flags=re.S) for fr in frags)
        tag = "GUARDED  " if ok else "MISSING  "
        via = {"prompt_tok/start_pos/s_len": "u_req_fifo (async FIFO)",
               "next_tok": "u_tok_fifo (async FIFO)",
               "busy (level)": "busy_s1/busy_s2 2-FF sync",
               "done (toggle)": "done_tgl_c -> done_tgl_h1..h3",
               "host reset": "u_host_rst_sync (reset_sync)",
               "core reset": "u_core_rst_sync (reset_sync)"}[label]
        print(f"  [{tag}] {label:28s} {direction:11s} via {via}")
        if not ok:
            fails += 1

    # Raw-crossing guard: no reg written in one domain may be the SOLE rhs of a
    # multi-bit assignment in the other domain unless it is a first-stage sync
    # flop (name endswith _s1/_h1) or a FIFO/reset_sync output. We flag direct
    # `X <= <other-domain-reg>` where the LHS is NOT a recognized sync stage.
    wd = domain_of_always_blocks(body)
    allow_lhs = re.compile(r"(_s1$|_h1$|_meta$|_sync\d?$)")
    raw = []
    for m in re.finditer(r"always\s*@\s*\((.*?)\)(.*?)(?=always\s*@|\bendmodule\b)",
                         body, flags=re.S):
        ck = re.search(r"posedge\s+(\w+)", m.group(1))
        if not ck:
            continue
        dom = ck.group(1)
        for line in re.split(r";", m.group(2)):
            if "<=" not in line:
                continue
            before = line.split("<=", 1)[0]
            # LHS = the identifier IMMEDIATELY before '<=' (strip any bit-select),
            # not the first word of the segment (which may be if/else/end).
            lm = re.search(r"(\w+)\s*(?:\[[^\]]*\])?\s*$", before)
            if not lm:
                continue
            lhs = lm.group(1)
            rhs = line.split("<=", 1)[1]
            ids = [t for t in re.findall(r"[A-Za-z_]\w*", rhs)]
            for sig in ids:
                if wd.get(sig) and wd[sig] != dom and not allow_lhs.search(lhs):
                    # crossing captured somewhere other than a first-stage flop
                    raw.append((sig, wd[sig], dom, lhs))
    raw = sorted(set(raw))
    if raw:
        print("\ncdc_check: potential RAW cross-domain uses (not a first-stage sync flop):")
        for sig, s, d, lhs in raw:
            print(f"  [UNGUARDED] {sig} ({s}->{d}) captured into '{lhs}'")
        fails += len(raw)

    print()
    if fails:
        print(f"cdc_check: FAIL -- {fails} issue(s)")
        return 1
    print(f"cdc_check: PASS -- all {len(CHECKS)} documented crossings guarded, "
          f"no raw cross-domain register capture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
