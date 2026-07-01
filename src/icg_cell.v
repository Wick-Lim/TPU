`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// icg_cell.v  --  GLITCH-FREE INTEGRATED CLOCK GATE (ICG) cell for the FP8 die
//                                              (docs/IMPROVEMENT_PLAN.md P2.4)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   clk_en_ctrl.v decides, per cluster and per cycle, whether a cluster's clock
//   MAY be gated -- it emits a clock-ENABLE (clk_en).  That enable is a normal
//   data signal: it can change at ANY point in the cycle, including while the
//   clock is HIGH.  If you naively gate a clock with a bare AND
//       gated_clk = clk & enable
//   and `enable` falls (or rises) while `clk` is high, you chop the clock high
//   phase mid-pulse -- a GLITCH / runt pulse that can double-clock or corrupt
//   every flop on that gated domain.  You must NOT hand-AND a clock with a
//   free-running enable.
//
//   The standard fix is the INTEGRATED CLOCK GATE (ICG): a LOW-PHASE
//   TRANSPARENT LATCH samples `enable` while clk is LOW (the phase where the
//   downstream flops are not about to capture), HOLDS that sampled value while
//   clk is HIGH, and only then ANDs it with the clock:
//       en_latched  = enable   while clk==0  (latch transparent)
//       en_latched  = en_latched while clk==1 (latch opaque / holds)
//       gated_clk   = clk & en_latched
//   Because en_latched is FROZEN for the entire high phase, gated_clk can only
//   ever be a WHOLE clk pulse (if the value latched on the preceding low phase
//   was 1) or stay LOW (if it was 0).  No partial/runt pulse is possible no
//   matter when `enable` toggles.  This is exactly the cell a synthesis tool
//   infers for an enable-based clock gate; we spell out the standard, portable,
//   synthesis-friendly behavioral model (level-sensitive latch + AND) so the
//   gating is explicit and testable in simulation.
//
//----------------------------------------------------------------------------
// TEST-ENABLE (scan / DFT bypass)
//   `test_en` ORs into the latched enable path so the gate is forced
//   TRANSPARENT (gated_clk follows clk) during scan/test.  It defaults unused
//   in normal operation (tie 0 == pure functional gating) and, like `enable`,
//   is sampled only on the low phase so it can never itself create a glitch.
//   Tying test_en=0 reproduces a plain enable-only ICG (old behavior).
//
//----------------------------------------------------------------------------
// STRUCTURE / SAFETY
//   - Pure clock-gate primitive: NO reset (a clock gate has no data state; the
//     latch powers up transparent-on-low like a real ICG and self-corrects on
//     the first low phase -- matching standard-cell behavior).
//   - The latch is level-sensitive on ~clk; the AND is combinational.  There is
//     no register clocked by `clk` here (this cell MAKES a clock, it does not
//     consume one), no combinational loop (en_latch depends on clk+enable, not
//     on gated_clk), and no free-running-enable glitch by construction.
//   - GATE_POLARITY selects active-high (default, enable=1 -> pass) vs the
//     rarely-needed inverted sense, defaulting to the ordinary behavior.
//============================================================================
module icg_cell #(
    // Polarity of `enable`: 1 (default) = enable HIGH passes the clock.
    // 0 = enable LOW passes the clock (inverted-sense gate).  Leave at 1 for
    // the ordinary clk_en_ctrl-driven gate.
    parameter integer GATE_POLARITY = 1
)(
    input  wire clk,        // free-running source clock
    input  wire enable,     // clock enable (data signal, may change any time)
    input  wire test_en,    // scan/DFT force-transparent (tie 0 in functional use)
    output wire gated_clk   // clk if enabled, else held LOW (glitch-free)
);

    // Normalize the requested-enable to active-high "pass the clock" sense.
    // GATE_POLARITY==1 : pass when enable==1 (ordinary).
    // GATE_POLARITY==0 : pass when enable==0 (inverted-sense).
    wire enable_h = (GATE_POLARITY != 0) ? enable : ~enable;

    // Combined enable request: functional enable OR scan force-transparent.
    wire en_req = enable_h | test_en;

    // ---- LOW-PHASE-TRANSPARENT ENABLE LATCH -------------------------------
    // Transparent while clk is LOW (samples en_req), opaque while clk is HIGH
    // (holds the value sampled on the preceding low phase).  This is the whole
    // point of the ICG: en_latch is FROZEN across the entire high phase, so the
    // AND below can only pass a whole pulse or nothing -- never a runt.
    reg en_latch;
    always @(*) begin
        if (!clk)               // clk low  -> latch transparent
            en_latch = en_req;
        // clk high -> latch opaque: en_latch retains its value (no assignment)
    end

    // ---- GATED CLOCK ------------------------------------------------------
    // AND the (now stable-across-high-phase) latched enable with the clock.
    assign gated_clk = clk & en_latch;

endmodule
/* verilator lint_on DECLFILENAME */
