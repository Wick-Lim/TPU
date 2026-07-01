`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// reset_sync.v  --  ASYNC-ASSERT / SYNC-DEASSERT reset synchronizer  (P2.2)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   Every clock domain in the accelerator needs a locally-clean reset.  A raw
//   external / power-on reset (or a reset generated in another clock domain) is
//   ASYNCHRONOUS to this domain's clock: if it were used directly, its RELEASE
//   edge could land inside a flop's setup/hold window and drive that flop
//   metastable -- the classic "reset recovery / removal" failure that a static
//   timing tool cannot fix once the reset is fanned out.  The canonical, CDC-
//   signoff-approved fix is this reset synchronizer:
//
//       * ASSERT is ASYNCHRONOUS  -- the moment the async reset input goes
//         active, the output reset asserts (same delta), WITHOUT needing a
//         clock edge.  So the domain is held in reset even if its clock is not
//         yet toggling (e.g. PLL not locked).
//       * DEASSERT is SYNCHRONOUS -- after the async input releases, the "not
//         reset" value has to walk through STAGES flip-flops clocked by THIS
//         domain's clock before the output reset drops.  Any metastability from
//         the release edge is caught in the first flop and settles out over the
//         remaining STAGES-1 flops, so the reset the rest of the domain sees
//         releases cleanly and synchronously to its own clock.
//
//----------------------------------------------------------------------------
// STRUCTURE
//   A shift chain of STAGES flip-flops clocked by `clk`.  The chain shifts in a
//   constant "release" value (1) from the tail; the driven output reset is the
//   INVERSE of the head flop.  Every flop has an ASYNCHRONOUS CLEAR driven by
//   the async reset input, so while that input is active all STAGES flops are
//   held at 0 => output reset = ~0 = 1 immediately, with no clock edge.  When
//   the input releases, the 1 walks the chain one dest-clock edge at a time;
//   after STAGES edges the head flop is 1 and the output reset drops to 0.
//   (Async CLEAR of the chain + output inversion makes the ASYNC event ASSERT
//   the reset -- the metastability-safe async-assert / sync-deassert form.)
//
// POLARITY (matches the die-wide convention: ACTIVE-HIGH reset)
//   arst_in   : async reset request, ACTIVE-HIGH (1 = assert reset).
//   rst_out   : synchronized reset, ACTIVE-HIGH, registered.  1 = in reset.
//   An active-LOW consumer can simply use ~rst_out (or set INIT accordingly);
//   this module standardizes on active-high to match the rest of the design.
//
//----------------------------------------------------------------------------
// PROPERTIES (proven by test/reset_sync_tb.v)
//   (a) arst_in rising -> rst_out = 1 in the SAME delta (no clock edge needed).
//   (b) after arst_in falls, rst_out stays 1 for EXACTLY STAGES clk edges, then
//       deasserts on the STAGES-th edge.
//   (c) a sub-cycle glitch on arst_in still produces a full, clean multi-cycle
//       (STAGES-edge) reset pulse on rst_out -- the async assert latches into the
//       chain and only a synchronous walk can release it.
//
//   NO latch (the chain is edge-triggered + async set only), NO combinational
//   loop (rst_out is a pure inverse of a flop), X-safe (async set forces a
//   defined 0 into every flop the instant reset asserts).
//============================================================================
module reset_sync #(
    parameter integer STAGES = 2   // synchronizer depth (dest-clock deassert edges); >= 1
)(
    input  wire clk,        // DESTINATION clock -- the domain being reset
    input  wire arst_in,    // ASYNC reset request, ACTIVE-HIGH (1 = assert)
    output wire rst_out     // synchronized reset, ACTIVE-HIGH, registered (1 = in reset)
);

    // guard the degenerate STAGES<1 case (always keep at least one flop)
    localparam integer N = (STAGES < 1) ? 1 : STAGES;

    // Synchronizer shift chain.  We drive the chain toward the "release" value
    // 1 and take rst_out as its INVERSE, so the ASYNC reset event must force the
    // flops to 0: async reset -> chain cleared to 0 -> rst_out = ~0 = 1
    // (asserted).  Using async CLEAR (not async set) here, with the output
    // inversion, is the standard async-assert / sync-deassert form.
    reg [N-1:0] sync_chain;

    // ASYNC ASSERT / SYNC DEASSERT flip-flop chain.  The async reset appears in
    // the sensitivity list (posedge arst_in) so its ASSERT takes effect without
    // a clk edge; the DEASSERT walks synchronously through the chain on `clk`.
    always @(posedge clk or posedge arst_in) begin
        if (arst_in)
            sync_chain <= {N{1'b0}};                 // async assert: force all 0
        else
            sync_chain <= (sync_chain << 1) | {{(N-1){1'b0}}, 1'b1};
                                                     // sync deassert: shift a 1
                                                     // in at the LSB (MSB shifts
                                                     // out); clean for N==1 too
                                                     // ((0<<1)|1 == 1).
    end

    // rst_out is the INVERSE of the head flop: while any 0 is still walking the
    // chain the head is 0 => rst_out = 1 (in reset).  Once the 1 reaches the
    // head (after N clk edges past release) rst_out drops to 0.  Pure inverse of
    // a registered value -> output is effectively registered, no comb loop.
    assign rst_out = ~sync_chain[N-1];

endmodule
/* verilator lint_on DECLFILENAME */
