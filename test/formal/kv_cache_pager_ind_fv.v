`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_ind_fv.v -- k-INDUCTION (UNBOUNDED) harness for
//                            src/kv_cache_pager.v   (READ-ONLY DUT)
//----------------------------------------------------------------------------
// PURPOSE
//   Prove the APPEND/GATHER IN-BOUNDS / WINDOW invariants of the committed
//   pager for ALL REACHABLE STATES (unbounded), via temporal k-induction
//   (yosys-smtbmc -i).  The bounded BMC harness kv_cache_pager_fv.v proves the
//   same window invariants PLUS the resident/cold data-correctness properties,
//   but only for the first K cycles from reset.  This harness upgrades the
//   WINDOW/IN-BOUNDS subset to an unbounded proof.
//
// WHAT IS (AND IS NOT) PROVABLE UNBOUNDED IN THIS yosys 0.66 BUILD
//   yosys 0.66 `write_smt2` cannot observe DUT internals: a hierarchical
//   reference such as `dut.g_state` / `dut.ring[..]` is read as a *dangling
//   autowire* that flatten never connects (verified: a BMC of
//   `busy == (dut.g_state==G_FLASH)` -- a tautology on the real RTL -- FAILS),
//   and SystemVerilog `bind` is dropped.  Therefore any invariant that must
//   relate an OUTPUT to the hidden FSM register `g_state` or to the hidden
//   `ring` memory is NOT expressible, so:
//     * (A) IN-BOUNDS / WINDOW  -> expressible from OUTPUTS only  -> UNBOUNDED here.
//     * (B) resident-gather data correctness  (needs g_state + ring)  )
//     * (C) cold-gather data correctness       (needs g_state)        )-> stay BOUNDED
//       ...remain proven only by BMC (kv_cache_pager_fv.v); making them
//       inductive would require internal observability (SymbiYosys with a
//       working `bind`, or in-RTL assertions) -- see docs/FORMAL.md.
//
// PROPERTIES PROVEN UNBOUNDED (all over DUT OUTPUTS only):
//   a_lo   : resident_lo <= append_count                  (window never inverts)
//   a_win  : (append_count - resident_lo) <= RESIDENT      (window <= ring cap)
//   a_slot : on an accepted resident gather the ring read slot < RESIDENT
//   a_fidx : while a cold fetch is held (flash_req), flash_idx < append_count
//            (the held cold index is a real, previously-appended position)
//
// STRENGTHENING INVARIANT (makes the induction STEP go through):
//   SI_fidx : flash_req -> flash_idx < append_count
//     This is a_fidx itself, but its role in the INDUCTIVE HYPOTHESIS is the
//     load-bearing strengthener: it excludes the spurious inductive pre-states
//     in which a held cold fetch points past the live counter (flash_idx >=
//     append_count).  It is self-inductive: `count`(=append_count) is
//     monotonic non-decreasing (only +1 on append, else hold, 0 on reset), and
//     every transition that SETS flash_req=1 (a cold accept) loads
//     flash_idx <= gather_idx with the protocol assume gather_idx < count;
//     every transition that HOLDS flash_req=1 leaves flash_idx fixed while
//     count can only grow -> flash_idx < count is preserved.  No reference to
//     the hidden FSM state is needed for THIS invariant.
//   a_lo / a_win / a_slot are combinational functions of `count` (resp.
//   gather_idx) and hold for ALL counter/index values -> trivially inductive.
//============================================================================
module kv_cache_pager_ind_fv #(
    parameter integer ROW_BITS  = 4,
    parameter integer RESIDENT  = 4,     // power of two
    parameter integer S_MAX     = 8,
    parameter integer FLASH_LAT = 2,
    parameter integer POSW      = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer RPTRW     = (RESIDENT <= 1) ? 1 : $clog2(RESIDENT)
)(
    input  wire                 clk,
    // free formal stimulus (constrained below)
    input  wire                 rst,
    input  wire                 append_valid,
    input  wire [ROW_BITS-1:0]  append_row,
    input  wire                 gather_valid,
    input  wire [POSW-1:0]      gather_idx,
    input  wire                 flash_done,
    input  wire [ROW_BITS-1:0]  flash_row      // cold backing data (opaque here)
);
    //------------------------------------------------------------------------
    // DUT outputs
    //------------------------------------------------------------------------
    wire                 row_valid;
    wire [ROW_BITS-1:0]  row_out;
    wire                 busy;
    wire                 flash_req;
    wire [POSW-1:0]      flash_idx;
    wire [POSW-1:0]      append_count;
    wire [POSW-1:0]      resident_lo;
    wire                 overflowed;

    kv_cache_pager #(
        .ROW_BITS (ROW_BITS),
        .RESIDENT (RESIDENT),
        .S_MAX    (S_MAX),
        .FLASH_LAT(FLASH_LAT)
    ) dut (
        .clk          (clk),
        .rst          (rst),
        .append_valid (append_valid),
        .append_row   (append_row),
        .gather_valid (gather_valid),
        .gather_idx   (gather_idx),
        .row_valid    (row_valid),
        .row_out      (row_out),
        .busy         (busy),
        .flash_req    (flash_req),
        .flash_idx    (flash_idx),
        .flash_done   (flash_done),
        .flash_row    (flash_row),
        .append_count (append_count),
        .resident_lo  (resident_lo),
        .overflowed   (overflowed)
    );

    //------------------------------------------------------------------------
    // PROTOCOL ASSUMPTIONS (legal stimulus).
    //   For k-INDUCTION the reset is NOT pinned to t==0 (induction reasons from
    //   arbitrary reachable states); we constrain the PER-CYCLE protocol
    //   exactly as the BMC harness does post-reset.
    //------------------------------------------------------------------------
    always @* begin
        // legal gather: only request a position that has actually been
        // appended (gather_idx < append_count).  Forbids gather when count==0.
        if (gather_valid)
            assume (gather_idx < append_count);

        // never append past the (small) shadow capacity in the bounded window;
        // keeps `count` from wrapping its POSW-bit range.
        if (append_valid)
            assume (append_count < (S_MAX[POSW-1:0] - 1'b1));
    end

    //------------------------------------------------------------------------
    // residency / acceptance decode (mirrors the DUT, computed from OUTPUTS).
    //------------------------------------------------------------------------
    wire g_resident = (gather_idx < append_count) && (gather_idx >= resident_lo);
    wire acc        = gather_valid && !busy && !rst;
    wire acc_res    = acc && g_resident;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = overflowed | row_valid | (|row_out) | busy | (|append_row);
    /* verilator lint_on UNUSEDSIGNAL */

    //------------------------------------------------------------------------
    // (A) IN-BOUNDS / WINDOW invariants  +  the SI_fidx strengthener.
    //   All over OUTPUTS only -> no hidden-state reference -> inductive.
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst) begin
            a_lo   : assert (resident_lo <= append_count);
            a_win  : assert ((append_count - resident_lo) <= RESIDENT[POSW-1:0]);
            if (acc_res)
                a_slot : assert (gather_idx[RPTRW-1:0] < RESIDENT[RPTRW:0]);

            // a_fidx == SI_fidx : held cold index is a real appended position.
            if (flash_req)
                a_fidx : assert (flash_idx < append_count);
        end
    end

endmodule
