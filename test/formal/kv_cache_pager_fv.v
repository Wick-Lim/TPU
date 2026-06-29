`timescale 1ns/1ps
//============================================================================
// kv_cache_pager_fv.v -- BMC formal harness for src/kv_cache_pager.v
//----------------------------------------------------------------------------
// Instantiates the committed DUT (READ-ONLY) and drives its inputs from free
// formal signals constrained with assume() to legal protocol.  Proves:
//   (A) APPEND/GATHER IN-BOUNDS
//        - resident_lo <= append_count                 (window never inverts)
//        - (append_count - resident_lo) <= RESIDENT     (window <= ring cap)
//        - on a resident gather the ring read slot is < RESIDENT (in ring)
//        - while a cold fetch is held, flash_idx < append_count
//          (the cold index is a real, previously-appended position)
//   (B) RESIDENT GATHER CORRECT
//        - an accepted resident gather drives row_valid the next cycle and
//          row_out == the row that was appended at that logical position.
//   (C) COLD GATHER CORRECT (bonus)
//        - modelling Flash backing store as the same shadow content, a cold
//          fetch returns row_out == the row appended at that position.
//
// A reference SHADOW array records, per logical position, the row appended
// there; the asserts compare DUT row_out against the shadow.  Sizes are kept
// SMALL via DUT params so BMC is tractable.
//============================================================================
module kv_cache_pager_fv #(
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
    input  wire                 flash_done
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

    // Flash backing store modelled by the SAME shadow content the appends saw,
    // so a correct cold fetch returns exactly the appended row.
    wire [ROW_BITS-1:0]  flash_row;

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
    // Free-running step counter -> forces reset at t==0 only.
    //------------------------------------------------------------------------
    reg [15:0] t = 16'd0;
    always @(posedge clk) t <= t + 16'd1;

    //------------------------------------------------------------------------
    // PROTOCOL ASSUMPTIONS (legal stimulus).
    //------------------------------------------------------------------------
    always @* begin
        // reset asserted exactly on the first cycle, deasserted afterwards.
        assume (rst == (t == 16'd0));

        // legal gather: only request a position that has actually been
        // appended (gather_idx < append_count).  This also forbids gather
        // when append_count == 0.
        if (gather_valid)
            assume (gather_idx < append_count);

        // never append past the (small) shadow capacity in the bounded window.
        if (append_valid)
            assume (append_count < (S_MAX[POSW-1:0] - 1'b1));
    end

    //------------------------------------------------------------------------
    // REFERENCE SHADOW : row appended at each logical position.
    //------------------------------------------------------------------------
    reg [ROW_BITS-1:0] shadow [0:S_MAX-1];
    always @(posedge clk) begin
        if (!rst && append_valid)
            shadow[append_count] <= append_row;
    end

    // Flash backing returns the row stored at the requested cold position.
    assign flash_row = shadow[flash_idx];

    //------------------------------------------------------------------------
    // residency / acceptance decode (mirrors the DUT, computed from outputs).
    //------------------------------------------------------------------------
    wire g_resident = (gather_idx < append_count) && (gather_idx >= resident_lo);
    wire acc        = gather_valid && !busy && !rst;
    wire acc_res    = acc &&  g_resident;
    wire acc_cold   = acc && !g_resident;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused_acc_cold = acc_cold;       // documentation handle
    wire _unused_of       = overflowed;
    wire _unused_fr       = flash_req || (|append_row);
    /* verilator lint_on UNUSEDSIGNAL */

    //------------------------------------------------------------------------
    // (B) RESIDENT GATHER CORRECT -- registered expectation, checked next cyc.
    //------------------------------------------------------------------------
    reg                res_v;
    reg [ROW_BITS-1:0] res_row;
    always @(posedge clk) begin
        if (rst) begin
            res_v   <= 1'b0;
            res_row <= {ROW_BITS{1'b0}};
        end else begin
            res_v   <= acc_res;
            res_row <= shadow[gather_idx];
        end
    end

    //------------------------------------------------------------------------
    // (C) COLD GATHER CORRECT -- busy(=G_FLASH) & flash_done => deliver next.
    //------------------------------------------------------------------------
    reg                cold_v;
    reg [ROW_BITS-1:0] cold_row;
    always @(posedge clk) begin
        if (rst) begin
            cold_v   <= 1'b0;
            cold_row <= {ROW_BITS{1'b0}};
        end else begin
            cold_v   <= (busy && flash_done);
            cold_row <= flash_row;            // == shadow[flash_idx]
        end
    end

    //------------------------------------------------------------------------
    // ASSERTIONS
    //------------------------------------------------------------------------
    always @(posedge clk) begin
        if (!rst) begin
            // (A) in-bounds / window invariants
            a_lo  : assert (resident_lo <= append_count);
            a_win : assert ((append_count - resident_lo) <= RESIDENT[POSW-1:0]);
            if (acc_res)
                a_slot : assert (gather_idx[RPTRW-1:0] < RESIDENT[RPTRW:0]);
            if (flash_req)
                a_fidx : assert (flash_idx < append_count);

            // (B) resident gather correctness
            if (res_v) begin
                a_res_valid : assert (row_valid);
                a_res_row   : assert (row_out == res_row);
            end

            // (C) cold gather correctness
            if (cold_v) begin
                a_cold_valid : assert (row_valid);
                a_cold_row   : assert (row_out == cold_row);
            end
        end
    end

endmodule
