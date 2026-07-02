`timescale 1ns/1ps
/* verilator lint_off DECLFILENAME */
//============================================================================
// clk_gate_cluster.v  --  CLOCK-GATING INTEGRATION WRAPPER (C7)
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   src/clk_en_ctrl.v DECIDES (per cycle, from activity hints) whether a
//   cluster's clock may be gated -- it emits a clock-ENABLE.  src/icg_cell.v
//   is the glitch-free integrated clock gate that turns that enable + the
//   primary clock into a clean GATED clock.  Neither, on its own, clocks any
//   real logic.  This wrapper wires the two together and hangs a real
//   synchronous leaf (register_file) off the resulting gated clock, so the
//   whole "activity -> enable -> ICG -> gated leaf" path exists as one
//   synthesizable, testable block.
//
// DATAFLOW
//       req  --> clk_en_ctrl.has_pending_work
//                clk_en_ctrl (clocked by the FREE-RUNNING clk) --> clk_en
//       clk_en  --> icg_cell.enable
//       scan_enable --> icg_cell.test_en   (DFT force-transparent)
//       icg_cell(clk, enable, test_en) --> gated_clk
//       register_file clocked by gated_clk  (the gated leaf)
//
// SAFETY INVARIANT (preserved from clk_en_ctrl)
//   clk_en_ctrl guarantees  clk_en >= req  every cycle: an advancing cluster
//   is NEVER gated.  Here enable == clk_en, so req=1 => enable=1 the SAME
//   cycle (wake margin), and pending work is never gated away.
//
// NOTES
//   - clk_en_ctrl is deliberately clocked by the primary (ungated) clk: its
//     hysteresis/measurement state must advance even while the leaf is gated.
//   - register_file's own reset clears its state; drive req (or scan_enable)
//     high while rst is asserted so the gated clock actually toggles through
//     the reset.
//   - HOLD is exposed so a testbench can shorten the idle-detect hysteresis.
//============================================================================
module clk_gate_cluster #(
    parameter integer HOLD = 1   // clk_en_ctrl hysteresis (idle cycles kept on)
)(
    input  wire        clk,           // free-running primary clock
    input  wire        rst,           // synchronous, active-high
    input  wire        scan_enable,   // DFT: force gate transparent (test_en)
    input  wire        req,           // activity/work pending -> wants the clock

    // ---- register_file (the gated leaf) stimulus ----
    input  wire [3:0]  read_addr1,
    input  wire [3:0]  read_addr2,
    input  wire [3:0]  read_addr3,
    input  wire [3:0]  write_addr,
    input  wire [31:0] write_data,
    input  wire        write_enable,

    // ---- register_file read results (from the gated leaf) ----
    output wire [31:0] read_data1,
    output wire [31:0] read_data2,
    output wire [31:0] read_data3,

    // ---- observability ----
    output wire        clk_en,        // enable driven into the ICG (== safety bound)
    output wire        gated_clk       // the gated clock feeding the leaf
);

    // -------- activity-based clock-enable controller (single cluster) --------
    wire        clk_en_w;
    wire [31:0] gated_cycles_unused;   // measurement counter (unused here)

    clk_en_ctrl #(
        .N_CLUSTER (1),
        .HOLD      (HOLD),
        .CNT_W     (32)
    ) u_clk_en_ctrl (
        .clk                     (clk),
        .rst                     (rst),
        .boot_active             (1'b0),
        .stall                   (1'b0),
        .has_pending_work        (req),        // the work/activity signal
        .input_valid             (1'b0),
        .output_ready_downstream (1'b0),
        .clk_en                  (clk_en_w),
        .gated_cycles            (gated_cycles_unused)
    );

    assign clk_en = clk_en_w;

    // -------- glitch-free integrated clock gate --------
    // enable == clk_en (so req=1 => enable=1, safety preserved);
    // test_en == scan_enable (DFT force-transparent).
    wire gated_clk_w;
    icg_cell #(
        .GATE_POLARITY (1)
    ) u_icg (
        .clk       (clk),
        .enable    (clk_en_w),
        .test_en   (scan_enable),
        .gated_clk (gated_clk_w)
    );

    assign gated_clk = gated_clk_w;

    // -------- the gated leaf: register_file clocked on the GATED clock --------
    register_file u_rf_gated (
        .clk          (gated_clk_w),
        .rst          (rst),
        .read_addr1   (read_addr1),
        .read_addr2   (read_addr2),
        .read_addr3   (read_addr3),
        .write_addr   (write_addr),
        .write_data   (write_data),
        .write_enable (write_enable),
        .read_data1   (read_data1),
        .read_data2   (read_data2),
        .read_data3   (read_data3)
    );

endmodule
/* verilator lint_on DECLFILENAME */
