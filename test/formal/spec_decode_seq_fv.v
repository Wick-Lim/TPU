`timescale 1ns/1ps
//============================================================================
// spec_decode_seq_fv.v -- FORMAL HARNESS (bounded model checking)
//
// Wraps the committed, READ-ONLY DUT src/spec_decode_seq.v.  All DUT inputs are
// top-level PORTS of this harness: yosys-smtbmc treats unconstrained top-level
// inputs as FREE primary inputs that the solver re-picks every cycle, so the
// proof covers ALL legal input sequences.  (NB: the `(* anyseq *) reg` idiom is
// avoided -- when several signals share one declaration only the first is freed;
// top-input ports are the reliable way to get per-cycle free inputs.)
//
// LEGAL-PROTOCOL constraints (assume):
//   * synchronous active-high reset is forced for the first cycle (t=0) via an
//     internal first_cycle flag OR'd into rst; thereafter rst_in is free;
//   * `start` may only pulse while not already running (arm-once discipline).
//     The monotonic proof does NOT rely on this; it just trims silly traces.
//
// SAFETY PROPERTIES proven (all UNGATED except for the cross-reset exclusion,
// which only skips the cycle right after a reset where counters legally reset
// to 0):
//   (P1) committed-token count `total_tokens` MONOTONIC non-decreasing
//   (P2) main_passes MONOTONIC non-decreasing
//   (P3) accepts      MONOTONIC non-decreasing
//   (P4) rejects      MONOTONIC non-decreasing
//   (P5) per step total_tokens advances by AT MOST 2 (1 verified + <=1 bonus)
//
// TOKW kept tiny (4) so BMC is fast; monotonicity is width-independent.
//============================================================================
module spec_decode_seq_fv #(
    parameter integer TOKW    = 4,
    parameter integer DRAFT_K = 1
)(
    input  wire                 clk,
    // free primary inputs (solver-chosen every cycle)
    input  wire                 rst_in,
    input  wire                 start,
    input  wire                 pass_valid,
    input  wire [TOKW-1:0]      verified_tok,
    input  wire [TOKW-1:0]      draft_tok,
    input  wire                 draft_present
);
    // ---- DUT outputs ----
    wire                 commit_valid;
    wire [TOKW-1:0]      commit_tok;
    wire                 accepted;
    wire [31:0]          total_tokens;
    wire [31:0]          main_passes;
    wire [31:0]          accepts;
    wire [31:0]          rejects;

    // ---- reset: forced high in the first cycle, free thereafter ----
    reg first_cycle = 1'b1;
    always @(posedge clk) first_cycle <= 1'b0;
    wire rst = first_cycle | rst_in;

    // shadow of "running" intent for the arm-once assume
    reg running_shadow = 1'b0;
    always @(posedge clk) begin
        if (rst)        running_shadow <= 1'b0;
        else if (start) running_shadow <= 1'b1;
    end

    // ---- DUT instance (committed module, unmodified) ----
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(DRAFT_K)) dut (
        .clk          (clk),
        .rst          (rst),
        .start        (start),
        .pass_valid   (pass_valid),
        .verified_tok (verified_tok),
        .draft_tok    (draft_tok),
        .draft_present(draft_present),
        .commit_valid (commit_valid),
        .commit_tok   (commit_tok),
        .accepted     (accepted),
        .total_tokens (total_tokens),
        .main_passes  (main_passes),
        .accepts      (accepts),
        .rejects      (rejects)
    );

    // ---- previous-cycle snapshots for monotonicity checks ----
    reg [31:0] prev_total   = 32'd0;
    reg [31:0] prev_passes  = 32'd0;
    reg [31:0] prev_accepts = 32'd0;
    reg [31:0] prev_rejects = 32'd0;
    reg        prev_rst     = 1'b1;
    reg        seen         = 1'b0;
    always @(posedge clk) begin
        prev_total   <= total_tokens;
        prev_passes  <= main_passes;
        prev_accepts <= accepts;
        prev_rejects <= rejects;
        prev_rst     <= rst;
        seen         <= 1'b1;
    end

    // Compare only across cycles where the PREVIOUS cycle was not a reset
    // (reset legally zeroes counters; we don't police that drop). The CURRENT
    // cycle may be anything -- counters keep their value during a reset cycle's
    // own combinational read, and the next-state zeroing is what prev_rst masks.
    wire chk = seen & ~prev_rst;

    // ---- assume legal protocol ----
    always @(posedge clk) begin
        if (running_shadow) assume (~start);   // arm-once
    end

    // ---- SAFETY ASSERTIONS ----
    always @(posedge clk) begin
        if (chk) begin
            a_p1_total_mono   : assert (total_tokens >= prev_total);
            a_p2_passes_mono  : assert (main_passes  >= prev_passes);
            a_p3_accepts_mono : assert (accepts      >= prev_accepts);
            a_p4_rejects_mono : assert (rejects      >= prev_rejects);
            a_p5_step_le2     : assert (total_tokens - prev_total <= 32'd2);
        end
    end
endmodule
