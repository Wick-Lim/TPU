`timescale 1ns/1ps
//============================================================================
// spec_decode_seq.v  --  GLM-5.2 MTP SPECULATIVE-DECODE CONTROLLER
//                        (docs/SYSTEM_SINGLE_PACKAGE.md -- on-chip decode loop)
//----------------------------------------------------------------------------
// PURPOSE
//   On-chip CONTROLLER for DeepSeek-V3-style MTP (Multi-Token-Prediction)
//   speculative decoding, as shipped by GLM-5.2 (num_nextn_predict_layers=1,
//   the appended `mtp_head`).  It avoids a host round-trip per token by doing
//   the accept/reject bookkeeping of the speculative loop in hardware.
//
//   THE LOOP (K=1 draft):
//     * a MAIN-MODEL pass (glm_model_fp8) produces the VERIFIED next token for
//       the position being decoded, AND -- via the appended MTP head -- a
//       DRAFT for the FOLLOWING position, in the SAME pass;
//     * the NEXT main pass VERIFIES that draft: its own logits give the true
//       token for the drafted position.
//         - draft == true  -> ACCEPT : the drafted token is confirmed "for
//                             free"; this pass commits TWO tokens (the verified
//                             token + the accepted draft) and the next pass
//                             starts AFTER the accepted token;
//         - draft != true  -> REJECT : discard the draft; this pass commits
//                             ONE token (the verified token) and the next pass
//                             re-drafts from it.
//
//   So each main pass advances 1 token ALWAYS + 1 BONUS when the prior draft
//   was correct  =>  effective tokens / main-pass = 1 + alpha, where alpha is
//   the MTP acceptance rate (alpha = accepts / (accepts+rejects)).  Derive it
//   externally as total_tokens / main_passes.
//
//----------------------------------------------------------------------------
// MODEL ABSTRACTION
//   This unit consumes RESULTS, never weights.  The main model + MTP head are
//   the existing units; this controller only sees, once per completed pass:
//     verified_tok   -- the token the main model actually produced for the
//                       position being verified THIS pass;
//     draft_tok      -- the MTP draft this pass produced for the NEXT position;
//     draft_present  -- 1 if a draft from the PREVIOUS pass is being verified
//                       this pass (0 on the very first pass: nothing to verify).
//   The previous pass's draft is held INTERNALLY (`pending_draft`) and compared
//   to verified_tok to decide accept/reject -- the controller is self-contained.
//
//----------------------------------------------------------------------------
// COMMIT STREAM
//   Tokens leave on a single-beat stream {commit_valid, commit_tok}: 1 beat for
//   a reject / first pass, 2 beats for an accept.  Because the port is one beat
//   wide, the bonus (accepted-draft) beat lands the cycle AFTER the verified
//   beat -- in the idle gap before the next pass (a real main-model pass takes
//   many cycles, so pass_valid pulses are always >= 2 cycles apart; the bonus
//   never collides with the next pass).
//
//----------------------------------------------------------------------------
// ROLLBACK
//   There is NO state to unwind on a reject.  "Rollback" is simply: do NOT emit
//   the bonus beat and do NOT count the draft as a token; the stored draft is
//   overwritten by THIS pass's fresh draft_tok like any other pass.  The model
//   guarantees the next pass re-drafts from the verified token; the controller
//   just keeps committing verified_tok every pass, so no commit is ever lost or
//   duplicated across accept / reject / rollback.
//
//----------------------------------------------------------------------------
// PARAMETERS
//   TOKW     -- token-index width (default 16; GLM-5.2 vocab fits in 18b).
//   DRAFT_K  -- number of MTP drafts per pass.  GLM-5.2 ships K=1; the storage
//               and the +1-bonus accounting are written K-ready, but the K=1
//               verification path is the one implemented (a K>1 chain would
//               verify drafts in order, accepting the longest correct prefix).
//
// DISCIPLINE: synchronous active-high reset, every output registered, no latch,
//   no combinational loop, deterministic -- pure integer / control logic.
//============================================================================
module spec_decode_seq #(
    parameter integer TOKW    = 16,
    parameter integer DRAFT_K = 1
)(
    input  wire                 clk,
    input  wire                 rst,           // sync, active-high
    input  wire                 start,         // 1-cycle pulse: arm the loop

    // ---- per completed main-model pass (pass_valid is a 1-cycle pulse) ----
    input  wire                 pass_valid,    // a main pass completed this cycle
    input  wire [TOKW-1:0]      verified_tok,  // true token for the verified pos
    input  wire [TOKW-1:0]      draft_tok,     // MTP draft for the NEXT pos
    input  wire                 draft_present, // verifying a prior-pass draft?

    // ---- committed-token stream (1 or 2 beats per pass) ----
    output reg                  commit_valid,
    output reg  [TOKW-1:0]      commit_tok,
    output reg                  accepted,      // pulse: prior draft accepted

    // ---- running counters ----
    output reg  [31:0]          total_tokens,  // committed tokens so far
    output reg  [31:0]          main_passes,   // completed main passes
    output reg  [31:0]          accepts,       // drafts accepted
    output reg  [31:0]          rejects        // drafts rejected
);
    // K=1 -> exactly 2 committed tokens on an accept (verified + 1 bonus draft).
    localparam integer MAX_COMMITS_PER_PASS = DRAFT_K + 1;

    // K-ready draft store (K=1 uses slot 0).  Flag marks a draft is held.
    reg [TOKW-1:0] pending_draft [0:DRAFT_K-1];
    reg            have_draft;     // a prior-pass draft is stored
    reg            running;        // armed by `start`

    // queued bonus (accepted-draft) commit, emitted one cycle after the verified
    reg            second_pending;
    reg [TOKW-1:0] second_tok;

    // ---- combinational verify/accept decision (no state, no loop) ----
    wire do_verify = pass_valid & (running | start) & draft_present & have_draft;
    wire accept    = do_verify & (verified_tok == pending_draft[0]);

    integer i;
    always @(posedge clk) begin
        if (rst) begin
            running        <= 1'b0;
            have_draft     <= 1'b0;
            second_pending <= 1'b0;
            second_tok     <= {TOKW{1'b0}};
            commit_valid   <= 1'b0;
            commit_tok     <= {TOKW{1'b0}};
            accepted       <= 1'b0;
            total_tokens   <= 32'd0;
            main_passes    <= 32'd0;
            accepts        <= 32'd0;
            rejects        <= 32'd0;
            for (i = 0; i < DRAFT_K; i = i + 1)
                pending_draft[i] <= {TOKW{1'b0}};
        end else begin
            // pulse defaults (overridden below)
            commit_valid <= 1'b0;
            accepted     <= 1'b0;

            if (start)
                running <= 1'b1;

            if (pass_valid & (running | start)) begin
                // (1) ALWAYS commit the verified token (advance 1)
                commit_valid <= 1'b1;
                commit_tok   <= verified_tok;
                main_passes  <= main_passes + 32'd1;
                accepted     <= accept;

                if (accept) begin
                    // (2)(3) ACCEPT: commit the prior draft too (advance a 2nd)
                    total_tokens   <= total_tokens + MAX_COMMITS_PER_PASS[31:0];
                    accepts        <= accepts + 32'd1;
                    second_pending <= 1'b1;
                    second_tok     <= pending_draft[0];
                end else begin
                    total_tokens <= total_tokens + 32'd1;
                    // (4) REJECT: prior draft present but mispredicted -> discard
                    if (do_verify)
                        rejects <= rejects + 32'd1;
                end

                // store THIS pass's fresh draft for the next pass to verify.
                // (rollback == simply overwriting/discarding -- no unwind state)
                pending_draft[0] <= draft_tok;
                have_draft       <= 1'b1;
            end else if (second_pending) begin
                // emit the queued bonus (accepted-draft) commit, one cycle later
                commit_valid   <= 1'b1;
                commit_tok     <= second_tok;
                second_pending <= 1'b0;
            end
        end
    end

    // K>1 verification chain is not yet implemented (storage is K-ready).
    // For the default DRAFT_K=1 this generate is empty -> no effect on lint/synth.
    generate
        if (DRAFT_K != 1) begin : g_unsupported_k
            initial $fatal(1, "spec_decode_seq: DRAFT_K>1 not yet implemented");
        end
    endgenerate
endmodule
