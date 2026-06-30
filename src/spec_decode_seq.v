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
//   DRAFT_K  -- number of MTP drafts per pass.  GLM-5.2 ships K=1 (the rolling
//               path, g_k1).  DRAFT_K>1 selects the BATCH verifier (g_kn): it
//               consumes K chained drafts d_1..d_K + the model's K+1 true
//               argmaxes m_1..m_{K+1} per pass and commits the LONGEST ACCEPTED
//               PREFIX m_1..m_{p+1} (p drafts accepted in order, then the one
//               always-correct model token).  Every committed token is the
//               model's greedy argmax, so the committed stream == greedy for
//               ANY K.  The two paths are generate-split: K=1 is byte-identical.
//
// DISCIPLINE: synchronous active-high reset, every output registered, no latch,
//   no combinational loop, deterministic -- pure integer / control logic.
//============================================================================
module spec_decode_seq #(
    parameter integer TOKW    = 16,
    parameter integer DRAFT_K = 1,
    // ---- derived (do NOT override) -- width for a 0..K draft count ----
    parameter integer DKW     = (DRAFT_K <= 1) ? 1 : $clog2(DRAFT_K + 1)
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
    output reg  [31:0]          rejects,       // drafts rejected

    // ---- K>1 BATCH interface (used ONLY when DRAFT_K>1; ignored at K=1 so the
    //      committed K=1 instantiations stay drop-in / byte-identical) ----
    //   Per pass, for the K positions t+1..t+K after the last committed token:
    //     draft_vec[j*TOKW +: TOKW] = j-th chained MTP draft  d_{j+1}   (j=0..K-1)
    //     truth_vec[j*TOKW +: TOKW] = model true argmax        m_{j+1}   (j=0..K)
    //         (truth_vec carries K+1 entries: m_1..m_K plus the bonus m_{K+1})
    //     n_draft                   = #valid drafts this pass (<= K)
    input  wire [DRAFT_K*TOKW-1:0]     draft_vec,
    input  wire [(DRAFT_K+1)*TOKW-1:0] truth_vec,
    input  wire [DKW-1:0]              n_draft
);
    // K=1 -> exactly 2 committed tokens on an accept (verified + 1 bonus draft);
    // the +1 bonus is added inline to total_tokens (see accept path below).
    //
    // The two decode schemes are split by generate so EXACTLY ONE is elaborated:
    //   g_k1 -- DRAFT_K==1 : the ORIGINAL rolling MTP controller, verbatim.  This
    //           keeps the committed test/spec_decode_seq_tb.v (621 tests) and the
    //           formal harness byte-identical (the K>1 ports are dead/sunk here).
    //   g_kn -- DRAFT_K> 1 : the BATCH multi-token verifier (longest accepted
    //           prefix).  It constant-folds away entirely at K=1.
    generate
    if (DRAFT_K == 1) begin : g_k1
        // the K>1 batch ports are inactive in this branch -> sink them
        /* verilator lint_off UNUSEDSIGNAL */
        wire _u_kn = &{1'b0, draft_vec, truth_vec, n_draft};
        /* verilator lint_on UNUSEDSIGNAL */

        // K-ready draft store (K=1 uses slot 0).  Flag marks a draft is held.
        reg [TOKW-1:0] pending_draft [0:DRAFT_K-1];
        reg            have_draft;     // a prior-pass draft is stored
        reg            running;        // armed by `start`

        // queued bonus (accepted-draft) commit, emitted one cycle after verified
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

                    // advance 1 token ALWAYS + 1 BONUS when the prior draft was
                    // accepted (K=1).  Single 32-bit adder instead of muxed adders.
                    total_tokens <= total_tokens + 32'd1 + {31'd0, accept};

                    if (accept) begin
                        // (2)(3) ACCEPT: commit the prior draft too (advance a 2nd)
                        accepts        <= accepts + 32'd1;
                        second_pending <= 1'b1;
                        second_tok     <= pending_draft[0];
                    end else begin
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
    end else begin : g_kn
        //====================================================================
        // DRAFT_K > 1 : BATCH multi-token verify, commit the LONGEST ACCEPTED
        //   PREFIX.  Per main pass the harness/top presents K chained MTP drafts
        //   d_1..d_K (draft_vec) for positions t+1..t+K and the model's K+1 true
        //   argmaxes m_1..m_{K+1} (truth_vec) for those positions plus the one
        //   after.  We accept the longest prefix p (0..n_draft) with d_i==m_i for
        //   every i<=p (the scan stops at the FIRST mismatch -- a later matching
        //   draft can NOT be accepted because an earlier one was wrong), then
        //   COMMIT m_1..m_{p+1}: the p accepted drafts (== their matching m_i) and
        //   the always-correct model token m_{p+1} at the stop point (the "+1" /
        //   the bonus when p==K).  Every committed token is therefore the model's
        //   OWN greedy argmax -> the committed stream == the non-speculative
        //   (greedy) stream for ANY K; rejected drafts past the prefix never
        //   commit.  total_tokens advances by (#accepted + 1) = (p + 1) per pass.
        //
        //   COMMIT STREAM: the 1-beat port emits m_1 on the pass cycle and drains
        //   m_2..m_{p+1} on the next p idle cycles, so pass_valid pulses must be
        //   spaced >= K+1 cycles apart (a real main pass is far longer).  No
        //   rollback state: a mismatch simply truncates the prefix; obuf is
        //   overwritten each pass like any other.
        //====================================================================
        localparam integer OCW = $clog2(DRAFT_K + 2);   // width for a 0..K+1 count

        localparam integer OIW = $clog2(DRAFT_K + 1);   // obuf index width (0..K)

        // the K=1 rolling ports are inactive in this branch -> sink them
        /* verilator lint_off UNUSEDSIGNAL */
        wire _u_k1 = &{1'b0, verified_tok, draft_tok, draft_present};
        /* verilator lint_on UNUSEDSIGNAL */

        reg            running;
        reg [TOKW-1:0] obuf [0:DRAFT_K];   // queued commits m_1..m_{p+1}
        reg [OCW-1:0]  ocnt;               // #tokens to emit this pass (= p+1)
        reg [OCW-1:0]  oidx;               // next drain index (1..p)
        reg            draining;

        integer j;

        // ---- COMBINATIONAL longest-accepted-prefix (no state, no loop) ----
        //   p = #leading drafts that equal the model's argmax, stopping at the
        //   FIRST mismatch (a later match can NOT be accepted once an earlier
        //   draft was wrong).  Pure function of the batch inputs -> a wire.
        function automatic [OCW-1:0] acc_prefix;
            input [DRAFT_K*TOKW-1:0]     dv;
            input [(DRAFT_K+1)*TOKW-1:0] tv;
            input [OCW-1:0]              ndi;
            integer    fj;
            reg        fb;
            reg [OCW-1:0] fp;
            begin
                fp = {OCW{1'b0}};
                fb = 1'b0;
                for (fj = 0; fj < DRAFT_K; fj = fj + 1) begin
                    if (!fb && (fj < ndi) &&
                        (dv[fj*TOKW +: TOKW] == tv[fj*TOKW +: TOKW]))
                        fp = fp + 1'b1;
                    else
                        fb = 1'b1;
                end
                acc_prefix = fp;
            end
        endfunction

        // clamp the presented draft count to K, then the accepted prefix p
        localparam [OCW-1:0] K_OCW = DRAFT_K[OCW-1:0];   // K, OCW-bit (no overflow)
        /* verilator lint_off WIDTHEXPAND */
        wire [OCW-1:0] nd_ext = n_draft;                 // zero-extend DKW -> OCW
        /* verilator lint_on WIDTHEXPAND */
        wire [OCW-1:0] nd_w   = (nd_ext > K_OCW) ? K_OCW : nd_ext;
        wire [OCW-1:0] pfx_w  = acc_prefix(draft_vec, truth_vec, nd_w);  // p (0..K)

        always @(posedge clk) begin
            if (rst) begin
                running      <= 1'b0;
                draining     <= 1'b0;
                ocnt         <= {OCW{1'b0}};
                oidx         <= {OCW{1'b0}};
                commit_valid <= 1'b0;
                commit_tok   <= {TOKW{1'b0}};
                accepted     <= 1'b0;
                total_tokens <= 32'd0;
                main_passes  <= 32'd0;
                accepts      <= 32'd0;
                rejects      <= 32'd0;
                for (j = 0; j <= DRAFT_K; j = j + 1)
                    obuf[j] <= {TOKW{1'b0}};
            end else begin
                // pulse defaults (overridden below)
                commit_valid <= 1'b0;
                accepted     <= 1'b0;

                if (start)
                    running <= 1'b1;

                if (pass_valid & (running | start)) begin
                    // counters: advance (p) accepted + 1 always-correct token
                    main_passes  <= main_passes + 32'd1;
                    total_tokens <= total_tokens + {{(32-OCW){1'b0}}, pfx_w} + 32'd1;
                    accepts      <= accepts + {{(32-OCW){1'b0}}, pfx_w};      // p accepted
                    rejects      <= rejects + {{(32-OCW){1'b0}}, (nd_w - pfx_w)}; // discarded
                    accepted     <= (pfx_w != {OCW{1'b0}});

                    // commit m_1 now; queue m_2..m_{p+1} for the drain cycles
                    commit_valid <= 1'b1;
                    commit_tok   <= truth_vec[0 +: TOKW];     // m_1 (always correct)
                    for (j = 1; j <= DRAFT_K; j = j + 1)
                        obuf[j] <= truth_vec[j*TOKW +: TOKW];
                    ocnt     <= pfx_w + 1'b1;                 // p+1 total beats
                    oidx     <= {{(OCW-1){1'b0}}, 1'b1};      // m_1 already emitted
                    draining <= (pfx_w >= 1);
                end else if (draining) begin
                    // emit the next queued model token, one per idle cycle
                    commit_valid <= 1'b1;
                    commit_tok   <= obuf[oidx[OIW-1:0]];
                    if ((oidx + 1'b1) >= ocnt)
                        draining <= 1'b0;
                    oidx <= oidx + 1'b1;
                end
            end
        end
    end
    endgenerate
endmodule
