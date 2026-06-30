`timescale 1ns/1ps
//============================================================================
// spec_decode_seq_ind_fv.v -- k-INDUCTION (UNBOUNDED) harness for
//                             src/spec_decode_seq.v  (READ-ONLY DUT, K=1).
//----------------------------------------------------------------------------
// WHY A SEPARATE HARNESS  (vs the committed spec_decode_seq_fv.v BMC harness):
//   The committed harness asserts STRICT UNSIGNED monotonicity
//       total_tokens >= prev_total ,  main_passes >= prev_passes , ...
//   Those are NOT k-inductive -- and in fact NOT even true unboundedly: every
//   counter is a free-running 32-bit register, so after 2^32 events it WRAPS
//   (0xFFFFFFFF + 1 -> 0) and the unsigned `>=` is violated.  yosys-smtbmc -i
//   finds exactly this: a state with main_passes==0xFFFFFFFF that increments to
//   0.  No strengthening invariant can rescue an property that is mathematically
//   false on the reachable state space -- so P1..P4 stay BOUNDED (see report).
//
//   This harness instead proves the GENUINELY-UNBOUNDED safety core, which is
//   what the monotonic counters actually guarantee cycle-by-cycle:
//
//   (U1) RELATIONAL EQUALITY  total_tokens == main_passes + accepts   (mod 2^32)
//        -- exact: every pass adds 1 to main_passes and (1+accept) to total and
//        accept to accepts; resets zero all three.  1-INDUCTIVE on its own and
//        the key STRENGTHENING invariant: it excludes the spurious "partial
//        reset" inductive states (e.g. main_passes==0 while total!=0) that make
//        naive induction fail.
//        (The order relation accepts+rejects <= main_passes is NOT included:
//        like strict monotonicity it is an unsigned inequality that breaks when
//        main_passes wraps at 2^32 -- so it is bounded-only, not in this core.)
//   (U3) PER-CYCLE MODULAR INCREMENT BOUNDS (the honest replacement for strict
//        monotonicity -- TRUE even across a 2^32 wrap, since modular subtraction
//        recovers the true delta):
//          d_total   := total_tokens - prev_total  in {0,1,2}
//          d_passes  := main_passes  - prev_passes in {0,1}
//          d_accepts := accepts      - prev_accepts in {0,1}
//          d_rejects := rejects      - prev_rejects in {0,1}
//        (U3 d_total<=2 is the committed P5, now proven UNBOUNDED.)
//   (U4) STEP-FORM MONOTONICITY (the unbounded form of P1..P4): each counter
//        either is >= its previous value OR a wrap just happened
//        (prev == 0xFFFFFFFF), i.e. it never decreases except by wraparound.
//        Equivalent to d in {0,1,2}; stated explicitly so the report can map it
//        back to the original monotonicity intent.
//
//   OPTIONAL no-overflow corollary: under the explicit assumption that no
//   counter is at its max (NO_OVF=1), STRICT monotonicity P1..P4 IS k-inductive
//   -- i.e. the counters are monotone in the entire pre-wrap regime.  Selected
//   by `chparam -set NO_OVF 1`; OFF by default so the unconditional core stands
//   on its own.
//
//   RESET GATING: a reset legally zeroes the counters; the modular delta across
//   a reset (0 - big) is huge, so U3/U4 are checked only on transitions with no
//   reset at either endpoint (~rst & ~prev_rst & seen).  U1/U2 hold in EVERY
//   state (including reset -> all zero) and are asserted ungated.
//
//   TOKW kept tiny (4); the counters are full 32-bit so the wrap behaviour and
//   all relational invariants are width-faithful.
//============================================================================
module spec_decode_seq_ind_fv #(
    parameter integer TOKW    = 4,
    parameter integer DRAFT_K = 1,
    parameter integer NO_OVF  = 0   // 1 => assume no counter at max (strict-mono corollary)
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

    // ---- previous-cycle snapshots ----
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

    // step is checked only across a reset-free transition.
    // NB: gates use the architectural reset signals (rst/prev_rst) and NOT the
    // harness `seen` flag.  In k-induction `seen` is a FREE register in the base
    // state, so a `seen`-gated invariant can be DODGED (solver picks seen=0 with
    // garbage counters) -- whereas `rst` is pinned: the forced first-cycle reset
    // makes every reset-free state carry the relational invariants, so the
    // hypothesis actually constrains the inductive base.
    wire chk = ~prev_rst & ~rst;

    localparam [31:0] MAXU = 32'hFFFF_FFFF;

    // modular per-cycle deltas (true increment even across wrap)
    wire [31:0] d_total   = total_tokens - prev_total;
    wire [31:0] d_passes  = main_passes  - prev_passes;
    wire [31:0] d_accepts = accepts      - prev_accepts;
    wire [31:0] d_rejects = rejects      - prev_rejects;

    // ---- assume legal protocol (arm-once; trims silly traces only) ----
    always @(posedge clk) begin
        if (running_shadow) assume (~start);
    end

    // ---- OPTIONAL no-overflow assumption (strict-mono corollary) ----
    generate if (NO_OVF != 0) begin : g_noovf
        always @(posedge clk) begin
            assume (main_passes  != MAXU);
            assume (total_tokens != MAXU);
            assume (total_tokens != MAXU - 32'd1); // total can jump by 2
            assume (accepts      != MAXU);
            assume (rejects      != MAXU);
        end
    end endgenerate

    // single-state invariants are checked outside a reset cycle.  `~rst`
    // excludes both the forced t=0 reset (free uninitialised registers) and any
    // later reset cycle; it is the architectural gate that survives k-induction.
    wire act = ~rst;

    // ---- UNBOUNDED-TRUE invariants + safety ----
    always @(posedge clk) begin
        if (act) begin
            // (U1) relational equality -- holds in EVERY operational state and is
            //      MODULAR-SAFE (both sides wrap together at 2^32), so it is a
            //      genuinely UNBOUNDED invariant and the crucial STRENGTHENING
            //      fact: it excludes partial-reset / inconsistent inductive
            //      states (e.g. main_passes==0 while total!=0).
            u1_total_eq : assert (total_tokens == (main_passes + accepts));
        end

        // NB: no separate "snapshot consistency" assertion is needed.  prev_* is
        // a pure delay register of the DUT counters, so in any induction step the
        // transition relation forces prev_*(t) == counters(t-1); and U1 is
        // asserted at t-1 (gate ~rst there is implied by chk's ~prev_rst), so the
        // pre-state the step properties read is already pinned consistent.

        if (chk) begin
            // (U3) per-cycle modular increment bounds (== honest monotonicity)
            u3_dtotal   : assert (d_total   <= 32'd2);
            u3_dpasses  : assert (d_passes  <= 32'd1);
            u3_daccepts : assert (d_accepts <= 32'd1);
            u3_drejects : assert (d_rejects <= 32'd1);

            // (U4) step-form monotonicity: non-decreasing EXCEPT across a wrap.
            u4_total_mono   : assert ((total_tokens >= prev_total) || (prev_total == MAXU) || (prev_total == MAXU-32'd1));
            u4_passes_mono  : assert ((main_passes  >= prev_passes)  || (prev_passes  == MAXU));
            u4_accepts_mono : assert ((accepts      >= prev_accepts) || (prev_accepts == MAXU));
            u4_rejects_mono : assert ((rejects      >= prev_rejects) || (prev_rejects == MAXU));

            // accepts and rejects never both advance in the same cycle (K=1: a
            // pass is accept XOR reject XOR neither).
            u5_excl : assert (~((d_accepts != 32'd0) && (d_rejects != 32'd0)));

            if (NO_OVF != 0) begin
                // strict P1..P4 -- inductive ONLY under the no-overflow assume.
                p1_total_mono   : assert (total_tokens >= prev_total);
                p2_passes_mono  : assert (main_passes  >= prev_passes);
                p3_accepts_mono : assert (accepts      >= prev_accepts);
                p4_rejects_mono : assert (rejects      >= prev_rejects);
            end
        end
    end
endmodule
