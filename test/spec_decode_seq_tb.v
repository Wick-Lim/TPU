`timescale 1ns/1ps
//============================================================================
// spec_decode_seq_tb.v -- TB for the GLM-5.2 MTP speculative-decode controller
//----------------------------------------------------------------------------
// PART A -- CORRECTNESS vs an INDEPENDENT, X-aware software model.
//   A behavioral re-derivation of the draft/verify/accept-reject loop (the
//   `gold_*` tasks below) runs in lock-step with the DUT.  For every pass we
//   assert the committed-token STREAM (order AND values) and ALL FOUR counters
//   (total_tokens, main_passes, accepts, rejects) match bit-exactly, and that
//   no committed token is ever X.  Directed cases: first-pass (no prior draft),
//   all-accept, all-reject, alternating, accept-after-reject.  Then randomized
//   pass sequences driven by a PRNG at several target acceptance rates alpha.
//
// PART B -- EFFECTIVE SPEEDUP vs alpha (the system value).
//   For alpha in {0.0,0.3,0.5,0.6,0.7,0.8,0.9} a long PRNG-driven pass sequence
//   is run and the effective tokens / main-pass (= total_tokens/main_passes) is
//   printed -- it tracks 1+alpha, the FP8-compatible throughput multiplier (a
//   main pass loads the routed-expert weights from Flash ONCE, so tokens/s
//   scales by the effective tokens/pass).
//
//   Emits "ALL <N> TESTS PASSED" + the alpha table; $fatal on any mismatch.
//============================================================================
module spec_decode_seq_tb;
    localparam integer TOKW = 16;
    localparam integer CAP  = 8192;     // capture/golden array depth

    reg                  clk, rst, start;
    reg                  pass_valid, draft_present;
    reg  [TOKW-1:0]      verified_tok, draft_tok;

    wire                 commit_valid, accepted;
    wire [TOKW-1:0]      commit_tok;
    wire [31:0]          total_tokens, main_passes, accepts, rejects;

    integer tests = 0;

    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(1)) dut (
        .clk(clk), .rst(rst), .start(start),
        .pass_valid(pass_valid), .verified_tok(verified_tok),
        .draft_tok(draft_tok), .draft_present(draft_present),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects)
    );

    // ---- clock ----
    initial clk = 1'b0;
    always #5 clk = ~clk;

    // ---- captured DUT commit stream (X-aware) ----
    reg  [TOKW-1:0] got [0:CAP-1];
    integer         got_n = 0;
    reg             cap_en = 1'b0;

    always @(negedge clk) if (cap_en && commit_valid) begin
        if (^commit_tok === 1'bx) begin
            $display("FAIL: X on commit_tok at beat %0d", got_n);
            $fatal(1, "X-aware check failed");
        end
        got[got_n] = commit_tok;
        got_n = got_n + 1;
    end

    // ---- INDEPENDENT golden (re-derived from the spec, not DUT internals) ----
    reg  [TOKW-1:0] exp [0:CAP-1];
    integer         exp_n = 0;
    reg  [TOKW-1:0] g_pending;
    reg             g_have;
    integer         g_total, g_passes, g_acc, g_rej;

    task gold_reset;
        begin
            exp_n=0; g_have=0; g_pending=0;
            g_total=0; g_passes=0; g_acc=0; g_rej=0;
        end
    endtask

    // golden bookkeeping for one pass (mirrors the spec exactly)
    task gold_pass(input dp, input [TOKW-1:0] v, input [TOKW-1:0] d);
        reg acc;
        begin
            acc = dp & g_have & (v == g_pending);
            exp[exp_n]=v; exp_n=exp_n+1;             // ALWAYS commit verified
            g_total=g_total+1; g_passes=g_passes+1;
            if (acc) begin
                exp[exp_n]=g_pending; exp_n=exp_n+1; // bonus = accepted draft
                g_total=g_total+1; g_acc=g_acc+1;
            end else if (dp & g_have) begin
                g_rej=g_rej+1;
            end
            g_pending=d; g_have=1;
        end
    endtask

    // ---- DUT driver: one pass pulse, spaced 3 cycles (room for the bonus) ----
    task drive_pass(input dp, input [TOKW-1:0] v, input [TOKW-1:0] d);
        begin
            @(negedge clk);
            pass_valid=1'b1; draft_present=dp; verified_tok=v; draft_tok=d;
            @(negedge clk);
            pass_valid=1'b0; draft_present=1'b0; verified_tok=0; draft_tok=0;
            @(negedge clk);  // idle: queued bonus beat (if any) emits here
        end
    endtask

    // apply the same pass to golden + DUT
    task do_pass(input dp, input [TOKW-1:0] v, input [TOKW-1:0] d);
        begin
            gold_pass(dp, v, d);
            drive_pass(dp, v, d);
        end
    endtask

    // ---- compare captured stream + counters against golden ----
    task check_scenario(input [255:0] name);
        integer k;
        begin
            if (got_n !== exp_n) begin
                $display("FAIL[%0s]: stream length got %0d exp %0d", name, got_n, exp_n);
                $fatal(1, "stream length mismatch");
            end
            for (k=0; k<exp_n; k=k+1) begin
                if (got[k] !== exp[k]) begin
                    $display("FAIL[%0s]: beat %0d got %0d exp %0d", name, k, got[k], exp[k]);
                    $fatal(1, "stream value mismatch");
                end
                tests = tests + 1;
            end
            if (total_tokens !== g_total)
                begin $display("FAIL[%0s]: total_tokens %0d exp %0d", name, total_tokens, g_total);
                      $fatal(1,"total_tokens mismatch"); end
            if (main_passes !== g_passes)
                begin $display("FAIL[%0s]: main_passes %0d exp %0d", name, main_passes, g_passes);
                      $fatal(1,"main_passes mismatch"); end
            if (accepts !== g_acc)
                begin $display("FAIL[%0s]: accepts %0d exp %0d", name, accepts, g_acc);
                      $fatal(1,"accepts mismatch"); end
            if (rejects !== g_rej)
                begin $display("FAIL[%0s]: rejects %0d exp %0d", name, rejects, g_rej);
                      $fatal(1,"rejects mismatch"); end
            // invariant: nothing lost/duplicated -> stream length == total_tokens
            if (got_n !== g_total) $fatal(1, "stream/total invariant broken");
            tests = tests + 5;
            $display("PASS[%0s]: passes=%0d total=%0d acc=%0d rej=%0d (eff x100=%0d)",
                     name, main_passes, total_tokens, accepts, rejects,
                     (main_passes==0)?0:(total_tokens*100)/main_passes);
        end
    endtask

    // ---- reset + arm; restart capture+golden for a fresh scenario ----
    task arm;
        begin
            rst=1'b1; start=1'b0; pass_valid=1'b0; draft_present=1'b0;
            verified_tok=0; draft_tok=0; cap_en=1'b0;
            @(negedge clk); @(negedge clk);
            rst=1'b0;
            got_n=0; gold_reset(); cap_en=1'b1;
            @(negedge clk); start=1'b1; @(negedge clk); start=1'b0;  // 1-cycle arm
        end
    endtask

    // ---- PRNG-driven pass generator with target acceptance alpha (per-mille) ----
    //   accept => verified == prior draft (so DUT+golden both ACCEPT);
    //   reject => verified = prior_draft ^ nonzero  (guaranteed mismatch).
    //   Returns nothing; caller checks the stream/counters afterwards.
    integer seed;
    reg [TOKW-1:0] gprev;            // last draft we fed (what next pass verifies)
    task run_alpha(input integer n, input integer alpha_pm);
        integer i, r;
        reg acc;
        reg [TOKW-1:0] v, d;
        begin
            arm;
            // first pass: nothing to verify (draft_present=0)
            d = $random(seed);
            do_pass(1'b0, 16'h0001, d);
            gprev = d;
            for (i=1; i<n; i=i+1) begin
                r = $random(seed); if (r<0) r=-r; r = r % 1000;
                acc = (r < alpha_pm);
                v = acc ? gprev : (gprev ^ 16'hA53C);   // ^nonzero => sure mismatch
                d = $random(seed);
                do_pass(1'b1, v, d);
                gprev = d;
            end
            @(negedge clk);
        end
    endtask

    integer ai, alpha_pm, eff100, ideal100;
    integer sweep [0:6];
    initial begin
        // ============================ PART A ============================
        // --- directed: FIRST PASS only (no prior draft) ---
        arm;
        do_pass(1'b0, 16'd100, 16'd101);   // commit 100; store draft=101
        @(negedge clk);
        check_scenario("first-pass");      // 1 pass, 1 tok, 0 acc, 0 rej

        // --- directed: ALL-ACCEPT (2 tok/pass) ---
        arm;
        do_pass(1'b0, 16'd100, 16'd101);   // commit 100; store 101
        do_pass(1'b1, 16'd101, 16'd102);   // 101==101 ACCEPT -> 101,101
        do_pass(1'b1, 16'd102, 16'd103);   // ACCEPT -> 102,102
        do_pass(1'b1, 16'd103, 16'd104);   // ACCEPT -> 103,103
        @(negedge clk);
        check_scenario("all-accept");      // 4 passes, 7 tok, 3 acc -> eff 1.75

        // --- directed: ALL-REJECT (1 tok/pass) ---
        arm;
        do_pass(1'b0, 16'd200, 16'd201);
        do_pass(1'b1, 16'd210, 16'd211);   // 210!=201 REJECT
        do_pass(1'b1, 16'd220, 16'd221);   // REJECT
        do_pass(1'b1, 16'd230, 16'd231);   // REJECT
        @(negedge clk);
        check_scenario("all-reject");      // 4 passes, 4 tok, 3 rej -> eff 1.00

        // --- directed: ALTERNATING ---
        arm;
        do_pass(1'b0, 16'd300, 16'd301);
        do_pass(1'b1, 16'd301, 16'd999);   // ACCEPT -> 301,301; store 999
        do_pass(1'b1, 16'd400, 16'd401);   // 400!=999 REJECT; store 401
        do_pass(1'b1, 16'd401, 16'd555);   // ACCEPT -> 401,401; store 555
        do_pass(1'b1, 16'd500, 16'd501);   // 500!=555 REJECT
        @(negedge clk);
        check_scenario("alternating");     // 5 passes, 7 tok, 2 acc, 2 rej

        // --- directed: ACCEPT-AFTER-REJECT (recovery has no unwind state) ---
        arm;
        do_pass(1'b0, 16'd600, 16'd601);
        do_pass(1'b1, 16'd700, 16'd701);   // 700!=601 REJECT; store 701
        do_pass(1'b1, 16'd701, 16'd702);   // 701==701 ACCEPT -> 701,701; store 702
        do_pass(1'b1, 16'd702, 16'd703);   // ACCEPT -> 702,702
        @(negedge clk);
        check_scenario("accept-after-reject"); // 4 passes, 6 tok, 2 acc, 1 rej

        // --- randomized alpha sweep checked vs independent golden ---
        seed = 32'hC0FFEE01;
        run_alpha(40, 1000); check_scenario("rand-a1.0");
        run_alpha(40, 0);    check_scenario("rand-a0.0");
        run_alpha(60, 500);  check_scenario("rand-a0.5");
        run_alpha(60, 300);  check_scenario("rand-a0.3");
        run_alpha(60, 700);  check_scenario("rand-a0.7");
        run_alpha(80, 850);  check_scenario("rand-a0.85");

        // --- counters non-X after a fresh reset ---
        arm;
        if (^{total_tokens,main_passes,accepts,rejects} === 1'bx)
            $fatal(1, "counters are X after reset");
        tests = tests + 1;

        $display("ALL %0d TESTS PASSED", tests);

        // ============================ PART B ============================
        // Effective tokens/main-pass vs alpha -> tracks 1+alpha (the throughput
        // multiplier).  Each row is ALSO checked vs the independent golden.
        sweep[0]=0; sweep[1]=300; sweep[2]=500; sweep[3]=600;
        sweep[4]=700; sweep[5]=800; sweep[6]=900;
        seed = 32'h5EED1234;
        $display("");
        $display("==== PART B: effective tokens / main-pass vs alpha (n=1000 passes) ====");
        $display("  alpha | eff tok/pass (measured) | ideal 1+alpha | throughput x");
        $display("  ------+-------------------------+---------------+-------------");
        for (ai=0; ai<7; ai=ai+1) begin
            alpha_pm = sweep[ai];
            run_alpha(1000, alpha_pm);
            // validate the run against golden (counters + stream) too
            check_scenario("partB");
            eff100   = (total_tokens*100)/main_passes;     // measured eff*100
            ideal100 = 100 + (alpha_pm/10);                // (1+alpha)*100
            $display("   0.%01d0 |          %0d.%02d           |     %0d.%02d      |    %0d.%02dx",
                     alpha_pm/100,
                     eff100/100, eff100%100,
                     ideal100/100, ideal100%100,
                     eff100/100, eff100%100);
        end
        $display("  (effective tokens/pass tracks 1+alpha => tokens/s scales by ~1+alpha)");

        $finish;
    end

    // safety timeout
    initial begin
        #50000000;
        $display("FAIL: timeout");
        $fatal(1, "timeout");
    end
endmodule
