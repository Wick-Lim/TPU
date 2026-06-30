`timescale 1ns/1ps
//============================================================================
// spec_decode_seq_k_tb.v -- BINDING spec==greedy check ACROSS DRAFT_K = 1,2,3
//                           for the GLM-5.2 MTP speculative-decode controller.
//----------------------------------------------------------------------------
// THE OVERRIDING PROPERTY (docs/IMPROVEMENT_PLAN.md P2.2, repo principle):
//   The committed-token stream of the speculative loop MUST equal the
//   NON-speculative (greedy) stream for ANY K -- rejected drafts NEVER commit.
//   This TB proves that for K=1 (rolling MTP path g_k1) AND K=2,3 (batch
//   longest-accepted-prefix path g_kn) against INDEPENDENT, X-aware goldens.
//
// PART A -- spec==greedy for K=1,2,3:
//   * K=1 (rolling): an independent re-derivation of the accounting model
//     (commit the model argmax `verified_tok` every pass; on ACCEPT commit the
//     accepted draft, which == verified_tok so it too is a model argmax). Every
//     committed token is a model argmax; a mispredicted (rejected) draft is
//     NEVER emitted. Asserted bit-exact incl. all 4 counters + no-X.
//   * K=2,3 (batch): a POSITION-ACCURATE greedy reference G. Per pass we present
//     K chained drafts (the first `first_miss` matching G, the rest a guaranteed
//     mismatch) + the K+1 true argmaxes G[c..c+K]; the DUT must commit exactly
//     G[c..c+p], p=min(first_miss,n_draft). The captured stream is asserted to
//     equal the SAME greedy G for every acceptance pattern.
//   Coverage for every K: all-accept, all-reject, mixed/random, and
//   reject-at-each-position (first mismatch walked across positions 0..K).
//
// PART B -- effective accepted-tokens/pass vs per-position acceptance alpha,
//   for K=1,2,3 (the throughput benefit + the chained-MTP decay). Two models:
//     (i)  FLAT  : each of the K positions matches independently w.p. alpha.
//          Because the prefix STOPS at the first miss, eff_K = 1+sum_{j=1..K}
//          alpha^j even with a flat per-position alpha (deeper positions are
//          reached only if all shallower drafts hit).
//     (ii) CHAINED-DECAY : position j matches w.p. alpha^(j+1) (the realistic
//          1-MTP-layer chain: each extra autoregressive draft step is worse),
//          eff_K = 1+sum_{j=1..K} prod_{i=1..j} alpha^i.
//   Measured eff (= total_tokens/main_passes) is printed against theory.
//
// Emits "ALL <N> TESTS PASSED"; $fatal on ANY spec!=greedy / counter mismatch.
//============================================================================
module spec_decode_seq_k_tb;
    localparam integer TOKW = 16;
    localparam integer CAP  = 16384;     // capture/greedy array depth
    localparam [TOKW-1:0] MISMASK = 16'h5A3C;  // flip bits => guaranteed mismatch

    integer tests = 0;
    integer seed;

    reg clk = 1'b0;
    always #5 clk = ~clk;

    // shared greedy reference sequence G (model argmax at each absolute pos).
    // Nonzero so X is detectable; reused unchanged for every K and scenario.
    reg [TOKW-1:0] G [0:CAP-1];
    integer gi;

    //========================================================================
    // K = 1  (rolling MTP controller, g_k1) -- ports verified_tok/draft_tok/...
    //========================================================================
    reg               rst1, st1;
    reg               pv1, dp1;
    reg  [TOKW-1:0]   vt1, dt1;
    wire              cv1, ac1_pulse;
    wire [TOKW-1:0]   ct1;
    wire [31:0]       tt1, mp1, ac1, rj1;

    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(1)) dut1 (
        .clk(clk), .rst(rst1), .start(st1),
        .pass_valid(pv1), .verified_tok(vt1), .draft_tok(dt1), .draft_present(dp1),
        .commit_valid(cv1), .commit_tok(ct1), .accepted(ac1_pulse),
        .total_tokens(tt1), .main_passes(mp1), .accepts(ac1), .rejects(rj1),
        .draft_vec(16'd0), .truth_vec(32'd0), .n_draft(1'd0)
    );

    // captured K=1 commit stream (X-aware)
    reg  [TOKW-1:0] got1 [0:CAP-1];
    integer got1_n = 0;
    reg cap1 = 1'b0;
    always @(negedge clk) if (cap1 && cv1) begin
        if (^ct1 === 1'bx) begin $display("FAIL: X on K1 commit beat %0d", got1_n);
                                 $fatal(1,"X"); end
        got1[got1_n] = ct1; got1_n = got1_n + 1;
    end

    // independent K=1 golden (accounting re-derivation, mirrors g_k1 spec)
    reg  [TOKW-1:0] e1 [0:CAP-1];
    integer e1_n; reg [TOKW-1:0] gp1; reg gh1;
    integer t1_tot, t1_pas, t1_acc, t1_rej;

    task k1_reset; begin
        e1_n=0; gh1=0; gp1=0; t1_tot=0; t1_pas=0; t1_acc=0; t1_rej=0; end
    endtask

    task k1_arm; begin
        rst1=1; st1=0; pv1=0; dp1=0; vt1=0; dt1=0; cap1=0;
        @(negedge clk); @(negedge clk); rst1=0;
        got1_n=0; k1_reset(); cap1=1;
        @(negedge clk); st1=1; @(negedge clk); st1=0;
    end endtask

    // one rolling pass; updates golden + drives DUT (spaced 3 cycles for bonus)
    task k1_pass(input dp, input [TOKW-1:0] v, input [TOKW-1:0] d);
        reg acc;
        begin
            // golden
            acc = dp & gh1 & (v == gp1);
            e1[e1_n]=v; e1_n=e1_n+1; t1_tot=t1_tot+1; t1_pas=t1_pas+1;
            if (acc) begin e1[e1_n]=gp1; e1_n=e1_n+1; t1_tot=t1_tot+1; t1_acc=t1_acc+1; end
            else if (dp & gh1) t1_rej=t1_rej+1;
            gp1=d; gh1=1;
            // drive
            @(negedge clk); pv1=1; dp1=dp; vt1=v; dt1=d;
            @(negedge clk); pv1=0; dp1=0; vt1=0; dt1=0;
            @(negedge clk);
        end
    endtask

    task k1_check(input [255:0] name);
        integer k;
        begin
            if (got1_n !== e1_n) begin
                $display("FAIL[K1 %0s]: stream len got %0d exp %0d", name, got1_n, e1_n);
                $fatal(1,"len"); end
            for (k=0;k<e1_n;k=k+1) begin
                if (got1[k] !== e1[k]) begin
                    $display("FAIL[K1 %0s]: beat %0d got %0d exp greedy %0d",name,k,got1[k],e1[k]);
                    $fatal(1,"spec!=greedy"); end
                tests=tests+1;
            end
            if (tt1!==t1_tot)begin $display("FAIL[K1 %0s] total %0d exp %0d",name,tt1,t1_tot);$fatal(1,"t");end
            if (mp1!==t1_pas)begin $display("FAIL[K1 %0s] pass %0d exp %0d",name,mp1,t1_pas);$fatal(1,"p");end
            if (ac1!==t1_acc)begin $display("FAIL[K1 %0s] acc %0d exp %0d",name,ac1,t1_acc);$fatal(1,"a");end
            if (rj1!==t1_rej)begin $display("FAIL[K1 %0s] rej %0d exp %0d",name,rj1,t1_rej);$fatal(1,"r");end
            if (got1_n!==t1_tot) $fatal(1,"K1 stream/total invariant");
            tests=tests+5;
            $display("PASS[K1 %0s] passes=%0d total=%0d acc=%0d rej=%0d (eff x100=%0d)",
                name, mp1, tt1, ac1, rj1, (mp1==0)?0:(tt1*100)/mp1);
        end
    endtask

    //========================================================================
    // K = 2 / K = 3  (batch longest-accepted-prefix, g_kn) -- batch ports.
    //========================================================================
    // ---- K=2 ----
    localparam integer DKW2 = 2;   // $clog2(3)
    reg                 rst2, st2;
    reg                 pv2;
    reg [2*TOKW-1:0]    dv2;
    reg [3*TOKW-1:0]    tv2;
    reg [DKW2-1:0]      nd2;
    wire                cv2;
    wire [TOKW-1:0]     ct2;
    wire [31:0]         tt2, mp2, ac2, rj2;
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(2)) dut2 (
        .clk(clk), .rst(rst2), .start(st2),
        .pass_valid(pv2), .verified_tok(16'd0), .draft_tok(16'd0), .draft_present(1'b0),
        .commit_valid(cv2), .commit_tok(ct2), .accepted(),
        .total_tokens(tt2), .main_passes(mp2), .accepts(ac2), .rejects(rj2),
        .draft_vec(dv2), .truth_vec(tv2), .n_draft(nd2)
    );
    reg  [TOKW-1:0] got2 [0:CAP-1];
    integer got2_n = 0; reg cap2 = 1'b0;
    always @(negedge clk) if (cap2 && cv2) begin
        if (^ct2 === 1'bx) begin $display("FAIL: X on K2 commit beat %0d", got2_n);$fatal(1,"X"); end
        got2[got2_n]=ct2; got2_n=got2_n+1;
    end
    integer c2, p2pas, p2acc, p2rej;
    task k2_arm; begin
        rst2=1; st2=0; pv2=0; dv2=0; tv2=0; nd2=0; cap2=0;
        @(negedge clk); @(negedge clk); rst2=0;
        got2_n=0; c2=0; p2pas=0; p2acc=0; p2rej=0; cap2=1;
        @(negedge clk); st2=1; @(negedge clk); st2=0;
    end endtask
    // nd valid drafts, first_miss = index of first mismatch (K=>all match) => p=min(fm,nd)
    task k2_pass(input integer nd, input integer first_miss);
        integer j, p; reg [TOKW-1:0] m;
        begin
            dv2=0; tv2=0;
            for (j=0;j<=2;j=j+1) tv2[j*TOKW +: TOKW]=G[c2+j];
            for (j=0;j<2;j=j+1) begin
                m=G[c2+j];
                dv2[j*TOKW +: TOKW] = (j<first_miss)? m : (m ^ MISMASK);
            end
            nd2=nd[DKW2-1:0];
            p=(first_miss<nd)?first_miss:nd;
            p2pas=p2pas+1; p2acc=p2acc+p; p2rej=p2rej+(nd-p); c2=c2+p+1;
            @(negedge clk); pv2=1;
            @(negedge clk); pv2=0; dv2=0; tv2=0; nd2=0;
            for (j=0;j<4;j=j+1) @(negedge clk);  // K+2 drain cycles
        end
    endtask
    task k2_check(input [255:0] name);
        integer k; begin
            if (got2_n !== c2) begin $display("FAIL[K2 %0s] len got %0d exp %0d",name,got2_n,c2);$fatal(1,"len"); end
            for (k=0;k<c2;k=k+1) begin
                if (got2[k] !== G[k]) begin
                    $display("FAIL[K2 %0s] beat %0d got %0d exp greedy %0d",name,k,got2[k],G[k]);
                    $fatal(1,"spec!=greedy"); end
                tests=tests+1; end
            if (tt2!==c2)   begin $display("FAIL[K2 %0s] total %0d exp %0d",name,tt2,c2);$fatal(1,"t");end
            if (mp2!==p2pas)begin $display("FAIL[K2 %0s] pass %0d exp %0d",name,mp2,p2pas);$fatal(1,"p");end
            if (ac2!==p2acc)begin $display("FAIL[K2 %0s] acc %0d exp %0d",name,ac2,p2acc);$fatal(1,"a");end
            if (rj2!==p2rej)begin $display("FAIL[K2 %0s] rej %0d exp %0d",name,rj2,p2rej);$fatal(1,"r");end
            tests=tests+4;
            $display("PASS[K2 %0s] passes=%0d total=%0d acc=%0d rej=%0d (eff x100=%0d)",
                name, mp2, tt2, ac2, rj2, (mp2==0)?0:(tt2*100)/mp2);
        end
    endtask

    // ---- K=3 ----
    localparam integer DKW3 = 2;   // $clog2(4)
    reg                 rst3, st3;
    reg                 pv3;
    reg [3*TOKW-1:0]    dv3;
    reg [4*TOKW-1:0]    tv3;
    reg [DKW3-1:0]      nd3;
    wire                cv3;
    wire [TOKW-1:0]     ct3;
    wire [31:0]         tt3, mp3, ac3, rj3;
    spec_decode_seq #(.TOKW(TOKW), .DRAFT_K(3)) dut3 (
        .clk(clk), .rst(rst3), .start(st3),
        .pass_valid(pv3), .verified_tok(16'd0), .draft_tok(16'd0), .draft_present(1'b0),
        .commit_valid(cv3), .commit_tok(ct3), .accepted(),
        .total_tokens(tt3), .main_passes(mp3), .accepts(ac3), .rejects(rj3),
        .draft_vec(dv3), .truth_vec(tv3), .n_draft(nd3)
    );
    reg  [TOKW-1:0] got3 [0:CAP-1];
    integer got3_n = 0; reg cap3 = 1'b0;
    always @(negedge clk) if (cap3 && cv3) begin
        if (^ct3 === 1'bx) begin $display("FAIL: X on K3 commit beat %0d", got3_n);$fatal(1,"X"); end
        got3[got3_n]=ct3; got3_n=got3_n+1;
    end
    integer c3, p3pas, p3acc, p3rej;
    task k3_arm; begin
        rst3=1; st3=0; pv3=0; dv3=0; tv3=0; nd3=0; cap3=0;
        @(negedge clk); @(negedge clk); rst3=0;
        got3_n=0; c3=0; p3pas=0; p3acc=0; p3rej=0; cap3=1;
        @(negedge clk); st3=1; @(negedge clk); st3=0;
    end endtask
    task k3_pass(input integer nd, input integer first_miss);
        integer j, p; reg [TOKW-1:0] m;
        begin
            dv3=0; tv3=0;
            for (j=0;j<=3;j=j+1) tv3[j*TOKW +: TOKW]=G[c3+j];
            for (j=0;j<3;j=j+1) begin
                m=G[c3+j];
                dv3[j*TOKW +: TOKW] = (j<first_miss)? m : (m ^ MISMASK);
            end
            nd3=nd[DKW3-1:0];
            p=(first_miss<nd)?first_miss:nd;
            p3pas=p3pas+1; p3acc=p3acc+p; p3rej=p3rej+(nd-p); c3=c3+p+1;
            @(negedge clk); pv3=1;
            @(negedge clk); pv3=0; dv3=0; tv3=0; nd3=0;
            for (j=0;j<5;j=j+1) @(negedge clk);  // K+2 drain cycles
        end
    endtask
    task k3_check(input [255:0] name);
        integer k; begin
            if (got3_n !== c3) begin $display("FAIL[K3 %0s] len got %0d exp %0d",name,got3_n,c3);$fatal(1,"len"); end
            for (k=0;k<c3;k=k+1) begin
                if (got3[k] !== G[k]) begin
                    $display("FAIL[K3 %0s] beat %0d got %0d exp greedy %0d",name,k,got3[k],G[k]);
                    $fatal(1,"spec!=greedy"); end
                tests=tests+1; end
            if (tt3!==c3)   begin $display("FAIL[K3 %0s] total %0d exp %0d",name,tt3,c3);$fatal(1,"t");end
            if (mp3!==p3pas)begin $display("FAIL[K3 %0s] pass %0d exp %0d",name,mp3,p3pas);$fatal(1,"p");end
            if (ac3!==p3acc)begin $display("FAIL[K3 %0s] acc %0d exp %0d",name,ac3,p3acc);$fatal(1,"a");end
            if (rj3!==p3rej)begin $display("FAIL[K3 %0s] rej %0d exp %0d",name,rj3,p3rej);$fatal(1,"r");end
            tests=tests+4;
            $display("PASS[K3 %0s] passes=%0d total=%0d acc=%0d rej=%0d (eff x100=%0d)",
                name, mp3, tt3, ac3, rj3, (mp3==0)?0:(tt3*100)/mp3);
        end
    endtask

    //========================================================================
    // PRNG helper: 0..999 uniform
    //========================================================================
    function integer rnd1000; input dummy; integer r; begin
        r=$random(seed); if (r<0) r=-r; rnd1000=r%1000; end
    endfunction

    integer i, alpha_pm, fm, nd_r, j;
    integer e1_m, e2_m, e3_m;     // measured eff*100 per K
    integer ideal100;             // theoretical eff*100 for the FLAT model
    integer sweep [0:6];

    //========================================================================
    // MAIN
    //========================================================================
    initial begin
        seed = 32'h5EC0DE11;
        for (gi=0; gi<CAP; gi=gi+1) G[gi] = ($random(seed) & 16'h7FFF) | 16'h1;

        //================= PART A : spec==greedy, K=1,2,3 =================
        $display("==== PART A : committed stream == greedy, for DRAFT_K = 1,2,3 ====");

        // ---------------- K = 1 ----------------
        // all-accept
        k1_arm;
        k1_pass(0,16'd100,16'd101);
        k1_pass(1,16'd101,16'd102); k1_pass(1,16'd102,16'd103); k1_pass(1,16'd103,16'd104);
        @(negedge clk); k1_check("all-accept");
        // all-reject
        k1_arm;
        k1_pass(0,16'd200,16'd201);
        k1_pass(1,16'd210,16'd211); k1_pass(1,16'd220,16'd221); k1_pass(1,16'd230,16'd231);
        @(negedge clk); k1_check("all-reject");
        // reject-at-each-position (K=1: positions = {accept, reject})
        k1_arm;
        k1_pass(0,16'd300,16'd301);
        k1_pass(1,16'd301,16'd302);  // accept (p=1)
        k1_pass(1,16'd999,16'd303);  // reject (p=0)
        @(negedge clk); k1_check("reject-each-pos");
        // mixed random sweep over alpha
        for (i=0;i<5;i=i+1) begin
            k1_arm;
            k1_pass(0,16'd1,G[0]);
            for (j=1;j<60;j=j+1) begin
                alpha_pm = (i*200);  // 0,200,400,600,800
                fm = (rnd1000(0) < alpha_pm) ? 1 : 0;
                // accept => verified == prior draft; reject => guaranteed mismatch
                if (fm==1) k1_pass(1, gp1, G[j]);
                else       k1_pass(1, gp1 ^ MISMASK, G[j]);
            end
            @(negedge clk); k1_check("mixed");
        end

        // ---------------- K = 2 ----------------
        k2_arm; for(i=0;i<5;i=i+1) k2_pass(2,2); @(negedge clk); k2_check("all-accept");
        k2_arm; for(i=0;i<5;i=i+1) k2_pass(2,0); @(negedge clk); k2_check("all-reject");
        // reject-at-each-position: first_miss walks 0,1,2
        k2_arm; k2_pass(2,0); k2_pass(2,1); k2_pass(2,2);
                k2_pass(1,1); k2_pass(1,0);  // short batches
                @(negedge clk); k2_check("reject-each-pos");
        // mixed random
        k2_arm;
        for (i=0;i<200;i=i+1) begin
            nd_r = (rnd1000(0)%2)+1;            // 1..2
            fm   = rnd1000(0)%3;               // 0..2
            k2_pass(nd_r, fm);
        end
        @(negedge clk); k2_check("mixed");

        // ---------------- K = 3 ----------------
        k3_arm; for(i=0;i<5;i=i+1) k3_pass(3,3); @(negedge clk); k3_check("all-accept");
        k3_arm; for(i=0;i<5;i=i+1) k3_pass(3,0); @(negedge clk); k3_check("all-reject");
        k3_arm; k3_pass(3,0); k3_pass(3,1); k3_pass(3,2); k3_pass(3,3);
                k3_pass(1,1); k3_pass(2,1);
                @(negedge clk); k3_check("reject-each-pos");
        k3_arm;
        for (i=0;i<200;i=i+1) begin
            nd_r = (rnd1000(0)%3)+1;            // 1..3
            fm   = rnd1000(0)%4;               // 0..3
            k3_pass(nd_r, fm);
        end
        @(negedge clk); k3_check("mixed");

        $display("ALL %0d TESTS PASSED", tests);

        //================= PART B : eff tok/pass vs alpha =================
        // FLAT per-position acceptance: each position matches w.p. alpha. The
        // longest-prefix stop => eff_K = 1 + sum_{j=1..K} alpha^j.
        sweep[0]=0; sweep[1]=300; sweep[2]=500; sweep[3]=600;
        sweep[4]=700; sweep[5]=800; sweep[6]=900;
        $display("");
        $display("==== PART B(i): FLAT per-position alpha -- eff tok/pass (n=1500 passes) ====");
        $display("  alpha | K=1 eff | K=2 eff | K=3 eff |  ideal K3 (1+a+a^2+a^3)");
        $display("  ------+---------+---------+---------+------------------------");
        for (i=0;i<7;i=i+1) begin
            alpha_pm = sweep[i];
            // K=1 : eff = 1+alpha
            k1_arm; k1_pass(0,16'd1,G[0]);
            for (j=1;j<1500;j=j+1) begin
                if (rnd1000(0)<alpha_pm) k1_pass(1, gp1, G[j]);
                else                     k1_pass(1, gp1 ^ MISMASK, G[j]);
            end
            e1_m = (tt1*100)/mp1;
            // K=2
            k2_arm;
            for (j=0;j<1500;j=j+1) begin
                fm=0;
                if (rnd1000(0)<alpha_pm) begin fm=1; if (rnd1000(0)<alpha_pm) fm=2; end
                k2_pass(2, fm);
            end
            e2_m = (tt2*100)/mp2;
            // K=3
            k3_arm;
            for (j=0;j<1500;j=j+1) begin
                fm=0;
                if (rnd1000(0)<alpha_pm) begin fm=1;
                    if (rnd1000(0)<alpha_pm) begin fm=2;
                        if (rnd1000(0)<alpha_pm) fm=3; end end
                k3_pass(3, fm);
            end
            e3_m = (tt3*100)/mp3;
            // ideal eff*100 = 100 + a + a^2 + a^3 (a = alpha_pm/1000), integer math
            ideal100 = 100 + (alpha_pm/10) + ((alpha_pm*alpha_pm)/10000)
                           + ((alpha_pm*alpha_pm/100*alpha_pm)/100000);
            $display("   0.%01d0 |  %0d.%02d  |  %0d.%02d  |  %0d.%02d  |  %0d.%02d",
                alpha_pm/100,
                e1_m/100,e1_m%100, e2_m/100,e2_m%100, e3_m/100,e3_m%100,
                ideal100/100, ideal100%100);
        end

        // CHAINED-DECAY: position j matches w.p. alpha^(j+1) (realistic 1-MTP
        // chain -- deeper autoregressive drafts decay). eff_K = 1 + sum prod.
        $display("");
        $display("==== PART B(ii): CHAINED-DECAY alpha_j=alpha^(j+1) -- eff tok/pass (n=1500) ====");
        $display("  alpha | K=1 eff | K=2 eff | K=3 eff   (the SHIPPED 1-MTP-layer reality)");
        $display("  ------+---------+---------+---------");
        for (i=0;i<7;i=i+1) begin
            alpha_pm = sweep[i];
            // K=1 : pos0 matches w.p. alpha  => eff=1+alpha (same as flat)
            k1_arm; k1_pass(0,16'd1,G[0]);
            for (j=1;j<1500;j=j+1) begin
                if (rnd1000(0)<alpha_pm) k1_pass(1, gp1, G[j]);
                else                     k1_pass(1, gp1 ^ MISMASK, G[j]);
            end
            e1_m = (tt1*100)/mp1;
            // K=2 : pos0 w.p. alpha, pos1 w.p. alpha^2
            k2_arm;
            for (j=0;j<1500;j=j+1) begin
                fm=0;
                if (rnd1000(0)<alpha_pm) begin fm=1;
                    if (rnd1000(0) < (alpha_pm*alpha_pm)/1000) fm=2; end
                k2_pass(2, fm);
            end
            e2_m = (tt2*100)/mp2;
            // K=3 : pos j w.p. alpha^(j+1)
            k3_arm;
            for (j=0;j<1500;j=j+1) begin
                fm=0;
                if (rnd1000(0)<alpha_pm) begin fm=1;
                    if (rnd1000(0) < (alpha_pm*alpha_pm)/1000) begin fm=2;
                        if (rnd1000(0) < (alpha_pm*alpha_pm/1000*alpha_pm)/1000) fm=3; end end
                k3_pass(3, fm);
            end
            e3_m = (tt3*100)/mp3;
            $display("   0.%01d0 |  %0d.%02d  |  %0d.%02d  |  %0d.%02d",
                alpha_pm/100,
                e1_m/100,e1_m%100, e2_m/100,e2_m%100, e3_m/100,e3_m%100);
        end
        $display("  (K>1 raises eff only while drafts hit; chained decay erodes the deep positions)");

        $finish;
    end

    // safety timeout
    initial begin #200000000; $display("FAIL: timeout"); $fatal(1,"timeout"); end
endmodule
