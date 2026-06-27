`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
//============================================================================
// swiglu_expert_fp8_tb.v  --  self-checking TB for the FP8-native SwiGLU expert.
//
//   BINDING GOLDEN: an INDEPENDENT fp64 model of the SAME fp8 math the DUT runs:
//     * weights are E4M3 with [128,128] bf16 BLOCK scales (per out-block,k-block),
//     * activations (x for gate/up, h for down) are DYNAMICALLY quantized to
//       E4M3 with the SAME per-vector pow2 a_shift the DUT computes
//       (a_shift = clamp(134 - max_exp)); xsh is exact (same x); hsh is predicted
//       from the golden's own bf16 h (chained bf16-rounded -> matches the DUT's h
//       except at rare power-of-two boundaries, covered by the tolerance),
//     * the three GEMMs are block-scaled fp8 reductions accumulated in fp64,
//     * the TAIL is bf16: silu(gate) (glm_act SILU semantics: sigmoid arg clamped
//       to +/-16, raw x in the multiply), then h = bf16(silu*up); y rounded bf16.
//   The fp8 per-element quantization error lives INSIDE the golden; the residual
//   DUT/golden gap is fp32-accum (vs fp64) over the gate+up and down reductions
//   plus the bf16 grid rounding of gate/up/h/y -- the tolerance is built from
//   those depths, NOT from the raw 2^-3 E4M3 ulp.
//
//   X-AWARE: any X in a captured output bit is a hard failure.
//   Emits "ALL <N> TESTS PASSED" and $fatal on any mismatch / X.
//
//   Covers DENSE (INTER>HIDDEN) and MoE (INTER<HIDDEN) modes via two DUT
//   instances differing only in INTER (HIDDEN identical, as in GLM-5.2).
//   As a SANITY scaffold a bf16 reference (swiglu_expert) max-deviation is also
//   printed, but the BINDING check is the fp8 golden above.
//============================================================================
module swiglu_expert_fp8_tb;

    localparam integer HIDDEN     = 256;   // both modes (as in GLM-5.2)
    localparam integer INTER_MOE  = 128;   // MoE expert  (INTER < HIDDEN)
    localparam integer INTER_DENSE= 256;   // dense front (INTER >= HIDDEN, 2 k-blk)
    localparam integer TN         = 4;
    localparam integer KMAX       = 256;
    localparam integer BLK        = 128;
    localparam integer NB         = (KMAX + BLK - 1) / BLK;   // 2 K-blocks
    localparam integer NBg        = (HIDDEN + BLK - 1) / BLK; // gate/up K-blocks
    localparam integer KW         = $clog2(KMAX+1);
    localparam integer GW         = $clog2(HIDDEN/TN + 1);

    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;

    reg clk = 1'b0, rst = 1'b1;
    always #5 clk = ~clk;

    integer pass_cnt = 0, fail_cnt = 0;
    real    max_ratio = 0.0;
    real    bf16_maxdev = 0.0;          // sanity: vs bf16 swiglu_expert
    integer INTER_CUR;                  // INTER of the test in flight

    //----------------------------------------------------------------------
    //  Shared scenario ROM (sized to the larger dims; per-test fill).
    //----------------------------------------------------------------------
    reg [15:0] X     [0:HIDDEN-1];                 // bf16 token
    reg [ 7:0] Wgate [0:HIDDEN-1][0:HIDDEN-1];     // E4M3 (rows<=INTER used)
    reg [ 7:0] Wup   [0:HIDDEN-1][0:HIDDEN-1];
    reg [ 7:0] Wdown [0:HIDDEN-1][0:HIDDEN-1];
    reg [15:0] SCgate[0:1][0:NB-1];                // bf16 block scale [out-blk][k-blk]
    reg [15:0] SCup  [0:1][0:NB-1];
    reg [15:0] SCdown[0:1][0:NB-1];

    reg [16*HIDDEN-1:0] x_vec_w;                   // packed token for the DUTs
    integer pk;
    always @* begin
        for (pk = 0; pk < HIDDEN; pk = pk + 1)
            x_vec_w[16*pk +: 16] = X[pk];
    end

    //----------------------------------------------------------------------
    //  Independent fp64 decoders / scale helpers (NOT the DUT's fp32 funcs).
    //----------------------------------------------------------------------
    function real e4m3_real(input [7:0] x);
        reg s; reg [3:0] e; reg [2:0] m; real v;
        begin
            s=x[7]; e=x[6:3]; m=x[2:0];
            if (e==4'hF && m==3'h7)      v=0.0;
            else if (e==4'h0)            v=m*(2.0**(-9));
            else                         v=(1.0+m/8.0)*(2.0**(real'($signed({1'b0,e}))-7.0));
            e4m3_real = s ? -v : v;
        end
    endfunction

    function real bf16_real(input [15:0] b);
        reg s; reg [7:0] e; reg [6:0] m; real v;
        begin
            s=b[15]; e=b[14:7]; m=b[6:0];
            if (e==8'hFF)      v=0.0;
            else if (e==8'h00) v=m*(2.0**(-133));
            else               v=(1.0+m/128.0)*(2.0**(real'($signed({1'b0,e}))-127.0));
            bf16_real = s ? -v : v;
        end
    endfunction

    // bf16*2^k as the exact fp32 word the DUT encodes (matches fp32_scale_pow2).
    function [31:0] scaled_fp32(input [15:0] b, input signed [9:0] k);
        reg s; reg [7:0] e; reg [22:0] m; reg signed [10:0] ne;
        begin
            s=b[15]; e=b[14:7]; m={b[6:0],16'b0};
            if (e==8'hFF)      scaled_fp32={s,e,m};
            else if (e==8'h00) scaled_fp32={s,31'b0};
            else begin
                ne=$signed({3'b0,e})+k;
                if (ne>=11'sd255)    scaled_fp32={s,8'hFF,23'b0};
                else if (ne<=11'sd0) scaled_fp32={s,31'b0};
                else                 scaled_fp32={s,ne[7:0],m};
            end
        end
    endfunction

    // EXACT RNE round of an fp64 real to bf16 (via the IEEE double bit pattern).
    function [15:0] real_to_bf16(input real v);
        reg [63:0] db; reg s; reg [10:0] de; reg [51:0] dm;
        reg signed [12:0] be; reg [7:0] mr; reg rnd, sticky;
        begin
            if (v == 0.0) real_to_bf16 = 16'h0000;
            else begin
                db=$realtobits(v); s=db[63]; de=db[62:52]; dm=db[51:0];
                be = $signed({2'b0,de}) - 13'sd1023 + 13'sd127;
                rnd = dm[44]; sticky = (dm[43:0] != 0);
                mr = {1'b0,dm[51:45]} + ((rnd & (sticky | dm[45])) ? 8'd1 : 8'd0);
                if (mr[7]) begin be = be + 13'sd1; mr = 8'd0; end
                if (be >= 13'sd255)     real_to_bf16 = {s,8'hFF,7'b0};
                else if (be <= 13'sd0)  real_to_bf16 = {s,15'b0};
                else                    real_to_bf16 = {s,be[7:0],mr[6:0]};
            end
        end
    endfunction

    // 2^s for small signed s (pure scaling; bounded).
    function real pow2r(input signed [9:0] s);
        real p; integer i;
        begin
            p = 1.0;
            if (s >= 0) for (i=0;i<s;i=i+1)  p = p*2.0;
            else        for (i=0;i<-s;i=i+1) p = p/2.0;
            pow2r = p;
        end
    endfunction

    // dynamic per-vector pow2 shift, IDENTICAL to the DUT's dyn_shift.
    function signed [7:0] dyn_shift_tb(input [7:0] emax);
        reg signed [9:0] sh;
        begin
            if (emax==8'd0) dyn_shift_tb = 8'sd0;
            else begin
                sh = 10'sd134 - $signed({2'b0,emax});
                if (sh > 10'sd127)  sh = 10'sd127;
                if (sh < -10'sd128) sh = -10'sd128;
                dyn_shift_tb = sh[7:0];
            end
        end
    endfunction

    //----------------------------------------------------------------------
    //  Golden result store (per-test).
    //----------------------------------------------------------------------
    reg [15:0] Hbf [0:HIDDEN-1];      // golden bf16 h (sized to max INTER)
    real GY     [0:HIDDEN-1];         // golden y (fp64)
    real GSUMABS[0:HIDDEN-1];         // sum of |terms| of the down reduction

    task compute_golden(input integer IT);
        integer i,o,k,b,k0,k1,outblk,NBd;
        reg [7:0] emax; reg signed [7:0] xs, hs;
        reg [31:0] sfp; reg [7:0] aq; real aqr;
        real gsum,usum,gateR,upR,gbf,ubf,gclamp,sigv,siluv,hR,ashf;
        real bs,ba,scr,dnsum,dnabs;
        reg [15:0] gate_bf, up_bf, silu_bf;
        begin
            // ---- xsh from the token (exact: golden sees the same X) ----
            emax=8'd0;
            for (k=0;k<HIDDEN;k=k+1) if (X[k][14:7] > emax) emax = X[k][14:7];
            xs = dyn_shift_tb(emax);
            ashf = pow2r({{2{xs[7]}}, xs});
            // ---- gate/up (block-scaled fp8, fp64 accum) + bf16 tail -> Hbf ----
            for (i=0;i<IT;i=i+1) begin
                gateR=0.0; upR=0.0;
                for (b=0;b<NBg;b=b+1) begin
                    k0=b*BLK; k1=(b+1)*BLK; if (k1>HIDDEN) k1=HIDDEN;
                    gsum=0.0; usum=0.0;
                    for (k=k0;k<k1;k=k+1) begin
                        sfp=scaled_fp32(X[k],{{2{xs[7]}},xs});
                        aq =fp32_to_fp8e4m3(sfp);
                        aqr=e4m3_real(aq);
                        gsum = gsum + aqr*e4m3_real(Wgate[i][k]);
                        usum = usum + aqr*e4m3_real(Wup[i][k]);
                    end
                    outblk = i/128;
                    gateR = gateR + gsum*bf16_real(SCgate[outblk][b]);
                    upR   = upR   + usum*bf16_real(SCup  [outblk][b]);
                end
                gateR = gateR/ashf;  upR = upR/ashf;
                gate_bf = real_to_bf16(gateR);
                up_bf   = real_to_bf16(upR);
                gbf = bf16_real(gate_bf);  ubf = bf16_real(up_bf);
                // glm_act SILU: sigmoid arg clamped to +/-16, raw gbf in multiply.
                gclamp = gbf; if (gclamp>16.0) gclamp=16.0; if (gclamp<-16.0) gclamp=-16.0;
                sigv  = 1.0/(1.0+$exp(-gclamp));
                siluv = gbf*sigv;
                silu_bf = real_to_bf16(siluv);
                hR = bf16_real(silu_bf)*ubf;
                Hbf[i] = real_to_bf16(hR);
            end
            // ---- hsh from the golden bf16 h (predicted; matches DUT h) ----
            emax=8'd0;
            for (i=0;i<IT;i=i+1) if (Hbf[i][14:7] > emax) emax = Hbf[i][14:7];
            hs = dyn_shift_tb(emax);
            ashf = pow2r({{2{hs[7]}}, hs});
            NBd = (IT + BLK - 1)/BLK;
            // ---- down (block-scaled fp8, fp64 accum) -> GY ----
            for (o=0;o<HIDDEN;o=o+1) begin
                dnsum=0.0; dnabs=0.0;
                for (b=0;b<NBd;b=b+1) begin
                    k0=b*BLK; k1=(b+1)*BLK; if (k1>IT) k1=IT;
                    bs=0.0; ba=0.0;
                    for (k=k0;k<k1;k=k+1) begin
                        sfp=scaled_fp32(Hbf[k],{{2{hs[7]}},hs});
                        aq =fp32_to_fp8e4m3(sfp);
                        aqr=e4m3_real(aq);
                        bs = bs + aqr*e4m3_real(Wdown[o][k]);
                        ba = ba + ((aqr*e4m3_real(Wdown[o][k])>=0.0) ?
                                    aqr*e4m3_real(Wdown[o][k]) : -(aqr*e4m3_real(Wdown[o][k])));
                    end
                    outblk = o/128;
                    scr = bf16_real(SCdown[outblk][b]);
                    dnsum = dnsum + bs*scr;
                    dnabs = dnabs + ba*((scr>=0.0)?scr:-scr);
                end
                GY[o]      = dnsum/ashf;
                GSUMABS[o] = dnabs/ashf;
            end
        end
    endtask

    task check_y(input [16*HIDDEN-1:0] yv, input [127:0] tag);
        integer o; reg [15:0] cb; real dutr,err,tol,g,sa;
        begin
            for (o=0;o<HIDDEN;o=o+1) begin
                cb = yv[16*o +: 16];
                if (^cb === 1'bx) begin
                    $display("FAIL [%0s] o=%0d : X in output = %b", tag, o, cb);
                    fail_cnt = fail_cnt + 1;
                end else begin
                    dutr=bf16_real(cb); g=GY[o]; sa=GSUMABS[o];
                    err = dutr-g; if (err<0.0) err=-err;
                    // principled fp8 tolerance:
                    //   down fp32-accum (depth*ulp scaled by sum|terms|)
                    // + h bf16-grid mismatch propagated through the down reduction
                    // + final bf16 output rounding + a tiny floor.
                    tol = sa*(real'(INTER_CUR)*(2.0**(-22)))
                        + sa*(2.0**(-6))
                        + ((g>=0.0)?g:-g)*(2.0**(-7))
                        + (2.0**(-14));
                    if (err > tol) begin
                        $display("FAIL [%0s] o=%0d : dut=%g golden=%g err=%g tol=%g",
                                 tag, o, dutr, g, err, tol);
                        fail_cnt = fail_cnt + 1;
                    end else begin
                        pass_cnt = pass_cnt + 1;
                        if (tol>0.0 && err/tol > max_ratio) max_ratio = err/tol;
                    end
                end
            end
        end
    endtask

    //----------------------------------------------------------------------
    //  Random scenario generators.
    //----------------------------------------------------------------------
    function [15:0] rnd_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m; reg s;
        begin
            s=$random&1; e=lo_e+({$random}%(hi_e-lo_e+1)); m=$random;
            rnd_bf16={s,e,m};
        end
    endfunction
    // positive bf16 (for scales).
    function [15:0] rnd_pos_bf16(input integer lo_e, input integer hi_e);
        reg [7:0] e; reg [6:0] m;
        begin
            e=lo_e+({$random}%(hi_e-lo_e+1)); m=$random;
            rnd_pos_bf16={1'b0,e,m};
        end
    endfunction
    // E4M3 with bounded exponent (keeps products O(1), dodges NaN).
    function [7:0] rnd_e4m3_b(input integer elo, input integer ehi);
        reg s; reg [3:0] e; reg [2:0] m;
        begin
            s=$random&1; e=elo+({$random}%(ehi-elo+1)); m=$random;
            if (e==4'hF && m==3'h7) m=3'h6;
            rnd_e4m3_b={s,e,m};
        end
    endfunction

    integer i2,k2,b2;
    task gen_scenario(input integer IT);
        begin
            for (k2=0;k2<HIDDEN;k2=k2+1) X[k2]=rnd_bf16(125,129);  // ~0.25..8
            for (i2=0;i2<IT;i2=i2+1)
                for (k2=0;k2<HIDDEN;k2=k2+1) begin
                    Wgate[i2][k2]=rnd_e4m3_b(4,9);
                    Wup  [i2][k2]=rnd_e4m3_b(4,9);
                end
            for (i2=0;i2<HIDDEN;i2=i2+1)
                for (k2=0;k2<IT;k2=k2+1)
                    Wdown[i2][k2]=rnd_e4m3_b(4,9);
            for (b2=0;b2<NB;b2=b2+1) begin
                SCgate[0][b2]=rnd_pos_bf16(122,124); SCgate[1][b2]=rnd_pos_bf16(122,124);
                SCup  [0][b2]=rnd_pos_bf16(122,124); SCup  [1][b2]=rnd_pos_bf16(122,124);
                SCdown[0][b2]=rnd_pos_bf16(122,124); SCdown[1][b2]=rnd_pos_bf16(122,124);
            end
        end
    endtask

    //======================================================================
    //  DUT: MoE (INTER_MOE) and DENSE (INTER_DENSE).  Identical port widths.
    //======================================================================
    // ---- MoE ----
    reg                     start_moe;
    wire                    busy_moe, done_moe;
    wire                    wreq_moe;
    wire [1:0]              wsel_moe;
    wire [GW-1:0]           wgrp_moe;
    wire [KW-1:0]           wk_moe;
    reg  [8*TN-1:0]         wcol_moe, wcolup_moe;
    reg  [16*TN*NB-1:0]     wscG_moe, wscU_moe;
    wire [16*HIDDEN-1:0]    y_moe;

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER_MOE), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK)) dut_moe (
        .clk(clk), .rst(rst), .start(start_moe), .busy(busy_moe), .done(done_moe),
        .x_vec(x_vec_w),
        .w_req(wreq_moe), .w_sel(wsel_moe), .w_grp(wgrp_moe), .w_k(wk_moe),
        .w_col(wcol_moe), .w_col_up(wcolup_moe),
        .w_scale_g(wscG_moe), .w_scale_u(wscU_moe),
        .y_out(y_moe)
    );

    // ---- DENSE ----
    reg                     start_den;
    wire                    busy_den, done_den;
    wire                    wreq_den;
    wire [1:0]              wsel_den;
    wire [GW-1:0]           wgrp_den;
    wire [KW-1:0]           wk_den;
    reg  [8*TN-1:0]         wcol_den, wcolup_den;
    reg  [16*TN*NB-1:0]     wscG_den, wscU_den;
    wire [16*HIDDEN-1:0]    y_den;

    swiglu_expert_fp8 #(.HIDDEN(HIDDEN), .INTER(INTER_DENSE), .TN(TN),
                        .KMAX(KMAX), .BLK(BLK)) dut_den (
        .clk(clk), .rst(rst), .start(start_den), .busy(busy_den), .done(done_den),
        .x_vec(x_vec_w),
        .w_req(wreq_den), .w_sel(wsel_den), .w_grp(wgrp_den), .w_k(wk_den),
        .w_col(wcol_den), .w_col_up(wcolup_den),
        .w_scale_g(wscG_den), .w_scale_u(wscU_den),
        .y_out(y_den)
    );

    // ---- combinational weight responders (shared ROM, per-DUT request) ----
    //  Driven from always @* (sensitive to the whole ROM array) so a freshly
    //  filled ROM is reflected even when the request indices have not changed
    //  from their reset value -- a plain function-in-continuous-assign read is
    //  NOT array-sensitive and would latch the stale (pre-fill X) first read.
    integer rt, rb;
    always @* begin
        for (rt=0; rt<TN; rt=rt+1) begin
            wcol_moe  [8*rt +: 8] = (wsel_moe==SEL_DOWN) ? Wdown[wgrp_moe*TN+rt][wk_moe]
                                                         : Wgate[wgrp_moe*TN+rt][wk_moe];
            wcolup_moe[8*rt +: 8] = Wup[wgrp_moe*TN+rt][wk_moe];
            for (rb=0; rb<NB; rb=rb+1) begin
                wscG_moe[16*(rb*TN+rt) +: 16] = (wsel_moe==SEL_DOWN)
                        ? SCdown[(wgrp_moe*TN+rt)/128][rb] : SCgate[(wgrp_moe*TN+rt)/128][rb];
                wscU_moe[16*(rb*TN+rt) +: 16] = SCup[(wgrp_moe*TN+rt)/128][rb];
            end
        end
    end
    integer dt, db;
    always @* begin
        for (dt=0; dt<TN; dt=dt+1) begin
            wcol_den  [8*dt +: 8] = (wsel_den==SEL_DOWN) ? Wdown[wgrp_den*TN+dt][wk_den]
                                                         : Wgate[wgrp_den*TN+dt][wk_den];
            wcolup_den[8*dt +: 8] = Wup[wgrp_den*TN+dt][wk_den];
            for (db=0; db<NB; db=db+1) begin
                wscG_den[16*(db*TN+dt) +: 16] = (wsel_den==SEL_DOWN)
                        ? SCdown[(wgrp_den*TN+dt)/128][db] : SCgate[(wgrp_den*TN+dt)/128][db];
                wscU_den[16*(db*TN+dt) +: 16] = SCup[(wgrp_den*TN+dt)/128][db];
            end
        end
    end

    //----------------------------------------------------------------------
    //  Drivers (one per DUT; identical flow, different instance signals).
    //----------------------------------------------------------------------
    integer wd;
    task run_moe(input [127:0] tag);
        begin
            INTER_CUR = INTER_MOE;
            compute_golden(INTER_MOE);
            @(negedge clk); start_moe = 1'b1;
            @(negedge clk); start_moe = 1'b0;
            wd = 0;
            while (done_moe !== 1'b1) begin
                @(negedge clk); wd = wd + 1;
                if (wd > 400000) begin $display("FATAL [%0s]: done timeout", tag);
                                       $fatal(1,"timeout"); end
            end
            check_y(y_moe, tag);
            @(negedge clk);
        end
    endtask
    task run_den(input [127:0] tag);
        begin
            INTER_CUR = INTER_DENSE;
            compute_golden(INTER_DENSE);
            @(negedge clk); start_den = 1'b1;
            @(negedge clk); start_den = 1'b0;
            wd = 0;
            while (done_den !== 1'b1) begin
                @(negedge clk); wd = wd + 1;
                if (wd > 400000) begin $display("FATAL [%0s]: done timeout", tag);
                                       $fatal(1,"timeout"); end
            end
            check_y(y_den, tag);
            @(negedge clk);
        end
    endtask

    integer t;
    initial begin
        start_moe=1'b0; start_den=1'b0;
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        // ---- MoE mode (INTER<HIDDEN): two weight/x sets ----
        for (t=0; t<2; t=t+1) begin
            gen_scenario(INTER_MOE);
            run_moe("MOE");
        end
        // ---- DENSE mode (INTER>=HIDDEN, 2 K-blocks both passes): two sets ----
        for (t=0; t<2; t=t+1) begin
            gen_scenario(INTER_DENSE);
            run_den("DENSE");
        end

        if (fail_cnt != 0) begin
            $display("FAILED: %0d mismatch(es), %0d passed.", fail_cnt, pass_cnt);
            $fatal(1, "swiglu_expert_fp8 verification FAILED");
        end else begin
            $display("ALL %0d TESTS PASSED  (worst err/tol = %.4f)", pass_cnt, max_ratio);
        end
        $finish;
    end
endmodule
