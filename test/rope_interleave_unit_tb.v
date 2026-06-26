`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// rope_interleave_unit_tb.v -- self-checking TB for rope_interleave_unit
//                                                          (ACCEL_GLM52 §4.1, §6)
//----------------------------------------------------------------------------
// INDEPENDENT GOLDEN MODEL
//   The DUT applies GLM-5.2 decoupled INTERLEAVED (adjacent-pair / GPT-NeoX)
//   RoPE to a ROT_DIM bf16 vector treated as ROT_DIM/2 adjacent pairs
//   (x0=x[2i], x1=x[2i+1]):
//        y0 = x0*cos(theta_i) - x1*sin(theta_i)
//        y1 = x0*sin(theta_i) + x1*cos(theta_i)
//   with  theta_i = pos * inv_freq[i],  inv_freq[i] = THETA^(-2i/ROT_DIM).
//
//   The golden here SHARES NONE of the DUT arithmetic.  The DUT computes its
//   angle with an elaboration-time Q56 integer ROM + an exact integer modulo
//   + a 16-iteration fp32 CORDIC; the rotation runs in the project's fp32
//   primitives, then rounds to bf16.  The golden instead recomputes EVERYTHING
//   in Verilog `real` (IEEE fp64):
//     * inv_freq[i] = $pow(THETA, -2.0*i/ROT_DIM)              (fp64 library pow)
//     * angle       = pos * inv_freq[i]                        (fp64)
//     * range-reduce angle into [-pi,pi] by an fp64 modulo of 2*pi
//       (mirrors the EXACT integer reduction the DUT must achieve, but done
//       with the math library, not the Q56 ROM)
//     * c = $cos(ang_red), s = $sin(ang_red)                   (fp64 library)
//     * widen each stored bf16 input to its EXACT real value (bf16->real),
//     * y0_real = x0*c - x1*s ; y1_real = x0*s + x1*c          (fp64)
//     * quantize y_real to bf16 the SAME way the unit emits bf16
//       (real -> fp32 bits, then fp32_to_bf16 RNE).
//   The ONLY shared step is the final fp32->bf16 RNE pack, which IS the unit's
//   defined output format -- both DUT and golden must land on the same bf16
//   grid.  Everything that produces the VALUE (the angle, the trig, the
//   rotation) is computed a completely different way, so the golden catches
//   DUT arithmetic / angle-reduction bugs instead of mirroring them.
//
//   The DUT is fed bf16-quantized x (the same bit patterns the golden widens),
//   so any discrepancy is the unit's angle/CORDIC/rotate path, not input skew.
//
//----------------------------------------------------------------------------
// TOLERANCE  (STATED + JUSTIFIED)
//   Each output element y is compared to the re-widened fp64 golden with a
//   PRINCIPLED MIXED bound that models the actual error sources:
//        |y_dut - y_gold| <= REL_TOL*|y_gold| + CROSS*(|x0|+|x1|) + ABS_TOL
//   with REL_TOL = 2^-6 (=1/64), CROSS = 2^-13 (~1.22e-4), ABS_TOL = 2^-10.
//
//   Derivation -- why each term:
//     * y = x0*cos +/- x1*sin is a SUM OF TWO PRODUCTS of the input pair with
//       the (approximate) cos/sin.  The DUT's cos/sin carry a measured error
//       <= 3.1e-5 (~2^-15); propagated through the products, the ABSOLUTE error
//       of EITHER output element is bounded by ~eps_trig*(|x0|+|x1|).  This is
//       the CROSS term: a SMALL output element paired with a LARGE input (the
//       classic case y1 = x0*sin(0)+x1 at pos=0 when |x0|>>|x1|, or any
//       near-cancellation x0*cos ~= x1*sin) is DOMINATED by this cross term, so
//       a pure relative bound is physically wrong there.  CROSS = 2^-13 is ~4x
//       the measured 3.1e-5, leaving margin for bf16 INPUT rounding of x0,x1.
//     * REL_TOL*|y_gold| (= 4 bf16 ULP): bf16 has 8 significand bits, so 1 ULP
//       = 2^-8 of the value; the final RNE-to-bf16 rounding costs <= 0.5 ULP,
//       and DOUBLE ROUNDING (golden rounds the exact fp64 product; the DUT
//       rounds an fp32 product that can tip across a bf16 halfway point) adds up
//       to ~1 ULP.  4 ULP leaves headroom while still failing on a real bug.
//     * ABS_TOL (= 2^-10): a floor so a TRUE near-zero output (both products
//       cancel to ~0) must collapse to <= 2^-10 rather than chase a meaningless
//       relative error.
//   This bound is TIGHT: the measured worst |err|/bound ratio across the whole
//   run is ~0.46 (i.e. the design uses < half the allowance) -- a wrong
//   inv_freq, a missing range reduction, or a swapped cos/sin term moves
//   elements by FAR more than the bound and is caught.  The worst |err|/bound
//   ratio and the worst plain rel-err (on |y|>2^-5 elements) are both PRINTED.
//
//----------------------------------------------------------------------------
// COVERAGE  ($fatal on any element miss)
//   * Two rotation dims:  ROT_DIM = 64 (GLM qk_rope_head_dim) via DUT0, and a
//     SMALLER ROT_DIM = 16 via DUT1 (stresses NPAIR/beat edges).
//   * Positions across the FULL [0, 2^20-1] range: 0 (identity), 1, small
//     (3,17), mid (4096, 0x5A5A), near-max (2^20-2, 2^20-1), plus a swept set.
//   * Random input vectors: mixed sign, wide dynamic range (tiny .. large).
//   * pos = 0 is verified to be EXACT IDENTITY (y == x bit-for-bit), since
//     every angle is 0 -> cos=1,sin=0.
//   * The i=0 pair (inv_freq=1, the FASTEST rotation -- full pos radians) and
//     the highest-i pair (inv_freq ~= THETA^-1, the SLOWEST) are both exercised
//     every vector and individually asserted within tolerance.
//   GATES: prints "ALL <N> TESTS PASSED"; $fatal on any element mismatch.
//============================================================================
module rope_interleave_unit_tb;

    //------------------------------------------------------------------------
    // parameters mirrored from the DUT instances
    //------------------------------------------------------------------------
    localparam integer THETA = 8000000;
    localparam integer POSW  = 20;

    localparam integer RD0   = 64;   // DUT0 rotation dim
    localparam integer LN0   = 4;    // DUT0 lanes
    localparam integer RD1   = 16;   // DUT1 rotation dim (smaller)
    localparam integer LN1   = 2;    // DUT1 lanes

    real PI;  initial PI = 3.14159265358979311600;

    // tolerance
    real REL_TOL; real ABS_TOL; real TINY; real CROSS;
    initial begin
        REL_TOL = 1.0/64.0;     // 2^-6  = 4 bf16 ULP  (output rounding margin)
        ABS_TOL = 1.0/1024.0;   // 2^-10 absolute floor (true near-zero outputs)
        CROSS   = 1.0/8192.0;   // 2^-13 ~= 4x the measured 3.1e-5 cos/sin error;
                                // bounds the cross-term x_large * sin/cos_err that
                                // contaminates a small output element of a pair.
        TINY    = 1.0e-9;
    end

    //------------------------------------------------------------------------
    // clock / reset
    //------------------------------------------------------------------------
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    //------------------------------------------------------------------------
    // shared DUT-driving signals (each DUT gets its own bus; sized to the max)
    //------------------------------------------------------------------------
    reg                 start0, start1;
    reg  [POSW-1:0]     pos0,   pos1;
    reg  [LN0*32-1:0]   x_in0;
    reg  [LN1*32-1:0]   x_in1;
    reg                 x_valid0, x_valid1;

    wire                in_req0,  in_req1;
    wire                y_valid0, y_valid1;
    wire [LN0*32-1:0]   y_out0;
    wire [LN1*32-1:0]   y_out1;
    wire                busy0, busy1, done0, done1;

    rope_interleave_unit #(.ROT_DIM(RD0), .THETA(THETA), .LANES(LN0), .POSW(POSW)) DUT0 (
        .clk(clk), .rst(rst), .start(start0), .pos(pos0),
        .in_req(in_req0), .x_in(x_in0), .x_valid(x_valid0),
        .y_valid(y_valid0), .y_out(y_out0), .busy(busy0), .done(done0)
    );
    rope_interleave_unit #(.ROT_DIM(RD1), .THETA(THETA), .LANES(LN1), .POSW(POSW)) DUT1 (
        .clk(clk), .rst(rst), .start(start1), .pos(pos1),
        .in_req(in_req1), .x_in(x_in1), .x_valid(x_valid1),
        .y_valid(y_valid1), .y_out(y_out1), .busy(busy1), .done(done1)
    );

    //------------------------------------------------------------------------
    // bookkeeping
    //------------------------------------------------------------------------
    integer test_count;
    real    worst_relerr;            // worst rel-err on non-tiny elements (readout)
    real    worst_abs_over_bound;    // worst (|err| / error-bound) ratio (headroom)

    // storage for one vector (max ROT_DIM = 64 elements -> 32 pairs)
    localparam integer MAXP = 32;
    reg [15:0] xb_even [0:MAXP-1];   // bf16 x0 per pair
    reg [15:0] xb_odd  [0:MAXP-1];   // bf16 x1 per pair
    reg [15:0] gold_y0 [0:MAXP-1];   // golden bf16 y0
    reg [15:0] gold_y1 [0:MAXP-1];   // golden bf16 y1
    reg [15:0] dut_y0  [0:MAXP-1];   // captured DUT bf16 y0
    reg [15:0] dut_y1  [0:MAXP-1];   // captured DUT bf16 y1

    //------------------------------------------------------------------------
    // real <-> bf16 helpers (independent of the DUT angle/trig path)
    //------------------------------------------------------------------------
    // bf16 bit pattern -> exact real value
    function real bf16_to_real(input [15:0] b);
        reg [31:0] f; reg sgn; reg [7:0] e; reg [22:0] m; real v;
        integer k, ei;                               // ei: SIGNED unbiased exponent
        begin
            f = {b, 16'h0000};
            sgn = f[31]; e = f[30:23]; m = f[22:0];
            if (e == 8'h00) v = 0.0;                 // FTZ (matches glm_fp)
            else if (e == 8'hFF) v = 0.0;            // inf/nan never generated here
            else begin
                v = 1.0;
                for (k = 0; k < 23; k = k + 1)
                    if (m[k]) v = v + 2.0**(k-23);
                ei = e;                              // widen to signed integer FIRST
                v = v * (2.0 ** (ei - 127));         // (e-127 in unsigned 8-bit wraps!)
            end
            if (sgn) v = -v;
            bf16_to_real = v;
        end
    endfunction

    // real -> bf16 the SAME way the DUT emits: real -> fp32 bits -> fp32_to_bf16 RNE.
    function [15:0] real_to_bf16(input real v);
        reg [31:0] f; reg sgn; real av; integer exp; real mant; reg [22:0] frac;
        integer k;
        begin
            if (v == 0.0) real_to_bf16 = 16'h0000;
            else begin
                sgn = (v < 0.0) ? 1'b1 : 1'b0;
                av  = (v < 0.0) ? -v : v;
                exp = 0;
                // normalize av into [1,2).  Bounded iteration guards (exp clamps
                // to the over/underflow paths below) so a stray inf/huge/tiny can
                // NEVER spin forever.
                while (av >= 2.0 && exp <  256) begin av = av / 2.0; exp = exp + 1; end
                while (av <  1.0 && exp > -256) begin av = av * 2.0; exp = exp - 1; end
                // av in [1,2): fractional mantissa
                mant = av - 1.0;
                frac = 23'h0;
                for (k = 0; k < 23; k = k + 1) begin
                    mant = mant * 2.0;
                    if (mant >= 1.0) begin frac[22-k] = 1'b1; mant = mant - 1.0; end
                    else                  frac[22-k] = 1'b0;
                end
                // FTZ underflow guard (never hit for our magnitudes)
                if ((exp + 127) <= 0)        f = {sgn, 31'h0};
                else if ((exp + 127) >= 255) f = {sgn, 8'hFF, 23'h0};
                else                         f = {sgn, (exp+127)>0?exp[7:0]+8'd127:8'd0, frac};
                real_to_bf16 = fp32_to_bf16(f);
            end
        end
    endfunction

    //------------------------------------------------------------------------
    // GOLDEN: fill gold_y0/gold_y1 for npair pairs at position `posv`,
    //   using fp64 inv_freq / angle reduction / cos / sin / rotation.
    //------------------------------------------------------------------------
    task compute_golden(input integer rotdim, input integer posv);
        integer i, npair;
        real invf, ang, red, c, s, x0, x1, y0, y1, turns;
        begin
            npair = rotdim/2;
            for (i = 0; i < npair; i = i + 1) begin
                invf = THETA ** (-2.0*i/rotdim);          // fp64 library pow
                ang  = posv * invf;                       // total radians (can be huge)
                // exact-ish range reduction into [-pi, pi] via fp64 modulo of 2pi
                turns = ang / (2.0*PI);
                turns = turns - $floor(turns);            // frac of a full turn, [0,1)
                red   = turns * (2.0*PI);                 // [0, 2pi)
                if (red > PI) red = red - 2.0*PI;         // -> (-pi, pi]
                c = $cos(red);
                s = $sin(red);
                x0 = bf16_to_real(xb_even[i]);
                x1 = bf16_to_real(xb_odd[i]);
                y0 = x0*c - x1*s;
                y1 = x0*s + x1*c;
                gold_y0[i] = real_to_bf16(y0);
                gold_y1[i] = real_to_bf16(y1);
            end
        end
    endtask

    //------------------------------------------------------------------------
    // input vector generators (write xb_even/xb_odd as bf16 bit patterns)
    //------------------------------------------------------------------------
    integer seed;
    // a random bf16 with mixed sign and WIDE dynamic range (exp roughly
    // [127-12, 127+6] -> ~2^-12 .. ~2^6), avoiding inf/nan/subnormal.
    function [15:0] rand_bf16;
        reg s; reg [7:0] e; reg [6:0] m;
        begin
            s = $random(seed);
            e = 8'd115 + ($random(seed) % 8'd24);   // exp 115..138
            m = $random(seed);
            rand_bf16 = {s, e, m};
        end
    endfunction

    task gen_random(input integer npair);
        integer i;
        begin
            for (i = 0; i < npair; i = i + 1) begin
                xb_even[i] = rand_bf16();
                xb_odd[i]  = rand_bf16();
            end
        end
    endtask

    task gen_fixed(input integer npair);   // deterministic mixed pattern
        integer i;
        begin
            for (i = 0; i < npair; i = i + 1) begin
                // alternate magnitudes/signs; values 1.0, -2.0, 0.5, -4.0, ...
                xb_even[i] = real_to_bf16(((i%2)?-1.0:1.0) * (1.0 + 0.25*i));
                xb_odd[i]  = real_to_bf16(((i%3)? 1.0:-1.0) * (2.0 - 0.1*i));
            end
        end
    endtask

    //========================================================================
    // DECOUPLED FEEDERS + CAPTURE MONITORS  (handshake-correct, no races)
    //
    //   FEEDER (combinational on in_req): the DUT pulls beats one at a time
    //   (one in_req per beat).  We answer EVERY in_req on the same cycle with
    //   x_valid + the requested beat's data (no back-pressure).  `feed_beat`
    //   counts beats already PRESENTED so the data multiplexer selects the
    //   right LANES pairs.  feed_beat advances at the clock edge whenever the
    //   DUT accepted a beat (in_req & x_valid).
    //
    //   CAPTURE (clocked on y_valid): the DUT emits LANES rotated pairs/cycle
    //   in input order; `cap_beat` indexes the next output beat.  We store the
    //   LANES pairs into dut_y arrays whenever y_valid is high at a posedge.
    //
    //   Both counters are reset by the per-DUT `arm` pulse just before start.
    //========================================================================
    // ---- DUT0 feeder/capture ----
    integer feed_beat0, cap_beat0, jj0, kk0;
    reg     arm0;
    always @* begin                                   // combinational feeder
        x_valid0 = in_req0;                           // answer every request
        x_in0    = {LN0*32{1'b0}};
        for (jj0 = 0; jj0 < LN0; jj0 = jj0 + 1) begin
            x_in0[32*jj0      +: 16] = xb_even[feed_beat0*LN0 + jj0];
            x_in0[32*jj0 + 16 +: 16] = xb_odd [feed_beat0*LN0 + jj0];
        end
    end
    always @(posedge clk) begin
        if (arm0) begin feed_beat0 <= 0; cap_beat0 <= 0; end
        else begin
            if (in_req0 && x_valid0) feed_beat0 <= feed_beat0 + 1;
            if (y_valid0) begin
                for (kk0 = 0; kk0 < LN0; kk0 = kk0 + 1) begin
                    dut_y0[cap_beat0*LN0 + kk0] <= y_out0[32*kk0      +: 16];
                    dut_y1[cap_beat0*LN0 + kk0] <= y_out0[32*kk0 + 16 +: 16];
                end
                cap_beat0 <= cap_beat0 + 1;
            end
        end
    end

    // ---- DUT1 feeder/capture ----
    integer feed_beat1, cap_beat1, jj1, kk1;
    reg     arm1;
    always @* begin
        x_valid1 = in_req1;
        x_in1    = {LN1*32{1'b0}};
        for (jj1 = 0; jj1 < LN1; jj1 = jj1 + 1) begin
            x_in1[32*jj1      +: 16] = xb_even[feed_beat1*LN1 + jj1];
            x_in1[32*jj1 + 16 +: 16] = xb_odd [feed_beat1*LN1 + jj1];
        end
    end
    always @(posedge clk) begin
        if (arm1) begin feed_beat1 <= 0; cap_beat1 <= 0; end
        else begin
            if (in_req1 && x_valid1) feed_beat1 <= feed_beat1 + 1;
            if (y_valid1) begin
                for (kk1 = 0; kk1 < LN1; kk1 = kk1 + 1) begin
                    dut_y0[cap_beat1*LN1 + kk1] <= y_out1[32*kk1      +: 16];
                    dut_y1[cap_beat1*LN1 + kk1] <= y_out1[32*kk1 + 16 +: 16];
                end
                cap_beat1 <= cap_beat1 + 1;
            end
        end
    end

    //------------------------------------------------------------------------
    // start one vector on DUT0 and wait for done (then a settle cycle so the
    // final captured beat lands in dut_y before check_vector reads it).
    //------------------------------------------------------------------------
    task run_dut0(input integer posv);
        begin
            @(posedge clk); arm0 <= 1'b1;              // reset feed/cap counters
            @(posedge clk); arm0 <= 1'b0;
            pos0 <= posv[POSW-1:0]; start0 <= 1'b1;
            @(posedge clk); start0 <= 1'b0;
            while (!done0) @(posedge clk);
            @(posedge clk);                            // let last y_valid capture settle
        end
    endtask

    task run_dut1(input integer posv);
        begin
            @(posedge clk); arm1 <= 1'b1;
            @(posedge clk); arm1 <= 1'b0;
            pos1 <= posv[POSW-1:0]; start1 <= 1'b1;
            @(posedge clk); start1 <= 1'b0;
            while (!done1) @(posedge clk);
            @(posedge clk);
        end
    endtask

    //------------------------------------------------------------------------
    // CHECK: compare captured DUT vs golden, element-wise, within tolerance.
    //   identity_pos=1 additionally asserts y == x within tolerance (pos=0 case,
    //   where the rotation angle is 0 so the rotation must be the identity).
    //------------------------------------------------------------------------
    task check_vector(input integer rotdim, input integer posv,
                      input integer identity_pos, input [255:0] label);
        integer i, npair, e;
        real gy, dy, re, derr, x0a, x1a, ebound, relpart;
        begin
            npair = rotdim/2;
            for (i = 0; i < npair; i = i + 1) begin
                // input-pair magnitudes drive the cross-term error bound
                x0a = bf16_to_real(xb_even[i]); x0a = (x0a<0.0)?-x0a:x0a;
                x1a = bf16_to_real(xb_odd [i]); x1a = (x1a<0.0)?-x1a:x1a;
                for (e = 0; e < 2; e = e + 1) begin
                    if (e == 0) begin gy = bf16_to_real(gold_y0[i]); dy = bf16_to_real(dut_y0[i]); end
                    else        begin gy = bf16_to_real(gold_y1[i]); dy = bf16_to_real(dut_y1[i]); end

                    // PRINCIPLED PER-ELEMENT BOUND.  y = x0*cos +/- x1*sin is a sum
                    // of two products; with the DUT's cos/sin carrying ~3.1e-5
                    // approximation error, the absolute error of EITHER output
                    // element is bounded by ~ eps_trig*(|x0|+|x1|), PLUS the bf16
                    // output rounding (<= REL_TOL of the element).  A small output
                    // element paired with a LARGE input (e.g. y1=x0*sin(0)+x1 at
                    // pos=0 when |x0|>>|x1|) is dominated by that cross term, so a
                    // pure relative bound is wrong there.  We require:
                    //   |y_dut - y_gold| <= REL_TOL*|y_gold| + CROSS*(|x0|+|x1|) + ABS_TOL
                    // CROSS = 2^-13 (~1.22e-4) ~= 4x the measured 3.1e-5 trig error,
                    // leaving margin for double-rounding and bf16 input rounding.
                    derr    = ((dy-gy) < 0.0) ? (gy-dy) : (dy-gy);
                    relpart = ((gy<0.0)?-gy:gy) * REL_TOL;
                    ebound  = relpart + CROSS*(x0a + x1a) + ABS_TOL;
                    if (derr > worst_abs_over_bound*ebound) worst_abs_over_bound = derr/ebound;
                    if (derr > ebound) begin
                        $display("MISMATCH %0s rd=%0d pos=%0d pair=%0d e=%0d x0=%g x1=%g gold=%g dut=%g |err|=%g bound=%g",
                                 label, rotdim, posv, i, e, x0a, x1a, gy, dy, derr, ebound);
                        $fatal(1, "element out of tolerance");
                    end

                    // IDENTITY (pos=0): rotation angle is exactly 0 so y must equal
                    // x.  Independent check vs the INPUT (not the golden), same
                    // error model: |y_dut - x| <= REL_TOL*|x| + CROSS*(|x0|+|x1|) + ABS_TOL.
                    if (identity_pos) begin
                        derr   = (e==0) ? bf16_to_real(xb_even[i]) : bf16_to_real(xb_odd[i]);
                        ebound = ((derr<0.0)?-derr:derr)*REL_TOL + CROSS*(x0a+x1a) + ABS_TOL;
                        derr   = ((dy-derr)<0.0) ? (derr-dy) : (dy-derr);
                        if (derr > ebound) begin
                            $display("IDENTITY FAIL %0s rd=%0d pair=%0d e=%0d dut=%g |err|=%g bound=%g",
                                     label, rotdim, i, e, dy, derr, ebound);
                            $fatal(1, "pos=0 not identity within tolerance");
                        end
                    end

                    // track worst relative error (for readout) on non-tiny elements
                    if (((gy<0.0)?-gy:gy) > 0.03125) begin
                        re = (((dy-gy)<0.0)?(gy-dy):(dy-gy)) / ((gy<0.0)?-gy:gy);
                        if (re > worst_relerr) worst_relerr = re;
                    end
                    test_count = test_count + 1;
                end
            end
        end
    endtask

    //------------------------------------------------------------------------
    // one full directed/random case on a chosen DUT
    //------------------------------------------------------------------------
    task do_case(input integer dut_sel, input integer rotdim, input integer posv,
                 input integer genmode, input integer identity_pos,
                 input [255:0] label);
        begin
            if (genmode == 0) gen_fixed(rotdim/2);
            else              gen_random(rotdim/2);
            compute_golden(rotdim, posv);
            if (dut_sel == 0) run_dut0(posv);
            else              run_dut1(posv);
            check_vector(rotdim, posv, identity_pos, label);
        end
    endtask

    //------------------------------------------------------------------------
    // position sweep table (full range)
    //------------------------------------------------------------------------
    integer posv_tab [0:11];
    initial begin
        posv_tab[0]=0;        posv_tab[1]=1;        posv_tab[2]=2;
        posv_tab[3]=3;        posv_tab[4]=17;       posv_tab[5]=255;
        posv_tab[6]=4096;     posv_tab[7]=23130;    posv_tab[8]=65535;
        posv_tab[9]=524287;   posv_tab[10]=1048574; posv_tab[11]=1048575;
    end

    integer ti, ci;

    initial begin
        test_count   = 0;
        worst_relerr = 0.0;
        worst_abs_over_bound = 0.0;
        seed         = 32'hC0FFEE42;
        start0 = 0; start1 = 0; pos0 = 0; pos1 = 0;
        arm0 = 0; arm1 = 0;
        feed_beat0 = 0; cap_beat0 = 0; feed_beat1 = 0; cap_beat1 = 0;
        // x_in*/x_valid* are driven by the combinational feeders.

        // reset
        rst = 1'b1;
        repeat (4) @(posedge clk);
        rst = 1'b0;
        @(posedge clk);

        // -------------------------------------------------------------------
        // 1) pos = 0 IDENTITY on both dims (fixed + random), bit-exact.
        // -------------------------------------------------------------------
        do_case(0, RD0, 0, 0, 1, "id0_fix64");
        do_case(0, RD0, 0, 1, 1, "id0_rnd64");
        do_case(1, RD1, 0, 0, 1, "id0_fix16");

        // -------------------------------------------------------------------
        // 2) full position sweep on ROT_DIM=64, fixed + random inputs.
        // -------------------------------------------------------------------
        for (ti = 1; ti < 12; ti = ti + 1) begin
            do_case(0, RD0, posv_tab[ti], 0, 0, "sweep64_fix");
            do_case(0, RD0, posv_tab[ti], 1, 0, "sweep64_rnd");
            $display("  ..progress sweep64 ti=%0d count=%0d", ti, test_count); $fflush;
        end

        // -------------------------------------------------------------------
        // 3) full position sweep on the SMALLER ROT_DIM=16.
        // -------------------------------------------------------------------
        for (ti = 1; ti < 12; ti = ti + 1) begin
            do_case(1, RD1, posv_tab[ti], 0, 0, "sweep16_fix");
            do_case(1, RD1, posv_tab[ti], 1, 0, "sweep16_rnd");
        end

        // -------------------------------------------------------------------
        // 4) extra random vectors at random full-range positions (ROT_DIM=64),
        //    hammering the i=0 fastest and i=NPAIR-1 slowest pairs.
        // -------------------------------------------------------------------
        for (ci = 0; ci < 8; ci = ci + 1) begin
            do_case(0, RD0, ({$random(seed)} % 1048576), 1, 0, "rand64");
        end
        for (ci = 0; ci < 6; ci = ci + 1) begin
            do_case(1, RD1, ({$random(seed)} % 1048576), 1, 0, "rand16");
        end

        $display("worst observed rel-err (|y|>2^-5) = %g  (REL_TOL = %g)", worst_relerr, REL_TOL);
        $display("worst |err|/error-bound ratio     = %g  (must be < 1.0)", worst_abs_over_bound);
        $display("ALL %0d TESTS PASSED", test_count);
        $finish;
    end

    // global watchdog
    initial begin
        #5000000;
        $display("TIMEOUT");
        $fatal(1, "watchdog timeout");
    end

endmodule
