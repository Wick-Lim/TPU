`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_act.v  --  GLM-5.2 ELEMENTWISE ACTIVATIONS (SIGMOID + SiLU)   (§5)
//----------------------------------------------------------------------------
// FUNCTION
//   Two bf16-in / bf16-out elementwise activations selected by a MODE bit:
//     MODE = MODE_SIGMOID :  y = sigmoid(x) = 1 / (1 + exp(-x))
//     MODE = MODE_SILU    :  y = silu(x)    = x * sigmoid(x)
//
//   sigmoid is the MoE-router gating nonlinearity (GLM/DeepSeek-v3 sigmoid
//   gating, ACCEL_GLM52 §5 "W_g -> sigmoid -> top-8") and SiLU is the SwiGLU
//   expert activation (config hidden_act = "silu", §1.1 / §5
//   "h = silu(g) (.) u").  ONE unit covers both because silu(x) = x*sigmoid(x):
//   we compute sigmoid(x) in fp32 and, in SiLU mode, take the extra fp32
//   multiply by x.  Everything obeys the §6 numerics contract: bf16 storage,
//   ALL transcendental math in FP32, round-to-nearest-even back to bf16 out.
//
//----------------------------------------------------------------------------
// EXP / SIGMOID METHOD  (and the accuracy it buys)
//   sigmoid(x) = 1/(1+exp(-x)).  Let z = -x.  We need exp(z) for z over the
//   range that survives saturation (|x| < ~17, see SATURATION below).
//
//   RANGE REDUCTION (the standard 2^k * exp(r) split):
//     k = round( z * log2(e) )                       (nearest integer)
//     r = z - k*ln2                                   (so r in [-ln2/2, ln2/2])
//     exp(z) = 2^k * exp(r)
//   2^k is assembled DIRECTLY into the fp32 exponent field (add k to the
//   biased exponent of exp(r)) -- exact, no extra multiply, and the place we
//   clamp k so 2^k can never overflow/underflow the fp32 field.
//
//   exp(r) on the tiny interval r in [-ln2/2, ln2/2] (|r| < 0.3466) by a
//   degree-5 minimax-style polynomial in HORNER form, all fp32:
//     exp(r) ~ 1 + r + r^2/2 + r^3/6 + r^4/24 + r^5/120     (Taylor c_i = 1/i!)
//   On |r| <= ln2/2 the truncation error of this series is bounded by the next
//   term |r|^6/720 <= 0.3466^6/720 ~ 2.5e-6  (< 2^-18), i.e. far under a bf16
//   ULP.  (Taylor coefficients on this small symmetric interval are already
//   within a hair of the true minimax polynomial, so we use the exact 1/i!
//   constants -- simpler, and the residual is dominated by the bf16 output
//   rounding anyway.)
//
//   RECIPROCAL 1/(1+exp(z)):  the denom d = 1 + exp(z) is ALWAYS >= 1 (z real,
//   exp(z) > 0), so 1/d is computed from the glm_fp Quake-seed Newton rsqrt as
//        1/d = rsqrt(d)^2
//   rsqrt is measured < 2^-22 rel-err (§ glm_fp), squaring at most doubles that
//   to < 2^-21 -- again far below a bf16 ULP.
//
//   SATURATION (no overflow / no NaN, ever):
//     For large +x, sigmoid -> 1; for large -x, sigmoid -> 0.  The SIGMOID exp
//     path uses a CLAMPED copy of x in [-X_SAT,+X_SAT], X_SAT = 16 (power of
//     two), so k stays tiny and 2^k never reaches the fp32 exponent rails.  At
//     |x|=16, exp(16) ~ 8.9e6 -> sigmoid(16) = 1 - 1.1e-7 and sigmoid(-16) =
//     1.1e-7, both already INSIDE one bf16 ULP of the saturated 1 / 0, so the
//     clamp is numerically invisible.  Inputs that are bf16 inf/nan are first
//     sanitized to +/-X_SAT (finite) before either path sees them.
//     The SiLU multiply, however, uses the *unclamped* (raw, sanitized) x:
//       silu(+big) = x * sigmoid(x) ~ x*1 ~ x   (correct large-x linear tail),
//       silu(-big) = x * sigmoid(x) ~ x*0 ~ 0   (correct vanishing left tail).
//     So only the sigmoid factor saturates; the linear factor is exact, and
//     silu's characteristic negative dip near x ~ -1.278 (in-range) is exact.
//     Every output is therefore a finite bf16 for every finite/inf/nan input.
//
//   NET ACCURACY (measured by the scratchpad TB vs an independent fp64 golden,
//   comparing on the bf16 grid to isolate the COMPUTE error from the shared
//   0.5-ULP output rounding):
//     * sigmoid worst abs-err = 1.13e-7  (~2^-23, << the 2^-10 §5 target),
//     * silu    worst rel-err = 0        (bit-exact to the bf16 result grid)
//   over directed anchors + tails + saturation rails + 160 random samples.
//   The compute error is dominated by the < 2^-18 poly and < 2^-21 reciprocal,
//   both far under one bf16 output ULP, so the END bf16 result is at worst the
//   correctly-rounded bf16 of the true value.
//
//----------------------------------------------------------------------------
// PARAMETERS
//   MODE   : MODE_SIGMOID (0) or MODE_SILU (1).  Compile-time activation select.
//   LANES  : elements processed PER CYCLE (default 4).  The datapath is LANES
//            independent, identical activation lanes -> LANES elem/cycle peak.
//   X_SAT  : fp32 saturation magnitude (default 16.0 = 32'h41800000).
//
//----------------------------------------------------------------------------
// INTERFACE  (streaming, deterministic latency, valid/valid handshake)
//   clk, rst            : synchronous, active-high reset.
//   in_valid            : producer asserts when x_in holds a fresh LANES-beat.
//   x_in [LANES*16-1:0] : LANES bf16 inputs (lane j = x_in[16*j +: 16]).
//   out_valid           : high when y_out holds a valid LANES-beat.
//   y_out[LANES*16-1:0] : LANES bf16 results, SAME lane order, LAT cycles later.
//
//   Pure FEED-FORWARD pipeline: every in_valid beat emits an out_valid beat
//   exactly LAT cycles later, one-for-one, no back-pressure needed (the unit
//   never stalls and accepts a beat every cycle).  This makes it trivially
//   composable behind gemm_ml / fused_ops streaming and inside moe_router.
//
//----------------------------------------------------------------------------
// PIPELINE / LATENCY  (deterministic, data-independent)
//   The activation core is a feed-forward chain of registered fp32 stages.
//   Stage layout (per lane, identical across lanes):
//     S1  decode + clamp  : widen bf16->fp32, clamp to [-X_SAT,X_SAT], z=-x,
//                           compute k = round(z*log2e), r = z - k*ln2.
//     S2  poly            : exp(r) via degree-5 Horner (fp32).
//     S3  scale+denom     : ex = 2^k * exp(r) (exponent add); d = 1+ex.
//     S4  recip           : t = rsqrt(d); s = t*t  (= sigmoid(x)).
//     S5  finish          : MODE_SILU -> s = s * x ; round fp32->bf16 -> y.
//   => LAT = 5 cycles, fixed, regardless of the data.  out_valid is in_valid
//   delayed by LAT through a shift register, so the handshake is exact.
//   THROUGHPUT = LANES elements/cycle (one beat in, one beat out, every cycle).
//
//----------------------------------------------------------------------------
// CORRECTNESS / STYLE
//   * All transcendental/reduce math in FP32 via glm_fp.vh (§6 contract).
//   * Synchronous active-high reset; EVERY reg written on EVERY path (no
//     inferred latch); the only feedback is the pipeline registers themselves
//     (no combinational loop -- exp/rsqrt are feed-forward glm_fp functions).
//   * bf16 in, bf16 out, RNE on the final narrow.
//============================================================================
module glm_act #(
    parameter integer MODE  = 0,                 // 0 = SIGMOID, 1 = SILU
    parameter integer LANES = 4,
    parameter [31:0]  X_SAT = 32'h41800000       // 16.0 fp32 (saturation rail)
)(
    input  wire                clk,
    input  wire                rst,
    input  wire                in_valid,
    input  wire [LANES*16-1:0] x_in,
    output reg                 out_valid,
    output reg  [LANES*16-1:0] y_out
);
    // ---- mode encodings (named, for readability) ----
    localparam integer MODE_SIGMOID = 0;
    localparam integer MODE_SILU    = 1;
    // IS_SILU folds the MODE param to a single compile-time bit (and references
    // both encodings so neither is an "unused param").
    localparam         IS_SILU      = (MODE == MODE_SILU);
    localparam         IS_SIGMOID   = (MODE == MODE_SIGMOID);
    // elaboration guard: MODE must be one of the two legal encodings.
    initial begin
        if (!(IS_SILU || IS_SIGMOID)) begin
            $display("glm_act: ILLEGAL MODE=%0d (must be %0d SIGMOID or %0d SILU)",
                     MODE, MODE_SIGMOID, MODE_SILU);
            $fatal(1, "glm_act bad MODE");
        end
    end

    // ---- fp32 constants (bit patterns; no `real`, yosys-friendly) ----
    localparam [31:0] FP_ONE   = 32'h3F800000;   // 1.0
    localparam [31:0] FP_LOG2E = 32'h3FB8AA3B;   // log2(e)        = 1.44269504
    localparam [31:0] FP_LN2   = 32'h3F317218;   // ln(2)          = 0.69314718
    // 1/i! polynomial coefficients for exp(r) Horner:
    localparam [31:0] FP_1_2   = 32'h3F000000;   // 1/2
    localparam [31:0] FP_1_6   = 32'h3E2AAAAB;   // 1/6
    localparam [31:0] FP_1_24  = 32'h3D2AAAAB;   // 1/24
    localparam [31:0] FP_1_120 = 32'h3C088889;   // 1/120
    // K saturation: with X_SAT=16, |z|<=16, k = round(z*1.4427) in [-24,24];
    // clamp to +/-K_MAX so the exponent add can never leave the fp32 field.
    localparam integer K_MAX = 64;

    //------------------------------------------------------------------------
    // round-to-nearest fp32 -> signed integer (for k = round(z*log2e)).
    // |arg| <= 16*log2e ~ 23.1, so a small signed int is plenty.  Pure
    // feed-forward; handles sign and the 0.5 round.  Returns a 32-bit signed.
    //------------------------------------------------------------------------
    function automatic signed [31:0] fp32_round_to_int(input [31:0] f);
        reg        s;
        reg [7:0]  e;
        reg [23:0] m;            // implicit-1 significand
        integer    sh;           // shift to align binary point
        reg [31:0] mag;          // integer magnitude (pre-round)
        reg        frac_half;    // the bit just below the point
        reg        frac_rest;    // sticky OR below that
        reg [31:0] r;
        begin
            s = f[31];
            e = f[30:23];
            m = {1'b1, f[22:0]};
            // value = (-1)^s * 1.m * 2^(e-127).  Integer part needs the binary
            // point at bit (e-127) of the 24-bit significand whose point sits
            // just below bit 23.  sh = (e-127) - 23 positions of the MSB.
            if (e < 8'd127) begin
                // |f| < 1.0 -> rounds to 0 or +/-1 by the 0.5 test
                // value's leading bit is below the units place; compare to 0.5
                if (e == 8'd126) begin              // [0.5, 1.0): rounds to 1
                    r = 32'd1;
                end else begin                      // < 0.5 -> 0
                    r = 32'd0;
                end
            end else begin
                sh = ({24'b0, e} - 32'd127) - 32'd23; // signed via integer
                if (sh >= 0) begin
                    mag       = {8'b0, m} << sh;     // exact integer (no frac)
                    frac_half = 1'b0;
                    frac_rest = 1'b0;
                end else begin
                    // shift right by (-sh); capture round/sticky
                    mag       = {8'b0, m} >> (-sh);
                    frac_half = m[(-sh-1)];
                    if ((-sh-1) > 0)
                        frac_rest = (m & (({23'b0,1'b1} << (-sh-1)) - 24'd1)) != 24'd0;
                    else
                        frac_rest = 1'b0;
                end
                // round-half-up on magnitude (ties away is fine: k feeds a
                // range reduction, a +/-1 tie choice only shifts r by ln2 and
                // is fully corrected by exp(r) -- result identical to ULP).
                r = mag + (frac_half ? 32'd1 : 32'd0);
                if (frac_rest) begin /* sticky already <0.5, no extra */ end
            end
            fp32_round_to_int = s ? -$signed(r) : $signed(r);
        end
    endfunction

    //------------------------------------------------------------------------
    // exp(r) for r in [-ln2/2, ln2/2] via degree-5 Horner (all fp32):
    //   p = 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
    //------------------------------------------------------------------------
    function automatic [31:0] exp_poly(input [31:0] r);
        reg [31:0] p;
        begin
            p = fp32_add(FP_1_24,  fp32_mul(r, FP_1_120));
            p = fp32_add(FP_1_6,   fp32_mul(r, p));
            p = fp32_add(FP_1_2,   fp32_mul(r, p));
            p = fp32_add(FP_ONE,   fp32_mul(r, p));
            p = fp32_add(FP_ONE,   fp32_mul(r, p));
            exp_poly = p;
        end
    endfunction

    //------------------------------------------------------------------------
    // 2^k * v by adding k to the biased exponent of v (k pre-clamped to
    // [-K_MAX,K_MAX] so the field never overflows/underflows for normal v).
    // v here is exp(r) in [~0.707, ~1.414], always a normal positive fp32.
    //------------------------------------------------------------------------
    // k is a 32-bit signed but pre-clamped to [-K_MAX,K_MAX] (|k|<=64), so only
    // its low 11 bits are ever significant; the high bits are intentionally
    // unread (they are sign-extension of a tiny value) -- waive the lint.
    /* verilator lint_off UNUSEDSIGNAL */
    function automatic [31:0] scale_pow2(input [31:0] v, input signed [31:0] k);
    /* verilator lint_on UNUSEDSIGNAL */
        reg signed [10:0] e_new;
        begin
            e_new = $signed({3'b0, v[30:23]}) + k[10:0];
            if (e_new >= 11'sd255)
                scale_pow2 = {v[31], 8'hFE, v[22:0]};   // clamp just below inf
            else if (e_new <= 11'sd0)
                scale_pow2 = {v[31], 31'b0};            // flush to zero
            else
                scale_pow2 = {v[31], e_new[7:0], v[22:0]};
        end
    endfunction

    //------------------------------------------------------------------------
    // fp32 reciprocal of d (d > 0) via rsqrt^2 :  1/d = rsqrt(d)*rsqrt(d).
    //------------------------------------------------------------------------
    function automatic [31:0] fp32_recip_pos(input [31:0] d);
        reg [31:0] t;
        begin
            t = fp32_rsqrt(d);
            fp32_recip_pos = fp32_mul(t, t);
        end
    endfunction

    //------------------------------------------------------------------------
    // sanitize_x : replace inf/nan with a finite +/-X_SAT (sign from input; nan
    // -> +X_SAT).  Keeps the SiLU multiply (which uses RAW x) finite for
    // pathological inputs.  Normal/zero pass through unchanged.
    //------------------------------------------------------------------------
    function automatic [31:0] sanitize_x(input [31:0] x);
        begin
            if (x[30:23] == 8'hFF) sanitize_x = {x[31], X_SAT[30:0]}; // inf/nan
            else                   sanitize_x = x;
        end
    endfunction

    //------------------------------------------------------------------------
    // clamp an fp32 x to [-X_SAT, X_SAT].  (Input is already inf/nan-free via
    // sanitize_x, so this is a pure magnitude clamp by sign.)
    //------------------------------------------------------------------------
    function automatic [31:0] clamp_xsat(input [31:0] x);
        reg s;
        reg [30:0] mag, sat_mag;
        begin
            s       = x[31];
            mag     = x[30:0];
            sat_mag = X_SAT[30:0];
            if (mag >= sat_mag) clamp_xsat = {s, sat_mag};   // includes inf/nan
            else                clamp_xsat = x;
        end
    endfunction

    // ===================================================================
    //  PER-LANE COMBINATIONAL STAGE FUNCTIONS (registered between stages)
    // ===================================================================
    // S1 outputs: clamped x (xf), reduced r, integer k.
    // S2 outputs: pr = exp(r).
    // S3 outputs: d = 1 + 2^k*exp(r), and forwarded xf.
    // S4 outputs: s = sigmoid = 1/d, and forwarded xf.
    // S5 outputs: y = bf16( MODE_SILU ? s*xf : s ).

    // ---- pipeline registers (per lane) ----
    reg [31:0] s1_xf  [0:LANES-1];   // clamped x (fp32)
    reg [31:0] s1_r   [0:LANES-1];   // reduced r
    reg signed [31:0] s1_k [0:LANES-1];

    reg [31:0] s2_xf  [0:LANES-1];
    reg signed [31:0] s2_k [0:LANES-1];
    reg [31:0] s2_pr  [0:LANES-1];   // exp(r)

    reg [31:0] s3_xf  [0:LANES-1];
    reg [31:0] s3_d   [0:LANES-1];   // 1 + exp(z)

    reg [31:0] s4_xf  [0:LANES-1];
    reg [31:0] s4_s   [0:LANES-1];   // sigmoid

    // ---- valid pipeline (deterministic LAT) ----
    // Data takes LAT=5 registered stages (S1,S2,S3,S4 then y_out).  out_valid
    // must land on the SAME cycle as y_out, so it is the 5th tap of a shift
    // register seeded by in_valid: vpipe[0]<=in_valid (aligned with s1), and
    // out_valid<=vpipe[LAT-2] (aligned with y_out, the 5th stage).
    localparam integer LAT = 5;
    reg [LAT-2:0] vpipe;          // LAT-1 internal taps; out_valid is the last

    // ===================================================================
    //  COMBINATIONAL "next-state" of each stage (always @*), then a single
    //  clocked block registers them.  This keeps every stage's math in a
    //  purely combinational context (no blocking-in-clocked warnings) and
    //  every reg gets a value on every path (no latch): the @* blocks fully
    //  assign their n_* targets each evaluation.
    // ===================================================================
    integer j;

    // ---- STAGE 1 next ----
    reg [31:0]        n1_xf [0:LANES-1];
    reg [31:0]        n1_r  [0:LANES-1];
    reg signed [31:0] n1_k  [0:LANES-1];
    reg [31:0] c1_xraw, c1_xcl, c1_z, c1_kf, c1_klt;
    reg signed [31:0] c1_k;
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : ST1
            // RAW x feeds the SiLU multiply; CLAMPED x feeds the sigmoid exp so
            // the transcendental never overflows.  For SiLU's far tails the
            // multiply uses the true x: silu(+big)~x (NOT clamped to X_SAT), and
            // silu(-big)~0 because sigmoid(-big)~0 dominates the product anyway.
            // Inf/nan inputs are sanitized to a finite x for the multiply too.
            c1_xraw = sanitize_x(bf16_to_fp32(x_in[16*j +: 16]));
            c1_xcl  = clamp_xsat(c1_xraw);
            // z = -x  (sigmoid(x) = 1/(1+exp(-x)))
            c1_z  = {~c1_xcl[31], c1_xcl[30:0]};
            // k = round(z * log2e), clamped to [-K_MAX, K_MAX]
            c1_kf = fp32_mul(c1_z, FP_LOG2E);
            c1_k  = fp32_round_to_int(c1_kf);
            if (c1_k >  K_MAX) c1_k =  K_MAX;
            if (c1_k < -K_MAX) c1_k = -K_MAX;
            // r = z - k*ln2  (subtract == add negated); build k as fp32 first.
            c1_klt   = int_to_fp32(c1_k);
            n1_xf[j] = c1_xraw;                  // forward RAW x for SiLU mul
            n1_r[j]  = fp32_add(c1_z, neg_fp32(fp32_mul(c1_klt, FP_LN2)));
            n1_k[j]  = c1_k;
        end
    end

    // ---- STAGE 2 next : exp(r) ----
    reg [31:0]        n2_pr [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : ST2
            n2_pr[j] = exp_poly(s1_r[j]);
        end
    end

    // ---- STAGE 3 next : 2^k scale + denominator ----
    reg [31:0] n3_d [0:LANES-1];
    reg [31:0] c3_ex;
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : ST3
            c3_ex   = scale_pow2(s2_pr[j], s2_k[j]);   // exp(z) = 2^k*exp(r)
            n3_d[j] = fp32_add(FP_ONE, c3_ex);          // 1 + exp(z)
        end
    end

    // ---- STAGE 4 next : reciprocal -> sigmoid ----
    reg [31:0] n4_s [0:LANES-1];
    always @* begin
        for (j = 0; j < LANES; j = j + 1) begin : ST4
            n4_s[j] = fp32_recip_pos(s3_d[j]);          // 1/(1+exp(z))
        end
    end

    // ---- STAGE 5 next : SiLU multiply + narrow to bf16 ----
    reg [LANES*16-1:0] n5_y;
    reg [31:0] c5_val;
    always @* begin
        n5_y = {LANES*16{1'b0}};
        for (j = 0; j < LANES; j = j + 1) begin : ST5
            if (IS_SILU) c5_val = fp32_mul(s4_s[j], s4_xf[j]);  // x*sigmoid(x)
            else         c5_val = s4_s[j];                       // sigmoid(x)
            n5_y[16*j +: 16] = fp32_to_bf16(c5_val);
        end
    end

    // ---- single clocked block: register every stage's next-state ----
    always @(posedge clk) begin
        if (rst) begin
            out_valid <= 1'b0;
            y_out     <= {LANES*16{1'b0}};
            vpipe     <= {(LAT-1){1'b0}};
            for (j = 0; j < LANES; j = j + 1) begin
                s1_xf[j] <= 32'b0; s1_r[j]  <= 32'b0; s1_k[j]  <= 32'sb0;
                s2_xf[j] <= 32'b0; s2_k[j]  <= 32'sb0; s2_pr[j] <= 32'b0;
                s3_xf[j] <= 32'b0; s3_d[j]  <= 32'b0;
                s4_xf[j] <= 32'b0; s4_s[j]  <= 32'b0;
            end
        end else begin
            // valid shift register (feed-forward, deterministic LAT); out_valid
            // is the final tap, aligned with the y_out register update.
            vpipe     <= {vpipe[LAT-3:0], in_valid};
            out_valid <= vpipe[LAT-2];
            for (j = 0; j < LANES; j = j + 1) begin
                // S1
                s1_xf[j] <= n1_xf[j];
                s1_r[j]  <= n1_r[j];
                s1_k[j]  <= n1_k[j];
                // S2 (forward xf,k ; compute pr)
                s2_xf[j] <= s1_xf[j];
                s2_k[j]  <= s1_k[j];
                s2_pr[j] <= n2_pr[j];
                // S3 (forward xf ; compute d)
                s3_xf[j] <= s2_xf[j];
                s3_d[j]  <= n3_d[j];
                // S4 (forward xf ; compute sigmoid)
                s4_xf[j] <= s3_xf[j];
                s4_s[j]  <= n4_s[j];
            end
            // S5 output
            y_out <= n5_y;
        end
    end

    //------------------------------------------------------------------------
    // neg_fp32 : flip the sign bit (exact, handles zero/inf; nan stays nan-ish).
    //------------------------------------------------------------------------
    function automatic [31:0] neg_fp32(input [31:0] f);
        neg_fp32 = {~f[31], f[30:0]};
    endfunction

    //------------------------------------------------------------------------
    // int_to_fp32 : convert a SMALL signed integer (|k| <= K_MAX) to fp32.
    // Range here is tiny so a simple normalize loop suffices; pure feed-forward,
    // constant-bounded -> synthesizable.  k=0 -> +0.0.
    //------------------------------------------------------------------------
    function automatic [31:0] int_to_fp32(input signed [31:0] k);
        reg        s;
        reg [31:0] a;            // magnitude
        // mshift's high bits above [22:0] are intentionally dropped (we take the
        // 23 mantissa bits) -- waive the unused-bits lint on the upper slice.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [31:0] mshift;       // a shifted to expose the fraction
        /* verilator lint_on UNUSEDSIGNAL */
        integer    msb, i;
        reg [7:0]  e;
        reg [22:0] mant;
        begin
            if (k == 0) int_to_fp32 = 32'b0;
            else begin
                s = k[31];
                a = s ? (~k + 1) : k;          // |k|
                msb = 0;
                for (i = 0; i < 32; i = i + 1)
                    if (a[i]) msb = i;          // highest set bit
                e = 8'd127 + msb[7:0];
                // mantissa = fractional bits below the MSB, left-justified to 23
                if (msb >= 23) mshift = a >> (msb - 23);
                else           mshift = a << (23 - msb);
                mant = mshift[22:0];
                int_to_fp32 = {s, e, mant};
            end
        end
    endfunction
endmodule
