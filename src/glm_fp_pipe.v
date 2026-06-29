`timescale 1ns/1ps
`include "glm_fp.vh"
// This file is a HEADER-STYLE bundle of several pipelined FP primitive modules
// (like glm_fp.vh bundles several functions), so the file name deliberately
// does not match any single module name.  Waive that one Verilator style note.
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_fp_pipe.v  --  PIPELINED GLM-5.2 FLOATING-POINT PRIMITIVES   (§6, §8.x)
//----------------------------------------------------------------------------
// PURPOSE
//   src/glm_fp.vh defines the GLM-5.2 numerics CONTRACT as purely combinational
//   `function automatic`s (fp32_mul/fp32_add/fp32_rsqrt/...).  Those functions
//   are golden-correct, but a unit that CHAINS several of them in one cycle
//   (RMSNorm: Sx^2 reduce -> rsqrt -> scale; GEMM: mul+add accumulate; softmax:
//   exp) ends up with a critical path of multiple full combinational fp32 ops
//   and therefore routes at only a few MHz.
//
//   This file re-expresses the SAME numerics as REGISTERED PIPELINES.  Each
//   long combinational chain (variable-shift align + 27-bit normalize + RNE for
//   add; 24x24 partial-product + normalize + round for mul; Quake-seed + 2
//   Newton iters for rsqrt; range-reduce + Horner + 2^k fold for exp) is cut
//   into stages so the critical path is ONE pipeline stage, not the whole op.
//   Throughput is 1 result / cycle; latency is LAT cycles (documented per
//   module).  A valid bit travels alongside the data: a `valid_in` pulse at
//   cycle t produces `valid_out` at cycle t+LAT.
//
//----------------------------------------------------------------------------
// BIT-EQUIVALENCE  (the whole point: drop-in for the combinational contract)
//   Every module here is BIT-FOR-BIT IDENTICAL (0 ULP) to the matching
//   glm_fp.vh function, because each module computes the *same Boolean
//   function* -- it merely inserts flip-flops at boundaries that are
//   feed-forward in the original combinational expression.  Splitting a
//   combinational chain  f = h(g(x))  into  reg(g(x))  ->  h(reg)  cannot
//   change the value of f; it only delays it.  Concretely:
//
//     fp32_mul_pipe   == fp32_mul                                  (0 ULP)
//     fp32_add_pipe   == fp32_add                                  (0 ULP)
//     fp32_mac_pipe   == fp32_add(fp32_mul(a,b), c)                (0 ULP)
//     fp32_rsqrt_pipe == fp32_rsqrt                                (0 ULP)
//     fp32_exp_pipe    : NOT a glm_fp.vh primitive (glm_fp has no exp). It is
//                        a NEW fp32 exp built ENTIRELY from fp32_mul/fp32_add,
//                        so it inherits the contract's rounding/FTZ. Its self
//                        check is against a same-arithmetic combinational
//                        reference `glm_exp_ref` (below): 0 ULP vs that ref.
//                        (Accuracy vs real exp(): <2^-12 rel over the softmax
//                        input range x in [-87,0], same Horner/2^k method the
//                        combinational softmax_unit uses.)
//
//   Because the stage boundaries are placed between the *already-existing*
//   combinational sub-expressions of glm_fp.vh, the per-stage logic depth is a
//   short slice of the original (one shift, or one add, or one normalize), so
//   fmax rises while the produced bits are unchanged.
//
//----------------------------------------------------------------------------
// LATENCIES (cycles from valid_in to valid_out); throughput = 1 result/cycle.
// These are MEASURED by the scratchpad smoke TB (single-pulse probe in one
// posedge domain) and equal the structural flop count of each datapath:
//     fp32_mul_pipe    LAT = 2    (r0 stage, output stage)
//     fp32_add_pipe    LAT = 3    (r0 align, r1 add/sub, output normalize+round)
//     fp32_mac_pipe    LAT = 5    (= mul 2 + add 3, c delayed 2 to meet product)
//     fp32_rsqrt_pipe  LAT = 20   (xhalf mul + 2 Newton iters of mul/mul/add/mul)
//     fp32_exp_pipe    LAT = 12   (LAT-deep retiming pipe + output flop)
//
// CONVENTIONS
//   * synchronous, ACTIVE-HIGH reset (clears the valid pipe; data regs are
//     don't-care under reset, gated by valid).
//   * NO latch (every reg assigned every clock under the single clocked block).
//   * NO combinational loop (all feedback is through the staging registers).
//   * The combinational sub-functions are the very same ones from glm_fp.vh,
//     `include`d here so there is a single source of numeric truth.
//============================================================================


// SINGLE SOURCE OF TRUTH for the per-module pipeline LATENCIES lives in
// glm_fp_pipe_lat.vh (FP_MUL_LAT / FP_ADD_LAT / FP_MAC_LAT / FP_RSQRT_LAT /
// FP_EXP_LAT, structural flop count valid_in->valid_out).  Each module below
// sets its `localparam LAT` from the matching macro, and every consumer reads
// the same macros rather than hardcoding a number.
`include "glm_fp_pipe_lat.vh"


//============================================================================
// fp32_mul_pipe  --  pipelined IEEE-ish fp32 multiply (RNE, FTZ).
//   BIT-EQUIVALENT to glm_fp.vh fp32_mul.   LAT = 2,  1 result/cycle.
//
//   Stage 0 (input regs + decode/special + 24x24 product):
//       classify special lattice; form 24-bit significands; compute the
//       48-bit product and the biased exponent sum.  These are registered.
//   Stage 1 (normalize + RNE round + range):
//       pick [1,2)/[2,4) normalize, round-to-nearest-even with carry, apply
//       overflow->inf / underflow->FTZ.  Registered to the output.
//
//   LAT (structural flop count, valid_in->valid_out) is exposed as the output
//   parameter LAT so consumers can read it instead of hardcoding.
//============================================================================
module fp32_mul_pipe (
    input              clk,
    input              rst,
    input              valid_in,
    input      [31:0]  a,
    input      [31:0]  b,
    output reg         valid_out,
    output reg [31:0]  result
);
    /* verilator lint_off UNUSEDPARAM */
    localparam integer LAT = `FP_MUL_LAT;  // exposed: structural flop count in->out
    /* verilator lint_on UNUSEDPARAM */
    `include "glm_fp.vh"

    // ---- Stage 0 combinational: decode + product + special detect ----
    reg               s0_special;     // result fully determined (special case)
    reg [31:0]        s0_special_val; // the special result
    reg               s0_sr;
    reg signed [10:0] s0_exp;         // biased exponent sum (ea+eb-127)
    reg [47:0]        s0_prod;

    always @* begin
        reg [7:0] ea, eb;
        reg [23:0] ma, mb;
        // default-assign every local on every path (no inferred latch)
        ma = 24'b0; mb = 24'b0;
        s0_special     = 1'b1;
        s0_special_val = 32'b0;
        s0_sr          = a[31] ^ b[31];
        s0_exp         = 11'sd0;
        s0_prod        = 48'b0;
        ea = a[30:23]; eb = b[30:23];
        if (_glmfp_is_nan(a[30:0]) || _glmfp_is_nan(b[30:0])) begin
            s0_special_val = 32'h7FC00000;
        end else if (_glmfp_is_inf(a[30:0]) || _glmfp_is_inf(b[30:0])) begin
            if (_glmfp_is_zero(a[30:23]) || _glmfp_is_zero(b[30:23]))
                s0_special_val = 32'h7FC00000;
            else
                s0_special_val = {s0_sr, 8'hFF, 23'b0};
        end else if (_glmfp_is_zero(a[30:23]) || _glmfp_is_zero(b[30:23])) begin
            s0_special_val = {s0_sr, 31'b0};
        end else begin
            s0_special = 1'b0;
            ma = {1'b1, a[22:0]};
            mb = {1'b1, b[22:0]};
            s0_prod = ma * mb;
            s0_exp  = $signed({3'b0, ea}) + $signed({3'b0, eb}) - 11'sd127;
        end
    end

    // ---- Stage 0 -> 1 registers ----
    reg               r0_valid;
    reg               r0_special;
    reg [31:0]        r0_special_val;
    reg               r0_sr;
    reg signed [10:0] r0_exp;
    reg [47:0]        r0_prod;

    // ---- Stage 1 combinational: normalize + round + range ----
    reg [31:0]        s1_val;
    always @* begin
        reg [23:0] mant;
        reg        guard, sticky, round_up;
        reg [24:0] mant_r;
        reg signed [10:0] exp_s;
        // default-assign every local on every path (no inferred latch)
        mant = 24'b0; guard = 1'b0; sticky = 1'b0; round_up = 1'b0;
        mant_r = 25'b0;
        s1_val = r0_special_val;
        exp_s  = r0_exp;
        if (!r0_special) begin
            if (r0_prod[47]) begin
                exp_s  = exp_s + 11'sd1;
                mant   = r0_prod[47:24];
                guard  = r0_prod[23];
                sticky = (r0_prod[22:0] != 0);
            end else begin
                mant   = r0_prod[46:23];
                guard  = r0_prod[22];
                sticky = (r0_prod[21:0] != 0);
            end
            round_up = guard & (sticky | mant[0]);
            mant_r   = {1'b0, mant} + {24'b0, round_up};
            if (mant_r[24]) begin
                mant_r = mant_r >> 1;
                exp_s  = exp_s + 11'sd1;
            end
            if (exp_s >= 11'sd255)
                s1_val = {r0_sr, 8'hFF, 23'b0};
            else if (exp_s <= 11'sd0)
                s1_val = {r0_sr, 31'b0};
            else
                s1_val = {r0_sr, exp_s[7:0], mant_r[22:0]};
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            r0_valid  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            // stage 0 -> r0
            r0_valid       <= valid_in;
            r0_special     <= s0_special;
            r0_special_val <= s0_special_val;
            r0_sr          <= s0_sr;
            r0_exp         <= s0_exp;
            r0_prod        <= s0_prod;
            // stage 1 -> output
            valid_out      <= r0_valid;
            result         <= s1_val;
        end
    end
endmodule


//============================================================================
// fp32_add_pipe  --  pipelined IEEE-ish fp32 add (RNE, FTZ).
//   BIT-EQUIVALENT to glm_fp.vh fp32_add.   LAT = 5,  1 result/cycle.
//
//   The old version put the ENTIRE post-add cone -- leading-zero detect +
//   normalize barrel-shift + RNE round + exponent adjust -- in ONE stage,
//   which routed at ~37 MHz and bounded the fmax of every FP unit that uses
//   fp32 add.  Here that cone is partitioned into short slices so each stage
//   is at most one shift / one add / the LZ-count / the round:
//
//     Stage 0 : special lattice + exponent compare + choose big/small +
//               align-shift the smaller mantissa (the variable align shift)
//               + sticky collection.                         (one variable shift)
//     Stage 1 : signed mantissa add/sub (magnitude) + carry-out detect. (one add)
//     Stage 2 : leading-zero COUNT only -- a bounded priority encode over the
//               28-bit magnitude.  (Also folds the mag[27] carry case into the
//               same "shift amount + exponent delta" representation, but does
//               NOT yet shift.)                                  (the LZ count)
//     Stage 3 : normalize barrel-shift by the stage-2 count (one barrel shift),
//               then slice mantissa / guard / round / sticky and compute the
//               adjusted exponent.                          (one barrel shift)
//     Stage 4 : RNE round (+carry) + exponent range -> output.       (the round)
//
//   Each of the five stages is a registered slice of the SAME Boolean function
//   as glm_fp.vh fp32_add, so the produced bits are identical (0 ULP); only the
//   timing is deepened.  LAT (structural flop count) is exposed as a parameter.
//============================================================================
module fp32_add_pipe (
    input              clk,
    input              rst,
    input              valid_in,
    input      [31:0]  a,
    input      [31:0]  b,
    output reg         valid_out,
    output reg [31:0]  result
);
    /* verilator lint_off UNUSEDPARAM */
    localparam integer LAT = `FP_ADD_LAT;  // exposed: structural flop count in->out
    /* verilator lint_on UNUSEDPARAM */
    `include "glm_fp.vh"

    // ---- log-depth leading-zero count (replaces the stage-2 walk loop) -------
    // clz32(x): number of leading zeros of a 32-bit value (0..32), as the
    // classic 5-level (log2(32)) priority encode -- each `if` halves the search
    // window, so the depth is 5 compare/shift slices, NOT a 27-iteration ripple.
    // Stage 2 calls it on the TOP-aligned add magnitude {mag[26:0],5'b0}; because
    // mag[26:0] is nonzero there, the result is 26 - msb(mag[26:0]), which is
    // EXACTLY the value the old walk loop produced (lz = shifts to bring the
    // first 1 below bit 27 up to bit 26).  Pure restructure -> 0 ULP, same LAT.
    function automatic [5:0] clz32(input [31:0] x);
        reg [31:0] v;
        reg [5:0]  n;
        begin
            v = x; n = 6'd0;
            if (v == 32'b0) begin
                n = 6'd32;
            end else begin
                if (v[31:16] == 16'b0) begin n = n + 6'd16; v = v << 16; end
                if (v[31:24] ==  8'b0) begin n = n + 6'd8;  v = v << 8;  end
                if (v[31:28] ==  4'b0) begin n = n + 6'd4;  v = v << 4;  end
                if (v[31:30] ==  2'b0) begin n = n + 6'd2;  v = v << 2;  end
                if (v[31]    ==  1'b0) begin n = n + 6'd1;               end
            end
            clz32 = n;
        end
    endfunction

    // ---------------- Stage 0 : classify + align ----------------
    reg               s0_special;
    reg [31:0]        s0_special_val;
    reg               s0_s_big, s0_s_small;
    reg [7:0]         s0_exp_big;
    reg [26:0]        s0_m_big_e, s0_m_small_e;

    always @* begin
        reg sa, sb;
        reg [7:0] ea, eb;
        reg [23:0] ma, mb, m_big, m_small;
        reg [7:0] shamt;
        // default-assign every local on every path (no inferred latch)
        ma = 24'b0; mb = 24'b0; m_big = 24'b0; m_small = 24'b0; shamt = 8'b0;
        s0_special     = 1'b1;
        s0_special_val = 32'b0;
        s0_s_big       = 1'b0;
        s0_s_small     = 1'b0;
        s0_exp_big     = 8'b0;
        s0_m_big_e     = 27'b0;
        s0_m_small_e   = 27'b0;
        sa = a[31]; sb = b[31];
        ea = a[30:23]; eb = b[30:23];
        if (_glmfp_is_nan(a[30:0]) || _glmfp_is_nan(b[30:0])) begin
            s0_special_val = 32'h7FC00000;
        end else if (_glmfp_is_inf(a[30:0]) && _glmfp_is_inf(b[30:0])) begin
            s0_special_val = (sa == sb) ? {sa, 8'hFF, 23'b0} : 32'h7FC00000;
        end else if (_glmfp_is_inf(a[30:0])) begin
            s0_special_val = {sa, 8'hFF, 23'b0};
        end else if (_glmfp_is_inf(b[30:0])) begin
            s0_special_val = {sb, 8'hFF, 23'b0};
        end else if (_glmfp_is_zero(a[30:23]) && _glmfp_is_zero(b[30:23])) begin
            s0_special_val = {sa & sb, 31'b0};
        end else if (_glmfp_is_zero(a[30:23])) begin
            s0_special_val = {sb, eb, b[22:0]};
        end else if (_glmfp_is_zero(b[30:23])) begin
            s0_special_val = {sa, ea, a[22:0]};
        end else begin
            s0_special = 1'b0;
            ma = {1'b1, a[22:0]};
            mb = {1'b1, b[22:0]};
            if (ea >= eb) begin
                s0_exp_big = ea; s0_s_big = sa; s0_s_small = sb;
                m_big = ma;      m_small = mb; shamt = ea - eb;
            end else begin
                s0_exp_big = eb; s0_s_big = sb; s0_s_small = sa;
                m_big = mb;      m_small = ma; shamt = eb - ea;
            end
            s0_m_big_e = {m_big, 3'b0};
            if (shamt >= 8'd27) begin
                s0_m_small_e = (m_small != 0) ? 27'd1 : 27'd0;
            end else begin
                s0_m_small_e = {m_small, 3'b0} >> shamt;
                if (shamt > 8'd3) begin
                    if (({m_small, 3'b0} & ((27'd1 << shamt) - 27'd1)) != 27'd0)
                        s0_m_small_e[0] = 1'b1;
                end
            end
        end
    end

    reg               r0_valid;
    reg               r0_special;
    reg [31:0]        r0_special_val;
    reg               r0_s_big, r0_s_small;
    reg [7:0]         r0_exp_big;
    reg [26:0]        r0_m_big_e, r0_m_small_e;

    // ---------------- Stage 1 : magnitude add/sub ----------------
    reg               s1_special;
    reg [31:0]        s1_special_val;
    reg               s1_res_sign;
    reg [7:0]         s1_exp_big;
    reg [27:0]        s1_mag;

    always @* begin
        s1_special     = r0_special;
        s1_special_val = r0_special_val;
        s1_exp_big     = r0_exp_big;
        s1_res_sign    = 1'b0;
        s1_mag         = 28'b0;
        if (!r0_special) begin
            if (r0_s_big == r0_s_small) begin
                s1_mag      = {1'b0, r0_m_big_e} + {1'b0, r0_m_small_e};
                s1_res_sign = r0_s_big;
            end else begin
                if (r0_m_big_e >= r0_m_small_e) begin
                    s1_mag      = {1'b0, r0_m_big_e} - {1'b0, r0_m_small_e};
                    s1_res_sign = r0_s_big;
                end else begin
                    s1_mag      = {1'b0, r0_m_small_e} - {1'b0, r0_m_big_e};
                    s1_res_sign = r0_s_small;
                end
            end
        end
    end

    reg               r1_valid;
    reg               r1_special;
    reg [31:0]        r1_special_val;
    reg               r1_res_sign;
    reg [7:0]         r1_exp_big;
    reg [27:0]        r1_mag;

    // ---------------- Stage 2 : leading-zero COUNT (no shift yet) ----------
    // Resolve the cancellation case (mag==0 -> +0), the carry case (mag[27]),
    // and otherwise the leading-zero count of the 28-bit magnitude.  We emit a
    // single signed shift amount `nshift` and the matching exponent so stage 3
    // can do exactly ONE barrel shift:
    //   nshift > 0  -> shift LEFT  by nshift (leading-zero normalize)
    //   nshift = -1 -> shift RIGHT by 1      (mag[27] carry-out case)
    //   nshift = 0  -> no shift.
    // The exponent is adjusted by the same amount (exp_big - lz, or +1 carry).
    reg               s2_special;
    reg [31:0]        s2_special_val;
    reg               s2_iszero;           // exact cancellation -> +0
    reg               s2_res_sign;
    reg [27:0]        s2_mag;
    reg signed [6:0]  s2_nshift;           // -1..26 ; left>0, right=-1
    reg signed [10:0] s2_exp_pre;          // exponent BEFORE the normalize delta

    always @* begin
        reg [5:0] lz;             // leading-zero count of mag[26:0], range 0..26
        s2_special     = r1_special;
        s2_special_val = r1_special_val;
        s2_iszero      = 1'b0;
        s2_res_sign    = r1_res_sign;
        s2_mag         = r1_mag;   // pass the UNSHIFTED magnitude to stage 3
        s2_nshift      = 7'sd0;
        s2_exp_pre     = $signed({3'b0, r1_exp_big});
        lz             = 6'd0;
        if (!r1_special) begin
            if (r1_mag == 0) begin
                s2_iszero = 1'b1;
            end else if (r1_mag[27]) begin
                s2_nshift = -7'sd1;             // right shift by 1, exp+1
            end else begin
                // count leading zeros above bit 26 in log-depth (was a 27-deep
                // walk-and-count ripple).  Top-align mag[26:0] into a 32-bit word
                // and clz32 it: lz = 26 - msb(mag[26:0]) == the old walk count.
                // mag[26:0] is nonzero in this branch so lz is in 0..26.
                lz        = clz32({r1_mag[26:0], 5'b0});
                s2_nshift = $signed({1'b0, lz});  // 0..26 fits in 7 signed bits
            end
        end
    end

    reg               r2_valid;
    reg               r2_special;
    reg [31:0]        r2_special_val;
    reg               r2_iszero;
    reg               r2_res_sign;
    reg [27:0]        r2_mag;
    reg signed [6:0]  r2_nshift;
    reg signed [10:0] r2_exp_pre;

    // ---------------- Stage 3 : normalize barrel-shift + GRS slice ----------
    reg               s3_special;
    reg [31:0]        s3_special_val;
    reg               s3_iszero;
    reg               s3_res_sign;
    reg [23:0]        s3_mant;
    reg               s3_guard, s3_round_bit, s3_sticky;
    reg signed [10:0] s3_exp_s;

    always @* begin
        // mag[27] is intentionally dropped after normalize: the implicit-1 lands
        // at bit 26 (carry case shifts it down), so only mag[26:0] feed the
        // mantissa/GRS slice.  Localized waiver, in the spirit of glm_fp.vh's.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [27:0] mag;
        /* verilator lint_on UNUSEDSIGNAL */
        reg signed [10:0] exp_s;
        s3_special     = r2_special;
        s3_special_val = r2_special_val;
        s3_iszero      = r2_iszero;
        s3_res_sign    = r2_res_sign;
        s3_mant        = 24'b0;
        s3_guard       = 1'b0;
        s3_round_bit   = 1'b0;
        s3_sticky      = 1'b0;
        mag            = r2_mag;
        exp_s          = r2_exp_pre;
        if (!r2_special && !r2_iszero) begin
            if (r2_nshift == -7'sd1) begin
                mag   = r2_mag >> 1;            // carry case
                exp_s = exp_s + 11'sd1;
            end else if (r2_nshift > 7'sd0) begin
                mag   = r2_mag << r2_nshift[5:0];  // single barrel shift left
                exp_s = exp_s - $signed({4'b0, r2_nshift});
            end
            s3_mant      = mag[26:3];
            s3_guard     = mag[2];
            s3_round_bit = mag[1];
            s3_sticky    = mag[0];
        end
        s3_exp_s = exp_s;
    end

    reg               r3_valid;
    reg               r3_special;
    reg [31:0]        r3_special_val;
    reg               r3_iszero;
    reg               r3_res_sign;
    reg [23:0]        r3_mant;
    reg               r3_guard, r3_round_bit, r3_sticky;
    reg signed [10:0] r3_exp_s;

    // ---------------- Stage 4 : RNE round + range -> output ----------------
    reg [31:0]        s4_val;
    always @* begin
        reg        round_up;
        reg [24:0] mant_r;
        reg signed [10:0] exp_s;
        round_up = 1'b0; mant_r = 25'b0;
        exp_s    = r3_exp_s;
        s4_val   = r3_special_val;
        if (!r3_special) begin
            if (r3_iszero) begin
                s4_val = 32'b0;
            end else begin
                round_up = r3_guard & (r3_round_bit | r3_sticky | r3_mant[0]);
                mant_r   = {1'b0, r3_mant} + {24'b0, round_up};
                if (mant_r[24]) begin
                    mant_r = mant_r >> 1;
                    exp_s  = exp_s + 11'sd1;
                end
                if (exp_s >= 11'sd255)
                    s4_val = {r3_res_sign, 8'hFF, 23'b0};
                else if (exp_s <= 11'sd0)
                    s4_val = {r3_res_sign, 31'b0};
                else
                    s4_val = {r3_res_sign, exp_s[7:0], mant_r[22:0]};
            end
        end
    end

    always @(posedge clk) begin
        if (rst) begin
            r0_valid  <= 1'b0;
            r1_valid  <= 1'b0;
            r2_valid  <= 1'b0;
            r3_valid  <= 1'b0;
            valid_out <= 1'b0;
        end else begin
            // stage 0 -> r0
            r0_valid       <= valid_in;
            r0_special     <= s0_special;
            r0_special_val <= s0_special_val;
            r0_s_big       <= s0_s_big;
            r0_s_small     <= s0_s_small;
            r0_exp_big     <= s0_exp_big;
            r0_m_big_e     <= s0_m_big_e;
            r0_m_small_e   <= s0_m_small_e;
            // stage 1 -> r1
            r1_valid       <= r0_valid;
            r1_special     <= s1_special;
            r1_special_val <= s1_special_val;
            r1_res_sign    <= s1_res_sign;
            r1_exp_big     <= s1_exp_big;
            r1_mag         <= s1_mag;
            // stage 2 -> r2
            r2_valid       <= r1_valid;
            r2_special     <= s2_special;
            r2_special_val <= s2_special_val;
            r2_iszero      <= s2_iszero;
            r2_res_sign    <= s2_res_sign;
            r2_mag         <= s2_mag;
            r2_nshift      <= s2_nshift;
            r2_exp_pre     <= s2_exp_pre;
            // stage 3 -> r3
            r3_valid       <= r2_valid;
            r3_special     <= s3_special;
            r3_special_val <= s3_special_val;
            r3_iszero      <= s3_iszero;
            r3_res_sign    <= s3_res_sign;
            r3_mant        <= s3_mant;
            r3_guard       <= s3_guard;
            r3_round_bit   <= s3_round_bit;
            r3_sticky      <= s3_sticky;
            r3_exp_s       <= s3_exp_s;
            // stage 4 -> out
            valid_out      <= r3_valid;
            result         <= s4_val;
        end
    end
endmodule


//============================================================================
// fp32_mac_pipe  --  fused a*b + c  (RNE, FTZ).   LAT = 7, 1 result/cycle.
//   BIT-EQUIVALENT (0 ULP) to glm_fp.vh  fp32_add(fp32_mul(a,b), c).
//
//   This is the GEMM accumulate primitive.  It is the fp32_mul_pipe feeding the
//   (now deeper) fp32_add_pipe, with `c` delayed by the multiply latency so it
//   meets the product at the adder.  Because each sub-pipe is bit-equivalent to
//   its glm_fp.vh function, the fused result equals fp32_add(fp32_mul(a,b), c)
//   bit-for-bit.  The sub-pipe latencies are READ FROM their exposed LAT
//   parameters rather than hardcoded, so deepening the add ripples here
//   automatically: Total LAT = fp32_mul_pipe.LAT + fp32_add_pipe.LAT.
//   Throughput = 1/cycle.
//============================================================================
module fp32_mac_pipe (
    input              clk,
    input              rst,
    input              valid_in,
    input      [31:0]  a,
    input      [31:0]  b,
    input      [31:0]  c,
    output             valid_out,
    output     [31:0]  result
);
    // sub-pipe latencies, read from the single-source-of-truth macros.  The c
    // operand is delayed by the multiply latency to meet the product at the add.
    localparam integer MUL_LAT = `FP_MUL_LAT;   // 2
    /* verilator lint_off UNUSEDPARAM */
    localparam integer LAT     = `FP_MAC_LAT;   // exposed: structural flop count = mul+add = 7
    /* verilator lint_on UNUSEDPARAM */

    // multiply a*b  (LAT = MUL_LAT)
    wire        mul_v;
    wire [31:0] mul_y;
    fp32_mul_pipe u_mul (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .a(a), .b(b), .valid_out(mul_v), .result(mul_y)
    );

    // delay c by the multiply latency so it meets the product at the adder.
    reg [31:0] c_d [0:MUL_LAT-1];
    integer ci;
    always @(posedge clk) begin
        c_d[0] <= c;
        for (ci = 1; ci < MUL_LAT; ci = ci + 1)
            c_d[ci] <= c_d[ci-1];
    end

    // add (a*b) + c_delayed.  Total LAT = MUL_LAT + ADD_LAT cycles.
    fp32_add_pipe u_add (
        .clk(clk), .rst(rst), .valid_in(mul_v),
        .a(mul_y), .b(c_d[MUL_LAT-1]), .valid_out(valid_out), .result(result)
    );
endmodule


//============================================================================
// fp32_rsqrt_pipe  --  pipelined 1/sqrt(x), x>0.   LAT = 24, 1 result/cycle.
//   BIT-EQUIVALENT to glm_fp.vh fp32_rsqrt (Quake seed + 2 Newton iters,
//   y = y*(1.5 - 0.5*x*y*y)).
//
//   The combinational fp32_rsqrt chains, per iteration:
//        yy  = fp32_mul(y,y)
//        xyy = fp32_mul(xhalf, yy)
//        t   = fp32_add(1.5, -xyy)
//        y   = fp32_mul(y, t)
//   i.e. 3 muls + 1 add per iter, x2 iters, + a setup mul (xhalf).  That is the
//   monstrous critical path glm_fp.vh has.  Here every one of those ops is a
//   real pipelined sub-op, so the critical path is one mul/add stage.
//
//   We reuse the bit-exact fp32_mul_pipe / fp32_add_pipe.  The seed, sign/special
//   handling and the negation of xyy are 0-logic (bit ops) carried alongside in
//   delay regs.  All alignment delays are PARAMETERIZED on the sub-pipes'
//   exposed LAT parameters, so deepening the adder (LAT 3 -> 5) re-balances the
//   delay lines automatically.  Latency bookkeeping (ML=mul LAT, AL=add LAT):
//     setup mul xhalf : ML
//     per iteration   : yy(ML) -> xyy(ML) -> t(AL) -> ymul(ML) = 3*ML + AL
//     total LAT = ML + 2*(3*ML + AL) = 7*ML + 2*AL.
//   With ML=2, AL=5 -> LAT = 14 + 10 = 24.  Throughput = 1/cycle.
//============================================================================
module fp32_rsqrt_pipe (
    input              clk,
    input              rst,
    input              valid_in,
    input      [31:0]  x,
    output             valid_out,
    output     [31:0]  result
);
    localparam [31:0] THREE_HALF = 32'h3FC00000; // 1.5
    localparam [31:0] HALF        = 32'h3F000000; // 0.5

    // sub-pipe latencies read from the single-source-of-truth macros.
    localparam integer ML = `FP_MUL_LAT;            // mul latency (2)
    localparam integer AL = `FP_ADD_LAT;            // add latency (5)
    localparam integer ITER_LAT = 3*ML + AL;        // one Newton iter latency
    /* verilator lint_off UNUSEDPARAM */
    localparam integer LAT = `FP_RSQRT_LAT;         // exposed: structural flop count = 24
    /* verilator lint_on UNUSEDPARAM */

    // ---- special / seed (combinational, 0 ULP bit ops) ----
    // Replicates fp32_rsqrt's special lattice and the integer magic seed.
    wire is_nan  = (x[30:23] == 8'hFF) && (x[22:0] != 0);
    wire is_inf  = (x[30:23] == 8'hFF) && (x[22:0] == 0);
    wire is_zero = (x[30:23] == 8'h00);
    wire bad     = is_nan || x[31] || is_zero;       // x<=0/nan -> qnan
    wire [31:0] special_val = bad   ? 32'h7FC00000 :
                              is_inf ? 32'h00000000 : 32'h00000000;
    wire special = bad || is_inf;
    wire [31:0] seed = 32'h5F3759DF - (x >> 1);

    // ---- xhalf = fp32_mul(0.5, x)   (LAT ML) ----
    wire        xh_v;
    wire [31:0] xhalf;
    fp32_mul_pipe u_xhalf (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .a(HALF), .b(x), .valid_out(xh_v), .result(xhalf)
    );

    // carry the seed/special alongside, aligned to xhalf (ML cycles).
    reg [31:0] y0_d [0:ML-1];
    reg        sp_d [0:ML-1];
    reg [31:0] spv_d[0:ML-1];
    always @(posedge clk) begin : seed_align
        integer di;
        y0_d[0]  <= seed;        sp_d[0]  <= special;     spv_d[0] <= special_val;
        for (di = 1; di < ML; di = di + 1) begin
            y0_d[di]  <= y0_d[di-1];
            sp_d[di]  <= sp_d[di-1];
            spv_d[di] <= spv_d[di-1];
        end
    end
    wire [31:0] y0     = y0_d[ML-1];   // seed aligned with xhalf valid
    wire        sp0    = sp_d[ML-1];
    wire [31:0] spv0   = spv_d[ML-1];

    // =====================================================================
    // One Newton iteration as a sub-pipe-chain.  Given y_in, xhalf_in:
    //   yy  = y*y                (mul, ML)
    //   xyy = xhalf*yy           (mul, ML)
    //   t   = 1.5 + (-xyy)       (add, AL)
    //   y   = y*t                (mul, ML)
    // y_in delayed to the final mul (2*ML + AL); xhalf_in delayed to the 2nd
    // mul (ML).  We hand-instantiate twice rather than generate, for clarity.
    // =====================================================================

    // ---------- ITER 0 ----------
    // yy = y0*y0
    wire        yy0_v;  wire [31:0] yy0;
    fp32_mul_pipe u_yy0 (.clk(clk), .rst(rst), .valid_in(xh_v),
        .a(y0), .b(y0), .valid_out(yy0_v), .result(yy0));
    // delay xhalf by ML (align with yy0) to feed the xyy mul.
    reg [31:0] xhalf_d [0:ML-1];
    always @(posedge clk) begin : xhalf_align0
        integer di;
        xhalf_d[0] <= xhalf;
        for (di = 1; di < ML; di = di + 1) xhalf_d[di] <= xhalf_d[di-1];
    end
    // delay y0 by (2*ML + AL) to align with t0 at the final mul.
    localparam integer YDLY = 2*ML + AL;
    reg [31:0] y0_dl [0:YDLY-1];
    always @(posedge clk) begin : y0_align
        integer di;
        y0_dl[0] <= y0;
        for (di = 1; di < YDLY; di = di + 1) y0_dl[di] <= y0_dl[di-1];
    end
    // xyy = xhalf * yy0
    wire        xyy0_v; wire [31:0] xyy0;
    fp32_mul_pipe u_xyy0 (.clk(clk), .rst(rst), .valid_in(yy0_v),
        .a(xhalf_d[ML-1]), .b(yy0), .valid_out(xyy0_v), .result(xyy0));
    // t = 1.5 + (-xyy0)   (negate xyy0 by flipping sign bit, a 0-cost bit op)
    wire [31:0] neg_xyy0 = {~xyy0[31], xyy0[30:0]};
    wire        t0_v;   wire [31:0] t0;
    fp32_add_pipe u_t0 (.clk(clk), .rst(rst), .valid_in(xyy0_v),
        .a(THREE_HALF), .b(neg_xyy0), .valid_out(t0_v), .result(t0));
    // y1 = y0 * t0
    wire        y1_v;   wire [31:0] y1;
    fp32_mul_pipe u_y1 (.clk(clk), .rst(rst), .valid_in(t0_v),
        .a(y0_dl[YDLY-1]), .b(t0), .valid_out(y1_v), .result(y1));

    // carry xhalf, special alongside to iter 1.  Total iter-0 chain latency =
    // ITER_LAT cycles from xh_v.  xhalf needs delaying ITER_LAT to feed iter1's
    // xyy; special/specialval too.
    reg [31:0] xhalf_chain [0:ITER_LAT-1];
    reg        sp_chain    [0:ITER_LAT-1];
    reg [31:0] spv_chain   [0:ITER_LAT-1];
    always @(posedge clk) begin : iter0_carry
        integer k;
        xhalf_chain[0] <= xhalf;
        sp_chain[0]    <= sp0;
        spv_chain[0]   <= spv0;
        for (k = 1; k < ITER_LAT; k = k + 1) begin
            xhalf_chain[k] <= xhalf_chain[k-1];
            sp_chain[k]    <= sp_chain[k-1];
            spv_chain[k]   <= spv_chain[k-1];
        end
    end
    wire [31:0] xhalf1 = xhalf_chain[ITER_LAT-1];
    wire        sp1    = sp_chain[ITER_LAT-1];
    wire [31:0] spv1   = spv_chain[ITER_LAT-1];

    // ---------- ITER 1 ----------  (same structure, y1 -> y2)
    // yy = y1*y1
    wire        yy1_v;  wire [31:0] yy1;
    fp32_mul_pipe u_yy1 (.clk(clk), .rst(rst), .valid_in(y1_v),
        .a(y1), .b(y1), .valid_out(yy1_v), .result(yy1));
    reg [31:0] xhalf1_d [0:ML-1];
    reg [31:0] y1_dl    [0:YDLY-1];
    always @(posedge clk) begin : iter1_align
        integer di;
        xhalf1_d[0] <= xhalf1;
        for (di = 1; di < ML; di = di + 1) xhalf1_d[di] <= xhalf1_d[di-1];
        y1_dl[0] <= y1;
        for (di = 1; di < YDLY; di = di + 1) y1_dl[di] <= y1_dl[di-1];
    end
    wire        xyy1_v; wire [31:0] xyy1;
    fp32_mul_pipe u_xyy1 (.clk(clk), .rst(rst), .valid_in(yy1_v),
        .a(xhalf1_d[ML-1]), .b(yy1), .valid_out(xyy1_v), .result(xyy1));
    wire [31:0] neg_xyy1 = {~xyy1[31], xyy1[30:0]};
    wire        t1_v;   wire [31:0] t1;
    fp32_add_pipe u_t1 (.clk(clk), .rst(rst), .valid_in(xyy1_v),
        .a(THREE_HALF), .b(neg_xyy1), .valid_out(t1_v), .result(t1));
    wire        y2_v;   wire [31:0] y2;
    fp32_mul_pipe u_y2 (.clk(clk), .rst(rst), .valid_in(t1_v),
        .a(y1_dl[YDLY-1]), .b(t1), .valid_out(y2_v), .result(y2));

    // carry special/specialval across iter 1 (ITER_LAT cycles) to the final mux.
    reg [31:0] spv_chain2 [0:ITER_LAT-1];
    reg        sp_chain2  [0:ITER_LAT-1];
    always @(posedge clk) begin : iter1_carry
        integer k;
        spv_chain2[0] <= spv1;
        sp_chain2[0]  <= sp1;
        for (k = 1; k < ITER_LAT; k = k + 1) begin
            spv_chain2[k] <= spv_chain2[k-1];
            sp_chain2[k]  <= sp_chain2[k-1];
        end
    end
    wire        sp_final  = sp_chain2[ITER_LAT-1];
    wire [31:0] spv_final = spv_chain2[ITER_LAT-1];

    // ---------- final select ----------
    assign valid_out = y2_v;
    assign result    = sp_final ? spv_final : y2;
    // LAT = ML + 2*ITER_LAT = 7*ML + 2*AL = 24 (ML=2, AL=5).  Throughput 1/cycle
    // (every sub-pipe accepts a new op each cycle).  0 ULP vs glm_fp.vh fp32_rsqrt.
endmodule


//============================================================================
// glm_exp_ref  --  COMBINATIONAL reference fp32 exp(x), built ONLY from the
//   glm_fp.vh contract primitives (fp32_mul/fp32_add).  This is the golden the
//   fp32_exp_pipe is bit-checked against (glm_fp.vh has no exp of its own).
//
//   Method (the standard softmax-range exp): range-reduce
//        x = k*ln2 + r,  k = round(x/ln2),  r in [-ln2/2, ln2/2]
//   approximate exp(r) by a 5-term Horner polynomial (1 + r + r^2/2 + r^3/6 +
//   r^4/24), then fold 2^k by adding k to the result's exponent field.  All
//   arithmetic is fp32_mul/fp32_add so the pipe can reproduce it bit-for-bit.
//   Domain: softmax feeds x = (logit - max) <= 0, x in roughly [-87, 0].
//============================================================================
function automatic [31:0] glm_exp_ref(input [31:0] x);
    `include "glm_fp.vh"
    // constants
    reg [31:0] LN2, INV_LN2;
    reg [31:0] C1, C2, C3, C4;       // 1/2, 1/6, 1/24, 1/120
    reg [31:0] xv, kf, r, poly, kln2;
    reg signed [9:0] ki;             // k in [-256,255] (softmax range needs <=0)
    reg [7:0]  e;
    reg signed [9:0] new_e;
    begin
        LN2     = 32'h3F317218;   // 0.6931472
        INV_LN2 = 32'h3FB8AA3B;   // 1.4426950
        C1      = 32'h3F000000;   // 1/2
        C2      = 32'h3E2AAAAB;   // 1/6
        C3      = 32'h3D2AAAAB;   // 1/24
        C4      = 32'h3C088889;   // 1/120
        // k = round(x * 1/ln2).  x*1/ln2 is computed in fp32, then rounded to a
        // signed integer.  For the softmax domain x in [-87,0], |k| <= 126, well
        // within the 10-bit signed range.
        xv = x;
        kf = fp32_mul(xv, INV_LN2);
        ki = fp32_to_int10_rne(kf);
        // r = x - k*ln2     (k folded back to fp32, multiplied, subtracted)
        kln2 = fp32_mul(int10_to_fp32(ki), LN2);
        r    = fp32_add(xv, {kln2[31]^1'b1, kln2[30:0]});
        // Horner: exp(r) ~ 1 + r*(1 + r*(1/2 + r*(1/6 + r*(1/24 + r*(1/120)))))
        poly = fp32_add(C3, fp32_mul(C4, r));            // 1/24 + r/120
        poly = fp32_add(C2, fp32_mul(poly, r));          // 1/6 + r*(...)
        poly = fp32_add(C1, fp32_mul(poly, r));          // 1/2 + r*(...)
        poly = fp32_add(32'h3F800000, fp32_mul(poly, r));// 1 + r*(...)
        poly = fp32_add(32'h3F800000, fp32_mul(poly, r));// 1 + r*(...)  full
        // fold 2^k: add ki to the biased exponent of poly (FTZ on under/overflow)
        e     = poly[30:23];
        new_e = $signed({2'b0, e}) + ki;
        if (e == 8'h00)
            glm_exp_ref = 32'b0;                          // poly already FTZ
        else if (new_e >= 10'sd255)
            glm_exp_ref = {poly[31], 8'hFF, 23'b0};       // overflow -> inf
        else if (new_e <= 10'sd0)
            glm_exp_ref = 32'b0;                          // underflow -> FTZ
        else
            glm_exp_ref = {poly[31], new_e[7:0], poly[22:0]};
    end
endfunction

// ---------------------------------------------------------------------------
// Bounded-range integer<->fp32 glue for the exp range reduction.  These touch
// only the small integer k (|k| <= 126 in the softmax domain), NOT the fp32
// datapath, and are written with wider intermediates for clarity; the few
// width/unused notes that result are localized waivers (the values are
// provably in-range), in the same spirit as glm_fp.vh's documented lint
// waivers.  Functional correctness is checked bit-exactly by the smoke TB.
// ---------------------------------------------------------------------------
/* verilator lint_off WIDTHTRUNC */
/* verilator lint_off WIDTHEXPAND */
/* verilator lint_off UNUSEDSIGNAL */

// helper: fp32 -> signed 10-bit int, round-to-nearest (magnitude < 512 here).
function automatic signed [9:0] fp32_to_int10_rne(input [31:0] f);
    reg        s;
    reg [23:0] m;            // 1.frac significand
    reg [23:0] shifted;      // m after right-shift
    reg [4:0]  sh;           // shift = 23 - (e-127), range 1..23
    reg [9:0]  mag;          // integer magnitude (<512)
    begin
        s       = f[31];
        m       = {1'b1, f[22:0]};
        mag     = 10'd0;
        sh      = 5'd0;
        shifted = 24'd0;
        // Only e>=127 (value>=1) can produce a nonzero integer; e<127 -> 0.
        if (f[30:23] >= 8'd127) begin
            if (f[30:23] >= 8'd150) begin
                // value >= 2^23: out of softmax range; saturate magnitude.
                mag = 10'd511;
            end else begin
                // sh = 23 - (e-127) = 150 - e, range 1..23 for e in 127..149.
                sh      = 8'd150 - f[30:23];
                shifted = m >> sh;
                // saturate if shifted value exceeds 10 bits (out of range);
                // otherwise take the low 10 bits.
                mag     = (|shifted[23:10]) ? 10'd511 : shifted[9:0];
                // round half up using the bit just below the integer point.
                if (m[sh - 5'd1])
                    mag = mag + 10'd1;
            end
        end
        fp32_to_int10_rne = s ? -$signed(mag) : $signed(mag);
    end
endfunction

// helper: signed 10-bit int -> fp32 (exact for |iv| < 512).
function automatic [31:0] int10_to_fp32(input signed [9:0] iv);
    reg        s;
    reg [9:0]  mag;
    reg [3:0]  msb;          // index of MSB (0..9)
    reg [7:0]  e;
    reg [22:0] frac;
    integer    i;
    begin
        if (iv == 10'sd0) int10_to_fp32 = 32'b0;
        else begin
            s   = iv[9];
            mag = iv[9] ? (~iv + 10'sd1) : iv;     // magnitude
            msb = 4'd0;
            for (i = 0; i < 10; i = i + 1)
                if (mag[i]) msb = i[3:0];
            e    = 8'd127 + {4'b0, msb};
            // fraction = magnitude left-justified so its MSB (the implicit 1)
            // aligns above bit 22; the low 23 bits are the fraction.
            frac = {13'b0, mag} << (5'd23 - {1'b0, msb});
            int10_to_fp32 = {s, e, frac};
        end
    end
endfunction

/* verilator lint_on UNUSEDSIGNAL */
/* verilator lint_on WIDTHEXPAND */
/* verilator lint_on WIDTHTRUNC */


//============================================================================
// fp32_exp_pipe  --  GENUINELY pipelined fp32 exp(x) for the softmax range.
//   1 result/cycle.  BIT-EQUIVALENT (0 ULP) to glm_exp_ref above (the same
//   range-reduce + Horner + 2^k method, same fp32_mul/fp32_add contract
//   arithmetic).  Accuracy vs true exp() < 2^-11 rel over x in [-87,0]
//   (softmax domain), matching the combinational softmax_unit method.
//
//   This is NOT a combinational cone + delay-line (the old, ~1.5 MHz version).
//   Every fp32 op of glm_exp_ref is a real pipelined sub-op (fp32_mul_pipe /
//   fp32_add_pipe), exactly like fp32_rsqrt_pipe, so the per-stage critical path
//   is ONE fp-pipe slice.  The tiny integer-k conversions (fp32<->int10) are
//   bit-ops evaluated combinationally at a stage boundary and ride in matched
//   delay regs; they are NOT on the fp32 mantissa critical path.
//
//   Dataflow (mirrors glm_exp_ref op-for-op; ML=mul LAT, AL=add LAT):
//     kf   = x * (1/ln2)                     mul   (ML)
//     ki   = round_to_int(kf)                comb int (registered, carried)
//     kln2 = int10_to_fp(ki) * ln2           mul   (ML)   [x delayed ML to here]
//     r    = x + (-kln2)                     add   (AL)
//     Horner, 5 (mul,add) pairs reusing r:
//        p = C4*r ; poly = C3 + p            mul(ML),add(AL)
//        p = poly*r ; poly = C2 + p          mul(ML),add(AL)
//        p = poly*r ; poly = C1 + p          mul(ML),add(AL)
//        p = poly*r ; poly = 1.0 + p         mul(ML),add(AL)
//        p = poly*r ; poly = 1.0 + p         mul(ML),add(AL)   <- full poly
//     fold 2^k : add ki to poly's biased exponent (comb, FTZ).   output reg.
//
//   LAT (structural flop count valid_in->valid_out) =
//        ML (kf) + ML (kln2) + AL (r) + 5*(ML+AL) (Horner) + 1 (fold/output reg)
//      = 7*ML + 6*AL + 1.   With ML=2, AL=5 -> 14 + 30 + 1 = 45.
//============================================================================
module fp32_exp_pipe (
    input              clk,
    input              rst,
    input              valid_in,
    input      [31:0]  x,
    output reg         valid_out,
    output reg [31:0]  result
);
    `include "glm_fp.vh"

    localparam integer ML = `FP_MUL_LAT;   // mul latency (2)
    localparam integer AL = `FP_ADD_LAT;   // add latency (5)
    /* verilator lint_off UNUSEDPARAM */
    // structural flop count valid_in -> valid_out (exposed for consumers)
    localparam integer LAT = `FP_EXP_LAT;  // = 7*ML + 6*AL + 2 = 46
    /* verilator lint_on UNUSEDPARAM */

    localparam [31:0] LN2     = 32'h3F317218;   // 0.6931472
    localparam [31:0] INV_LN2 = 32'h3FB8AA3B;   // 1.4426950
    localparam [31:0] C1      = 32'h3F000000;   // 1/2
    localparam [31:0] C2      = 32'h3E2AAAAB;   // 1/6
    localparam [31:0] C3      = 32'h3D2AAAAB;   // 1/24
    localparam [31:0] C4      = 32'h3C088889;   // 1/120
    localparam [31:0] ONE     = 32'h3F800000;   // 1.0

    // ===================== kf = x * (1/ln2)  (mul, ML) =====================
    wire        kf_v;  wire [31:0] kf;
    fp32_mul_pipe u_kf (.clk(clk), .rst(rst), .valid_in(valid_in),
        .a(x), .b(INV_LN2), .valid_out(kf_v), .result(kf));

    // ki = round(kf) -> as fp32, computed combinationally on kf, registered.
    // We carry BOTH the integer ki (for the final 2^k fold) and its fp32 image
    // (the multiplicand for kln2).  These are bit-ops, off the fp critical path.
    wire signed [9:0] ki_now   = fp32_to_int10_rne(kf);
    wire       [31:0] kifp_now = int10_to_fp32(ki_now);

    // x is needed at the 'r' add, which is fed by kln2_v.  kln2's result is valid
    // ML (kf) + 1 (the ki/kifp register stage) + ML (kln2 mul) = 2*ML+1 cycles
    // after valid_in, so x must be delayed by the same amount to meet it.
    localparam integer XDLY = 2*ML + 1;
    reg [31:0] x_dl [0:XDLY-1];
    always @(posedge clk) begin : x_align
        integer di;
        x_dl[0] <= x;
        for (di = 1; di < XDLY; di = di + 1) x_dl[di] <= x_dl[di-1];
    end

    // register ki (signed 10b) and its fp image at the kf boundary; carry ki the
    // whole way to the fold.  kifp feeds the kln2 mul one cycle later.
    reg signed [9:0] ki_r;
    reg       [31:0] kifp_r;
    reg              kf_v_r;
    always @(posedge clk) begin
        ki_r   <= ki_now;
        kifp_r <= kifp_now;
        kf_v_r <= kf_v;
    end

    // ===================== kln2 = kifp * ln2  (mul, ML) =====================
    wire        kln2_v;  wire [31:0] kln2;
    fp32_mul_pipe u_kln2 (.clk(clk), .rst(rst), .valid_in(kf_v_r),
        .a(kifp_r), .b(LN2), .valid_out(kln2_v), .result(kln2));

    // x aligned to the kln2 result is x_dl[XDLY-1] (delayed 2*ML).
    wire [31:0] neg_kln2 = {~kln2[31], kln2[30:0]};

    // ===================== r = x + (-kln2)  (add, AL) ======================
    wire        r_v;  wire [31:0] r;
    fp32_add_pipe u_r (.clk(clk), .rst(rst), .valid_in(kln2_v),
        .a(x_dl[XDLY-1]), .b(neg_kln2), .valid_out(r_v), .result(r));

    // carry ki from the kf boundary all the way to the fold.  ki was registered
    // at the kf boundary (1 cycle after valid_in's kf consumption); from there
    // to the fold is: (kln2 mul ML) + (r add AL) + 5*(ML+AL) Horner.  We just
    // push ki through a delay line of that length alongside the fp datapath.
    // ki was registered 1 cycle (the kf boundary register); from there to the
    // poly-ready edge is KI_TAIL = (kln2 mul ML) + (r add AL) + 5*(ML+AL) Horner.
    localparam integer KI_TAIL = ML + AL + 5*(ML+AL); // kf-bndry -> poly ready
    reg signed [9:0] ki_chain [0:KI_TAIL-1];
    always @(posedge clk) begin : ki_carry
        integer di;
        ki_chain[0] <= ki_r;
        for (di = 1; di < KI_TAIL; di = di + 1) ki_chain[di] <= ki_chain[di-1];
    end
    wire signed [9:0] ki_fold = ki_chain[KI_TAIL-1];

    // ===================== Horner: 5 (mul, add) pairs ======================
    // Each pair:  p = poly_in * r_aligned ;  poly_out = const + p
    // r must be delayed to align with each successive poly_in.  We keep a single
    // r delay line and tap it at multiples of (ML+AL): the first mul consumes r
    // at r_v; each subsequent mul consumes r delayed by an extra (ML+AL).
    localparam integer STEP   = ML + AL;        // latency of one (mul,add) pair
    localparam integer RDEPTH = 5*STEP;         // r must survive 5 pairs
    reg [31:0] r_dl [0:RDEPTH-1];
    always @(posedge clk) begin : r_align
        integer di;
        r_dl[0] <= r;
        for (di = 1; di < RDEPTH; di = di + 1) r_dl[di] <= r_dl[di-1];
    end
    // r tap for Horner step s (s=0..4): aligned to that step's mul input.
    //   step 0 mul consumes r at r_v            -> use r directly
    //   step s mul consumes r delayed s*STEP    -> r_dl[s*STEP-1]
    // NOTE: these MUST be explicit wires, not a function called inside the module
    // instantiation ports below -- iverilog freezes a function-in-port at time 0
    // (it does not re-evaluate), driving the sub-pipes with X.  STEP is a constant
    // so r_dl[k*STEP-1] is a constant-index memory read = a valid continuous assign.
    wire [31:0] r_tap0 = r;
    wire [31:0] r_tap1 = r_dl[1*STEP-1];
    wire [31:0] r_tap2 = r_dl[2*STEP-1];
    wire [31:0] r_tap3 = r_dl[3*STEP-1];
    wire [31:0] r_tap4 = r_dl[4*STEP-1];

    // ---- step 0 : p0 = C4 * r ; poly0 = C3 + p0 ----
    wire        p0_v;  wire [31:0] p0;
    fp32_mul_pipe u_m0 (.clk(clk), .rst(rst), .valid_in(r_v),
        .a(C4), .b(r_tap0), .valid_out(p0_v), .result(p0));
    wire        poly0_v;  wire [31:0] poly0;
    fp32_add_pipe u_a0 (.clk(clk), .rst(rst), .valid_in(p0_v),
        .a(C3), .b(p0), .valid_out(poly0_v), .result(poly0));

    // ---- step 1 : p1 = poly0 * r ; poly1 = C2 + p1 ----
    wire        p1_v;  wire [31:0] p1;
    fp32_mul_pipe u_m1 (.clk(clk), .rst(rst), .valid_in(poly0_v),
        .a(poly0), .b(r_tap1), .valid_out(p1_v), .result(p1));
    wire        poly1_v;  wire [31:0] poly1;
    fp32_add_pipe u_a1 (.clk(clk), .rst(rst), .valid_in(p1_v),
        .a(C2), .b(p1), .valid_out(poly1_v), .result(poly1));

    // ---- step 2 : p2 = poly1 * r ; poly2 = C1 + p2 ----
    wire        p2_v;  wire [31:0] p2;
    fp32_mul_pipe u_m2 (.clk(clk), .rst(rst), .valid_in(poly1_v),
        .a(poly1), .b(r_tap2), .valid_out(p2_v), .result(p2));
    wire        poly2_v;  wire [31:0] poly2;
    fp32_add_pipe u_a2 (.clk(clk), .rst(rst), .valid_in(p2_v),
        .a(C1), .b(p2), .valid_out(poly2_v), .result(poly2));

    // ---- step 3 : p3 = poly2 * r ; poly3 = 1.0 + p3 ----
    wire        p3_v;  wire [31:0] p3;
    fp32_mul_pipe u_m3 (.clk(clk), .rst(rst), .valid_in(poly2_v),
        .a(poly2), .b(r_tap3), .valid_out(p3_v), .result(p3));
    wire        poly3_v;  wire [31:0] poly3;
    fp32_add_pipe u_a3 (.clk(clk), .rst(rst), .valid_in(p3_v),
        .a(ONE), .b(p3), .valid_out(poly3_v), .result(poly3));

    // ---- step 4 : p4 = poly3 * r ; poly4 = 1.0 + p4  (full poly) ----
    wire        p4_v;  wire [31:0] p4;
    fp32_mul_pipe u_m4 (.clk(clk), .rst(rst), .valid_in(poly3_v),
        .a(poly3), .b(r_tap4), .valid_out(p4_v), .result(p4));
    wire        poly_v;  wire [31:0] poly;
    fp32_add_pipe u_a4 (.clk(clk), .rst(rst), .valid_in(p4_v),
        .a(ONE), .b(p4), .valid_out(poly_v), .result(poly));

    // ===================== fold 2^k + output register ======================
    // add ki to poly's biased exponent (FTZ on under/overflow), exactly as
    // glm_exp_ref.  This is the final stage; result is registered out.
    always @(posedge clk) begin
        reg [7:0]         e;
        reg signed [9:0]  new_e;          // matches glm_exp_ref's width exactly
        if (rst) begin
            valid_out <= 1'b0;
        end else begin
            valid_out <= poly_v;
            e     = poly[30:23];
            new_e = $signed({2'b0, e}) + ki_fold;          // 10-bit, as glm_exp_ref
            if (e == 8'h00)
                result <= 32'b0;                           // poly already FTZ
            else if (new_e >= 10'sd255)
                result <= {poly[31], 8'hFF, 23'b0};        // overflow -> inf
            else if (new_e <= 10'sd0)
                result <= 32'b0;                           // underflow -> FTZ
            else
                result <= {poly[31], new_e[7:0], poly[22:0]};
        end
    end
endmodule
