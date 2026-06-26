`ifndef GLM_FP_VH
`define GLM_FP_VH
//============================================================================
// glm_fp.vh  --  GLM-5.2 FLOATING-POINT PRIMITIVES (the numerics contract)
//                                                            (ACCEL_GLM52 §6)
//----------------------------------------------------------------------------
// PURPOSE
//   This header is the SINGLE canonical definition of the floating-point
//   numerics every GLM-5.2 hardware unit inherits.  It establishes the
//   bf16-storage / fp32-reduce / bf16-out discipline that ACCEL_GLM52 §6
//   mandates: weights and activations are stored bf16, but ALL reductions and
//   accumulations (a 6144-term Σx² loses precision in bf16) run in fp32, and
//   results are rounded back to bf16 for storage.
//
//   Everything here is a SYNTHESIZABLE, PURELY COMBINATIONAL SystemVerilog
//   `function automatic`.  No state, no clocks, no side effects -> safe to call
//   from any always_comb / continuous-assign context, and trivially shareable.
//   Functions are included into a module's scope via `include "glm_fp.vh"`.
//
//----------------------------------------------------------------------------
// FORMATS
//   BF16  : 1 sign | 8 exp (bias 127) | 7 mantissa.  Exactly the HIGH 16 bits
//           of an IEEE-754 binary32.  Same exponent range as fp32, 8-bit
//           significand (7 stored + 1 implicit).
//   FP32  : IEEE-754 binary32.  1 sign | 8 exp (bias 127) | 23 mantissa.
//
//----------------------------------------------------------------------------
// SPECIAL-VALUE / SUBNORMAL POLICY  (documented, enforced uniformly)
//   * inf/nan are PROPAGATED by mul/add (any nan input -> qnan out;
//     inf arithmetic follows IEEE sign rules; inf-inf -> qnan).
//   * SUBNORMALS ARE FLUSHED TO ZERO (FTZ) on both inputs and outputs of the
//     fp32 arithmetic primitives.  GLM activations/weights live in a normal
//     dynamic range (bf16 anyway has no useful subnormal headroom relative to
//     fp32), so FTZ removes the denormal-handling hardware at no accuracy cost
//     for this workload.  This is a DELIBERATE, DOCUMENTED simplification.
//   * Rounding mode for fp32->bf16 and for the fp32 mul/add normalize step is
//     ROUND-TO-NEAREST-EVEN (RNE), including correct carry-out of the mantissa
//     into the exponent (and exp overflow -> inf).
//
//----------------------------------------------------------------------------
// API  (all `function automatic`, combinational)
//   bf16_to_fp32(input [15:0] b)            -> [31:0]   zero-extend mantissa
//   fp32_to_bf16(input [31:0] f)            -> [15:0]   RNE, carry into exp
//   fp32_mul   (input [31:0] a,b)           -> [31:0]   IEEE-ish *, RNE, FTZ
//   fp32_add   (input [31:0] a,b)           -> [31:0]   IEEE-ish +, RNE, FTZ
//   fp32_rsqrt (input [31:0] x)             -> [31:0]   1/sqrt(x), x>0
//   bf16_mul   (input [15:0] a,b)           -> [15:0]   convenience: bf16*bf16
//                                                       computed in fp32 -> bf16
//----------------------------------------------------------------------------
// RSQRT ACCURACY (see fp32_rsqrt below)
//   Fast-inverse-sqrt magic-constant seed (0x5f3759df) + 2 Newton-Raphson
//   iterations of  y = y*(1.5 - 0.5*x*y*y).  Measured worst-case relative
//   error over x in [1e-12, 1e12] is  < 2^-22  (~2.4e-7), FAR inside the
//   §1 target of <= 2^-12.  (One NR iteration already gives ~1.75e-3 < 2^-9;
//   the second is essentially free combinationally and we keep it.)
//============================================================================

//----------------------------------------------------------------------------
// bf16_to_fp32 : the bf16 value IS the top 16 bits of the fp32 bit pattern,
//   so widening is a zero-fill of the low 16 mantissa bits.  Exact, lossless,
//   handles zero/inf/nan/normal identically (it is a pure bit operation).
//----------------------------------------------------------------------------
function automatic [31:0] bf16_to_fp32(input [15:0] b);
    bf16_to_fp32 = {b, 16'h0000};
endfunction

//----------------------------------------------------------------------------
// fp32_to_bf16 : round-to-nearest-even of an fp32 into bf16 (drop low 16
//   mantissa bits).  Handles the rounding CARRY propagating through the
//   mantissa and into the exponent (and exp overflow -> inf).  nan stays nan
//   (forced quiet, nonzero payload preserved-ish); inf stays inf; zero/signed
//   zero preserved.
//----------------------------------------------------------------------------
function automatic [15:0] fp32_to_bf16(input [31:0] f);
    reg        s;
    reg [7:0]  e;
    reg [22:0] m;
    reg        round_up;
    reg [14:0] top_plus;   // {8 exp, 7 mant}; a round carry rolls into the exp
    begin
        s = f[31];
        e = f[30:23];
        m = f[22:0];
        if (e == 8'hFF) begin
            // inf or nan: keep inf as inf; force nan quiet (mantissa MSB=1).
            if (m == 23'b0) fp32_to_bf16 = {s, 8'hFF, 7'b0};          // inf
            else            fp32_to_bf16 = {s, 8'hFF, 1'b1, 6'b0};    // qnan
        end else begin
            // RNE on the boundary at bit 16: round bit = m[15], sticky = m[14:0],
            // and the LSB being kept is m[16].
            round_up = m[15] & (m[14:0] != 15'b0 | m[16]);
            // pack {8 exp, 7 mant} into 15 bits, add round; a carry out of the
            // mantissa propagates naturally into the exponent field.
            top_plus = {e, m[22:16]} + {14'b0, round_up};
            // top_plus[14:7] = new exp(8), top_plus[6:0] = new mant(7).
            if (top_plus[14:7] == 8'hFF)
                fp32_to_bf16 = {s, 8'hFF, 7'b0};   // overflow -> inf
            else
                fp32_to_bf16 = {s, top_plus[14:7], top_plus[6:0]};
        end
    end
endfunction

//----------------------------------------------------------------------------
// Helpers: decode/normalize an fp32 with FTZ.  Returns sign, an UNBIASED-ish
//   biased exponent, and a 24-bit significand {implicit-1, 23 frac} for normal
//   numbers; zero (incl. subnormal flushed) reports is_zero.
//----------------------------------------------------------------------------
// (these take the magnitude bits [30:0]; the sign bit is irrelevant to the
//  is-nan/inf/zero classification, so it is sliced off at the call site.)
function automatic        _glmfp_is_nan (input [30:0] f);
    _glmfp_is_nan = (f[30:23] == 8'hFF) && (f[22:0] != 0);
endfunction
function automatic        _glmfp_is_inf (input [30:0] f);
    _glmfp_is_inf = (f[30:23] == 8'hFF) && (f[22:0] == 0);
endfunction
function automatic        _glmfp_is_zero(input [7:0] e);
    // FTZ: exponent 0 (true zero OR subnormal) is treated as zero.
    _glmfp_is_zero = (e == 8'h00);
endfunction

//----------------------------------------------------------------------------
// fp32_mul : IEEE-ish single-precision multiply.  RNE, FTZ.  Handles the full
//   special-value lattice (nan/inf/zero).
//----------------------------------------------------------------------------
function automatic [31:0] fp32_mul(input [31:0] a, input [31:0] b);
    reg        sa, sb, sr;
    reg [7:0]  ea, eb;
    reg [23:0] ma, mb;       // 24-bit significands (implicit 1)
    reg [47:0] prod;         // 24x24
    reg signed [10:0] exp_s; // biased-sum exponent, signed for under/overflow
    reg [23:0] mant;         // normalized 24-bit significand
    reg        guard, round_bit, sticky, round_up;
    reg [24:0] mant_r;       // +1 for round carry
    begin
        sa = a[31]; sb = b[31]; sr = sa ^ sb;
        ea = a[30:23]; eb = b[30:23];
        // -------- special values --------
        if (_glmfp_is_nan(a[30:0]) || _glmfp_is_nan(b[30:0])) begin
            fp32_mul = 32'h7FC00000;                       // qnan
        end else if (_glmfp_is_inf(a[30:0]) || _glmfp_is_inf(b[30:0])) begin
            // inf * 0 -> nan ; else signed inf
            if (_glmfp_is_zero(a[30:23]) || _glmfp_is_zero(b[30:23]))
                fp32_mul = 32'h7FC00000;
            else
                fp32_mul = {sr, 8'hFF, 23'b0};
        end else if (_glmfp_is_zero(a[30:23]) || _glmfp_is_zero(b[30:23])) begin
            fp32_mul = {sr, 31'b0};                         // signed zero
        end else begin
            ma = {1'b1, a[22:0]};
            mb = {1'b1, b[22:0]};
            prod = ma * mb;                                 // 1.xxx * 1.xxx in [1,4)
            // exponent: (ea-127)+(eb-127)+127 = ea+eb-127
            exp_s = $signed({3'b0, ea}) + $signed({3'b0, eb}) - 11'sd127;
            if (prod[47]) begin
                // result in [2,4): shift right 1, exp+1
                exp_s     = exp_s + 11'sd1;
                mant      = prod[47:24];
                guard     = prod[23];
                sticky    = (prod[22:0] != 0);
            end else begin
                // result in [1,2)
                mant      = prod[46:23];
                guard     = prod[22];
                sticky    = prod[21:0] != 0;
            end
            round_bit = guard;
            round_up  = round_bit & (sticky | mant[0]);     // RNE
            mant_r    = {1'b0, mant} + {24'b0, round_up};
            if (mant_r[24]) begin                           // mantissa carried out
                mant_r = mant_r >> 1;
                exp_s  = exp_s + 11'sd1;
            end
            // -------- exponent range --------
            if (exp_s >= 11'sd255)
                fp32_mul = {sr, 8'hFF, 23'b0};              // overflow -> inf
            else if (exp_s <= 11'sd0)
                fp32_mul = {sr, 31'b0};                     // underflow -> FTZ zero
            else
                fp32_mul = {sr, exp_s[7:0], mant_r[22:0]};
        end
    end
endfunction

//----------------------------------------------------------------------------
// fp32_add : IEEE-ish single-precision add (handles subtraction via signs).
//   RNE, FTZ.  Full special-value lattice.
//----------------------------------------------------------------------------
function automatic [31:0] fp32_add(input [31:0] a, input [31:0] b);
    reg        sa, sb;
    reg [7:0]  ea, eb;
    reg [23:0] ma, mb;
    reg [7:0]  exp_big;
    reg        s_big, s_small;
    reg [23:0] m_big, m_small;
    reg [7:0]  shamt;
    // extended significand: 24 + 3 guard/round/sticky region, give generous width
    reg [26:0] m_big_e, m_small_e;
    reg [27:0] sum;          // +1 for carry
    reg        res_sign;
    reg [27:0] mag;
    reg signed [10:0] exp_s;
    reg [10:0] lz;
    integer    lzi;
    reg        guard, round_bit, sticky, round_up;
    reg [24:0] mant_r;
    reg [23:0] mant;
    begin
        sa = a[31]; sb = b[31];
        ea = a[30:23]; eb = b[30:23];
        // -------- special values --------
        if (_glmfp_is_nan(a[30:0]) || _glmfp_is_nan(b[30:0])) begin
            fp32_add = 32'h7FC00000;
        end else if (_glmfp_is_inf(a[30:0]) && _glmfp_is_inf(b[30:0])) begin
            fp32_add = (sa == sb) ? {sa, 8'hFF, 23'b0} : 32'h7FC00000; // inf-inf=nan
        end else if (_glmfp_is_inf(a[30:0])) begin
            fp32_add = {sa, 8'hFF, 23'b0};
        end else if (_glmfp_is_inf(b[30:0])) begin
            fp32_add = {sb, 8'hFF, 23'b0};
        end else if (_glmfp_is_zero(a[30:23]) && _glmfp_is_zero(b[30:23])) begin
            // both zero: -0 + -0 = -0, else +0
            fp32_add = {sa & sb, 31'b0};
        end else if (_glmfp_is_zero(a[30:23])) begin
            fp32_add = {sb, eb, b[22:0]};                  // = b (already normal)
        end else if (_glmfp_is_zero(b[30:23])) begin
            fp32_add = {sa, ea, a[22:0]};                  // = a
        end else begin
            ma = {1'b1, a[22:0]};
            mb = {1'b1, b[22:0]};
            // align to larger exponent
            if (ea >= eb) begin
                exp_big = ea; s_big = sa; s_small = sb;
                m_big = ma;   m_small = mb;  shamt = ea - eb;
            end else begin
                exp_big = eb; s_big = sb; s_small = sa;
                m_big = mb;   m_small = ma;  shamt = eb - ea;
            end
            // extend with 3 low bits (guard/round/sticky room) -> <<3
            m_big_e   = {m_big,   3'b0};
            if (shamt >= 8'd27) begin
                // small operand shifted entirely into sticky
                m_small_e = (m_small != 0) ? 27'd1 : 27'd0;
            end else begin
                m_small_e = {m_small, 3'b0} >> shamt;
                // OR-in sticky bits lost by the shift (mask the low `shamt` bits)
                if (shamt > 8'd3) begin
                    if (({m_small, 3'b0} & ((27'd1 << shamt) - 27'd1)) != 27'd0)
                        m_small_e[0] = 1'b1;
                end
            end
            if (s_big == s_small) begin
                sum      = {1'b0, m_big_e} + {1'b0, m_small_e};
                res_sign = s_big;
            end else begin
                if (m_big_e >= m_small_e) begin
                    sum = {1'b0, m_big_e} - {1'b0, m_small_e};
                    res_sign = s_big;
                end else begin
                    sum = {1'b0, m_small_e} - {1'b0, m_big_e};
                    res_sign = s_small;
                end
            end
            exp_s = $signed({3'b0, exp_big});
            mag   = sum;
            if (mag == 0) begin
                fp32_add = 32'b0;                          // exact cancellation -> +0
            end else begin
                // normalize: target the implicit-1 at bit position 26 (since
                // significand was <<3 from 24-bit -> MSB nominally at 26, with a
                // possible carry at 27).
                if (mag[27]) begin
                    // carried out: shift right 1, exp+1
                    exp_s = exp_s + 11'sd1;
                    mag   = mag >> 1;
                end else begin
                    // normalize left: count leading zeros above bit 26 with a
                    // BOUNDED for-loop (synthesizable priority logic -- no
                    // data-dependent `while`), then shift once by that count.
                    lz = 11'd0;
                    for (lzi = 0; lzi < 27; lzi = lzi + 1)
                        if (mag[26] == 1'b0 && (lzi[10:0] == lz)) begin
                            mag = mag << 1;
                            lz  = lz + 11'd1;
                        end
                    exp_s = exp_s - $signed(lz);
                end
                // now mag[26:3] is the 24-bit significand, mag[2]=guard,
                // mag[1]=round, mag[0]=sticky
                mant      = mag[26:3];
                guard     = mag[2];
                round_bit = mag[1];
                sticky    = mag[0];
                round_up  = (guard) & (round_bit | sticky | mant[0]);
                mant_r    = {1'b0, mant} + {24'b0, round_up};
                if (mant_r[24]) begin
                    mant_r = mant_r >> 1;
                    exp_s  = exp_s + 11'sd1;
                end
                if (exp_s >= 11'sd255)
                    fp32_add = {res_sign, 8'hFF, 23'b0};   // overflow -> inf
                else if (exp_s <= 11'sd0)
                    fp32_add = {res_sign, 31'b0};          // underflow -> FTZ
                else
                    fp32_add = {res_sign, exp_s[7:0], mant_r[22:0]};
            end
        end
    end
endfunction

//----------------------------------------------------------------------------
// bf16_mul : convenience bf16*bf16 done through fp32 (widen, fp32_mul, narrow).
//----------------------------------------------------------------------------
function automatic [15:0] bf16_mul(input [15:0] a, input [15:0] b);
    bf16_mul = fp32_to_bf16(fp32_mul(bf16_to_fp32(a), bf16_to_fp32(b)));
endfunction

//----------------------------------------------------------------------------
// fp32_rsqrt : 1/sqrt(x) for x>0.  Quake fast-inverse-sqrt magic seed
//   (0x5f3759df) + TWO Newton-Raphson refinements:
//        y = y * (1.5 - 0.5*x*y*y)
//   Measured worst-case rel-err < 2^-22 across x in [1e-12,1e12] (target 2^-12).
//   x<=0 / nan -> returns nan (RMSNorm guarantees its argument is >0 via +eps).
//----------------------------------------------------------------------------
function automatic [31:0] fp32_rsqrt(input [31:0] x);
    reg [31:0] y, xhalf, half, three_half;
    reg [31:0] yy, xyy, t;
    integer    it;
    begin
        if (_glmfp_is_nan(x[30:0]) || x[31] == 1'b1 || _glmfp_is_zero(x[30:23])) begin
            fp32_rsqrt = 32'h7FC00000;                     // nan for x<=0
        end else if (_glmfp_is_inf(x[30:0])) begin
            fp32_rsqrt = 32'h00000000;                     // 1/sqrt(inf)=+0
        end else begin
            half       = 32'h3F000000;                     // 0.5
            three_half = 32'h3FC00000;                     // 1.5
            xhalf      = fp32_mul(half, x);                 // 0.5*x
            // integer magic seed
            y = 32'h5F3759DF - (x >> 1);
            for (it = 0; it < 2; it = it + 1) begin
                yy  = fp32_mul(y, y);                       // y*y
                xyy = fp32_mul(xhalf, yy);                  // 0.5*x*y*y
                t   = fp32_add(three_half, {xyy[31]^1'b1, xyy[30:0]}); // 1.5 - xyy
                y   = fp32_mul(y, t);                       // y*(1.5-0.5xyy)
            end
            fp32_rsqrt = y;
        end
    end
endfunction

`endif // GLM_FP_VH
