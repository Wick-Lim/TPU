`ifndef FP8_E4M3_VH
`define FP8_E4M3_VH
//============================================================================
// fp8_e4m3.vh  --  GLM-5.2 FP8 E4M3 PRIMITIVES  (the FP8-native numerics)
//                                                            (ACCEL_GLM52 §6)
//----------------------------------------------------------------------------
// PURPOSE
//   GLM-5.2 ships officially in FP8 (zai-org/GLM-5.2-FP8): E4M3 weights with
//   per-block scales.  This header is the SINGLE canonical definition of the
//   FP8 E4M3 numerics so the published checkpoint runs with NO re-quantization.
//   It mirrors glm_fp.vh's style: every function here is a SYNTHESIZABLE,
//   PURELY COMBINATIONAL `function automatic` -- no state, no clocks, no side
//   effects -- included into a module's scope via `include "fp8_e4m3.vh"`.
//
//   THE HARDWARE WIN.  The product of two E4M3 values is a 4-bit x 4-bit
//   mantissa multiply (the implicit-1 significands "1.mmm" are 4-bit integers
//   8..15, or 1..7 for subnormals).  fp8_mul does EXACTLY that 4x4 integer
//   multiply (Mp = Ma*Mb, at most 8 bits), adds the integer exponents, xors
//   the signs, and normalizes -- it NEVER decodes to fp32 and uses a 24x24
//   fp32 multiply.  A 4x4 mult is pure LUT (the plentiful Gowin GW2A-18 LUT,
//   20,736) and frees the SCARCE DSP (only 48) -- the opposite resource
//   profile from the DSP-bound fp32 datapath (mla_attn ~396 DSP-equiv).
//
//----------------------------------------------------------------------------
// FORMAT  --  FP8 E4M3 (OCP fp8, the GLM-5.2 weight format)
//   1 sign | 4 exponent (bias 7) | 3 mantissa.   NO Infinity.
//     exp field 1..15 : NORMAL,  value = (-1)^s * 2^(e-7) * (1 + m/8)
//                       (the implicit leading 1 -> 4 significand bits "1.mmm")
//     exp field 0     : ZERO (m==0, signed) or SUBNORMAL (m!=0),
//                       value = (-1)^s * 2^-6 * (m/8) = (-1)^s * m * 2^-9
//     S.1111.111      : the ONE NaN pattern (exp all-ones AND mant all-ones)
//   Max normal = S.1111.110 = 2^8 * 1.75 = 448.   Smallest subnormal = 2^-9.
//   (S.1111.111 occupies what would be 480; it is NaN, so 480 is NOT
//    representable and overflow saturates to +/-448.)
//
//----------------------------------------------------------------------------
// SPECIAL-VALUE / ROUNDING POLICY  (documented, enforced uniformly)
//   * Decode preserves the sign of NaN and of signed zero; NaN -> a quiet
//     fp32 NaN {s,8'hFF,quiet}.  (E4M3 has no Inf so decode never makes Inf.)
//   * Encode (fp32 -> E4M3): ROUND-TO-NEAREST-EVEN with SATURATION to +/-448
//     (the max normal); small magnitudes produce SUBNORMALS (and round
//     correctly up into the smallest normal); +/-0 preserved; fp32 NaN ->
//     S.1111.111; fp32 Inf -> +/-448 (satfinite: E4M3 has no Inf).  fp32
//     subnormal inputs (|f| < 2^-126) are far below the FP8 grid -> +/-0.
//   * fp8_mul is EXACT: the product (Mp <= 225 fits 8 bits, small integer
//     exponent) is exactly representable in fp32, so it is returned WITHOUT
//     rounding.  Any NaN operand -> qnan; any zero operand (finite) -> signed
//     zero.  No E4M3 product overflows or underflows fp32 (range 2^-18..448^2).
//
//----------------------------------------------------------------------------
// API  (all `function automatic`, combinational)
//   fp8e4m3_to_fp32(input [7:0]  x)        -> [31:0]   decode E4M3  -> fp32
//   fp32_to_fp8e4m3(input [31:0] f)        -> [7:0]    encode fp32  -> E4M3 (RNE+sat)
//   fp8_mul        (input [7:0]  a, b)     -> [31:0]   E4M3*E4M3 -> EXACT fp32
//                                                      (via a 4x4 mantissa mult)
//============================================================================

//----------------------------------------------------------------------------
// fp8e4m3_to_fp32 : decode E4M3 -> fp32.  Exact (every E4M3 value is exactly
//   representable in fp32).  Handles zero, subnormals, normals, NaN.
//----------------------------------------------------------------------------
function automatic [31:0] fp8e4m3_to_fp32(input [7:0] x);
    reg        s;
    reg [3:0]  e;
    reg [2:0]  m;
    reg [7:0]  nexp;
    begin
        s = x[7];
        e = x[6:3];
        m = x[2:0];
        if (e == 4'hF && m == 3'h7) begin
            // the single NaN pattern -> quiet fp32 NaN, sign preserved
            fp8e4m3_to_fp32 = {s, 8'hFF, 1'b1, 22'b0};
        end else if (e == 4'h0 && m == 3'h0) begin
            // signed zero
            fp8e4m3_to_fp32 = {s, 31'b0};
        end else if (e == 4'h0) begin
            // subnormal: value = m * 2^-9 (m in 1..7), normalize to 1.f * 2^E
            if (m[2])
                fp8e4m3_to_fp32 = {s, 8'd120, m[1:0], 21'b0};   // 1.xx * 2^-7
            else if (m[1])
                fp8e4m3_to_fp32 = {s, 8'd119, m[0],   22'b0};   // 1.x  * 2^-8
            else
                fp8e4m3_to_fp32 = {s, 8'd118,         23'b0};   // 1.0  * 2^-9
        end else begin
            // normal: fp32 exp = (e - 7) + 127 = e + 120; mantissa = m left-justified
            nexp = 8'd120 + {4'b0, e};
            fp8e4m3_to_fp32 = {s, nexp, m, 20'b0};
        end
    end
endfunction

//----------------------------------------------------------------------------
// fp32_to_fp8e4m3 : encode fp32 -> E4M3, round-to-nearest-even + saturation.
//   Produces subnormals for small magnitudes (rounding up into the smallest
//   normal where appropriate); NaN -> S.1111.111; Inf -> +/-448 (satfinite);
//   +/-0 preserved.
//----------------------------------------------------------------------------
function automatic [7:0] fp32_to_fp8e4m3(input [31:0] f);
    reg               s;
    reg [7:0]         e;
    reg [22:0]        m;
    reg signed [9:0]  E;        // unbiased fp32 exponent (e - 127)
    reg [23:0]        sig;      // 1.m significand (24-bit)
    // --- normal path ---
    reg               g, st, rup;
    reg [3:0]         msum;     // {carry, 3-bit mantissa} after the RNE add
    reg signed [9:0]  efield;   // E4M3 exponent field (signed for range check)
    reg [2:0]         mant3;
    // --- subnormal path ---
    reg [3:0]         q;        // rounded subnormal mantissa (0..8)
    reg [3:0]         q_int;
    begin
        s = f[31];
        e = f[30:23];
        m = f[22:0];
        if (e == 8'hFF && m != 23'b0) begin
            fp32_to_fp8e4m3 = {s, 7'b1111111};            // NaN
        end else if (e == 8'hFF) begin
            fp32_to_fp8e4m3 = {s, 7'b1111110};            // Inf -> saturate to 448
        end else if (e == 8'h00) begin
            fp32_to_fp8e4m3 = {s, 7'b0000000};            // zero (FTZ fp32 subnormal)
        end else begin
            E   = $signed({2'b0, e}) - 10'sd127;
            sig = {1'b1, m};
            if (E >= -10'sd6) begin
                // ---------------- NORMAL E4M3 result ----------------
                efield = E + 10'sd7;                      // >= 1
                // RNE keeping the top 3 mantissa bits (drop low 20)
                g   = m[19];
                st  = |m[18:0];
                rup = g & (st | m[20]);
                // 4-bit add of the 3 kept mantissa bits + round; msum[3] is the
                // carry out of the mantissa (1.111 + ulp -> 10.000 = exp+1).
                msum = {1'b0, m[22:20]} + {3'b0, rup};
                if (msum[3]) begin                        // mantissa carried out
                    efield = efield + 10'sd1;
                    mant3  = 3'b000;                       // 1.000 * 2^(E+1)
                end else begin
                    mant3  = msum[2:0];
                end
                // overflow / NaN-slot -> saturate to max normal 448
                if (efield > 10'sd15 || (efield == 10'sd15 && mant3 == 3'b111))
                    fp32_to_fp8e4m3 = {s, 7'b1111110};
                else
                    fp32_to_fp8e4m3 = {s, efield[3:0], mant3};
            end else if (E <= -10'sd11) begin
                // below half the smallest subnormal step -> rounds to zero
                fp32_to_fp8e4m3 = {s, 7'b0000000};
            end else begin
                // ---------------- SUBNORMAL E4M3 result ----------------
                // value = 1.m * 2^E, E in [-10..-7].  Round to the nearest
                // multiple of 2^-9 (= round(value*2^9) = round(sig*2^(E-14))).
                // Use FIXED slices of sig per E (no variable shift): the integer
                // part q_int and guard g land at constant bit positions.
                case (E)
                    -10'sd7: begin q_int = {1'b0, sig[23:21]}; g = sig[20]; st = (sig[19:0] != 20'b0); end
                    -10'sd8: begin q_int = {2'b0, sig[23:22]}; g = sig[21]; st = (sig[20:0] != 21'b0); end
                    -10'sd9: begin q_int = {3'b0, sig[23]};    g = sig[22]; st = (sig[21:0] != 22'b0); end
                    default: begin q_int = 4'd0;               g = sig[23]; st = (sig[22:0] != 23'b0); end // E=-10
                endcase
                rup = g & (st | q_int[0]);                 // RNE
                q   = q_int + {3'b0, rup};                 // 0..8
                if (q == 4'd0)
                    fp32_to_fp8e4m3 = {s, 7'b0000000};     // -> signed zero
                else if (q == 4'd8)
                    fp32_to_fp8e4m3 = {s, 4'b0001, 3'b000};// rounded up to smallest normal
                else
                    fp32_to_fp8e4m3 = {s, 4'b0000, q[2:0]};// subnormal
            end
        end
    end
endfunction

//----------------------------------------------------------------------------
// fp8_mul : multiply two E4M3 values, returning the EXACT fp32 product.
//
//   THE 4x4 MANTISSA MULTIPLY (the DSP win).  Each nonzero operand is decoded
//   to an INTEGER significand M and an INTEGER exponent P with value = M*2^P:
//       normal    (efield 1..15) : M = {1,mmm} = 8 + mant   (4-bit, 8..15)
//                                   P = efield - 10
//       subnormal (efield 0)     : M = mant                 (3-bit, 1..7)
//                                   P = -9
//   The product significand is the single 4-bit x 4-bit integer multiply
//       Mp = Ma * Mb        (<= 15*15 = 225, an 8-bit integer)
//   with Pp = Pa + Pb and sign = sa ^ sb.  Mp*2^Pp is EXACT in fp32 (8-bit
//   significand, exponent in [109..144]), so it is normalized and returned
//   with NO rounding.  Note Ma*Mb is a small integer mult -> synthesizes to
//   LUT, NOT to the 24x24 fp32 DSP multiplier.
//----------------------------------------------------------------------------
function automatic [31:0] fp8_mul(input [7:0] a, input [7:0] b);
    reg               sa, sb, sr;
    reg [3:0]         ea, eb;
    reg [2:0]         ma, mb;
    reg [3:0]         Ma, Mb;     // 4-bit significands "1.mmm" (or subnormal mant)
    reg [3:0]         PaB, PbB;   // biased integer exponents (Pa+9, Pb+9) in [0..14]
    reg [7:0]         Mp;         // the 4x4 product, up to 8 bits (<=225)
    reg [7:0]         norm8;      // Mp normalized so its MSB sits at bit 7
    reg [3:0]         k7;         // exponent of the implicit-1 (7 - #left-shifts)
    reg [22:0]        mant23;
    reg [7:0]         fexp;       // fp32 biased exponent, always in [109..144]
    integer           i;
    begin
        sa = a[7]; sb = b[7]; sr = sa ^ sb;
        ea = a[6:3]; eb = b[6:3];
        ma = a[2:0]; mb = b[2:0];
        if ((ea == 4'hF && ma == 3'h7) || (eb == 4'hF && mb == 3'h7)) begin
            fp8_mul = 32'h7FC00000;                        // NaN operand -> qnan
        end else if ((a[6:0] == 7'b0) || (b[6:0] == 7'b0)) begin
            fp8_mul = {sr, 31'b0};                         // zero operand -> signed 0
        end else begin
            // decode each operand to (integer significand M, biased exponent PB).
            //   value = M * 2^(PB-9).   normal: M={1,mmm}, PB=efield-1 (in 0..14)
            //                           subnorm: M={0,mmm}, PB=0  (i.e. P=-9)
            if (ea == 4'h0) begin Ma = {1'b0, ma}; PaB = 4'd0;     end
            else            begin Ma = {1'b1, ma}; PaB = ea - 4'd1; end
            if (eb == 4'h0) begin Mb = {1'b0, mb}; PbB = 4'd0;     end
            else            begin Mb = {1'b1, mb}; PbB = eb - 4'd1; end
            // ----------------- THE 4x4 MANTISSA MULTIPLY -----------------
            Mp = Ma * Mb;                                  // 4b x 4b -> <=225 (8b)
            // normalize Mp so its MSB lands at bit 7 (the implicit one); count
            // the left shifts with a BOUNDED, same-width loop (no variable shift).
            norm8 = Mp;
            k7    = 4'd7;
            for (i = 0; i < 7; i = i + 1)
                if (norm8[7] == 1'b0) begin
                    norm8 = norm8 << 1;
                    k7    = k7 - 4'd1;
                end
            // norm8[7]=1 -> fp32 fraction = norm8[6:0] (left-justified into 23b).
            mant23 = {norm8[6:0], 16'b0};
            // value = 1.f * 2^(k7 + (PaB-9) + (PbB-9)); fp32 exp adds bias 127:
            //   fexp = k7 + PaB + PbB - 18 + 127 = k7 + PaB + PbB + 109  (>=0, <=144)
            fexp = {4'b0, k7} + {4'b0, PaB} + {4'b0, PbB} + 8'd109;
            fp8_mul = {sr, fexp, mant23};
        end
    end
endfunction

`endif // FP8_E4M3_VH
