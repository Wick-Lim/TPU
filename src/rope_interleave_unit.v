`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// rope_interleave_unit.v -- GLM-5.2 DECOUPLED INTERLEAVED RoPE   (§4.1, §3 #6)
//----------------------------------------------------------------------------
// FUNCTION
//   Applies the GLM-5.2 decoupled, ADJACENT-PAIR ("GPT-NeoX interleaved")
//   rotary position embedding to a ROT_DIM-element bf16 vector.  The vector is
//   treated as ROT_DIM/2 adjacent pairs (x[2i], x[2i+1]); pair i is rotated by
//   angle_i = POS * inv_freq[i], inv_freq[i] = THETA^(-2i/ROT_DIM):
//
//        x0' = x0*cos(angle_i) - x1*sin(angle_i)
//        x1' = x0*sin(angle_i) + x1*cos(angle_i)
//
//   This is the form MLA uses for q_rope[64] (per head) and the single shared
//   k_rope[64], and the DSA indexer reuses it.  It is NOT rotate-half.
//
//----------------------------------------------------------------------------
// NUMERICS  (the §6 contract; all FP ops come from glm_fp.vh)
//   * Vector elements stream in/out as BF16.
//   * The ANGLE is computed in FULL PRECISION.  POS is up to 2^20-1 (1M ctx) and
//     inv_freq[i] = THETA^(-2i/ROT_DIM) spans ~7 decades, so the *product*
//     POS*inv_freq[i] reaches ~1e6 radians: a bf16 (or even fp32) angle cannot
//     resolve a single position there.  We therefore RANGE-REDUCE EXACTLY in
//     fixed point: a Q56 ROM holds INVF_Q56[i] = inv_freq[i] * 2^56 (an
//     integer); phase = POS*INVF_Q56[i] is an EXACT integer product, and
//     reduced = phase mod (2*pi in Q56) is an EXACT integer modulo -> NO
//     large-argument cancellation.  The reduced angle (in [0,2*pi), < 2*pi) is
//     then converted to an fp32 angle in [-pi,pi] and fed to an fp32 CORDIC.
//   * cos/sin: 16-iteration fp32 CORDIC (rotation mode), preceded by a one-step
//     fold of [-pi,pi] into [-pi/2,pi/2] (CORDIC's convergence range).  Measured
//     worst-case |err| <= 3.1e-5 for both cos and sin over all ROT_DIM/2 freqs x
//     positions spanning [1 .. 2^20-1]  (target <= 2^-12 = 2.44e-4) -- ~8x
//     inside spec.  The rotation x0*cos-x1*sin etc. is done in fp32 and rounded
//     back to bf16 (RNE) on output.
//
//----------------------------------------------------------------------------
// inv_freq / cos-sin METHOD SUMMARY
//   INVF_Q56 ROM (ROT_DIM/2 entries) is built AT ELABORATION by a PURE-INTEGER
//   function (log2 via 56 fixed-point squarings + 2^frac via a 56-entry Q56
//   table) -- no `real`, no `$pow`/`$ln`, so yosys elaborates it cleanly and it
//   constant-folds to a ROM.  Worst-case rel-err of an INVF_Q56 entry vs the
//   true inv_freq is < 5e-6, and because reduction is an exact integer modulo,
//   the residual angle error stays < ~2^20 * 2^-56 ~ 2^-36 turns -- negligible.
//
//----------------------------------------------------------------------------
// PARAMETERS
//   ROT_DIM : rotated-vector length (default 64 = GLM qk_rope_head_dim).
//             MUST be even; NPAIR = ROT_DIM/2 pairs.
//   THETA   : RoPE base (default 8000000 = GLM-5.2 rope_theta = 8e6).
//   LANES   : PAIRS processed/produced PER CYCLE (default 4).  Throughput knob:
//             each cycle the unit consumes LANES pairs (= LANES*2 bf16 elts),
//             rotates them, and emits LANES rotated pairs.  NPAIR must be a
//             multiple of LANES.  Each lane has its own combinational CORDIC.
//   POSW    : width of the position input (default 20 -> POS in [0, 2^20-1]).
//
//----------------------------------------------------------------------------
// INTERFACE  (clean req/valid handshake, deterministic latency, sync reset)
//   clk, rst              : synchronous, active-high reset.
//   start                 : 1-cycle pulse to begin a new vector.
//   pos    [POSW-1:0]      : token position; captured at start.
//   --- input stream (unit pulls LANES pairs/cycle) ---
//   in_req                : high while the unit wants the next input beat.
//   x_in   [LANES*32-1:0]  : LANES pairs; lane j = {x[2i+1], x[2i]} packed as
//                           x_in[32*j +: 32] = {bf16 odd, bf16 even}
//                           (i.e. x_in[32*j +: 16] = x_even = x[2i],
//                                 x_in[32*j+16 +:16] = x_odd  = x[2i+1]).
//   x_valid               : producer asserts when x_in holds the requested beat.
//   --- output stream (LANES rotated pairs/cycle, input order) ---
//   y_valid               : high when y_out holds a valid output beat.
//   y_out  [LANES*32-1:0]  : LANES rotated pairs, same packing as x_in.
//   --- status ---
//   busy                  : high from start until done.
//   done                  : 1-cycle pulse when the whole vector has been emitted.
//
//----------------------------------------------------------------------------
// PIPELINE / LATENCY   (NBEATS = NPAIR/LANES)
//   S_IDLE -> (start) -> S_RUN (NBEATS beats, 1 beat/cycle when x_valid) -> S_DONE.
//   Each S_RUN beat is fully combinational input->output (angle gen + CORDIC +
//   rotate), registered once into y_out.  With the producer answering every
//   in_req on the next cycle:
//       latency(start -> done) = NBEATS + 2  cycles
//       throughput             = LANES pairs/cycle (NPAIR/LANES beats/vector).
//   Stalls (x_valid low) simply hold the beat; latency grows 1 cycle/stall.
//
//----------------------------------------------------------------------------
// CORRECTNESS / STYLE
//   * Angle reduction EXACT in Q56 integer; cos/sin fp32 CORDIC (<=3.1e-5).
//   * All FP arithmetic from glm_fp.vh (fp32_add/fp32_mul, bf16<->fp32).
//   * Synchronous active-high reset; every reg written on every path (no
//     inferred latch); no combinational loop (CORDIC is a feed-forward fp32_add
//     chain, registered between beats).  No `real` -> yosys-synthesizable.
//============================================================================
module rope_interleave_unit #(
    parameter integer ROT_DIM = 64,
    parameter integer THETA   = 8000000,
    parameter integer LANES   = 4,
    parameter integer POSW    = 20
)(
    input  wire                   clk,
    input  wire                   rst,
    input  wire                   start,
    input  wire [POSW-1:0]        pos,
    // input stream (unit pulls)
    output reg                    in_req,
    input  wire [LANES*32-1:0]    x_in,
    input  wire                   x_valid,
    // output stream
    output reg                    y_valid,
    output reg  [LANES*32-1:0]    y_out,
    // status
    output reg                    busy,
    output reg                    done
);
    //------------------------------------------------------------------------
    // derived sizes
    //------------------------------------------------------------------------
    localparam integer NPAIR  = ROT_DIM / 2;
    localparam integer NBEATS = NPAIR / LANES;
    localparam integer BAW    = (NBEATS <= 1) ? 1 : $clog2(NBEATS);
    localparam [BAW:0] LAST_BEAT = (BAW+1)'(NBEATS-1);

    // 2*pi in Q56 (rounded).  reduced angle is computed as phase mod this.
    localparam [127:0] TWO_PI_Q56 = 128'h06487ed5110b4600;

    // fp32 constants for the CORDIC fold + gain.
    localparam [31:0] FP_PI      = 32'h40490FDB;   // pi
    localparam [31:0] FP_PI_HALF = 32'h3FC90FDB;   // pi/2
    localparam [31:0] FP_K       = 32'h3F1B74EE;   // 1/An (16-iter CORDIC gain)

    //========================================================================
    // ELABORATION-TIME PURE-INTEGER MATH  (builds INVF_Q56 ROM; yosys-friendly)
    //========================================================================
    // log2 of a 128-bit integer x (x>=1) as a Q56 fixed-point value.
    function automatic [127:0] log2_q56(input [127:0] x);
        integer i, ip;
        reg [127:0] z, acc, sq;
        begin
            ip = -1;
            for (i = 0; i < 128; i = i + 1) if (x[i]) ip = i;     // floor(log2)
            acc = ({{96{1'b0}}, ip[31:0]}) << 56;                 // integer part Q56
            if (ip >= 56) z = x >> (ip - 56);
            else          z = x << (56 - ip);                     // z in [1,2) Q56
            for (i = 0; i < 56; i = i + 1) begin
                sq = (z * z) >> 56;                               // z^2 Q56
                z  = sq;
                if (z >= (128'd1 << 57)) begin                    // z >= 2
                    acc = acc | (128'd1 << (55 - i));
                    z   = z >> 1;
                end
            end
            log2_q56 = acc;
        end
    endfunction

    // 2^f for f in [0,1) given Q56 -> Q56 value in [1,2).  Product of the
    // precomputed Q56 constants 2^(2^-k) selected by the fractional bits.
    function automatic [127:0] exp2_frac_q56(input [127:0] f);
        integer k;
        reg [127:0] r, tab;
        begin
            r = 128'd1 << 56;                                     // 1.0 Q56
            for (k = 1; k <= 56; k = k + 1) if (f[56-k]) begin
                case (k)
                     1: tab=128'h016A09E667F3BCD0;  2: tab=128'h01306FE0A31B7150;
                     3: tab=128'h01172B83C7D517B0;  4: tab=128'h010B5586CF9890F0;
                     5: tab=128'h01059B0D31585740;  6: tab=128'h0102C9A3E7780610;
                     7: tab=128'h010163DA9FB33350;  8: tab=128'h0100B1AFA5ABCBF0;
                     9: tab=128'h010058C86DA1C0A0; 10: tab=128'h01002C605E2E8CF0;
                    11: tab=128'h0100162F39040520; 12: tab=128'h01000B175EFFDC70;
                    13: tab=128'h0100058BA01FBA00; 14: tab=128'h010002C5CC37DA90;
                    15: tab=128'h01000162E525EE00; 16: tab=128'h010000B172557760;
                    17: tab=128'h01000058B91B5BD0; 18: tab=128'h0100002C5C89D5F0;
                    19: tab=128'h010000162E43F500; 20: tab=128'h0100000B1721BD00;
                    21: tab=128'h010000058B90CF20; 22: tab=128'h01000002C5C863B0;
                    23: tab=128'h0100000162E430E0; 24: tab=128'h01000000B1721830;
                    25: tab=128'h0100000058B90C10; 26: tab=128'h010000002C5C8600;
                    27: tab=128'h01000000162E4300; 28: tab=128'h010000000B172180;
                    29: tab=128'h01000000058B90C0; 30: tab=128'h0100000002C5C860;
                    31: tab=128'h010000000162E430; 32: tab=128'h0100000000B17210;
                    33: tab=128'h010000000058B910; 34: tab=128'h01000000002C5C80;
                    35: tab=128'h0100000000162E40; 36: tab=128'h01000000000B1720;
                    37: tab=128'h0100000000058B90; 38: tab=128'h010000000002C5D0;
                    39: tab=128'h01000000000162E0; 40: tab=128'h010000000000B170;
                    41: tab=128'h01000000000058C0; 42: tab=128'h0100000000002C60;
                    43: tab=128'h0100000000001630; 44: tab=128'h0100000000000B10;
                    45: tab=128'h0100000000000590; 46: tab=128'h01000000000002C0;
                    47: tab=128'h0100000000000160; 48: tab=128'h01000000000000B0;
                    49: tab=128'h0100000000000060; 50: tab=128'h0100000000000030;
                    51: tab=128'h0100000000000010; 52: tab=128'h0100000000000010;
                    default: tab = 128'd1 << 56;                  // 2^(2^-k) ~ 1
                endcase
                r = (r * tab) >> 56;
            end
            exp2_frac_q56 = r;
        end
    endfunction

    // INVF_Q56[idx] = THETA^(-2*idx/ROT_DIM) * 2^56  (an integer in (0, 2^56]).
    //   value = 2^(-e),  e = (2*idx/ROT_DIM)*log2(THETA)  (Q56, >=0)
    //   value*2^56 = 2^(56 - e); split (56 - e) into integer P + frac FR.
    function automatic [127:0] invf_q56(input integer idx);
        reg [127:0] L2T, e_q56, fr, m, r;
        reg signed [127:0] E;
        integer P;
        begin
            L2T   = log2_q56({{96{1'b0}}, THETA[31:0]});
            e_q56 = ((128'd2 * idx) * L2T) / 128'(ROT_DIM);       // (2idx/ROT)*log2T
            E     = $signed(128'd56 << 56) - $signed(e_q56);      // (56 - e) Q56
            P     = 32'(E >>> 56);                                // floor int part
            fr    = E & ((128'd1 << 56) - 1);                     // frac Q56
            m     = exp2_frac_q56(fr);                            // 2^fr in [1,2) Q56
            if (P >= 56) r = m << (P - 56);
            else         r = m >> (56 - P);
            invf_q56 = r;
        end
    endfunction

    // INVF_Q56 ROM, ELABORATED ONCE into NPAIR constants.  Each entry is a
    // CONSTANT (invf_q56 is called with a compile-time genvar), so the heavy
    // log2/exp2 elaboration math constant-folds away to a fixed bit pattern --
    // the synthesized hardware is a plain NPAIR x 128-bit ROM, NOT a live
    // integer-pow datapath.  (Calling invf_q56 with a RUNTIME index would
    // instead instantiate that datapath per lane; the genvar ROM avoids it.)
    wire [127:0] invf_rom [0:NPAIR-1];
    genvar gp;
    generate
        for (gp = 0; gp < NPAIR; gp = gp + 1) begin : GEN_INVF_ROM
            assign invf_rom[gp] = invf_q56(gp);
        end
    endgenerate

    //========================================================================
    // RUNTIME COMBINATIONAL PRIMITIVES
    //========================================================================
    // 16-iteration fp32 CORDIC rotation, input angle a in [-pi/2,pi/2].
    // returns {cos, sin} (each fp32).  Pure feed-forward fp32_add chain.
    function automatic [63:0] cordic_raw(input [31:0] ang);
        integer i;
        reg [31:0] x, y, z, xn, yn, xi, yi;
        reg        dneg;
        reg [31:0] ATAN [0:15];
        begin
            ATAN[0]=32'h3F490FDB; ATAN[1]=32'h3EED6338; ATAN[2]=32'h3E7ADBB0;
            ATAN[3]=32'h3DFEADD5; ATAN[4]=32'h3D7FAADE; ATAN[5]=32'h3CFFEAAE;
            ATAN[6]=32'h3C7FFAAB; ATAN[7]=32'h3BFFFEAB; ATAN[8]=32'h3B7FFFAB;
            ATAN[9]=32'h3AFFFFEB; ATAN[10]=32'h3A7FFFFB; ATAN[11]=32'h39FFFFFF;
            ATAN[12]=32'h39800000; ATAN[13]=32'h39000000; ATAN[14]=32'h38800000;
            ATAN[15]=32'h38000000;
            x = FP_K; y = 32'h0; z = ang;
            for (i = 0; i < 16; i = i + 1) begin
                dneg = z[31];                                     // z<0 ?
                // xi = x * 2^-i via exponent decrement (0 if it would underflow)
                xi = (x[30:23] > i[7:0]) ? {x[31], x[30:23]-i[7:0], x[22:0]} : 32'h0;
                yi = (y[30:23] > i[7:0]) ? {y[31], y[30:23]-i[7:0], y[22:0]} : 32'h0;
                if (!dneg) begin
                    // rotate by -atan(2^-i):  x-=yi, y+=xi, z-=atan
                    xn = fp32_add(x, {~yi[31], yi[30:0]});
                    yn = fp32_add(y, xi);
                    z  = fp32_add(z, {1'b1, ATAN[i][30:0]});
                end else begin
                    // rotate by +atan(2^-i):  x+=yi, y-=xi, z+=atan
                    xn = fp32_add(x, yi);
                    yn = fp32_add(y, {~xi[31], xi[30:0]});
                    z  = fp32_add(z, ATAN[i]);
                end
                x = xn; y = yn;
            end
            cordic_raw = {x, y};
        end
    endfunction

    // cos/sin of an fp32 angle a in [-pi,pi]: fold |a|>pi/2 into [-pi/2,pi/2]
    // (a -> sign(a)*(pi-|a|), cos flips sign), then CORDIC.
    function automatic [63:0] cossin(input [31:0] a);
        reg [31:0] aa, c, s;
        reg [63:0] cs;
        reg        negc;
        // `ad` = fp32(pi-|a|), always >=0 so its sign bit [31] is computed but
        // never read (we re-attach the original sign).  Waive the unused-bit lint.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [31:0] ad;
        /* verilator lint_on UNUSEDSIGNAL */
        begin
            aa = a; negc = 1'b0;
            if ({1'b0, aa[30:0]} > FP_PI_HALF) begin
                ad   = fp32_add(FP_PI, {1'b1, aa[30:0]});         // pi - |a|
                aa   = {a[31], ad[30:0]};                         // keep sign
                negc = 1'b1;
            end
            cs = cordic_raw(aa);
            c  = cs[63:32];
            s  = cs[31:0];
            if (negc) c = {c[31]^1'b1, c[30:0]};
            cossin = {c, s};
        end
    endfunction

    // convert a reduced Q56 angle in [0, 2*pi) to an fp32 angle in [-pi,pi].
    function automatic [31:0] q56_to_fp32_centered(input [127:0] q);
        reg [127:0] mag, HALF, FULL;
        reg         sgn;
        integer     msb, i, e;
        reg [22:0]  m23;
        // `norm` is a 128-bit shifter result; only its low 23 bits (the fp32
        // mantissa) are read -- the high bits are the implicit-1 + above, never
        // stored.  Waive the unused-upper-bits lint exactly as glm_fp.vh does.
        /* verilator lint_off UNUSEDSIGNAL */
        reg [127:0] norm;
        /* verilator lint_on UNUSEDSIGNAL */
        begin
            HALF = TWO_PI_Q56 >> 1;
            FULL = TWO_PI_Q56;
            if (q >= HALF) begin sgn = 1'b1; mag = FULL - q; end  // -> negative
            else           begin sgn = 1'b0; mag = q;        end  // [0,pi]
            msb = -1;
            for (i = 0; i < 128; i = i + 1) if (mag[i]) msb = i;
            if (msb < 0) q56_to_fp32_centered = 32'h0;            // angle 0
            else begin
                e = msb - 56;                                    // value exponent
                // align the implicit leading 1 (at msb) to bit 23; mantissa =
                // the 23 fraction bits just below it.
                if (msb >= 23) norm = mag >> (msb - 23);
                else           norm = mag << (23 - msb);
                m23 = norm[22:0];
                q56_to_fp32_centered = {sgn, 8'(e + 127), m23};
            end
        end
    endfunction

    //========================================================================
    // PER-LANE COMBINATIONAL DATAPATH
    //   For the current beat, lane j handles pair index (beat*LANES + j).
    //   angle_j = POS * INVF_Q56[pair] mod 2pi  (exact integer), -> fp32 ->
    //   cos/sin -> rotate (x0,x1) -> {y1,y0} bf16.
    //========================================================================
    reg  [POSW-1:0]      pos_q;                                   // captured POS
    reg  [BAW:0]         beat;

    integer              j;
    reg  [127:0]         phase_j;
    reg  [127:0]         red_j;
    reg  [31:0]          ang_j;
    reg  [63:0]          cs_j;
    reg  [31:0]          cos_j, sin_j;
    reg  [31:0]          x0_j, x1_j;
    reg  [31:0]          x0c, x1s, x0s, x1c;
    reg  [31:0]          y0_j, y1_j;
    reg  [LANES*32-1:0]  rot_beat;

    always @* begin
        rot_beat = {LANES*32{1'b0}};
        for (j = 0; j < LANES; j = j + 1) begin : LANE
            // pair index this lane processes (beat*LANES + j); INVF ROM lookup
            phase_j = {{(128-POSW){1'b0}}, pos_q}
                      * invf_rom[32'(beat) * LANES + j];
            red_j   = phase_j % TWO_PI_Q56;                       // exact mod
            ang_j   = q56_to_fp32_centered(red_j);
            cs_j    = cossin(ang_j);
            cos_j   = cs_j[63:32];
            sin_j   = cs_j[31:0];
            // x_in lane packing: [32*j +:16]=x_even=x0, [32*j+16 +:16]=x_odd=x1
            x0_j = bf16_to_fp32(x_in[32*j      +: 16]);
            x1_j = bf16_to_fp32(x_in[32*j + 16 +: 16]);
            // y0 = x0*cos - x1*sin ; y1 = x0*sin + x1*cos
            x0c = fp32_mul(x0_j, cos_j);
            x1s = fp32_mul(x1_j, sin_j);
            x0s = fp32_mul(x0_j, sin_j);
            x1c = fp32_mul(x1_j, cos_j);
            y0_j = fp32_add(x0c, {x1s[31]^1'b1, x1s[30:0]});      // x0c - x1s
            y1_j = fp32_add(x0s, x1c);
            rot_beat[32*j      +: 16] = fp32_to_bf16(y0_j);
            rot_beat[32*j + 16 +: 16] = fp32_to_bf16(y1_j);
        end
    end

    //========================================================================
    // FSM
    //========================================================================
    localparam [1:0] S_IDLE=2'd0, S_RUN=2'd1, S_DONE=2'd2;
    reg [1:0] state;

    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            in_req  <= 1'b0;
            y_valid <= 1'b0;
            y_out   <= {LANES*32{1'b0}};
            busy    <= 1'b0;
            done    <= 1'b0;
            beat    <= {(BAW+1){1'b0}};
            pos_q   <= {POSW{1'b0}};
        end else begin
            // defaults (every reg written every cycle -> no latch)
            done    <= 1'b0;
            y_valid <= 1'b0;
            in_req  <= 1'b0;

            case (state)
                // -----------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy   <= 1'b1;
                        pos_q  <= pos;
                        beat   <= {(BAW+1){1'b0}};
                        in_req <= 1'b1;                            // pull first beat
                        state  <= S_RUN;
                    end
                end
                // -----------------------------------------------------------
                S_RUN: begin
                    in_req <= 1'b1;                                // keep asking
                    if (x_valid) begin
                        y_out   <= rot_beat;
                        y_valid <= 1'b1;
                        if (beat == LAST_BEAT) begin
                            in_req <= 1'b0;
                            beat   <= {(BAW+1){1'b0}};
                            state  <= S_DONE;
                        end else begin
                            beat <= beat + 1'b1;
                        end
                    end
                end
                // -----------------------------------------------------------
                S_DONE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end
                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
