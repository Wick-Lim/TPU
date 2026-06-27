`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// rope_interleave_unit.v -- GLM-5.2 DECOUPLED INTERLEAVED RoPE   (§4.1, §3 #6)
//----------------------------------------------------------------------------
// FUNCTION  (UNCHANGED interface / semantics)
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
// LEAN REBUILD  (drop-in; same module/params/ports as the prior CORDIC unit)
//   The previous unit reduced the angle with a 128-bit EXACT integer modulo and
//   computed cos/sin with a 16-iteration fp32 CORDIC.  That was ~1.34M cells and
//   stopped yosys synth_ecp5 from converging.  bf16 storage only needs ~1/256
//   accuracy, so both pieces are replaced with far leaner schemes that stay
//   inside the EXISTING golden tolerance:
//
//   (1) RANGE REDUCTION as FRACTIONAL TURNS (modulo becomes a bit-truncate).
//       Instead of reducing radians mod 2*pi (an expensive 128-bit `%`), we work
//       in TURNS.  Per pair we hold a Q48 constant
//            F[i] = round( inv_freq[i] / (2*pi) * 2^48 )            (turns/pos)
//       built AT ELABORATION by the SAME pure-integer log2/exp2 ROM math the old
//       unit used for inv_freq (no `real`, constant-folds to a ROM).  Then
//            phase = POS * F[i]                                     (exact int)
//            frac_turns = phase[47:0]   == (POS*inv_freq/2pi) mod 1
//       i.e. reduction mod one full turn is simply the LOW 48 BITS -- no divide,
//       no 128-bit modulo.  F[i] carries <2^-29 turn error over POS<=2^20, and
//       (like before) inv_freq[0] is EXACTLY 1 so the fastest pair is exact and
//       POS=0 is a bit-exact identity.
//
//   (2) cos/sin from the turn fraction by QUADRANT FOLD + fp32 TAYLOR.
//       frac_turns[47:46] selects the quadrant; the remaining 46 bits become an
//       fp32 angle theta in [0, pi/2).  cos/sin(theta) use Taylor series
//            sin = th*(1 - th^2/6 + th^4/120 - th^6/5040 + th^8/362880)
//            cos =     1 - th^2/2 + th^4/24  - th^6/720  + th^8/40320
//       (Horner in u=theta^2), then the quadrant applies the sign/swap mapping.
//       Worst-case |err| over [0,pi/2] is ~2.5e-5 for both -- inside the golden's
//       cross-term allowance (CROSS = 2^-13 ~= 1.22e-4).  All fp32 ops come from
//       glm_fp.vh; the interleaved rotation pairing is byte-identical to before.
//
//----------------------------------------------------------------------------
// PARAMETERS  (names/defaults UNCHANGED)
//   ROT_DIM : rotated-vector length (default 64 = GLM qk_rope_head_dim). Even.
//   THETA   : RoPE base (default 8000000 = GLM-5.2 rope_theta = 8e6).
//   LANES   : pairs processed/produced per cycle (default 4). NPAIR % LANES == 0.
//   POSW    : width of the position input (default 20 -> POS in [0, 2^20-1]).
//
//----------------------------------------------------------------------------
// INTERFACE  (PORT LIST / TYPES / SEMANTICS UNCHANGED -- byte-identical to old)
//   clk, rst              : synchronous, active-high reset.
//   start                 : 1-cycle pulse to begin a new vector.
//   pos    [POSW-1:0]      : token position; captured at start.
//   in_req                : high while the unit wants the next input beat.
//   x_in   [LANES*32-1:0]  : LANES pairs; x_in[32*j +:16]=x_even=x[2i],
//                           x_in[32*j+16 +:16]=x_odd=x[2i+1].
//   x_valid               : producer asserts when x_in holds the requested beat.
//   y_valid               : high when y_out holds a valid output beat.
//   y_out  [LANES*32-1:0]  : LANES rotated pairs, same packing as x_in.
//   busy                  : high from start until done.
//   done                  : 1-cycle pulse when the whole vector has been emitted.
//
//----------------------------------------------------------------------------
// PIPELINE / LATENCY   (UNCHANGED; NBEATS = NPAIR/LANES)
//   S_IDLE -> (start) -> S_RUN (NBEATS beats, 1 beat/cycle when x_valid) -> S_DONE.
//   Each S_RUN beat is fully combinational input->output (turn gen + poly trig +
//   rotate), registered once into y_out.
//       latency(start -> done) = NBEATS + 2  cycles   (same as before)
//       throughput             = LANES pairs/cycle.
//   Stalls (x_valid low) hold the beat; latency grows 1 cycle/stall.
//
//----------------------------------------------------------------------------
// STYLE
//   * Turn reduction via Q48 ROM + bit-truncate (no 128-bit modulo, no divide).
//   * cos/sin: quadrant fold + fp32 Taylor (no CORDIC).
//   * All FP arithmetic from glm_fp.vh.  Sync active-high reset; every reg
//     written on every path (no latch); no combinational loop (trig is a
//     feed-forward fp32 chain, registered between beats); no `real`.
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

    // turn fixed-point: F[i] = round(inv_freq[i]/(2*pi) * 2^BFR); phase = pos*F.
    localparam integer BFR = 48;                       // fractional-turn bits
    localparam integer PW  = POSW + BFR;               // phase width (pos*F)

    // 1/(2*pi) in Q64 (round(2^64 / (2*pi))) -- exact integer constant.
    localparam [127:0] INV_2PI_Q64 = 128'h0000000000000000_28BE60DB9391054A;

    // fp32 constants for the trig polynomial.
    localparam [31:0] FP_PI_HALF = 32'h3FC90FDB;       // pi/2
    // sin Taylor coeffs in u = theta^2:  sin = th*(S0 + u*(S1 + u*(S2 + u*(S3 + u*S4))))
    localparam [31:0] S0 = 32'h3F800000;               //  1
    localparam [31:0] S1 = 32'hBE2AAAAB;               // -1/6
    localparam [31:0] S2 = 32'h3C088889;               //  1/120
    localparam [31:0] S3 = 32'hB9500D01;               // -1/5040
    localparam [31:0] S4 = 32'h3638EF1D;               //  1/362880
    // cos Taylor coeffs in u = theta^2:  cos = C0 + u*(C1 + u*(C2 + u*(C3 + u*C4)))
    localparam [31:0] C0 = 32'h3F800000;               //  1
    localparam [31:0] C1 = 32'hBF000000;               // -1/2
    localparam [31:0] C2 = 32'h3D2AAAAB;               //  1/24
    localparam [31:0] C3 = 32'hBAB60B61;               // -1/720
    localparam [31:0] C4 = 32'h37D00D01;               //  1/40320

    //========================================================================
    // ELABORATION-TIME PURE-INTEGER MATH  (builds the per-pair turn ROM)
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

    // 2^f for f in [0,1) given Q56 -> Q56 value in [1,2).
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

    // inv_freq[idx] * 2^56  (an integer in (0, 2^56]).
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

    // F[idx] = round( inv_freq[idx] / (2*pi) * 2^BFR )  (turns per position, Q48).
    //   inv_freq*2^56 (Q56) times 1/(2pi)*2^64 (Q64) = inv_freq/(2pi) * 2^120;
    //   shift right (120-BFR) with round-to-nearest -> Q(BFR).
    function automatic [PW-1:0] turn_per_pos(input integer idx);
        reg [255:0] prod;
        localparam integer SH = 120 - BFR;                       // = 72
        begin
            prod = invf_q56(idx) * INV_2PI_Q64;                  // Q120, < 2^118
            prod = prod + (256'd1 << (SH-1));                    // round to nearest
            turn_per_pos = prod[SH +: PW];                       // (>>SH) Q48 turns/pos
        end
    endfunction

    // Per-pair turn ROM, ELABORATED ONCE into NPAIR constants (constant-folds to
    // a plain ROM; the log2/exp2 datapath disappears because idx is a genvar).
    wire [PW-1:0] turn_rom [0:NPAIR-1];
    genvar gp;
    generate
        for (gp = 0; gp < NPAIR; gp = gp + 1) begin : GEN_TURN_ROM
            assign turn_rom[gp] = turn_per_pos(gp);
        end
    endgenerate

    //========================================================================
    // RUNTIME COMBINATIONAL PRIMITIVES
    //========================================================================
    // Convert a 46-bit unsigned fraction (value/2^46 in [0,1)) to fp32.
    function automatic [31:0] frac46_to_fp32(input [45:0] fbits);
        integer i, msb, e;
        reg [22:0] m23;
        /* verilator lint_off UNUSEDSIGNAL */
        reg [68:0] norm;                                         // only low 23 read
        /* verilator lint_on UNUSEDSIGNAL */
        begin
            msb = -1;
            for (i = 0; i < 46; i = i + 1) if (fbits[i]) msb = i;
            if (msb < 0) frac46_to_fp32 = 32'h0;                 // exactly 0
            else begin
                e = msb - 46;                                    // value exponent
                if (msb >= 23) norm = {23'd0, fbits} >> (msb - 23);
                else           norm = {23'd0, fbits} << (23 - msb);
                m23 = norm[22:0];
                frac46_to_fp32 = {1'b0, 8'(e + 127), m23};
            end
        end
    endfunction

    // cos/sin of theta in [0, pi/2] (fp32) via Horner Taylor; returns {cos,sin}.
    function automatic [63:0] cossin_quad(input [31:0] th);
        reg [31:0] u, sp, cp, sinv, cosv;
        begin
            u = fp32_mul(th, th);                                // theta^2
            // sin = th * (S0 + u(S1 + u(S2 + u(S3 + u*S4))))
            sp   = fp32_add(S3, fp32_mul(u, S4));
            sp   = fp32_add(S2, fp32_mul(u, sp));
            sp   = fp32_add(S1, fp32_mul(u, sp));
            sp   = fp32_add(S0, fp32_mul(u, sp));
            sinv = fp32_mul(th, sp);
            // cos = C0 + u(C1 + u(C2 + u(C3 + u*C4)))
            cp   = fp32_add(C3, fp32_mul(u, C4));
            cp   = fp32_add(C2, fp32_mul(u, cp));
            cp   = fp32_add(C1, fp32_mul(u, cp));
            cosv = fp32_add(C0, fp32_mul(u, cp));
            cossin_quad = {cosv, sinv};
        end
    endfunction

    // cos/sin from a Q48 turn fraction.  Bits [47:46] = quadrant; the remaining
    // 46 bits scale to theta in [0,pi/2].  Apply quadrant sign/swap.
    function automatic [63:0] cossin_turn(input [BFR-1:0] frac);
        reg [1:0]  q;
        reg [31:0] r, th, cs_lo, cs_hi, sinv, cosv, cf, sf;
        reg [63:0] cs;
        begin
            q  = frac[BFR-1 -: 2];                               // quadrant 0..3
            r  = frac46_to_fp32(frac[BFR-3 -: 46]);              // r in [0,1)
            th = fp32_mul(r, FP_PI_HALF);                        // theta in [0,pi/2)
            cs = cossin_quad(th);
            cosv = cs[63:32];
            sinv = cs[31:0];
            case (q)
                2'd0: begin cf = cosv;                  sf = sinv;                  end
                2'd1: begin cf = {~sinv[31], sinv[30:0]}; sf = cosv;                end
                2'd2: begin cf = {~cosv[31], cosv[30:0]}; sf = {~sinv[31], sinv[30:0]}; end
                default: begin cf = sinv;               sf = {~cosv[31], cosv[30:0]}; end
            endcase
            cs_hi = cf; cs_lo = sf;
            cossin_turn = {cs_hi, cs_lo};
        end
    endfunction

    //========================================================================
    // PER-LANE COMBINATIONAL DATAPATH
    //   lane j handles pair index (beat*LANES + j):
    //   phase = POS * F[pair];  frac_turns = phase[BFR-1:0]  (mod-1 = truncate);
    //   cos/sin from frac_turns; rotate (x0,x1) -> {y1,y0} bf16.
    //========================================================================
    reg  [POSW-1:0]      pos_q;                                   // captured POS
    reg  [BAW:0]         beat;

    integer              j;
    // phase_j high bits [PW-1:BFR] are the whole-turn count, intentionally
    // discarded (mod-1 reduction keeps only [BFR-1:0]).  Waive the unused-upper
    // lint, exactly as glm_fp.vh does for its wide intermediates.
    /* verilator lint_off UNUSEDSIGNAL */
    reg  [PW-1:0]        phase_j;
    /* verilator lint_on UNUSEDSIGNAL */
    reg  [BFR-1:0]       frac_j;
    reg  [63:0]          cs_j;
    reg  [31:0]          cos_j, sin_j;
    reg  [31:0]          x0_j, x1_j;
    reg  [31:0]          x0c, x1s, x0s, x1c;
    reg  [31:0]          y0_j, y1_j;
    reg  [LANES*32-1:0]  rot_beat;

    always @* begin
        rot_beat = {LANES*32{1'b0}};
        for (j = 0; j < LANES; j = j + 1) begin : LANE
            // pair index this lane processes (beat*LANES + j); turn ROM lookup
            phase_j = {{(PW-POSW){1'b0}}, pos_q}
                      * turn_rom[32'(beat) * LANES + j];
            frac_j  = phase_j[BFR-1:0];                          // (pos*invf/2pi) mod 1
            cs_j    = cossin_turn(frac_j);
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
    // FSM  (UNCHANGED)
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
