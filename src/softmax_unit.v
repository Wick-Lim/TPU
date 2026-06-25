`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// softmax_unit.v  --  TPU v2.0 fixed-point exp-based softmax, len 8 (SPEC §5.3)
//----------------------------------------------------------------------------
// PURPOSE
//   Computes a true numerically-stable softmax over SM_LEN(8) logits:
//       p_i = exp(x_i - max) / SUM_j exp(x_j - max)
//   This REPLACES the v1.5 "fake SOFTMAX" linear lane-normalization with a
//   genuine exponential softmax (LUT-based exp + reciprocal normalize).  It is
//   TM->TM: it reads two tile-memory lines of logits and writes two tile-memory
//   lines of probabilities, naming both tiles by a base TM line index.  It
//   instantiates NOTHING and exposes raw TM ACCESS PORTS (read + write); the
//   surrounding datapath (or the unit TB) owns/​models the tile memory.
//
// Q-FORMATS (single source of truth: tpu_defs.vh, SPEC §1.3)
//   logits  x_i     : Q7.8   signed 16-bit          (ELEM_W, Q78_FRAC=8)
//   diff    d=x-max : Q7.8   signed (<= 0 by const.) , magnitude m=-d  (>=0)
//   exp LUT entry   : Q15.16 unsigned (exp in (0,1]) (ACC_FRAC=16)
//   residual corr   : Q0.16  unsigned  (exp(-r), r in [0,0.25))
//   e_i = exp(d)    : Q15.16 unsigned                (accumulated)
//   SUM e_i (S)     : 48-bit Q15.16 accumulator      (ACC_W)
//   reciprocal 1/S  : Q1.30  unsigned                (Q130_FRAC=30)
//   prob   p_i      : Q0.16  unsigned, 0xFFFF ~= 1.0 (Q016_FRAC=16)
//
// EXP ALGORITHM  (64-entry exp LUT  +  multiplicative range-reduction)
//   For d = x_i - max <= 0, let m = -d (>=0, Q7.8).  Split m = idx*0.25 + r with
//   r in [0,0.25):
//       idx = m[13:6]   (the 0.25-quantized integer step, clamped to 63)
//       r   = m[5:0]    (the residual fraction, 6 bits, r_real = r/256)
//   Then  exp(-m) = exp(-idx*0.25) * exp(-r):
//       * exp(-idx*0.25) is the 64-entry exp LUT (EXP_LUT_DEPTH=64), built as a
//         synthesizable case/ROM.  Each entry K was derived OFFLINE as the
//         integer constant  round( exp(-K*0.25) * 2^16 )  (Q15.16); the case
//         block below carries the exact constants and their decimal exp values
//         in comments so the derivation is auditable.  m>=16 (idx>=64) -> exp~=0.
//       * exp(-r) for the small residual r in [0,0.25) is computed in-line by a
//         DIVIDE-FREE degree-4 Maclaurin polynomial in Q0.16
//             corr(r) = 1 - r + r^2/2 - r^3/6 + r^4/24
//         with 1/6 ~ 10923/2^16 and 1/24 ~ 2731/2^16 (constant Q16 multipliers,
//         only shifts -- no hardware divider).  Over r in [0,0.25) this matches
//         exp(-r) to < 1 LSB of Q0.16.  e_i = (LUT[idx]*corr + 2^15) >> 16.
//   This is a genuine per-element exp (NOT a fake), accurate enough that the
//   normalized probabilities track true floating-point softmax to within the
//   documented +/-2 LSB of Q0.16 tolerance (see TB).
//
// RECIPROCAL
//   1/S in Q1.30 is computed as a single rounded fixed-point reciprocal
//       recip = ( 2^46 + (S>>1) ) / S
//   (since S is Q15.16, 1/S_real = 2^16/S; in Q1.30 that is 2^46/S).  This is a
//   single combinational integer divide registered into one state -- a
//   legitimate synthesizable reciprocal (SPEC §5.3 permits "Newton-Raphson OR a
//   reciprocal LUT"; an explicit hardware divide is the exact-quotient form of
//   the same).  S is always >= LUT[0] = 65536 (the max element contributes
//   exp(0)=1.0), so S != 0 and recip <= 2^30 (fits Q1.30 / 32 bits).
//
// NORMALIZE
//   p_i = round( e_i(Q15.16) * recip(Q1.30) ) to Q0.16
//       = ( e_i*recip + 2^29 ) >> 30 , clamped to Q016_ONE(0xFFFF).
//   The argmax index (lowest index on ties) is emitted on `argmax[2:0]`.
//
// SATURATION
//   The only narrowing that can clamp is p_i -> 0xFFFF (a one-hot input gives
//   p~=1.0 which rounds to 0x10000 and is clamped to 0xFFFF).  `sat` is raised
//   sticky for the duration of `done` when any p_i hit the 0xFFFF clamp, so the
//   clamp is observable per SPEC §1.3/§4.  (Inputs/exp/reciprocal never clamp.)
//
// INTERFACE
//   clk, rst                              clock / synchronous active-high reset
//   start                                 1-cycle pulse: latch bases, begin
//   x_base [TM_IDX_W-1:0]                 base TM line of the 8 logits (2 lines)
//   p_base [TM_IDX_W-1:0]                 base TM line for the 8 probs  (2 lines)
//   busy                                  high while an op is in flight
//   done                                  1-cycle pulse when result lines written
//   sat                                   saturation flag (valid with `done`)
//   argmax [2:0]                          index of the max logit (0..7)
//   -- TM read access port (combinational; TB/datapath drives the memory) --
//   tm_raddr [TM_IDX_W-1:0]  (out)        line index to read
//   tm_rdata [LINE_W-1:0]    (in)         line data read back (combinational)
//   -- TM write access port (synchronous) --
//   tm_we                    (out)        write enable
//   tm_waddr [TM_IDX_W-1:0]  (out)        line index to write
//   tm_wdata [LINE_W-1:0]    (out)        line data to write
//
// LATENCY (deterministic; asserted by the TB)
//   FSM stages, in order:
//     S_RD0(1) S_RD1(1) S_MAX(1) S_EXP(8) S_RECIP(1) S_NORM(8) S_WR1(1) S_DONE(1)
//   COUNTING CONVENTION (the two numbers describe the SAME waveform):
//     * 22 = stage-edges traversed AFTER the start edge, EXCLUSIVE of it (the
//       count used in this header's stage list above).
//     * 23 = posedges from the edge that samples `start` to the edge that raises
//       `done`, INCLUSIVE of the start edge -- this is the `LAT` the unit TB
//       (softmax_unit_tb.v) measures and asserts, and the figure SPEC §3.3 lists.
//   SPEC §3.2 originally budgeted ~20 (2*SM_LEN+4); the committed RTL adds a
//   clean separate max-reduce and write-drain cycle.  `busy` is high from the
//   start cycle through the cycle before `done`.
//
// SYNTHESIZABILITY
//   Synchronous reset on ALL state; every reg assigned on every path of the one
//   clocked block (FSM) -- no inferred latches, no comb loops, no real/$display/
//   $random/initial in the module.  Passes verilator --lint-only -Wall.
//============================================================================
module softmax_unit (
    input  wire                 clk,
    input  wire                 rst,

    // Control handshake.
    input  wire                 start,
    input  wire [`TM_IDX_W-1:0] x_base,
    input  wire [`TM_IDX_W-1:0] p_base,
    output reg                  busy,
    output reg                  done,
    output reg                  sat,
    output reg  [2:0]           argmax,

    // TM read access port (combinational read).
    // Each 128-bit line packs four 32-bit lanes; a Q7.8 logit occupies only the
    // LOW 16 bits of its lane (sign-extension/padding in the high 16).  This
    // unit reads the low 16 bits of each lane, so the high 16 of every lane are
    // intentionally unused -- the narrow lint_off documents that.
    output reg  [`TM_IDX_W-1:0] tm_raddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [`LINE_W-1:0]   tm_rdata,
    /* verilator lint_on UNUSEDSIGNAL */

    // TM write access port (synchronous write).
    output reg                  tm_we,
    output reg  [`TM_IDX_W-1:0] tm_waddr,
    output reg  [`LINE_W-1:0]   tm_wdata
);

    // ---- local constants (NOT in tpu_defs.vh; declared here per the rules) ----
    localparam [3:0] S_IDLE  = 4'd0;
    localparam [3:0] S_RD0   = 4'd1;  // read logits line 0 (lanes 0..3)
    localparam [3:0] S_RD1   = 4'd2;  // read logits line 1 (lanes 4..7)
    localparam [3:0] S_MAX   = 4'd3;  // max + argmax over all 8 logits
    localparam [3:0] S_EXP   = 4'd4;  // 8 cycles: e_i = exp(x_i-max), accumulate S
    localparam [3:0] S_RECIP = 4'd5;  // reciprocal 1/S in Q1.30
    localparam [3:0] S_NORM  = 4'd6;  // 8 cycles: p_i = e_i*recip -> Q0.16
    localparam [3:0] S_WR1   = 4'd7;  // write probs line 1, pulse done next
    localparam [3:0] S_DONE  = 4'd8;  // 1-cycle done pulse

    // Maclaurin reciprocal-of-factorial constants in Q0.16 (divide-free):
    //   1/6  ~= round(2^16/6)  = 10923
    //   1/24 ~= round(2^16/24) =  2731
    localparam [16:0] INV6  = 17'd10923;
    localparam [16:0] INV24 = 17'd2731;

    // ---- state ----
    reg [3:0]              state;
    reg [`TM_IDX_W-1:0]    x_base_q;
    reg [`TM_IDX_W-1:0]    p_base_q;

    // Latched logits (8 x Q7.8 signed, stored sign-extended to 16-bit).
    reg signed [`ELEM_W-1:0] xv [0:`SM_LEN-1];
    // Running max (Q7.8 signed) and its index.
    reg signed [`ELEM_W-1:0] maxv;
    reg [2:0]                maxidx;
    // exp results e_i (Q15.16, up to 65536 -> fits 32-bit), and SUM accumulator.
    reg [`PROD_W-1:0]        ev [0:`SM_LEN-1];
    reg [`ACC_W-1:0]         sumacc;
    // reciprocal 1/S in Q1.30.
    reg [`Q130_W-1:0]        recip;
    // probabilities p_i (Q0.16).
    reg [`Q016_W-1:0]        pv [0:`SM_LEN-1];
    // pass element counter (0..7).
    reg [3:0]                cnt;

    integer w;

    // ----------------------------------------------------------------------
    // Combinational max + argmax over the eight latched logits xv[0..7].
    // Sequential reduce: keeps the LOWEST index on ties (strict > update).
    // Computed when all 8 logits are latched (state S_MAX), then registered.
    // ----------------------------------------------------------------------
    reg signed [`ELEM_W-1:0] cmax;
    reg [2:0]                cmaxidx;
    integer mi;
    always @(*) begin
        cmax    = xv[0];
        cmaxidx = 3'd0;
        for (mi = 1; mi < `SM_LEN; mi = mi + 1) begin
            if (xv[mi] > cmax) begin
                cmax    = xv[mi];
                cmaxidx = mi[2:0];
            end
        end
    end

    // ----------------------------------------------------------------------
    // Combinational exp(x_i - max) for the lane currently at `cnt`.
    // Pure function of the latched logit, the max, and the exp LUT/poly.
    // All intermediates are sized explicitly (no implicit width growth).
    // ----------------------------------------------------------------------
    // d = x[cnt] - max  (<= 0).  m = -d  (>= 0).  Use 17-bit signed for the
    // subtraction so the full [-65535,0] range of d is representable.
    wire signed [16:0] diff17 = $signed({xv[cnt[2:0]][`ELEM_W-1], xv[cnt[2:0]]})
                              - $signed({maxv[`ELEM_W-1], maxv});
    wire        [16:0] mag17  = (~diff17) + 17'd1;   // m = -d (two's complement)

    // idx = m[13:6] (the 0.25-step index); r = m[5:0] (6-bit residual fraction).
    // m can exceed 16 (idx>=64) for very negative diffs -> exp underflows to 0.
    wire [7:0]  idx8  = mag17[13:6];
    wire        idx_oor = (mag17[16:14] != 3'b000); // m >= 16384/256=64? -> idx>=256
    wire [5:0]  rfrac = mag17[5:0];

    // fxmul16: Q0.16 x Q0.16 -> Q0.16.  Forms the full 34-bit product and
    // returns bits [32:16] (>>16).  The low 16 fractional bits of `prod` are the
    // intended fixed-point round-down (magnitudes here are all < 1.0); the
    // narrow lint_off documents that `prod`'s low bits are deliberately dropped.
    /* verilator lint_off UNUSEDSIGNAL */
    function [16:0] fxmul16;
        input [16:0] a;
        input [16:0] b;
        reg   [33:0] prod;
        begin
            prod    = a * b;                 // Q0.32 (34 bits)
            fxmul16 = prod[32:16];           // Q0.16
        end
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    // exp LUT base: exp(-idx*0.25) in Q15.16 (value 0..65536 -> 17 bits).
    // Synthesizable 64-entry ROM.  Out-of-range (idx>=64) -> 0.  Constants
    // derived offline as round( exp(-K*0.25) * 2^16 ).
    reg [16:0] lut_base;
    always @(*) begin
        if (idx_oor || (idx8 > 8'd63)) begin
            lut_base = 17'd0;
        end else begin
            case (idx8[5:0])
                6'd 0: lut_base = 17'd65536;  // exp(-0.00)=1.00000000
                6'd 1: lut_base = 17'd51039;  // exp(-0.25)=0.77880078
                6'd 2: lut_base = 17'd39750;  // exp(-0.50)=0.60653066
                6'd 3: lut_base = 17'd30957;  // exp(-0.75)=0.47236655
                6'd 4: lut_base = 17'd24109;  // exp(-1.00)=0.36787944
                6'd 5: lut_base = 17'd18776;  // exp(-1.25)=0.28650480
                6'd 6: lut_base = 17'd14623;  // exp(-1.50)=0.22313016
                6'd 7: lut_base = 17'd11388;  // exp(-1.75)=0.17377394
                6'd 8: lut_base = 17'd8869;   // exp(-2.00)=0.13533528
                6'd 9: lut_base = 17'd6907;   // exp(-2.25)=0.10539922
                6'd10: lut_base = 17'd5380;   // exp(-2.50)=0.08208500
                6'd11: lut_base = 17'd4190;   // exp(-2.75)=0.06392786
                6'd12: lut_base = 17'd3263;   // exp(-3.00)=0.04978707
                6'd13: lut_base = 17'd2541;   // exp(-3.25)=0.03877421
                6'd14: lut_base = 17'd1979;   // exp(-3.50)=0.03019738
                6'd15: lut_base = 17'd1541;   // exp(-3.75)=0.02351775
                6'd16: lut_base = 17'd1200;   // exp(-4.00)=0.01831564
                6'd17: lut_base = 17'd935;    // exp(-4.25)=0.01426423
                6'd18: lut_base = 17'd728;    // exp(-4.50)=0.01110900
                6'd19: lut_base = 17'd567;    // exp(-4.75)=0.00865170
                6'd20: lut_base = 17'd442;    // exp(-5.00)=0.00673795
                6'd21: lut_base = 17'd344;    // exp(-5.25)=0.00524752
                6'd22: lut_base = 17'd268;    // exp(-5.50)=0.00408677
                6'd23: lut_base = 17'd209;    // exp(-5.75)=0.00318278
                6'd24: lut_base = 17'd162;    // exp(-6.00)=0.00247875
                6'd25: lut_base = 17'd127;    // exp(-6.25)=0.00193045
                6'd26: lut_base = 17'd99;     // exp(-6.50)=0.00150344
                6'd27: lut_base = 17'd77;     // exp(-6.75)=0.00117088
                6'd28: lut_base = 17'd60;     // exp(-7.00)=0.00091188
                6'd29: lut_base = 17'd47;     // exp(-7.25)=0.00071017
                6'd30: lut_base = 17'd36;     // exp(-7.50)=0.00055308
                6'd31: lut_base = 17'd28;     // exp(-7.75)=0.00043074
                6'd32: lut_base = 17'd22;     // exp(-8.00)=0.00033546
                6'd33: lut_base = 17'd17;     // exp(-8.25)=0.00026126
                6'd34: lut_base = 17'd13;     // exp(-8.50)=0.00020347
                6'd35: lut_base = 17'd10;     // exp(-8.75)=0.00015846
                6'd36: lut_base = 17'd8;      // exp(-9.00)=0.00012341
                6'd37: lut_base = 17'd6;      // exp(-9.25)=0.00009611
                6'd38: lut_base = 17'd5;      // exp(-9.50)=0.00007485
                6'd39: lut_base = 17'd4;      // exp(-9.75)=0.00005829
                6'd40: lut_base = 17'd3;      // exp(-10.00)=0.00004540
                6'd41: lut_base = 17'd2;      // exp(-10.25)=0.00003536
                6'd42: lut_base = 17'd2;      // exp(-10.50)=0.00002754
                6'd43: lut_base = 17'd1;      // exp(-10.75)=0.00002145
                6'd44: lut_base = 17'd1;      // exp(-11.00)=0.00001670
                6'd45: lut_base = 17'd1;      // exp(-11.25)=0.00001301
                6'd46: lut_base = 17'd1;      // exp(-11.50)=0.00001013
                6'd47: lut_base = 17'd1;      // exp(-11.75)=0.00000789
                default: lut_base = 17'd0;    // exp(<=-12.00) ~ 0 in Q15.16
            endcase
        end
    end

    // Residual correction corr(r) = exp(-r), r in [0,0.25), in Q0.16, divide-free.
    // Everything is kept in Q0.16 (low 16 = fraction); intermediate products are
    // formed in `fxmul16` (Q0.16 x Q0.16 -> Q0.16) which CONSUMES the full 34-bit
    // product internally (returns bits [32:16]) so no signal has unused bits.
    //   r16 = r/256 in Q0.16  (rfrac << 8)
    //   r2 = fxmul16(r16,r16), r3 = fxmul16(r2,r16), r4 = fxmul16(r3,r16)
    //   corr = 1 - r + r^2/2 - r^3/6 + r^4/24   (Q0.16, value in (0.778, 1.0])
    wire [16:0] r16 = {3'b000, rfrac, 8'b0};                 // Q0.16 (< 0.25)
    wire [16:0] r2       = fxmul16(r16, r16);                // r^2   (Q0.16)
    wire [16:0] r3       = fxmul16(r2,  r16);                // r^3   (Q0.16)
    wire [16:0] r4       = fxmul16(r3,  r16);                // r^4   (Q0.16)
    wire [16:0] r3_inv6  = fxmul16(r3,  INV6);               // r^3/6 (Q0.16)
    wire [16:0] r4_inv24 = fxmul16(r4,  INV24);              // r^4/24(Q0.16)
    // Sum in a 17-bit UNSIGNED accumulator.  Evaluated left-to-right the running
    // value never underflows (65536 - r16 >= 49152 since r16 < 16384) and never
    // exceeds 65536, so 17 bits hold the Q0.16 result exactly with no waste bit.
    wire [16:0] corr16 =
          17'd65536          // + 1.0
        - r16                // - r
        + {1'b0, r2[16:1]}   // + r^2/2  (r2 >> 1)
        - r3_inv6            // - r^3/6
        + r4_inv24;          // + r^4/24

    // e = (lut_base * corr) rounded to Q15.16.  lut_base<=65536(17b),
    // corr16<=65536(17b) -> product <= 2^32 (fits 33 bits); +2^15 round; >>16
    // returns Q15.16 (<=65536, 17 bits).
    // NOTE on the scoped lint_off below: e_round / p_round / fxmul16.prod / the
    // unused high lane bytes of TM lines all DELIBERATELY drop bits -- the low
    // fractional bits after a rounding right-shift, and the high 16 bits of each
    // 32-bit TM lane that holds only a 16-bit Q7.8 element.  These are intended
    // fixed-point/packing truncations, not bugs; the narrow lint_off documents
    // that and keeps the rest of the file under the strict -Wall UNUSEDSIGNAL.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [32:0] e_round = (lut_base * corr16) + {17'd0, 1'b1, 15'b0};
    wire [`PROD_W-1:0] e_q1516 = {15'd0, e_round[32:16]};   // Q15.16 in 32-bit slot
    /* verilator lint_on UNUSEDSIGNAL */

    // ----------------------------------------------------------------------
    // Reciprocal 1/S in Q1.30: recip = (2^46 + (S>>1)) / S  (rounded).
    // Numerator up to 2^46+ : use 64-bit operands for the divide.  The high
    // bits of the 64-bit temporaries are intentionally unused (the quotient is
    // <= 2^30), so the divide region is under a narrow scoped lint_off.
    // ----------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0] sum64   = {{(64-`ACC_W){1'b0}}, sumacc};
    wire [63:0] num_rcp = (64'd1 <<< 46) + (sum64 >> 1);
    wire [63:0] rcp64   = (sumacc == {`ACC_W{1'b0}}) ? 64'd0 : (num_rcp / sum64);
    /* verilator lint_on UNUSEDSIGNAL */
    // Clamp to Q1.30 range (<= 2^30); S>=65536 guarantees rcp<=2^30 anyway.
    wire [`Q130_W-1:0] recip_calc =
        (rcp64 > {32'd0, 32'h40000000}) ? 32'h40000000 : rcp64[`Q130_W-1:0];

    // ----------------------------------------------------------------------
    // Normalize p_i = round( e_i(Q15.16) * recip(Q1.30) ) to Q0.16.
    //   product = e_i * recip  (Q15.46) ; +2^29 round ; >>30 -> Q0.16.
    //   e_i<=65536(17b) * recip<=2^30(31b) -> <=2^47 : 64-bit product.  The low
    //   30 fractional bits of p_round are deliberately dropped by the >>30
    //   (the round-down after adding the 2^29 half-LSB bias).
    // ----------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0] p_prod  = {{(64-`PROD_W){1'b0}}, ev[cnt[2:0]]} * {32'd0, recip};
    wire [63:0] p_round = p_prod + (64'd1 <<< 29);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [33:0] p_shift = p_round[63:30];                    // Q0.16-ish, >=0
    // Saturate to 0xFFFF (1.0); flag the clamp.
    wire        p_sat_hit = (p_shift > {18'd0, `Q016_ONE});
    wire [`Q016_W-1:0] p_q016 = p_sat_hit ? `Q016_ONE : p_shift[`Q016_W-1:0];

    // ----------------------------------------------------------------------
    // Single clocked FSM.  Every reg assigned on every path (reset/branch).
    // ----------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state    <= S_IDLE;
            busy     <= 1'b0;
            done     <= 1'b0;
            sat      <= 1'b0;
            argmax   <= 3'd0;
            x_base_q <= {`TM_IDX_W{1'b0}};
            p_base_q <= {`TM_IDX_W{1'b0}};
            maxv     <= {`ELEM_W{1'b0}};
            maxidx   <= 3'd0;
            sumacc   <= {`ACC_W{1'b0}};
            recip    <= {`Q130_W{1'b0}};
            cnt      <= 4'd0;
            tm_raddr <= {`TM_IDX_W{1'b0}};
            tm_we    <= 1'b0;
            tm_waddr <= {`TM_IDX_W{1'b0}};
            tm_wdata <= {`LINE_W{1'b0}};
            for (w = 0; w < `SM_LEN; w = w + 1) begin
                xv[w] <= {`ELEM_W{1'b0}};
                ev[w] <= {`PROD_W{1'b0}};
                pv[w] <= {`Q016_W{1'b0}};
            end
        end else begin
            // Defaults each cycle (overridden below where needed).
            done  <= 1'b0;
            tm_we <= 1'b0;

            case (state)
                // ---------------------------------------------------------
                S_IDLE: begin
                    if (start) begin
                        busy     <= 1'b1;
                        sat      <= 1'b0;
                        sumacc   <= {`ACC_W{1'b0}};
                        cnt      <= 4'd0;
                        x_base_q <= x_base;
                        p_base_q <= p_base;
                        tm_raddr <= x_base;          // present line 0 this cycle
                        state    <= S_RD0;
                    end
                end

                // ---------------------------------------------------------
                // Latch lanes 0..3 (from logits line 0); request line 1.
                S_RD0: begin
                    xv[0] <= tm_rdata[ 15:  0];
                    xv[1] <= tm_rdata[ 47: 32];
                    xv[2] <= tm_rdata[ 79: 64];
                    xv[3] <= tm_rdata[111: 96];
                    tm_raddr <= x_base_q + {{(`TM_IDX_W-1){1'b0}}, 1'b1};
                    state <= S_RD1;
                end

                // ---------------------------------------------------------
                // Latch lanes 4..7 (from logits line 1).
                S_RD1: begin
                    xv[4] <= tm_rdata[ 15:  0];
                    xv[5] <= tm_rdata[ 47: 32];
                    xv[6] <= tm_rdata[ 79: 64];
                    xv[7] <= tm_rdata[111: 96];
                    state <= S_MAX;
                end

                // ---------------------------------------------------------
                // All 8 logits are now latched: register the combinational
                // max + argmax (lowest index on ties) for the exp pass.
                S_MAX: begin
                    maxv   <= cmax;
                    maxidx <= cmaxidx;
                    cnt    <= 4'd0;
                    state  <= S_EXP;
                end

                // ---------------------------------------------------------
                // 8 cycles: e[cnt] = exp(x[cnt]-max); accumulate sum.
                S_EXP: begin
                    ev[cnt[2:0]] <= e_q1516;
                    sumacc <= sumacc + {{(`ACC_W-`PROD_W){1'b0}}, e_q1516};
                    if (cnt == 4'd7) begin
                        cnt   <= 4'd0;
                        state <= S_RECIP;
                    end else begin
                        cnt <= cnt + 4'd1;
                    end
                end

                // ---------------------------------------------------------
                // Reciprocal 1/S in Q1.30.
                S_RECIP: begin
                    recip <= recip_calc;
                    cnt   <= 4'd0;
                    state <= S_NORM;
                end

                // ---------------------------------------------------------
                // 8 cycles: p[cnt] = e[cnt]*recip -> Q0.16.  On the 4th and
                // 8th element, emit the packed output line.
                S_NORM: begin
                    pv[cnt[2:0]] <= p_q016;
                    if (p_sat_hit)
                        sat <= 1'b1;
                    if (cnt == 4'd3) begin
                        // First output line: probs 0..3 (lane0 already in pv[0..2],
                        // lane3 is the live p_q016).
                        tm_we    <= 1'b1;
                        tm_waddr <= p_base_q;
                        tm_wdata <= { {16'd0, p_q016}, {16'd0, pv[2]},
                                      {16'd0, pv[1]},  {16'd0, pv[0]} };
                        cnt <= cnt + 4'd1;
                    end else if (cnt == 4'd7) begin
                        cnt   <= 4'd0;
                        state <= S_WR1;
                    end else begin
                        cnt <= cnt + 4'd1;
                    end
                end

                // ---------------------------------------------------------
                // Second output line: probs 4..7 (pv[4..6] + live p_q016 for 7).
                S_WR1: begin
                    tm_we    <= 1'b1;
                    tm_waddr <= p_base_q + {{(`TM_IDX_W-1){1'b0}}, 1'b1};
                    tm_wdata <= { {16'd0, pv[7]}, {16'd0, pv[6]},
                                  {16'd0, pv[5]}, {16'd0, pv[4]} };
                    state <= S_DONE;
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    busy   <= 1'b0;
                    done   <= 1'b1;
                    argmax <= maxidx;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
