`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// softmax_unit.v  --  TPU v2.0 fixed-point exp-based softmax, LEN-generic
//                     (default LEN=`SM_LEN=8) (SPEC §5.3)
//----------------------------------------------------------------------------
// PURPOSE
//   Computes a true numerically-stable softmax over LEN logits:
//       p_i = exp(x_i - max) / SUM_j exp(x_j - max)
//   This REPLACES the v1.5 "fake SOFTMAX" linear lane-normalization with a
//   genuine exponential softmax (LUT-based exp + reciprocal normalize).  It is
//   TM->TM: it reads NLINES tile-memory lines of logits and writes NLINES
//   tile-memory lines of probabilities, naming both tiles by a base TM line
//   index.  It instantiates NOTHING and exposes raw TM ACCESS PORTS (read +
//   write); the surrounding datapath (or the unit TB) owns/models the tile
//   memory.
//
// PARAMETERIZATION (this file is LEN-generic)
//   parameter integer LEN  : the softmax vector length.  Defaults to `SM_LEN
//                            (8) so the module is BYTE-IDENTICAL in behavior to
//                            the committed len-8 unit at its default, and the
//                            attention_unit reuse (which instantiates the
//                            DEFAULT) is unaffected.
//   Softmax is a 1-D reduction, so it is naturally length-generic.  The vector
//   spans NLINES = ceil(LEN/LINE_LANES) TM lines (each line packs LINE_LANES=4
//   lanes of one element each).  Everything below -- the read loop, max+argmax
//   reduce, exp-LUT pass, reciprocal, normalize loop, and write loop -- runs
//   over LEN with parameter-derived index/counter widths ($clog2) and
//   line-span counts (NLINES); there are no size-specific bit tricks.
//
//   SUPPORTED RANGE:  2 <= LEN <= 8 * LINE_LANES = 32.
//     * Lower bound 2: a length-1 softmax is degenerate (p=1.0).
//     * Upper bound 32: the read/write loops address NLINES = ceil(LEN/4) TM
//       lines from the base; with TM_LINES=32 a comfortable in-range envelope
//       is LEN<=32 (NLINES<=8 lines).  LEN need NOT be a multiple of
//       LINE_LANES (the last line is partially populated; unused lanes are
//       never read/written meaningfully).
//     * The `argmax` STATUS PORT is fixed 3-bit (architectural status field,
//       SPEC §4 carries a 0..7 value).  argmax is exact for LEN<=8; for LEN>8
//       the reported index is the true argmax truncated to its low 3 bits
//       (the port width is an architectural invariant and is NOT widened).
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
// RECIPROCAL  (PIPELINED -- multi-cycle SEQUENTIAL divider)
//   1/S in Q1.30 is the rounded fixed-point reciprocal
//       recip = ( 2^46 + (S>>1) ) / S
//   (since S is Q15.16, 1/S_real = 2^16/S; in Q1.30 that is 2^46/S).  This used
//   to be a SINGLE-CYCLE combinational 64-bit integer divide -- the MEASURED
//   critical path (PPA.md §3.1, softmax routed at only ~3.4 MHz; the long CCU2C
//   ripple divider was the #1 worst path).  It is now a MULTI-CYCLE radix-2
//   RESTORING sequential unsigned divider that computes the SAME integer
//   quotient one bit per cycle, dramatically shortening the per-cycle path at
//   the cost of DIV_CYCLES added LATENCY (the probabilities are BIT-IDENTICAL:
//   integer division is exact, and a throwaway probe confirmed the radix-2
//   restoring quotient == Verilog "/" over the full operand range, sum64 in
//   [65536, 32*65536], num_rcp = 2^46 + sum64/2).  The S==0 guard (recip=0) is
//   preserved (div_zero latched from sumacc==0).  S is always >= LUT[0]=65536
//   (the max element contributes exp(0)=1.0), so S != 0 and recip <= 2^30 (fits
//   Q1.30 / 32 bits).  SPEC §5.3 permits "Newton-Raphson OR a reciprocal LUT";
//   an explicit sequential hardware divide is the exact-quotient form.
//   DIVIDER GEOMETRY:  DIV_W = 48-bit dividend/quotient, DIV_CYCLES = 48.  The
//   divide spans S_RECIP (1 operand-latch cycle) + S_DIV (DIV_W iterations).
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
//   x_base [TM_IDX_W-1:0]                 base TM line of the LEN logits (NLINES)
//   p_base [TM_IDX_W-1:0]                 base TM line for the LEN probs (NLINES)
//   busy                                  high while an op is in flight
//   done                                  1-cycle pulse when result lines written
//   sat                                   saturation flag (valid with `done`)
//   argmax [2:0]                          index of the max logit (low 3 bits)
//   -- TM read access port (combinational; TB/datapath drives the memory) --
//   tm_raddr [TM_IDX_W-1:0]  (out)        line index to read
//   tm_rdata [LINE_W-1:0]    (in)         line data read back (combinational)
//   -- TM write access port (synchronous) --
//   tm_we                    (out)        write enable
//   tm_waddr [TM_IDX_W-1:0]  (out)        line index to write
//   tm_wdata [LINE_W-1:0]    (out)        line data to write
//
// LATENCY (deterministic; asserted by the TB)
//   FSM stages, in order (the reciprocal is now a MULTI-CYCLE divide):
//     S_RD(NLINES) S_MAX(1) S_EXP(LEN) S_RECIP(1) S_DIV(DIV_CYCLES)
//     S_NORM(LEN) S_WR(1) S_DONE(1)
//   The NORM pass writes each FULLY-PROBED output line in the same cycle its
//   last lane is produced (the historic "write line 0 during S_NORM cnt==3"
//   trick, generalized to every line); the FINAL line is drained in the
//   dedicated S_WR cycle.  Generic start->done latency (posedges from the start
//   edge to the done edge, inclusive of both, == the TB's `LAT`):
//       LAT = 1 + NLINES + 1 + LEN + 1 + DIV_CYCLES + LEN + 1 + 1
//           = 5 + NLINES + 2*LEN + DIV_CYCLES
//   The single-cycle S_RECIP divide became S_RECIP(1 operand-latch) +
//   S_DIV(DIV_CYCLES iterations), so the divide region grew from 1 -> 1+
//   DIV_CYCLES cycles, i.e. the closed form gained exactly DIV_CYCLES vs. the
//   old  5 + NLINES + 2*LEN.  With DIV_CYCLES = DIV_W = 48:
//       LAT = 53 + NLINES + 2*LEN.
//   COUNTING CONVENTION (the two numbers describe the SAME waveform):
//     * (LAT-1) = stage-edges traversed AFTER the start edge, EXCLUSIVE of it.
//     * LAT     = posedges from the edge that samples `start` to the edge that
//       raises `done`, INCLUSIVE of the start edge -- this is what the unit TB
//       measures and asserts.
//   For the DEFAULT LEN=8 (NLINES=2):  LAT = 53 + 2 + 16 = 71  (was 23; the
//   probabilities/argmax/sat are UNCHANGED -- only the cycle-accurate latency
//   grew by DIV_CYCLES=48).  `busy` is high from the start cycle through the
//   cycle before `done`.
//
// SYNTHESIZABILITY
//   Synchronous reset on ALL state; every reg assigned on every path of the one
//   clocked block (FSM) -- no inferred latches, no comb loops, no real/$display/
//   $random/initial in the module.  Passes verilator --lint-only -Wall.
//============================================================================
module softmax_unit #(
    // Vector length.  DEFAULTS to `SM_LEN so behavior is byte-identical to the
    // committed len-8 unit (and the attention_unit reuse) at its default.
    parameter integer LEN = `SM_LEN
) (
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
    // Each 128-bit line packs LINE_LANES (4) 32-bit lanes; a Q7.8 logit occupies
    // only the LOW 16 bits of its lane (sign-extension/padding in the high 16).
    // This unit reads the low 16 bits of each lane, so the high 16 of every lane
    // are intentionally unused -- the narrow lint_off documents that.
    output reg  [`TM_IDX_W-1:0] tm_raddr,
    /* verilator lint_off UNUSEDSIGNAL */
    input  wire [`LINE_W-1:0]   tm_rdata,
    /* verilator lint_on UNUSEDSIGNAL */

    // TM write access port (synchronous write).
    output reg                  tm_we,
    output reg  [`TM_IDX_W-1:0] tm_waddr,
    output reg  [`LINE_W-1:0]   tm_wdata
);

    // ---- parameter-derived geometry (replaces the size-specific tricks) ----
    // NLINES   : TM lines the LEN-vector spans = ceil(LEN/LINE_LANES).
    // IDX_W    : bits to hold an element index 0..LEN-1.
    // LINE_W_  : bits to hold a line index 0..NLINES-1 (for the line counters).
    // CNT_W    : bits for the element pass counter (must hold LEN, i.e. 0..LEN).
    localparam integer NLANES = `LINE_LANES;                 // 4 lanes / line
    localparam integer NLINES = (LEN + NLANES - 1) / NLANES; // ceil(LEN/4)
    localparam integer IDX_W  = (LEN  > 1) ? $clog2(LEN)    : 1;
    localparam integer LINE_W_= (NLINES > 1) ? $clog2(NLINES) : 1;
    localparam integer CNT_W  = $clog2(LEN + 1);             // holds 0..LEN

    // ---- local constants (NOT in tpu_defs.vh; declared here per the rules) ----
    localparam [3:0] S_IDLE  = 4'd0;
    localparam [3:0] S_RD    = 4'd1;  // read logits lines 0..NLINES-1 (NLINES cyc)
    localparam [3:0] S_MAX   = 4'd2;  // max + argmax over all LEN logits
    localparam [3:0] S_EXP   = 4'd3;  // LEN cycles: e_i = exp(x_i-max), accum S
    localparam [3:0] S_RECIP = 4'd4;  // latch divider operands (1 cyc), -> S_DIV
    localparam [3:0] S_DIV   = 4'd8;  // sequential reciprocal divide (DIV_W cyc)
    localparam [3:0] S_NORM  = 4'd5;  // LEN cycles: p_i = e_i*recip -> Q0.16
    localparam [3:0] S_WR    = 4'd6;  // write the FINAL probs line, done next
    localparam [3:0] S_DONE  = 4'd7;  // 1-cycle done pulse

    // ---- sequential reciprocal divider geometry --------------------------
    // The reciprocal recip = num_rcp / sum64 was a single-cycle combinational
    // 64-bit divide (the measured critical path, PPA.md §3.1 ~3.4 MHz).  It is
    // now a MULTI-CYCLE radix-2 RESTORING unsigned divider producing the SAME
    // integer quotient bit-for-bit, one quotient bit per cycle.  The dividend
    // num_rcp = 2^46 + (S>>1) is <= 2^47 (47 significant bits); a 48-bit-wide
    // divider (48 iterations) covers the full operand range exactly (verified
    // by a throwaway probe against the Verilog "/" over the real operand ranges:
    // sum64 in [65536, 32*65536], num_rcp = 2^46 + sum64/2 -- IDENTICAL for all).
    //   DIV_W      : dividend / quotient register width and the iteration count.
    //   DIV_CYCLES : extra latency added vs. the old 1-cycle divide.  The divide
    //                region now spans S_RECIP(1 setup) + S_DIV(DIV_W iterations);
    //                it replaced the old single S_RECIP cycle, so the net added
    //                latency is DIV_CYCLES = (1 + DIV_W) - 1 = DIV_W.
    localparam integer DIV_W      = 48;     // 48-bit radix-2 restoring divider
    // DIV_CYCLES documents the latency the divide adds (used by the TBs' LAT
    // closed form); it is not referenced elsewhere in the RTL, so it is lint_off
    // for UNUSEDPARAM (it stays here as the single source of the cycle figure).
    /* verilator lint_off UNUSEDPARAM */
    localparam integer DIV_CYCLES = DIV_W;  // +48 cycles vs. the old 1-cyc divide
    /* verilator lint_on UNUSEDPARAM */

    // Maclaurin reciprocal-of-factorial constants in Q0.16 (divide-free):
    //   1/6  ~= round(2^16/6)  = 10923
    //   1/24 ~= round(2^16/24) =  2731
    localparam [16:0] INV6  = 17'd10923;
    localparam [16:0] INV24 = 17'd2731;

    // ---- state ----
    reg [3:0]              state;
    reg [`TM_IDX_W-1:0]    x_base_q;
    reg [`TM_IDX_W-1:0]    p_base_q;

    // Latched logits (LEN x Q7.8 signed, stored sign-extended to 16-bit).
    reg signed [`ELEM_W-1:0] xv [0:LEN-1];
    // Running max (Q7.8 signed) and its index.
    reg signed [`ELEM_W-1:0] maxv;
    reg [IDX_W-1:0]          maxidx;
    // exp results e_i (Q15.16, up to 65536 -> fits 32-bit), and SUM accumulator.
    reg [`PROD_W-1:0]        ev [0:LEN-1];
    reg [`ACC_W-1:0]         sumacc;
    // reciprocal 1/S in Q1.30.
    reg [`Q130_W-1:0]        recip;
    // probabilities p_i (Q0.16).
    reg [`Q016_W-1:0]        pv [0:LEN-1];
    // element pass counter (0..LEN-1) and line counter for the RD/WR loops.
    reg [CNT_W-1:0]          cnt;
    reg [LINE_W_-1:0]        lcnt;     // line index during the read loop

    // ---- sequential reciprocal divider state -----------------------------
    // div_dividend : the full DIV_W-bit dividend (latched num_rcp); the iteration
    //                shifts in dividend bits MSB-first via div_idx.
    // div_divisor  : the latched divisor (sum64).
    // div_quot     : accumulating quotient (one bit set per iteration).
    // div_rem      : partial remainder.  After each restoring step the remainder
    //                is < divisor <= 2^DIV_W and the dividend is < 2^DIV_W, so the
    //                registered remainder always fits in DIV_W bits; the trial
    //                subtract's extra sign bit lives only in the combinational
    //                wires below (div_rem_sh / div_sub).
    // div_idx      : current dividend bit index (DIV_W-1 down to 0); a 0..DIV_W
    //                counter (needs $clog2(DIV_W+1) bits).
    // div_zero     : sticky S==0 guard (sum64==0) latched at operand-latch time;
    //                forces recip=0 exactly as the old combinational guard did.
    localparam integer DIDX_W = $clog2(DIV_W + 1);     // holds 0..DIV_W
    reg [DIV_W-1:0]  div_dividend;
    reg [DIV_W-1:0]  div_divisor;
    reg [DIV_W-1:0]  div_quot;
    reg [DIV_W-1:0]  div_rem;
    reg [DIDX_W-1:0] div_idx;
    reg              div_zero;

    integer w;

    // ----------------------------------------------------------------------
    // Combinational max + argmax over the LEN latched logits xv[0..LEN-1].
    // Sequential reduce: keeps the LOWEST index on ties (strict > update).
    // Computed when all logits are latched (state S_MAX), then registered.
    // ----------------------------------------------------------------------
    reg signed [`ELEM_W-1:0] cmax;
    reg [IDX_W-1:0]          cmaxidx;
    integer mi;
    always @(*) begin
        cmax    = xv[0];
        cmaxidx = {IDX_W{1'b0}};
        for (mi = 1; mi < LEN; mi = mi + 1) begin
            if (xv[mi] > cmax) begin
                cmax    = xv[mi];
                cmaxidx = mi[IDX_W-1:0];
            end
        end
    end

    // ----------------------------------------------------------------------
    // Combinational exp(x_i - max) for the lane currently at `cnt`.
    // Pure function of the latched logit, the max, and the exp LUT/poly.
    // All intermediates are sized explicitly (no implicit width growth).
    // ----------------------------------------------------------------------
    // d = x[cnt] - max  (<= 0).  m = -d  (>= 0).  Use 17-bit signed for the
    // subtraction so the full [-65535,0] range of d is representable.  `cnt`
    // indexes the element array; the low IDX_W bits select the lane.
    wire [IDX_W-1:0] xidx = cnt[IDX_W-1:0];
    wire signed [16:0] diff17 = $signed({xv[xidx][`ELEM_W-1], xv[xidx]})
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
    // Reciprocal 1/S in Q1.30: recip = (2^46 + (S>>1)) / S  (rounded), computed
    // by a MULTI-CYCLE radix-2 RESTORING sequential divider (was a single-cycle
    // combinational 64-bit "/", the measured critical path, PPA.md §3.1).
    //
    //   * The divide operands are formed combinationally here (DIV_W-wide):
    //       sum64   = S (the Q15.16 exp-sum accumulator), the DIVISOR.
    //       num_rcp = 2^46 + (S>>1), the DIVIDEND (since 1/S_real = 2^16/S, in
    //                 Q1.30 that is 2^46/S; +(S>>1) is the round-half bias).
    //     These wires are LATCHED into div_divisor/div_dividend in S_RECIP, then
    //     the FSM iterates the divider one quotient bit per cycle in S_DIV.
    //   * The S==0 guard (recip=0) is preserved: div_zero is latched from
    //     (sumacc==0) at operand-latch time; if set the quotient is forced 0.
    //   * Bit-exactness vs. the Verilog "/" was confirmed by a throwaway probe
    //     over the full operand range (sum64 in [65536, 32*65536], num_rcp =
    //     2^46 + sum64/2): the radix-2 restoring quotient is IDENTICAL for all.
    // The high bits of the DIV_W operands above the meaningful range are unused
    // (the quotient is <= 2^30), so the divide region is under a scoped lint_off.
    // ----------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */
    wire [DIV_W-1:0] sum64   = {{(DIV_W-`ACC_W){1'b0}}, sumacc};
    wire [DIV_W-1:0] num_rcp = (DIV_W'(1) <<< 46) + (sum64 >> 1);

    // The bit index being processed this cycle, as a DIDX_W-wide value:
    // (div_idx - 1), i.e. DIV_W-1 down to 0 over the DIV_W iterations.
    wire [DIDX_W-1:0] div_bit = div_idx - DIDX_W'(1);
    // One radix-2 restoring iteration (combinational): shift the working
    // remainder left, bring in the dividend bit at div_bit, trial-subtract the
    // divisor; if non-negative keep the subtract (quotient bit 1) else restore.
    wire [DIV_W-1:0] div_bitsel = div_dividend >> div_bit;
    // shifted remainder (DIV_W+1 bits) and the trial subtract (one sign bit).
    wire [DIV_W:0]   div_rem_sh = {div_rem, div_bitsel[0]};
    wire [DIV_W:0]   div_sub    = div_rem_sh - {1'b0, div_divisor};
    wire             div_qbit   = (div_sub[DIV_W] == 1'b0);   // rem_sh >= divisor
    // next registered remainder: always fits in DIV_W bits (kept value's top bit
    // is 0 -- restore keeps rem_sh<divisor, subtract keeps the non-negative sub).
    wire [DIV_W-1:0] div_rem_nx = div_qbit ? div_sub[DIV_W-1:0]
                                           : div_rem_sh[DIV_W-1:0];
    // quotient bit goes to position div_bit (MSB-first generation).
    wire [DIV_W-1:0] div_quot_nx =
        div_quot | (div_qbit ? (DIV_W'(1) << div_bit)
                             : {DIV_W{1'b0}});

    // recip_clamp(q): S==0-guard + Q1.30 clamp of a completed quotient `q`.
    //   * div_zero (sumacc==0) forces 0 -- the preserved S==0 guard.
    //   * clamp to Q1.30 range (<= 2^30); S>=65536 guarantees rcp<=2^30 anyway.
    // Applied to the freshly-completed quotient at the last divide iteration.
    function [`Q130_W-1:0] recip_clamp;
        input [DIV_W-1:0] q;
        begin
            if (div_zero)
                recip_clamp = {`Q130_W{1'b0}};
            else if (q > {{(DIV_W-32){1'b0}}, 32'h40000000})
                recip_clamp = 32'h40000000;
            else
                recip_clamp = q[`Q130_W-1:0];
        end
    endfunction
    /* verilator lint_on UNUSEDSIGNAL */

    // ----------------------------------------------------------------------
    // Normalize p_i = round( e_i(Q15.16) * recip(Q1.30) ) to Q0.16.
    //   product = e_i * recip  (Q15.46) ; +2^29 round ; >>30 -> Q0.16.
    //   e_i<=65536(17b) * recip<=2^30(31b) -> <=2^47 : 64-bit product.  The low
    //   30 fractional bits of p_round are deliberately dropped by the >>30
    //   (the round-down after adding the 2^29 half-LSB bias).
    // ----------------------------------------------------------------------
    /* verilator lint_off UNUSEDSIGNAL */
    wire [63:0] p_prod  = {{(64-`PROD_W){1'b0}}, ev[xidx]} * {32'd0, recip};
    wire [63:0] p_round = p_prod + (64'd1 <<< 29);
    /* verilator lint_on UNUSEDSIGNAL */
    wire [33:0] p_shift = p_round[63:30];                    // Q0.16-ish, >=0
    // Saturate to 0xFFFF (1.0); flag the clamp.
    wire        p_sat_hit = (p_shift > {18'd0, `Q016_ONE});
    wire [`Q016_W-1:0] p_q016 = p_sat_hit ? `Q016_ONE : p_shift[`Q016_W-1:0];

    // ----------------------------------------------------------------------
    // Explicit-width constants derived from LEN/NLINES/NLANES.  Building these
    // as sized localparams (rather than bare 32-bit integer literals) keeps the
    // comparisons/adds below width-clean under verilator -Wall.
    // ----------------------------------------------------------------------
    localparam [CNT_W-1:0]     LEN_M1    = CNT_W'(LEN - 1);      // last element idx
    localparam [IDX_W-1:0]     LANE_M1   = IDX_W'(NLANES - 1);   // last lane idx/line
    localparam [`TM_IDX_W-1:0] NLINES_M1 = `TM_IDX_W'(NLINES-1); // last line idx
    localparam integer         LASTBASE  = (NLINES - 1) * NLANES;// base elem last line

    // Which element index is currently the last lane of a line (so its line is
    // complete and can be flushed in the same NORM cycle): (cnt % NLANES)==last.
    wire [IDX_W-1:0]  lane_in_line  = xidx % NLANES[IDX_W-1:0];
    wire              line_complete = (lane_in_line == LANE_M1);
    // Is `cnt` the very last element (LEN-1)?  Its line is drained in S_WR.
    wire              is_last_elem  = (cnt == LEN_M1);

    // argmax index forced to EXACTLY 3 bits for the fixed-width status port,
    // independent of IDX_W (zero-extend if IDX_W<3, truncate if IDX_W>3).
    // Zero-extend maxidx to a fixed 3+IDX_W vector, then take the low 3 bits --
    // legal for any IDX_W (avoids an out-of-range part-select when IDX_W<3).
    // The high bits [IDX_W+2:3] are deliberately unused (the status port is the
    // architectural 3-bit field, SPEC §4); the narrow lint_off documents that.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [IDX_W+2:0]  maxidx_ext = {{3{1'b0}}, maxidx};
    /* verilator lint_on UNUSEDSIGNAL */
    wire [2:0]        maxidx3    = maxidx_ext[2:0];

    // ----------------------------------------------------------------------
    // Combinationally assemble the two output lines (LEN-generic NLANES packing).
    //   norm_wline : the line that ENDS at the current NORM element `cnt` (its
    //                last lane is the live p_q016; the earlier lanes are pv[]).
    //                Used for every line EXCEPT the final one (flushed in NORM).
    //   final_wline: the line containing element LEN-1 (all lanes from pv[]); a
    //                partial line writes 0 into unused high lanes.  Used in S_WR.
    // Each Q7.8 probability sits in the low ELEM_W (16) bits of its 32-bit lane;
    // the high 16 bits of each lane are intentionally 0 (the packing matches the
    // committed len-8 layout exactly).
    // ----------------------------------------------------------------------
    reg  [`LINE_W-1:0] norm_wline;
    reg  [`LINE_W-1:0] final_wline;
    integer kk;
    // base element index of the line that ends at `cnt` (== cnt - (NLANES-1)).
    wire [IDX_W-1:0] norm_base = xidx - LANE_M1;
    always @(*) begin
        norm_wline  = {`LINE_W{1'b0}};
        final_wline = {`LINE_W{1'b0}};
        for (kk = 0; kk < NLANES; kk = kk + 1) begin
            // current (just-completed) line: last lane is live p_q016, rest pv[].
            if (kk == (NLANES-1))
                norm_wline[(kk*`LANE_W) +: `LANE_W] = {16'd0, p_q016};
            else
                norm_wline[(kk*`LANE_W) +: `LANE_W] =
                    {16'd0, pv[norm_base + kk[IDX_W-1:0]]};
            // final line: every populated lane from pv[]; unused high lanes = 0.
            if ((LASTBASE + kk) < LEN)
                final_wline[(kk*`LANE_W) +: `LANE_W] =
                    {16'd0, pv[LASTBASE + kk]};
        end
    end

    // ----------------------------------------------------------------------
    // Single clocked FSM.  Every reg assigned on every path (reset/branch).
    // ----------------------------------------------------------------------
    integer k;            // generic lane loop variable in the clocked block

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
            maxidx   <= {IDX_W{1'b0}};
            sumacc   <= {`ACC_W{1'b0}};
            recip    <= {`Q130_W{1'b0}};
            cnt      <= {CNT_W{1'b0}};
            lcnt     <= {LINE_W_{1'b0}};
            div_dividend <= {DIV_W{1'b0}};
            div_divisor  <= {DIV_W{1'b0}};
            div_quot     <= {DIV_W{1'b0}};
            div_rem      <= {DIV_W{1'b0}};
            div_idx      <= {DIDX_W{1'b0}};
            div_zero     <= 1'b0;
            tm_raddr <= {`TM_IDX_W{1'b0}};
            tm_we    <= 1'b0;
            tm_waddr <= {`TM_IDX_W{1'b0}};
            tm_wdata <= {`LINE_W{1'b0}};
            for (w = 0; w < LEN; w = w + 1) begin
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
                        cnt      <= {CNT_W{1'b0}};
                        lcnt     <= {LINE_W_{1'b0}};
                        x_base_q <= x_base;
                        p_base_q <= p_base;
                        tm_raddr <= x_base;          // present line 0 this cycle
                        state    <= S_RD;
                    end
                end

                // ---------------------------------------------------------
                // Read loop: NLINES cycles.  On cycle `lcnt` the live tm_rdata is
                // logits line `lcnt` (lanes lcnt*4 .. lcnt*4+3); latch up to 4
                // lanes into xv[], guarding the partial final line by LEN.  Then
                // request the NEXT line (lcnt+1) so it is live next cycle.
                S_RD: begin
                    for (k = 0; k < NLANES; k = k + 1) begin
                        // global element index of lane k on this line
                        if ((lcnt * NLANES + k) < LEN)
                            xv[lcnt * NLANES + k] <=
                                tm_rdata[(k*`LANE_W) +: `ELEM_W];
                    end
                    if (lcnt == NLINES_M1[LINE_W_-1:0]) begin
                        // all lines latched -> reduce
                        state <= S_MAX;
                    end else begin
                        // next logits line (widen lcnt to the TM index width).
                        tm_raddr <= x_base_q +
                            {{(`TM_IDX_W-LINE_W_){1'b0}}, lcnt} +
                            {{(`TM_IDX_W-1){1'b0}}, 1'b1};
                        lcnt  <= lcnt + 1'b1;
                    end
                end

                // ---------------------------------------------------------
                // All logits are now latched: register the combinational
                // max + argmax (lowest index on ties) for the exp pass.
                S_MAX: begin
                    maxv   <= cmax;
                    maxidx <= cmaxidx;
                    cnt    <= {CNT_W{1'b0}};
                    state  <= S_EXP;
                end

                // ---------------------------------------------------------
                // LEN cycles: e[cnt] = exp(x[cnt]-max); accumulate sum.
                S_EXP: begin
                    ev[xidx] <= e_q1516;
                    sumacc <= sumacc + {{(`ACC_W-`PROD_W){1'b0}}, e_q1516};
                    if (cnt == LEN_M1) begin
                        cnt   <= {CNT_W{1'b0}};
                        state <= S_RECIP;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // ---------------------------------------------------------
                // Reciprocal 1/S in Q1.30 -- OPERAND LATCH (1 cycle).  Latch the
                // dividend/divisor and the S==0 guard, initialize the radix-2
                // restoring divider (quotient=0, remainder=0, bit index=DIV_W),
                // then iterate one quotient bit per cycle in S_DIV.  Also rewind
                // `lcnt` to 0 so the NORM pass numbers OUTPUT lines from p_base
                // (the read loop left lcnt at NLINES-1).
                S_RECIP: begin
                    div_dividend <= num_rcp;
                    div_divisor  <= sum64;
                    div_zero     <= (sumacc == {`ACC_W{1'b0}});
                    div_quot     <= {DIV_W{1'b0}};
                    div_rem      <= {DIV_W{1'b0}};
                    div_idx      <= DIDX_W'(DIV_W);   // process bits DIV_W-1 .. 0
                    cnt          <= {CNT_W{1'b0}};
                    lcnt         <= {LINE_W_{1'b0}};
                    state        <= S_DIV;
                end

                // ---------------------------------------------------------
                // Sequential reciprocal divide: DIV_W cycles, one quotient bit
                // per cycle (radix-2 restoring).  Each cycle processes dividend
                // bit (div_idx-1): shift remainder, trial-subtract, set the
                // quotient bit on a non-negative result, else restore.  On the
                // last iteration (div_idx==1) the full quotient is ready, so
                // register recip (clamped, S==0-guarded) and advance to S_NORM.
                S_DIV: begin
                    div_quot <= div_quot_nx;
                    div_rem  <= div_rem_nx;
                    div_idx  <= div_idx - DIDX_W'(1);
                    if (div_idx == DIDX_W'(1)) begin
                        // div_quot_nx holds the COMPLETE quotient this cycle
                        // (the LSB, bit 0, is generated on this last iteration);
                        // S==0-guard + Q1.30-clamp it into recip.
                        recip <= recip_clamp(div_quot_nx);
                        state <= S_NORM;
                    end
                end

                // ---------------------------------------------------------
                // LEN cycles: p[cnt] = e[cnt]*recip -> Q0.16.  Whenever `cnt`
                // completes a line (last lane) AND it is NOT the final element,
                // flush that line in the SAME cycle (lanes are pv[] + live
                // p_q016).  The FINAL line is drained in S_WR.
                S_NORM: begin
                    pv[xidx] <= p_q016;
                    if (p_sat_hit)
                        sat <= 1'b1;

                    // When the current element completes a line (and it is NOT
                    // the final element), flush that just-completed line in the
                    // SAME cycle: its packed form is `norm_wline` (pv[] lanes +
                    // the live p_q016 as the last lane).  `lcnt` numbers output
                    // lines from p_base.
                    if (line_complete && !is_last_elem) begin
                        tm_we    <= 1'b1;
                        tm_waddr <= p_base_q +
                            {{(`TM_IDX_W-LINE_W_){1'b0}}, lcnt};   // line idx lcnt
                        tm_wdata <= norm_wline;
                        lcnt     <= lcnt + 1'b1;
                    end

                    if (cnt == LEN_M1) begin
                        cnt   <= {CNT_W{1'b0}};
                        state <= S_WR;
                    end else begin
                        cnt <= cnt + 1'b1;
                    end
                end

                // ---------------------------------------------------------
                // Drain the FINAL output line (the line containing element
                // LEN-1).  Its lanes are entirely in pv[] now (the last lane,
                // pv[LEN-1], was written on the previous NORM cycle).  This line
                // may be PARTIAL (LEN not a multiple of NLANES): unused high
                // lanes are written as 0.  Base lane = (NLINES-1)*NLANES.
                S_WR: begin
                    tm_we    <= 1'b1;
                    tm_waddr <= p_base_q + NLINES_M1;   // last output line index
                    tm_wdata <= final_wline;
                    state    <= S_DONE;
                end

                // ---------------------------------------------------------
                S_DONE: begin
                    busy   <= 1'b0;
                    done   <= 1'b1;
                    // Emit the low 3 bits of the argmax index on the fixed
                    // 3-bit status port (exact for LEN<=8).  `maxidx3` zero-
                    // extends/truncates the IDX_W-wide index to exactly 3 bits
                    // so the part-select is legal for any IDX_W (e.g. IDX_W=2
                    // at LEN=4, IDX_W=4 at LEN=16).
                    argmax <= maxidx3;
                    state  <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
