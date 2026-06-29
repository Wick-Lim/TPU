`timescale 1ns/1ps
`include "glm_fp.vh"
`include "fp8_e4m3.vh"
`include "glm_fp_pipe_lat.vh"   // FP pipeline latencies (single source of truth)
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_matmul_fp8.v  --  GLM-5.2 FP8-NATIVE GEMM datapath  (ACCEL_GLM52 §6)
//                       a DROP-IN sibling of glm_matmul_pipe.v.
//----------------------------------------------------------------------------
// FUNCTION
//   C[M,N] = A[M,K] x W[K,N], the SAME math as glm_matmul_pipe, but computed in
//   the OFFICIAL GLM-5.2-FP8 numerics so the published zai-org/GLM-5.2-FP8
//   checkpoint runs with NO re-quantization:
//
//     * Weights W arrive already in FP8 E4M3 (8-bit) carrying a [BLK x BLK]
//       BLOCK scale: one bf16 dequant scale per 128x128 weight block (the
//       DeepSeek-V3 / GLM-5.2-FP8 weight_block_size=[128,128] layout).  Block
//       (out-block, K-block) covers BLK output channels x BLK contraction (K)
//       columns.  Each tile output column pj therefore carries ONE scale per
//       K-block bj: w_scale[pj][bj]  (the caller selects the block-row for pj).
//     * Activations A arrive bf16 (the SAME bf16 activation interface as
//       glm_matmul_pipe, for drop-in compatibility) and are DYNAMICALLY
//       quantized to FP8 E4M3 on the fly.  Each array ROW carries a per-vector
//       (per-token) power-of-two activation scale  a_shift[pi]  (a signed
//       integer): the activation is pre-scaled by 2^a_shift (a cheap EXPONENT
//       add -- NO multiplier) to land in E4M3's dynamic range, encoded to E4M3,
//       and the exact inverse 2^-a_shift is folded back at dequant.
//
// BLOCK-SCALED ACCUMULATION (the [128,128] scheme):
//   out[pi][pj] = 2^-a_shift[pi] *
//                 SUM over K-blocks bj of ( w_scale[pj][bj] *
//                    SUM_{k in Kblock bj} fp8(A[pi][k]*2^a_shift) * fp8(W[k][pj]) )
//   i.e. the fp8 products are INNER-accumulated within each 128-wide K-block;
//   each block partial is multiplied by that block's weight scale; the scaled
//   block partials are OUTER-accumulated across K-blocks; finally the per-token
//   2^-a_shift is undone (an exact exponent add) and the result is rounded to
//   bf16.  For K <= BLK there is exactly ONE K-block, so this reduces to a
//   single per-column weight scale.
//
// THE HARDWARE WIN (why FP8 over the fp32 baseline):
//   The per-term product is  fp8_mul(act_e4m3, wgt_e4m3)  --- a 4-bit x 4-bit
//   MANTISSA multiply (src/fp8_e4m3.vh), which synthesizes to the plentiful
//   Gowin GW2A-18 LUT, NOT to the scarce 24x24 DSP the fp32 datapath burns.
//   The only 24x24 fp32 multiplies left in the whole unit are the time-shared
//   dequant multipliers (block_partial * w_scale): O(NB) of them, walked once
//   over the NOUT outputs at the very end.
//
//----------------------------------------------------------------------------
// ACCUMULATOR CHOICE:  BLOCK FIXED-POINT (BFP), NOT a full-fp32 adder tree.
//   The OLD datapath accumulated each fp8 product in a FULL fp32 adder
//   (fp32_add_pipe, 24-bit mantissa, renormalize+round EVERY add), L-way
//   interleaved per K-block plus a per-bank fp32 reduction tree.  Those fp32
//   adders dominated the cell count (research: 'Ultra-Low Accumulation Precision
//   Inference with BFP', openreview Dzamphz35c; arXiv 2502.01070 / DeepSeek-V3
//   Hopper FP8 GEMM -- block-float accumulation, no per-add normalize).
//
//   KEY NUMERIC FACT exploited here: every E4M3 value is an exact integer
//   multiple of 2^-9 (normal (8+m)*2^(e-10), subnormal m*2^-9), so every fp8x
//   fp8 PRODUCT is an exact integer multiple of 2^-18, and is therefore EXACTLY
//   representable as a fixed-point integer with ACC_FRAC=18 fractional bits.
//   We accumulate the products of one 128-wide K-block into a NARROW signed
//   fixed-point register (ACC_W bits, weight 2^-ACC_FRAC per LSB) with a plain
//   integer add -- NO per-add mantissa normalize, NO rounding, NO interleave,
//   NO reduction tree.  The within-block sum is thus EXACT (<= the fp32 path's
//   error -- the fp32 path rounded every add).  At block end the fixed-point
//   accumulator is converted to fp32 ONCE (RNE) and handed to the SAME proven
//   dequant pass (block_partial * w_scale, fold, undo 2^-a_shift, bf16 round).
//
//   ACC_W / ACC_FRAC are PARAMETERS so the accumulator width is tunable.  With
//   ACC_FRAC=18 and ACC_W>=44 the within-block accumulation is BIT-EXACT for
//   any K<=BLK (max |block sum| < 128*448^2 < 2^43); narrowing ACC_FRAC trades
//   precision for area (the BFP/Hopper regime).  Default is the exact width.
//
//----------------------------------------------------------------------------
// PER-K-BLOCK FIXED-POINT ACCUMULATION (banked)
//   Each PE owns NB independent fixed-point accumulator BANKS (NB=ceil(KMAX/BLK)).
//   The fp8 product of beat k is converted to fixed-point combinationally, then
//   registered one stage; the next cycle it is integer-added into the bank for
//   its K-block (kblk = k/BLK).  A single-cycle integer accumulate has no
//   pipeline hazard (the running sum is available every cycle), so NO L-way
//   interleave and NO drain/reduce tree are needed -- a short fixed drain after
//   the last beat lets the 1-stage term pipeline settle, then the dequant pass
//   reads the banks.
//
// LATENCY (deterministic):  K (stream) + ACC_DRAIN (term-pipe settle) +
//   DEQ_LAT (the time-shared dequant pipeline) + NOUT walk.  Lower than the old
//   tree-based path.
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop
//   (the accumulator feedback rides the integer-add register; fp8_mul, the pow2
//   scaling, the fixed-point convert and the dequant combine are feed-forward).
//----------------------------------------------------------------------------
// HANDSHAKE  (mirrors glm_matmul_pipe; activation interface identical)
//   start      : 1-cycle pulse; latches k_len, a_shift, w_scale; clears banks.
//   k_len      : number of K beats this tile.
//   in_valid   : a K-beat is presented (a_col / w_row).
//   a_col[pi]  : bf16 A[pi][k]  (PE_M packed)  -- SAME as glm_matmul_pipe.
//   w_row[pj]  : FP8 E4M3 W[k][pj] (PE_N packed, 8-bit each).
//   a_shift[pi]: signed-8 per-row activation pow2 quant exponent (at start).
//   w_scale    : bf16 BLOCK dequant scale per (output column pj, K-block bj),
//                packed  w_scale[16*(bj*PE_N + pj) +: 16]  (at start).
//   out_valid  : C tile (PE_M x PE_N bf16) valid for 1 cycle.  busy high in flight.
//============================================================================
module glm_matmul_fp8 #(
    parameter integer PE_M = 4,       // array rows (== tile M)
    parameter integer PE_N = 4,       // array cols (== tile N)
    parameter integer KMAX = 256,     // max supported K (counter width)
    parameter integer BLK  = 128,     // weight block size along K (and N) -- [128,128]
    // ---- fixed-point (BFP) accumulator geometry (tunable) ----
    // ACC_FRAC : fractional bits below the binary point (LSB weight 2^-ACC_FRAC).
    //            =18 makes every fp8xfp8 product EXACT (products are multiples of
    //            2^-18); smaller trades precision for area (BFP/Hopper regime).
    // ACC_W    : total signed accumulator width.  Must cover the max block sum:
    //            |sum| < BLK*448^2 < 2^43, so >= ACC_FRAC + 26 keeps it exact.
    parameter integer ACC_FRAC = 18,
    parameter integer ACC_W    = 48
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    input  wire                       start,      // begin a tile
    input  wire [$clog2(KMAX+1)-1:0]  k_len,      // number of K beats this tile

    input  wire                       in_valid,   // a K-beat is presented
    input  wire [16*PE_M-1:0]         a_col,      // bf16 A[*][k]  (drop-in)
    input  wire [ 8*PE_N-1:0]         w_row,      // FP8 E4M3 W[k][*], PE_N packed
    input  wire [ 8*PE_M-1:0]         a_shift,    // signed-8 per-row act pow2 scale
    // bf16 weight BLOCK scale per (col pj, K-block bj): w_scale[16*(bj*PE_N+pj)+:16]
    input  wire [16*PE_N*((KMAX+BLK-1)/BLK)-1:0] w_scale,

    output reg                        busy,
    output reg                        out_valid,  // C tile valid (1 cycle)
    output reg  [16*PE_M*PE_N-1:0]    c_out       // bf16 C[pi][pj] packed
);
    `include "glm_fp.vh"
    `include "fp8_e4m3.vh"

    //----------------------------------------------------------------------
    // fp32_scale_pow2 : multiply an fp32 by 2^k (k a small signed integer) by
    //   ADDING k to the biased exponent.  EXACT (no rounding), pure LUT (no
    //   multiplier).  inf/nan pass through; zero/FTZ-subnormal -> signed zero;
    //   exponent overflow -> inf, underflow -> signed zero (matches glm_fp FTZ).
    //----------------------------------------------------------------------
    function automatic [31:0] fp32_scale_pow2(input [31:0] f, input signed [9:0] k);
        reg               s;
        reg [7:0]         e;
        reg [22:0]        m;
        reg signed [10:0] ne;
        begin
            s = f[31]; e = f[30:23]; m = f[22:0];
            if (e == 8'hFF) begin
                fp32_scale_pow2 = f;                       // inf / nan unchanged
            end else if (e == 8'h00) begin
                fp32_scale_pow2 = {s, 31'b0};              // zero / FTZ subnormal
            end else begin
                ne = $signed({3'b0, e}) + $signed({{1{k[9]}}, k});
                if (ne >= 11'sd255)
                    fp32_scale_pow2 = {s, 8'hFF, 23'b0};   // overflow -> inf
                else if (ne <= 11'sd0)
                    fp32_scale_pow2 = {s, 31'b0};          // underflow -> FTZ
                else
                    fp32_scale_pow2 = {s, ne[7:0], m};
            end
        end
    endfunction

    //----------------------------------------------------------------------
    // fp32_to_fixed : exact-or-truncating convert of an fp32 fp8-product into a
    //   signed fixed-point integer with weight 2^-ACC_FRAC per LSB.  The fp8xfp8
    //   product is always either +/-0 or a NORMAL fp32 (biased exp in [109,144],
    //   no inf/nan, never fp32-subnormal), and is an exact multiple of 2^-18.
    //   With ACC_FRAC=18 the right-shift branch (small-exponent products) drops
    //   only PROVABLY-ZERO low mantissa bits, so the convert is BIT-EXACT.  Pure
    //   shift/mux LUT -- no multiplier, no rounding.
    //----------------------------------------------------------------------
    function automatic signed [ACC_W-1:0] fp32_to_fixed(input [31:0] f);
        reg             s;
        reg [7:0]       e;
        reg [ACC_W-1:0] m_ext, mag;
        integer         sh;
        begin
            s     = f[31];
            e     = f[30:23];
            m_ext = {{(ACC_W-24){1'b0}}, 1'b1, f[22:0]};   // 24-bit significand, 0-ext
            if (e == 8'h00) begin
                fp32_to_fixed = {ACC_W{1'b0}};             // zero / FTZ
            end else begin
                sh = {{24{1'b0}}, e} - 150 + ACC_FRAC;     // value*2^ACC_FRAC = m*2^sh
                if (sh >= 0) mag = m_ext << sh;
                else         mag = m_ext >> (-sh);         // exact: low m bits are 0
                fp32_to_fixed = s ? -$signed(mag) : $signed(mag);
            end
        end
    endfunction

    //----------------------------------------------------------------------
    // fixed_to_fp32 : convert a signed fixed-point block accumulator (weight
    //   2^-ACC_FRAC per LSB) to fp32 with round-to-nearest-even -- the SINGLE
    //   rounding of the whole within-block accumulation.  Pure LUT (priority
    //   encode + barrel shift + RNE); no multiplier.  Our magnitudes never
    //   overflow fp32 nor underflow to FTZ, but the range guards are kept.
    //----------------------------------------------------------------------
    function automatic [31:0] fixed_to_fp32(input signed [ACC_W-1:0] x);
        reg [ACC_W-1:0]   mag;
        /* verilator lint_off UNUSEDSIGNAL */ // only the low 24 bits of the shift feed mant
        reg [ACC_W-1:0]   shifted;
        /* verilator lint_on UNUSEDSIGNAL */
        reg               s, guard, sticky, round_up;
        reg [23:0]        mant;
        reg [24:0]        mant_r;
        integer           p, i, e_b;
        begin
            if (x == {ACC_W{1'b0}}) begin
                fixed_to_fp32 = 32'b0;
            end else begin
                s   = x[ACC_W-1];
                mag = s ? (~x + {{(ACC_W-1){1'b0}}, 1'b1}) : x;
                // highest set bit position p (mag != 0 here).
                p = 0;
                for (i = 0; i < ACC_W; i = i + 1)
                    if (mag[i]) p = i;
                // value = mag * 2^-ACC_FRAC ; unbiased exp = p - ACC_FRAC.
                e_b    = (p - ACC_FRAC) + 127;
                guard  = 1'b0;
                sticky = 1'b0;
                if (p >= 23) begin
                    shifted = mag >> (p - 23);
                    mant    = shifted[23:0];
                    if (p >= 24) guard = mag[(p-24)];
                    // sticky = OR of all bits strictly below the guard bit.
                    for (i = 0; i < ACC_W; i = i + 1)
                        if ((i <= (p - 25)) && mag[i]) sticky = 1'b1;
                end else begin
                    shifted = mag << (23 - p);             // p < 23: exact, no GRS
                    mant    = shifted[23:0];
                end
                round_up = guard & (sticky | mant[0]);
                mant_r   = {1'b0, mant} + {24'b0, round_up};
                if (mant_r[24]) begin
                    mant_r = mant_r >> 1;
                    e_b    = e_b + 1;
                end
                if (e_b >= 255)
                    fixed_to_fp32 = {s, 8'hFF, 23'b0};     // (won't occur) overflow
                else if (e_b <= 0)
                    fixed_to_fp32 = {s, 31'b0};            // (won't occur) FTZ
                else
                    fixed_to_fp32 = {s, e_b[7:0], mant_r[22:0]};
            end
        end
    endfunction

    localparam integer KW   = $clog2(KMAX+1);
    localparam integer NOUT = PE_M * PE_N;
    localparam integer OW   = (NOUT > 1) ? $clog2(NOUT) : 1;
    localparam integer MW   = (PE_M > 1) ? $clog2(PE_M) : 1;
    localparam integer NW   = (PE_N > 1) ? $clog2(PE_N) : 1;

    // K-block fan-out: NB = ceil(KMAX/BLK) accumulator banks per PE.
    localparam integer NB  = (KMAX + BLK - 1) / BLK;
    localparam integer BKW = $clog2(NB + 1);          // current-block index width
    localparam integer PSW = (BLK > 1) ? $clog2(BLK) : 1; // within-block position width

    // term pipeline depth (fp32->fixed register stage) + settle margin.
    localparam integer ACC_DRAIN = 3;

    // ----------------------------------------------------------------------
    // Control regs
    // ----------------------------------------------------------------------
    reg  [KW-1:0]      k_cnt;
    reg  [KW-1:0]      k_target;
    /* verilator lint_off UNUSEDSIGNAL */ // unread by design when NB==1 (single K-block)
    reg  [BKW-1:0]     kblk;          // current K-block 0..NB-1
    reg  [PSW-1:0]     kpos;          // position within current K-block 0..BLK-1
    /* verilator lint_on UNUSEDSIGNAL */
    reg                streaming;
    reg                draining;      // let the 1-stage term pipe settle into banks
    reg                rescaling;     // time-shared dequant pass
    reg  [7:0]         drain_cnt;

    // latched per-tile scales
    reg  [ 8*PE_M-1:0]                   a_shift_q;
    reg  [16*PE_N*NB-1:0]                w_scale_q;

    wire               issue      = streaming & in_valid;
    wire               last_issue = issue & (k_cnt == k_target - 1'b1);

    // ----------------------------------------------------------------------
    // K-block bookkeeping (kblk/kpos): only meaningful when NB > 1.  For a
    // single K-block (NB==1, KMAX<=BLK) every beat targets the one bank, so the
    // counters and the per-beat blk_last test collapse to constant 0.
    // ----------------------------------------------------------------------
    generate
    if (NB > 1) begin : GKBLK
        wire blk_last = (kpos == (BLK[PSW-1:0] - 1'b1)); // last beat of block
        always @(posedge clk) begin
            if (rst) begin
                kblk <= {BKW{1'b0}};
                kpos <= {PSW{1'b0}};
            end else begin
                if (start) begin
                    kblk <= {BKW{1'b0}};
                    kpos <= {PSW{1'b0}};
                end
                if (issue) begin
                    if (blk_last) begin
                        kblk <= kblk + 1'b1;   // next beat starts the next K-block
                        kpos <= {PSW{1'b0}};
                    end else begin
                        kpos <= kpos + 1'b1;
                    end
                end
            end
        end
    end else begin : GKBLK
        always @(posedge clk) begin           // NB==1: single K-block, counters tied 0
            kblk <= {BKW{1'b0}};
            kpos <= {PSW{1'b0}};
        end
    end
    endgenerate

    // per-PE/per-bank fixed-point block accumulators (written by the BANK
    // generate below with constant genvar indices; read by the dequant pass).
    reg signed [ACC_W-1:0] accx [0:NB-1][0:PE_M-1][0:PE_N-1];

    // ----------------------------------------------------------------------
    // Per-PE FP8-mul + NB-way banked fixed-point integer accumulate.
    // ----------------------------------------------------------------------
    genvar gi, gj, gb;
    generate
    for (gi = 0; gi < PE_M; gi = gi + 1) begin : ROW
      for (gj = 0; gj < PE_N; gj = gj + 1) begin : COL
        // ---- front-end: dynamic FP8 quantization of the activation ----
        // bf16 activation -> fp32 -> pow2 prescale (2^a_shift, exponent add) ->
        // encode to E4M3.  Weight already E4M3.  Product via the 4x4 fp8_mul.
        wire [15:0] a_bf  = a_col[16*gi +: 16];
        wire [ 7:0] w_q   = w_row[ 8*gj +:  8];
        wire signed [7:0] ash = $signed(a_shift_q[8*gi +: 8]);
        wire [31:0] a_f   = bf16_to_fp32(a_bf);
        wire [31:0] a_fs  = fp32_scale_pow2(a_f, {{2{ash[7]}}, ash}); // *2^ash
        wire [ 7:0] a_q   = fp32_to_fp8e4m3(a_fs);
        // THE 4x4 MANTISSA MULTIPLY -> EXACT fp32 product (LUT, not DSP).
        wire [31:0] prod_f = fp8_mul(a_q, w_q);
        // convert the exact product to fixed-point, then register one stage to
        // keep the front-end combinational path off the accumulate path.
        wire signed [ACC_W-1:0] termx = fp32_to_fixed(prod_f);
        reg  signed [ACC_W-1:0] term_r;
        reg                     tv_r;
        reg  [BKW-1:0]          bank_r;
        always @(posedge clk) begin
            if (rst) begin
                tv_r <= 1'b0;
            end else begin
                tv_r   <= issue;       // a product for this beat
                term_r <= termx;
                bank_r <= kblk;        // K-block this beat belongs to
            end
        end

        for (gb = 0; gb < NB; gb = gb + 1) begin : BANK
          // this bank integer-accumulates only the beats whose K-block == gb.
          wire sel_b;
          if (NB > 1) begin : GIB
              localparam [BKW-1:0] BIDX = gb[BKW-1:0];
              assign sel_b = tv_r & (bank_r == BIDX);
          end else begin : GIB
              assign sel_b = tv_r;          // single bank: every beat targets it
          end
          always @(posedge clk) begin
              if (start)      accx[gb][gi][gj] <= {ACC_W{1'b0}};   // clear bank
              else if (sel_b) accx[gb][gi][gj] <= accx[gb][gi][gj] + term_r;
          end
        end // BANK
      end
    end
    endgenerate

    // ----------------------------------------------------------------------
    // Time-shared dequant rescale: walk the NOUT output elements (one per
    // cycle).  For element (ri,rj):
    //   c = bf16( 2^-a_shift[ri] * SUM_bj ( fixed_to_fp32(accx[bj][ri][rj]) * w_scale[rj][bj] ) )
    // The fixed-point block accumulators are converted to fp32 ONCE here (RNE);
    // the per-block weight scales are the only 24x24 fp32 multiplies in the unit;
    // the 2^-a_shift undo is an exact exponent add (LUT).
    // ----------------------------------------------------------------------
    reg [MW-1:0] ri;
    reg [NW-1:0] rj;
    reg [OW:0]   rcnt;

    // PIPELINED dequant streaming ONE element/cycle.  The per-element math is the
    // left-fold  deq = ((0 + acc[0]*w[0]) + acc[1]*w[1]) + ...  reusing the
    // bit-exact fp32_mul_pipe / fp32_add_pipe.  DEQ_LAT = FP_MUL_LAT + NB*FP_ADD_LAT
    // (the fold depth); the shallow 2^-a_shift exponent-add and bf16 round stay
    // combinational at the tail.
    localparam integer DEQ_LAT = `FP_MUL_LAT + NB*`FP_ADD_LAT;

    // output side-band pipe: carry (valid, output-slot, a_shift) alongside the
    // dequant datapath so the result lands at the right c_out slot DEQ_LAT later.
    reg          dq_v_pipe    [0:DEQ_LAT-1];
    reg  [OW:0]  dq_slot_pipe [0:DEQ_LAT-1];
    reg  [7:0]   dq_ash_pipe  [0:DEQ_LAT-1];

    // feed: present element (ri,rj) for one cycle while walking 0..NOUT-1.
    wire         resc_feed_v = rescaling & (rcnt < NOUT[OW:0]);

    // --- stage A: NB parallel products fixed_to_fp32(accx[bb][ri][rj]) * w_scale[rj][bj] ---
    wire [31:0]  dq_prod  [0:NB-1];
    wire         dq_prodv [0:NB-1];
    // --- stage B: left-fold of {0, prod0, prod1, ...} (each add LAT=AL) ---
    wire [31:0]  dq_s     [0:NB-1];   // dq_s[k] = (((0+p0)+p1)+...+pk)
    /* verilator lint_off UNUSEDSIGNAL */ // fold valid_out: superseded by dq_v_pipe
    wire         dq_sv    [0:NB-1];
    /* verilator lint_on UNUSEDSIGNAL */
    genvar gd;
    generate
    for (gd = 0; gd < NB; gd = gd + 1) begin : DQ
        // bank selects (procedural: variable-base part-selects on w_scale_q, like
        // the original combinational deq, kept out of continuous assigns).  The
        // fixed-point bank is converted to fp32 ONCE here (the single rounding).
        reg [31:0] acc_sel, wf_sel;
        integer    dq_widx;
        always @* begin
            acc_sel = fixed_to_fp32(accx[gd][ri][rj]);
            dq_widx = gd*PE_N + {{(32-NW){1'b0}}, rj};   // linear (K-block, col)
            wf_sel  = bf16_to_fp32(w_scale_q[16*dq_widx +: 16]);
        end
        fp32_mul_pipe u_dqmul (
            .clk(clk), .rst(rst), .valid_in(resc_feed_v),
            .a(acc_sel), .b(wf_sel),
            .valid_out(dq_prodv[gd]), .result(dq_prod[gd])
        );
        // align product k to the running sum it folds into: delay by k*FP_ADD_LAT.
        wire [31:0] pmd;  wire pmdv;
        if (gd == 0) begin : DLY0
            assign pmd = dq_prod[0];  assign pmdv = dq_prodv[0];
        end else begin : DLYK
            localparam integer DD = gd*`FP_ADD_LAT;
            reg [31:0] pd [0:DD-1];
            reg        vd [0:DD-1];
            integer di;
            always @(posedge clk) begin
                pd[0] <= dq_prod[gd];  vd[0] <= dq_prodv[gd];
                for (di = 1; di < DD; di = di + 1) begin
                    pd[di] <= pd[di-1];  vd[di] <= vd[di-1];
                end
            end
            assign pmd = pd[DD-1];  assign pmdv = vd[DD-1];
        end
        // s_k = fp32_add(s_{k-1}, p_k)  (s_{-1} = +0) -- SAME order as the fold.
        if (gd == 0) begin : FOLD0
            fp32_add_pipe u_dqadd (
                .clk(clk), .rst(rst), .valid_in(pmdv),
                .a(32'h0000_0000), .b(pmd),
                .valid_out(dq_sv[gd]), .result(dq_s[gd])
            );
        end else begin : FOLDK
            // pmdv (product k delayed by k*AL) asserts on the SAME cycles as
            // dq_sv[gd-1]; use it as the stage valid so it is never dangling.
            fp32_add_pipe u_dqadd (
                .clk(clk), .rst(rst), .valid_in(pmdv),
                .a(dq_s[gd-1]), .b(pmd),
                .valid_out(dq_sv[gd]), .result(dq_s[gd])
            );
        end
    end
    endgenerate

    // tail (combinational, shallow): undo 2^-a_shift, round to bf16.
    wire signed [7:0] ash_o  = $signed(dq_ash_pipe[DEQ_LAT-1]);
    wire [31:0]       deq_un = fp32_scale_pow2(dq_s[NB-1], -{{2{ash_o[7]}}, ash_o});
    wire [15:0]       c_bf   = fp32_to_bf16(deq_un);
    // 32-bit-widened output slot for the c_out part-select base.
    wire [31:0]       dq_slot_o = {{(31-OW){1'b0}}, dq_slot_pipe[DEQ_LAT-1]};

    // ----------------------------------------------------------------------
    // Control FSM
    // ----------------------------------------------------------------------
    integer rp;
    always @(posedge clk) begin
        if (rst) begin
            busy          <= 1'b0;
            streaming     <= 1'b0;
            draining      <= 1'b0;
            rescaling     <= 1'b0;
            out_valid     <= 1'b0;
            k_cnt         <= {KW{1'b0}};
            k_target      <= {KW{1'b0}};
            drain_cnt     <= 8'd0;
            ri            <= {MW{1'b0}};
            rj            <= {NW{1'b0}};
            rcnt          <= {(OW+1){1'b0}};
            for (rp = 0; rp < DEQ_LAT; rp = rp + 1) dq_v_pipe[rp] <= 1'b0;
        end else begin
            out_valid <= 1'b0;

            // ---- dequant (valid,slot,a_shift) side-band pipe: shift every cycle ----
            dq_v_pipe[0]    <= resc_feed_v;
            dq_slot_pipe[0] <= rcnt;
            dq_ash_pipe[0]  <= a_shift_q[8*ri +: 8];
            for (rp = 1; rp < DEQ_LAT; rp = rp + 1) begin
                dq_v_pipe[rp]    <= dq_v_pipe[rp-1];
                dq_slot_pipe[rp] <= dq_slot_pipe[rp-1];
                dq_ash_pipe[rp]  <= dq_ash_pipe[rp-1];
            end

            // ---- write each dequantized element as it leaves the dequant pipe;
            //      the LAST slot (NOUT-1) completes the tile. ----
            if (dq_v_pipe[DEQ_LAT-1]) begin
                c_out[16*dq_slot_o +: 16] <= c_bf;
                if (dq_slot_pipe[DEQ_LAT-1] == NOUT[OW:0] - 1'b1) begin
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                end
            end

            if (start) begin
                busy          <= 1'b1;
                streaming     <= 1'b1;
                draining      <= 1'b0;
                rescaling     <= 1'b0;
                k_cnt         <= {KW{1'b0}};
                k_target      <= k_len;
                a_shift_q     <= a_shift;
                w_scale_q     <= w_scale;
                for (rp = 0; rp < DEQ_LAT; rp = rp + 1)
                    dq_v_pipe[rp]  <= 1'b0;   // drop any stale dequant-pipe entries
            end

            // ---- K streaming (route each beat to its K-block bank) ----
            if (issue) begin
                k_cnt <= k_cnt + 1'b1;
                if (last_issue) begin
                    streaming <= 1'b0;
                    draining  <= 1'b1;
                    drain_cnt <= ACC_DRAIN[7:0];
                end
            end

            // ---- let the 1-stage term pipeline settle into the banks, then
            //      begin the dequant rescale. ----
            if (draining) begin
                if (drain_cnt == 8'd0) begin
                    draining  <= 1'b0;
                    rescaling <= 1'b1;
                    ri        <= {MW{1'b0}};
                    rj        <= {NW{1'b0}};
                    rcnt      <= {(OW+1){1'b0}};
                end else begin
                    drain_cnt <= drain_cnt - 8'd1;
                end
            end

            // ---- dequant rescale FEED: present one element per cycle ----
            // rcnt walks 0..NOUT-1 in (ri*PE_N+rj) order, so it IS the linear slot;
            // the pipelined datapath writes c_out DEQ_LAT later (block above).  We
            // keep rescaling asserted until the final slot has been written.
            if (rescaling) begin
                if (rcnt < NOUT[OW:0]) begin
                    rcnt <= rcnt + 1'b1;
                    if (rj == PE_N[NW-1:0] - 1'b1) begin
                        rj <= {NW{1'b0}};
                        ri <= ri + 1'b1;
                    end else begin
                        rj <= rj + 1'b1;
                    end
                end
                // tile completes when the last slot leaves the dequant pipe.
                if (dq_v_pipe[DEQ_LAT-1] &&
                    (dq_slot_pipe[DEQ_LAT-1] == NOUT[OW:0] - 1'b1))
                    rescaling <= 1'b0;
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
