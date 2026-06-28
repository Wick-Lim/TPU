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
//   i.e. the fp8 products are INNER-accumulated in fp32 within each 128-wide
//   K-block; each block partial is multiplied by that block's weight scale; the
//   scaled block partials are OUTER-accumulated across K-blocks; finally the
//   per-token 2^-a_shift is undone (an exact exponent add) and the result is
//   rounded to bf16.  For K <= BLK there is exactly ONE K-block, so this reduces
//   to a single per-column weight scale.
//
// THE HARDWARE WIN (why FP8 over the fp32 baseline):
//   The per-term product is  fp8_mul(act_e4m3, wgt_e4m3)  --- a 4-bit x 4-bit
//   MANTISSA multiply (src/fp8_e4m3.vh), which synthesizes to the plentiful
//   Gowin GW2A-18 LUT, NOT to the scarce 24x24 DSP the fp32 datapath burns.
//   The KEY INVARIANT: the MULTIPLIER is fp8-width (sips DSP); ACCUMULATION
//   reuses the EXISTING, golden, pipelined fp32 adder (fp32_add_pipe).  The only
//   24x24 fp32 multiplies left in the whole unit are the time-shared dequant
//   multipliers (block_partial * w_scale): O(NB) of them (NB = #K-blocks),
//   walked once over the NOUT outputs at the very end -- independent of K-DEPTH
//   within a block and of PE_M*PE_N.
//
//----------------------------------------------------------------------------
// ACCUMULATOR CHOICE:  FP32 (reuse fp32_add_pipe), NOT wide fixed-point.
//   fp8_mul returns an EXACT fp32 product (8-bit significand, exp in [109,144]);
//   feeding it straight into the proven fp32_add_pipe means NO new rounding logic
//   to verify --- accumulation is the already-golden fp32 add, giving the full
//   fp8-product dynamic range (~2^-18..448^2) for free.
//
//----------------------------------------------------------------------------
// PER-K-BLOCK L-WAY INTERLEAVED FP32 ACCUMULATION
//   The stream cannot be back-pressured (one K-beat per cycle), and the L-deep
//   fp32_add_pipe has L beats in flight at any K-block boundary, so we cannot
//   "drain + reset" a single accumulator between blocks.  Instead each PE owns
//   NB independent accumulator BANKS (NB = ceil(KMAX/BLK)); beat k feeds ONLY
//   the bank for its K-block (kblk = k / BLK).  Each bank is the SAME L-way
//   interleaved fp32_add_pipe accumulator as the baseline: L sub-accumulators
//   ps[0..L-1], the term for a bank-local beat folds into lane (issue mod L),
//   lane j re-issued exactly L cycles after its prior issue (its add result has
//   just returned -- same-edge C-forwarding, never a stale c operand).  A GLOBAL
//   lane counter is shared by all banks: a bank's beats are contiguous, so its
//   re-issue gap is exactly L regardless of the starting lane.  After K beats,
//   bank b's L partials are summed by a fixed 8-leaf / 3-level fp32_add_pipe
//   tree into block_partial[b]; all banks' trees fire together.
//
// LATENCY (deterministic):  from first K-beat accepted to out_valid =
//     K (stream) + L (add drain) + TREE_LAT (3*FP_ADD_LAT) + PE_M*PE_N (the
//     time-shared dequant rescale) + 1.   Unchanged by NB (banks run in parallel
//     during streaming; all trees reduce concurrently).
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop
//   (all accumulator feedback rides through the fp32_add_pipe registers; fp8_mul,
//   the pow2 scaling and the dequant combine are feed-forward combinational).
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
    parameter integer BLK  = 128      // weight block size along K (and N) -- [128,128]
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

    // Accumulator interleave depth == the fp32 ADD latency (so each lane is
    // re-issued exactly when its previous add result returns).
    localparam integer L = `FP_ADD_LAT;            // fp32_add_pipe latency
    // Fixed 8-leaf / 3-level reduction tree (covers any L <= 8; FP_ADD_LAT=5).
    localparam integer TREE_LAT = 3 * `FP_ADD_LAT; // 3 levels of fp32_add_pipe

    localparam integer KW   = $clog2(KMAX+1);
    localparam integer NOUT = PE_M * PE_N;
    localparam integer OW   = (NOUT > 1) ? $clog2(NOUT) : 1;
    localparam integer MW   = (PE_M > 1) ? $clog2(PE_M) : 1;
    localparam integer NW   = (PE_N > 1) ? $clog2(PE_N) : 1;

    // K-block fan-out: NB = ceil(KMAX/BLK) accumulator banks per PE.
    localparam integer NB  = (KMAX + BLK - 1) / BLK;
    localparam integer BKW = $clog2(NB + 1);          // current-block index width
    localparam integer PSW = (BLK > 1) ? $clog2(BLK) : 1; // within-block position width

    // ----------------------------------------------------------------------
    // Control regs
    // ----------------------------------------------------------------------
    reg  [KW-1:0]      k_cnt;
    reg  [KW-1:0]      k_target;
    reg  [2:0]         lane;          // GLOBAL issue lane 0..L-1 (shared by banks)
    /* verilator lint_off UNUSEDSIGNAL */ // unread by design when NB==1 (single K-block)
    reg  [BKW-1:0]     kblk;          // current K-block 0..NB-1
    reg  [PSW-1:0]     kpos;          // position within current K-block 0..BLK-1
    /* verilator lint_on UNUSEDSIGNAL */
    reg                streaming;
    reg                mac_draining;  // wait L cycles for last add to retire
    reg                tree_draining; // wait TREE_LAT for the reduce tree
    reg                rescaling;     // time-shared dequant pass
    reg  [7:0]         mac_drain;
    reg  [7:0]         drain_cnt;
    reg                red_go;        // 1-cycle pulse: launch every PE's add-tree

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

    /* verilator lint_off WIDTHTRUNC */
    localparam [2:0] LANE_LAST = (L - 1);
    /* verilator lint_on WIDTHTRUNC */
    wire [2:0] lane_nxt = (lane == LANE_LAST) ? 3'd0 : (lane + 3'd1);

    // per-PE, per-bank final fp32 block partial (latched into acc at tree done)
    wire [31:0] pe_sum [0:NB-1][0:PE_M-1][0:PE_N-1];
    reg  [31:0] acc    [0:NB-1][0:PE_M-1][0:PE_N-1];

    // ----------------------------------------------------------------------
    // Per-PE FP8-mul + NB-way banked, L-way interleaved FP32 accumulate.
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

        for (gb = 0; gb < NB; gb = gb + 1) begin : BANK
          // this bank accumulates only the beats whose K-block == gb.
          wire issue_b;
          if (NB > 1) begin : GIB
              localparam [BKW-1:0] BIDX = gb[BKW-1:0];
              assign issue_b = issue & (kblk == BIDX);
          end else begin : GIB
              assign issue_b = issue;       // single bank: every beat targets it
          end

          // L sub-accumulators (one per interleave lane; right-sized to ps[0:L-1]).
          reg  [31:0] ps [0:L-1];
          // writeback-lane shift register (L deep, matches add latency).
          reg  [2:0]  lane_pipe [0:L-1];
          wire        add_v;
          wire [31:0] add_y;
          wire [2:0]  wb_lane = lane_pipe[L-1];

          // same-edge C forwarding (lane re-issued the cycle its result returns).
          wire        fwd  = add_v && (wb_lane == lane);
          wire [31:0] c_in = fwd ? add_y : ps[lane];

          // accumulate: result = prod_f + c_in   (LAT = L), gated to this bank.
          fp32_add_pipe u_acc (
              .clk(clk), .rst(rst), .valid_in(issue_b),
              .a(prod_f), .b(c_in),
              .valid_out(add_v), .result(add_y)
          );

          integer li;
          always @(posedge clk) begin
              lane_pipe[0] <= lane;          // global lane, delayed L cycles
              for (li = 1; li < L; li = li + 1)
                  lane_pipe[li] <= lane_pipe[li-1];
          end

          integer pj2;
          always @(posedge clk) begin
              if (start) begin
                  for (pj2 = 0; pj2 < L; pj2 = pj2 + 1)
                      ps[pj2] <= 32'h0000_0000;  // clear all L lane accumulators
              end else if (add_v) begin
                  ps[wb_lane] <= add_y;          // wb_lane in 0..L-1
              end
          end

          // ---- reduction: 5 live leaves (ps[0:L-1]) -> 3-level fp32_add_pipe tree.
          // The +0 pad adders (b = zpad) preserve the original 8-leaf grouping
          // bit-for-bit -- including the -0.0 -> +0.0 normalization that x + (+0)
          // performs -- and the 3-level TREE_LAT.  The provably-+0 leaves ps[5..7]
          // and their level-1 adder (lf6+lf7) are removed (constant-folded into
          // zpad), saving one fp32_add_pipe and three leaf regs per bank.
          wire [31:0] lf0  = ps[0];
          wire [31:0] lf1  = ps[1];
          wire [31:0] lf2  = ps[2];
          wire [31:0] lf3  = ps[3];
          wire [31:0] lf4  = ps[4];
          wire [31:0] zpad = 32'h0000_0000;   // +0 pad (was ps[5..7], provably +0)

          /* verilator lint_off UNUSEDSIGNAL */
          // level 1 (3 adders: 2 real + 1 +0-pad)
          wire        l1a_v, l1b_v, l1c_v;
          wire [31:0] l1a_y, l1b_y, l1c_y;
          fp32_add_pipe u_l1a (.clk(clk), .rst(rst), .valid_in(red_go),
              .a(lf0), .b(lf1), .valid_out(l1a_v), .result(l1a_y));
          fp32_add_pipe u_l1b (.clk(clk), .rst(rst), .valid_in(red_go),
              .a(lf2), .b(lf3), .valid_out(l1b_v), .result(l1b_y));
          fp32_add_pipe u_l1c (.clk(clk), .rst(rst), .valid_in(red_go),
              .a(lf4), .b(zpad), .valid_out(l1c_v), .result(l1c_y));
          // level 2 (2 adders: 1 real + 1 +0-pad)
          wire        l2a_v, l2b_v;
          wire [31:0] l2a_y, l2b_y;
          fp32_add_pipe u_l2a (.clk(clk), .rst(rst), .valid_in(l1a_v),
              .a(l1a_y), .b(l1b_y), .valid_out(l2a_v), .result(l2a_y));
          fp32_add_pipe u_l2b (.clk(clk), .rst(rst), .valid_in(l1c_v),
              .a(l1c_y), .b(zpad), .valid_out(l2b_v), .result(l2b_y));
          // level 3 (1 adder)
          wire        l3_v;
          wire [31:0] l3_y;
          fp32_add_pipe u_l3 (.clk(clk), .rst(rst), .valid_in(l2a_v),
              .a(l2a_y), .b(l2b_y), .valid_out(l3_v), .result(l3_y));
          /* verilator lint_on UNUSEDSIGNAL */

          assign pe_sum[gb][gi][gj] = l3_y;
        end
      end
    end
    endgenerate

    // ----------------------------------------------------------------------
    // Time-shared dequant rescale: walk the NOUT output elements (one per
    // cycle).  For element (ri,rj):
    //   c = bf16( 2^-a_shift[ri] * SUM_bj ( acc[bj][ri][rj] * w_scale[rj][bj] ) )
    // The per-block weight scales are the only 24x24 fp32 multiplies in the unit;
    // the 2^-a_shift undo is an exact exponent add (LUT).
    // ----------------------------------------------------------------------
    reg [MW-1:0] ri;
    reg [NW-1:0] rj;
    reg [OW:0]   rcnt;
    wire [31:0]  slot32 = {{(32-(OW+1)){1'b0}}, rcnt};   // linear output slot index

    // outer-accumulate the NB scaled block partials for the current element.
    reg  [31:0] deq_comb;
    integer     bb, widx;
    always @* begin
        deq_comb = 32'h0000_0000;
        for (bb = 0; bb < NB; bb = bb + 1) begin
            widx = bb*PE_N + {{(32-NW){1'b0}}, rj};   // linear (K-block, col) scale index
            deq_comb = fp32_add(
                           deq_comb,
                           fp32_mul(acc[bb][ri][rj],
                                    bf16_to_fp32(w_scale_q[16*widx +: 16])));
        end
    end
    wire signed [7:0] ash_b   = $signed(a_shift_q[8*ri +: 8]);
    // undo the activation pow2 prescale: * 2^(-a_shift)
    wire [31:0]       deq_un  = fp32_scale_pow2(deq_comb, -{{2{ash_b[7]}}, ash_b});
    wire [15:0]       c_bf    = fp32_to_bf16(deq_un);

    // ----------------------------------------------------------------------
    // Control FSM
    // ----------------------------------------------------------------------
    integer ci, cj, cb;
    always @(posedge clk) begin
        if (rst) begin
            busy          <= 1'b0;
            streaming     <= 1'b0;
            mac_draining  <= 1'b0;
            tree_draining <= 1'b0;
            rescaling     <= 1'b0;
            out_valid     <= 1'b0;
            red_go        <= 1'b0;
            k_cnt         <= {KW{1'b0}};
            k_target      <= {KW{1'b0}};
            lane          <= 3'd0;
            mac_drain     <= 8'd0;
            drain_cnt     <= 8'd0;
            ri            <= {MW{1'b0}};
            rj            <= {NW{1'b0}};
            rcnt          <= {(OW+1){1'b0}};
        end else begin
            out_valid <= 1'b0;
            red_go    <= 1'b0;

            if (start) begin
                busy          <= 1'b1;
                streaming     <= 1'b1;
                mac_draining  <= 1'b0;
                tree_draining <= 1'b0;
                rescaling     <= 1'b0;
                k_cnt         <= {KW{1'b0}};
                k_target      <= k_len;
                lane          <= 3'd0;
                a_shift_q     <= a_shift;
                w_scale_q     <= w_scale;
            end

            // ---- K streaming (route each beat to its K-block bank) ----
            if (issue) begin
                k_cnt <= k_cnt + 1'b1;
                lane  <= lane_nxt;
                if (last_issue) begin
                    streaming    <= 1'b0;
                    mac_draining <= 1'b1;
                    mac_drain    <= L[7:0];
                end
            end

            // ---- wait for the last add to retire, then fire the reduce tree ----
            if (mac_draining) begin
                if (mac_drain == 8'd0) begin
                    mac_draining  <= 1'b0;
                    red_go        <= 1'b1;
                    tree_draining <= 1'b1;
                    drain_cnt     <= TREE_LAT[7:0];
                end else begin
                    mac_drain <= mac_drain - 8'd1;
                end
            end

            // ---- wait for the reduce tree, then latch acc + start rescale ----
            if (tree_draining) begin
                if (drain_cnt == 8'd0) begin
                    tree_draining <= 1'b0;
                    rescaling     <= 1'b1;
                    ri            <= {MW{1'b0}};
                    rj            <= {NW{1'b0}};
                    rcnt          <= {(OW+1){1'b0}};
                    for (cb = 0; cb < NB; cb = cb + 1)
                        for (ci = 0; ci < PE_M; ci = ci + 1)
                            for (cj = 0; cj < PE_N; cj = cj + 1)
                                acc[cb][ci][cj] <= pe_sum[cb][ci][cj];
                end else begin
                    drain_cnt <= drain_cnt - 8'd1;
                end
            end

            // ---- time-shared dequant rescale: one element per cycle ----
            // rcnt walks 0..NOUT-1 in (ri*PE_N+rj) order, so it IS the linear slot.
            if (rescaling) begin
                c_out[16*slot32 +: 16] <= c_bf;
                if (rcnt == NOUT[OW:0] - 1'b1) begin
                    rescaling <= 1'b0;
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                end else begin
                    rcnt <= rcnt + 1'b1;
                    if (rj == PE_N[NW-1:0] - 1'b1) begin
                        rj <= {NW{1'b0}};
                        ri <= ri + 1'b1;
                    end else begin
                        rj <= rj + 1'b1;
                    end
                end
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
