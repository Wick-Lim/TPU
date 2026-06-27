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
//   the OFFICIAL GLM-5.2 FP8 numerics so the published zai-org/GLM-5.2-FP8
//   checkpoint runs with NO re-quantization:
//
//     * Weights W arrive already in FP8 E4M3 (8-bit) with a per-output-channel
//       (per-column) bf16 dequant scale  w_scale[pj]  --- exactly the published
//       checkpoint layout (E4M3 weights + per-block/channel scale).
//     * Activations A arrive bf16 (the SAME bf16 activation interface as
//       glm_matmul_pipe, for drop-in compatibility) and are DYNAMICALLY
//       quantized to FP8 E4M3 on the fly.  Each array ROW carries a per-vector
//       (per-token) power-of-two activation scale  a_shift[pi]  (a signed
//       integer): the activation is pre-scaled by 2^a_shift (a cheap EXPONENT
//       add -- NO multiplier) to land in E4M3's dynamic range, encoded to E4M3,
//       and the exact inverse 2^-a_shift is folded back at dequant.  A
//       power-of-two activation scale is the standard amax->2^e FP8 scheme and
//       costs ZERO DSP (it is an exponent bias, not a multiply).
//
// THE HARDWARE WIN (why FP8 over the fp32 baseline):
//   The per-term product is  fp8_mul(act_e4m3, wgt_e4m3)  --- a 4-bit x 4-bit
//   MANTISSA multiply (src/fp8_e4m3.vh), which synthesizes to the plentiful
//   Gowin GW2A-18 LUT, NOT to the scarce 24x24 DSP the fp32 datapath burns.
//   The KEY INVARIANT: the MULTIPLIER is fp8-width (sips DSP); ACCUMULATION
//   reuses the EXISTING, golden, pipelined fp32 adder (fp32_add_pipe).  The only
//   24x24 fp32 multiply left in the whole unit is ONE time-shared dequant
//   multiplier (acc * w_scale) used once per output element at the very end ---
//   so DSP cost is O(1), independent of K and of PE_M*PE_N.
//
//----------------------------------------------------------------------------
// ACCUMULATOR CHOICE:  FP32 (reuse fp32_add_pipe), NOT wide fixed-point.
//   * The contract (glm_fp.vh) is bf16-store / fp32-reduce / bf16-out.  An fp32
//     accumulator keeps this unit numerically in the SAME family as the fp32
//     baseline glm_matmul_pipe, so the two are directly comparable and the
//     golden reuses the same reduction discipline.
//   * fp8_mul returns an EXACT fp32 product (8-bit significand, exp in
//     [109,144]); feeding it straight into the proven fp32_add_pipe means NO new
//     rounding logic to verify --- accumulation is the already-golden fp32 add.
//   * fp8 products span ~2^-18 .. 2^17 (448^2); a fixed-point accumulator that
//     covered that range with K-deep headroom would be very wide (>~60 bits) and
//     would need its own rounding/normalize at the end -- more LUT, new numerics
//     to prove.  fp32 add gives the dynamic range for free.  We therefore reuse
//     the L-way interleaved fp32_add_pipe accumulator (identical structure to
//     glm_matmul_pipe), simply fed by fp8_mul instead of an fp32 multiply.
//
//----------------------------------------------------------------------------
// L-WAY INTERLEAVED FP32 ACCUMULATION  (same correctness proof as the baseline)
//   fp32_add_pipe has latency L = FP_ADD_LAT, throughput 1/cycle.  A naive
//   self-feedback acc<=acc+prod every cycle would read a stale acc.  So each PE
//   keeps L sub-accumulators ps[0..L-1]; the term for beat k folds into lane
//   (k mod L); lane j is re-issued exactly L cycles after its prior issue (when
//   its prior add result has just returned), so the c operand is never stale.
//   Same writeback-lane shift register + same-edge C-forwarding as the baseline.
//   After K beats: ps[j] = Sum_{k=j mod L} prod_k, and Sum_j ps[j] = full dot.
//   The L partials are summed by a fixed 8-leaf / 3-level fp32_add_pipe tree.
//
// LATENCY (deterministic):  from first K-beat accepted to out_valid =
//     K (stream) + L (add drain) + TREE_LAT (3*FP_ADD_LAT) + PE_M*PE_N (the
//     time-shared dequant rescale) + 1.
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop
//   (all accumulator feedback rides through the fp32_add_pipe registers; the
//   fp8_mul and pow2 scaling are feed-forward combinational).  Reuses
//   glm_fp_pipe.v (fp32_add_pipe) and the glm_fp.vh / fp8_e4m3.vh contracts.
//----------------------------------------------------------------------------
// HANDSHAKE  (mirrors glm_matmul_pipe; activation interface identical)
//   start     : 1-cycle pulse; latches k_len, a_shift, w_scale; clears ps.
//   k_len     : number of K beats this tile.
//   in_valid  : a K-beat is presented (a_col / w_row).
//   a_col[pi] : bf16 A[pi][k]  (PE_M packed)  -- SAME as glm_matmul_pipe.
//   w_row[pj] : FP8 E4M3 W[k][pj] (PE_N packed, 8-bit each).
//   a_shift[pi]: signed-8 per-row activation pow2 quant exponent (at start).
//   w_scale[pj]: bf16 per-col weight dequant scale (at start).
//   out_valid : C tile (PE_M x PE_N bf16) valid for 1 cycle.  busy high in flight.
//============================================================================
module glm_matmul_fp8 #(
    parameter integer PE_M = 4,       // array rows (== tile M)
    parameter integer PE_N = 4,       // array cols (== tile N)
    parameter integer KMAX = 256      // max supported K (counter width)
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    input  wire                       start,      // begin a tile
    input  wire [$clog2(KMAX+1)-1:0]  k_len,      // number of K beats this tile

    input  wire                       in_valid,   // a K-beat is presented
    input  wire [16*PE_M-1:0]         a_col,      // bf16 A[*][k]  (drop-in)
    input  wire [ 8*PE_N-1:0]         w_row,      // FP8 E4M3 W[k][*], PE_N packed
    input  wire [ 8*PE_M-1:0]         a_shift,    // signed-8 per-row act pow2 scale
    input  wire [16*PE_N-1:0]         w_scale,    // bf16 per-col weight dequant scale

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

    // ----------------------------------------------------------------------
    // Control regs
    // ----------------------------------------------------------------------
    reg  [KW-1:0]      k_cnt;
    reg  [KW-1:0]      k_target;
    reg  [2:0]         lane;          // issue lane 0..L-1
    reg                streaming;
    reg                mac_draining;  // wait L cycles for last add to retire
    reg                tree_draining; // wait TREE_LAT for the reduce tree
    reg                rescaling;     // time-shared dequant pass
    reg  [7:0]         mac_drain;
    reg  [7:0]         drain_cnt;
    reg                red_go;        // 1-cycle pulse: launch every PE's add-tree

    // latched per-tile scales
    reg  [ 8*PE_M-1:0] a_shift_q;
    reg  [16*PE_N-1:0] w_scale_q;

    wire               issue      = streaming & in_valid;
    wire               last_issue = issue & (k_cnt == k_target - 1'b1);

    /* verilator lint_off WIDTHTRUNC */
    localparam [2:0] LANE_LAST = (L - 1);
    /* verilator lint_on WIDTHTRUNC */
    wire [2:0] lane_nxt = (lane == LANE_LAST) ? 3'd0 : (lane + 3'd1);

    // per-PE final fp32 dot product (latched into acc at tree completion)
    wire [31:0] pe_sum [0:PE_M-1][0:PE_N-1];
    reg  [31:0] acc    [0:PE_M-1][0:PE_N-1];

    // ----------------------------------------------------------------------
    // Per-PE FP8-mul + L-way interleaved FP32 accumulate.  Generated grid.
    // ----------------------------------------------------------------------
    genvar gi, gj;
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

        // L sub-accumulators (sized to the fixed 8-leaf tree; ps[L..7] held +0).
        reg  [31:0] ps [0:7];
        // writeback-lane shift register (L deep, matches add latency).
        reg  [2:0]  lane_pipe [0:L-1];
        wire        add_v;
        wire [31:0] add_y;
        wire [2:0]  wb_lane = lane_pipe[L-1];

        // same-edge C forwarding (lane re-issued the cycle its result returns).
        wire        fwd  = add_v && (wb_lane == lane);
        wire [31:0] c_in = fwd ? add_y : ps[lane];

        // accumulate: result = prod_f + c_in   (LAT = L).  fp8 mul already done,
        // so the only pipelined FP op per PE per beat is ONE fp32 ADD.
        fp32_add_pipe u_acc (
            .clk(clk), .rst(rst), .valid_in(issue),
            .a(prod_f), .b(c_in),
            .valid_out(add_v), .result(add_y)
        );

        integer li;
        always @(posedge clk) begin
            lane_pipe[0] <= lane;
            for (li = 1; li < L; li = li + 1)
                lane_pipe[li] <= lane_pipe[li-1];
        end

        integer pj2;
        always @(posedge clk) begin
            if (start) begin
                for (pj2 = 0; pj2 < 8; pj2 = pj2 + 1)
                    ps[pj2] <= 32'h0000_0000;   // clear all 8 leaves (ps[L..7] stay +0)
            end else if (add_v) begin
                ps[wb_lane] <= add_y;           // wb_lane in 0..L-1
            end
        end

        // ---- reduction: fixed 8-leaf / 3-level fp32_add_pipe tree ----
        // leaves are ps[0..7]; lanes only ever write ps[0..L-1], so ps[L..7]==+0.
        wire [31:0] lf0 = ps[0];
        wire [31:0] lf1 = ps[1];
        wire [31:0] lf2 = ps[2];
        wire [31:0] lf3 = ps[3];
        wire [31:0] lf4 = ps[4];
        wire [31:0] lf5 = ps[5];
        wire [31:0] lf6 = ps[6];
        wire [31:0] lf7 = ps[7];

        /* verilator lint_off UNUSEDSIGNAL */
        // level 1 (4 adders)
        wire        l1a_v, l1b_v, l1c_v, l1d_v;
        wire [31:0] l1a_y, l1b_y, l1c_y, l1d_y;
        fp32_add_pipe u_l1a (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(lf0), .b(lf1), .valid_out(l1a_v), .result(l1a_y));
        fp32_add_pipe u_l1b (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(lf2), .b(lf3), .valid_out(l1b_v), .result(l1b_y));
        fp32_add_pipe u_l1c (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(lf4), .b(lf5), .valid_out(l1c_v), .result(l1c_y));
        fp32_add_pipe u_l1d (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(lf6), .b(lf7), .valid_out(l1d_v), .result(l1d_y));
        // level 2 (2 adders)
        wire        l2a_v, l2b_v;
        wire [31:0] l2a_y, l2b_y;
        fp32_add_pipe u_l2a (.clk(clk), .rst(rst), .valid_in(l1a_v),
            .a(l1a_y), .b(l1b_y), .valid_out(l2a_v), .result(l2a_y));
        fp32_add_pipe u_l2b (.clk(clk), .rst(rst), .valid_in(l1c_v),
            .a(l1c_y), .b(l1d_y), .valid_out(l2b_v), .result(l2b_y));
        // level 3 (1 adder)
        wire        l3_v;
        wire [31:0] l3_y;
        fp32_add_pipe u_l3 (.clk(clk), .rst(rst), .valid_in(l2a_v),
            .a(l2a_y), .b(l2b_y), .valid_out(l3_v), .result(l3_y));
        /* verilator lint_on UNUSEDSIGNAL */

        assign pe_sum[gi][gj] = l3_y;
      end
    end
    endgenerate

    // ----------------------------------------------------------------------
    // Time-shared dequant rescale: ONE fp32 multiply, walked over the NOUT
    // output elements.  c[pi][pj] = bf16( (acc[pi][pj] * 2^-a_shift[pi]) * w_scale[pj] ).
    // The 2^-a_shift undo is an exponent add (LUT); the w_scale apply is the
    // SINGLE 24x24 fp32 multiplier in the whole unit (shared across all outputs).
    // ----------------------------------------------------------------------
    reg [MW-1:0] ri;
    reg [NW-1:0] rj;
    reg [OW:0]   rcnt;
    wire [31:0]  slot32 = {{(32-(OW+1)){1'b0}}, rcnt};   // linear output slot index

    wire signed [7:0] ash_b   = $signed(a_shift_q[8*ri +: 8]);
    wire [31:0]       acc_cur = acc[ri][rj];
    // undo the activation pow2 prescale: * 2^(-a_shift)
    wire [31:0]       acc_un  = fp32_scale_pow2(acc_cur, -{{2{ash_b[7]}}, ash_b});
    // the single shared fp32 dequant multiply: * w_scale[pj]
    wire [31:0]       w_sc    = bf16_to_fp32(w_scale_q[16*rj +: 16]);
    wire [31:0]       deq     = fp32_mul(acc_un, w_sc);
    wire [15:0]       c_bf    = fp32_to_bf16(deq);

    // ----------------------------------------------------------------------
    // Control FSM
    // ----------------------------------------------------------------------
    integer ci, cj;
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

            // ---- K streaming ----
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
                    for (ci = 0; ci < PE_M; ci = ci + 1)
                        for (cj = 0; cj < PE_N; cj = cj + 1)
                            acc[ci][cj] <= pe_sum[ci][cj];
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
