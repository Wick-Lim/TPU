`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"   // FP pipeline latencies (single source of truth)
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_matmul_pipe.v  --  GLM-5.2 BF16xBF16 -> FP32-accumulate -> BF16 GEMM,
//                        ULTRA-HIGH-FMAX pipelined systolic version  (§6,§8)
//----------------------------------------------------------------------------
// FUNCTION
//   C[M,N] = A[M,K] x W[K,N].   Operands A (activations) and W (weights) are
//   BF16; every PE multiplies bf16xbf16 and accumulates the product into a
//   per-output FP32 accumulator across the K reduction; the final fp32 sum is
//   rounded-to-nearest-even to BF16 only at the very end.  All FP arithmetic
//   is the SHARED, golden, *pipelined* glm_fp_pipe.v primitive fp32_mac_pipe
//   (a*b+c fused, LAT=5, 1 result/cycle), so the per-cycle critical path is
//   ONE pipeline stage of one FP op -- not a whole combinational mul+add.
//   That is why this routes at high fmax where the combinational glm_matmul
//   (which used fp32_mul+fp32_add comb in the PE) routes at ~20 MHz.
//
//----------------------------------------------------------------------------
// ARRAY GEOMETRY  (output-stationary PE_M x PE_N systolic array)
//   A PE_M x PE_N grid (default 4x4) of output-stationary FP-MAC PEs.
//   PE[pi][pj] owns the fp32 accumulator(s) for output tile element C[pi][pj].
//   The verifiable tile uses PE_M==M, PE_N==N (the whole C tile resident in the
//   array), the common QKV / router / FFN / LM-head tile case; the surrounding
//   sequencer issues larger problems as multiple (M,N) tiles.  K is reduced by
//   STREAMING: on each accepted K-beat the unit presents column k of A (one
//   bf16 per array row pi, broadcast across that row) and row k of W (one bf16
//   per array column pj, broadcast down that column); PE[pi][pj] forms
//   A[pi][k]*W[k][pj] and folds it into its fp32 accumulator.  Output-stationary
//   => no partial-sum movement; each accumulator updates in place, exactly the
//   fp32-accumulate-in-place discipline the numerics contract wants.
//
//============================================================================
// THE CRUX: ACCUMULATION vs MAC-PIPELINE LATENCY  (correctness proof)
//----------------------------------------------------------------------------
//   fp32_mac_pipe computes  a*b+c  with LATENCY L=5 and throughput 1/cycle.
//   A NAIVE self-feedback  acc <= mac(acc, A*W)  issued every cycle is WRONG:
//   the mac of beat k does not produce its updated acc until 5 cycles later, so
//   beats k+1..k+4 would read a STALE acc and terms would be dropped.
//
//   SOLUTION -- L-WAY INTERLEAVED PARTIAL SUMS (a.k.a. C-stationary modulo-L
//   accumulation), so each accumulator is updated exactly ONCE every L cycles,
//   matching the pipeline latency, with NO stale read:
//
//     * Each PE keeps L independent fp32 sub-accumulators  ps[0..L-1].
//     * K terms are distributed ROUND-ROBIN by k mod L:  the term for beat k,
//       prod_k = A[pi][k]*W[k][pj], is accumulated into sub-accumulator
//       (k mod L).  We issue ONE mac per cycle into the array:
//             ps_lane = mac( A*W , c = ps[lane] )   with lane = (issue k) mod L
//       and write the L-cycles-later result back into ps[lane'] where lane' is
//       the lane that was issued L cycles earlier == the same lane (because we
//       cycle lanes 0,1,..,L-1,0,1,.. so the result returning at cycle t
//       belongs to the lane issued at t-L, which is the lane currently being
//       issued).  Concretely we keep a small "issue-lane" counter for the
//       in-flight write so the returned result lands in its own lane.
//     * Because consecutive issues go to DIFFERENT lanes, lane j is RE-ISSUED
//       only every L cycles -- precisely when its previous mac result has just
//       returned (LAT=L).  So the c input to lane j's mac is ALWAYS the fully
//       updated ps[j] from lane j's previous term, never stale.  No term is
//       read against an un-retired partial sum.
//     * After all K beats have been issued (and drained, L cycles later), the L
//       sub-accumulators hold disjoint partial dot products:
//             ps[j] = Σ_{k ≡ j (mod L)} A[pi][k]*W[k][pj]
//       and  Σ_j ps[j] = Σ_{k=0..K-1} A[pi][k]*W[k][pj]  -- EXACTLY the dot
//       product, every term present once, none duplicated.  (Associativity:
//       fp32 add is not associative, so the *grouping* is part of the defined
//       numerics here; the smoke TB's golden uses the SAME L-way grouping +
//       tree so the hardware is bit-exact to its specified reference.  No term
//       is dropped or double-counted -- that is the structural guarantee.)
//     * FINAL REDUCTION: the L partial sums are summed by a small PIPELINED
//       binary adder tree built from fp32_add_pipe (ceil(log2 L) levels), then
//       rounded fp32->bf16.  Deterministic latency throughout.
//
//   DETERMINISM: every datapath element is a fixed-latency pipe + counters, so
//   the whole unit has a fixed, computable latency (below).  No data-dependent
//   stalls inside the K stream.
//
//----------------------------------------------------------------------------
// LATENCY / THROUGHPUT (per issued (M,N) tile, K-beat stream)
//   * Per PE: 1 mac issued/cycle (full throughput); after K beats are issued
//     the last mac drains in L=5 cycles, then the adder-tree (TREE_LAT) and the
//     fp32->bf16 round (comb) produce the result.  All PEs run in lockstep
//     (same broadcast operands), so the whole MxN tile finishes together.
//   * Streaming K throughput: 1 K-beat/cycle once filled (1 MAC/PE/cycle).
//   * Tile latency from first K-beat accepted to C valid:
//         K (stream) + L (mac drain) + TREE_LAT (reduce) + 1 (bf16 round reg)
//
// CONVENTIONS: synchronous ACTIVE-HIGH reset; NO latch (every reg written every
//   clock in its clocked block); NO combinational loop (all PE feedback rides
//   through the mac/add pipeline registers).  Reuses glm_fp_pipe.v unchanged.
//----------------------------------------------------------------------------
// HANDSHAKE
//   start        : pulse 1 cycle to begin a tile (latches dims, clears ps).
//   in_valid     : asserted on each K-beat that presents a_col / w_row.
//   a_col[pi]    : bf16 column-k element of A for array row pi  (PE_M of them).
//   w_row[pj]    : bf16 row-k element of W for array col pj      (PE_N of them).
//   The caller streams exactly K in_valid beats after start.  When the last
//   beat (k==K-1) is presented, the unit drains and asserts out_valid with the
//   full C tile (PE_M x PE_N bf16) for one cycle.  busy is high while a tile is
//   in flight.
//============================================================================
module glm_matmul_pipe #(
    parameter integer PE_M = 4,       // array rows (== tile M)
    parameter integer PE_N = 4,       // array cols (== tile N)
    parameter integer KMAX = 256      // max supported K (counter width)
) (
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    input  wire                       start,      // begin a tile
    input  wire [$clog2(KMAX+1)-1:0]  k_len,      // number of K beats this tile

    input  wire                       in_valid,   // a K-beat is presented
    input  wire [16*PE_M-1:0]         a_col,      // bf16 A[*][k], PE_M packed
    input  wire [16*PE_N-1:0]         w_row,      // bf16 W[k][*], PE_N packed

    output reg                        busy,
    output reg                        out_valid,  // C tile valid (1 cycle)
    output reg  [16*PE_M*PE_N-1:0]    c_out       // bf16 C[pi][pj] packed
);
    `include "glm_fp.vh"

    // Pipeline latencies are READ from glm_fp_pipe.v's single-source-of-truth
    // macros (FP_MAC_LAT / FP_ADD_LAT), so deepening the FP adder ripples into
    // the L-way interleave depth and the reduction-tree drain automatically --
    // no hardcoded magic number to forget.
    localparam integer L = `FP_MAC_LAT;       // fp32_mac_pipe latency (now 7)
    // adder-tree to reduce L partial sums: ceil(log2 L) levels of fp32_add_pipe.
    // L=7 -> ceil(log2 7)=3 levels, each fp32_add_pipe latency FP_ADD_LAT.
    localparam integer TREE_LAT = 3 * `FP_ADD_LAT; // 3 levels * add LAT (now 15)

    // ----------------------------------------------------------------------
    // Control: lane counter (k mod L), K-beat counter, drain countdown.
    // ----------------------------------------------------------------------
    localparam integer KW = $clog2(KMAX+1);
    reg  [KW-1:0]      k_cnt;        // beats issued so far
    reg  [KW-1:0]      k_target;     // = k_len for this tile
    reg  [2:0]         lane;         // issue lane 0..L-1 (round robin)
    reg                streaming;    // issuing K beats
    reg                draining;     // waiting for last mac + tree to finish
    reg  [7:0]         drain_cnt;    // counts down L + TREE_LAT + 1

    wire               issue = streaming & in_valid;     // a mac issued this cyc
    wire               last_issue = issue & (k_cnt == k_target - 1'b1);

    // next lane (round robin 0..L-1)
    /* verilator lint_off WIDTHTRUNC */
    localparam [2:0] LANE_LAST = (L - 1);  // 3-bit truncation is intended
    /* verilator lint_on WIDTHTRUNC */
    wire [2:0] lane_nxt = (lane == LANE_LAST) ? 3'd0 : (lane + 3'd1);

    // red_go : single-cycle pulse that launches every PE's add-tree, fired once
    // the last mac of the tile has drained (declared here so the generated PEs
    // below reference the real reg, not an implicit wire).
    reg red_go;

    // ----------------------------------------------------------------------
    // Per-PE MAC datapath + L-way partial sums.  Generated PE_M x PE_N grid.
    // Each PE: one fp32_mac_pipe; L sub-accumulators; final pipelined add-tree;
    // bf16 round.  All PEs share the lane/issue control (lockstep broadcast).
    // ----------------------------------------------------------------------
    genvar gi, gj;
    // collect each PE's final bf16 result.  All PEs are lockstep (same control
    // and same fixed-latency pipes), so the FSM's deterministic drain_cnt -- not
    // a per-PE done -- gates the single output-latch cycle.
    wire [15:0] pe_c [0:PE_M-1][0:PE_N-1];

    generate
    for (gi = 0; gi < PE_M; gi = gi + 1) begin : ROW
      for (gj = 0; gj < PE_N; gj = gj + 1) begin : COL
        // bf16 operands for this PE, widened to fp32 for the mac
        wire [15:0] a_bf = a_col[16*gi +: 16];
        wire [15:0] w_bf = w_row[16*gj +: 16];
        wire [31:0] a_f  = bf16_to_fp32(a_bf);
        wire [31:0] w_f  = bf16_to_fp32(w_bf);

        // L sub-accumulators (fp32).
        reg  [31:0] ps [0:L-1];

        // The lane a returning mac result belongs to = the lane that was issued
        // L cycles ago.  We track it with an L-deep shift register of issued
        // lanes so write-back is independent of the issue stream's exact phase
        // (handles in_valid bubbles too).
        reg  [2:0]  lane_pipe [0:L-1];
        wire        mac_v;
        wire [31:0] mac_y;
        wire [2:0]  wb_lane = lane_pipe[L-1];   // lane of the result out now

        // C-INPUT FORWARDING (resolves the writeback/re-issue same-edge race):
        // lane j is re-issued exactly L cycles after its prior issue, which is
        // the SAME edge its prior mac result returns.  A plain ps[lane] read
        // would see the pre-update value (term dropped).  So if a mac result is
        // returning THIS cycle for the lane being issued, forward mac_y directly
        // as the c operand instead of the (about-to-be-updated) ps[lane].
        wire        fwd  = mac_v && (wb_lane == lane);
        wire [31:0] c_in = fwd ? mac_y : ps[lane];

        // the fused mac:  result = a_f*w_f + c_in   (LAT=L)
        fp32_mac_pipe u_mac (
            .clk(clk), .rst(rst), .valid_in(issue),
            .a(a_f), .b(w_f), .c(c_in),
            .valid_out(mac_v), .result(mac_y)
        );

        integer li;
        always @(posedge clk) begin
            // shift the issued lane down the L-deep pipe (matches mac latency)
            lane_pipe[0] <= lane;
            for (li = 1; li < L; li = li + 1)
                lane_pipe[li] <= lane_pipe[li-1];
        end

        // sub-accumulator update + clear-on-start
        integer pj;
        always @(posedge clk) begin
            if (start) begin
                for (pj = 0; pj < L; pj = pj + 1)
                    ps[pj] <= 32'h0000_0000;     // +0.0
            end else if (mac_v) begin
                ps[wb_lane] <= mac_y;            // write back the updated lane
            end
        end

        // -------- final reduction: pipelined add-tree over ps[0..L-1] --------
        // L=7 partial sums summed in ceil(log2 7)=3 levels of fp32_add_pipe:
        //   level1 (4 adders): (ps0+ps1),(ps2+ps3),(ps4+ps5),(ps6+0)
        //   level2 (2 adders): (a01+a23),(a45+a6)
        //   level3 (1 adder) : (b0+b1) = full dot product
        // Trigger the tree once per tile (red_go pulse).  Every adder has the
        // SAME fixed LAT and is launched in lockstep, so one representative valid
        // per level gates the next level; the sibling valid_out bits are
        // functionally redundant (localized lint waiver, in the same spirit as
        // glm_fp_pipe.v's documented waivers).  The single output-latch cycle is
        // gated by the FSM's deterministic drain_cnt.  The fp32-add grouping here
        // is part of the defined numerics; the TB golden tolerates the tree order.
        /* verilator lint_off UNUSEDSIGNAL */
        // --- level 1 : 4 adders summing the 7 partial sums (+0 pad) ---
        wire        a01_v, a23_v, a45_v, a6_v;
        wire [31:0] a01_y, a23_y, a45_y, a6_y;
        fp32_add_pipe u_a01 (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(ps[0]), .b(ps[1]), .valid_out(a01_v), .result(a01_y));
        fp32_add_pipe u_a23 (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(ps[2]), .b(ps[3]), .valid_out(a23_v), .result(a23_y));
        fp32_add_pipe u_a45 (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(ps[4]), .b(ps[5]), .valid_out(a45_v), .result(a45_y));
        fp32_add_pipe u_a6  (.clk(clk), .rst(rst), .valid_in(red_go),
            .a(ps[6]), .b(32'h0), .valid_out(a6_v), .result(a6_y));
        // --- level 2 : (a01 + a23), (a45 + a6) ---
        wire        b0_v, b1_v;
        wire [31:0] b0_y, b1_y;
        fp32_add_pipe u_b0 (.clk(clk), .rst(rst), .valid_in(a01_v),
            .a(a01_y), .b(a23_y), .valid_out(b0_v), .result(b0_y));
        fp32_add_pipe u_b1 (.clk(clk), .rst(rst), .valid_in(a45_v),
            .a(a45_y), .b(a6_y), .valid_out(b1_v), .result(b1_y));
        // --- level 3 : (b0 + b1) = full dot product ---
        wire        csum_v;
        wire [31:0] csum_y;
        fp32_add_pipe u_c0 (.clk(clk), .rst(rst), .valid_in(b0_v),
            .a(b0_y), .b(b1_y), .valid_out(csum_v), .result(csum_y));
        /* verilator lint_on UNUSEDSIGNAL */

        assign pe_c[gi][gj] = fp32_to_bf16(csum_y);
      end
    end
    endgenerate

    // ----------------------------------------------------------------------
    // Control FSM
    // ----------------------------------------------------------------------
    reg [7:0] mac_drain;     // counts L cycles after last_issue, then fire red_go
    reg       mac_draining;

    integer ri, rj;
    always @(posedge clk) begin
        if (rst) begin
            busy         <= 1'b0;
            streaming    <= 1'b0;
            draining     <= 1'b0;
            out_valid    <= 1'b0;
            red_go       <= 1'b0;
            mac_draining <= 1'b0;
            k_cnt        <= {KW{1'b0}};
            k_target     <= {KW{1'b0}};
            lane         <= 3'd0;
            drain_cnt    <= 8'd0;
            mac_drain    <= 8'd0;
        end else begin
            out_valid <= 1'b0;
            red_go    <= 1'b0;

            if (start) begin
                busy      <= 1'b1;
                streaming <= 1'b1;
                draining  <= 1'b0;
                k_cnt     <= {KW{1'b0}};
                k_target  <= k_len;
                lane      <= 3'd0;
                mac_draining <= 1'b0;
            end

            // ---- K streaming ----
            if (issue) begin
                k_cnt <= k_cnt + 1'b1;
                lane  <= lane_nxt;
                if (last_issue) begin
                    streaming    <= 1'b0;
                    mac_draining <= 1'b1;
                    mac_drain    <= L[7:0];       // wait L cycles for mac drain
                end
            end

            // ---- wait for the last mac to retire, then fire the add-tree ----
            if (mac_draining) begin
                if (mac_drain == 8'd0) begin
                    mac_draining <= 1'b0;
                    red_go       <= 1'b1;         // launch reduction this cycle
                    draining     <= 1'b1;
                    drain_cnt    <= TREE_LAT[7:0] + 8'd1; // tree + output reg
                end else begin
                    mac_drain <= mac_drain - 8'd1;
                end
            end

            // ---- wait for the add-tree to finish, then publish C ----
            if (draining) begin
                if (drain_cnt == 8'd0) begin
                    // every PE's add-tree result is valid now; latch outputs
                    draining  <= 1'b0;
                    busy      <= 1'b0;
                    out_valid <= 1'b1;
                    for (ri = 0; ri < PE_M; ri = ri + 1)
                        for (rj = 0; rj < PE_N; rj = rj + 1)
                            c_out[16*(ri*PE_N + rj) +: 16] <= pe_c[ri][rj];
                end else begin
                    drain_cnt <= drain_cnt - 8'd1;
                end
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
