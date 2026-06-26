`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_matmul.v  --  GLM-5.2 BF16xBF16 -> FP32-accumulate -> BF16 matmul
//                                                            (ACCEL_GLM52 §6,§8)
//----------------------------------------------------------------------------
// FUNCTION
//   C[M,N] = A[M,K] x W[K,N]
//   This is the FLOATING-POINT matmul workhorse behind QKV/output projections,
//   the FFN/SwiGLU experts, the router GEMV, and the LM head.  It inherits the
//   GLM-5.2 numerics contract (glm_fp.vh):
//     * Operands A (activations) and W (weights) are BF16.
//     * Each PE multiplies bf16 x bf16 (via glm_fp.fp32_mul on widened inputs)
//       and accumulates the product into a per-output FP32 accumulator across
//       the K reduction (fp32 accumulate is mandatory: a long K dot product
//       loses precision in bf16).
//     * The final fp32 accumulator is rounded-to-nearest-even to BF16 only at
//       the very end (glm_fp.fp32_to_bf16).
//   All FP arithmetic is the shared, golden-verified glm_fp.vh primitives.
//
//   NOTE (precision): the existing gemm_systolic / gemm_ml are Q7.8 FIXED-POINT
//   and are NOT usable for the GLM bf16 datapath.  This unit reuses the
//   OUTPUT-STATIONARY systolic DATAFLOW idea from gemm_systolic but the PE
//   multiply is bf16xbf16 and the accumulator is FP32.
//
//----------------------------------------------------------------------------
// ARRAY / TILING SCHEME  (output-stationary tiled array)
//   The unit instantiates a PE_M x PE_N grid of output-stationary FP MAC PEs.
//   PE[pi][pj] owns the FP32 accumulator for output tile element C[pi][pj].
//   The verifiable slice uses PE_M==M and PE_N==N (the whole C tile is resident
//   in the array at once), which is the common case for the small QKV/router/
//   FFN tiles that drive this unit; the geometry is fully parameterized so a
//   larger problem is handled by the surrounding sequencer issuing multiple
//   (M,N) tiles.  K is reduced by STREAMING: on each accepted K-beat the unit
//   presents column k of A (one bf16 per array row pi) and row k of W (one bf16
//   per array column pj); every PE forms A[pi][k]*W[k][pj] and adds it to its
//   fp32 accumulator.  After K beats the accumulators hold the full dot
//   products and are rounded to bf16 and streamed out, one C row per cycle.
//
//   This is algebraically the standard systolic outer-product schedule
//   (rank-1 update per K-beat over the whole MxN tile) with the operands
//   broadcast to the array edges; output-stationary means no partial-sum
//   movement -- each accumulator is updated in place, which is exactly the
//   fp32-accumulate-in-place discipline the numerics contract wants.
//
//----------------------------------------------------------------------------
// PIPELINED FP MAC DATAPATH (high fmax)
//   Per PE the bf16*bf16 product and the fp32 add are the two heavy FP ops.
//   To keep fmax high the MAC is a 2-stage pipeline:
//       stage 0 (MUL) : prod_pij = fp32_mul(bf16_to_fp32(A[pi][k]),
//                                            bf16_to_fp32(W[k][pj]))   -> reg
//       stage 1 (ADD) : acc_pij  = fp32_add(acc_pij, prod_pij_reg)
//   So a K-beat accepted on cycle t is multiplied (registered) at t+1 and
//   accumulated at t+2.  The accumulate of beat k and the multiply of beat k+1
//   run concurrently in different pipe stages -> one K-beat retired per cycle
//   (full throughput) with the fp32_mul and fp32_add on SEPARATE clock cycles
//   (the critical path is one FP op, not a fused mul+add).
//
//----------------------------------------------------------------------------
// HANDSHAKE / OPERAND PORTS  (streaming, the unit is the consumer)
//   clk, rst (synchronous, active-high reset).
//   start              : 1-cycle pulse -> begin a new C = A x W tile.
//   --- K-streaming operand input (the unit pulls one K-beat/cycle) ---
//   a_req              : high while the unit wants the next operand beat.
//   a_col [M*16-1:0]   : column k of A as M bf16 lanes (lane pi = a_col[16*pi+:16]
//                        = A[pi][k]).
//   w_row [N*16-1:0]   : row k of W as N bf16 lanes (lane pj = w_row[16*pj+:16]
//                        = W[k][pj]).
//   ab_valid           : producer asserts when a_col/w_row hold beat k.
//                        (A single ready/valid feeds BOTH operands in lock-step,
//                        since the outer-product schedule consumes A-col k and
//                        W-row k on the SAME beat.)
//   --- output stream (one C row per cycle, in row order) ---
//   c_valid            : high when c_row holds a valid output row.
//   c_row [N*16-1:0]   : row of C as N bf16 lanes (lane pj = C[row][pj]).
//   --- status ---
//   busy               : high from start until done.
//   done               : 1-cycle pulse when the whole C tile has been emitted.
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic; producer answers each req on the next cycle)
//   K operand beats are pulled over K cycles; the 2-stage MAC adds a fixed
//   drain of 2 cycles after the last beat (mul of beat K-1, then its add), then
//   the M C-rows are streamed over M cycles, plus the start/done bookkeeping.
//       L(start->done) = K (stream-in) + 2 (mac drain) + M (stream-out) + const
//   With producer stalls add one cycle per stalled beat.  Throughput in each
//   streaming pass is one K-beat (a full MxN rank-1 update) per cycle.
//
//----------------------------------------------------------------------------
// (OPTIONAL, documented, NOT built) INT8-weight quantized mode: a future
//   low-memory path would store W as int8 with per-group fp32 scales and
//   dequantize (int8*scale) into the same fp32 accumulator.  For now the
//   bf16xbf16 -> fp32 -> bf16 path is the correctness baseline and is what
//   this module implements.
//
//----------------------------------------------------------------------------
// SYNTHESIZABILITY / STYLE
//   * All FP ops via glm_fp.vh (bf16<->fp32, fp32_mul, fp32_add).
//   * Synchronous active-high reset; every reg assigned on every path (no
//     inferred latch); no combinational loop (the MAC is feed-forward through
//     the glm_fp functions, registered between the two pipe stages and into the
//     accumulator).  Passes verilator --lint-only -Wall and yosys check.
//============================================================================
module glm_matmul #(
    parameter integer M    = 8,   // output rows (tile)
    parameter integer N    = 8,   // output cols (tile)
    parameter integer K    = 8,   // reduction depth (tile)
    parameter integer PE_M = M,   // array rows  (resident output-tile rows)
    parameter integer PE_N = N    // array cols  (resident output-tile cols)
)(
    input  wire                 clk,
    input  wire                 rst,
    input  wire                 start,
    // K-streaming operand input (unit pulls one beat/cycle)
    output reg                  a_req,
    input  wire [M*16-1:0]      a_col,    // A[:,k]  (M bf16 lanes)
    input  wire [N*16-1:0]      w_row,    // W[k,:]  (N bf16 lanes)
    input  wire                 ab_valid,
    // output stream (one C row / cycle)
    output reg                  c_valid,
    output reg  [N*16-1:0]      c_row,
    // status
    output reg                  busy,
    output reg                  done
);
    // ------------------------------------------------------------------
    // This verifiable slice keeps the whole MxN output tile resident in the
    // PE array (PE_M==M, PE_N==N).  The parameters PE_M/PE_N are exposed so a
    // larger problem is decomposed by the surrounding sequencer into resident
    // tiles; within one issued tile the array spans the full M x N.
    // ------------------------------------------------------------------
    // verilator lint_off UNUSEDPARAM
    localparam integer PE_M_U = PE_M;  // documented knobs; tile is MxN-resident
    localparam integer PE_N_U = PE_N;
    // verilator lint_on UNUSEDPARAM

    // counter widths (one extra bit so the ==K / ==M compares are exact)
    localparam integer KCW = (K <= 1) ? 1 : $clog2(K);
    localparam integer MCW = (M <= 1) ? 1 : $clog2(M);
    localparam [KCW:0] LAST_K = (KCW+1)'(K-1);
    localparam [MCW:0] LAST_M = (MCW+1)'(M-1);

    // FSM
    localparam [2:0] S_IDLE   = 3'd0,
                     S_STREAM = 3'd1,  // pull K beats, run the MAC pipeline
                     S_DRAIN  = 3'd2,  // flush the 2-stage MAC pipeline
                     S_OUT    = 3'd3,  // stream M C-rows out
                     S_DONE   = 3'd4;
    reg [2:0]      state;
    reg [KCW:0]    kbeat;     // K-beat counter
    reg [MCW:0]    orow;      // output-row counter (stream-out)
    reg [1:0]      drain;     // MAC pipeline drain counter

    integer pi, pj;

    // ---- FP32 accumulators: PE[pi][pj] owns C[pi][pj] ----
    reg [31:0] acc [0:M-1][0:N-1];

    // ---- MAC pipeline stage-0 (multiply) registers + valid ----
    reg        mul_v;                 // a product is in the stage-0 registers
    reg [31:0] prod [0:M-1][0:N-1];   // registered bf16*bf16 -> fp32 products

    // ------------------------------------------------------------------
    // Combinational stage-0 products of the CURRENT operand beat: for every
    // PE, prod_c[pi][pj] = fp32( A[pi][kbeat] * W[kbeat][pj] ).  Registered
    // into `prod` when a beat is accepted.
    // ------------------------------------------------------------------
    reg [31:0] prod_c [0:M-1][0:N-1];
    always @* begin
        for (pi = 0; pi < M; pi = pi + 1)
            for (pj = 0; pj < N; pj = pj + 1)
                prod_c[pi][pj] =
                    fp32_mul(bf16_to_fp32(a_col[16*pi +: 16]),
                             bf16_to_fp32(w_row[16*pj +: 16]));
    end

    // ------------------------------------------------------------------
    // Combinational pack of output row `orow`: round each of that row's N fp32
    // accumulators to bf16.  (Driven onto c_row when c_valid is asserted.)
    // ------------------------------------------------------------------
    reg [N*16-1:0] crow_c;
    always @* begin
        crow_c = {N*16{1'b0}};
        for (pj = 0; pj < N; pj = pj + 1)
            crow_c[16*pj +: 16] = fp32_to_bf16(acc[orow[MCW-1:0]][pj]);
    end

    // ------------------------------------------------------------------
    // Sequential core.
    // ------------------------------------------------------------------
    always @(posedge clk) begin
        if (rst) begin
            state   <= S_IDLE;
            a_req   <= 1'b0;
            c_valid <= 1'b0;
            c_row   <= {N*16{1'b0}};
            busy    <= 1'b0;
            done    <= 1'b0;
            kbeat   <= {(KCW+1){1'b0}};
            orow    <= {(MCW+1){1'b0}};
            drain   <= 2'd0;
            mul_v   <= 1'b0;
            for (pi = 0; pi < M; pi = pi + 1)
                for (pj = 0; pj < N; pj = pj + 1) begin
                    acc[pi][pj]  <= 32'b0;
                    prod[pi][pj] <= 32'b0;
                end
        end else begin
            // defaults (every reg gets a value every cycle -> no latch)
            done    <= 1'b0;
            c_valid <= 1'b0;
            a_req   <= 1'b0;

            // -----------------------------------------------------------
            // MAC pipeline stage-1 (ACCUMULATE): whenever a product is sitting
            // in the stage-0 registers (mul_v), add it into the accumulators.
            // This runs every cycle, independent of the FSM, so the accumulate
            // of beat k overlaps the multiply of beat k+1 (full throughput) and
            // the fp32_add sits on its own clock cycle (high fmax).
            // -----------------------------------------------------------
            if (mul_v) begin
                for (pi = 0; pi < M; pi = pi + 1)
                    for (pj = 0; pj < N; pj = pj + 1)
                        acc[pi][pj] <= fp32_add(acc[pi][pj], prod[pi][pj]);
            end
            // mul_v de-asserts by default; re-asserted below when a beat is
            // multiplied this cycle.
            mul_v <= 1'b0;

            case (state)
                // -------------------------------------------------------
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy  <= 1'b1;
                        kbeat <= {(KCW+1){1'b0}};
                        orow  <= {(MCW+1){1'b0}};
                        drain <= 2'd0;
                        mul_v <= 1'b0;
                        for (pi = 0; pi < M; pi = pi + 1)
                            for (pj = 0; pj < N; pj = pj + 1)
                                acc[pi][pj] <= 32'b0;
                        a_req <= 1'b1;          // request first K-beat
                        state <= S_STREAM;
                    end
                end
                // -------------------- STREAM K-BEATS -------------------
                // Pull one (A-col k, W-row k) beat per cycle; on each accepted
                // beat register the MxN products (stage 0) and advance k.
                S_STREAM: begin
                    a_req <= 1'b1;              // keep asking until accepted
                    if (ab_valid) begin
                        for (pi = 0; pi < M; pi = pi + 1)
                            for (pj = 0; pj < N; pj = pj + 1)
                                prod[pi][pj] <= prod_c[pi][pj];
                        mul_v <= 1'b1;          // product latched -> accumulate next cyc
                        if (kbeat == LAST_K) begin
                            a_req <= 1'b0;
                            drain <= 2'd1;       // one more accumulate to flush
                            state <= S_DRAIN;
                        end else begin
                            kbeat <= kbeat + 1'b1;
                        end
                    end
                end
                // ---------------------- DRAIN MAC ----------------------
                // The last beat's product was registered on the cycle we left
                // S_STREAM; its accumulate happens on the FIRST S_DRAIN cycle
                // (mul_v was set then).  One drain cycle guarantees that final
                // accumulate has committed before we read the accumulators.
                S_DRAIN: begin
                    if (drain != 2'd0)
                        drain <= drain - 2'd1;
                    else begin
                        orow  <= {(MCW+1){1'b0}};
                        state <= S_OUT;
                    end
                end
                // -------------------- STREAM C-ROWS --------------------
                // Emit one bf16 C row per cycle, in row order.
                S_OUT: begin
                    c_row   <= crow_c;
                    c_valid <= 1'b1;
                    if (orow == LAST_M) begin
                        state <= S_DONE;
                    end else begin
                        orow <= orow + 1'b1;
                    end
                end
                // -------------------------- DONE -----------------------
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
