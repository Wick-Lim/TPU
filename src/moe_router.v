`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// moe_router.v  --  GLM-5.2 / DeepSeek-v3 MoE ROUTER  (ACCEL_GLM52 §5)
//----------------------------------------------------------------------------
// FUNCTION
//   Per token, route to the TOP-8 of 256 experts using SIGMOID gating, then
//   RENORMALIZE the selected gates and SCALE by routed_scaling_factor (2.5):
//
//       logits = x @ W_g          (W_g : HIDDEN x N_EXPERT, GEMV, K=HIDDEN)
//       gate   = sigmoid(logits)  (elementwise, N_EXPERT)
//       (idx, g) = TOP-K(gate)    (K=TOPK, lower-index tie-break)
//       s      = Σ_{j in TopK} gate_j
//       w_j    = (gate_j / s) * SCALE      for each selected j
//       OUTPUT : {idx_j, bf16(w_j)}  for the TOPK routed experts.
//
//   This is the EXACT GLM/DeepSeek-v3 order: sigmoid-gate -> top-k ->
//   RENORMALIZE-THEN-SCALE (NOT scale-then-renormalize, NOT softmax).  The
//   shared expert is always-on and handled OUTSIDE this unit (§5).
//
//   bf16 storage (x, W_g, weights out); ALL gating/renorm math in FP32 via the
//   glm_fp / glm_fp_pipe numerics contract (§6).  Reuses the foundation blocks
//   unchanged: glm_matmul_pipe (GEMV), glm_act (SIGMOID), topk_select (top-K),
//   glm_fp (fp32 add / rsqrt-reciprocal / mul).  Nothing re-implemented.
//
//----------------------------------------------------------------------------
// DATAFLOW / FSM  (deterministic, data-independent latency)
//   S_IDLE : wait for `start` (x_vec is latched into xbuf by the GEMV stream).
//   S_MMP  : prime -- glm_matmul_pipe.start asserted the prior cycle; let its
//            internal `streaming` go live before the first K-beat.
//   S_MM   : stream K=HIDDEN beats into the GEMV.  Each beat presents
//              a_col = x[k]  (PE_M=1 lane) ,  w_row = W_g[*, k]  (N_EXPERT lanes)
//            and asserts the WEIGHT-PULL request (w_req/w_k) so the surrounding
//            system answers w_col = column k of W_g combinationally that cycle
//            (mirrors swiglu_expert's pull interface).
//   S_MMW  : wait the GEMV drain; on out_valid latch the N_EXPERT bf16 logits
//            and fire one sigmoid beat (glm_act consumes all N_EXPERT lanes).
//   S_ACT  : wait LAT_ACT for sigmoid; latch the N_EXPERT bf16 gates, widen to
//            fp32 (gate_f), and pulse topk_select.start.
//   S_TKL  : feed topk's score-pull: it requests scores 1/cycle (LANES_IN=1);
//            we answer score_in = gate_f[load index].
//   S_TKW  : wait topk.done; capture the TOPK indices + the TOPK selected fp32
//            gates (topk returns them in descending score order).
//   S_SUM  : fire the renorm add-tree over the TOPK selected gates (pipelined
//            fp32_add_pipe tree, ceil(log2 TOPK) levels) -> s = Σ gate_j.
//   S_RCP  : reciprocal r = 1/s  (fp32_rsqrt_pipe then square: 1/s=rsqrt(s)^2),
//            and precompute rs = r * SCALE = SCALE/s once (single multiply).
//   S_MUL  : per selected gate, w_j = gate_j * rs = gate_j * (SCALE/s); narrow
//            each to bf16.  RENORM (÷s) THEN SCALE (×2.5) folded into one mul by
//            the shared (SCALE/s) factor -- algebraically identical, fewer ops.
//   S_DONE : pulse `done`; outputs {sel_idx, weight_bf16} held until next start.
//
//   Every stage is a fixed-latency pipe + counters: the whole router latency is
//   data-independent and computable (see LATENCY).  busy high throughout.
//
//----------------------------------------------------------------------------
// RENORM-THEN-SCALE SEQUENCING (the load-bearing order; do NOT reorder)
//   1. gate_j  = sigmoid(logit_j)            (fp32, from glm_act, widened)
//   2. TOP-K picks the K largest gate_j      (topk_select, indices + gates)
//   3. s       = Σ_{j in TopK} gate_j        (fp32 add tree)            <- SUM
//   4. r       = 1 / s                       (fp32, rsqrt^2)            <- RECIP
//   5. rs      = r * SCALE                   (= 2.5 / s, one fp32 mul)
//   6. w_j     = gate_j * rs                 (= (gate_j/s)*SCALE)       <- SCALE
//   7. weight_j= bf16(w_j)                   (RNE narrow)
//   The renormalization (step 3-4, divide by the SUM of the SELECTED gates) is
//   done BEFORE the scale (step 5-6), exactly per GLM/DeepSeek-v3.  Selecting
//   first (step 2) then summing only the K winners (step 3) is what makes the K
//   routed weights sum to SCALE (=2.5), not 1.
//
//----------------------------------------------------------------------------
// LATENCY (deterministic)
//   GEMV pass   : HIDDEN + L + TREE_GEMM + 1   (L=`FP_MAC_LAT, TREE_GEMM=3*ADD)
//   sigmoid     : LAT_ACT (=5)
//   top-K       : TOPK load beats + KEFF extract passes + handshake (topk FSM)
//   renorm      : TREE_SUM (ceil(log2 TOPK)*ADD) + RSQRT + 2*MUL + few control
//   All fixed counts -> the router exposes a fixed latency for given params.
//
//----------------------------------------------------------------------------
// STYLE
//   synchronous ACTIVE-HIGH reset; NO latch (every reg assigned on every path);
//   NO combinational loop (all FP feedback rides pipeline registers; the renorm
//   tree/rsqrt/mul are feed-forward glm_fp_pipe instances).  Reuses
//   glm_matmul_pipe + glm_act + topk_select + glm_fp(_pipe) UNCHANGED.
//============================================================================
module moe_router #(
    parameter integer HIDDEN  = 128,           // model hidden size (scales to 6144)
    parameter integer N_EXPERT= 8,             // routed experts (real 256)
    parameter integer TOPK    = 2,             // experts per token (real 8)
    parameter [31:0]  SCALE   = 32'h40200000,  // routed_scaling_factor 2.5 (fp32)
    parameter integer KMAX    = 16384,         // >= HIDDEN (matmul K counter)
    // IDXW DERIVED ($clog2(N_EXPERT)); exposed only to size the index port.
    // Do NOT override -- always leave default.
    parameter integer IDXW    = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT)
)(
    input  wire                       clk,
    input  wire                       rst,        // sync, active-high

    // ---- control handshake ----
    input  wire                       start,      // begin one token route
    output reg                        busy,
    output reg                        done,       // 1-cycle pulse, outputs valid

    // ---- token input (consumed by the GEMV stream) ----
    input  wire [16*HIDDEN-1:0]       x_vec,      // HIDDEN bf16 token, packed

    // ---- gate-weight pull (mirror swiglu_expert's w_req/w_k/w_col pattern) ----
    //   The router drives, every K-beat, a request for column `w_k` of W_g; the
    //   system answers COMBINATIONALLY the same cycle with the N_EXPERT bf16
    //   lanes:  w_col[16*e +: 16] = W_g[ w_k , e ]   (e = expert 0..N_EXPERT-1).
    output wire                       w_req,      // need a W_g column this cycle
    output wire [$clog2(KMAX+1)-1:0]  w_k,        // reduction index k (= row of W_g)
    input  wire [16*N_EXPERT-1:0]     w_col,      // N_EXPERT bf16 lanes = W_g[k,*]

    // ---- routed result (held from done until next start) ----
    output reg  [TOPK*IDXW-1:0]       sel_idx,    // TOPK expert indices (slot 0=top)
    output reg  [TOPK*16-1:0]         sel_weight  // TOPK bf16 routed weights
);
    `include "glm_fp.vh"

    // ---------------- derived latencies / sizes (documentation + control) ----
    /* verilator lint_off UNUSEDPARAM */
    localparam integer L          = `FP_MAC_LAT;        // matmul mac drain (7)
    localparam integer TREE_GEMM  = 3 * `FP_ADD_LAT;    // matmul reduce tree (15)
    localparam integer GEMV_LAT   = HIDDEN + L + TREE_GEMM + 1;
    localparam integer LAT_ACT    = 5;                  // glm_act fixed latency (doc)
    /* verilator lint_on UNUSEDPARAM */
    localparam integer KW         = $clog2(KMAX+1);

    // renorm add-tree depth: ceil(log2 TOPK) levels of fp32_add_pipe.  TOPK is a
    // small power-of-two in practice (2 / 8); we build a CONSTANT-bounded tree.
    localparam integer SUMLEV = (TOPK <= 1) ? 1 : $clog2(TOPK);

    // ===================================================================
    //  Token buffer (latched at start; streamed into the GEMV).
    // ===================================================================
    reg [15:0] xbuf [0:HIDDEN-1];
    integer xi;
    always @(posedge clk) begin
        if (start) begin
            for (xi = 0; xi < HIDDEN; xi = xi + 1)
                xbuf[xi] <= x_vec[16*xi +: 16];
        end
    end

    // ===================================================================
    //  FSM state
    // ===================================================================
    localparam [3:0] S_IDLE = 4'd0,
                     S_MMP  = 4'd1,   // prime: let matmul `streaming` go live
                     S_MM   = 4'd2,   // stream K=HIDDEN beats into the GEMV
                     S_MMW  = 4'd3,   // wait GEMV drain -> sigmoid launch
                     S_ACT  = 4'd4,   // wait sigmoid -> gates -> topk start
                     S_TKL  = 4'd5,   // feed topk score-pull, wait done
                     S_SUM  = 4'd6,   // renorm add-tree (Σ selected gates)
                     S_RCP  = 4'd7,   // reciprocal + (1/s)*SCALE precompute
                     S_MUL  = 4'd8,   // per-gate * (SCALE/s) -> bf16 weights
                     S_DONE = 4'd9;
    reg [3:0]  state;
    reg [KW-1:0] kcnt;               // K beat counter (GEMV stream)

    // ===================================================================
    //  (1) GEMV : logits = x @ W_g   (PE_M=1 token row, PE_N=N_EXPERT logits)
    // ===================================================================
    reg          mm_start;
    reg  [KW-1:0] mm_k_len;
    wire             mm_ov;
    wire [16*N_EXPERT-1:0] mm_c;     // 1 x N_EXPERT bf16 logits tile

    // streaming operand + weight-pull drive (continuous assigns: a_col/w_row/
    // in_valid and the w_req/w_k request all describe the SAME beat, coherent
    // with the system's combinational w_col answer -- no a_col/w_row skew).
    // Declared BEFORE the GEMV instance so iverilog binds them (decl-before-use).
    wire stream_mm = (state == S_MM);
    localparam integer XIW = (HIDDEN   > 1) ? $clog2(HIDDEN)   : 1;
    localparam integer EIW = (N_EXPERT > 1) ? $clog2(N_EXPERT) : 1;
    wire [XIW-1:0]         x_idx = kcnt[XIW-1:0];
    wire [15:0]            a_col = stream_mm ? xbuf[x_idx] : 16'b0;
    wire [16*N_EXPERT-1:0] w_row = w_col;          // system weight response
    wire                   mm_in_valid = stream_mm;
    assign w_req = stream_mm;
    assign w_k   = kcnt;

    // mm_busy is unused (FSM gates on out_valid + deterministic drain). Waived.
    /* verilator lint_off UNUSEDSIGNAL */
    wire             mm_busy;
    glm_matmul_pipe #(.PE_M(1), .PE_N(N_EXPERT), .KMAX(KMAX)) u_gemv (
        .clk(clk), .rst(rst),
        .start(mm_start), .k_len(mm_k_len),
        .in_valid(mm_in_valid), .a_col(a_col), .w_row(w_row),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_on UNUSEDSIGNAL */

    // ===================================================================
    //  (2) SIGMOID gating : gate = sigmoid(logits)  (N_EXPERT lanes, 1 beat)
    // ===================================================================
    reg                    act_in_valid;
    reg  [16*N_EXPERT-1:0] act_x_in;     // logits fed to sigmoid
    wire                   act_ov;
    wire [16*N_EXPERT-1:0] act_y;        // sigmoid(logits), bf16
    glm_act #(.MODE(0), .LANES(N_EXPERT)) u_sigmoid (  // MODE=0 => SIGMOID
        .clk(clk), .rst(rst),
        .in_valid(act_in_valid), .x_in(act_x_in),
        .out_valid(act_ov), .y_out(act_y)
    );

    // fp32 widened gates (topk scores + renorm operands).  Captured at act_ov.
    reg [31:0] gate_f [0:N_EXPERT-1];

    // ===================================================================
    //  (3) TOP-K : pick the TOPK largest gates -> indices + selected gates.
    //      topk pulls scores 1/cycle (LANES_IN=1); we answer gate_f[load idx].
    // ===================================================================
    reg               tk_start;
    reg               tk_score_valid;
    reg  [31:0]       tk_score_in;
    wire              tk_load_req;
    wire [TOPK*IDXW-1:0]  tk_sel_idx;
    wire [TOPK*32-1:0]    tk_sel_score;   // selected gates (fp32), descending
    // sel_valid / mask / busy unused here (TOPK<=N_EXPERT always, FSM gates on
    // done).  Waived.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [TOPK-1:0]       tk_sel_valid;
    wire [N_EXPERT-1:0]   tk_mask;
    wire                  tk_busy, tk_done;
    /* verilator lint_on UNUSEDSIGNAL */
    topk_select #(.N(N_EXPERT), .K(TOPK), .SCORE_W(32), .LANES_IN(1)) u_topk (
        .clk(clk), .rst(rst), .start(tk_start),
        .load_req(tk_load_req), .score_in(tk_score_in),
        .score_valid(tk_score_valid),
        .sel_idx_o(tk_sel_idx), .sel_score_o(tk_sel_score),
        .sel_valid_o(tk_sel_valid), .mask_o(tk_mask),
        .busy(tk_busy), .done(tk_done)
    );

    // score-load address counter (which gate_f we hand topk this beat).
    // EIW+1 bits so it counts 0..N_EXPERT cleanly (one spare bit for the == N).
    reg [EIW:0] tk_addr;             // expert load index 0..N_EXPERT
    // captured TOPK winners (indices + fp32 gates) at topk.done
    reg [IDXW-1:0] win_idx [0:TOPK-1];
    reg [31:0]     win_gate[0:TOPK-1];

    // ===================================================================
    //  (4) RENORM add-tree : s = Σ_{j in TopK} win_gate[j]   (fp32_add_pipe).
    //      A balanced binary tree of SUMLEV levels.  We allocate the worst-case
    //      node count (TOPK-1 adders) as a flat generate and chain by levels.
    // ===================================================================
    reg sum_go;                      // 1-cycle pulse: launch the add-tree

    // Tree node arrays.  level 0 = the TOPK leaves (win_gate), padded to a power
    // of two with +0.0.  We instantiate one fp32_add_pipe per internal node.
    localparam integer NPOW = (1 << SUMLEV);          // padded leaf count >= TOPK
    // node storage: lvl-wide arrays.  lvl[l] has NPOW>>l entries.
    // We carry data through a packed 2D via generate; simplest correct form is a
    // per-level wire array sized NPOW (upper entries unused at deeper levels).
    wire [31:0] tnode_y [0:SUMLEV][0:NPOW-1];   // tnode_y[l][i] = node value
    wire        tnode_v [0:SUMLEV][0:NPOW-1];   // its valid

    // level 0 leaves: win_gate padded with +0.0; valid = sum_go pulse.
    genvar gl, gi;
    generate
      for (gi = 0; gi < NPOW; gi = gi + 1) begin : LEAF
        assign tnode_y[0][gi] = (gi < TOPK) ? win_gate[gi] : 32'h0000_0000;
        assign tnode_v[0][gi] = sum_go;
      end
      // internal levels: each node = add of its two children (one fp32_add_pipe).
      for (gl = 0; gl < SUMLEV; gl = gl + 1) begin : LVL
        for (gi = 0; gi < (NPOW >> (gl+1)); gi = gi + 1) begin : NODE
          /* verilator lint_off UNUSEDSIGNAL */
          wire nv;                  // representative valid (siblings lockstep)
          /* verilator lint_on UNUSEDSIGNAL */
          fp32_add_pipe u_add (
            .clk(clk), .rst(rst), .valid_in(tnode_v[gl][2*gi]),
            .a(tnode_y[gl][2*gi]), .b(tnode_y[gl][2*gi+1]),
            .valid_out(nv), .result(tnode_y[gl+1][gi])
          );
          assign tnode_v[gl+1][gi] = nv;
        end
        // pad the unused upper slots of this output level (deeper levels read
        // only [0 .. (NPOW>>(l+1))-1]; tie the rest off so they are defined).
        for (gi = (NPOW >> (gl+1)); gi < NPOW; gi = gi + 1) begin : PAD
          assign tnode_y[gl+1][gi] = 32'h0000_0000;
          assign tnode_v[gl+1][gi] = 1'b0;
        end
      end
    endgenerate

    wire [31:0] sum_y = tnode_y[SUMLEV][0];   // s = Σ selected gates (fp32)
    wire        sum_v = tnode_v[SUMLEV][0];   // tree result valid
    reg  [31:0] s_reg;                        // latched s

    // ===================================================================
    //  (5) RECIPROCAL : r = 1/s  via rsqrt(s)^2  (pipelined fp32_rsqrt_pipe).
    // ===================================================================
    reg          rcp_go;
    wire         rsq_v;
    wire [31:0]  rsq_y;            // rsqrt(s)
    fp32_rsqrt_pipe u_rsqrt (
        .clk(clk), .rst(rst), .valid_in(rcp_go),
        .x(s_reg), .valid_out(rsq_v), .result(rsq_y)
    );
    // r = rsqrt(s)^2 = 1/s ; then rs = r * SCALE = SCALE/s.  Both via combinational
    // glm_fp.vh muls registered the cycle rsq_v lands (feed-forward, no loop).
    reg  [31:0]  rs_reg;          // SCALE / s   (the single fold factor)

    // ===================================================================
    //  Main FSM + datapath
    // ===================================================================
    integer i, t;
    // combinational reciprocal fold (registered below at rsq_v):
    reg [31:0] c_recip, c_rs;
    always @* begin
        c_recip = fp32_mul(rsq_y, rsq_y);     // 1/s
        c_rs    = fp32_mul(c_recip, SCALE);   // SCALE/s
    end

    always @(posedge clk) begin
        if (rst) begin
            state          <= S_IDLE;
            busy           <= 1'b0;
            done           <= 1'b0;
            kcnt           <= {KW{1'b0}};
            mm_start       <= 1'b0;
            mm_k_len       <= {KW{1'b0}};
            act_in_valid   <= 1'b0;
            act_x_in       <= {16*N_EXPERT{1'b0}};
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            tk_score_in    <= 32'b0;
            tk_addr        <= {(EIW+1){1'b0}};
            sum_go         <= 1'b0;
            rcp_go         <= 1'b0;
            s_reg          <= 32'b0;
            rs_reg         <= 32'b0;
            sel_idx        <= {TOPK*IDXW{1'b0}};
            sel_weight     <= {TOPK*16{1'b0}};
            for (i = 0; i < N_EXPERT; i = i + 1) gate_f[i] <= 32'b0;
            for (i = 0; i < TOPK; i = i + 1) begin
                win_idx[i]  <= {IDXW{1'b0}};
                win_gate[i] <= 32'b0;
            end
        end else begin
            // ---- defaults (deassert pulses) ----
            done           <= 1'b0;
            mm_start       <= 1'b0;
            act_in_valid   <= 1'b0;
            tk_start       <= 1'b0;
            tk_score_valid <= 1'b0;
            sum_go         <= 1'b0;
            rcp_go         <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy     <= 1'b1;
                    kcnt     <= {KW{1'b0}};
                    mm_start <= 1'b1;                 // start the GEMV
                    mm_k_len <= HIDDEN[KW-1:0];
                    state    <= S_MMP;
                end
            end

            // ---- prime: matmul `streaming` goes live this cycle ----
            S_MMP: begin
                kcnt  <= {KW{1'b0}};
                state <= S_MM;
            end

            // ---- stream K=HIDDEN beats into the GEMV (operands driven comb) ----
            S_MM: begin
                if (kcnt == HIDDEN[KW-1:0] - 1'b1)
                    state <= S_MMW;
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait GEMV drain; on out_valid launch sigmoid ----
            S_MMW: begin
                if (mm_ov) begin
                    act_in_valid <= 1'b1;
                    act_x_in     <= mm_c;             // logits -> sigmoid
                    state        <= S_ACT;
                end
            end

            // ---- wait sigmoid; on out_valid latch gates (fp32), start topk ----
            S_ACT: begin
                if (act_ov) begin
                    for (i = 0; i < N_EXPERT; i = i + 1)
                        gate_f[i] <= bf16_to_fp32(act_y[16*i +: 16]);
                    tk_start <= 1'b1;                 // begin top-K
                    tk_addr  <= {(EIW+1){1'b0}};
                    state    <= S_TKL;
                end
            end

            // ---- feed topk score-pull (1 score/cycle), wait for topk.done ----
            // topk asserts load_req while it wants the next score; we answer
            // gate_f[tk_addr] that cycle (score_valid + the score) and advance.
            // After all N_EXPERT scores are loaded topk runs its extract passes;
            // we hold here until tk_done, then capture the winners.
            S_TKL: begin
                if (tk_load_req) begin
                    tk_score_valid <= 1'b1;
                    tk_score_in    <= gate_f[tk_addr[EIW-1:0]];
                    tk_addr        <= tk_addr + 1'b1;
                end
                if (tk_done) begin
                    for (t = 0; t < TOPK; t = t + 1) begin
                        win_idx[t]  <= tk_sel_idx[IDXW*t +: IDXW];
                        win_gate[t] <= tk_sel_score[32*t +: 32];
                    end
                    sum_go <= 1'b1;                   // launch renorm add-tree
                    state  <= S_SUM;
                end
            end

            // ---- wait add-tree -> s ; then launch reciprocal ----
            S_SUM: begin
                if (sum_v) begin
                    s_reg  <= sum_y;                  // s = Σ selected gates
                    rcp_go <= 1'b1;                   // launch rsqrt(s)
                    state  <= S_RCP;
                end
            end

            // ---- wait rsqrt; compute rs = SCALE/s ; then per-gate multiply ----
            S_RCP: begin
                if (rsq_v) begin
                    rs_reg <= c_rs;                   // SCALE / s (one fold factor)
                    state  <= S_MUL;
                end
            end

            // ---- per selected gate: w_j = gate_j * (SCALE/s) -> bf16 ----
            S_MUL: begin
                for (t = 0; t < TOPK; t = t + 1) begin
                    sel_idx[IDXW*t +: IDXW] <= win_idx[t];
                    sel_weight[16*t +: 16]  <=
                        fp32_to_bf16(fp32_mul(win_gate[t], rs_reg));
                end
                state <= S_DONE;
            end

            // ---- done ----
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
/* verilator lint_on DECLFILENAME */
