`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// swiglu_expert_fp8.v  --  GLM-5.2 SwiGLU FFN EXPERT, FP8-NATIVE MATMULS
//                          (the FP8 sibling of swiglu_expert.v)        (§5,§6)
//----------------------------------------------------------------------------
// FUNCTION  (identical math to swiglu_expert.v; only the GEMM numerics change)
//   Given a single token's hidden vector x (HIDDEN bf16) and this expert's three
//   weight matrices (now FP8 E4M3 + [128,128] block scales), compute the SwiGLU
//   FFN output y (HIDDEN bf16):
//
//       gate = W_gate @ x      (W_gate : INTER x HIDDEN  -> INTER outputs)
//       up   = W_up   @ x      (W_up   : INTER x HIDDEN  -> INTER outputs)
//       h    = silu(gate) (.) up        (elementwise, INTER)
//       y    = W_down @ h      (W_down : HIDDEN x INTER  -> HIDDEN outputs)
//
//   THE FP8 SPLIT (this module's whole reason to exist):
//     * The THREE GEMMs (gate, up, down) run through glm_matmul_fp8 -- the
//       official GLM-5.2-FP8 numerics: weights are E4M3 carrying a [128,128]
//       BLOCK scale (DeepSeek-V3 weight_block_size=[128,128]); activations are
//       bf16 and DYNAMICALLY quantized to E4M3 with a per-vector (per-token)
//       power-of-two scale; products are 4x4-mantissa fp8 muls, fp32-accumulated
//       per K-block, block-scaled, then de-scaled and rounded to bf16.
//     * The "tail" stays bf16: silu(gate) via glm_act #(SILU) and the
//       elementwise silu(gate) (.) up via bf16_mul (glm_fp.vh).  Only the matmuls
//       are FP8; the activation/merge are the SAME bf16 path as swiglu_expert.v.
//
//----------------------------------------------------------------------------
// DYNAMIC ACTIVATION QUANT (per-vector pow2 scale; activation_scheme=dynamic)
//   glm_matmul_fp8 takes a per-row signed-8 pow2 exponent a_shift and prescales
//   the bf16 activation by 2^a_shift before E4M3 encode (undoing 2^-a_shift at
//   dequant -- exact, a free exponent add).  Because PE_M=1 (one token row) each
//   matmul pass needs ONE a_shift for the whole activation vector.  We pick it
//   so the LARGEST-magnitude element lands near E4M3's top of range:
//       a_shift = clamp_{[-128,127]}( 134 - emax )      (emax = max bf16 exp field)
//   i.e. the max element's scaled exponent becomes 7 (value in [128,256), well
//   under the 448 E4M3 max -> no saturation), giving the small elements ~2^16 of
//   headroom before they underflow.  emax==0 (all-zero vector) -> a_shift = 0.
//   * x's a_shift (xsh) is reduced combinationally from x_vec, latched at start.
//   * h's a_shift (hsh) is reduced INCREMENTALLY as the h tiles are written
//     (running max exponent h_emax), so it is final the cycle the DOWN pass
//     starts.  Both are pure exponent-field maxes -> no multiplier.
//
//----------------------------------------------------------------------------
// MAPPING / SEQUENCING  (UNCHANGED from swiglu_expert.v)
//   PE_M=1 token row, PE_N=TN output columns per tile-group.  GATE+UP run in
//   lockstep (two glm_matmul_fp8 instances, identical control/a_col/a_shift,
//   different weights+scales) over K=HIDDEN; DOWN reduces over K=INTER.  The FSM
//   (IDLE->GUP->GU->GUW->DNP->DN->DNW->DONE) is purely handshake-driven off the
//   matmul out_valid + glm_act out_valid, so the (different) FP8 matmul latency
//   is absorbed automatically -- no latency constant is hardcoded in the FSM.
//
//----------------------------------------------------------------------------
// WEIGHT DELIVERY (pull; FP8 + block scales)
//   Per-beat code request (combinational, same cycle as the beat sampled):
//       w_req            : high on a beat needing weights (== matmul in_valid)
//       w_sel (0=GATE,2=DOWN) , w_grp (tile-group) : REGISTERED pass descriptor,
//                          valid from the pass prime through its whole stream.
//       w_k              : reduction index of this beat.
//     System answers COMBINATIONALLY with the FP8 codes for (sel,grp,k):
//       w_col   [8*TN] : E4M3 W_{gate|down}[grp*TN+t , k]
//       w_col_up[8*TN] : E4M3 W_up[grp*TN+t , k]   (GATE/UP pass only)
//   Per-pass BLOCK scales (combinational from the SAME w_sel/w_grp; latched by
//   the matmul at its start):
//       w_scale_g[16*TN*NB] : bf16 block scale per (col pj, K-block bj) for the
//                             gate(or down) tile group, packed [16*(bj*TN+pj)+:16]
//       w_scale_u[16*TN*NB] : same, for the up tile group.
//   (NB = ceil(KMAX/BLK) K-blocks.  Scales for K-blocks past k_len are don't-care
//    -- those banks stay zero, 0*scale=0.)
//
//----------------------------------------------------------------------------
// LATENCY (deterministic, data-independent)
//   Per FP8 matmul pass:  PASS(K) = K + L + TREE_LAT + (PE_M*PE_N) + 1
//                                  = K + `FP_ADD_LAT + 3*`FP_ADD_LAT + TN + 1.
//   Merge tail = LAT_ACT + 1.  Total = NG_GU*(PASS(HIDDEN)+merge+ctl)
//                                     + NG_D*(PASS(INTER)+ctl) + tail.  Fixed.
//
//----------------------------------------------------------------------------
// STYLE: synchronous ACTIVE-HIGH reset; NO latch; NO comb loop (all feedback
//   rides the matmul/act pipeline registers); reuses glm_matmul_fp8 + glm_act +
//   glm_fp unchanged.  DENSE vs MoE = the INTER parameter only.
//============================================================================
module swiglu_expert_fp8 #(
    parameter integer HIDDEN = 128,   // model hidden size (scales to 6144)
    parameter integer INTER  = 64,    // FFN inter size (MoE 2048 / dense 12288)
    parameter integer TN     = 4,     // output-tile width = matmul PE_N
    parameter integer KMAX   = 16384, // >= max(HIDDEN, INTER) for matmul counter
    parameter integer BLK    = 128    // weight block size -- [128,128]
)(
    input  wire                     clk,
    input  wire                     rst,        // sync, active-high

    // ---- control handshake ----
    input  wire                     start,      // begin one expert evaluation
    output reg                      busy,
    output reg                      done,       // 1-cycle pulse when y_out valid

    // ---- token input (latched at start) ----
    input  wire [16*HIDDEN-1:0]     x_vec,      // HIDDEN bf16 token, packed

    // ---- weight request (to surrounding system / DMA buffer) ----
    output wire                     w_req,      // need weight codes this cycle
    output wire [1:0]               w_sel,      // 0=GATE(+UP),2=DOWN (registered)
    output wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp, // tile-group
    output wire [$clog2(KMAX+1)-1:0] w_k,       // reduction index k of this beat
    // ---- weight response: FP8 codes (combinational, same cycle as w_req) ----
    input  wire [8*TN-1:0]          w_col,      // E4M3 W_{gate|down} lanes
    input  wire [8*TN-1:0]          w_col_up,   // E4M3 W_up lanes (GATE/UP pass)
    // ---- weight response: [128,128] BLOCK scales (combinational from w_sel/w_grp,
    //      latched by the matmul at its start) ----
    input  wire [16*TN*((KMAX+BLK-1)/BLK)-1:0] w_scale_g,  // gate/down block scales
    input  wire [16*TN*((KMAX+BLK-1)/BLK)-1:0] w_scale_u,  // up block scales

    // ---- result ----
    output reg  [16*HIDDEN-1:0]     y_out       // HIDDEN bf16 result, packed
);
    `include "glm_fp.vh"

    // ---------------- derived sizes ----------------
    localparam integer KW    = $clog2(KMAX+1);
    localparam integer NG_GU = (INTER  + TN - 1) / TN;   // gate/up: OUT=INTER
    localparam integer NG_D  = (HIDDEN + TN - 1) / TN;   // down   : OUT=HIDDEN
    localparam integer GW    = $clog2((INTER>HIDDEN?INTER:HIDDEN)/TN + 1);
    // Partial-tile bound guards are needed ONLY for ragged tiles (OUT%TN!=0).
    // When OUT divides evenly (default/typical sizes) the comparator+write-mask
    // mux fold away at elaboration (DN_FULL/GU_FULL are constants).
    localparam DN_FULL = (HIDDEN % TN == 0);   // down  output (HIDDEN) tiles even?
    localparam GU_FULL = (INTER  % TN == 0);   // gate/up output (INTER) tiles even?

    // pass-select encodings (also drive w_sel)
    localparam [1:0] SEL_GATE = 2'd0, SEL_DOWN = 2'd2;

    // ===================================================================
    //  Token + intermediate buffers (the only data state this unit owns)
    // ===================================================================
    reg [15:0] xbuf [0:HIDDEN-1];     // latched token x
    reg [15:0] hbuf [0:INTER-1];      // h = silu(gate) (.) up

    // latch x at start (flat, deterministic; token is small)
    integer xi;
    always @(posedge clk) begin
        if (start) begin
            for (xi = 0; xi < HIDDEN; xi = xi + 1)
                xbuf[xi] <= x_vec[16*xi +: 16];
        end
    end

    // ===================================================================
    //  Dynamic per-vector pow2 activation scale (a_shift) -- pure exp maxes.
    // ===================================================================
    // a_shift = clamp(134 - emax) so the max element's scaled exp becomes 7.
    function automatic signed [7:0] dyn_shift(input [7:0] emax);
        reg signed [9:0] sh;
        begin
            if (emax == 8'd0) dyn_shift = 8'sd0;
            else begin
                sh = 10'sd134 - $signed({2'b0, emax});
                if (sh > 10'sd127)  sh = 10'sd127;
                if (sh < -10'sd128) sh = -10'sd128;
                dyn_shift = sh[7:0];
            end
        end
    endfunction

    // ---- x activation shift: BALANCED-TREE max over x_vec exponent fields ----
    // max is associative+commutative, so a depth-ceil(log2 HIDDEN) halving tree
    // is BIT-IDENTICAL to the old serial fold (depth HIDDEN-1).  Built statically
    // (heap-layout: node n has children 2n+1/2n+2; root = node 0), leaves padded
    // up to a power of two with 0 (>=0, never beats a real exponent).
    localparam integer EM_LV = (HIDDEN > 1) ? $clog2(HIDDEN) : 0; // tree depth
    localparam integer EM_W  = 1 << EM_LV;                        // padded leaves
    wire [8*(2*EM_W-1)-1:0] em_node;
    genvar emg;
    generate
        // leaves occupy heap indices EM_W-1 .. 2*EM_W-2
        for (emg = 0; emg < EM_W; emg = emg + 1) begin : g_emleaf
            assign em_node[8*(EM_W-1+emg) +: 8] =
                       (emg < HIDDEN) ? x_vec[16*emg + 7 +: 8] : 8'd0;
        end
        // internal nodes EM_W-2 .. 0 : pairwise max of the two children
        for (emg = 0; emg < EM_W-1; emg = emg + 1) begin : g_emnode
            wire [7:0] em_ca = em_node[8*(2*emg+1) +: 8];
            wire [7:0] em_cb = em_node[8*(2*emg+2) +: 8];
            assign em_node[8*emg +: 8] = (em_ca > em_cb) ? em_ca : em_cb;
        end
    endgenerate
    wire [7:0] xemax_c = em_node[7:0];     // root (max exponent field)
    wire signed [7:0] xsh_comb = dyn_shift(xemax_c);
    reg  signed [7:0] xsh;            // latched at start

    // ---- h activation shift: running max exponent, updated as h tiles land ----
    reg [7:0] h_emax;                 // max bf16 exp field seen across h so far
    wire signed [7:0] hsh = dyn_shift(h_emax);

    // ===================================================================
    //  FSM
    // ===================================================================
    localparam [2:0] S_IDLE = 3'd0,
                     S_GUP  = 3'd1,   // prime gate/up matmuls (streaming goes live)
                     S_GU   = 3'd2,   // stream a gate/up group
                     S_GUW  = 3'd3,   // wait gate/up matmul drain + silu/merge
                     S_DNP  = 3'd4,   // prime the down matmul
                     S_DN   = 3'd5,   // stream a down group
                     S_DNW  = 3'd6,   // wait down matmul drain
                     S_DONE = 3'd7;

    reg [2:0]    state;
    reg [GW-1:0] grp;                 // current output tile-group
    reg [KW-1:0] kcnt;                // K beat counter while streaming
    reg [1:0]    pass_sel;            // registered pass descriptor -> w_sel

    wire [KW-1:0] k_len_gu = HIDDEN[KW-1:0];
    wire [KW-1:0] k_len_dn = INTER [KW-1:0];

    // ===================================================================
    //  Shared GEMM operand drive.  PE_M = 1 (single token row).
    // ===================================================================
    reg           mm_start;           // clocked pulse (gate/down matmul)
    reg  [KW-1:0] mm_k_len;           // clocked (latched at start)
    reg           mm_start_u;         // clocked pulse (up matmul)
    wire          mm_in_valid;
    wire          mm_in_valid_u;
    wire [15:0]   a_col;              // x[k] / h[k]  (PE_M = 1 lane)
    wire [8*TN-1:0] w_row_g;          // gate-or-down E4M3 weight lanes
    wire [8*TN-1:0] w_row_u;          // up E4M3 weight lanes (gate/up pass only)
    // the active activation pow2 scale for whichever pass is starting.
    wire signed [7:0] mm_a_shift = (pass_sel == SEL_DOWN) ? hsh : xsh;

    // GATE/DOWN matmul (reused across both projections: gate then down).
    /* verilator lint_off UNUSEDSIGNAL */
    wire            g_busy, g_ov;
    wire [16*TN-1:0] g_c;             // 1 x TN bf16 tile
    glm_matmul_fp8 #(.PE_M(1), .PE_N(TN), .KMAX(KMAX), .BLK(BLK)) u_mm_g (
        .clk(clk), .rst(rst),
        .start(mm_start), .k_len(mm_k_len),
        .in_valid(mm_in_valid), .a_col(a_col), .w_row(w_row_g),
        .a_shift(mm_a_shift), .w_scale(w_scale_g),
        .busy(g_busy), .out_valid(g_ov), .c_out(g_c)
    );

    // UP matmul (lockstep with GATE during the gate/up pass; idle in down).
    wire            u_busy, u_ov;
    wire [16*TN-1:0] u_c;
    glm_matmul_fp8 #(.PE_M(1), .PE_N(TN), .KMAX(KMAX), .BLK(BLK)) u_mm_u (
        .clk(clk), .rst(rst),
        .start(mm_start_u), .k_len(mm_k_len),
        .in_valid(mm_in_valid_u), .a_col(a_col), .w_row(w_row_u),
        .a_shift(mm_a_shift), .w_scale(w_scale_u),
        .busy(u_busy), .out_valid(u_ov), .c_out(u_c)
    );
    /* verilator lint_on UNUSEDSIGNAL */

    // ===================================================================
    //  SiLU(gate) and the silu*up merge (bf16 tail, UNCHANGED from bf16 expert)
    // ===================================================================
    reg              act_in_valid;
    reg  [16*TN-1:0] act_x_in;        // gate lanes fed to silu
    wire             act_ov;
    wire [16*TN-1:0] act_y;           // silu(gate)
    glm_act #(.MODE(1), .LANES(TN)) u_silu (   // MODE=1 => SILU
        .clk(clk), .rst(rst),
        .in_valid(act_in_valid), .x_in(act_x_in),
        .out_valid(act_ov), .y_out(act_y)
    );

    reg [16*TN-1:0] up_hold;          // up tile captured at g_ov
    reg [GW-1:0]    grp_hold;         // its destination group

    integer mt;

    // ===================================================================
    //  COMBINATIONAL matmul-operand + weight-request drive.
    // ===================================================================
    wire stream_gu = (state == S_GU);
    wire stream_dn = (state == S_DN);
    localparam integer XIW = (HIDDEN > 1) ? $clog2(HIDDEN) : 1;
    localparam integer HIW = (INTER  > 1) ? $clog2(INTER)  : 1;
    wire [XIW-1:0] x_idx = kcnt[XIW-1:0];
    wire [HIW-1:0] h_idx = kcnt[HIW-1:0];
    assign a_col         = stream_gu ? xbuf[x_idx] :
                           stream_dn ? hbuf[h_idx] : 16'b0;
    assign w_row_g       = w_col;          // system FP8 weight response (gate/down)
    assign w_row_u       = w_col_up;       // up companion lanes
    assign mm_in_valid   = stream_gu | stream_dn;
    assign mm_in_valid_u = stream_gu;      // up matmul only during the gate/up pass
    // per-beat weight code request.  w_sel/w_grp are the REGISTERED pass
    // descriptor (also address the per-pass block scales); w_k is the beat index.
    assign w_req = stream_gu | stream_dn;
    assign w_sel = pass_sel;
    assign w_grp = grp[GW-1:0];
    assign w_k   = kcnt;

    // ===================================================================
    //  Main control + datapath registers
    // ===================================================================
    integer yi;
    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            grp          <= {GW{1'b0}};
            kcnt         <= {KW{1'b0}};
            pass_sel     <= SEL_GATE;
            mm_start     <= 1'b0;
            mm_start_u   <= 1'b0;
            mm_k_len     <= {KW{1'b0}};
            act_in_valid <= 1'b0;
            act_x_in     <= {16*TN{1'b0}};
            up_hold      <= {16*TN{1'b0}};
            grp_hold     <= {GW{1'b0}};
            xsh          <= 8'sd0;
        end else begin
            // ---- defaults (deassert pulses) ----
            done         <= 1'b0;
            mm_start     <= 1'b0;
            mm_start_u   <= 1'b0;
            act_in_valid <= 1'b0;

            case (state)
            // ----------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    busy       <= 1'b1;
                    grp        <= {GW{1'b0}};
                    state      <= S_GUP;
                    kcnt       <= {KW{1'b0}};
                    pass_sel   <= SEL_GATE;
                    mm_start   <= 1'b1;     // start gate matmul
                    mm_start_u <= 1'b1;     // start up   matmul
                    mm_k_len   <= k_len_gu;
                    xsh        <= xsh_comb; // latch dynamic x activation shift
                end
            end

            // ---- prime: streaming goes live this cycle; begin issuing next ----
            S_GUP: begin
                kcnt  <= {KW{1'b0}};
                state <= S_GU;
            end

            // ---- GATE/UP: stream HIDDEN K-beats into both matmuls ----
            S_GU: begin
                if (kcnt == k_len_gu - 1'b1) begin
                    state <= S_GUW;
                end
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait gate/up matmul -> silu -> *up -> h buffer ----
            S_GUW: begin
                if (g_ov) begin
                    act_in_valid <= 1'b1;
                    act_x_in     <= g_c;      // gate tile -> silu
                    up_hold      <= u_c;      // pair up tile with silu output
                    grp_hold     <= grp;
                end

                if (act_ov) begin
                    if (grp == NG_GU[GW-1:0] - 1'b1) begin
                        // all gate/up groups done -> begin down projection
                        state    <= S_DNP;
                        grp      <= {GW{1'b0}};
                        kcnt     <= {KW{1'b0}};
                        pass_sel <= SEL_DOWN;
                        mm_start <= 1'b1;
                        mm_k_len <= k_len_dn;
                    end else begin
                        grp        <= grp + 1'b1;
                        kcnt       <= {KW{1'b0}};
                        state      <= S_GUP;
                        pass_sel   <= SEL_GATE;
                        mm_start   <= 1'b1;
                        mm_start_u <= 1'b1;
                        mm_k_len   <= k_len_gu;
                    end
                end
            end

            // ---- prime the down matmul ----
            S_DNP: begin
                kcnt  <= {KW{1'b0}};
                state <= S_DN;
            end

            // ---- DOWN: stream INTER K-beats from h buffer into down matmul ----
            S_DN: begin
                if (kcnt == k_len_dn - 1'b1) begin
                    state <= S_DNW;
                end
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait down matmul -> write y tile ----
            S_DNW: begin
                if (g_ov) begin
                    for (yi = 0; yi < TN; yi = yi + 1) begin
                        if (DN_FULL || (grp*TN + yi < HIDDEN))
                            y_out[16*(grp*TN + yi) +: 16] <= g_c[16*yi +: 16];
                    end
                    if (grp == NG_D[GW-1:0] - 1'b1) begin
                        state <= S_DONE;
                    end else begin
                        grp      <= grp + 1'b1;
                        kcnt     <= {KW{1'b0}};
                        state    <= S_DNP;
                        pass_sel <= SEL_DOWN;
                        mm_start <= 1'b1;
                        mm_k_len <= k_len_dn;
                    end
                end
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

    // ===================================================================
    //  silu*up MERGE -> h buffer write + running h_emax (for hsh).
    //  Fires on act_ov; pairs silu(gate) (act_y) with the matched up tile
    //  (up_hold) for group grp_hold, writes h = bf16(silu*up), and folds the
    //  new h exponents into the running max h_emax used by the DOWN a_shift.
    // ===================================================================
    // combinational next-state: per-lane h = bf16(silu*up) and the running max
    // exponent over the (masked) lanes -- blocking is fine here (always @*).
    reg [15:0] n_hval [0:TN-1];
    reg [7:0]  n_h_emax;
    integer    mc;
    always @* begin
        n_h_emax = h_emax;
        for (mc = 0; mc < TN; mc = mc + 1) begin
            n_hval[mc] = bf16_mul(act_y[16*mc +: 16], up_hold[16*mc +: 16]);
            if (GU_FULL || (grp_hold*TN + mc < INTER))
                if (n_hval[mc][14:7] > n_h_emax) n_h_emax = n_hval[mc][14:7];
        end
    end
    // clocked register: only nonblocking writes (no latch, no BLKSEQ).
    always @(posedge clk) begin
        if (rst)        h_emax <= 8'd0;
        else if (start) h_emax <= 8'd0;
        else if (act_ov) begin
            for (mt = 0; mt < TN; mt = mt + 1)
                if (GU_FULL || (grp_hold*TN + mt < INTER))
                    hbuf[grp_hold*TN + mt] <= n_hval[mt];
            h_emax <= n_h_emax;
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
