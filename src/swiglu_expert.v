`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// swiglu_expert.v  --  GLM-5.2 SwiGLU FFN EXPERT  (ACCEL_GLM52 §5, §1.1)
//----------------------------------------------------------------------------
// FUNCTION
//   Given a single token's hidden vector x (HIDDEN bf16) and this expert's three
//   weight matrices, compute the SwiGLU FFN output y (HIDDEN bf16):
//
//       gate = W_gate @ x      (W_gate : INTER x HIDDEN  -> INTER outputs)
//       up   = W_up   @ x      (W_up   : INTER x HIDDEN  -> INTER outputs)
//       h    = silu(gate) (.) up        (elementwise, INTER)
//       y    = W_down @ h      (W_down : HIDDEN x INTER  -> HIDDEN outputs)
//
//   bf16 operands everywhere; fp32 accumulation INSIDE the GEMMs (glm_matmul_pipe
//   reduces the K dot products in fp32); silu and the silu*up multiply run in
//   fp32 and round to bf16 (glm_act + glm_fp).  This unit does NOT reimplement
//   GEMM or SiLU -- it ORCHESTRATES the foundation blocks:
//     * glm_matmul_pipe  (bf16xbf16 -> fp32-accum -> bf16 GEMM)   x3 logical passes
//     * glm_act #(MODE=SILU)                                       silu(gate)
//     * bf16_mul (glm_fp.vh)                                       silu(gate) (.) up
//
//----------------------------------------------------------------------------
// MAPPING (single token => M = 1 systolic row)
//   Each output element is a length-K dot product.  We use glm_matmul_pipe as a
//   1 x TN tile (PE_M = 1, PE_N = TN): the single A-row is the token, the TN
//   array columns own TN consecutive output elements.  On each K-beat we present
//       a_col = x[k]                  (1 bf16 lane, the token element k)
//       w_row = { W[out0..outTN-1, k] }   (TN bf16 lanes: column k of the tile)
//   and the unit folds x[k]*W[out,k] into each output's fp32 accumulator.  After
//   K beats it emits the 1 x TN bf16 tile.  We sweep the output dimension in
//   ceil(OUT/TN) tile-groups; one tile-group = one glm_matmul_pipe pass.
//
//   The GATE and UP projections share the SAME x stream and SAME K=HIDDEN, so a
//   gate matmul and an up matmul run IN LOCKSTEP (two glm_matmul_pipe instances,
//   identical control, identical a_col, different w_row) -- one combined pass
//   yields both the gate tile and the up tile for a group.  The DOWN projection
//   reduces over K=INTER and produces the HIDDEN-wide y, tile-group by tile-group.
//
//----------------------------------------------------------------------------
// SEQUENCING (FSM)
//   IDLE  : wait for start (latches x into the x buffer over HIDDEN load beats,
//           or x is preloaded -- see X LOAD below).
//   GU    : for g in 0..NG_GU-1  (NG_GU = ceil(INTER/TN) gate/up tile-groups):
//             stream K=HIDDEN beats into the gate+up matmuls (lockstep);
//             on their out_valid, push the TN gate lanes through glm_act(SILU)
//             and, LAT_ACT later, multiply by the saved TN up lanes -> TN h
//             elements written to the h buffer (INTER bf16).
//   DOWN  : for g in 0..NG_D-1   (NG_D = ceil(HIDDEN/TN) down tile-groups):
//             stream K=INTER beats (from the h buffer) into the down matmul;
//             on out_valid, write the TN y lanes to the y buffer (HIDDEN bf16).
//   DONE  : assert done for one cycle; y_out holds the HIDDEN result.
//
//   Each matmul pass is the deterministic glm_matmul_pipe latency
//       PASS(K) = K + L + TREE_LAT + 1            (L=`FP_MAC_LAT, TREE_LAT=3*ADD)
//   The silu*up merge adds a fixed LAT_ACT (glm_act) + 1 (bf16 mul reg) tail that
//   overlaps the next group's stream, so the whole expert latency is data-
//   independent and computable (see LATENCY below).
//
//----------------------------------------------------------------------------
// WEIGHT DELIVERY (pull / streamed by the surrounding system)
//   The expert OWNS only the small x (HIDDEN) and h (INTER) buffers; the weights
//   (INTER x HIDDEN, etc. -- up to 6144x12288) are streamed by the system.  The
//   expert drives, every beat it issues, a fully-decoded weight REQUEST:
//       w_sel    : which projection (0=GATE,1=UP,2=DOWN)
//       w_grp    : the output tile-group index g (its TN outputs start at g*TN)
//       w_k      : the reduction index k of this beat
//       w_req    : high on a beat that needs weights (== matmul in_valid)
//   The system must present, COMBINATIONALLY in the same cycle, the TN weight
//   lanes for that (sel,grp,k):  w_col[16*t +: 16] = W_sel[g*TN + t, k].
//   (For the verifiable slice the smoke TB is a flat weight ROM indexed by these;
//   a real system points them at the DMA'd resident expert buffer.)  Lanes whose
//   output index g*TN+t >= OUT are don't-care (the expert masks them out of the
//   buffer write), so OUT need not be a multiple of TN.
//
//----------------------------------------------------------------------------
// X LOAD
//   start pulses with x_valid streaming: the caller drives HIDDEN beats of
//   x_in (1 bf16/beat, low-latency) right after start, OR preloads via the same
//   port.  To keep the handshake trivial and deterministic we accept x as a
//   single wide bus x_vec (HIDDEN bf16 packed) latched at start -- the token is
//   tiny (HIDDEN<=12288 -> wide but flat) and this removes a load-FSM.  (A
//   streaming x port is a drop-in alternative; the orchestration is identical.)
//
//----------------------------------------------------------------------------
// DENSE vs MoE
//   Same datapath; INTER selects the size:  MoE expert INTER=2048, dense-front
//   INTER=12288 (HIDDEN=6144 both).  No mode logic beyond the INTER parameter --
//   the tile-group counts (NG_GU, NG_D) and K lengths derive from HIDDEN/INTER,
//   so a single instance covers both FFN modes by parameterization.
//
//----------------------------------------------------------------------------
// LATENCY (deterministic, data-independent)
//   Let Pgu = HIDDEN + L + TREE_LAT + 1   (one gate/up pass, K=HIDDEN)
//       Pd  = INTER  + L + TREE_LAT + 1   (one down  pass, K=INTER)
//       Tmerge = LAT_ACT + 1              (silu + bf16 mul tail per group)
//   Total ~ NG_GU*(Pgu + a few control cycles) + Tmerge + NG_D*Pd + tail.
//   Every term is a fixed count of cycles -> the expert exposes a fixed latency
//   for given (HIDDEN, INTER, TN).  busy is high throughout; done pulses once.
//
//----------------------------------------------------------------------------
// STYLE
//   synchronous ACTIVE-HIGH reset; NO latch (every reg assigned on every path);
//   NO combinational loop (all feedback rides matmul/act pipeline registers);
//   reuses glm_matmul_pipe + glm_act + glm_fp unchanged.
//============================================================================
module swiglu_expert #(
    parameter integer HIDDEN = 128,   // model hidden size (scales to 6144)
    parameter integer INTER  = 64,    // FFN inter size (MoE 2048 / dense 12288)
    parameter integer TN     = 4,     // output-tile width = matmul PE_N
    parameter integer KMAX   = 16384  // >= max(HIDDEN, INTER) for matmul counter
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
    output wire                     w_req,      // need weights this cycle
    output wire [1:0]               w_sel,      // 0=GATE,1=UP,2=DOWN
    output wire [$clog2((INTER>HIDDEN?INTER:HIDDEN)/TN+1)-1:0] w_grp, // tile-group
    output wire [$clog2(KMAX+1)-1:0] w_k,       // reduction index k of this beat
    // ---- weight response (combinational, same cycle as w_req) ----
    //   GATE/UP : w_col[t] = W_{gate|up}[ w_grp*TN + t , w_k ]
    //   DOWN    : w_col[t] = W_down    [ w_grp*TN + t , w_k ]
    input  wire [16*TN-1:0]         w_col,      // TN bf16 weight lanes
    input  wire [16*TN-1:0]         w_col_up,   // TN bf16 W_up lanes (GATE/UP pass)

    // ---- result ----
    output reg  [16*HIDDEN-1:0]     y_out       // HIDDEN bf16 result, packed
);
    `include "glm_fp.vh"

    // ---------------- derived sizes ----------------
    // Per-pass GEMM latency = K + L(mac drain) + TREE_LAT(reduce) + 1(round reg),
    // and the silu tail = LAT_ACT + 1.  These are documentation localparams (the
    // FSM is purely handshake-driven off out_valid/done, so no site hardcodes the
    // numbers); exposed here as a single PASS-latency formula and lint-waived.
    /* verilator lint_off UNUSEDPARAM */
    localparam integer L         = `FP_MAC_LAT;       // matmul mac drain (7)
    localparam integer TREE_LAT  = 3 * `FP_ADD_LAT;   // matmul reduce tree (15)
    localparam integer LAT_ACT   = 5;                 // glm_act fixed latency
    localparam integer PASS_GU_LAT = HIDDEN + L + TREE_LAT + 1; // one gate/up pass
    localparam integer PASS_DN_LAT = INTER  + L + TREE_LAT + 1; // one down  pass
    localparam integer MERGE_LAT   = LAT_ACT + 1;     // silu + bf16-mul tail
    /* verilator lint_on UNUSEDPARAM */
    localparam integer KW       = $clog2(KMAX+1);
    // number of output tile-groups for the projections (ceil(OUT/TN))
    localparam integer NG_GU = (INTER  + TN - 1) / TN;   // gate/up: OUT=INTER
    localparam integer NG_D  = (HIDDEN + TN - 1) / TN;   // down   : OUT=HIDDEN
    localparam integer GW    = $clog2((INTER>HIDDEN?INTER:HIDDEN)/TN + 1);

    // ===================================================================
    //  Token + intermediate buffers (the only state this unit owns)
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
    //  FSM
    // ===================================================================
    localparam [2:0] S_IDLE = 3'd0,
                     S_GUP  = 3'd1,   // prime: mm_start asserted prior cycle; let
                                      //        the matmul's `streaming` go live
                     S_GU   = 3'd2,   // stream a gate/up group
                     S_GUW  = 3'd3,   // wait gate/up matmul drain + merge
                     S_DNP  = 3'd4,   // prime the down matmul
                     S_DN   = 3'd5,   // stream a down group
                     S_DNW  = 3'd6,   // wait down matmul drain
                     S_DONE = 3'd7;

    reg [2:0]    state;
    reg [GW-1:0] grp;                 // current output tile-group
    reg [KW-1:0] kcnt;                // K beat counter while streaming

    // current pass K length (HIDDEN for gate/up, INTER for down)
    wire [KW-1:0] k_len_gu = HIDDEN[KW-1:0];
    wire [KW-1:0] k_len_dn = INTER [KW-1:0];

    // ===================================================================
    //  Shared GEMM operand drive.  PE_M = 1 (single token row).
    //  a_col is the single token element x[k]; w_row is the TN weight lanes.
    // ===================================================================
    reg          mm_start;            // clocked pulse
    reg  [KW-1:0] mm_k_len;           // clocked (latched at start)
    reg  mm_start_u;                  // clocked pulse (up matmul)
    // COMBINATIONAL operand drive via CONTINUOUS ASSIGNS (coherent with the
    // weight request the same cycle, so a_col/w_row/in_valid and w_col all
    // describe the SAME beat).  Continuous assigns -- not an always@* that reads
    // the w_col input -- so there is no array-sensitivity re-trigger with the
    // system's own combinational weight responder.
    wire         mm_in_valid;
    wire         mm_in_valid_u;
    wire [15:0]  a_col;               // x[k] (PE_M = 1 lane)
    wire [16*TN-1:0] w_row_g;         // gate-or-down weight lanes
    wire [16*TN-1:0] w_row_u;         // up weight lanes (gate/up pass only)

    // GATE/DOWN matmul (reused across both projections: gate then down).
    // busy is unused -- the FSM gates on out_valid (g_ov) + its deterministic
    // drain, not on busy; both matmuls run in lockstep so the up unit's busy/
    // out_valid are functionally redundant with the gate unit's.  Lint-waived.
    /* verilator lint_off UNUSEDSIGNAL */
    wire             g_busy, g_ov;
    wire [16*TN-1:0] g_c;             // 1 x TN bf16 tile
    glm_matmul_pipe #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm_g (
        .clk(clk), .rst(rst),
        .start(mm_start), .k_len(mm_k_len),
        .in_valid(mm_in_valid), .a_col(a_col), .w_row(w_row_g),
        .busy(g_busy), .out_valid(g_ov), .c_out(g_c)
    );

    // UP matmul (runs in lockstep with GATE during the gate/up pass; idle in down)
    wire             u_busy, u_ov;
    wire [16*TN-1:0] u_c;
    glm_matmul_pipe #(.PE_M(1), .PE_N(TN), .KMAX(KMAX)) u_mm_u (
        .clk(clk), .rst(rst),
        .start(mm_start_u), .k_len(mm_k_len),
        .in_valid(mm_in_valid_u), .a_col(a_col), .w_row(w_row_u),
        .busy(u_busy), .out_valid(u_ov), .c_out(u_c)
    );
    /* verilator lint_on UNUSEDSIGNAL */

    // ===================================================================
    //  SiLU(gate) and the silu*up merge.
    //  glm_act consumes the TN gate lanes when g_ov fires; LAT_ACT later it
    //  emits silu(gate); we then bf16-multiply by the up tile (saved from u_ov,
    //  which fires the SAME cycle as g_ov since the two matmuls are lockstep).
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

    // up tile + its destination group must be paired with the silu result when
    // it emerges LAT_ACT cycles later.  Because EXACTLY ONE gate/up group is in
    // flight at a time (the FSM streams a group, then waits in S_GUW for its
    // matmul + silu + merge before issuing the next), a simple LATCH of the up
    // tile and group at the g_ov edge suffices -- no shift register, no aliasing.
    reg [16*TN-1:0] up_hold;          // up tile captured at g_ov
    reg [GW-1:0]    grp_hold;         // its destination group

    // ===================================================================
    //  Down-projection output capture (the silu/merge is bypassed).
    // ===================================================================

    // ---- merge: silu(gate) (.) up, write TN h elements ----
    integer mt;

    // ===================================================================
    //  COMBINATIONAL matmul-operand + weight-request drive.
    //  Streaming states (S_GU, S_DN) present beat `kcnt` of group `grp`:
    //    GATE/UP : a_col = x[kcnt] ; w_row_g = W_gate[grp tile, kcnt] ;
    //              w_row_u = W_up[grp tile, kcnt] ; both matmuls in_valid.
    //    DOWN    : a_col = h[kcnt] ; w_row_g = W_down[grp tile, kcnt].
    //  The weight request (w_req/w_sel/w_grp/w_k) is asserted the SAME cycle, so
    //  the system's combinational w_col / w_col_up answer the very beat the
    //  matmul samples at the next edge -- no skew between a_col and w_row.
    // ===================================================================
    wire stream_gu = (state == S_GU);
    wire stream_dn = (state == S_DN);
    // sized buffer indices (kcnt counts 0..K-1; K<=HIDDEN or <=INTER per pass)
    localparam integer XIW = (HIDDEN > 1) ? $clog2(HIDDEN) : 1;
    localparam integer HIW = (INTER  > 1) ? $clog2(INTER)  : 1;
    wire [XIW-1:0] x_idx = kcnt[XIW-1:0];
    wire [HIW-1:0] h_idx = kcnt[HIW-1:0];
    // matmul operands: token element (x in gate/up, h in down) and weight lanes.
    assign a_col        = stream_gu ? xbuf[x_idx] :
                          stream_dn ? hbuf[h_idx] : 16'b0;
    assign w_row_g      = w_col;          // system weight response (pass-through)
    assign w_row_u      = w_col_up;       // up companion lanes
    assign mm_in_valid  = stream_gu | stream_dn;
    assign mm_in_valid_u= stream_gu;      // up matmul only during the gate/up pass
    // weight request: same beat the matmul samples this cycle.
    assign w_req = stream_gu | stream_dn;
    assign w_sel = stream_dn ? 2'd2 : 2'd0;   // DOWN vs GATE(+UP companion)
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
            mm_start     <= 1'b0;
            mm_start_u   <= 1'b0;
            mm_k_len     <= {KW{1'b0}};
            act_in_valid <= 1'b0;
            act_x_in     <= {16*TN{1'b0}};
            up_hold      <= {16*TN{1'b0}};
            grp_hold     <= {GW{1'b0}};
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
                    busy     <= 1'b1;
                    grp      <= {GW{1'b0}};
                    // assert start now; prime next cycle so `streaming` is live
                    // before the first K-beat is presented (no dropped beat).
                    state    <= S_GUP;
                    kcnt     <= {KW{1'b0}};
                    mm_start   <= 1'b1;     // start gate matmul
                    mm_start_u <= 1'b1;     // start up   matmul
                    mm_k_len   <= k_len_gu;
                end
            end

            // ---- prime: streaming goes live this cycle; begin issuing next ----
            S_GUP: begin
                kcnt  <= {KW{1'b0}};
                state <= S_GU;
            end

            // ---- GATE/UP: stream HIDDEN K-beats into both matmuls ----
            // (operands + weight request driven combinationally above)
            S_GU: begin
                if (kcnt == k_len_gu - 1'b1) begin
                    state <= S_GUW;         // last beat issued; wait drain
                end
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait gate/up matmul -> silu -> *up -> h buffer ----
            S_GUW: begin
                // when both matmuls finish (lockstep -> same cycle), launch silu
                if (g_ov) begin
                    act_in_valid <= 1'b1;
                    act_x_in     <= g_c;      // gate tile -> silu
                    // latch the up tile + its group to pair with silu's output
                    up_hold      <= u_c;
                    grp_hold     <= grp;
                end

                // when silu emits, multiply by the matching up tile and store h
                if (act_ov) begin
                    // (h written in the combinational merge block below via hbuf)
                    // advance to next group or to DOWN once all groups done
                    if (grp == NG_GU[GW-1:0] - 1'b1) begin
                        // all gate/up groups done -> begin down projection
                        state    <= S_DNP;
                        grp      <= {GW{1'b0}};
                        kcnt     <= {KW{1'b0}};
                        mm_start <= 1'b1;       // start down matmul
                        mm_k_len <= k_len_dn;
                    end else begin
                        // next gate/up group
                        grp      <= grp + 1'b1;
                        kcnt     <= {KW{1'b0}};
                        state    <= S_GUP;
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
            // (operands + weight request driven combinationally above)
            S_DN: begin
                if (kcnt == k_len_dn - 1'b1) begin
                    state <= S_DNW;
                end
                kcnt <= kcnt + 1'b1;
            end

            // ---- wait down matmul -> write y tile ----
            S_DNW: begin
                if (g_ov) begin
                    // write TN y lanes for group grp (mask tail past HIDDEN)
                    for (yi = 0; yi < TN; yi = yi + 1) begin
                        if (grp*TN + yi < HIDDEN)
                            y_out[16*(grp*TN + yi) +: 16] <= g_c[16*yi +: 16];
                    end
                    if (grp == NG_D[GW-1:0] - 1'b1) begin
                        state <= S_DONE;
                    end else begin
                        grp      <= grp + 1'b1;
                        kcnt     <= {KW{1'b0}};
                        state    <= S_DNP;
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
    //  silu*up MERGE -> h buffer write (clocked, fires on act_ov).
    //  A clean per-lane masked store of h = silu(gate) (.) up.  Pairs the silu
    //  result (act_y) with the up tile latched at the matched g_ov edge (up_hold)
    //  and writes the destination group (grp_hold) -- so lanes pair correctly.
    // ===================================================================
    always @(posedge clk) begin
        if (act_ov) begin
            for (mt = 0; mt < TN; mt = mt + 1) begin
                // h = silu(gate) * up  (bf16 multiply via fp32, RNE -> bf16)
                if (grp_hold*TN + mt < INTER)
                    hbuf[grp_hold*TN + mt] <= bf16_mul(act_y[16*mt +: 16],
                                                       up_hold[16*mt +: 16]);
            end
        end
    end
endmodule
/* verilator lint_on DECLFILENAME */
