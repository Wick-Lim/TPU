`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// mtp_head.v  --  GLM-5.2 Multi-Token Prediction head (num_nextn_predict_layers=1)
//                 DeepSeek-V3-style t+2 speculative head  (ACCEL_GLM52 §3 / #18)
//----------------------------------------------------------------------------
// FUNCTION  (predict token t+2 from main-model hidden h_t and the embedding of
//            the already-predicted token t+1)
//
//     a    = RMSNorm(h_t)                          // own pre-combine norm (gamma_a)
//     b    = RMSNorm(embed(tok_{t+1}))             // own pre-combine norm (gamma_b)
//     cat  = [ a ; b ]                              // concat -> 2*MODEL_DIM (bf16)
//     h'   = W_proj @ cat                           // W_proj is MODEL_DIM x 2*MODEL_DIM
//     y    = decoder_block( h' , pos , kv_cache )   // ONE GLM-5.2 decoder layer
//     xN   = RMSNorm_final( y )                     // shared final norm (gamma_final)
//     logits[V] = W_lm[V,MODEL_DIM] . xN            // shared LM head GEMV
//     argmax    = arg max_v logits[v]               // speculative next-next token
//
//   PURE ORCHESTRATOR.  It REIMPLEMENTS NO ARITHMETIC -- it ORCHESTRATES the
//   already-verified sub-units, exactly the discipline glm_model.v uses:
//     * ONE rmsnorm_unit (LEN=MODEL_DIM, LANES=1) SERIALLY REUSED for the THREE
//       norms (RMSNorm(h_t), RMSNorm(emb), final RMSNorm).  A 2-bit `cn_which`
//       qualifies the gamma pull so the system answers gamma_a / gamma_b /
//       gamma_final.  Minimal area (one norm datapath, three passes); a parallel
//       trio is a drop-in alternative -- not chosen.
//     * ONE glm_matmul_pipe(PE_M=1,PE_N=PROJ_TN,KMAX=2*MODEL_DIM) as the COMBINE
//       PROJECTION GEMV:  h'[1,MODEL_DIM] = cat[1,2*MODEL_DIM] x W_proj^T, issued
//       over MODEL_DIM/PROJ_TN output tiles (K=2*MODEL_DIM reduction per tile).
//     * ONE glm_decoder_block (the verified GLM-5.2 layer; dense or MoE per the
//       `mode` input) run ONCE on h'.  All its weight/cache pulls are forwarded.
//     * ONE rmsnorm_unit pass (the SAME reused instance) for the final norm.
//     * ONE glm_matmul_pipe(PE_M=1,PE_N=LM_TN,KMAX=MODEL_DIM) as the LM-head GEMV
//       over VOCAB/LM_TN tiles, then a 1-elt/cycle argmax scan -- identical to
//       glm_model.v's tail (the "shared LM head").
//   Two matmul_pipe instances (proj vs LM head) are used for a clean, deterministic
//   FSM; a single shared instance (PE_N=max, KMAX=2*MODEL_DIM, w_row/a muxed) is a
//   drop-in area optimisation -- not chosen here.
//
//----------------------------------------------------------------------------
// LATENCY  (deterministic, data-independent for given params + S)
//   L_mtp = 2*L_rmsnorm(MODEL_DIM)                       // RMSNorm(h_t)+RMSNorm(emb)
//         + (MODEL_DIM/PROJ_TN) * (2*MODEL_DIM + `FP_MAC_LAT + 3*`FP_ADD_LAT + few)
//         + L_decoder_block(params, S)                   // the ONE layer
//         + L_rmsnorm(MODEL_DIM)                          // final norm
//         + (VOCAB/LM_TN) * (MODEL_DIM + `FP_MAC_LAT + 3*`FP_ADD_LAT + few)
//         + VOCAB                                         // argmax scan
//   Every term is a fixed cycle count -> the head exposes a fixed, computable
//   latency.  No data-dependent stall; sync active-high reset; no latch (every reg
//   written every clock in its block); no comb loop (all feedback rides the
//   rmsnorm / decoder_block / matmul_pipe pipeline registers).
//
//----------------------------------------------------------------------------
// CONVENTIONS: `timescale + glm_fp.vh; synchronous ACTIVE-HIGH reset; no latch;
//   no combinational loop.  All weight / cache / gamma delivery is via
//   combinational PULL interfaces answered the SAME cycle by the system/TB.
//============================================================================
module mtp_head #(
    // ---- model / slice config (small-but-faithful, ACCEL_GLM52 §8.1) ----
    parameter integer MODEL_DIM  = 128,
    parameter integer VOCAB      = 256,
    // ---- decoder_block slice params (passed straight through) ----
    parameter integer H_HEADS    = 4,
    parameter integer NOPE       = 16,
    parameter integer ROPE       = 16,
    parameter integer V_DIM      = 32,
    parameter integer Q_LORA     = 64,
    parameter integer KV_LORA    = 32,
    parameter integer S_MAX      = 8,
    parameter integer TOPK_ATTN  = 8,
    parameter integer THETA      = 8000000,
    parameter integer PE_N       = 4,
    parameter integer POSW       = 20,
    parameter integer N_EXPERT   = 8,
    parameter integer TOPK       = 2,
    parameter integer INTER_MOE  = 64,
    parameter integer INTER_DENSE= 256,
    parameter [31:0]  RSCALE     = 32'h40200000,// 2.5 fp32
    parameter integer TN         = 4,
    // ---- GEMV tile widths.  VOCAB % LM_TN == 0 ; MODEL_DIM % PROJ_TN == 0. ----
    parameter integer LM_TN      = 4,           // LM-head VOCAB cols/pass
    parameter integer PROJ_TN    = 4,           // combine-proj output cols/pass
    // ====================================================================
    // derived (do NOT override) -- mirror decoder_block's port-width derivations
    // ====================================================================
    parameter integer QK_DIM     = NOPE + ROPE,
    parameter integer IDXW       = (S_MAX <= 1) ? 1 : $clog2(S_MAX),
    parameter integer HQK        = H_HEADS * QK_DIM,
    parameter integer HNOPE      = H_HEADS * NOPE,
    parameter integer HV         = H_HEADS * V_DIM,
    parameter integer EIDXW      = (N_EXPERT <= 1) ? 1 : $clog2(N_EXPERT),
    parameter integer A_KMAX     = (MODEL_DIM > Q_LORA) ?
                               ((MODEL_DIM > KV_LORA) ?
                                ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV))
                             : ((Q_LORA > KV_LORA) ?
                                ((Q_LORA > HV) ? Q_LORA : HV)
                              : ((KV_LORA > HV) ? KV_LORA : HV)),
    parameter integer A_OMAX     = (HQK > MODEL_DIM) ?
                               ((HQK > HNOPE) ?
                                 ((HQK > HV) ? HQK : HV)
                               : ((HNOPE > HV) ? HNOPE : HV))
                             : ((MODEL_DIM > HNOPE) ?
                                 ((MODEL_DIM > HV) ? MODEL_DIM : HV)
                               : ((HNOPE > HV) ? HNOPE : HV)),
    parameter integer A_NGMAX    = (A_OMAX + PE_N - 1) / PE_N,
    parameter integer A_GRPW     = (A_NGMAX <= 1) ? 1 : $clog2(A_NGMAX),
    parameter integer A_KCW      = (A_KMAX  <= 1) ? 1 : $clog2(A_KMAX),
    parameter integer FF_GWD     = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN + 1),
    parameter integer FF_KMAX_D  = (INTER_DENSE > MODEL_DIM) ? INTER_DENSE : MODEL_DIM,
    parameter integer FF_KWD     = $clog2(FF_KMAX_D + 1),
    parameter integer FF_KMAX_M  = (INTER_MOE  > MODEL_DIM) ? INTER_MOE  : MODEL_DIM,
    parameter integer R_KW       = $clog2(FF_KMAX_M + 1),
    // head-level derived
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer LMKW       = $clog2(MODEL_DIM + 1),    // LM matmul k_len width
    // combine-projection derived
    parameter integer CK         = 2 * MODEL_DIM,            // concat length (K)
    parameter integer CKIW       = $clog2(CK),               // concat index width
    parameter integer PKW        = $clog2(CK + 1),           // proj matmul k_len width
    parameter integer NPTILE     = MODEL_DIM / PROJ_TN,      // proj output tiles
    parameter integer PTW        = (NPTILE <= 1) ? 1 : $clog2(NPTILE)
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    // ---- control ----
    input  wire                          start,      // 1-cycle pulse: begin
    output reg                           busy,
    output reg                           done,       // 1-cycle pulse: logits valid
    input  wire                          mode,       // 0=DENSE FFN, 1=MoE FFN (block)
    input  wire [POSW-1:0]               pos,        // query position t (RoPE)
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)

    // ---- data in (bf16) ----
    input  wire [MODEL_DIM*16-1:0]       h_t,        // main-model hidden state @ t
    input  wire [MODEL_DIM*16-1:0]       emb_t1,     // embedding of predicted tok t+1

    // ---- outputs ----
    output reg  [VOCAB*16-1:0]           logits,     // VOCAB bf16 t+2 logits
    output reg  [TOKW-1:0]               argmax,     // arg max logit (spec. t+2 token)

    // ---- combine/final RMSNorm gamma pull (cn_which: 0=h_t,1=emb,2=final) ----
    output wire                          cn_req,
    output wire [1:0]                    cn_which,
    output wire [DIMW-1:0]               cn_idx,
    input  wire [15:0]                   cn_val,

    // ---- combine-projection weight pull: pw_col[t] = W_proj[ptile*PROJ_TN+t][pw_k]
    output wire                          pw_req,
    output wire [PTW-1:0]                pw_ptile,   // which MODEL_DIM output tile
    output wire [CKIW-1:0]               pw_k,       // concat reduction index 0..2*MD-1
    input  wire [PROJ_TN*16-1:0]         pw_col,     // PROJ_TN bf16 weight lanes

    // ---- decoder_block RMSNorm gamma pull (pre-attn/pre-FFN) ----
    output wire                          gn_req,
    output wire                          gn_which,   // 0=pre-attn, 1=pre-FFN
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,

    // ---- decoder_block attention weight pull ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*16-1:0]            aw_col,

    // ---- decoder_block attention KV-cache read ----
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    input  wire                          kc_valid,

    // ---- decoder_block MoE router weight pull (W_g column) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [16*N_EXPERT-1:0]        rw_col,

    // ---- decoder_block FFN expert weight pull (qualified) ----
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [16*TN-1:0]              fw_col,
    input  wire [16*TN-1:0]              fw_col_up,

    // ---- shared LM-head weight pull: lw_col[t] = W_lm[vtile*LM_TN+t][lw_k] ----
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col
);
    `include "glm_fp.vh"

    integer ii;

    //========================================================================
    // latched inputs + working buffers (bf16)
    //========================================================================
    reg [15:0] hbuf [0:MODEL_DIM-1];   // latched h_t  (RMSNorm source, phase 0)
    reg [15:0] ebuf [0:MODEL_DIM-1];   // latched emb  (RMSNorm source, phase 1)
    reg [15:0] cbuf [0:CK-1];          // concat [a;b]  (= RMSNorm outputs)
    reg [15:0] hprime [0:MODEL_DIM-1]; // h' = W_proj @ cat  (decoder block input)
    reg [15:0] xcur   [0:MODEL_DIM-1]; // decoder-block output y (final-norm source)
    reg [15:0] xn     [0:MODEL_DIM-1]; // final-normed (LM-head input)
    reg [15:0] lbuf   [0:VOCAB-1];     // LM-head logits scratch (bf16)

    reg            mode_q;
    reg [POSW-1:0] pos_q;
    reg [IDXW:0]   slen_q;

    // packed view of h' for the decoder block's wide x_vec port
    reg [MODEL_DIM*16-1:0] hp_vec;
    always @* begin
        for (ii = 0; ii < MODEL_DIM; ii = ii + 1)
            hp_vec[16*ii +: 16] = hprime[ii];
    end

    //========================================================================
    // ONE rmsnorm_unit (LEN=MODEL_DIM, LANES=1) reused for the 3 norms.
    //   It PULLS x (reduce pass) from the phase-selected source, then gamma
    //   (normalize pass) via cn_* (cn_which = phase tells the system which gamma).
    //========================================================================
    reg              cn_start;
    reg  [1:0]       cn_phase;          // 0=norm(h_t),1=norm(emb),2=final norm
    wire             cn_in_req, cn_g_req, cn_y_valid, cn_busy, cn_done;
    wire [15:0]      cn_y_out;
    reg  [15:0]      cn_x_in;
    reg              cn_x_valid;
    reg  [15:0]      cn_gamma_in;
    reg              cn_g_valid;
    rmsnorm_unit #(.LEN(MODEL_DIM), .LANES(1)) u_norm (
        .clk(clk), .rst(rst), .start(cn_start),
        .in_req(cn_in_req), .x_in(cn_x_in), .x_valid(cn_x_valid),
        .g_req(cn_g_req), .gamma_in(cn_gamma_in), .g_valid(cn_g_valid),
        .y_valid(cn_y_valid), .y_out(cn_y_out), .busy(cn_busy), .done(cn_done)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _cn_busy_unused = &{1'b0, cn_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // norm beat counters (LANES=1 -> beat == element index).  Reset at cn_start.
    reg [DIMW:0] cn_ridx;   // reduce read index (x pull)
    reg [DIMW:0] cn_widx;   // normalize write index (y store)
    reg [DIMW:0] cn_gidx;   // gamma pull index

    // gamma pull is COMBINATIONAL (answered same cycle), registered 1 cycle.
    assign cn_req   = cn_g_req;
    assign cn_which = cn_phase;
    assign cn_idx   = cn_gidx[DIMW-1:0];

    //========================================================================
    // ONE glm_decoder_block (the verified GLM-5.2 layer) run ONCE on h'.
    //   All weight/cache pulls forwarded straight out.
    //========================================================================
    reg                       db_start;
    wire                      db_busy, db_done;
    wire [MODEL_DIM*16-1:0]   db_y;
    glm_decoder_block #(
        .MODEL_DIM(MODEL_DIM), .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE),
        .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX),
        .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N), .POSW(POSW),
        .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN)
    ) u_block (
        .clk(clk), .rst(rst), .start(db_start), .busy(db_busy), .done(db_done),
        .mode(mode_q), .pos(pos_q), .s_len(slen_q),
        .x_vec(hp_vec), .y_out(db_y),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k), .aw_col(aw_col),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx), .fw_col(fw_col), .fw_col_up(fw_col_up)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _db_busy_unused = &{1'b0, db_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    //========================================================================
    // COMBINE-PROJECTION GEMV : glm_matmul_pipe as a 1xPROJ_TN tile, K=2*MD.
    //   A row (M=1) = cat[1,2*MODEL_DIM] ; W tile (N=PROJ_TN) = W_proj[ptile..][k]
    //   transposed.  On beat k present pp_a = cat[k] and pw_col = W_proj column.
    //========================================================================
    reg                  pp_start;
    reg                  pp_in_valid;
    reg  [PKW-1:0]       pp_klen;
    reg  [15:0]          pp_a;            // cat[k] (1 lane)
    wire                 pp_busy, pp_ov;
    wire [16*PROJ_TN-1:0] pp_c;           // 1 x PROJ_TN result tile (bf16)
    glm_matmul_pipe #(.PE_M(1), .PE_N(PROJ_TN), .KMAX(CK)) u_proj (
        .clk(clk), .rst(rst), .start(pp_start), .k_len(pp_klen),
        .in_valid(pp_in_valid), .a_col(pp_a), .w_row(pw_col),
        .busy(pp_busy), .out_valid(pp_ov), .c_out(pp_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _pp_busy_unused = &{1'b0, pp_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    // projection sequencing
    reg [PTW-1:0]   ptile;          // current output tile
    reg [CKIW:0]    pk;             // current K beat (0..CK)
    reg             pp_streaming;
    reg [CKIW-1:0]  pk_present;     // K index currently registered in pp_a
    reg             pp_pres_valid;  // mirrors a presented beat (weight pull qualifier)

    assign pw_req   = pp_pres_valid;
    assign pw_ptile = ptile;
    assign pw_k     = pk_present;

    //========================================================================
    // SHARED LM-HEAD GEMV : glm_matmul_pipe as a 1xLM_TN tile, K=MODEL_DIM.
    //========================================================================
    reg                  mm_start;
    reg                  mm_in_valid;
    reg  [LMKW-1:0]      mm_klen;
    reg  [15:0]          mm_a;            // xN[k] (1 lane)
    wire                 mm_busy, mm_ov;
    wire [16*LM_TN-1:0]  mm_c;            // 1 x LM_TN result tile (bf16)
    glm_matmul_pipe #(.PE_M(1), .PE_N(LM_TN), .KMAX(MODEL_DIM)) u_lm (
        .clk(clk), .rst(rst), .start(mm_start), .k_len(mm_klen),
        .in_valid(mm_in_valid), .a_col(mm_a), .w_row(lw_col),
        .busy(mm_busy), .out_valid(mm_ov), .c_out(mm_c)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _mm_busy_unused = &{1'b0, mm_busy};
    /* verilator lint_on UNUSEDSIGNAL */

    reg [VTW-1:0]  vtile;          // current VOCAB tile
    reg [DIMW:0]   lm_k;           // current K beat (0..MODEL_DIM)
    reg            lm_streaming;
    reg [DIMW-1:0] lk_present;     // K index currently registered in mm_a
    reg            mm_pres_valid;  // mirrors a presented beat

    assign lw_req   = mm_pres_valid;
    assign lw_vtile = vtile;
    assign lw_k     = lk_present;

    //========================================================================
    // MASTER FSM
    //========================================================================
    localparam [3:0]
        S_IDLE   = 4'd0,
        S_NORM   = 4'd1,    // run rmsnorm pass (phase 0/1/2)
        S_PROJ   = 4'd2,    // stream K beats of combine projection (current ptile)
        S_PROJW  = 4'd3,    // wait pp_ov; store PROJ_TN h' elts; next ptile / block
        S_DB     = 4'd4,    // start decoder block, wait one cycle
        S_DBW    = 4'd5,    // wait db_done; xcur <= y; launch final norm
        S_LMTILE = 4'd6,    // stream K beats for current vtile
        S_LMWAIT = 4'd7,    // wait mm_ov; store LM_TN logits; next vtile
        S_ARGMAX = 4'd8,    // scan lbuf for argmax (fp32 compare)
        S_DONE   = 4'd9;
    reg [3:0] state;

    reg [TOKW:0]   am_i;           // argmax scan index
    reg [31:0]     am_best;        // best logit value (fp32)
    reg [TOKW-1:0] am_arg;

    always @(posedge clk) begin
        if (rst) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            logits       <= {VOCAB*16{1'b0}};
            argmax       <= {TOKW{1'b0}};
            mode_q       <= 1'b0;
            pos_q        <= {POSW{1'b0}};
            slen_q       <= {(IDXW+1){1'b0}};
            cn_start     <= 1'b0; cn_phase <= 2'd0;
            cn_x_in      <= 16'h0; cn_x_valid <= 1'b0;
            cn_gamma_in  <= 16'h0; cn_g_valid <= 1'b0;
            db_start     <= 1'b0;
            pp_start     <= 1'b0; pp_in_valid <= 1'b0;
            pp_klen      <= {PKW{1'b0}}; pp_a <= 16'h0;
            ptile        <= {PTW{1'b0}};
            pk           <= {(CKIW+1){1'b0}};
            pp_streaming <= 1'b0;
            pk_present   <= {CKIW{1'b0}};
            pp_pres_valid<= 1'b0;
            mm_start     <= 1'b0; mm_in_valid <= 1'b0;
            mm_klen      <= {LMKW{1'b0}}; mm_a <= 16'h0;
            vtile        <= {VTW{1'b0}};
            lm_k         <= {(DIMW+1){1'b0}};
            lm_streaming <= 1'b0;
            lk_present   <= {DIMW{1'b0}};
            mm_pres_valid<= 1'b0;
            am_i         <= {(TOKW+1){1'b0}};
            am_best      <= 32'h0;
            am_arg       <= {TOKW{1'b0}};
            for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                hbuf[ii] <= 16'h0; ebuf[ii] <= 16'h0;
                hprime[ii] <= 16'h0; xcur[ii] <= 16'h0; xn[ii] <= 16'h0;
            end
            for (ii=0; ii<CK; ii=ii+1)    cbuf[ii] <= 16'h0;
            for (ii=0; ii<VOCAB; ii=ii+1) lbuf[ii] <= 16'h0;
        end else begin
            // ---- default pulse deassert ----
            done     <= 1'b0;
            cn_start <= 1'b0;
            db_start <= 1'b0;
            pp_start <= 1'b0;
            mm_start <= 1'b0;

            case (state)
            //----------------------------------------------------------------
            S_IDLE: begin
                busy <= 1'b0;
                if (start) begin
                    busy   <= 1'b1;
                    mode_q <= mode;
                    pos_q  <= pos;
                    slen_q <= s_len;
                    for (ii=0; ii<MODEL_DIM; ii=ii+1) begin
                        hbuf[ii] <= h_t   [16*ii +: 16];
                        ebuf[ii] <= emb_t1[16*ii +: 16];
                    end
                    // launch RMSNorm(h_t) : phase 0 (reduce source = hbuf)
                    cn_phase <= 2'd0;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- rmsnorm pass
            // Reduce pass: answer cn_in_req from the phase source.  Normalize pass:
            // answer cn_g_req with the combinational gamma (registered 1 cycle).
            // Store y to the phase destination: phase0 -> cbuf[0..MD-1] (=a),
            // phase1 -> cbuf[MD..2MD-1] (=b), phase2 -> xn (final-normed).
            S_NORM: begin
                cn_x_valid <= 1'b0; cn_g_valid <= 1'b0;
                if (cn_in_req) begin
                    case (cn_phase)
                        2'd0:    cn_x_in <= hbuf[cn_ridx[DIMW-1:0]];
                        2'd1:    cn_x_in <= ebuf[cn_ridx[DIMW-1:0]];
                        default: cn_x_in <= xcur[cn_ridx[DIMW-1:0]];
                    endcase
                    cn_x_valid <= 1'b1;
                end
                if (cn_g_req) begin
                    cn_gamma_in <= cn_val;      // combinational gamma answer
                    cn_g_valid  <= 1'b1;
                end
                if (cn_y_valid) begin
                    case (cn_phase)
                        2'd0:    cbuf[cn_widx[CKIW-1:0]]                            <= cn_y_out;
                        2'd1:    cbuf[MODEL_DIM[CKIW-1:0] + cn_widx[CKIW-1:0]]      <= cn_y_out;
                        default: xn[cn_widx[DIMW-1:0]]                              <= cn_y_out;
                    endcase
                end
                if (cn_done) begin
                    if (cn_phase == 2'd0) begin
                        // a done -> run RMSNorm(emb) : phase 1
                        cn_phase <= 2'd1;
                        cn_start <= 1'b1;
                        state    <= S_NORM;
                    end else if (cn_phase == 2'd1) begin
                        // concat ready -> launch combine projection (tile 0)
                        ptile        <= {PTW{1'b0}};
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        pp_in_valid  <= 1'b0;       // first beat presented in S_PROJ
                        state        <= S_PROJ;
                    end else begin
                        // final norm done -> launch LM head (vtile 0)
                        vtile        <= {VTW{1'b0}};
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        mm_in_valid  <= 1'b0;       // first beat presented in S_LMTILE
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile stream
            // Present cat[k] as a_col and W_proj[ptile][k] (pw_col) as w_row each
            // K beat.  pk_present latches the beat index so the weight pull aligns.
            S_PROJ: begin
                if (pp_streaming) begin
                    if (pk < CK[CKIW:0]) begin
                        pp_a          <= cbuf[pk[CKIW-1:0]];
                        pk_present    <= pk[CKIW-1:0];
                        pp_in_valid   <= 1'b1;
                        pp_pres_valid <= 1'b1;
                        pk            <= pk + 1'b1;
                    end else begin
                        pp_in_valid   <= 1'b0;
                        pp_pres_valid <= 1'b0;
                        pp_streaming  <= 1'b0;
                        state         <= S_PROJW;
                    end
                end
            end
            //---------------------------------------------------------------- proj tile wait
            S_PROJW: begin
                pp_in_valid   <= 1'b0;
                pp_pres_valid <= 1'b0;
                if (pp_ov) begin
                    for (ii=0; ii<PROJ_TN; ii=ii+1)
                        hprime[ptile*PROJ_TN + ii] <= pp_c[16*ii +: 16];
                    if (ptile == (NPTILE[PTW-1:0]-1'b1)) begin
                        // all output tiles done -> run the decoder block on h'
                        db_start <= 1'b1;
                        state    <= S_DB;
                    end else begin
                        ptile        <= ptile + 1'b1;
                        pp_klen      <= CK[PKW-1:0];
                        pp_start     <= 1'b1;
                        pp_streaming <= 1'b1;
                        pk           <= {(CKIW+1){1'b0}};
                        state        <= S_PROJ;
                    end
                end
            end
            //---------------------------------------------------------------- decoder block
            S_DB: begin
                // db_start pulsed entering this state; wait for done.
                state <= S_DBW;
            end
            S_DBW: begin
                if (db_done) begin
                    for (ii=0; ii<MODEL_DIM; ii=ii+1)
                        xcur[ii] <= db_y[16*ii +: 16];
                    // launch final rmsnorm over xcur : phase 2.  (xcur is updated
                    // this edge; the reduce pass starts next cycle so it is in place.)
                    cn_phase <= 2'd2;
                    cn_start <= 1'b1;
                    state    <= S_NORM;
                end
            end
            //---------------------------------------------------------------- LM head tile stream
            S_LMTILE: begin
                if (lm_streaming) begin
                    if (lm_k < MODEL_DIM[DIMW:0]) begin
                        mm_a          <= xn[lm_k[DIMW-1:0]];
                        lk_present    <= lm_k[DIMW-1:0];
                        mm_in_valid   <= 1'b1;
                        mm_pres_valid <= 1'b1;
                        lm_k          <= lm_k + 1'b1;
                    end else begin
                        mm_in_valid   <= 1'b0;
                        mm_pres_valid <= 1'b0;
                        lm_streaming  <= 1'b0;
                        state         <= S_LMWAIT;
                    end
                end
            end
            //---------------------------------------------------------------- LM head tile wait
            S_LMWAIT: begin
                mm_in_valid   <= 1'b0;
                mm_pres_valid <= 1'b0;
                if (mm_ov) begin
                    for (ii=0; ii<LM_TN; ii=ii+1)
                        lbuf[vtile*LM_TN + ii] <= mm_c[16*ii +: 16];
                    if (vtile == (NVTILE[VTW-1:0]-1'b1)) begin
                        am_i    <= {(TOKW+1){1'b0}};
                        am_best <= 32'hFF80_0000;   // -inf (fp32)
                        am_arg  <= {TOKW{1'b0}};
                        state   <= S_ARGMAX;
                    end else begin
                        vtile        <= vtile + 1'b1;
                        mm_klen      <= MODEL_DIM[LMKW-1:0];
                        mm_start     <= 1'b1;
                        lm_streaming <= 1'b1;
                        lm_k         <= {(DIMW+1){1'b0}};
                        state        <= S_LMTILE;
                    end
                end
            end
            //---------------------------------------------------------------- argmax
            S_ARGMAX: begin
                if (am_i < VOCAB[TOKW:0]) begin
                    if (fp32_gt(bf16_to_fp32(lbuf[am_i[TOKW-1:0]]), am_best)) begin
                        am_best <= bf16_to_fp32(lbuf[am_i[TOKW-1:0]]);
                        am_arg  <= am_i[TOKW-1:0];
                    end
                    am_i <= am_i + 1'b1;
                end else begin
                    for (ii=0; ii<VOCAB; ii=ii+1)
                        logits[16*ii +: 16] <= lbuf[ii];
                    argmax <= am_arg;
                    state  <= S_DONE;
                end
            end
            //----------------------------------------------------------------
            S_DONE: begin
                done  <= 1'b1;
                busy  <= 1'b0;
                state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end

    //========================================================================
    // rmsnorm pull beat counters (mirror the unit's beat order; LANES=1 so
    // beat == element index).  Reset at each cn_start.
    //========================================================================
    always @(posedge clk) begin
        if (rst) begin
            cn_ridx <= {(DIMW+1){1'b0}};
            cn_widx <= {(DIMW+1){1'b0}};
            cn_gidx <= {(DIMW+1){1'b0}};
        end else begin
            if (cn_start) begin
                cn_ridx <= {(DIMW+1){1'b0}};
                cn_widx <= {(DIMW+1){1'b0}};
                cn_gidx <= {(DIMW+1){1'b0}};
            end else begin
                if (cn_in_req)  cn_ridx <= cn_ridx + 1'b1;
                if (cn_y_valid) cn_widx <= cn_widx + 1'b1;
                if (cn_g_req)   cn_gidx <= cn_gidx + 1'b1;
            end
        end
    end

    //========================================================================
    // fp32 greater-than (strict).  Treats -0 == +0; ignores nan (finite logits).
    // Used for the argmax compare ONLY.
    //========================================================================
    function automatic fp32_gt(input [31:0] a, input [31:0] b);
        reg sa, sb;
        reg [30:0] ma, mb;
        begin
            sa = a[31]; sb = b[31];
            ma = a[30:0]; mb = b[30:0];
            if (sa != sb) begin
                if ((ma == 31'b0) && (mb == 31'b0)) fp32_gt = 1'b0; // +0 vs -0
                else fp32_gt = (sb == 1'b1);
            end else if (sa == 1'b0) begin
                fp32_gt = (ma > mb);
            end else begin
                fp32_gt = (ma < mb);
            end
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
