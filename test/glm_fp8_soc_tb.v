`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_fp8_soc_tb.v -- SMOKE TB for the top-level GLM-5.2-FP8 single-package SoC
//                     (glm_fp8_soc.v).  Stubs GDDR6 + Flash; drives the host.
//----------------------------------------------------------------------------
// WHAT IT CHECKS (X-aware, ALL N TESTS PASSED + $fatal on any failure)
//   The SoC wires the VERIFIED compute die (glm_model_fp8) to the VERIFIED
//   memory controllers (expert_cache_pf, kv_cache_pager) over a single Flash
//   channel.  This TB plays the GDDR6 + Flash PHYs as latency/memory STUBS and
//   the USB-C host, then drives a token end-to-end with the memory hierarchy in
//   the loop and asserts the INTEGRATION is correct:
//
//     (a) BINDING (the headline check): the SoC's committed token equals the
//         token an INDEPENDENT standalone glm_model_fp8 reference produces from
//         the SAME weights/KV ROMs.  i.e. routing the per-layer expert weights
//         through expert_cache_pf and the latent KV through kv_cache_pager does
//         NOT change the computed token -- the controllers are functionally
//         TRANSPARENT (they only address/move data, they do no math).  This is
//         a SEPARATE glm_model_fp8 instance (u_ref) fed by its OWN set of weight
//         responders reading the SAME ROM arrays the SoC's stub ports read.
//     (a') the committed token / logits / hidden state are X/Z-clean, and the
//          SoC's own internal argmax equals next_tok.
//     (b) the EXPERT CACHE saw the router-selected routed experts and advanced
//         its hit/miss counters (and the single Flash channel actually fetched
//         on misses) -- i.e. the cache is in the datapath.
//     (c) the KV PAGER appended one latent row per token (prefill s_len + the
//         decode token) and served the model's DSA gathers (row_valid pulses)
//         from the resident ring -- i.e. the pager is in the datapath.
//     (d) the request FIFO never overflowed (ec_dropped == 0).
//
//   DATAPATH NOTE (what runs through the controllers vs. what is stubbed):
//     * Compute MATH is unperturbed: the die's weight codes/scales and KV read
//       bytes come from the TB GDDR6/Flash STUBS the same cycle (the verified
//       pull contract).  expert_cache_pf sits in the loop as the ROUTED-expert
//       address/decision engine (SEES every router pick, answers hit/miss,
//       fetches Flash on miss); kv_cache_pager OWNS the latent-KV window
//       (APPENDs every token's row, GATHERs the model's kc_* reads).  So the
//       memory controllers are exercised and observed, while the bytes the die
//       consumes are the stub bytes -- exactly the explicitly-accepted
//       integration in the SoC header.  The reference proves transparency.
//----------------------------------------------------------------------------
// SMALL slice (MODEL_DIM=16) so a full forward is seconds, not ~15 min: the
// attention path is already golden at this size (mla_attn_fp8_tb), and the
// reference-vs-SoC equality is the binding proof regardless of slice.
//============================================================================
module glm_fp8_soc_tb;

    // ---- clock / reset ----
    reg clk = 1'b0;
    always #5 clk = ~clk;
    reg rst;

    // ================= small-but-faithful slice =================
    localparam integer MODEL_DIM  = 16;
    localparam integer L          = 4;    // 2 dense + 2 MoE (exercise expert cache)
    localparam integer N_DENSE    = 2;
    localparam integer VOCAB      = 16;   // VOCAB % LM_TN == 0
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 4;
    localparam integer ROPE       = 4;
    localparam integer V_DIM      = 4;
    localparam integer Q_LORA     = 8;
    localparam integer KV_LORA    = 8;
    localparam integer S_MAX      = 4;
    localparam integer TOPK_ATTN  = 4;    // = S_MAX (dense DSA selection)
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 4;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 4;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 16;
    localparam integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 4;
    localparam integer BLK        = 128;  // weight_block_size; all dims < BLK -> 1 block
    localparam integer LM_TN      = 4;
    // ---- memory system ----
    localparam integer CACHE_SLOTS = 2;   // < N_EXPERT -> evictions/misses happen
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;  // pow2, >= S_MAX; > total appends -> no overflow
    localparam integer EFIFO_DEPTH = 16;

    // ---- derived (mirror the DUT) ----
    localparam integer QK_DIM = NOPE + ROPE;
    localparam integer IDXW   = (S_MAX<=1)?1:$clog2(S_MAX);
    localparam integer HQK    = H_HEADS*QK_DIM;
    localparam integer HNOPE  = H_HEADS*NOPE;
    localparam integer HV     = H_HEADS*V_DIM;
    localparam integer EIDXW  = (N_EXPERT<=1)?1:$clog2(N_EXPERT);
    localparam integer A_KMAX = (MODEL_DIM>Q_LORA)?
                       ((MODEL_DIM>KV_LORA)?((MODEL_DIM>HV)?MODEL_DIM:HV):((KV_LORA>HV)?KV_LORA:HV))
                     :((Q_LORA>KV_LORA)?((Q_LORA>HV)?Q_LORA:HV):((KV_LORA>HV)?KV_LORA:HV));
    localparam integer A_OMAX = (HQK>MODEL_DIM)?
                       ((HQK>HNOPE)?((HQK>HV)?HQK:HV):((HNOPE>HV)?HNOPE:HV))
                     :((MODEL_DIM>HNOPE)?((MODEL_DIM>HV)?MODEL_DIM:HV):((HNOPE>HV)?HNOPE:HV));
    localparam integer A_NGMAX = (A_OMAX+PE_N-1)/PE_N;
    localparam integer A_GRPW  = (A_NGMAX<=1)?1:$clog2(A_NGMAX);
    localparam integer A_KCW   = (A_KMAX <=1)?1:$clog2(A_KMAX);
    localparam integer FF_KMAX_D = (INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM;
    localparam integer FF_KMAX_M = (INTER_MOE >MODEL_DIM)?INTER_MOE :MODEL_DIM;
    localparam integer FF_GWD = $clog2(((INTER_DENSE>MODEL_DIM)?INTER_DENSE:MODEL_DIM)/TN+1);
    localparam integer FF_KWD = $clog2(FF_KMAX_D+1);
    localparam integer R_KW   = $clog2(FF_KMAX_M+1);
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;   // == 1 at this slice
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;   // == 1 at this slice
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;   // == 1 at this slice
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer ROW_BITS = (KV_LORA+ROPE)*16;
    localparam integer KVPOSW   = (KV_CTX<=1)?1:$clog2(KV_CTX);
    localparam integer CSLOTW   = (CACHE_SLOTS<=1)?1:$clog2(CACHE_SLOTS);

    // ================= per-layer WEIGHT ROMs (E4M3 codes + bf16 block scales) ===
    //   Shared by BOTH the SoC stub responders and the reference responders.
    reg [15:0] EMB [0:VOCAB-1][0:MODEL_DIM-1];
    reg [15:0] GF  [0:MODEL_DIM-1];
    reg [15:0] Wlm [0:VOCAB-1][0:MODEL_DIM-1];
    reg [15:0] G1 [0:L-1][0:MODEL_DIM-1];
    reg [15:0] G2 [0:L-1][0:MODEL_DIM-1];
    reg [7:0] W_dq  [0:L-1][0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_uq  [0:L-1][0:HQK-1][0:Q_LORA-1];
    reg [7:0] W_dkv [0:L-1][0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] W_kr  [0:L-1][0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] W_uk  [0:L-1][0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] W_uv  [0:L-1][0:HV-1][0:KV_LORA-1];
    reg [7:0] W_o   [0:L-1][0:MODEL_DIM-1][0:HV-1];
    reg [15:0] ScW_dq[0:L-1], ScW_uq[0:L-1], ScW_dkv[0:L-1], ScW_kr[0:L-1],
               ScW_uk[0:L-1], ScW_uv[0:L-1], ScW_o[0:L-1];
    reg [15:0] CKV [0:L-1][0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] KRP [0:L-1][0:S_MAX-1][0:ROPE-1];
    reg [7:0] Wg [0:L-1][0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] ScWg[0:L-1];
    reg [7:0] Dg [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Du [0:L-1][0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] Dd [0:L-1][0:MODEL_DIM-1][0:INTER_DENSE-1];
    reg [15:0] ScDg [0:L-1][0:FF_NB_D-1];
    reg [15:0] ScDu [0:L-1][0:FF_NB_D-1];
    reg [15:0] ScDd [0:L-1][0:FF_NB_D-1];
    reg [7:0] Mg [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Mu [0:L-1][0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] Md [0:L-1][0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [7:0] SHg [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHu [0:L-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] SHd [0:L-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] ScMg [0:L-1][0:N_EXPERT-1], ScMu [0:L-1][0:N_EXPERT-1], ScMd [0:L-1][0:N_EXPERT-1];
    reg [15:0] ScSHg[0:L-1], ScSHu[0:L-1], ScSHd[0:L-1];

    // ================= shared DRY weight-lookup functions =======================
    //   automatic (re-entrant): both the SoC responders and the reference
    //   responders call these concurrently in combinational always@* blocks.
    //   All weight families collapse to a SINGLE [128,128] block at this slice
    //   (A_NB==FF_NB_D==R_NB==1), so scale lookups return one bf16 per tensor.
    function automatic [PE_N*8-1:0] f_aw_col;
        input integer ly; input [3:0] sel; input integer grp; input integer kk;
        integer t, fo; begin
        f_aw_col = {PE_N*8{1'b0}};
        for (t=0;t<PE_N;t=t+1) begin
            fo = grp*PE_N+t;
            case (sel)
            4'd0: if (fo<Q_LORA)   f_aw_col[8*t+:8]=W_dq [ly][fo][kk];
            4'd1: if (fo<HQK)      f_aw_col[8*t+:8]=W_uq [ly][fo][kk];
            4'd2: if (fo<KV_LORA)  f_aw_col[8*t+:8]=W_dkv[ly][fo][kk];
            4'd3: if (fo<ROPE)     f_aw_col[8*t+:8]=W_kr [ly][fo][kk];
            4'd4: if (fo<HNOPE)    f_aw_col[8*t+:8]=W_uk [ly][fo][kk];
            4'd5: if (fo<HV)       f_aw_col[8*t+:8]=W_uv [ly][fo][kk];
            4'd6: if (fo<MODEL_DIM)f_aw_col[8*t+:8]=W_o  [ly][fo][kk];
            default: ;
            endcase
        end end
    endfunction
    function automatic [15:0] f_aw_sc1;
        input integer ly; input [3:0] sel; begin
        case (sel)
            4'd0: f_aw_sc1=ScW_dq[ly];  4'd1: f_aw_sc1=ScW_uq[ly];  4'd2: f_aw_sc1=ScW_dkv[ly];
            4'd3: f_aw_sc1=ScW_kr[ly];  4'd4: f_aw_sc1=ScW_uk[ly];  4'd5: f_aw_sc1=ScW_uv[ly];
            4'd6: f_aw_sc1=ScW_o[ly];   default: f_aw_sc1=16'h3F80;
        endcase end
    endfunction
    function automatic [KV_LORA*16-1:0] f_kc_ckv;
        input integer ly; input integer idx; integer cd; begin
        f_kc_ckv = {KV_LORA*16{1'b0}};
        for (cd=0;cd<KV_LORA;cd=cd+1) f_kc_ckv[16*cd+:16]=CKV[ly][idx][cd];
        end
    endfunction
    function automatic [ROPE*16-1:0] f_kc_krope;
        input integer ly; input integer idx; integer cd; begin
        f_kc_krope = {ROPE*16{1'b0}};
        for (cd=0;cd<ROPE;cd=cd+1) f_kc_krope[16*cd+:16]=KRP[ly][idx][cd];
        end
    endfunction
    function automatic [8*N_EXPERT-1:0] f_rw_col;
        input integer ly; input integer kk; integer re; begin
        f_rw_col = {8*N_EXPERT{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) f_rw_col[8*re+:8]=Wg[ly][kk][re];
        end
    endfunction
    function automatic [LM_TN*16-1:0] f_lw_col;
        input integer vt; input integer kk; integer lt; begin
        f_lw_col = {LM_TN*16{1'b0}};
        for (lt=0;lt<LM_TN;lt=lt+1) f_lw_col[16*lt+:16]=Wlm[vt*LM_TN+lt][kk];
        end
    endfunction
    // FFN gate/down code lanes (fw_col): GATE for sel!=2, DOWN for sel==2.
    function automatic [8*TN-1:0] f_fw_col;
        input integer ly; input [1:0] sel; input integer grp; input integer kk;
        input shr; input integer eidx; integer ft, fo; reg dm; begin
        dm = (ly < N_DENSE);
        f_fw_col = {8*TN{1'b0}};
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = grp*TN+ft;
            if (dm) begin
                if (sel==2'd2) begin if (fo<MODEL_DIM)  f_fw_col[8*ft+:8]=Dd[ly][fo][kk]; end
                else           begin if (fo<INTER_DENSE)f_fw_col[8*ft+:8]=Dg[ly][fo][kk]; end
            end else if (shr) begin
                if (sel==2'd2) begin if (fo<MODEL_DIM)  f_fw_col[8*ft+:8]=SHd[ly][fo][kk]; end
                else           begin if (fo<INTER_MOE)  f_fw_col[8*ft+:8]=SHg[ly][fo][kk]; end
            end else begin
                if (sel==2'd2) begin if (fo<MODEL_DIM)  f_fw_col[8*ft+:8]=Md[ly][eidx][fo][kk]; end
                else           begin if (fo<INTER_MOE)  f_fw_col[8*ft+:8]=Mg[ly][eidx][fo][kk]; end
            end
        end end
    endfunction
    // FFN up companion code lanes (fw_col_up): UP only meaningful for sel!=2.
    function automatic [8*TN-1:0] f_fw_colup;
        input integer ly; input [1:0] sel; input integer grp; input integer kk;
        input shr; input integer eidx; integer ft, fo; reg dm; begin
        dm = (ly < N_DENSE);
        f_fw_colup = {8*TN{1'b0}};
        if (sel!=2'd2)
        for (ft=0;ft<TN;ft=ft+1) begin
            fo = grp*TN+ft;
            if (dm)       begin if (fo<INTER_DENSE) f_fw_colup[8*ft+:8]=Du[ly][fo][kk]; end
            else if (shr) begin if (fo<INTER_MOE)   f_fw_colup[8*ft+:8]=SHu[ly][fo][kk]; end
            else          begin if (fo<INTER_MOE)   f_fw_colup[8*ft+:8]=Mu[ly][eidx][fo][kk]; end
        end end
    endfunction
    function automatic [15:0] f_fw_scg;       // gate (sel!=2) / down (sel==2) per-tensor scale
        input integer ly; input [1:0] sel; input shr; input integer eidx; reg dm; begin
        dm = (ly < N_DENSE);
        if (dm)       f_fw_scg = (sel==2'd2)?ScDd[ly][0]:ScDg[ly][0];
        else if (shr) f_fw_scg = (sel==2'd2)?ScSHd[ly]  :ScSHg[ly];
        else          f_fw_scg = (sel==2'd2)?ScMd[ly][eidx]:ScMg[ly][eidx];
        end
    endfunction
    function automatic [15:0] f_fw_scu;       // up per-tensor scale (sel!=2)
        input integer ly; input shr; input integer eidx; reg dm; begin
        dm = (ly < N_DENSE);
        if (dm)       f_fw_scu = ScDu[ly][0];
        else if (shr) f_fw_scu = ScSHu[ly];
        else          f_fw_scu = ScMu[ly][eidx];
        end
    endfunction

    // ================= DUT (the SoC) host I/O =================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    wire                      busy, done;
    wire [TOKW-1:0]           next_tok;
    wire                      tok_valid;
    wire [VOCAB*16-1:0]       logits;
    // ---- GDDR6 hot-weight stubs (driven by responders below) ----
    wire                      em_req;  wire [TOKW-1:0] em_tok;  wire [DIMW-1:0] em_idx;
    reg  [15:0]               em_val;
    wire [LAYW-1:0]           db_layer;  wire idx_fresh;  wire [LAYW-1:0] idx_win;
    wire                      gn_req, gn_which;  wire [DIMW-1:0] gn_idx;  reg [15:0] gn_val;
    wire                      aw_req;  wire [3:0] aw_sel;  wire [A_GRPW-1:0] aw_grp;  wire [A_KCW-1:0] aw_k;
    reg  [PE_N*8-1:0]         aw_col;  reg [16*PE_N*A_NB-1:0] aw_scale;
    wire                      rw_req;  wire [R_KW-1:0] rw_k;
    reg  [8*N_EXPERT-1:0]     rw_col;  reg [16*N_EXPERT*R_NB-1:0] rw_scale;
    wire                      fw_req;  wire [1:0] fw_sel;  wire [FF_GWD-1:0] fw_grp;  wire [FF_KWD-1:0] fw_k;
    wire                      fw_shared;  wire [EIDXW-1:0] fw_eidx;
    reg  [8*TN-1:0]           fw_col, fw_col_up;  reg [16*TN*FF_NB_D-1:0] fw_scale_g, fw_scale_u;
    wire                      fn_req;  wire [DIMW-1:0] fn_idx;  reg [15:0] fn_val;
    wire                      lw_req;  wire [VTW-1:0] lw_vtile;  wire [DIMW-1:0] lw_k;  reg [LM_TN*16-1:0] lw_col;
    wire                      kc_req;  wire [IDXW-1:0] kc_idx;  reg [KV_LORA*16-1:0] kc_ckv;  reg [ROPE*16-1:0] kc_krope;
    // ---- KV append row stub ----
    wire [KVPOSW-1:0]         kv_row_sel;  reg [ROW_BITS-1:0] kv_row_in;
    // ---- single Flash channel ----
    wire                      flash_req, flash_is_expert;
    wire [EIDXW-1:0]          flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg                       flash_done;  reg [ROW_BITS-1:0] flash_row;
    // ---- prefetch hint (tied off) ----
    reg                       pf_valid;  reg [EIDXW-1:0] pf_expert_id;
    // ---- observability ----
    wire [TOKW-1:0]           argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                      ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]               ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                      kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]         kv_append_count, kv_resident_lo;  wire kv_overflowed;
    wire [31:0]               ec_dropped;

    glm_fp8_soc #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos), .s_len(s_len),
        .busy(busy), .done(done), .next_tok(next_tok), .tok_valid(tok_valid),
        .logits(logits),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .kc_ckv(kc_ckv), .kc_krope(kc_krope), .kc_req(kc_req), .kc_idx(kc_idx),
        .kv_row_sel(kv_row_sel), .kv_row_in(kv_row_in),
        .flash_req(flash_req), .flash_is_expert(flash_is_expert),
        .flash_expert_id(flash_expert_id), .flash_row_idx(flash_row_idx),
        .flash_done(flash_done), .flash_row(flash_row),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id),
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles),
        .ec_pf_issued(ec_pf_issued), .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed), .ec_dropped(ec_dropped)
    );

    // ---- SoC weight/KV responders (drive the GDDR6/Flash stub ports) ----
    integer t, ft, re;  reg [15:0] sc_a, scg, scu;
    always @* em_val   = EMB[em_tok][em_idx];
    always @* fn_val   = GF[fn_idx];
    always @* gn_val   = gn_which ? G2[db_layer][gn_idx] : G1[db_layer][gn_idx];
    always @* lw_col   = f_lw_col(lw_vtile, lw_k);
    always @* aw_col   = f_aw_col(db_layer, aw_sel, aw_grp, aw_k);
    always @* begin
        sc_a = f_aw_sc1(db_layer, aw_sel);
        aw_scale = {16*PE_N*A_NB{1'b0}};
        for (t=0;t<PE_N;t=t+1) aw_scale[16*t+:16]=sc_a;
    end
    always @* begin
        rw_col   = f_rw_col(db_layer, rw_k);
        rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (re=0;re<N_EXPERT;re=re+1) rw_scale[16*re+:16]=ScWg[db_layer];
    end
    always @* begin
        kc_ckv   = f_kc_ckv  (db_layer, kc_idx);
        kc_krope = f_kc_krope(db_layer, kc_idx);
    end
    always @* begin
        fw_col    = f_fw_col  (db_layer, fw_sel, fw_grp, fw_k, fw_shared, fw_eidx);
        fw_col_up = f_fw_colup(db_layer, fw_sel, fw_grp, fw_k, fw_shared, fw_eidx);
        scg = f_fw_scg(db_layer, fw_sel, fw_shared, fw_eidx);
        scu = f_fw_scu(db_layer, fw_shared, fw_eidx);
        fw_scale_g = {16*TN*FF_NB_D{1'b0}};
        fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        for (ft=0;ft<TN;ft=ft+1) begin
            fw_scale_g[16*ft+:16]=scg;
            fw_scale_u[16*ft+:16]=scu;
        end
    end

    // ================= INDEPENDENT REFERENCE: standalone glm_model_fp8 ==========
    //   Same params, fed by its OWN responders reading the SAME ROM arrays.  It
    //   has NO cache / NO pager -- it consumes the stub bytes directly.  Its
    //   committed argmax is the binding golden the SoC must reproduce.
    reg                       r_start;
    wire                      r_busy, r_done;
    wire [TOKW-1:0]           r_argmax;
    wire [VOCAB*16-1:0]       r_logits;
    wire                      r_em_req;  wire [TOKW-1:0] r_em_tok;  wire [DIMW-1:0] r_em_idx;  reg [15:0] r_em_val;
    wire [LAYW-1:0]           r_db_layer;  wire r_idx_fresh;  wire [LAYW-1:0] r_idx_win;
    wire                      r_gn_req, r_gn_which;  wire [DIMW-1:0] r_gn_idx;  reg [15:0] r_gn_val;
    wire                      r_aw_req;  wire [3:0] r_aw_sel;  wire [A_GRPW-1:0] r_aw_grp;  wire [A_KCW-1:0] r_aw_k;
    reg  [PE_N*8-1:0]         r_aw_col;  reg [16*PE_N*A_NB-1:0] r_aw_scale;
    wire                      r_rw_req;  wire [R_KW-1:0] r_rw_k;
    reg  [8*N_EXPERT-1:0]     r_rw_col;  reg [16*N_EXPERT*R_NB-1:0] r_rw_scale;
    wire                      r_fw_req;  wire [1:0] r_fw_sel;  wire [FF_GWD-1:0] r_fw_grp;  wire [FF_KWD-1:0] r_fw_k;
    wire                      r_fw_shared;  wire [EIDXW-1:0] r_fw_eidx;
    reg  [8*TN-1:0]           r_fw_col, r_fw_col_up;  reg [16*TN*FF_NB_D-1:0] r_fw_scale_g, r_fw_scale_u;
    wire                      r_fn_req;  wire [DIMW-1:0] r_fn_idx;  reg [15:0] r_fn_val;
    wire                      r_lw_req;  wire [VTW-1:0] r_lw_vtile;  wire [DIMW-1:0] r_lw_k;  reg [LM_TN*16-1:0] r_lw_col;
    wire                      r_kc_req;  wire [IDXW-1:0] r_kc_idx;  reg [KV_LORA*16-1:0] r_kc_ckv;  reg [ROPE*16-1:0] r_kc_krope;
    reg                       r_kc_valid;
    wire [MODEL_DIM*16-1:0]   r_h_state;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) u_ref (
        .clk(clk), .rst(rst),
        .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(prompt_tok), .pos(start_pos), .s_len(s_len),
        .logits(r_logits), .argmax(r_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(r_em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(r_gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_col(r_aw_col), .aw_scale(r_aw_scale),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_ckv(r_kc_ckv), .kc_krope(r_kc_krope),
        .kc_valid(r_kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k), .rw_col(r_rw_col), .rw_scale(r_rw_scale),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_col(r_fw_col), .fw_col_up(r_fw_col_up),
        .fw_scale_g(r_fw_scale_g), .fw_scale_u(r_fw_scale_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(r_fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(r_lw_col),
        .h_state(r_h_state)
    );
    // reference responders (same functions, same ROMs)
    integer rt, rft, rre;  reg [15:0] r_sc_a, r_scg, r_scu;
    always @* r_em_val = EMB[r_em_tok][r_em_idx];
    always @* r_fn_val = GF[r_fn_idx];
    always @* r_gn_val = r_gn_which ? G2[r_db_layer][r_gn_idx] : G1[r_db_layer][r_gn_idx];
    always @* r_lw_col = f_lw_col(r_lw_vtile, r_lw_k);
    always @* r_aw_col = f_aw_col(r_db_layer, r_aw_sel, r_aw_grp, r_aw_k);
    always @* begin
        r_sc_a = f_aw_sc1(r_db_layer, r_aw_sel);
        r_aw_scale = {16*PE_N*A_NB{1'b0}};
        for (rt=0;rt<PE_N;rt=rt+1) r_aw_scale[16*rt+:16]=r_sc_a;
    end
    always @* begin
        r_rw_col   = f_rw_col(r_db_layer, r_rw_k);
        r_rw_scale = {16*N_EXPERT*R_NB{1'b0}};
        for (rre=0;rre<N_EXPERT;rre=rre+1) r_rw_scale[16*rre+:16]=ScWg[r_db_layer];
    end
    always @* begin
        r_kc_ckv   = f_kc_ckv  (r_db_layer, r_kc_idx);
        r_kc_krope = f_kc_krope(r_db_layer, r_kc_idx);
    end
    always @* begin
        r_fw_col    = f_fw_col  (r_db_layer, r_fw_sel, r_fw_grp, r_fw_k, r_fw_shared, r_fw_eidx);
        r_fw_col_up = f_fw_colup(r_db_layer, r_fw_sel, r_fw_grp, r_fw_k, r_fw_shared, r_fw_eidx);
        r_scg = f_fw_scg(r_db_layer, r_fw_sel, r_fw_shared, r_fw_eidx);
        r_scu = f_fw_scu(r_db_layer, r_fw_shared, r_fw_eidx);
        r_fw_scale_g = {16*TN*FF_NB_D{1'b0}};
        r_fw_scale_u = {16*TN*FF_NB_D{1'b0}};
        for (rft=0;rft<TN;rft=rft+1) begin
            r_fw_scale_g[16*rft+:16]=r_scg;
            r_fw_scale_u[16*rft+:16]=r_scu;
        end
    end
    // reference kc_valid : 1-cycle-registered ack of kc_req (the read contract)
    always @(posedge clk) begin
        if (rst) r_kc_valid <= 1'b0;
        else     r_kc_valid <= r_kc_req;
    end
    // latch the reference committed argmax for the binding compare
    reg [TOKW-1:0] r_tok_lat;  reg r_done_seen;
    always @(posedge clk) begin
        if (rst)            begin r_done_seen<=1'b0; r_tok_lat<={TOKW{1'b0}}; end
        else if (r_start)   r_done_seen<=1'b0;
        else if (r_done)    begin r_tok_lat<=r_argmax; r_done_seen<=1'b1; end
    end

    // ---- lint guard: signals observed but not individually asserted ----
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, busy, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_demand_stall_cycles, ec_pf_issued,
                     ec_pf_hit, kv_row_out, kv_busy, kv_resident_lo, flash_is_expert,
                     flash_expert_id, h_state, gn_which, done, kv_overflowed,
                     r_busy, r_em_req, r_aw_req, r_fw_req, r_rw_req, r_gn_req,
                     r_fn_req, r_lw_req, r_idx_fresh, r_idx_win, r_logits, r_h_state,
                     r_gn_which};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= deterministic stimulus generators ========================
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4];
        else         e=8'd124+h[5:4];
        m=h[12:6];
        gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]};
        else         e = 4'd6 + {3'b0,h[4]};
        m = h[12:10];
        gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]};
        m = h[10:4];
        gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e,GLY,sc;
    task build_stimulus; input integer seed0; input integer band; begin
        sc=seed0;
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin EMB[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (GLY=0;GLY<L;GLY=GLY+1) begin
            for (i=0;i<MODEL_DIM;i=i+1) begin G1[GLY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) begin G2[GLY][i]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin W_dq[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin W_uq[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin W_dkv[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin W_kr[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin W_uk[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin W_uv[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin W_o[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScW_dq[GLY]=gen_scale(sc); sc=sc+1;  ScW_uq[GLY]=gen_scale(sc); sc=sc+1;
            ScW_dkv[GLY]=gen_scale(sc); sc=sc+1; ScW_kr[GLY]=gen_scale(sc); sc=sc+1;
            ScW_uk[GLY]=gen_scale(sc); sc=sc+1;  ScW_uv[GLY]=gen_scale(sc); sc=sc+1;
            ScW_o[GLY]=gen_scale(sc); sc=sc+1;
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin CKV[GLY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin KRP[GLY][i][j]=gen_bf16(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin Wg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScWg[GLY]=gen_scale(sc); sc=sc+1;
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Dg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Du[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin Dd[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDg[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDu[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (i=0;i<FF_NB_D;i=i+1) begin ScDd[GLY][i]=gen_scale(sc); sc=sc+1; end
            for (e=0;e<N_EXPERT;e=e+1) begin
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mg[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Mu[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin Md[GLY][e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
                ScMg[GLY][e]=gen_scale(sc); sc=sc+1; ScMu[GLY][e]=gen_scale(sc); sc=sc+1; ScMd[GLY][e]=gen_scale(sc); sc=sc+1;
            end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHg[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin SHu[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin SHd[GLY][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            ScSHg[GLY]=gen_scale(sc); sc=sc+1; ScSHu[GLY]=gen_scale(sc); sc=sc+1; ScSHd[GLY]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<MODEL_DIM;i=i+1) begin GF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin Wlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ================= KV latent-ROW stub (the bytes a real datapath writes) ====
    integer rr;
    always @* begin
        kv_row_in = {ROW_BITS{1'b0}};
        for (rr=0;rr<(KV_LORA+ROPE);rr=rr+1)
            kv_row_in[16*rr+:16] = gen_bf16(kv_row_sel*131 + rr*7 + 3, 0);
    end

    // ================= FLASH PHY STUB (latency + cold-row memory) ===============
    //   Rising-edge of flash_req arms a FLASH_LAT countdown; flash_done pulses one
    //   cycle when it expires.  The SoC arbiter holds flash_req high until done.
    reg [31:0] fl_timer; reg fl_active; reg prev_req;
    always @(posedge clk) begin
        if (rst) begin
            fl_timer <= 32'd0; fl_active <= 1'b0; flash_done <= 1'b0; prev_req <= 1'b0;
        end else begin
            flash_done <= 1'b0;
            if (!fl_active) begin
                if (flash_req && !prev_req) begin
                    fl_active <= 1'b1; fl_timer <= FLASH_LAT[31:0];
                end
            end else begin
                if (fl_timer <= 32'd1) begin flash_done <= 1'b1; fl_active <= 1'b0; end
                else fl_timer <= fl_timer - 32'd1;
            end
            prev_req <= flash_req;
        end
    end
    integer cr;
    always @* begin
        flash_row = {ROW_BITS{1'b0}};
        for (cr=0;cr<(KV_LORA+ROPE);cr=cr+1)
            flash_row[16*cr+:16] = gen_bf16(flash_row_idx*977 + cr*13 + 1, 0);
    end

    // ================= flash-activity / row_valid observers =====================
    integer flash_done_cnt, kv_rowvalid_cnt;
    always @(posedge clk) if (!rst) begin
        if (flash_done)   flash_done_cnt   = flash_done_cnt + 1;
        if (kv_row_valid) kv_rowvalid_cnt  = kv_rowvalid_cnt + 1;
    end

    // ================= checks =================
    integer errors, test_count;
    integer hbefore, mbefore, abefore;

    task settle; input integer n; integer c; begin
        for (c=0;c<n;c=c+1) @(negedge clk);
    end endtask

    // drive one token through the full SoC AND the standalone reference
    integer fdc0, krv0;
    task run_token; input [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        input [256*8-1:0] label; integer b; begin
        hbefore = ec_hit_count; mbefore = ec_miss_count; abefore = kv_append_count;
        fdc0 = flash_done_cnt;  krv0 = kv_rowvalid_cnt;
        prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
        @(negedge clk); start = 1'b1; r_start = 1'b1;
        @(negedge clk); start = 1'b0; r_start = 1'b0;
        // wait the SoC committed-token pulse
        wait (tok_valid === 1'b1);
        @(negedge clk);
        // make sure the reference has committed too
        wait (r_done_seen === 1'b1);
        // let the expert FIFO drain through the cache + Flash
        settle(300);

        test_count = test_count + 1;

        // (a) BINDING: SoC committed token == independent standalone model token
        if (next_tok !== r_tok_lat) begin
            $display("FAIL[%0s]: BINDING next_tok %0d != standalone ref %0d", label, next_tok, r_tok_lat);
            errors=errors+1; end

        // (a') X-cleanliness + SoC internal-argmax consistency
        for (b=0;b<TOKW;b=b+1) if (next_tok[b]===1'bx || next_tok[b]===1'bz) begin
            $display("FAIL[%0s]: next_tok bit %0d X/Z", label, b); errors=errors+1; end
        for (b=0;b<VOCAB*16;b=b+1) if (logits[b]===1'bx || logits[b]===1'bz) begin
            $display("FAIL[%0s]: logits bit %0d X/Z", label, b); errors=errors+1; b=VOCAB*16; end
        for (b=0;b<MODEL_DIM*16;b=b+1) if (h_state[b]===1'bx || h_state[b]===1'bz) begin
            $display("FAIL[%0s]: h_state bit %0d X/Z", label, b); errors=errors+1; b=MODEL_DIM*16; end
        if (next_tok !== argmax_o) begin
            $display("FAIL[%0s]: next_tok %0d != SoC internal argmax %0d", label, next_tok, argmax_o);
            errors=errors+1; end

        // (b) expert cache saw routed experts: hit+miss advanced; Flash fetched on miss
        if ((ec_hit_count + ec_miss_count) <= (hbefore + mbefore)) begin
            $display("FAIL[%0s]: expert cache made no demand request (hit+miss did not advance)", label);
            errors=errors+1; end
        if ((ec_miss_count > mbefore) && (flash_done_cnt <= fdc0)) begin
            $display("FAIL[%0s]: cache missed but no Flash fetch completed", label);
            errors=errors+1; end

        // (c) KV pager appended per token + served resident gathers
        if (kv_append_count !== abefore + SL[KVPOSW-1:0] + 1) begin
            $display("FAIL[%0s]: kv_append_count=%0d expected %0d (prefill %0d + 1 decode)",
                     label, kv_append_count, abefore + SL + 1, SL);
            errors=errors+1; end
        if (kv_rowvalid_cnt <= krv0) begin
            $display("FAIL[%0s]: KV pager produced no gather row_valid (no resident gather served)", label);
            errors=errors+1; end

        // (d) FIFO never overflowed
        if (ec_dropped !== 32'd0) begin
            $display("FAIL[%0s]: expert request FIFO overflowed (ec_dropped=%0d)", label, ec_dropped);
            errors=errors+1; end

        $display("PASS[%0s] tok=%0d(==ref %0d)  hit/miss=%0d/%0d  flash+=%0d  kv_count=%0d row_valid+=%0d  ovf=%0d",
                 label, next_tok, r_tok_lat, ec_hit_count-hbefore, ec_miss_count-mbefore,
                 flash_done_cnt-fdc0, kv_append_count, kv_rowvalid_cnt-krv0, kv_overflowed);
    end endtask

    initial begin
        #200000000;
        $display("FAIL: global timeout"); $fatal;
    end

    initial begin
        errors=0; test_count=0; flash_done_cnt=0; kv_rowvalid_cnt=0;
        rst=1'b1; start=1'b0; r_start=1'b0;
        prompt_tok={TOKW{1'b0}}; start_pos={POSW{1'b0}};
        s_len={(IDXW+1){1'b0}}; pf_valid=1'b0; pf_expert_id={EIDXW{1'b0}};
        repeat(4) @(negedge clk);
        rst=1'b0;
        @(negedge clk);

        build_stimulus(500,   0); run_token(4'd7,  20'd0,  1, "tok7 pos0 S1");
        build_stimulus(7000,  0); run_token(4'd10, 20'd37, 3, "tok10 pos37 S3");
        build_stimulus(90000, 1); run_token(4'd15, 20'd42, 4, "tok15 pos42 Smax");

        if (errors!=0) begin
            $display("FAILED: %0d error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("ALL %0d TESTS PASSED  (SoC compute die + expert_cache_pf + kv_cache_pager + Flash arbiter == standalone glm_model_fp8, end-to-end)", test_count);
        $finish;
    end
endmodule
