`timescale 1ns/1ps
`include "glm_fp.vh"
//============================================================================
// glm_fp8_system_cdc_tb.v -- BINDING TB for the TWO-CLOCK wrapper
//                            glm_fp8_system_cdc (src/glm_fp8_system_cdc.v).
//----------------------------------------------------------------------------
// WHAT IT PROVES  (the binding)
//   The verified single-clock glm_fp8_system, wrapped so its HOST interface is
//   sampled on an asynchronous host_clk while the compute box runs on core_clk,
//   generates EXACTLY the same token stream as the standalone compute die
//   (glm_model_fp8, u_ref) fed the SAME input directly -- i.e. crossing the
//   two unrelated clock domains does NOT change the generated tokens.
//
//     1) REFERENCE  : a standalone glm_model_fp8 (u_ref) is driven directly,
//        single-clock, with the SAME weight/KV stubs (shared ROMs), producing a
//        few-token autoregressive reference stream  ref_tok[0..NTOK-1].
//        (The wrapped glm_fp8_system simply wires glm_model_fp8 to these exact
//         same hot-weight / KV pull stubs and forwards mdl_argmax as next_tok;
//         the prefill/pager/fabric blocks are an observability path and do not
//         perturb the model's compute -- so the standalone die is the golden
//         token oracle for the wrapped system.)
//
//     2) BOTH CLOCK ORDERINGS : the wrapped DUT is then run autoregressively
//        through its host_clk interface TWICE, across ASYNCHRONOUS clocks:
//          PHASE A : host FASTER than core (host 7 ns / core 10 ns)
//          PHASE B : host SLOWER than core (host 11 ns / core 8 ns)
//        Each phase the host-collected token stream must EQUAL ref_tok exactly.
//        The live host/core phase offset is sampled and shown to drift widely
//        (a genuine CDC stress), and the two periods are asserted unrelated.
//
//     3) CDC SOUNDNESS :
//          * the host-visible synchronized outputs (busy/done/tok_valid/next_tok)
//            are NEVER X/Z after reset (2-FF + toggle-sync + FIFO all settle clean);
//          * the REQUEST fifo never OVERFLOWS (no start_rise dropped while full)
//            and the TOKEN fifo never OVERFLOWS (no produced token dropped while
//            full);  every issued request produces exactly one token, IN ORDER
//            (#requests == #tokens == NTOK each phase -> no loss/dup/underflow);
//          * busy was observed asserted during the run and done pulsed on
//            completion (status synchronizers live).
//
//   The memory side (GDDR6/Flash/DDR5/loader staging/KV) is modeled here exactly
//   as the single-clock glm_fp8_system_tb does, clocked on core_clk (the compute
//   box and all its memory ports live entirely in the core domain).
//
//   Prints "ALL <N> TESTS PASSED"; $fatal on any mismatch.
//============================================================================
module glm_fp8_system_cdc_tb;

    // ---- TWO ASYNCHRONOUS CLOCKS (period set per phase from these reals) ----
    real HOST_HALF = 3.5;   // host_clk half-period (USB-C device domain)
    real CORE_HALF = 5.0;   // core_clk half-period (compute die)
    reg  host_clk = 1'b0;
    reg  core_clk = 1'b0;
    always #(HOST_HALF) host_clk = ~host_clk;
    always #(CORE_HALF) core_clk = ~core_clk;
    reg host_rst, core_rst;

    // ================= small-but-faithful slice (mirrors glm_fp8_system_tb) =====
    localparam integer MODEL_DIM  = 16;
    localparam integer L          = 4;
    localparam integer N_DENSE    = 2;
    localparam integer VOCAB      = 16;
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 4;
    localparam integer ROPE       = 4;
    localparam integer V_DIM      = 4;
    localparam integer Q_LORA     = 8;
    localparam integer KV_LORA    = 8;
    localparam integer S_MAX      = 4;
    localparam integer TOPK_ATTN  = 4;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 4;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 4;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 16;
    localparam integer INTER_DENSE= 32;
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 4;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 4;
    localparam integer CACHE_SLOTS = 2;
    localparam integer FLASH_LAT   = 8;
    localparam integer KV_CTX      = 1024;
    localparam integer KV_RESIDENT = 16;
    localparam integer EFIFO_DEPTH = 16;
    localparam integer DDR_NCH     = 4;
    localparam integer DDR_ADDR_W  = 32;
    localparam integer DDR_DATA_W  = 256;
    localparam integer DDR_TAG_W   = 8;
    localparam integer DDR_ROW_LAT = 10;
    localparam integer DDR_RESP_QD = 4;
    localparam integer WL_KMAX     = 256;
    localparam integer WL_ADDR_W   = 24;
    localparam integer LOADER_KLEN = MODEL_DIM;

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
    localparam integer A_NB    = (A_KMAX   +BLK-1)/BLK;
    localparam integer FF_NB_D = (FF_KMAX_D+BLK-1)/BLK;
    localparam integer R_NB    = (FF_KMAX_M+BLK-1)/BLK;
    localparam integer LAYW   = (L<=1)?1:$clog2(L);
    localparam integer TOKW   = (VOCAB<=1)?1:$clog2(VOCAB);
    localparam integer DIMW   = (MODEL_DIM<=1)?1:$clog2(MODEL_DIM);
    localparam integer NVTILE = VOCAB/LM_TN;
    localparam integer VTW    = (NVTILE<=1)?1:$clog2(NVTILE);
    localparam integer ROW_BITS = (KV_LORA+ROPE)*16;
    localparam integer KVPOSW   = (KV_CTX<=1)?1:$clog2(KV_CTX);
    localparam integer CSLOTW   = (CACHE_SLOTS<=1)?1:$clog2(CACHE_SLOTS);
    localparam integer WL_DATA_W= (8*PE_N>=16)?8*PE_N:16;

    // ================= per-layer WEIGHT ROMs (SHARED by DUT + u_ref) ============
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
    function automatic [15:0] f_fw_scg;
        input integer ly; input [1:0] sel; input shr; input integer eidx; reg dm; begin
        dm = (ly < N_DENSE);
        if (dm)       f_fw_scg = (sel==2'd2)?ScDd[ly][0]:ScDg[ly][0];
        else if (shr) f_fw_scg = (sel==2'd2)?ScSHd[ly]  :ScSHg[ly];
        else          f_fw_scg = (sel==2'd2)?ScMd[ly][eidx]:ScMg[ly][eidx];
        end
    endfunction
    function automatic [15:0] f_fw_scu;
        input integer ly; input shr; input integer eidx; reg dm; begin
        dm = (ly < N_DENSE);
        if (dm)       f_fw_scu = ScDu[ly][0];
        else if (shr) f_fw_scu = ScSHu[ly];
        else          f_fw_scu = ScMu[ly][eidx];
        end
    endfunction

    // ================= DUT host I/O (host_clk) =================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    wire                      busy, done;
    wire [TOKW-1:0]           next_tok;
    wire                      tok_valid;
    // ---- core-domain (pass-through) ports ----
    wire [VOCAB*16-1:0]       logits;
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
    wire [KVPOSW-1:0]         kv_row_sel;  reg [ROW_BITS-1:0] kv_row_in;
    wire                      flash_req, flash_is_expert;
    wire [EIDXW-1:0]          flash_expert_id;  wire [KVPOSW-1:0] flash_row_idx;
    reg                       flash_done;  reg [ROW_BITS-1:0] flash_row;
    reg                       pf_valid;  reg [EIDXW-1:0] pf_expert_id;
    wire [TOKW-1:0]           argmax_o;  wire [MODEL_DIM*16-1:0] h_state;  wire mdl_busy;
    wire                      ec_resp_valid, ec_hit;  wire [CSLOTW-1:0] ec_resp_slot;  wire ec_busy;
    wire [31:0]               ec_hit_count, ec_miss_count, ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit;
    wire                      kv_row_valid;  wire [ROW_BITS-1:0] kv_row_out;  wire kv_busy;
    wire [KVPOSW-1:0]         kv_append_count, kv_resident_lo;  wire kv_overflowed;
    wire [31:0]               ec_dropped;
    wire [DDR_NCH-1:0]            mem_req_valid;
    wire [DDR_NCH-1:0]           mem_req_ready;
    wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr;
    wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag;
    reg  [DDR_NCH-1:0]            mem_resp_valid;
    wire [DDR_NCH-1:0]            mem_resp_ready;
    reg  [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data;
    reg  [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag;
    wire                      wl_mem_en;  wire [WL_ADDR_W-1:0] wl_mem_addr;  reg [WL_DATA_W-1:0] wl_mem_data;
    wire [31:0]               xbar_req_count, xbar_resp_count;
    wire                      xbar_resp_valid;  wire [DDR_DATA_W-1:0] xbar_resp_data;
    wire                      loader_busy;  wire [31:0] loader_done_count, loader_beat_count;
    wire [8*PE_N-1:0]         loader_w_row;  wire loader_in_valid;

    glm_fp8_system_cdc #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN),
        .CACHE_SLOTS(CACHE_SLOTS), .FLASH_LAT(FLASH_LAT), .KV_CTX(KV_CTX),
        .KV_RESIDENT(KV_RESIDENT), .EFIFO_DEPTH(EFIFO_DEPTH),
        .DDR_NCH(DDR_NCH), .DDR_ADDR_W(DDR_ADDR_W), .DDR_DATA_W(DDR_DATA_W),
        .DDR_TAG_W(DDR_TAG_W), .DDR_ROW_LAT(DDR_ROW_LAT), .DDR_RESP_QD(DDR_RESP_QD),
        .WL_KMAX(WL_KMAX), .WL_ADDR_W(WL_ADDR_W), .LOADER_KLEN(LOADER_KLEN)
    ) dut (
        .host_clk(host_clk), .host_rst(host_rst),
        .core_clk(core_clk), .core_rst(core_rst),
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
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .wl_mem_en(wl_mem_en), .wl_mem_addr(wl_mem_addr), .wl_mem_data(wl_mem_data),
        .argmax_o(argmax_o), .h_state(h_state), .mdl_busy(mdl_busy),
        .ec_resp_valid(ec_resp_valid), .ec_hit(ec_hit), .ec_resp_slot(ec_resp_slot),
        .ec_busy(ec_busy), .ec_hit_count(ec_hit_count), .ec_miss_count(ec_miss_count),
        .ec_demand_stall_cycles(ec_demand_stall_cycles),
        .ec_pf_issued(ec_pf_issued), .ec_pf_hit(ec_pf_hit),
        .kv_row_valid(kv_row_valid), .kv_row_out(kv_row_out), .kv_busy(kv_busy),
        .kv_append_count(kv_append_count), .kv_resident_lo(kv_resident_lo),
        .kv_overflowed(kv_overflowed), .ec_dropped(ec_dropped),
        .xbar_req_count(xbar_req_count), .xbar_resp_count(xbar_resp_count),
        .xbar_resp_valid(xbar_resp_valid), .xbar_resp_data(xbar_resp_data),
        .loader_busy(loader_busy), .loader_done_count(loader_done_count),
        .loader_beat_count(loader_beat_count),
        .loader_w_row(loader_w_row), .loader_in_valid(loader_in_valid)
    );

    // ---- DUT weight/KV responders (combinational; core-domain ports) ----
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

    //========================================================================
    // GOLDEN ORACLE -- standalone glm_model_fp8 (u_ref), single-clock (core_clk),
    // fed DIRECTLY with the SAME weight/KV stubs (shared ROMs).  This is the
    // exact compute die the wrapped glm_fp8_system instantiates; its argmax is
    // the per-token reference the CDC DUT must reproduce across async clocks.
    //========================================================================
    reg                       r_start;
    reg  [TOKW-1:0]           r_token_id;
    reg  [POSW-1:0]           r_pos;
    reg  [IDXW:0]             r_s_len;
    wire                      r_busy, r_done;
    wire [VOCAB*16-1:0]       r_logits;
    wire [TOKW-1:0]           r_argmax;
    wire                      r_em_req;  wire [TOKW-1:0] r_em_tok;  wire [DIMW-1:0] r_em_idx;
    reg  [15:0]               r_em_val;
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
        .clk(core_clk), .rst(core_rst),
        .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(r_token_id), .pos(r_pos), .s_len(r_s_len),
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

    // kc_valid : 1-cycle-registered ack of kc_req (the verified read contract,
    // exactly as glm_fp8_system wires it to glm_model_fp8).
    always @(posedge core_clk) begin
        if (core_rst) r_kc_valid <= 1'b0;
        else          r_kc_valid <= r_kc_req;
    end

    // ---- u_ref weight/KV responders (same shared ROMs, ref's own indices) ----
    integer rt, rft, rre;  reg [15:0] r_sc_a, r_scg, r_scu;
    always @* r_em_val   = EMB[r_em_tok][r_em_idx];
    always @* r_fn_val   = GF[r_fn_idx];
    always @* r_gn_val   = r_gn_which ? G2[r_db_layer][r_gn_idx] : G1[r_db_layer][r_gn_idx];
    always @* r_lw_col   = f_lw_col(r_lw_vtile, r_lw_k);
    always @* r_aw_col   = f_aw_col(r_db_layer, r_aw_sel, r_aw_grp, r_aw_k);
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

    // ---- lint guard for observability ports not actively checked ----
    /* verilator lint_off UNUSEDSIGNAL */
    wire _unused = &{1'b0, em_req, aw_req, fw_req, rw_req, gn_req, fn_req,
                     lw_req, idx_fresh, idx_win, mdl_busy, ec_resp_valid, ec_hit,
                     ec_resp_slot, ec_busy, ec_hit_count, ec_miss_count,
                     ec_demand_stall_cycles, ec_pf_issued, ec_pf_hit, kv_row_out,
                     kv_busy, kv_resident_lo, kv_append_count, kv_overflowed,
                     flash_is_expert, flash_expert_id, h_state, gn_which,
                     ec_dropped, xbar_req_count, xbar_resp_count, xbar_resp_valid,
                     xbar_resp_data, loader_busy, loader_done_count,
                     loader_beat_count, loader_w_row, loader_in_valid, mem_req_tag,
                     mem_req_valid, kv_row_valid,
                     r_em_req, r_aw_req, r_fw_req, r_rw_req, r_gn_req, r_fn_req,
                     r_lw_req, r_idx_fresh, r_idx_win, r_busy, r_logits, r_h_state};
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

    // ================= KV latent-ROW stub (combinational; core-domain) ==========
    integer rr;
    always @* begin
        kv_row_in = {ROW_BITS{1'b0}};
        for (rr=0;rr<(KV_LORA+ROPE);rr=rr+1)
            kv_row_in[16*rr+:16] = gen_bf16(kv_row_sel*131 + rr*7 + 3, 0);
    end

    // ================= FLASH PHY STUB (core_clk) =================
    reg [31:0] fl_timer; reg fl_active; reg prev_req;
    always @(posedge core_clk) begin
        if (core_rst) begin
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

    // ================= LOADER STAGING-TIER RAM (latency-1, core_clk) =============
    localparam integer STG_DEPTH = 2048;
    reg [WL_DATA_W-1:0] STAGE [0:STG_DEPTH-1];
    integer si;
    initial for (si=0; si<STG_DEPTH; si=si+1)
        STAGE[si] = {gen_bf16(si*53+9,0), gen_bf16(si*29+5,1)};
    always @(posedge core_clk) begin
        if (wl_mem_en) wl_mem_data <= STAGE[wl_mem_addr[10:0]];
        else           wl_mem_data <= {WL_DATA_W{1'b0}};
    end

    // ================= DDR5 PER-CHANNEL MEMORY MODEL (core_clk) ==================
    localparam integer NINF = 64;
    reg                  infv  [0:NINF-1];
    reg [DDR_TAG_W-1:0]  inftg [0:NINF-1];
    reg [DDR_ADDR_W-1:0] infad [0:NINF-1];
    reg [15:0]           inftm [0:NINF-1];
    integer ii, cc, kk;

    assign mem_req_ready = {DDR_NCH{1'b1}};

    reg [31:0] presIdx [0:DDR_NCH-1];
    reg        presV   [0:DDR_NCH-1];
    function automatic [DDR_DATA_W-1:0] gen_beat;
        input [DDR_TAG_W-1:0] tg; input [DDR_ADDR_W-1:0] ad; integer ln; begin
        gen_beat = {DDR_DATA_W{1'b0}};
        for (ln=0; ln<DDR_DATA_W/16; ln=ln+1)
            gen_beat[16*ln+:16] = gen_bf16(({16'd0,tg}*7 + ad*3 + ln*5 + 1), 0);
        end
    endfunction
    always @* begin
        mem_resp_valid = {DDR_NCH{1'b0}};
        mem_resp_data  = {(DDR_NCH*DDR_DATA_W){1'b0}};
        mem_resp_tag   = {(DDR_NCH*DDR_TAG_W){1'b0}};
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            presV[cc]   = 1'b0;
            presIdx[cc] = 32'd0;
        end
        for (cc=0; cc<DDR_NCH; cc=cc+1) begin
            for (ii=NINF-1; ii>=0; ii=ii-1) begin
                if (infv[ii] && (inftm[ii]=={16{1'b0}}) &&
                    (infad[ii][$clog2(DDR_NCH)-1:0] == cc[$clog2(DDR_NCH)-1:0])) begin
                    presV[cc]   = 1'b1;
                    presIdx[cc] = ii;
                end
            end
            if (presV[cc]) begin
                mem_resp_valid[cc] = 1'b1;
                mem_resp_data[cc*DDR_DATA_W +: DDR_DATA_W] =
                    gen_beat(inftg[presIdx[cc]], infad[presIdx[cc]]);
                mem_resp_tag[cc*DDR_TAG_W +: DDR_TAG_W] = inftg[presIdx[cc]];
            end
        end
    end

    integer freeslot; reg got_free;
    always @(posedge core_clk) begin
        if (core_rst) begin
            for (ii=0; ii<NINF; ii=ii+1) begin
                infv[ii]  <= 1'b0; inftg[ii] <= {DDR_TAG_W{1'b0}};
                infad[ii] <= {DDR_ADDR_W{1'b0}}; inftm[ii] <= 16'd0;
            end
        end else begin
            for (ii=0; ii<NINF; ii=ii+1)
                if (infv[ii] && (inftm[ii]!={16{1'b0}})) inftm[ii] <= inftm[ii]-16'd1;
            for (cc=0; cc<DDR_NCH; cc=cc+1)
                if (presV[cc] && mem_resp_ready[cc]) infv[presIdx[cc]] <= 1'b0;
            got_free = 1'b0; freeslot = 0;
            for (ii=NINF-1; ii>=0; ii=ii-1)
                if (!infv[ii]) begin got_free = 1'b1; freeslot = ii; end
            for (cc=0; cc<DDR_NCH; cc=cc+1) begin
                if (mem_req_valid[cc] && mem_req_ready[cc] && got_free) begin
                    infv[freeslot]  <= 1'b1;
                    inftg[freeslot] <= mem_req_tag[cc*DDR_TAG_W +: DDR_TAG_W];
                    infad[freeslot] <= mem_req_addr[cc*DDR_ADDR_W +: DDR_ADDR_W];
                    inftm[freeslot] <= DDR_ROW_LAT[15:0];
                end
            end
        end
    end

    // ================= ASYNC-CLOCK WITNESS (live phase drift) ====================
    real last_host_edge;
    real phase_min, phase_max;
    integer phase_samples;
    real off;
    always @(posedge host_clk) last_host_edge = $realtime;
    always @(posedge core_clk) begin
        if (last_host_edge >= 0.0) begin
            off = $realtime - last_host_edge;
            if (phase_samples == 0) begin phase_min = off; phase_max = off; end
            else begin
                if (off < phase_min) phase_min = off;
                if (off > phase_max) phase_max = off;
            end
            phase_samples = phase_samples + 1;
        end
    end
    task reset_witness; begin
        last_host_edge=-1.0; phase_min=0.0; phase_max=0.0; phase_samples=0;
    end endtask

    // ================= CDC SOUNDNESS monitors ===================================
    // host-domain: request issue / token receive counters + busy/done observers +
    // overflow + X-clean of the synchronized host-visible outputs.
    integer reqs_issued, toks_recv;
    reg     saw_busy, saw_done;
    reg     req_overflow;   // a start_rise dropped because REQUEST fifo was full
    reg     tok_overflow;   // a produced token dropped because TOKEN fifo was full
    reg     sync_x;         // busy/done/tok_valid X/Z after reset
    reg     tok_x;          // next_tok X/Z when a token was presented

    always @(posedge host_clk) if (!host_rst) begin
        if (dut.req_wr_en)  reqs_issued <= reqs_issued + 1;
        if (tok_valid)      toks_recv   <= toks_recv   + 1;
        if (busy)           saw_busy    <= 1'b1;
        if (done)           saw_done    <= 1'b1;
        if (dut.start_rise && dut.req_wr_full) req_overflow <= 1'b1;
        if ((busy===1'bx)||(busy===1'bz)||
            (done===1'bx)||(done===1'bz)||
            (tok_valid===1'bx)||(tok_valid===1'bz)) sync_x <= 1'b1;
        if (tok_valid && ((^next_tok)===1'bx)) tok_x <= 1'b1;
    end
    // core-domain: TOKEN fifo overflow witness (produced while full)
    always @(posedge core_clk) if (!core_rst) begin
        if (dut.sys_tok_valid && dut.tok_wr_full) tok_overflow <= 1'b1;
    end

    // ================= scoreboard ===============================================
    integer errors, test_count, b, k;
    localparam integer NTOK = 3;
    reg [TOKW-1:0] ref_in  [0:NTOK-1];
    reg [TOKW-1:0] ref_tok [0:NTOK-1];
    reg [TOKW-1:0] dut_tok [0:NTOK-1];
    reg [TOKW-1:0] cur, outtok;

    task chk_true; input [255:0] name; input cond; begin
        test_count = test_count + 1;
        if (cond === 1'b1) $display("  PASS %0s", name);
        else begin errors=errors+1; $display("  FAIL %0s", name); end
    end endtask

    // s_len schedule (deterministic, <= S_MAX); pos = step index
    function integer slen_of; input integer kk2; begin
        slen_of = (kk2+1 <= S_MAX) ? (kk2+1) : S_MAX;
    end endfunction

    // ---- drive ONE token directly on the standalone reference (core_clk) ----
    task ref_one_token;
        input  [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        output [TOKW-1:0] o;
        begin
            @(negedge core_clk);
            r_token_id = tk; r_pos = ps; r_s_len = SL[IDXW:0];
            r_start = 1'b1;
            @(negedge core_clk);
            r_start = 1'b0;
            wait (r_done === 1'b1);
            o = r_argmax;
            @(negedge core_clk);
        end
    endtask

    // ---- drive ONE token across the CDC boundary on host_clk ----
    task dut_one_token;
        input  [TOKW-1:0] tk; input [POSW-1:0] ps; input integer SL;
        output [TOKW-1:0] o;
        begin
            @(negedge host_clk);
            prompt_tok = tk; start_pos = ps; s_len = SL[IDXW:0];
            start = 1'b1;
            @(negedge host_clk);
            start = 1'b0;
            wait (tok_valid === 1'b1);
            o = next_tok;
            @(negedge host_clk);
        end
    endtask

    // ---- reset both domains (independent, asynchronous lengths) ----
    task do_reset; begin
        host_rst = 1'b1; core_rst = 1'b1;
        start = 1'b0; r_start = 1'b0;
        repeat(6) @(negedge core_clk); core_rst = 1'b0;
        repeat(6) @(negedge host_clk); host_rst = 1'b0;
        repeat(4) @(negedge host_clk);
    end endtask

    // ---- run NTOK tokens autoregressively through the CDC DUT, compare to ref --
    task run_dut_phase; input [127:0] phname; begin
        reqs_issued=0; toks_recv=0; saw_busy=1'b0; saw_done=1'b0;
        reset_witness;
        cur = ref_in[0];
        for (k=0;k<NTOK;k=k+1) begin
            dut_one_token(cur, k[POSW-1:0], slen_of(k), outtok);
            dut_tok[k] = outtok;
            cur = outtok;           // autoregress on the DUT's own output
        end
        repeat(120) @(negedge core_clk);  // let in-flight core memory traffic settle

        $display("[%0s] host=%.1f ns core=%.1f ns  reqs=%0d toks=%0d saw_busy=%0b saw_done=%0b  phase[min=%.2f max=%.2f over %0d]",
                 phname, 2.0*HOST_HALF, 2.0*CORE_HALF, reqs_issued, toks_recv,
                 saw_busy, saw_done, phase_min, phase_max, phase_samples);
        for (k=0;k<NTOK;k=k+1)
            $display("    tok[%0d]: dut=%0d  ref=%0d  %0s",
                     k, dut_tok[k], ref_tok[k], (dut_tok[k]===ref_tok[k])?"OK":"MISMATCH");

        // ---- per-token token-match (the binding) ----
        for (k=0;k<NTOK;k=k+1)
            chk_true({phname, " token matches standalone glm_model_fp8"},
                     (dut_tok[k] === ref_tok[k]));
        // ---- FIFO soundness: one token per request, in order, no loss ----
        chk_true({phname, " #requests == NTOK"}, (reqs_issued === NTOK));
        chk_true({phname, " #tokens   == NTOK"}, (toks_recv   === NTOK));
        chk_true({phname, " REQUEST fifo never overflowed"}, ~req_overflow);
        chk_true({phname, " TOKEN fifo never overflowed"},  ~tok_overflow);
        // ---- synchronized outputs X/Z-clean ----
        chk_true({phname, " busy/done/tok_valid never X/Z"}, ~sync_x);
        chk_true({phname, " next_tok X/Z-clean on every token"}, ~tok_x);
        // ---- status synchronizers live ----
        chk_true({phname, " busy observed asserted during run"}, saw_busy);
        chk_true({phname, " done pulsed on completion"}, saw_done);
        // ---- genuinely asynchronous clocks this phase ----
        chk_true({phname, " host/core periods differ"}, (HOST_HALF != CORE_HALF));
        chk_true({phname, " host/core phase drift > 2 ns"}, ((phase_max - phase_min) > 2.0));
        // ---- core-domain pass-through state X/Z-clean after the run ----
        begin : xlog
            reg bad; bad=1'b0;
            for (b=0;b<VOCAB*16;b=b+1) if ((logits[b]===1'bx)||(logits[b]===1'bz)) bad=1'b1;
            chk_true({phname, " logits X/Z-clean"}, ~bad);
        end
        begin : xhs
            reg bad; bad=1'b0;
            for (b=0;b<MODEL_DIM*16;b=b+1) if ((h_state[b]===1'bx)||(h_state[b]===1'bz)) bad=1'b1;
            chk_true({phname, " h_state X/Z-clean"}, ~bad);
        end
    end endtask

    // ================= global timeout ===========================================
    initial begin
        #400000000;
        $display("FAIL: global timeout"); $fatal;
    end

    // ================= main =====================================================
    initial begin
        errors=0; test_count=0;
        reqs_issued=0; toks_recv=0; saw_busy=0; saw_done=0;
        req_overflow=0; tok_overflow=0; sync_x=0; tok_x=0;
        reset_witness;
        host_rst=1'b1; core_rst=1'b1;
        start=1'b0; prompt_tok={TOKW{1'b0}}; start_pos={POSW{1'b0}}; s_len={(IDXW+1){1'b0}};
        r_start=1'b0; r_token_id={TOKW{1'b0}}; r_pos={POSW{1'b0}}; r_s_len={(IDXW+1){1'b0}};
        pf_valid=1'b0; pf_expert_id={EIDXW{1'b0}};
        wl_mem_data={WL_DATA_W{1'b0}};
        flash_done=1'b0; flash_row={ROW_BITS{1'b0}};

        build_stimulus(500, 0);

        // ---------------------------------------------------------------------
        // PHASE 0 : build the GOLDEN reference token stream from the standalone
        // glm_model_fp8 (u_ref) -- single clock, fed directly.  host_rst stays
        // asserted so the host-domain monitors are quiet.  Clock-period choice is
        // irrelevant to the (deterministic) reference compute.
        // ---------------------------------------------------------------------
        HOST_HALF = 3.5; CORE_HALF = 5.0;
        host_rst = 1'b1; core_rst = 1'b1;
        repeat(6) @(negedge core_clk); core_rst = 1'b0;
        repeat(4) @(negedge core_clk);
        cur = 4'd7;                          // prompt seed token
        for (k=0;k<NTOK;k=k+1) begin
            ref_in[k] = cur;
            ref_one_token(cur, k[POSW-1:0], slen_of(k), outtok);
            ref_tok[k] = outtok;
            cur = outtok;                    // autoregress
        end
        $display("[ref] standalone glm_model_fp8 stream: tok0=%0d tok1=%0d tok2=%0d (seed=%0d)",
                 ref_tok[0], ref_tok[1], ref_tok[2], ref_in[0]);

        // ---------------------------------------------------------------------
        // PHASE A : host FASTER than core (host 7 ns / core 10 ns), async.
        // ---------------------------------------------------------------------
        HOST_HALF = 3.5; CORE_HALF = 5.0;
        do_reset;
        $display("[glm_fp8_system_cdc TB] PHASE A (host FASTER): host_clk=%.1f ns core_clk=%.1f ns",
                 2.0*HOST_HALF, 2.0*CORE_HALF);
        run_dut_phase("A:host-faster");

        // ---------------------------------------------------------------------
        // PHASE B : host SLOWER than core (host 11 ns / core 8 ns), async.
        // ---------------------------------------------------------------------
        HOST_HALF = 5.5; CORE_HALF = 4.0;
        do_reset;
        $display("[glm_fp8_system_cdc TB] PHASE B (host SLOWER): host_clk=%.1f ns core_clk=%.1f ns",
                 2.0*HOST_HALF, 2.0*CORE_HALF);
        run_dut_phase("B:host-slower");

        // ---- cross-phase integrity: both orderings yielded the SAME stream ----
        begin : xphase
            reg same; same=1'b1;
            for (k=0;k<NTOK;k=k+1) if (dut_tok[k] !== ref_tok[k]) same=1'b0;
            chk_true("both clock orderings produce the SAME token stream", same);
        end

        if (errors!=0) begin
            $display("\nFAILED: %0d error(s) across %0d tests", errors, test_count);
            $fatal;
        end
        $display("\nALL %0d TESTS PASSED  (glm_fp8_system_cdc: host_clk<->core_clk CDC wrapper -- standalone glm_model_fp8 token stream reproduced across BOTH host-faster and host-slower async clocks; request/token FIFOs never over/underflow; synchronized outputs X-clean)", test_count);
        $finish;
    end
endmodule
