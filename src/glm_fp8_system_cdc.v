`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_fp8_system_cdc.v  --  TWO-CLOCK (multi-domain) WRAPPER around the verified
//                           single-clock glm_fp8_system GLM-5.2-FP8 box.
//----------------------------------------------------------------------------
// WHY THIS EXISTS
//   A real GLM-5.2-FP8 chip is NOT single-clock: the USB-C device controller that
//   talks to the host runs on the USB SerDes recovered clock, while the compute
//   die (glm_fp8_system: glm_model_fp8 + expert_cache_pf + kv_cache_pager +
//   ddr5_xbar + weight_loader) runs on its own core clock.  The two clocks are
//   ASYNCHRONOUS -- unrelated frequency and phase.  This wrapper keeps the verified
//   compute box UNCHANGED, instantiates it entirely on core_clk, and presents the
//   SAME host-facing interface (start / prompt_tok / start_pos / s_len ->
//   busy / done / next_tok / tok_valid) but now sampled on host_clk.  EVERY signal
//   that crosses between the two domains does so ONLY through a cdc_async_fifo
//   (gray-coded pointers + 2-FF synchronizers) or an explicit single-bit 2-FF
//   synchronizer.  NO raw multi-bit value is ever sampled directly across the
//   boundary; there is NO combinational path between the two domains.
//
//============================================================================
// 2-CLOCK BLOCK DIAGRAM  (which signal crosses which way, through what)
//
//   ┌──────────────── host_clk domain (USB-C device) ───────────────┐
//   │  start ─(rising-edge)─┐                                        │
//   │  prompt_tok ┐         │   pack {prompt_tok,start_pos,s_len}    │
//   │  start_pos  ┼─────────┴──▶ REQUEST cdc_async_fifo  ───────────┐│  host_clk ─▶ core_clk
//   │  s_len      ┘             (gray ptrs + 2-FF sync)             ││  (multi-bit, FIFO)
//   │                                                               ││
//   │  busy  ◀── (host_pending | 2-FF sync of sys_busy) ◀───────────┼┼─ sys_busy   (1-bit, 2-FF)
//   │  done  ◀── (edge-detect of 2-FF-synced done TOGGLE) ◀─────────┼┼─ done_tgl_c (1-bit, 2-FF)
//   │  next_tok ◀┐                                                  ││
//   │  tok_valid◀┴── TOKEN cdc_async_fifo read side ◀───────────────┼┼─ sys_next_tok (multi-bit,
//   │                (gray ptrs + 2-FF sync)                        ││   pushed on sys_tok_valid)
//   └───────────────────────────────────────────────────────────────┘│  core_clk ─▶ host_clk
//                                                                      │
//   ┌──────────────── core_clk domain (compute die) ─────────────────┐│
//   │  REQUEST fifo read ─▶ unpack ─▶ 1-cycle sys_start pulse ─▶┐     ││
//   │                                                          ▼     ││
//   │   glm_fp8_system  (UNCHANGED, clk=core_clk, rst=core_rst)       ││
//   │     .busy=sys_busy  .done=sys_done                             ││
//   │     .tok_valid=sys_tok_valid  .next_tok=sys_next_tok           ││
//   │     ...all weight/KV/Flash/DDR5/loader/observability ports.....││─▶ wrapper memory-side
//   │   sys_tok_valid ─▶ TOKEN fifo write side                       ││   ports (core_clk domain)
//   │   sys_done ─▶ done_tgl_c (toggle)                              ││
//   └───────────────────────────────────────────────────────────────┘
//
//   CROSSINGS (every one is gray-FIFO or 2-FF -- NO raw multi-bit crossing):
//     host->core : {prompt_tok,start_pos,s_len}  via REQUEST cdc_async_fifo
//     core->host : next_tok                       via TOKEN   cdc_async_fifo
//     core->host : busy (level)                   via 2-FF synchronizer
//     core->host : done (pulse -> TOGGLE)         via 2-FF synchronizer + edge det
//
//   The memory-side ports (GDDR6/Flash/DDR5/loader/observability) belong wholly to
//   the core_clk domain -- glm_fp8_system runs entirely on core_clk -- so they are
//   passed straight through to the wrapper ports with NO crossing (the host never
//   touches them; in a full SoC their own controllers live in the memory clocks).
//
// STYLE: synchronous ACTIVE-HIGH resets per domain; NO latch (every reg assigned on
//   every path); NO combinational loop; NO combinational path between domains.
//============================================================================
module glm_fp8_system_cdc #(
    // ---- compute-die (glm_model_fp8) slice config -- passed straight through --
    parameter integer MODEL_DIM  = 128,
    parameter integer L          = 6,
    parameter integer N_DENSE    = 3,
    parameter integer VOCAB      = 256,
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
    parameter [31:0]  RSCALE     = 32'h40200000,
    parameter integer TN         = 4,
    parameter integer BLK        = 128,
    parameter integer LM_TN      = 4,
    // ---- memory-system config ----
    parameter integer CACHE_SLOTS = 4,
    parameter integer FLASH_LAT   = 8,
    parameter integer KV_CTX      = 1024,
    parameter integer KV_RESIDENT = 16,
    parameter integer EFIFO_DEPTH = 16,
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 4,
    parameter integer DDR_ADDR_W  = 32,
    parameter integer DDR_DATA_W  = 256,
    parameter integer DDR_TAG_W   = 8,
    parameter integer DDR_ROW_LAT = 10,
    parameter integer DDR_RESP_QD = 4,
    // ---- weight_loader (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,
    parameter integer WL_ADDR_W   = 24,
    parameter integer LOADER_KLEN = MODEL_DIM,
    // ---- CDC FIFO depths (this wrapper) ----
    parameter integer REQ_AW      = 2,      // request FIFO addr width (depth 2**AW)
    parameter integer TOK_AW      = 3,      // token   FIFO addr width (depth 2**AW)
    // ====================================================================
    // derived (do NOT override) -- mirror glm_fp8_system's port-width derivations
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
    parameter integer A_NB       = (A_KMAX    + BLK - 1) / BLK,
    parameter integer FF_NB_D    = (FF_KMAX_D + BLK - 1) / BLK,
    parameter integer R_NB       = (FF_KMAX_M + BLK - 1) / BLK,
    parameter integer LAYW       = (L     <= 1) ? 1 : $clog2(L),
    parameter integer TOKW       = (VOCAB <= 1) ? 1 : $clog2(VOCAB),
    parameter integer DIMW       = (MODEL_DIM <= 1) ? 1 : $clog2(MODEL_DIM),
    parameter integer NVTILE     = VOCAB / LM_TN,
    parameter integer VTW        = (NVTILE <= 1) ? 1 : $clog2(NVTILE),
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer WL_PE_N    = PE_N,
    parameter integer WL_DATA_W  = (8*PE_N >= 16) ? 8*PE_N : 16,
    // ---- this wrapper's packed-request width ----
    parameter integer REQ_W      = TOKW + POSW + (IDXW+1)
)(
    //========================== TWO ASYNCHRONOUS CLOCK DOMAINS ===============
    input  wire                          host_clk,   // USB-C device domain
    input  wire                          host_rst,   // sync, active-high (host)
    input  wire                          core_clk,   // compute-die domain
    input  wire                          core_rst,   // sync, active-high (core)

    //========================== HOST interface (sampled on host_clk) ========
    input  wire                          start,
    input  wire [TOKW-1:0]               prompt_tok,
    input  wire [POSW-1:0]               start_pos,
    input  wire [IDXW:0]                 s_len,
    output reg                           busy,
    output reg                           done,
    output reg  [TOKW-1:0]               next_tok,
    output reg                           tok_valid,

    //====== everything below is the core_clk domain, passed straight through ===
    output wire [VOCAB*16-1:0]           logits,

    //========================== GDDR6 HOT-weight STUBS ======================
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,
    output wire                          fw_req,
    output wire [1:0]                    fw_sel,
    output wire [FF_GWD-1:0]             fw_grp,
    output wire [FF_KWD-1:0]             fw_k,
    output wire                          fw_shared,
    output wire [EIDXW-1:0]              fw_eidx,
    input  wire [8*TN-1:0]               fw_col,
    input  wire [8*TN-1:0]               fw_col_up,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_g,
    input  wire [16*TN*FF_NB_D-1:0]      fw_scale_u,
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,

    //========================== KV append (latent ROW source) ===============
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,

    //========================== SINGLE FLASH CHANNEL (to PHY/TB) ============
    output wire                          flash_req,
    output wire                          flash_is_expert,
    output wire [EIDXW-1:0]              flash_expert_id,
    output wire [KVPOSW-1:0]             flash_row_idx,
    input  wire                          flash_done,
    input  wire [ROW_BITS-1:0]           flash_row,

    //========================== expert prefetch hint (optional) =============
    input  wire                          pf_valid,
    input  wire [EIDXW-1:0]              pf_expert_id,

    //========================== DDR5 fabric channels ========================
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,

    //========================== weight_loader staging memory ================
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    //========================== observability ===============================
    output wire [TOKW-1:0]               argmax_o,
    output wire [MODEL_DIM*16-1:0]       h_state,
    output wire                          mdl_busy,
    output wire                          ec_resp_valid,
    output wire                          ec_hit,
    output wire [CSLOTW-1:0]             ec_resp_slot,
    output wire                          ec_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [31:0]                   ec_demand_stall_cycles,
    output wire [31:0]                   ec_pf_issued,
    output wire [31:0]                   ec_pf_hit,
    output wire                          kv_row_valid,
    output wire [ROW_BITS-1:0]           kv_row_out,
    output wire                          kv_busy,
    output wire [KVPOSW-1:0]             kv_append_count,
    output wire [KVPOSW-1:0]             kv_resident_lo,
    output wire                          kv_overflowed,
    output wire [31:0]                   ec_dropped,
    output wire [31:0]                   xbar_req_count,
    output wire [31:0]                   xbar_resp_count,
    output wire                          xbar_resp_valid,
    output wire [DDR_DATA_W-1:0]         xbar_resp_data,
    output wire                          loader_busy,
    output wire [31:0]                   loader_done_count,
    output wire [31:0]                   loader_beat_count,
    output wire [8*WL_PE_N-1:0]          loader_w_row,
    output wire                          loader_in_valid
);
    // ======================================================================
    // Per-domain active-LOW resets for the cdc_async_fifo primitive (its reset
    // convention is active-low, sampled synchronously in each clock).
    // ======================================================================
    wire host_rst_n = ~host_rst;
    wire core_rst_n = ~core_rst;

    // ======================================================================
    // ============================ host_clk DOMAIN =========================
    //  Request push : pack {prompt_tok,start_pos,s_len}, push on a rising edge of
    //  `start` (one request per assertion) when the request FIFO is not full.
    // ======================================================================
    reg  start_d;
    always @(posedge host_clk) begin
        if (host_rst) start_d <= 1'b0;
        else          start_d <= start;
    end
    wire start_rise = start & ~start_d;

    wire             req_wr_full;
    wire             req_wr_en   = start_rise & ~req_wr_full;
    wire [REQ_W-1:0] req_wr_data = {prompt_tok, start_pos, s_len};

    // ======================================================================
    // REQUEST cdc_async_fifo : host_clk (write) -> core_clk (read).
    //  The ONLY path the multi-bit host request takes across the boundary.
    // ======================================================================
    wire             req_rd_empty;
    wire [REQ_W-1:0] req_rd_data;
    wire             req_rd_en;

    cdc_async_fifo #(
        .DATA_W (REQ_W),
        .ADDR_W (REQ_AW)
    ) u_req_fifo (
        .wclk   (host_clk), .wrst_n (host_rst_n),
        .wr_en  (req_wr_en), .wr_data(req_wr_data), .full(req_wr_full),
        .rclk   (core_clk), .rrst_n (core_rst_n),
        .rd_en  (req_rd_en), .rd_data(req_rd_data), .empty(req_rd_empty)
    );

    // ======================================================================
    // ============================ core_clk DOMAIN =========================
    //  Request pop : single-outstanding pop.  The cdc_async_fifo read is
    //  REGISTERED, so a word popped this cycle is valid on req_rd_data NEXT cycle
    //  (req_rd_d).  Unpack it into holding regs and pulse sys_start for exactly
    //  one core_clk -- start + fields are driven from the SAME edge so they are
    //  aligned at the glm_fp8_system input (which samples start in its H_IDLE).
    // ======================================================================
    reg               req_rd_d;
    reg               sys_start;
    reg  [TOKW-1:0]   sys_prompt_tok;
    reg  [POSW-1:0]   sys_start_pos;
    reg  [IDXW:0]     sys_s_len;

    assign req_rd_en = ~req_rd_empty & ~req_rd_d;

    always @(posedge core_clk) begin
        if (core_rst) begin
            req_rd_d       <= 1'b0;
            sys_start      <= 1'b0;
            sys_prompt_tok <= {TOKW{1'b0}};
            sys_start_pos  <= {POSW{1'b0}};
            sys_s_len      <= {(IDXW+1){1'b0}};
        end else begin
            sys_start <= 1'b0;
            req_rd_d  <= req_rd_en;
            if (req_rd_d) begin
                {sys_prompt_tok, sys_start_pos, sys_s_len} <= req_rd_data;
                sys_start <= 1'b1;
            end
        end
    end

    // ----------------------------------------------------------------------
    // THE VERIFIED COMPUTE BOX -- instantiated UNCHANGED, entirely on core_clk.
    // ----------------------------------------------------------------------
    wire             sys_busy;
    wire             sys_done;
    wire [TOKW-1:0]  sys_next_tok;
    wire             sys_tok_valid;

    glm_fp8_system #(
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
    ) u_sys (
        .clk(core_clk), .rst(core_rst),
        // host port -- now driven from the core-side request unpack / captured back
        .start(sys_start), .prompt_tok(sys_prompt_tok),
        .start_pos(sys_start_pos), .s_len(sys_s_len),
        .busy(sys_busy), .done(sys_done),
        .next_tok(sys_next_tok), .tok_valid(sys_tok_valid),
        .logits(logits),
        // ---- everything else: pure core-domain pass-through ----
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

    // ======================================================================
    // TOKEN cdc_async_fifo : core_clk (write) -> host_clk (read).
    //  Push the produced token on each sys_tok_valid pulse.
    // ======================================================================
    wire            tok_wr_full;
    wire            tok_wr_en   = sys_tok_valid & ~tok_wr_full;

    wire            tok_rd_empty;
    wire [TOKW-1:0] tok_rd_data;
    wire            tok_rd_en;

    cdc_async_fifo #(
        .DATA_W (TOKW),
        .ADDR_W (TOK_AW)
    ) u_tok_fifo (
        .wclk   (core_clk), .wrst_n (core_rst_n),
        .wr_en  (tok_wr_en), .wr_data(sys_next_tok), .full(tok_wr_full),
        .rclk   (host_clk), .rrst_n (host_rst_n),
        .rd_en  (tok_rd_en), .rd_data(tok_rd_data), .empty(tok_rd_empty)
    );

    // ----------------------------------------------------------------------
    // host_clk token pop : single-outstanding read; the registered FIFO read
    // makes the popped word valid one host_clk later (tok_rd_d), which we latch
    // into next_tok and surface as a one-cycle tok_valid pulse.
    // ----------------------------------------------------------------------
    reg tok_rd_d;
    assign tok_rd_en = ~tok_rd_empty & ~tok_rd_d;

    always @(posedge host_clk) begin
        if (host_rst) begin
            tok_rd_d  <= 1'b0;
            next_tok  <= {TOKW{1'b0}};
            tok_valid <= 1'b0;
        end else begin
            tok_valid <= 1'b0;
            tok_rd_d  <= tok_rd_en;
            if (tok_rd_d) begin
                next_tok  <= tok_rd_data;
                tok_valid <= 1'b1;
            end
        end
    end

    // ======================================================================
    // STATUS crossings core_clk -> host_clk (single-bit only).
    //   busy : a LEVEL  -> plain 2-FF synchronizer.
    //   done : a PULSE  -> a TOGGLE in the core domain, 2-FF synced, edge-detected
    //          in the host domain (a level-sync could miss a 1-core-cycle pulse;
    //          a toggle survives because the host domain only needs to see the
    //          single-bit flip eventually -- metastability-safe).
    // ======================================================================
    // ---- busy (level) 2-FF sync ----
    reg busy_s1, busy_s2;
    always @(posedge host_clk) begin
        if (host_rst) begin
            busy_s1 <= 1'b0;
            busy_s2 <= 1'b0;
        end else begin
            busy_s1 <= sys_busy;
            busy_s2 <= busy_s1;
        end
    end

    // ---- done toggle (core) ----
    reg done_tgl_c;
    always @(posedge core_clk) begin
        if (core_rst)        done_tgl_c <= 1'b0;
        else if (sys_done)   done_tgl_c <= ~done_tgl_c;
    end

    // ---- done toggle 2-FF sync + edge detect (host) ----
    reg done_tgl_h1, done_tgl_h2, done_tgl_h3;
    wire done_edge = done_tgl_h3 ^ done_tgl_h2;
    always @(posedge host_clk) begin
        if (host_rst) begin
            done_tgl_h1 <= 1'b0;
            done_tgl_h2 <= 1'b0;
            done_tgl_h3 <= 1'b0;
        end else begin
            done_tgl_h1 <= done_tgl_c;
            done_tgl_h2 <= done_tgl_h1;
            done_tgl_h3 <= done_tgl_h2;
        end
    end

    // ----------------------------------------------------------------------
    // host-facing busy/done.
    //   host_pending : set when a request is accepted into the REQUEST FIFO,
    //   cleared when the run's done edge is observed.  This covers the launch gap
    //   before the (sync-delayed) core busy rises, so busy is asserted immediately
    //   on accept and stays high until the host-visible completion.
    // ----------------------------------------------------------------------
    reg host_pending;
    always @(posedge host_clk) begin
        if (host_rst) host_pending <= 1'b0;
        else begin
            if (req_wr_en)       host_pending <= 1'b1;
            else if (done_edge)  host_pending <= 1'b0;
        end
    end

    always @(posedge host_clk) begin
        if (host_rst) begin
            busy <= 1'b0;
            done <= 1'b0;
        end else begin
            busy <= host_pending | busy_s2;   // launch gap | synced core busy
            done <= done_edge;                // one host_clk completion pulse
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
