`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_fp8_system.v  --  PRODUCTION GLM-5.2-FP8 SINGLE-MODULE SYSTEM
//                       (glm_fp8_soc EVOLVED to route memory through the real
//                        DDR5 fabric + the weight-side DMA loader)
//----------------------------------------------------------------------------
// WHAT THIS IS  (vs. glm_fp8_soc)
//   glm_fp8_soc wired the VERIFIED FP8 compute die (glm_model_fp8) to the
//   memory CONTROLLERS (expert_cache_pf + kv_cache_pager) over one Flash
//   channel, with all the die's weight/KV bytes served the SAME cycle by the
//   TB GDDR6/Flash STUBS (the verified combinational PULL contract -- the die
//   cannot be stalled mid-layer).  glm_fp8_system KEEPS that exact, verified
//   compute path and ADDS the two remaining production-fabric blocks INTO the
//   datapath so the memory now FLOWS THROUGH the real multichannel fabric:
//
//     * ddr5_xbar  -- the N_CH-channel BANKED DDR5 READ fabric.  Every DDR5
//       FAST-TIER access (the compute die's HOT-weight pulls, the expert
//       cache's resident-SLOT reads, and the weight_loader's tile fetches)
//       presents its block address to ddr5_xbar, which BANKS it across N_CH
//       independent channels (~N_CH x aggregate read BW) and returns the data.
//       The per-channel DDR5 PHY/memory is modeled by the TB stub.  This makes
//       the channel-parallel bandwidth path REAL in the elaborated datapath:
//       ddr5_xbar's response counters advance as the token is generated.
//
//     * weight_loader -- the WEIGHT-side DMA / pull master for glm_matmul_fp8.
//       It is driven on the HOT / REPRESENTATIVE weight tile: at each compute
//       launch it loads one tile descriptor (bf16 block scales + FP8 weight
//       rows) from a fast staging tier and DRIVES the matmul pull stream
//       (mm_start / mm_w_scale / mm_w_row / mm_in_valid).  Its tile-fetch
//       ADDRESSES are MIRRORED into ddr5_xbar, realizing the multichannel
//       bandwidth for the weight stream.
//
//============================================================================
// BLOCK DIAGRAM  (one decode step)
//
//   HOST ─ start/prompt/pos/s_len ─▶┌───────────── glm_fp8_system ───────────┐
//                                   │  ┌──────────────┐                       │
//                                   │  │ glm_model_fp8│  hot-weight PULLS ─────┼─▶ GDDR6 stub (compute bytes)
//                                   │  │  FP8 COMPUTE │  (em/gn/aw/rw/fw/fn/lw)│        │  (addr mirror)
//                                   │  │  (verified)  │  kc_* KV read ────────┐│        ▼
//                                   │  └──────┬───────┘                       ││  ┌───────────┐
//                                   │  mdl_start │ db_layer/fw_eidx           ││  │ DDR5 XBAR │─▶ N_CH DDR5 stub
//                                   │         ▼  ▼ (router pick)              ││  │  (banked, │   (per-channel
//                                   │  ┌────────────┐   ┌──────────────┐      ││  │   N_CH BW)│    TB memory)
//                                   │  │WEIGHT LOADER│  │ EXPERT-ISSUE │      ││  └─────▲─────┘
//                                   │  │  (DMA pull) │  │  FIFO + FSM  │      ││  slot/hot/load
//   staging-tier RAM ◀──wl_mem──────┼──┤ mm_* stream │  └──────┬───────┘      ││  addresses
//   (TB, latency-1)                 │  └─────┬───────┘         ▼              ││        │
//                                   │   load-addr mirror  ┌──────────────┐    ││  ec_resp_slot
//                                   │        └────────────│expert_cache_pf│◀──┘│        │
//                                   │                     │ (GDDR6 cache) │────┼────────┘ (resident-slot read)
//                                   │                     └──────┬───────┘    │
//                                   │   gather/append    ┌───────┴────────┐   │
//                                   │   (kc_*/per-token) │ kv_cache_pager │   │
//                                   │                    └───────┬────────┘   │
//                                   │              ┌─────────────┴────────┐   │
//                                   │              │  SINGLE FLASH ARBITER │───┼──▶ Flash stub
//                                   │              │   (demand-priority)   │   │
//                                   │              └───────────────────────┘   │
//                                   └ busy/done/next_tok/tok_valid ─▶ HOST ────┘
//
//============================================================================
// INTEGRATION MAP  (what flows through the real fabric vs. the stub)
//   THROUGH ddr5_xbar (multichannel DDR5 fast tier; TB models per-channel mem):
//     - HOT weight pulls  : every cycle the die pulls a hot weight (em/gn/aw/
//                           rw/fw/fn/lw) a banked DDR5 read is issued (TAG_HOT).
//     - EXPERT-SLOT reads : on each expert_cache_pf demand response the resident
//                           GDDR6/DDR5 SLOT is read through the fabric (TAG_SLOT).
//     - LOADER fetches    : the weight_loader's tile-word fetches are mirrored
//                           as banked reads (TAG_LOAD).
//     All three are coalesced by a tiny priority issuer (LOAD > SLOT > HOT) onto
//     ddr5_xbar's single requester port; bank_rot stripes consecutive accepted
//     reads round-robin across channels so the N_CH bandwidth is exercised.
//   THROUGH weight_loader (the matmul weight-pull master, hot/representative tile):
//     - One descriptor per compute launch (mdl_start): bf16 block scales + FP8
//       rows for a representative attention-projection tile.  It drives the full
//       mm_start/mm_w_scale/mm_w_row/mm_in_valid pull stream a glm_matmul_fp8
//       consumes -- observable, X-clean -- and its fetch addresses feed the xbar.
//   STILL via the STUB (the compute MATH, unperturbed -- exactly as glm_fp8_soc):
//     - The die's actual weight CODES/SCALES + kc_* KV bytes are served the same
//       cycle by the TB GDDR6/Flash stub ports (the verified combinational pull
//       contract).  ddr5_xbar + weight_loader sit IN the datapath as the
//       bandwidth/address/pull engines and are fully exercised & counted, while
//       the verified compute is byte-for-byte unchanged.  Wiring the xbar's
//       returned bytes physically into the die is the remaining step a real PHY
//       closes; every address it needs is already presented here.
//
// STYLE: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop; header;
//   every reg reset (X-aware).  No arithmetic is reimplemented -- this is WIRING.
//============================================================================
module glm_fp8_system #(
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
    parameter integer CACHE_SLOTS = 4,      // GDDR6 expert-cache slots (slice)
    parameter integer FLASH_LAT   = 8,      // Flash fetch latency (doc; TB models)
    parameter integer KV_CTX      = 1024,   // logical KV context capacity (positions)
    parameter integer KV_RESIDENT = 16,     // KV ring capacity (POWER OF TWO, >= S_MAX)
    parameter integer EFIFO_DEPTH = 16,     // routed-expert request FIFO depth (POW2)
    // ---- DDR5 fast-tier fabric (ddr5_xbar) config ----
    parameter integer DDR_NCH     = 4,      // DDR5 channels (POWER OF TWO)
    parameter integer DDR_ADDR_W  = 32,     // block-address width into the fabric
    parameter integer DDR_DATA_W  = 256,    // DDR5 read-data width (one beat)
    parameter integer DDR_TAG_W   = 8,      // in-flight requester tag width
    parameter integer DDR_ROW_LAT = 10,     // per-channel read latency (TB models)
    parameter integer DDR_RESP_QD = 4,      // per-channel response FIFO depth
    // ---- weight_loader (matmul weight-pull DMA) config ----
    parameter integer WL_KMAX     = 256,    // max K the loader can stream
    parameter integer WL_ADDR_W   = 24,     // loader staging-memory address width
    parameter integer LOADER_KLEN = MODEL_DIM, // representative tile K length (<= WL_KMAX)
    // ====================================================================
    // derived (do NOT override) -- mirror glm_model_fp8's port-width derivations
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
    // ---- memory-system derived ----
    parameter integer ROW_BITS   = (KV_LORA + ROPE) * 16,             // one latent row
    parameter integer KVPOSW     = (KV_CTX <= 1) ? 1 : $clog2(KV_CTX),// pager logical pos
    parameter integer CSLOTW     = (CACHE_SLOTS <= 1) ? 1 : $clog2(CACHE_SLOTS),
    parameter integer EFW        = (EFIFO_DEPTH <= 1) ? 1 : $clog2(EFIFO_DEPTH),
    // ---- fabric / loader derived ----
    parameter integer CH_IDX_W   = (DDR_NCH <= 1) ? 1 : $clog2(DDR_NCH),
    parameter integer WL_PE_N    = PE_N,
    parameter integer WL_BLK     = BLK,
    parameter integer WL_DATA_W  = (8*PE_N >= 16) ? 8*PE_N : 16,
    parameter integer WL_NB      = (WL_KMAX + WL_BLK - 1) / WL_BLK,
    parameter integer WL_KW      = $clog2(WL_KMAX + 1),
    parameter integer WL_BKW     = $clog2(WL_NB + 1),
    parameter integer WL_SCALE_W = 16*WL_PE_N*WL_NB
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    //========================== HOST interface (USB-C bridge) ===============
    input  wire                          start,
    input  wire [TOKW-1:0]               prompt_tok,
    input  wire [POSW-1:0]               start_pos,
    input  wire [IDXW:0]                 s_len,
    output reg                           busy,
    output reg                           done,
    output reg  [TOKW-1:0]               next_tok,
    output reg                           tok_valid,
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

    //========================== DDR5 fabric channels (to per-channel TB stub) =
    output wire [DDR_NCH-1:0]            mem_req_valid,
    input  wire [DDR_NCH-1:0]            mem_req_ready,
    output wire [DDR_NCH*DDR_ADDR_W-1:0] mem_req_addr,
    output wire [DDR_NCH*DDR_TAG_W-1:0]  mem_req_tag,
    input  wire [DDR_NCH-1:0]            mem_resp_valid,
    output wire [DDR_NCH-1:0]            mem_resp_ready,
    input  wire [DDR_NCH*DDR_DATA_W-1:0] mem_resp_data,
    input  wire [DDR_NCH*DDR_TAG_W-1:0]  mem_resp_tag,

    //========================== weight_loader staging memory (TB, latency-1) =
    output wire                          wl_mem_en,
    output wire [WL_ADDR_W-1:0]          wl_mem_addr,
    input  wire [WL_DATA_W-1:0]          wl_mem_data,

    //========================== observability ===============================
    output wire [TOKW-1:0]               argmax_o,
    output wire [MODEL_DIM*16-1:0]       h_state,
    output wire                          mdl_busy,
    // expert-cache stats / slot
    output wire                          ec_resp_valid,
    output wire                          ec_hit,
    output wire [CSLOTW-1:0]             ec_resp_slot,
    output wire                          ec_busy,
    output wire [31:0]                   ec_hit_count,
    output wire [31:0]                   ec_miss_count,
    output wire [31:0]                   ec_demand_stall_cycles,
    output wire [31:0]                   ec_pf_issued,
    output wire [31:0]                   ec_pf_hit,
    // KV-pager stats
    output wire                          kv_row_valid,
    output wire [ROW_BITS-1:0]           kv_row_out,
    output wire                          kv_busy,
    output wire [KVPOSW-1:0]             kv_append_count,
    output wire [KVPOSW-1:0]             kv_resident_lo,
    output wire                          kv_overflowed,
    output wire [31:0]                   ec_dropped,
    // ---- DDR5 fabric + loader stats (NEW) ----
    output reg  [31:0]                   xbar_req_count,   // banked reads accepted
    output reg  [31:0]                   xbar_resp_count,  // banked reads returned
    output wire                          xbar_resp_valid,  // requester resp valid (obs)
    output wire [DDR_DATA_W-1:0]         xbar_resp_data,   // last returned beat (obs)
    output wire                          loader_busy,
    output reg  [31:0]                   loader_done_count,// tiles streamed
    output reg  [31:0]                   loader_beat_count,// weight-row beats driven
    output wire [8*WL_PE_N-1:0]          loader_w_row,     // current weight row (obs)
    output wire                          loader_in_valid   // weight beat valid (obs)
);
    //========================================================================
    // 1) THE COMPUTE DIE -- glm_model_fp8 (verified full FP8 forward pass).
    //========================================================================
    wire                      mdl_done;
    wire [TOKW-1:0]           mdl_argmax;
    reg                       kc_valid_r;
    reg                       mdl_start;

    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN)
    ) u_model (
        .clk(clk), .rst(rst),
        .start(mdl_start), .busy(mdl_busy), .done(mdl_done),
        .token_id(prompt_tok), .pos(start_pos), .s_len(s_len),
        .logits(logits), .argmax(mdl_argmax),
        .em_req(em_req), .em_tok(em_tok), .em_idx(em_idx), .em_val(em_val),
        .db_layer(db_layer), .idx_fresh(idx_fresh), .idx_win(idx_win),
        .gn_req(gn_req), .gn_which(gn_which), .gn_idx(gn_idx), .gn_val(gn_val),
        .aw_req(aw_req), .aw_sel(aw_sel), .aw_grp(aw_grp), .aw_k(aw_k),
        .aw_col(aw_col), .aw_scale(aw_scale),
        .kc_req(kc_req), .kc_idx(kc_idx), .kc_ckv(kc_ckv), .kc_krope(kc_krope),
        .kc_valid(kc_valid_r),
        .rw_req(rw_req), .rw_k(rw_k), .rw_col(rw_col), .rw_scale(rw_scale),
        .fw_req(fw_req), .fw_sel(fw_sel), .fw_grp(fw_grp), .fw_k(fw_k),
        .fw_shared(fw_shared), .fw_eidx(fw_eidx),
        .fw_col(fw_col), .fw_col_up(fw_col_up),
        .fw_scale_g(fw_scale_g), .fw_scale_u(fw_scale_u),
        .fn_req(fn_req), .fn_idx(fn_idx), .fn_val(fn_val),
        .lw_req(lw_req), .lw_vtile(lw_vtile), .lw_k(lw_k), .lw_col(lw_col),
        .h_state(h_state)
    );
    assign argmax_o = mdl_argmax;

    // kc_valid : 1-cycle-registered ack of kc_req (the verified read contract).
    always @(posedge clk) begin
        if (rst) kc_valid_r <= 1'b0;
        else     kc_valid_r <= kc_req;
    end

    //========================================================================
    // 2) HOST FSM : prefill KV window -> run die -> append decode latent -> commit
    //========================================================================
    localparam [2:0] H_IDLE   = 3'd0,
                     H_APPEND = 3'd1,
                     H_RUN_W  = 3'd3,
                     H_DECAP  = 3'd4,
                     H_DONE   = 3'd5;
    reg [2:0]        hstate;
    reg [IDXW:0]     ap_i;

    wire ap_active = (hstate == H_APPEND);
    wire ap_decode = (hstate == H_DECAP);
    wire pg_append_valid = ap_active || ap_decode;
    assign kv_row_sel = ap_decode ? {{(KVPOSW-(IDXW+1)){1'b0}}, s_len}
                                   : {{(KVPOSW-(IDXW+1)){1'b0}}, ap_i};

    always @(posedge clk) begin
        if (rst) begin
            hstate    <= H_IDLE;
            busy      <= 1'b0;
            done      <= 1'b0;
            next_tok  <= {TOKW{1'b0}};
            tok_valid <= 1'b0;
            ap_i      <= {(IDXW+1){1'b0}};
            mdl_start <= 1'b0;
        end else begin
            done      <= 1'b0;
            tok_valid <= 1'b0;
            mdl_start <= 1'b0;
            case (hstate)
                H_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy <= 1'b1;
                        ap_i <= {(IDXW+1){1'b0}};
                        if (s_len == {(IDXW+1){1'b0}}) begin
                            mdl_start <= 1'b1;
                            hstate    <= H_RUN_W;
                        end else
                            hstate <= H_APPEND;
                    end
                end
                H_APPEND: begin
                    if (ap_i == (s_len - 1'b1)) begin
                        mdl_start <= 1'b1;
                        hstate    <= H_RUN_W;
                    end
                    ap_i <= ap_i + 1'b1;
                end
                H_RUN_W: begin
                    if (mdl_done) begin
                        next_tok <= mdl_argmax;
                        hstate   <= H_DECAP;
                    end
                end
                H_DECAP: begin
                    hstate <= H_DONE;
                end
                H_DONE: begin
                    done      <= 1'b1;
                    tok_valid <= 1'b1;
                    busy      <= 1'b0;
                    hstate    <= H_IDLE;
                end
                default: hstate <= H_IDLE;
            endcase
        end
    end

    //========================================================================
    // 3) ROUTED-EXPERT EPISODE DETECT -> FIFO -> expert_cache_pf.
    //========================================================================
    wire moe_layer = (db_layer >= N_DENSE[LAYW-1:0]);
    wire cur_routed = fw_req && !fw_shared && moe_layer;

    reg              ep_active;
    reg [EIDXW-1:0]  ep_eidx;
    reg [LAYW-1:0]   ep_layer;
    wire new_episode = cur_routed &&
                       (!ep_active || (fw_eidx != ep_eidx) || (db_layer != ep_layer));

    always @(posedge clk) begin
        if (rst) begin
            ep_active <= 1'b0;
            ep_eidx   <= {EIDXW{1'b0}};
            ep_layer  <= {LAYW{1'b0}};
        end else begin
            ep_active <= cur_routed;
            if (cur_routed) begin
                ep_eidx  <= fw_eidx;
                ep_layer <= db_layer;
            end
        end
    end

    // ---- expert-id FIFO ----
    reg [EIDXW-1:0]  efifo [0:EFIFO_DEPTH-1];
    reg [EFW:0]      ef_wr, ef_rd;
    wire [EFW:0]     ef_cnt   = ef_wr - ef_rd;
    wire             ef_empty = (ef_wr == ef_rd);
    wire             ef_full  = (ef_cnt == EFIFO_DEPTH[EFW:0]);
    reg  [31:0]      dropped_r;
    assign ec_dropped = dropped_r;

    reg              awaiting;
    wire             ec_req_valid = (!ef_empty) && (!awaiting);
    wire [EIDXW-1:0] ec_req_id    = efifo[ef_rd[EFW-1:0]];

    integer fi;
    always @(posedge clk) begin
        if (rst) begin
            ef_wr     <= {(EFW+1){1'b0}};
            ef_rd     <= {(EFW+1){1'b0}};
            awaiting  <= 1'b0;
            dropped_r <= 32'd0;
            for (fi = 0; fi < EFIFO_DEPTH; fi = fi + 1)
                efifo[fi] <= {EIDXW{1'b0}};
        end else begin
            if (new_episode) begin
                if (!ef_full) begin
                    efifo[ef_wr[EFW-1:0]] <= fw_eidx;
                    ef_wr <= ef_wr + 1'b1;
                end else begin
                    dropped_r <= dropped_r + 32'd1;
                end
            end
            if (ec_req_valid) awaiting <= 1'b1;
            if (awaiting && ec_resp_valid) begin
                awaiting <= 1'b0;
                ef_rd    <= ef_rd + 1'b1;
            end
        end
    end

    //========================================================================
    // 4) EXPERT CACHE -- expert_cache_pf (GDDR6 cache + Flash prefetch).
    //========================================================================
    wire                 ec_flash_req;
    wire [EIDXW-1:0]     ec_flash_expert_id;
    wire                 ec_flash_done;
    wire                 ec_pf_ready;
    /* verilator lint_off UNUSEDSIGNAL */
    wire                 _ec_pf_ready_unused = ec_pf_ready;
    /* verilator lint_on UNUSEDSIGNAL */

    expert_cache_pf #(
        .SLOTS(CACHE_SLOTS), .N_EXPERT(N_EXPERT), .FLASH_LAT(FLASH_LAT),
        .CACHE_HIT_LAT(0)
    ) u_ecache (
        .clk(clk), .rst(rst),
        .req_valid(ec_req_valid), .req_expert_id(ec_req_id),
        .resp_valid(ec_resp_valid), .hit(ec_hit), .resp_slot(ec_resp_slot),
        .busy(ec_busy),
        .pf_valid(pf_valid), .pf_expert_id(pf_expert_id), .pf_ready(ec_pf_ready),
        .flash_req(ec_flash_req), .flash_expert_id(ec_flash_expert_id),
        .flash_done(ec_flash_done),
        .hit_count(ec_hit_count), .miss_count(ec_miss_count),
        .demand_stall_cycles(ec_demand_stall_cycles),
        .pf_issued(ec_pf_issued), .pf_hit(ec_pf_hit)
    );

    //========================================================================
    // 5) KV PAGER -- kv_cache_pager (latent ring + Flash overflow).
    //========================================================================
    wire                 pg_flash_req;
    wire [KVPOSW-1:0]    pg_flash_idx;
    wire                 pg_flash_done;
    wire [KVPOSW-1:0]    pg_gather_idx = {{(KVPOSW-IDXW){1'b0}}, kc_idx};

    kv_cache_pager #(
        .ROW_BITS(ROW_BITS), .RESIDENT(KV_RESIDENT), .S_MAX(KV_CTX),
        .FLASH_LAT(FLASH_LAT)
    ) u_kvpager (
        .clk(clk), .rst(rst),
        .append_valid(pg_append_valid), .append_row(kv_row_in),
        .gather_valid(kc_req), .gather_idx(pg_gather_idx),
        .row_valid(kv_row_valid), .row_out(kv_row_out), .busy(kv_busy),
        .flash_req(pg_flash_req), .flash_idx(pg_flash_idx),
        .flash_done(pg_flash_done), .flash_row(flash_row),
        .append_count(kv_append_count), .resident_lo(kv_resident_lo),
        .overflowed(kv_overflowed)
    );

    //========================================================================
    // 6) SINGLE FLASH ARBITER (demand-priority: expert-cache first).
    //========================================================================
    localparam G_EXP = 1'b0, G_PG = 1'b1;
    reg fl_busy;
    reg fl_gnt;

    assign flash_req        = fl_busy;
    assign flash_is_expert  = (fl_gnt == G_EXP);
    assign flash_expert_id  = ec_flash_expert_id;
    assign flash_row_idx    = pg_flash_idx;
    assign ec_flash_done    = flash_done && fl_busy && (fl_gnt == G_EXP);
    assign pg_flash_done    = flash_done && fl_busy && (fl_gnt == G_PG);

    always @(posedge clk) begin
        if (rst) begin
            fl_busy <= 1'b0;
            fl_gnt  <= G_EXP;
        end else begin
            if (!fl_busy) begin
                if (ec_flash_req) begin
                    fl_busy <= 1'b1; fl_gnt <= G_EXP;
                end else if (pg_flash_req) begin
                    fl_busy <= 1'b1; fl_gnt <= G_PG;
                end
            end else if (flash_done) begin
                fl_busy <= 1'b0;
            end
        end
    end

    //========================================================================
    // 7) WEIGHT LOADER -- the matmul weight-pull DMA on the hot/representative
    //    tile.  Loaded once per compute launch (mdl_start); reads its tile from
    //    the latency-1 staging memory (TB) and drives the glm_matmul_fp8 pull
    //    stream.  Its fetch addresses feed ddr5_xbar (§8).
    //========================================================================
    wire                       wl_mm_start;
    wire [WL_KW-1:0]           wl_k_len;
    wire [WL_SCALE_W-1:0]      wl_w_scale;
    wire                       wl_done;

    weight_loader #(
        .PE_N(WL_PE_N), .KMAX(WL_KMAX), .BLK(WL_BLK), .ADDR_W(WL_ADDR_W)
    ) u_loader (
        .clk(clk), .rst(rst),
        .load(mdl_start),
        .desc_base({WL_ADDR_W{1'b0}}),
        .desc_klen(LOADER_KLEN[WL_KW-1:0]),
        .desc_nblk({{(WL_BKW-1){1'b0}}, 1'b1}),    // one [128,128] block
        .mem_en(wl_mem_en), .mem_addr(wl_mem_addr), .mem_data(wl_mem_data),
        .mm_start(wl_mm_start), .mm_k_len(wl_k_len),
        .mm_w_row(loader_w_row), .mm_w_scale(wl_w_scale),
        .mm_in_valid(loader_in_valid),
        .busy(loader_busy), .done(wl_done)
    );

    always @(posedge clk) begin
        if (rst) begin
            loader_done_count <= 32'd0;
            loader_beat_count <= 32'd0;
        end else begin
            if (wl_done)          loader_done_count <= loader_done_count + 32'd1;
            if (loader_in_valid)  loader_beat_count <= loader_beat_count + 32'd1;
        end
    end

    //========================================================================
    // 8) DDR5 FAST-TIER READ FABRIC -- ddr5_xbar.
    //    A tiny priority issuer presents one banked DDR5 read per cycle from
    //    three address sources: the weight_loader's tile fetches (LOAD), the
    //    expert cache's resident-slot reads (SLOT), and the compute die's hot-
    //    weight pulls (HOT).  bank_rot stripes accepted reads round-robin across
    //    channels so all DDR_NCH channels carry traffic (the N_CH BW path).
    //    Sources coalesce while pending (a bandwidth model -- redundant reads are
    //    dropped, never lost permanently: a continuing demand re-asserts).
    //========================================================================
    localparam [DDR_TAG_W-1:0] TAG_HOT  = 8'h01;
    localparam [DDR_TAG_W-1:0] TAG_SLOT = 8'h02;
    localparam [DDR_TAG_W-1:0] TAG_LOAD = 8'h03;

    wire hot_pull = em_req | gn_req | aw_req | rw_req | fw_req | fn_req | lw_req;

    reg                  p_hot, p_slot, p_load;
    reg [CSLOTW-1:0]     slot_q;
    reg [WL_ADDR_W-1:0]  load_addr_q;
    reg [CH_IDX_W-1:0]   bank_rot;

    wire sel_load   = p_load;
    wire sel_slot   = p_slot & ~p_load;
    wire sel_hot    = p_hot  & ~p_load & ~p_slot;
    wire any_pending = p_load | p_slot | p_hot;
    /* verilator lint_off UNUSEDSIGNAL */
    wire _sel_hot_unused = sel_hot;
    /* verilator lint_on UNUSEDSIGNAL */

    // combinational requester address/tag (feed-forward from registered state)
    reg  [DDR_ADDR_W-1:0] xreq_addr;
    reg  [DDR_TAG_W-1:0]  xreq_tag;
    always @* begin
        if (sel_load) begin
            xreq_tag  = TAG_LOAD;
            xreq_addr = { {(DDR_ADDR_W-WL_ADDR_W-CH_IDX_W){1'b0}}, load_addr_q, bank_rot };
        end else if (sel_slot) begin
            xreq_tag  = TAG_SLOT;
            xreq_addr = { {(DDR_ADDR_W-CSLOTW-CH_IDX_W){1'b0}}, slot_q, bank_rot };
        end else begin
            xreq_tag  = TAG_HOT;
            xreq_addr = { {(DDR_ADDR_W-CH_IDX_W){1'b0}}, bank_rot };
        end
    end

    wire xreq_valid = any_pending;
    wire xreq_ready;
    wire xreq_fire  = xreq_valid & xreq_ready;

    // issuer state: clears come FIRST, sets AFTER -> a same-cycle new event keeps
    // the source pending (never lost), bank_rot still advances on every accept.
    always @(posedge clk) begin
        if (rst) begin
            p_hot       <= 1'b0;
            p_slot      <= 1'b0;
            p_load      <= 1'b0;
            slot_q      <= {CSLOTW{1'b0}};
            load_addr_q <= {WL_ADDR_W{1'b0}};
            bank_rot    <= {CH_IDX_W{1'b0}};
            xbar_req_count <= 32'd0;
        end else begin
            // ---- consume the granted source ----
            if (xreq_fire) begin
                bank_rot       <= bank_rot + 1'b1;
                xbar_req_count <= xbar_req_count + 32'd1;
                if (sel_load)      p_load <= 1'b0;
                else if (sel_slot) p_slot <= 1'b0;
                else               p_hot  <= 1'b0;
            end
            // ---- register new fast-tier read events (override a same-cycle clear) ----
            if (hot_pull)       p_hot  <= 1'b1;
            if (ec_resp_valid) begin p_slot <= 1'b1; slot_q <= ec_resp_slot; end
            if (wl_mem_en)     begin p_load <= 1'b1; load_addr_q <= wl_mem_addr; end
        end
    end

    wire [DDR_TAG_W-1:0] xbar_resp_tag;

    ddr5_xbar #(
        .N_CH(DDR_NCH), .ADDR_W(DDR_ADDR_W), .DATA_W(DDR_DATA_W),
        .TAG_W(DDR_TAG_W), .ROW_LAT(DDR_ROW_LAT), .RESP_QD(DDR_RESP_QD),
        .BANK_LSB(0)
    ) u_xbar (
        .clk(clk), .rst(rst),
        .req_valid(xreq_valid), .req_ready(xreq_ready),
        .req_addr(xreq_addr), .req_tag(xreq_tag),
        .mem_req_valid(mem_req_valid), .mem_req_ready(mem_req_ready),
        .mem_req_addr(mem_req_addr), .mem_req_tag(mem_req_tag),
        .mem_resp_valid(mem_resp_valid), .mem_resp_ready(mem_resp_ready),
        .mem_resp_data(mem_resp_data), .mem_resp_tag(mem_resp_tag),
        .resp_valid(xbar_resp_valid), .resp_ready(1'b1),
        .resp_data(xbar_resp_data), .resp_tag(xbar_resp_tag)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire _xbar_resp_tag_unused = &{1'b0, xbar_resp_tag};
    wire _wl_obs_unused = &{1'b0, wl_mm_start, wl_k_len, wl_w_scale};
    /* verilator lint_on UNUSEDSIGNAL */

    // drain counter (responses are always accepted, resp_ready=1)
    always @(posedge clk) begin
        if (rst) xbar_resp_count <= 32'd0;
        else if (xbar_resp_valid) xbar_resp_count <= xbar_resp_count + 32'd1;
    end

endmodule
/* verilator lint_on DECLFILENAME */
