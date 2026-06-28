`timescale 1ns/1ps
`include "glm_fp.vh"
`include "glm_fp_pipe_lat.vh"
/* verilator lint_off DECLFILENAME */
//============================================================================
// glm_fp8_soc.v  --  TOP-LEVEL GLM-5.2-FP8 SINGLE-PACKAGE SoC
//                    (docs/SYSTEM_SINGLE_PACKAGE.md  --  the integrated top)
//----------------------------------------------------------------------------
// WHAT THIS IS
//   The top-level System-on-Chip that wires the VERIFIED FP8 COMPUTE die
//   (glm_model_fp8 -- the full GLM-5.2-FP8 forward pass for one token) to the
//   VERIFIED MEMORY-SYSTEM controllers (expert_cache_pf -- the MoE routed-
//   expert GDDR6 cache + Flash prefetch; kv_cache_pager -- the latent-KV ring
//   cache + Flash overflow), so a single token can be generated with the FULL
//   memory hierarchy in the loop.  This is WIRING, not new math: no arithmetic
//   is reimplemented here.
//
// BLOCK DIAGRAM  (one decode step)
//
//      HOST (USB-C bridge)                                   GDDR6/Flash STUBS
//      start/prompt_tok ─┐                                   (TB-driven, the
//      start_pos/s_len   │                                    PHY would serve)
//                        ▼
//            ┌────────────────────────── glm_fp8_soc ──────────────────────┐
//            │  ┌──────────────┐  em_/gn_/aw_/rw_/fn_/lw_  (HOT weights) ── │ ◀── GDDR6 stub
//            │  │ glm_model_fp8│  kc_  (latent KV read) ───┐                │
//            │  │  FP8 COMPUTE │  fw_  (FFN expert pull) ──┐│                │
//            │  └──────┬───────┘                          ││                │
//            │   db_layer/fw_eidx (router pick)           ││ kc_idx         │
//            │         │ routed-expert episode detect      ││               │
//            │         ▼                                   ▼▼               │
//            │   ┌────────────┐  req/slot/hit/miss   ┌──────────────┐       │
//            │   │ EXPERT-ISSUE│────────────────────▶│expert_cache_pf│      │
//            │   │  FIFO + FSM │◀───resp_valid───────│ (GDDR6 cache) │      │
//            │   └────────────┘                      └──────┬───────┘       │
//            │                                  gather/append│ flash_req     │
//            │                                  ┌────────────┴───┐           │
//            │                                  │ kv_cache_pager │ flash_req │
//            │                                  │  (KV ring)     │           │
//            │                                  └───────┬────────┘           │
//            │                          ┌───────────────┴────────┐          │
//            │                          │  SINGLE FLASH ARBITER  │──────────│ ◀──▶ Flash stub
//            │                          │   (demand-priority)    │ flash_*  │
//            │                          └────────────────────────┘          │
//            └─────────────────────────────────────────────────────────────┘
//                        busy / done / next_tok / tok_valid  ▶ HOST
//
// HOW THE CONTROLLERS SIT IN THE DATAPATH  (observable-but-pass-through)
//   The compute die's weight/KV PULLS are COMBINATIONAL and answered the SAME
//   cycle (the verified handshake of glm_model_fp8) -- the die cannot be stalled
//   mid-layer.  So the HOT weights and the cache/pager BACKING (GDDR6 + Flash
//   contents) flow to the die from the STUB ports the TB drives (exactly the
//   bytes a real GDDR6/Flash PHY would serve), and the two controllers sit IN
//   THE DATAPATH as the address/decision engines:
//     * expert_cache_pf SEES every router-selected ROUTED expert (db_layer >=
//       N_DENSE, fw_shared==0): the SoC presents the expert id, the cache
//       answers HIT (resident GDDR6 slot) or MISS (background Flash fetch into
//       an LRU-victim slot), advancing hit/miss/stall counters and returning the
//       slot the GDDR6 read would use.  The episode->cache requests are decoupled
//       by a small FIFO so the fast die never waits on the slower cache.
//     * kv_cache_pager OWNS the latent-KV window: the SoC APPENDS one latent row
//       per prompt/decode token and GATHERS the model's DSA-selected rows
//       (gather_valid=kc_req, gather_idx=kc_idx), serving resident rows in 1
//       cycle and Flash-overflow rows over the shared channel.
//   This is the explicitly-accepted integration: the controllers are elaborated,
//   simulatable, in the datapath, and report hit/miss/stall while the verified
//   compute is unperturbed.  Wiring the cache's slot back through the GDDR6 read
//   address (so weights physically come FROM the slot) is the remaining step a
//   real PHY closes; the slot is exposed (ec_resp_slot) for exactly that.
//
//   SINGLE FLASH CHANNEL, DEMAND-PRIORITY ARBITER: expert_cache_pf and
//   kv_cache_pager share ONE Flash port (the module has one Flash bus).  The
//   arbiter grants the idle channel to a requester, EXPERT-cache first, and holds
//   the grant until that client's flash_done; the TB models Flash latency + data.
//
// STYLE: synchronous ACTIVE-HIGH reset; NO latch; NO combinational loop;
//   header; X-aware friendly (every reg is reset).  All weight/KV delivery is
//   via combinational PULL answered the same cycle by the system/TB stubs.
//============================================================================
module glm_fp8_soc #(
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
    parameter integer EFW        = (EFIFO_DEPTH <= 1) ? 1 : $clog2(EFIFO_DEPTH)
)(
    input  wire                          clk,
    input  wire                          rst,        // sync, active-high

    //========================== HOST interface (USB-C bridge) ===============
    input  wire                          start,      // 1-cycle pulse: begin a token
    input  wire [TOKW-1:0]               prompt_tok, // input token to embed
    input  wire [POSW-1:0]               start_pos,  // query position (RoPE)
    input  wire [IDXW:0]                 s_len,      // S causal keys (<= S_MAX)
    output reg                           busy,       // SoC busy (prefill+compute+append)
    output reg                           done,       // 1-cycle pulse: token committed
    output reg  [TOKW-1:0]               next_tok,   // committed next token (argmax)
    output reg                           tok_valid,  // 1-cycle pulse with next_tok
    output wire [VOCAB*16-1:0]           logits,     // bf16 logit vector (observability)

    //========================== GDDR6 HOT-weight STUBS ======================
    //  request lines OUT (address the GDDR6 ROM), data IN (the PHY serves it).
    // ---- embedding (bf16) ----
    output wire                          em_req,
    output wire [TOKW-1:0]               em_tok,
    output wire [DIMW-1:0]               em_idx,
    input  wire [15:0]                   em_val,
    // ---- per-layer annotation (DSA IndexShare schedule) ----
    output wire [LAYW-1:0]               db_layer,
    output wire                          idx_fresh,
    output wire [LAYW-1:0]               idx_win,
    // ---- per-layer RMSNorm gamma (bf16) ----
    output wire                          gn_req,
    output wire                          gn_which,
    output wire [DIMW-1:0]               gn_idx,
    input  wire [15:0]                   gn_val,
    // ---- attention weights (FP8 E4M3 + [128,128] block scales) ----
    output wire                          aw_req,
    output wire [3:0]                    aw_sel,
    output wire [A_GRPW-1:0]             aw_grp,
    output wire [A_KCW-1:0]              aw_k,
    input  wire [PE_N*8-1:0]             aw_col,
    input  wire [16*PE_N*A_NB-1:0]       aw_scale,
    // ---- MoE router W_g (FP8 + scales) ----
    output wire                          rw_req,
    output wire [R_KW-1:0]               rw_k,
    input  wire [8*N_EXPERT-1:0]         rw_col,
    input  wire [16*N_EXPERT*R_NB-1:0]   rw_scale,
    // ---- FFN expert weights (FP8 + scales) -- routed pass through the cache ----
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
    // ---- final RMSNorm gamma (bf16) ----
    output wire                          fn_req,
    output wire [DIMW-1:0]               fn_idx,
    input  wire [15:0]                   fn_val,
    // ---- LM-head weights (bf16) ----
    output wire                          lw_req,
    output wire [VTW-1:0]                lw_vtile,
    output wire [DIMW-1:0]               lw_k,
    input  wire [LM_TN*16-1:0]           lw_col,
    // ---- latent-KV read DATA STUB (the GDDR6/Flash bytes for kc_*; the pager
    //      observes the access -- see header).  Driven by the TB the same cycle. ----
    input  wire [KV_LORA*16-1:0]         kc_ckv,
    input  wire [ROPE*16-1:0]            kc_krope,
    output wire                          kc_req,
    output wire [IDXW-1:0]               kc_idx,

    //========================== KV append (latent ROW source) ===============
    //  The SoC APPENDS one latent row per token; the ROW BYTES come from this
    //  stub (the computed latent a real datapath would write back).  kv_row_sel
    //  is the logical position being appended (prompt prefill 0..s_len-1, then
    //  the decode token at s_len).
    output wire [KVPOSW-1:0]             kv_row_sel,
    input  wire [ROW_BITS-1:0]           kv_row_in,

    //========================== SINGLE FLASH CHANNEL (to PHY/TB) ============
    output wire                          flash_req,       // a Flash fetch is in flight
    output wire                          flash_is_expert, // 1=expert-cache, 0=KV-pager
    output wire [EIDXW-1:0]              flash_expert_id, // expert id (when is_expert)
    output wire [KVPOSW-1:0]             flash_row_idx,   // cold KV row (when !is_expert)
    input  wire                          flash_done,      // 1-cycle: fetch complete
    input  wire [ROW_BITS-1:0]           flash_row,       // cold KV row payload

    //========================== expert prefetch hint (optional) =============
    input  wire                          pf_valid,
    input  wire [EIDXW-1:0]              pf_expert_id,

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
    output wire [31:0]                   ec_dropped     // FIFO overflow guard (should stay 0)
);
    //========================================================================
    // 1) THE COMPUTE DIE -- glm_model_fp8 (verified full FP8 forward pass).
    //    Its HOT-weight pulls flow straight to the SoC's GDDR6 stub ports; its
    //    kc_* read data comes from the kc_ckv/kc_krope stub (the pager observes
    //    the access).  kc_valid is the 1-cycle-registered ack of kc_req (the
    //    contract every verified TB uses).
    //========================================================================
    wire                      mdl_done;
    wire [TOKW-1:0]           mdl_argmax;
    reg                       kc_valid_r;     // registered kc_req ack to the die
    reg                       mdl_start;      // 1-cycle compute-die launch pulse

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
    // 2) HOST FSM : prefill the KV window (append s_len prompt latents) -> run
    //    the compute die -> append the new decode token's latent -> commit.
    //========================================================================
    localparam [2:0] H_IDLE   = 3'd0,
                     H_APPEND = 3'd1,   // append prompt latents 0..s_len-1
                     H_RUN_W  = 3'd3,   // wait model done (mdl_start pulsed on entry)
                     H_DECAP  = 3'd4,   // append the decode token's latent
                     H_DONE   = 3'd5;
    reg [2:0]        hstate;
    reg [IDXW:0]     ap_i;              // prefill append index

    // append control (combinational from FSM state)
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
                            mdl_start <= 1'b1;          // empty prompt -> launch compute now
                            hstate    <= H_RUN_W;
                        end else
                            hstate <= H_APPEND;
                    end
                end
                H_APPEND: begin
                    // pg_append_valid is high this cycle -> the pager writes row ap_i.
                    if (ap_i == (s_len - 1'b1)) begin
                        mdl_start <= 1'b1;             // last append beat -> launch compute now
                        hstate    <= H_RUN_W;
                    end
                    ap_i <= ap_i + 1'b1;
                end
                H_RUN_W: begin
                    if (mdl_done) begin
                        next_tok <= mdl_argmax;         // commit the sampled token
                        hstate   <= H_DECAP;
                    end
                end
                H_DECAP: begin
                    // append the decode token's latent (position s_len) -- "append
                    // new rows per token".  pg_append_valid is high this cycle.
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
    //    A routed-expert episode = the model pulling FFN weights in a MoE layer
    //    (db_layer >= N_DENSE) for a NON-shared expert (fw_shared==0).  Each
    //    distinct (layer, fw_eidx) run is one cache request, pushed into a small
    //    FIFO so the fast die never waits on the slower cache.
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

    // ---- expert-id FIFO (depth EFIFO_DEPTH) ----
    reg [EIDXW-1:0]  efifo [0:EFIFO_DEPTH-1];
    reg [EFW:0]      ef_wr, ef_rd;          // extra bit for full/empty disambiguation
    wire [EFW:0]     ef_cnt   = ef_wr - ef_rd;
    wire             ef_empty = (ef_wr == ef_rd);
    wire             ef_full  = (ef_cnt == EFIFO_DEPTH[EFW:0]);
    reg  [31:0]      dropped_r;
    assign ec_dropped = dropped_r;

    // ---- issue handshake to the cache ----
    reg              awaiting;             // a cache request is outstanding
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
            // push a new episode
            if (new_episode) begin
                if (!ef_full) begin
                    efifo[ef_wr[EFW-1:0]] <= fw_eidx;
                    ef_wr <= ef_wr + 1'b1;
                end else begin
                    dropped_r <= dropped_r + 32'd1;   // guard (should stay 0)
                end
            end
            // issue / complete
            if (ec_req_valid) awaiting <= 1'b1;        // we present the head this cycle
            if (awaiting && ec_resp_valid) begin
                awaiting <= 1'b0;
                ef_rd    <= ef_rd + 1'b1;              // pop on response
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
    //    Append driven by the host FSM; gather driven by the die's kc_* read.
    //========================================================================
    wire                 pg_flash_req;
    wire [KVPOSW-1:0]    pg_flash_idx;
    wire                 pg_flash_done;
    // zero-extend the die's KV read index (IDXW) to the pager's logical pos (KVPOSW)
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
    //    One Flash channel shared by the two controllers; the grant is held
    //    until the granted client's flash_done.  The TB models Flash latency.
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
                    fl_busy <= 1'b1; fl_gnt <= G_EXP;   // expert-cache priority
                end else if (pg_flash_req) begin
                    fl_busy <= 1'b1; fl_gnt <= G_PG;
                end
            end else if (flash_done) begin
                fl_busy <= 1'b0;
            end
        end
    end

endmodule
/* verilator lint_on DECLFILENAME */
