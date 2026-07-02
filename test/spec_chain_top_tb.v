`timescale 1ns/1ps
//============================================================================
// spec_chain_top_tb.v -- BINDING spec==greedy test for the K-STEP MTP-CHAIN
//                        speculative-decode top (spec_chain_top).
//----------------------------------------------------------------------------
// WHAT IT BINDS (per the B8 brief):
//
//  (1) spec == greedy  (EXACT, the safety invariant):
//      An INDEPENDENT reference (a SEPARATE PE_M=1 glm_model_fp8 + a SEPARATE
//      mtp_head_fp8, both driven token-by-token off the SAME weight ROMs) replays
//      the whole loop in software:
//        * greedy rollout gtok[0..K]  : gtok[0]=argmax(model(cur_tok@pos)),
//          gtok[j]=argmax(model(gtok[j-1]@pos))  (all rows share the pass base pos,
//          exactly as the PE_M=K+1 verify batch does);
//        * chain drafts d[0..K-1]      : minted by RECURRENTLY running the ref MTP
//          head under the DUT's documented SEED CONVENTION -- step 0's h_t is the
//          main model's POST-final-norm h_state, step k>=1's h_t is the prior
//          step's PRE-final-norm h_mtp; emb_k = embed(prev token) from the model's
//          embedding table (m_0 for k=0, else d_{k-1}); pos_k = pos+1+k;
//        * p = longest prefix with d[j]==gtok[j] (== spec_decode_seq's rule, since
//          within the accepted prefix the verify argmax of d[j-1] equals gtok[j]);
//        * commit gtok[0..p]; advance cur_tok=gtok[p], pos+=p+1.
//      Because spec_chain_top commits ONLY the verify model's OWN argmaxes
//      (truth_vec), never a raw draft, that committed stream IS the model's greedy
//      rollout for ANY K -- the drafts only gate HOW MANY tokens commit per pass.
//      We assert, beat-for-beat and X-free, that the DUT committed stream EXACTLY
//      equals this reference, for K=2 AND K=3.
//
//  (2) K_eff sanity:
//      * with NONZERO embed/weights we SEARCH (prompt x pos x weight-seed) for a
//        config whose chained drafts get accepted (accepted>0 on >=1 pass) and
//        bind the DUT accept count == the reference on that config -- proving the
//        chain CAN accept and that K_eff (reported, not fixed) is exercised;
//      * with FORCED-ZERO mtp weights the chained drafts are deterministic garbage
//        (typically all rejected), and we bind that the DUT STILL commits EXACTLY
//        the greedy stream (p=0 => 1 token/pass) -- safety holds independent of the
//        draft quality.
//
//  The model + MTP weight ROMs are the committed tiny slice (== the pem / mtp_head
//  faithful-slice TBs).  K=2 and K=3 each run in their own parameterized engine;
//  the top aggregates and prints "ALL <N> TESTS PASSED"; $fatal on any
//  spec!=greedy / miscount / X / timeout.
//============================================================================

// ---------------------------------------------------------------------------
// MDL(P) : declare + combinationally answer ONE glm_model_fp8 pull bus (prefix P,
//   e.g. m_ / v_ / r_).  Reads the shared per-layer model ROMs, keyed on P``db_layer.
// ---------------------------------------------------------------------------
`define MDL(P) \
  wire P``em_req; wire [TOKW-1:0] P``em_tok; wire [DIMW-1:0] P``em_idx; reg [15:0] P``em_val; \
  wire [LAYW-1:0] P``db_layer; wire P``idx_fresh; wire [LAYW-1:0] P``idx_win; \
  wire P``gn_req; wire P``gn_which; wire [DIMW-1:0] P``gn_idx; reg [15:0] P``gn_val; \
  wire P``aw_req; wire [3:0] P``aw_sel; wire [A_GRPW-1:0] P``aw_grp; wire [A_KCW-1:0] P``aw_k; \
  reg [PE_N*8-1:0] P``aw_col; reg [16*PE_N*A_NB-1:0] P``aw_scale; \
  wire P``kc_req; wire [IDXW-1:0] P``kc_idx; reg [KV_LORA*16-1:0] P``kc_ckv; reg [ROPE*16-1:0] P``kc_krope; reg P``kc_valid; \
  wire P``rw_req; wire [R_KW-1:0] P``rw_k; reg [8*N_EXPERT-1:0] P``rw_col; reg [16*N_EXPERT*R_NB-1:0] P``rw_scale; \
  wire P``fw_req; wire [1:0] P``fw_sel; wire [FF_GWD-1:0] P``fw_grp; wire [FF_KWD-1:0] P``fw_k; \
  wire P``fw_shared; wire [EIDXW-1:0] P``fw_eidx; \
  reg [8*TN-1:0] P``fw_col, P``fw_col_up; reg [16*TN*FF_NB_D-1:0] P``fw_scale_g, P``fw_scale_u; \
  wire P``fn_req; wire [DIMW-1:0] P``fn_idx; reg [15:0] P``fn_val; \
  wire P``lw_req; wire [VTW-1:0] P``lw_vtile; wire [DIMW-1:0] P``lw_k; reg [LM_TN*16-1:0] P``lw_col; \
  integer P``ts, P``re, P``ft, P``fo, P``cd, P``ob, P``bj; reg [15:0] P``sca; reg P``dm; \
  always @* P``em_val = EMB[P``em_tok][P``em_idx]; \
  always @* P``fn_val = GF[P``fn_idx]; \
  always @* begin P``lw_col = {LM_TN*16{1'b0}}; \
    for (P``ts=0;P``ts<LM_TN;P``ts=P``ts+1) P``lw_col[16*P``ts+:16] = Wlm[P``lw_vtile*LM_TN + P``ts][P``lw_k]; end \
  always @* P``gn_val = P``gn_which ? G2[P``db_layer][P``gn_idx] : G1[P``db_layer][P``gn_idx]; \
  always @* begin \
    P``aw_col = {PE_N*8{1'b0}}; P``aw_scale = {16*PE_N*A_NB{1'b0}}; \
    for (P``ts=0;P``ts<PE_N;P``ts=P``ts+1) case (P``aw_sel) \
    4'd0: if (P``aw_grp*PE_N+P``ts<Q_LORA)   P``aw_col[8*P``ts+:8]=W_dq [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd1: if (P``aw_grp*PE_N+P``ts<HQK)      P``aw_col[8*P``ts+:8]=W_uq [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd2: if (P``aw_grp*PE_N+P``ts<KV_LORA)  P``aw_col[8*P``ts+:8]=W_dkv[P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd3: if (P``aw_grp*PE_N+P``ts<ROPE)     P``aw_col[8*P``ts+:8]=W_kr [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd4: if (P``aw_grp*PE_N+P``ts<HNOPE)    P``aw_col[8*P``ts+:8]=W_uk [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd5: if (P``aw_grp*PE_N+P``ts<HV)       P``aw_col[8*P``ts+:8]=W_uv [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    4'd6: if (P``aw_grp*PE_N+P``ts<MODEL_DIM)P``aw_col[8*P``ts+:8]=W_o  [P``db_layer][P``aw_grp*PE_N+P``ts][P``aw_k]; \
    default: P``aw_col[8*P``ts+:8]=8'h0; endcase \
    case (P``aw_sel) 4'd0:P``sca=ScW_dq[P``db_layer]; 4'd1:P``sca=ScW_uq[P``db_layer]; 4'd2:P``sca=ScW_dkv[P``db_layer]; \
    4'd3:P``sca=ScW_kr[P``db_layer]; 4'd4:P``sca=ScW_uk[P``db_layer]; 4'd5:P``sca=ScW_uv[P``db_layer]; \
    4'd6:P``sca=ScW_o[P``db_layer]; default:P``sca=16'h3F80; endcase \
    for (P``ts=0;P``ts<PE_N;P``ts=P``ts+1) P``aw_scale[16*P``ts+:16]=P``sca; \
  end \
  always @* begin P``kc_ckv={KV_LORA*16{1'b0}}; P``kc_krope={ROPE*16{1'b0}}; \
    for (P``cd=0;P``cd<KV_LORA;P``cd=P``cd+1) P``kc_ckv[16*P``cd+:16]=CKV[P``db_layer][P``kc_idx][P``cd]; \
    for (P``cd=0;P``cd<ROPE;P``cd=P``cd+1)    P``kc_krope[16*P``cd+:16]=KRP[P``db_layer][P``kc_idx][P``cd]; end \
  always @* begin P``rw_col={8*N_EXPERT{1'b0}}; P``rw_scale={16*N_EXPERT*R_NB{1'b0}}; \
    for (P``re=0;P``re<N_EXPERT;P``re=P``re+1) begin P``rw_col[8*P``re+:8]=Wg[P``db_layer][P``rw_k][P``re]; P``rw_scale[16*P``re+:16]=ScWg[P``db_layer]; end end \
  always @* begin P``dm=(P``db_layer<N_DENSE)?1'b0:1'b1; \
    P``fw_col={8*TN{1'b0}}; P``fw_col_up={8*TN{1'b0}}; P``fw_scale_g={16*TN*FF_NB_D{1'b0}}; P``fw_scale_u={16*TN*FF_NB_D{1'b0}}; \
    P``ob=(P``fw_grp*TN)/BLK; \
    for (P``ft=0;P``ft<TN;P``ft=P``ft+1) begin P``fo=P``fw_grp*TN+P``ft; \
      if (P``dm==1'b0) begin \
        if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=Dd[P``db_layer][P``fo][P``fw_k]; end \
        else begin if (P``fo<INTER_DENSE) begin P``fw_col[8*P``ft+:8]=Dg[P``db_layer][P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=Du[P``db_layer][P``fo][P``fw_k]; end end \
      end else begin \
        if (P``fw_shared) begin \
          if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=SHd[P``db_layer][P``fo][P``fw_k]; end \
          else if (P``fo<INTER_MOE) begin P``fw_col[8*P``ft+:8]=SHg[P``db_layer][P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=SHu[P``db_layer][P``fo][P``fw_k]; end \
        end else begin \
          if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=Md[P``db_layer][P``fw_eidx][P``fo][P``fw_k]; end \
          else if (P``fo<INTER_MOE) begin P``fw_col[8*P``ft+:8]=Mg[P``db_layer][P``fw_eidx][P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=Mu[P``db_layer][P``fw_eidx][P``fo][P``fw_k]; end \
        end end end \
    for (P``ft=0;P``ft<TN;P``ft=P``ft+1) begin \
      if (P``dm==1'b0) begin \
        for (P``bj=0;P``bj<FF_NB_D;P``bj=P``bj+1) begin \
          if (P``fw_sel==2'd2) P``fw_scale_g[16*(P``bj*TN+P``ft)+:16]=ScDd[P``db_layer][P``bj]; \
          else begin P``fw_scale_g[16*(P``bj*TN+P``ft)+:16]=ScDg[P``db_layer][P``ob]; P``fw_scale_u[16*(P``bj*TN+P``ft)+:16]=ScDu[P``db_layer][P``ob]; end end \
      end else begin \
        if (P``fw_shared) begin \
          if (P``fw_sel==2'd2) P``fw_scale_g[16*(0*TN+P``ft)+:16]=ScSHd[P``db_layer]; \
          else begin P``fw_scale_g[16*(0*TN+P``ft)+:16]=ScSHg[P``db_layer]; P``fw_scale_u[16*(0*TN+P``ft)+:16]=ScSHu[P``db_layer]; end \
        end else begin \
          if (P``fw_sel==2'd2) P``fw_scale_g[16*(0*TN+P``ft)+:16]=ScMd[P``db_layer][P``fw_eidx]; \
          else begin P``fw_scale_g[16*(0*TN+P``ft)+:16]=ScMg[P``db_layer][P``fw_eidx]; P``fw_scale_u[16*(0*TN+P``ft)+:16]=ScMu[P``db_layer][P``fw_eidx]; end \
        end end end \
  end \
  always @(posedge clk) begin if (rst) P``kc_valid<=1'b0; else P``kc_valid<=P``kc_req; end

// ---------------------------------------------------------------------------
// MTP(P) : declare + combinationally answer ONE mtp_head_fp8 pull bus (prefix P,
//   e.g. t_ / rm_).  Reads the single-layer MTP ROMs (m-prefixed).  force_zero_mtp
//   zeroes every weight answer so the chained drafts become deterministic garbage.
// ---------------------------------------------------------------------------
`define MTP(P) \
  wire P``cn_req; wire [1:0] P``cn_which; wire [DIMW-1:0] P``cn_idx; reg [15:0] P``cn_val; \
  wire P``pw_req; wire [PTW-1:0] P``pw_ptile; wire [CKIW-1:0] P``pw_k; reg [PROJ_TN*8-1:0] P``pw_col; reg [16*PROJ_TN*PROJ_NB-1:0] P``pw_scale; \
  wire P``gn_req; wire P``gn_which; wire [DIMW-1:0] P``gn_idx; reg [15:0] P``gn_val; \
  wire P``aw_req; wire [3:0] P``aw_sel; wire [A_GRPW-1:0] P``aw_grp; wire [A_KCW-1:0] P``aw_k; \
  reg [PE_N*8-1:0] P``aw_col; reg [16*PE_N*A_NB-1:0] P``aw_scale; \
  wire P``kc_req; wire [IDXW-1:0] P``kc_idx; reg [KV_LORA*16-1:0] P``kc_ckv; reg [ROPE*16-1:0] P``kc_krope; reg P``kc_valid; \
  wire P``rw_req; wire [R_KW-1:0] P``rw_k; reg [8*N_EXPERT-1:0] P``rw_col; reg [16*N_EXPERT*R_NB-1:0] P``rw_scale; \
  wire P``fw_req; wire [1:0] P``fw_sel; wire [FF_GWD-1:0] P``fw_grp; wire [FF_KWD-1:0] P``fw_k; \
  wire P``fw_shared; wire [EIDXW-1:0] P``fw_eidx; \
  reg [8*TN-1:0] P``fw_col, P``fw_col_up; reg [16*TN*FF_NB_D-1:0] P``fw_scale_g, P``fw_scale_u; \
  wire P``lw_req; wire [VTW-1:0] P``lw_vtile; wire [DIMW-1:0] P``lw_k; reg [LM_TN*16-1:0] P``lw_col; \
  integer P``ts, P``re, P``ft, P``fo, P``cd, P``ob, P``pq, P``bj; reg [15:0] P``sca; \
  always @* begin if (force_zero_mtp) P``cn_val=16'h0; \
    else case (P``cn_which) 2'd0: P``cn_val=mGA[P``cn_idx]; 2'd1: P``cn_val=mGB[P``cn_idx]; default: P``cn_val=mGF[P``cn_idx]; endcase end \
  always @* begin P``pw_col={PROJ_TN*8{1'b0}}; P``pw_scale={16*PROJ_TN*PROJ_NB{1'b0}}; \
    if (!force_zero_mtp) begin \
      for (P``pq=0;P``pq<PROJ_TN;P``pq=P``pq+1) P``pw_col[8*P``pq+:8]=mWp[P``pw_ptile*PROJ_TN+P``pq][P``pw_k]; \
      for (P``cd=0;P``cd<PROJ_NB;P``cd=P``cd+1) for (P``pq=0;P``pq<PROJ_TN;P``pq=P``pq+1) P``pw_scale[16*(P``cd*PROJ_TN+P``pq)+:16]=mScWp[P``cd]; end end \
  always @* begin if (force_zero_mtp) P``gn_val=16'h0; else P``gn_val = P``gn_which ? mG2[P``gn_idx] : mG1[P``gn_idx]; end \
  always @* begin P``aw_col={PE_N*8{1'b0}}; P``aw_scale={16*PE_N*A_NB{1'b0}}; \
    if (!force_zero_mtp) begin \
      for (P``ts=0;P``ts<PE_N;P``ts=P``ts+1) case (P``aw_sel) \
      4'd0: if (P``aw_grp*PE_N+P``ts<Q_LORA)   P``aw_col[8*P``ts+:8]=mW_dq [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd1: if (P``aw_grp*PE_N+P``ts<HQK)      P``aw_col[8*P``ts+:8]=mW_uq [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd2: if (P``aw_grp*PE_N+P``ts<KV_LORA)  P``aw_col[8*P``ts+:8]=mW_dkv[P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd3: if (P``aw_grp*PE_N+P``ts<ROPE)     P``aw_col[8*P``ts+:8]=mW_kr [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd4: if (P``aw_grp*PE_N+P``ts<HNOPE)    P``aw_col[8*P``ts+:8]=mW_uk [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd5: if (P``aw_grp*PE_N+P``ts<HV)       P``aw_col[8*P``ts+:8]=mW_uv [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      4'd6: if (P``aw_grp*PE_N+P``ts<MODEL_DIM)P``aw_col[8*P``ts+:8]=mW_o  [P``aw_grp*PE_N+P``ts][P``aw_k]; \
      default: P``aw_col[8*P``ts+:8]=8'h0; endcase \
      case (P``aw_sel) 4'd0:P``sca=mScW_dq; 4'd1:P``sca=mScW_uq; 4'd2:P``sca=mScW_dkv; \
      4'd3:P``sca=mScW_kr; 4'd4:P``sca=mScW_uk; 4'd5:P``sca=mScW_uv; \
      4'd6:P``sca=mScW_o; default:P``sca=16'h3F80; endcase \
      for (P``ts=0;P``ts<PE_N;P``ts=P``ts+1) P``aw_scale[16*P``ts+:16]=P``sca; end end \
  always @* begin P``kc_ckv={KV_LORA*16{1'b0}}; P``kc_krope={ROPE*16{1'b0}}; \
    if (!force_zero_mtp) begin \
      for (P``cd=0;P``cd<KV_LORA;P``cd=P``cd+1) P``kc_ckv[16*P``cd+:16]=mCKV[P``kc_idx][P``cd]; \
      for (P``cd=0;P``cd<ROPE;P``cd=P``cd+1)    P``kc_krope[16*P``cd+:16]=mKRP[P``kc_idx][P``cd]; end end \
  always @* begin P``rw_col={8*N_EXPERT{1'b0}}; P``rw_scale={16*N_EXPERT*R_NB{1'b0}}; \
    if (!force_zero_mtp) for (P``re=0;P``re<N_EXPERT;P``re=P``re+1) begin P``rw_col[8*P``re+:8]=mWg[P``rw_k][P``re]; P``rw_scale[16*P``re+:16]=mScWg; end end \
  always @* begin \
    P``fw_col={8*TN{1'b0}}; P``fw_col_up={8*TN{1'b0}}; P``fw_scale_g={16*TN*FF_NB_D{1'b0}}; P``fw_scale_u={16*TN*FF_NB_D{1'b0}}; \
    P``ob=(P``fw_grp*TN)/BLK; \
    if (!force_zero_mtp) begin \
      for (P``ft=0;P``ft<TN;P``ft=P``ft+1) begin P``fo=P``fw_grp*TN+P``ft; \
        if (cur_mode==1'b0) begin \
          if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=mDd[P``fo][P``fw_k]; end \
          else begin if (P``fo<INTER_DENSE) begin P``fw_col[8*P``ft+:8]=mDg[P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=mDu[P``fo][P``fw_k]; end end \
        end else begin \
          if (P``fw_shared) begin \
            if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=mSHd[P``fo][P``fw_k]; end \
            else if (P``fo<INTER_MOE) begin P``fw_col[8*P``ft+:8]=mSHg[P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=mSHu[P``fo][P``fw_k]; end \
          end else begin \
            if (P``fw_sel==2'd2) begin if (P``fo<MODEL_DIM) P``fw_col[8*P``ft+:8]=mMd[P``fw_eidx][P``fo][P``fw_k]; end \
            else if (P``fo<INTER_MOE) begin P``fw_col[8*P``ft+:8]=mMg[P``fw_eidx][P``fo][P``fw_k]; P``fw_col_up[8*P``ft+:8]=mMu[P``fw_eidx][P``fo][P``fw_k]; end \
          end end end \
      for (P``ft=0;P``ft<TN;P``ft=P``ft+1) begin \
        if (cur_mode==1'b0) begin \
          for (P``bj=0;P``bj<FF_NB_D;P``bj=P``bj+1) begin \
            if (P``fw_sel==2'd2) P``fw_scale_g[16*(P``bj*TN+P``ft)+:16]=mScDd[P``bj]; \
            else begin P``fw_scale_g[16*(P``bj*TN+P``ft)+:16]=mScDg[P``ob]; P``fw_scale_u[16*(P``bj*TN+P``ft)+:16]=mScDu[P``ob]; end end \
        end else begin \
          if (P``fw_shared) begin \
            if (P``fw_sel==2'd2) P``fw_scale_g[16*(0*TN+P``ft)+:16]=mScSHd; \
            else begin P``fw_scale_g[16*(0*TN+P``ft)+:16]=mScSHg; P``fw_scale_u[16*(0*TN+P``ft)+:16]=mScSHu; end \
          end else begin \
            if (P``fw_sel==2'd2) P``fw_scale_g[16*(0*TN+P``ft)+:16]=mScMd[P``fw_eidx]; \
            else begin P``fw_scale_g[16*(0*TN+P``ft)+:16]=mScMg[P``fw_eidx]; P``fw_scale_u[16*(0*TN+P``ft)+:16]=mScMu[P``fw_eidx]; end \
          end end end end end \
  always @* begin P``lw_col = {LM_TN*16{1'b0}}; \
    if (!force_zero_mtp) for (P``pq=0;P``pq<LM_TN;P``pq=P``pq+1) P``lw_col[16*P``pq+:16] = mWlm[P``lw_vtile*LM_TN + P``pq][P``lw_k]; end \
  always @(posedge clk) begin if (rst) P``kc_valid<=1'b0; else P``kc_valid<=P``kc_req; end

//============================================================================
//  sct_engine : one full DRAFT_K-parameterized binding harness (self-checking)
//============================================================================
module sct_engine #(
    parameter integer DRAFT_K = 2
)(
    input  wire        clk,
    input  wire        en,          // gate: start this engine's sequence (serialize)
    output reg  [31:0] tests_out,
    output reg  [31:0] errors_out,
    output reg         finished
);
    reg rst;

    // ================= tiny faithful slice (== glm_model_fp8_pem_tb) =================
    // Tiny slice chosen to keep the FULL chain (main + K mtp + PE_M=K+1 verify)
    // simulable in iverilog: VOCAB / N_EXPERT / INTER_DENSE are reduced vs the
    // pem TB (INTER_DENSE=32 => FF_NB_D=1; the responders loop over FF_NB_D so
    // they stay correct for FF_NB_D 1 or 2).  All ratios still valid.
    localparam integer MODEL_DIM  = 8;
    localparam integer L          = 1;
    localparam integer N_DENSE    = 1;
    localparam integer VOCAB      = 8;
    localparam integer H_HEADS    = 2;
    localparam integer NOPE       = 2;
    localparam integer ROPE       = 2;
    localparam integer V_DIM      = 2;
    localparam integer Q_LORA     = 4;
    localparam integer KV_LORA    = 4;
    localparam integer S_MAX      = 2;
    localparam integer TOPK_ATTN  = 2;
    localparam integer THETA      = 8000000;
    localparam integer PE_N       = 2;
    localparam integer POSW       = 20;
    localparam integer N_EXPERT   = 2;
    localparam integer TOPK       = 2;
    localparam integer INTER_MOE  = 4;
    localparam integer INTER_DENSE= 16;
    localparam [31:0]  RSCALE     = 32'h40200000;
    localparam integer TN         = 2;
    localparam integer BLK        = 128;
    localparam integer LM_TN      = 2;
    localparam integer PROJ_TN    = 2;
    localparam integer B          = DRAFT_K + 1;  // PE_M verify rows
    localparam integer NP         = 2;            // # passes per case (multi-pass)

    // ---- derived ----
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
    localparam integer DKW    = (DRAFT_K<=1)?1:$clog2(DRAFT_K+1);
    // MTP combine-projection derived
    localparam integer CK      = 2*MODEL_DIM;
    localparam integer CKIW    = $clog2(CK);
    localparam integer NPTILE  = MODEL_DIM/PROJ_TN;
    localparam integer PTW     = (NPTILE<=1)?1:$clog2(NPTILE);
    localparam integer PROJ_NB = (CK+BLK-1)/BLK;

    integer test_count;
    integer errors;

    // ================= per-layer MODEL WEIGHT ROMs (== spec_batched_top_tb) ======
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
    reg [7:0]  Wg [0:L-1][0:MODEL_DIM-1][0:N_EXPERT-1];
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

    // ================= single-layer MTP WEIGHT ROMs (== mtp_head_fp8_tb) =========
    reg [15:0] mGA  [0:MODEL_DIM-1];
    reg [15:0] mGB  [0:MODEL_DIM-1];
    reg [15:0] mGF  [0:MODEL_DIM-1];
    reg [15:0] mWlm [0:VOCAB-1][0:MODEL_DIM-1];
    reg [7:0]  mWp  [0:MODEL_DIM-1][0:CK-1];
    reg [15:0] mScWp[0:PROJ_NB-1];
    reg [15:0] mG1 [0:MODEL_DIM-1];
    reg [15:0] mG2 [0:MODEL_DIM-1];
    reg [15:0] mCKV [0:S_MAX-1][0:KV_LORA-1];
    reg [15:0] mKRP [0:S_MAX-1][0:ROPE-1];
    reg [7:0] mW_dq  [0:Q_LORA-1][0:MODEL_DIM-1];
    reg [7:0] mW_uq  [0:HQK-1][0:Q_LORA-1];
    reg [7:0] mW_dkv [0:KV_LORA-1][0:MODEL_DIM-1];
    reg [7:0] mW_kr  [0:ROPE-1][0:MODEL_DIM-1];
    reg [7:0] mW_uk  [0:HNOPE-1][0:KV_LORA-1];
    reg [7:0] mW_uv  [0:HV-1][0:KV_LORA-1];
    reg [7:0] mW_o   [0:MODEL_DIM-1][0:HV-1];
    reg [15:0] mScW_dq, mScW_uq, mScW_dkv, mScW_kr, mScW_uk, mScW_uv, mScW_o;
    reg [7:0]  mWg [0:MODEL_DIM-1][0:N_EXPERT-1];
    reg [15:0] mScWg;
    reg [7:0] mDg [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] mDu [0:INTER_DENSE-1][0:MODEL_DIM-1];
    reg [7:0] mDd [0:MODEL_DIM-1][0:INTER_DENSE-1];
    reg [15:0] mScDg [0:FF_NB_D-1];
    reg [15:0] mScDu [0:FF_NB_D-1];
    reg [15:0] mScDd [0:FF_NB_D-1];
    reg [7:0] mMg [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] mMu [0:N_EXPERT-1][0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] mMd [0:N_EXPERT-1][0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [7:0] mSHg [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] mSHu [0:INTER_MOE-1][0:MODEL_DIM-1];
    reg [7:0] mSHd [0:MODEL_DIM-1][0:INTER_MOE-1];
    reg [15:0] mScMg [0:N_EXPERT-1], mScMu [0:N_EXPERT-1], mScMd [0:N_EXPERT-1];
    reg [15:0] mScSHg, mScSHu, mScSHd;

    // ================= deterministic generators (== pem_tb / mtp_tb) =================
    function [15:0] gen_bf16; input integer seed; input integer band;
        reg s; reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e=8'd125+h[6:4]; else e=8'd124+h[5:4];
        m=h[12:6]; gen_bf16={s,e,m};
    end endfunction
    function [7:0] gen_e4m3; input integer seed; input integer band;
        reg s; reg [3:0] e; reg [2:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*40503);
        s=h[3];
        if (band==1) e = 4'd7 + {3'b0,h[4]}; else e = 4'd6 + {3'b0,h[4]};
        m = h[12:10]; gen_e4m3 = {s,e,m};
    end endfunction
    function [15:0] gen_scale; input integer seed;
        reg [7:0] e; reg [6:0] m; integer h; begin
        h=(seed*2654435761)^(seed<<13)^(seed*22229);
        e = 8'd122 + {7'b0,h[2]}; m = h[10:4]; gen_scale={1'b0,e,m};
    end endfunction

    integer i,j,e,GLY,sc;
    task build_stim; input integer seed0; integer band; begin
        band=0; sc=seed0;
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

    // single-layer MTP weight build (distinct seed space)
    task build_mtp_stim; input integer seed0; integer band; begin
        band=0; sc=seed0;
        for (i=0;i<MODEL_DIM;i=i+1) begin mGA[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin mGB[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin mGF[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<CK;j=j+1) begin mWp[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<PROJ_NB;i=i+1) begin mScWp[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin mG1[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) begin mG2[i]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<Q_LORA;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mW_dq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HQK;i=i+1)    for (j=0;j<Q_LORA;j=j+1)    begin mW_uq[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<KV_LORA;i=i+1)for (j=0;j<MODEL_DIM;j=j+1) begin mW_dkv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<ROPE;i=i+1)   for (j=0;j<MODEL_DIM;j=j+1) begin mW_kr[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HNOPE;i=i+1)  for (j=0;j<KV_LORA;j=j+1)   begin mW_uk[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<HV;i=i+1)     for (j=0;j<KV_LORA;j=j+1)   begin mW_uv[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1)for (j=0;j<HV;j=j+1)      begin mW_o[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        mScW_dq=gen_scale(sc); sc=sc+1;  mScW_uq=gen_scale(sc); sc=sc+1;
        mScW_dkv=gen_scale(sc); sc=sc+1; mScW_kr=gen_scale(sc); sc=sc+1;
        mScW_uk=gen_scale(sc); sc=sc+1;  mScW_uv=gen_scale(sc); sc=sc+1;
        mScW_o=gen_scale(sc); sc=sc+1;
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<KV_LORA;j=j+1) begin mCKV[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<S_MAX;i=i+1) for (j=0;j<ROPE;j=j+1)    begin mKRP[i][j]=gen_bf16(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<N_EXPERT;j=j+1) begin mWg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        mScWg=gen_scale(sc); sc=sc+1;
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mDg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_DENSE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mDu[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_DENSE;j=j+1) begin mDd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin mScDg[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin mScDu[i]=gen_scale(sc); sc=sc+1; end
        for (i=0;i<FF_NB_D;i=i+1) begin mScDd[i]=gen_scale(sc); sc=sc+1; end
        for (e=0;e<N_EXPERT;e=e+1) begin
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mMg[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mMu[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin mMd[e][i][j]=gen_e4m3(sc,band); sc=sc+1; end
            mScMg[e]=gen_scale(sc); sc=sc+1; mScMu[e]=gen_scale(sc); sc=sc+1; mScMd[e]=gen_scale(sc); sc=sc+1;
        end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mSHg[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<INTER_MOE;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mSHu[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        for (i=0;i<MODEL_DIM;i=i+1) for (j=0;j<INTER_MOE;j=j+1) begin mSHd[i][j]=gen_e4m3(sc,band); sc=sc+1; end
        mScSHg=gen_scale(sc); sc=sc+1; mScSHu=gen_scale(sc); sc=sc+1; mScSHd=gen_scale(sc); sc=sc+1;
        for (i=0;i<VOCAB;i=i+1) for (j=0;j<MODEL_DIM;j=j+1) begin mWlm[i][j]=gen_bf16(sc,band); sc=sc+1; end
    end endtask

    // ================= shared responder mode / zeroing controls =================
    reg cur_mode;         // mtp head FFN mode for THIS case (dense/moe)
    reg force_zero_mtp;   // zero all mtp weight answers (garbage-draft scenario)

    // ================= DUT : spec_chain_top =====================================
    reg                       start;
    reg  [TOKW-1:0]           prompt_tok;
    reg  [POSW-1:0]           start_pos;
    reg  [IDXW:0]             s_len;
    reg                       mtp_mode;
    reg  [15:0]               num_passes;
    wire                      busy, done;
    wire                      commit_valid; wire [TOKW-1:0] commit_tok; wire accepted;
    wire [31:0]               total_tokens, main_passes, accepts, rejects;

    // embedding pull (DUT drives em_req/em_tok ; TB answers em_vec = embed(em_tok))
    wire                      em_req; wire [TOKW-1:0] em_tok; reg [MODEL_DIM*16-1:0] em_vec;
    integer emi;
    always @* begin em_vec = {MODEL_DIM*16{1'b0}};
        for (emi=0;emi<MODEL_DIM;emi=emi+1) em_vec[16*emi+:16] = EMB[em_tok][emi]; end

    // three model pull buses (m_=main, v_=verify) + one mtp bus (t_)
    `MDL(m_)
    `MDL(v_)
    `MTP(t_)

    spec_chain_top #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PROJ_TN(PROJ_TN), .DRAFT_K(DRAFT_K)
    ) dut (
        .clk(clk), .rst(rst),
        .start(start), .prompt_tok(prompt_tok), .start_pos(start_pos),
        .s_len(s_len), .mtp_mode(mtp_mode), .num_passes(num_passes),
        .busy(busy), .done(done),
        .commit_valid(commit_valid), .commit_tok(commit_tok), .accepted(accepted),
        .total_tokens(total_tokens), .main_passes(main_passes),
        .accepts(accepts), .rejects(rejects),
        .em_req(em_req), .em_tok(em_tok), .em_vec(em_vec),
        // ---- main model ----
        .m_em_req(m_em_req), .m_em_tok(m_em_tok), .m_em_idx(m_em_idx), .m_em_val(m_em_val),
        .m_db_layer(m_db_layer), .m_idx_fresh(m_idx_fresh), .m_idx_win(m_idx_win),
        .m_gn_req(m_gn_req), .m_gn_which(m_gn_which), .m_gn_idx(m_gn_idx), .m_gn_val(m_gn_val),
        .m_aw_req(m_aw_req), .m_aw_sel(m_aw_sel), .m_aw_grp(m_aw_grp), .m_aw_k(m_aw_k),
        .m_aw_col(m_aw_col), .m_aw_scale(m_aw_scale),
        .m_kc_req(m_kc_req), .m_kc_idx(m_kc_idx), .m_kc_ckv(m_kc_ckv), .m_kc_krope(m_kc_krope), .m_kc_valid(m_kc_valid),
        .m_rw_req(m_rw_req), .m_rw_k(m_rw_k), .m_rw_col(m_rw_col), .m_rw_scale(m_rw_scale),
        .m_fw_req(m_fw_req), .m_fw_sel(m_fw_sel), .m_fw_grp(m_fw_grp), .m_fw_k(m_fw_k),
        .m_fw_shared(m_fw_shared), .m_fw_eidx(m_fw_eidx),
        .m_fw_col(m_fw_col), .m_fw_col_up(m_fw_col_up), .m_fw_scale_g(m_fw_scale_g), .m_fw_scale_u(m_fw_scale_u),
        .m_fn_req(m_fn_req), .m_fn_idx(m_fn_idx), .m_fn_val(m_fn_val),
        .m_lw_req(m_lw_req), .m_lw_vtile(m_lw_vtile), .m_lw_k(m_lw_k), .m_lw_col(m_lw_col),
        // ---- mtp head ----
        .t_cn_req(t_cn_req), .t_cn_which(t_cn_which), .t_cn_idx(t_cn_idx), .t_cn_val(t_cn_val),
        .t_pw_req(t_pw_req), .t_pw_ptile(t_pw_ptile), .t_pw_k(t_pw_k), .t_pw_col(t_pw_col), .t_pw_scale(t_pw_scale),
        .t_gn_req(t_gn_req), .t_gn_which(t_gn_which), .t_gn_idx(t_gn_idx), .t_gn_val(t_gn_val),
        .t_aw_req(t_aw_req), .t_aw_sel(t_aw_sel), .t_aw_grp(t_aw_grp), .t_aw_k(t_aw_k),
        .t_aw_col(t_aw_col), .t_aw_scale(t_aw_scale),
        .t_kc_req(t_kc_req), .t_kc_idx(t_kc_idx), .t_kc_ckv(t_kc_ckv), .t_kc_krope(t_kc_krope), .t_kc_valid(t_kc_valid),
        .t_rw_req(t_rw_req), .t_rw_k(t_rw_k), .t_rw_col(t_rw_col), .t_rw_scale(t_rw_scale),
        .t_fw_req(t_fw_req), .t_fw_sel(t_fw_sel), .t_fw_grp(t_fw_grp), .t_fw_k(t_fw_k),
        .t_fw_shared(t_fw_shared), .t_fw_eidx(t_fw_eidx),
        .t_fw_col(t_fw_col), .t_fw_col_up(t_fw_col_up), .t_fw_scale_g(t_fw_scale_g), .t_fw_scale_u(t_fw_scale_u),
        .t_lw_req(t_lw_req), .t_lw_vtile(t_lw_vtile), .t_lw_k(t_lw_k), .t_lw_col(t_lw_col),
        // ---- verify model ----
        .v_em_req(v_em_req), .v_em_tok(v_em_tok), .v_em_idx(v_em_idx), .v_em_val(v_em_val),
        .v_db_layer(v_db_layer), .v_idx_fresh(v_idx_fresh), .v_idx_win(v_idx_win),
        .v_gn_req(v_gn_req), .v_gn_which(v_gn_which), .v_gn_idx(v_gn_idx), .v_gn_val(v_gn_val),
        .v_aw_req(v_aw_req), .v_aw_sel(v_aw_sel), .v_aw_grp(v_aw_grp), .v_aw_k(v_aw_k),
        .v_aw_col(v_aw_col), .v_aw_scale(v_aw_scale),
        .v_kc_req(v_kc_req), .v_kc_idx(v_kc_idx), .v_kc_ckv(v_kc_ckv), .v_kc_krope(v_kc_krope), .v_kc_valid(v_kc_valid),
        .v_rw_req(v_rw_req), .v_rw_k(v_rw_k), .v_rw_col(v_rw_col), .v_rw_scale(v_rw_scale),
        .v_fw_req(v_fw_req), .v_fw_sel(v_fw_sel), .v_fw_grp(v_fw_grp), .v_fw_k(v_fw_k),
        .v_fw_shared(v_fw_shared), .v_fw_eidx(v_fw_eidx),
        .v_fw_col(v_fw_col), .v_fw_col_up(v_fw_col_up), .v_fw_scale_g(v_fw_scale_g), .v_fw_scale_u(v_fw_scale_u),
        .v_fn_req(v_fn_req), .v_fn_idx(v_fn_idx), .v_fn_val(v_fn_val),
        .v_lw_req(v_lw_req), .v_lw_vtile(v_lw_vtile), .v_lw_k(v_lw_k), .v_lw_col(v_lw_col)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _u = &{1'b0, busy, accepted, commit_valid, em_req,
                m_em_req, m_aw_req, m_fw_req, m_rw_req, m_gn_req, m_fn_req, m_lw_req, m_idx_fresh, m_idx_win,
                v_em_req, v_aw_req, v_fw_req, v_rw_req, v_gn_req, v_fn_req, v_lw_req, v_idx_fresh, v_idx_win,
                t_cn_req, t_pw_req, t_aw_req, t_fw_req, t_rw_req, t_gn_req, t_lw_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= committed-stream capture (X-aware) =================
    integer commit_n;
    reg [TOKW-1:0] got [0:4095];
    reg cap;
    always @(negedge clk) if (cap && commit_valid) begin
        if (^commit_tok === 1'bx) begin
            $display("FAIL K=%0d: X on commit beat %0d", DRAFT_K, commit_n); errors=errors+1;
        end
        got[commit_n] = commit_tok; commit_n = commit_n + 1;
    end

    // ================= INDEPENDENT reference : PE_M=1 model + one mtp head =======
    reg                       r_start; wire r_busy, r_done;
    reg  [TOKW-1:0]           r_tok;
    reg  [POSW-1:0]           r_pos;
    reg  [IDXW:0]             r_slen;
    wire [VOCAB*16-1:0]       r_logits; wire [TOKW-1:0] r_argmax;
    wire [MODEL_DIM*16-1:0]   r_hstate;
    `MDL(r_)
    glm_model_fp8 #(
        .MODEL_DIM(MODEL_DIM), .L(L), .N_DENSE(N_DENSE), .VOCAB(VOCAB),
        .H_HEADS(H_HEADS), .NOPE(NOPE), .ROPE(ROPE), .V_DIM(V_DIM),
        .Q_LORA(Q_LORA), .KV_LORA(KV_LORA), .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN),
        .THETA(THETA), .PE_N(PE_N), .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK),
        .INTER_MOE(INTER_MOE), .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN),
        .BLK(BLK), .LM_TN(LM_TN), .PE_M(1)
    ) ref_model (
        .clk(clk), .rst(rst), .start(r_start), .busy(r_busy), .done(r_done),
        .token_id(r_tok), .pos(r_pos), .s_len(r_slen),
        .logits(r_logits), .argmax(r_argmax),
        .em_req(r_em_req), .em_tok(r_em_tok), .em_idx(r_em_idx), .em_val(r_em_val),
        .db_layer(r_db_layer), .idx_fresh(r_idx_fresh), .idx_win(r_idx_win),
        .gn_req(r_gn_req), .gn_which(r_gn_which), .gn_idx(r_gn_idx), .gn_val(r_gn_val),
        .aw_req(r_aw_req), .aw_sel(r_aw_sel), .aw_grp(r_aw_grp), .aw_k(r_aw_k),
        .aw_col(r_aw_col), .aw_scale(r_aw_scale),
        .kc_req(r_kc_req), .kc_idx(r_kc_idx), .kc_ckv(r_kc_ckv), .kc_krope(r_kc_krope), .kc_valid(r_kc_valid),
        .rw_req(r_rw_req), .rw_k(r_rw_k), .rw_col(r_rw_col), .rw_scale(r_rw_scale),
        .fw_req(r_fw_req), .fw_sel(r_fw_sel), .fw_grp(r_fw_grp), .fw_k(r_fw_k),
        .fw_shared(r_fw_shared), .fw_eidx(r_fw_eidx),
        .fw_col(r_fw_col), .fw_col_up(r_fw_col_up), .fw_scale_g(r_fw_scale_g), .fw_scale_u(r_fw_scale_u),
        .fn_req(r_fn_req), .fn_idx(r_fn_idx), .fn_val(r_fn_val),
        .lw_req(r_lw_req), .lw_vtile(r_lw_vtile), .lw_k(r_lw_k), .lw_col(r_lw_col),
        .h_state(r_hstate)
    );

    reg                       rm_start; wire rm_busy, rm_done;
    reg                       rm_mode;
    reg  [POSW-1:0]           rm_pos;
    reg  [IDXW:0]             rm_slen;
    reg  [MODEL_DIM*16-1:0]   rm_h_t, rm_emb;
    wire [VOCAB*16-1:0]       rm_logits; wire [TOKW-1:0] rm_argmax;
    wire [MODEL_DIM*16-1:0]   rm_h_mtp;
    `MTP(rm_)
    mtp_head_fp8 #(
        .MODEL_DIM(MODEL_DIM), .VOCAB(VOCAB), .H_HEADS(H_HEADS), .NOPE(NOPE),
        .ROPE(ROPE), .V_DIM(V_DIM), .Q_LORA(Q_LORA), .KV_LORA(KV_LORA),
        .S_MAX(S_MAX), .TOPK_ATTN(TOPK_ATTN), .THETA(THETA), .PE_N(PE_N),
        .POSW(POSW), .N_EXPERT(N_EXPERT), .TOPK(TOPK), .INTER_MOE(INTER_MOE),
        .INTER_DENSE(INTER_DENSE), .RSCALE(RSCALE), .TN(TN), .BLK(BLK),
        .LM_TN(LM_TN), .PROJ_TN(PROJ_TN)
    ) ref_mtp (
        .clk(clk), .rst(rst), .start(rm_start), .busy(rm_busy), .done(rm_done),
        .mode(rm_mode), .pos(rm_pos), .s_len(rm_slen), .h_t(rm_h_t), .emb_t1(rm_emb),
        .logits(rm_logits), .argmax(rm_argmax), .h_mtp(rm_h_mtp),
        .cn_req(rm_cn_req), .cn_which(rm_cn_which), .cn_idx(rm_cn_idx), .cn_val(rm_cn_val),
        .pw_req(rm_pw_req), .pw_ptile(rm_pw_ptile), .pw_k(rm_pw_k), .pw_col(rm_pw_col), .pw_scale(rm_pw_scale),
        .gn_req(rm_gn_req), .gn_which(rm_gn_which), .gn_idx(rm_gn_idx), .gn_val(rm_gn_val),
        .aw_req(rm_aw_req), .aw_sel(rm_aw_sel), .aw_grp(rm_aw_grp), .aw_k(rm_aw_k),
        .aw_col(rm_aw_col), .aw_scale(rm_aw_scale),
        .kc_req(rm_kc_req), .kc_idx(rm_kc_idx), .kc_ckv(rm_kc_ckv), .kc_krope(rm_kc_krope), .kc_valid(rm_kc_valid),
        .rw_req(rm_rw_req), .rw_k(rm_rw_k), .rw_col(rm_rw_col), .rw_scale(rm_rw_scale),
        .fw_req(rm_fw_req), .fw_sel(rm_fw_sel), .fw_grp(rm_fw_grp), .fw_k(rm_fw_k),
        .fw_shared(rm_fw_shared), .fw_eidx(rm_fw_eidx),
        .fw_col(rm_fw_col), .fw_col_up(rm_fw_col_up), .fw_scale_g(rm_fw_scale_g), .fw_scale_u(rm_fw_scale_u),
        .lw_req(rm_lw_req), .lw_vtile(rm_lw_vtile), .lw_k(rm_lw_k), .lw_col(rm_lw_col)
    );
    /* verilator lint_off UNUSEDSIGNAL */
    wire _ur = &{1'b0, r_busy, r_logits, r_em_req, r_aw_req, r_fw_req, r_rw_req, r_gn_req,
                 r_fn_req, r_lw_req, r_idx_fresh, r_idx_win,
                 rm_busy, rm_logits, rm_cn_req, rm_pw_req, rm_aw_req, rm_fw_req, rm_rw_req,
                 rm_gn_req, rm_lw_req};
    /* verilator lint_on UNUSEDSIGNAL */

    // ================= reference drive tasks =================
    reg [TOKW-1:0]         r_arg;
    reg [MODEL_DIM*16-1:0] r_hs;
    task run_model; input [TOKW-1:0] tk; input [POSW-1:0] pz; integer w; begin
        r_tok=tk; r_pos=pz;
        @(negedge clk); r_start=1'b1; @(negedge clk); r_start=1'b0;
        w=0; while (!r_done && w<2000000) begin @(negedge clk); w=w+1; end
        if (!r_done) begin $display("FAIL K=%0d: ref model TIMEOUT", DRAFT_K); errors=errors+1; end
        r_arg=r_argmax; r_hs=r_hstate;
        if (^r_arg===1'bx) begin $display("FAIL K=%0d: ref model argmax X", DRAFT_K); errors=errors+1; end
    end endtask

    reg [TOKW-1:0]         rm_arg;
    reg [MODEL_DIM*16-1:0] rm_hout;
    task run_mtp; input [MODEL_DIM*16-1:0] hin; input [MODEL_DIM*16-1:0] emb;
                  input [POSW-1:0] pz; input [IDXW:0] sl; input md; integer w; begin
        rm_h_t=hin; rm_emb=emb; rm_pos=pz; rm_slen=sl; rm_mode=md; cur_mode=md;
        @(negedge clk); rm_start=1'b1; @(negedge clk); rm_start=1'b0;
        w=0; while (!rm_done && w<2000000) begin @(negedge clk); w=w+1; end
        if (!rm_done) begin $display("FAIL K=%0d: ref mtp TIMEOUT", DRAFT_K); errors=errors+1; end
        rm_arg=rm_argmax; rm_hout=rm_h_mtp;
        if (^rm_arg===1'bx) begin $display("FAIL K=%0d: ref mtp argmax X", DRAFT_K); errors=errors+1; end
    end endtask

    // ================= INDEPENDENT golden : greedy + chained-draft reference =====
    reg [TOKW-1:0] ref_commit [0:4095];
    integer ref_n, ref_acc, ref_rej, ref_tot;
    reg [TOKW-1:0] gtok [0:DRAFT_K];
    reg [TOKW-1:0] draft [0:DRAFT_K-1];
    task compute_ref;
        input [TOKW-1:0]  p0;
        input [POSW-1:0]  pos0;
        input [IDXW:0]    sl;
        input             md;
        input integer     np;
        reg [TOKW-1:0] tok; reg [POSW-1:0] pos;
        reg [MODEL_DIM*16-1:0] hin, embv; reg [TOKW-1:0] prevt;
        integer pass, jj, p, brk, ei;
        begin
            ref_n=0; ref_acc=0; ref_rej=0; ref_tot=0;
            tok=p0; pos=pos0; r_slen=sl; cur_mode=md;
            for (pass=0; pass<np; pass=pass+1) begin
                // greedy rollout gtok[0..K] at the SHARED pass pos
                run_model(tok, pos); gtok[0]=r_arg; hin=r_hs;
                for (jj=1; jj<=DRAFT_K; jj=jj+1) begin
                    run_model(gtok[jj-1], pos); gtok[jj]=r_arg;
                end
                // chained MTP drafts under the DUT seed convention
                prevt = gtok[0];               // emb(m_0) for step 0
                for (jj=0; jj<DRAFT_K; jj=jj+1) begin
                    for (ei=0; ei<MODEL_DIM; ei=ei+1) embv[16*ei+:16] = EMB[prevt][ei];
                    run_mtp(hin, embv, pos + 1 + jj, sl, md);
                    draft[jj] = rm_arg;
                    hin       = rm_hout;       // chain the pre-final-norm state
                    prevt     = draft[jj];     // emb(d_{jj}) for next step
                end
                // longest accepted prefix p (== spec_decode_seq's rule)
                p=0; brk=0;
                for (jj=0; jj<DRAFT_K; jj=jj+1) begin
                    if (!brk && (draft[jj]==gtok[jj])) p=p+1; else brk=1;
                end
                // commit the greedy tokens gtok[0..p]
                for (jj=0; jj<=p; jj=jj+1) begin ref_commit[ref_n]=gtok[jj]; ref_n=ref_n+1; end
                ref_acc=ref_acc+p; ref_rej=ref_rej+(DRAFT_K-p); ref_tot=ref_tot+(p+1);
                tok=gtok[p]; pos=pos+(p+1);
            end
        end
    endtask

    // ================= one binding case : reference vs DUT =====================
    integer wd, ci;
    task run_case;
        input integer    np;
        input [TOKW-1:0]  ptok;
        input [POSW-1:0]  pos0;
        input [IDXW:0]    sl;
        input             md;
        input             fz;          // force-zero mtp weights
        input             exp_acc_pos; // assert reference accepts>0
        input [8*16-1:0]  nm;
        begin
            test_count = test_count + 1;
            force_zero_mtp = fz; cur_mode = md;
            $display(".. K=%0d %0s: start (np=%0d fz=%0d md=%0d)", DRAFT_K, nm, np, fz, md); $fflush;

            // (a) INDEPENDENT reference (DUT idle / in reset)
            rst=1'b1; repeat(3) @(negedge clk); rst=1'b0; @(negedge clk);
            compute_ref(ptok, pos0, sl, md, np);
            if (exp_acc_pos && !(ref_acc>0)) begin
                $display("FAIL K=%0d %0s: reference accepts=%0d not >0 (accept scenario not realized)",
                         DRAFT_K, nm, ref_acc); errors=errors+1;
            end

            // (b) run the DUT on the SAME inputs (fresh counters)
            rst=1'b1; repeat(3) @(negedge clk); rst=1'b0; @(negedge clk);
            prompt_tok=ptok; start_pos=pos0; s_len=sl; mtp_mode=md; num_passes=np[15:0];
            commit_n=0; cap=1'b1;
            @(negedge clk); start=1'b1; @(negedge clk); start=1'b0;
            wd=0;
            while (!done && wd<20000000) begin @(negedge clk); wd=wd+1; end
            cap=1'b0;
            if (!done) begin $display("FAIL K=%0d %0s: DUT TIMEOUT", DRAFT_K, nm); errors=errors+1; disable run_case; end
            @(negedge clk);

            // ---- spec == greedy : EXACT committed-stream equality ----
            if (commit_n !== ref_n) begin
                $display("FAIL K=%0d %0s: committed beats %0d != reference %0d", DRAFT_K, nm, commit_n, ref_n);
                errors=errors+1;
            end else begin
                for (ci=0; ci<ref_n; ci=ci+1)
                    if (got[ci] !== ref_commit[ci]) begin
                        $display("FAIL K=%0d %0s: commit[%0d]=%0d != greedy ref %0d (spec!=greedy)",
                                 DRAFT_K, nm, ci, got[ci], ref_commit[ci]);
                        errors=errors+1;
                    end
            end
            // ---- counters mirror spec_decode_seq exactly ----
            if (total_tokens !== ref_tot) begin $display("FAIL K=%0d %0s: total_tokens %0d != ref %0d", DRAFT_K, nm, total_tokens, ref_tot); errors=errors+1; end
            if (main_passes !== np)        begin $display("FAIL K=%0d %0s: main_passes %0d != %0d",    DRAFT_K, nm, main_passes, np);     errors=errors+1; end
            if (accepts !== ref_acc)       begin $display("FAIL K=%0d %0s: accepts %0d != ref %0d",     DRAFT_K, nm, accepts, ref_acc);    errors=errors+1; end
            if (rejects !== ref_rej)       begin $display("FAIL K=%0d %0s: rejects %0d != ref %0d",     DRAFT_K, nm, rejects, ref_rej);    errors=errors+1; end

            $display("ok  K=%0d %0s: passes=%0d commits=%0d(=%0.2f/pass) acc=%0d rej=%0d fz=%0d | committed==greedy EXACT | K_eff=%0.2f",
                     DRAFT_K, nm, np, commit_n, (1.0*commit_n)/np, accepts, rejects, fz,
                     (1.0*(accepts+np))/np);
            $fflush;
        end
    endtask

    // ================= the binding sequence =================
    initial begin
        start=1'b0; r_start=1'b0; rm_start=1'b0;
        prompt_tok=0; start_pos=0; s_len=1; mtp_mode=1'b0; num_passes=0;
        r_tok=0; r_pos=0; r_slen=1;
        rm_mode=1'b0; rm_pos=0; rm_slen=1; rm_h_t=0; rm_emb=0;
        cur_mode=1'b0; force_zero_mtp=1'b0;
        cap=1'b0; commit_n=0;
        test_count=0; errors=0; finished=1'b0; tests_out=0; errors_out=0;
        rst=1'b1; repeat(4) @(negedge clk); rst=1'b0; @(negedge clk);

        // Serialize engines: sit idle (rst=0, all FSMs in IDLE -> cheap) until
        // this engine is enabled, so only ONE engine actively computes at a time
        // (avoids paying for both full netlists' active logic every cycle).
        wait (en === 1'b1); @(negedge clk);

        // ONE fixed weight set for the whole engine.
        build_stim(500); build_mtp_stim(90000);

        // DELIBERATELY BOUNDED, FAST binding (B8): the spec==greedy safety property
        // is asserted EXACTLY on every committed beat; K_eff (acceptance) is only
        // REPORTED (not asserted) since with random weights the chained drafts may
        // or may not be accepted -- either way committed==greedy holds.  No search
        // over the vocab (each trial is a full chain pass -> too slow); just two
        // fixed multi-pass cases with NUM_PASSES=2 so the multi-pass cursor advance
        // is exercised end-to-end while the whole engine still finishes in minutes.

        // (1) nonzero, DENSE mtp, 2 passes.  committed==greedy EXACT; K_eff reported.
        run_case(NP, 4'd3, 0, 1'b1, 1'b0, 1'b0, 1'b0, "nonzero-den");

`ifdef B8_ZERO_CASE
        // (2) forced-zero mtp weights : garbage drafts -> still commits EXACTLY the
        //     greedy stream (safety independent of draft quality), 2 passes.
        run_case(NP, 4'd5, 1, 1'b1, 1'b0, 1'b1, 1'b0, "zero-mtp");
`endif

        tests_out=test_count; errors_out=errors; finished=1'b1;
    end
endmodule

//============================================================================
//  spec_chain_top_tb : run the K=2 and K=3 binding engines, aggregate
//============================================================================
module spec_chain_top_tb;
    reg clk = 1'b0;
    always #5 clk = ~clk;

    wire [31:0] t2, e2;
    wire        f2;

    // ONE engine only (K=2).  Each spec-chain pass runs a FULL glm_model_fp8
    // forward, and every extra elaborated model/MTP netlist costs per simulated
    // cycle -- so a single K=2 engine (main + verify + ref_model + 2 mtp) is the
    // largest slice that finishes in the fast budget.  K=3 is validated separately
    // (spec_decode_seq_k already binds the DRAFT_K=3 accept/reject rule); here the
    // costly part is the model itself, whose K-independence is proven by K=2.
    sct_engine #(.DRAFT_K(2)) engK2 (.clk(clk), .en(1'b1), .tests_out(t2), .errors_out(e2), .finished(f2));

    // global watchdog
    initial begin #4000000000; $display("FAIL: global timeout"); $fatal(1,"timeout"); end

    initial begin
        wait (f2 === 1'b1);
        @(negedge clk);
        if (e2 != 32'd0) begin
            $display("FAILED: %0d error(s) across %0d tests (K=2)", e2, t2);
            $fatal(1,"fail");
        end
        $display("ALL %0d TESTS PASSED", t2);
        $finish;
    end
endmodule
