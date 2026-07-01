# P2_MEMORY_MAP — On-chip memory / register-array classification (reliability effort)

> **Status:** analysis artifact for the P2 reliability track. **Read-only** — this
> document changes no RTL. It is the concrete work-list that drives **C6**
> (ECC-on-payload) and **C7** (MBIST integration).
>
> **Scope:** every on-chip memory / register array declared in the synthesizable
> RTL under `src/` (the `test/` testbenches are out of scope). Rows are cited to
> `file:line`. Off-die DDR5 / Flash payloads (modeled by the TB as latency
> memories) are called out explicitly and excluded from on-die ECC/MBIST.

---

## 1. Purpose & the three classes

A radiation / retention / manufacturing bit-flip in an on-die array has one of
three consequences, and each wants a different protection strategy:

| Class | What lives there | Failure mode | Required protection |
|-------|------------------|--------------|---------------------|
| **SECDED** | **Model data** — weights, activations, KV latents, logits, partial-sum accumulators, weight-defining decode tables. | A single flip **silently corrupts a numeric result** (wrong token). Indistinguishable from correct data without a code. | Single-Error-Correct / Double-Error-Detect (Hamming+overall-parity, see `src/ecc_secded.v` / `src/ecc_mem_wrap.v`). Correct SBUs transparently; surface DBUs. **(C6 target.)** |
| **PARITY+MBIST** | **Control state** — cache directory tags/valid/rank, FIFO pointers & occupancy counters, outstanding-request counters, expert-id queues, clock-gate counters, DSA/top-k index lists. | A flip mis-routes, mis-counts, or mis-selects. Usually self-healing over time, but a wrong tag can return the *wrong* resident line silently. | At least **parity-detect** (flag + recover/flush); every such array is on-die SRAM/flops and is **MBIST-testable** for manufacturing defects. **(C7 target; parity is a lighter C6 add.)** |
| **OFF-DIE** | **External DDR5 / Flash payload** (the actual weight/KV/expert bytes) reached *through* the xbars and loaders. Modeled in the TB as a latency memory; **not on our die**. | Handled by the DRAM/Flash device's own ECC and by the SECDED wrapper at the on-die memory-controller boundary. | **Out of scope** for on-die ECC/MBIST. Protect at the `ecc_mem_wrap` boundary where the payload lands on-die. |

The infrastructure to implement the SECDED class already exists and is **not** in
the table as a target (it *is* the mechanism):

- `src/ecc_secded.v` — parameterized extended-Hamming SECDED codec, `DATA_W`
  default **64** → `CODE_W = DATA_W + P + 1` (for 64b: 64+7+1 = **72**).
- `src/ecc_mem_wrap.v:125` `mem [0:DEPTH-1]` — a `DATA_W×DEPTH` RAM that stores
  the **codeword** (never raw payload), correcting SBUs / detecting DBUs on read.
  This is the drop-in wrapper C6 instantiates in front of every SECDED-class array.

---

## 2. Master classification table

Grouped by function. **Active GLM-5.2 FP8 datapath + memory system first**, then
the transient compute scratch, then the legacy TPU core and the superseded bf16
GLM path (kept separate per the task). Every `reg`-array found by the STEP-1
greps appears here or in the §5 exclusion list.

### 2.1 Memory system — persistent / semi-persistent on-die stores (primary C6/C7 targets)

| Module | Array (signal) | Width × Depth | Holds | Class | Rationale |
|--------|----------------|---------------|-------|-------|-----------|
| kv_cache_pager.v:109 | `ring [0:RESIDENT-1]` | `ROW_BITS`(768) × `RESIDENT`(32) | Resident-window **latent KV rows** `[c_kv \| k_rope]` (opaque bf16 vectors) | **SECDED** | This *is* model KV. A flip returns a corrupt latent to `mla_attn_fp8`'s `kc_*` read → wrong attention, silently. **ECC lane note below.** |
| weight_decomp.v:123 | `count_table [0:MAXLEN]` | `COUNTW` × `MAXLEN+1` | Canonical-Huffman length→count table | **SECDED** | Weight-*defining*: a flip shifts codeword boundaries → mis-decodes the entire FP8 weight stream. Corrupts model data as surely as flipping a weight. |
| weight_decomp.v:124 | `symbol_table [0:NSYMMAX-1]` | `SYMW` × `2^SYMW` | Canonical-order FP8 weight symbols | **SECDED** | A flip emits the *wrong FP8 weight byte* on decode. Model data. |
| weight_decomp2.v:113 | `count_table [0:CTDEP-1]` | `COUNTW` × `NCTX<<CTADW` | Per-context Huffman count table | **SECDED** | Same as above, context-indexed. |
| weight_decomp2.v:114 | `symbol_table [0:SYDEP-1]` | `SYMW` × `NCTX<<SYMW` | Per-context canonical symbols | **SECDED** | Same as above, context-indexed. |
| expert_cache_pf.v:112 | `tag_arr [0:SLOTS-1]` | `ID_W` × `SLOTS` | Resident expert-id per slot (directory) | **PARITY+MBIST** | Directory tag. A flip returns the *wrong* resident expert silently → parity must at least *detect* and force a miss/refetch. The cached expert **bytes are OFF-DIE** (this module only returns `resp_slot` = an HBM slot). |
| expert_cache_pf.v:111 | `valid_arr [0:SLOTS-1]` | 1 × `SLOTS` | Per-slot valid bit | **PARITY+MBIST** | Control; flip → false hit/miss. |
| expert_cache_pf.v:113 | `rank [0:SLOTS-1]` | `SLOT_W` × `SLOTS` | LRU recency position | **PARITY+MBIST** | Replacement control; flip only degrades hit-rate (self-heals), but MBIST-test the cells. |
| expert_cache_pf.v:114 | `pf_flag [0:SLOTS-1]` | 1 × `SLOTS` | "installed by prefetch, not yet demanded" | **PARITY+MBIST** | Stats/control only. |
| expert_cache_pf.v:126 | `freq [0:SLOTS-1]` | `FREQ_W` × `SLOTS` | Saturating access frequency (REPL_POLICY=1) | **PARITY+MBIST** | Replacement control; self-healing. |
| expert_cache_ctrl.v:86 | `tag_arr [0:SLOTS-1]` | `ID_W` × `SLOTS` | Resident expert-id (directory) | **PARITY+MBIST** | Same directory-tag role as `expert_cache_pf`; bytes OFF-DIE. |
| expert_cache_ctrl.v:85 | `valid_arr [0:SLOTS-1]` | 1 × `SLOTS` | Per-slot valid bit | **PARITY+MBIST** | Control. |
| expert_cache_ctrl.v:87 | `rank [0:SLOTS-1]` | `SLOT_W` × `SLOTS` | LRU recency | **PARITY+MBIST** | Replacement control. |
| expert_predictor.v:75 | `freq [0:TBL-1]` | `FREQ_W` × `TBL` | Per-expert access-frequency table | **PARITY+MBIST** | Prefetch *hint* table; a flip only mis-hints (prefetch is best-effort, self-correcting). MBIST the SRAM. |
| expert_predictor.v:77 | `age_ctr [0:N_LAYER-1]` | `AGE_W` × `N_LAYER` | Per-layer aging counter | **PARITY+MBIST** | Control counter. |
| cdc_async_fifo.v:92 | `mem [0:DEPTH-1]` | `DATA_W`(32) × `2^ADDR_W`(16) | **CDC-crossing** dual-clock FIFO payload | **PARITY+MBIST** | Two-clock crossing; payload transits for only a few cycles between SECDED-protected endpoints. Parity-detect on transit + MBIST; do **not** put a same-cycle SECDED decode in the CDC path (adds crossing latency). Gray pointers are the CDC-safe part; the RAM cells still need MBIST. |
| ddr5_xbar.v:156 | `fifo [0:N_CH-1][0:RESP_QD-1]` | `PAY_W`={tag,data} × `N_CH·RESP_QD` | Per-channel **response** {tag,data} in flight | **PARITY+MBIST** | Response-path transit buffer, not a resting store (payload's home is off-die DDR5, SECDED'd at the controller boundary). Classify as control/transit: parity-detect + MBIST. |
| ddr5_xbar.v:157–159 | `head`,`tail`,`cnt [0:N_CH-1]` | `PTR_W`/`CNT_W` × `N_CH` | FIFO read/write pointers + occupancy | **PARITY+MBIST** | Pure FIFO control; a flip mis-drains → parity-detect. |
| flash_xbar.v:177 | `fifo [0:N_CH-1][0:RESP_QD-1]` | `PAY_W`={tag,data} × `N_CH·RESP_QD` | Per-channel Flash **response** in flight | **PARITY+MBIST** | Same transit-buffer role as `ddr5_xbar` response FIFO. |
| flash_xbar.v:178–180 | `head`,`tail`,`cnt [0:N_CH-1]` | `PTR_W`/`CNT_W` × `N_CH` | FIFO pointers + occupancy | **PARITY+MBIST** | FIFO control. |
| flash_xbar.v:142 | `outst [0:N_CH-1]` | `OST_W` × `N_CH` | Per-channel outstanding-read counter | **PARITY+MBIST** | Flow-control counter; flip → lost/double issue → parity-detect. |
| boot_loader.v:134 | `fifo_mem [0:BURST-1]` | `DATA_W` × `BURST`(8) | Flash→DDR5 copy **skid FIFO** (weight/data words) | **PARITY+MBIST** | Payload-in-transit skid buffer; **both** endpoints (Flash source, DDR5 sink) are SECDED-protected, so detect-only + MBIST here suffices. (Optionally carry the SECDED codeword through it for end-to-end coverage — C6 decision.) |
| boot_loader.v:118–120 | `fbase_q`,`dbase_q`,`len_q [0:SEG_MAX-1]` | `ADDR_W`/`LEN_W` × `SEG_MAX` | Latched copy-segment descriptors (base/base/len) | **PARITY+MBIST** | Address/length control; a flip mis-addresses a DMA burst → parity-detect. |
| glm_fp8_soc.v:422 | `efifo [0:EFIFO_DEPTH-1]` | `EIDXW` × `EFIFO_DEPTH` | Expert-id request FIFO to the cache | **PARITY+MBIST** | Queue of expert **ids** (control), not weights; flip → wrong expert fetched → parity-detect. |
| glm_fp8_system.v:453 | `efifo [0:EFIFO_DEPTH-1]` | `EIDXW` × `EFIFO_DEPTH` | Expert-id request FIFO (system-level dup) | **PARITY+MBIST** | Same role as `glm_fp8_soc` efifo. |
| clk_en_ctrl.v:110–112 | `en_reg`,`hold_cnt`,`gcnt [0:N_CLUSTER-1]` | 1/`HOLD_W`/`CNT_W` × `N_CLUSTER` | Clock-gate enable + hold/grace counters | **PARITY+MBIST** | Clock-gating control; flip → spurious gate/ungate (no data corruption) → MBIST the cells, parity optional. |

**ECC lane-partition note (kv_cache_pager `ring`).** `ROW_BITS` defaults to
**768** = (KV_LORA 32 + ROPE 16) × 16b. 768 = **12 × 64**, so with the 64-bit
`ecc_secded` lane it tiles into exactly **12 clean SECDED lanes** (12 × 72 = 864
stored bits/row). **However `ROW_BITS` is a free parameter** that scales with
`KV_LORA`/`ROPE`; the real GLM-5.2 row (c_kv 512 + k_rope 64 elements) and other
LoRA configs are **not guaranteed to be a multiple of the 64-bit ECC lane**. C6
must therefore **partition the row into 64-bit lanes and pad a ragged final
lane** to 64 before encode (do not assume 64-alignment), and decide whether a DBU
in any lane poisons the whole row. This is the one SECDED target whose width is
not natively code-aligned across configurations — flag it in the C6 wrapper.

### 2.2 Active FP8 datapath — activation / KV / accumulator scratch (SECDED-class model data, single-token scope)

These hold live numeric intermediates (bf16 activations, FP8-block accumulators,
scores, logits). By the class rule a flip corrupts **model data**, so they are
**SECDED-class**; but each is *transient* (rebuilt every token) and lives behind
SECDED-protected weight/KV inputs — so they are **lower C6 priority** than the
persistent stores in §2.1. All are on-die → **MBIST-testable** regardless.

| Module | Array (signal) | Width × Depth | Holds | Class | Rationale |
|--------|----------------|---------------|-------|-------|-----------|
| glm_matmul_fp8.v:328 | `accx [0:NB-1][0:PE_M-1][0:PE_N-1]` | `ACC_W`(48) signed × NB·PE_M·PE_N | **GEMM accumulator banks** (per-block fixed-point partial sums) | **SECDED** | Partial-sum bank; a flip directly corrupts the output activation of every matmul. Highest-leverage scratch — one flip → wrong row of the GEMM result. |
| mla_attn_fp8.v:250 | `scores [0:PE_M-1][0:H_HEADS-1][0:S_MAX-1]` | 16 × PE_M·H_HEADS·S_MAX | Per-row attention **scores** scratch | **SECDED** | Activation; flip → wrong softmax → wrong context. |
| mla_attn_fp8.v:251 | `vstore [0:H_HEADS-1][0:S_MAX-1][0:V_DIM-1]` | 16 × H_HEADS·S_MAX·V_DIM | Cached **V** rows (shared key value store) | **SECDED** | Holds V (model KV/activation); flip → corrupt attention output. |
| mla_attn_fp8.v:252 | `probs [0:PE_M-1][0:H_HEADS-1][0:S_MAX-1]` | 16 × PE_M·H_HEADS·S_MAX | Softmax **probabilities** scratch | **SECDED** | Activation. |
| mla_attn_fp8.v:227–254 | `xbuf,qlora,qlora_n,qfull,qrot,ckv_cur,krope_cur,ckv_key,ckv_n,knope_j,v_j,krope_j,ctx,outbuf` | 16 × (per-row / shared) | Q/KV projection + RoPE + context bf16 activation scratch | **SECDED** | Live activations/KV latents of the MLA datapath. |
| mla_attn_fp8.v:445 | `ctx_acc [0:PE_M-1]` | 32 × PE_M | Per-row context fp32 accumulator | **SECDED** | Activation partial sum. |
| mla_attn_fp8.v:548 / :568 | `a_emax [0:PE_M-1]`, `a_emax_q [0:PE_M-1]` | 8 × PE_M | Per-row bf16 **exponent-max (block scale)** | **SECDED** | Shared FP8 block scale for a whole row — a flip mis-scales every element in the row → high-leverage magnitude corruption. |
| mla_attn_fp8.v:257 | `sel_list [0:TOPK-1]` | `IDXW` × TOPK | DSA-selected row indices | **PARITY+MBIST** | Index list (control), not data. |
| mtp_head_fp8.v:262–268 | `hbuf,ebuf,cbuf,hprime,xcur,xn,lbuf` | 16 × (MODEL_DIM / CK / VOCAB) | MTP RMSNorm/proj/decoder activation + **logits** scratch | **SECDED** | Activations and LM-head logits. |
| moe_router_fp8.v:303/378/402 | `tk_score_in`,`s_reg`,`rs_reg [0:PE_M-1]` | 32 × PE_M | Per-row router **scores** + fold factor | **SECDED** | Gating scores (data); flip → wrong expert weighting. |
| glm_decoder_block_fp8.v:446/476/477 | `cur_gate_f`,`sh_add_a`,`sh_add_b [0:PE_M-1]` | 32 × PE_M | Per-row gate value + adder operands | **SECDED** | Activation/gate data. |
| glm_decoder_block_fp8.v:445 | `row_active [0:PE_M-1]` | 1 × PE_M | Per-row active flag | **PARITY+MBIST** | Control bit. |
| glm_model_fp8.v:374 | `am_best [0:PE_M-1]` | 16 × PE_M | Per-row best logit value (argmax) | **SECDED** | Result data (drives the emitted token). |
| glm_model_fp8.v:375 | `am_arg [0:PE_M-1]` | `TOKW` × PE_M | Per-row argmax **token index** | **PARITY+MBIST** | Index/control (the final selected token id). |
| swiglu_expert_fp8.v:479 | `n_hval [0:MTN-1]` | 16 × MTN | SwiGLU intermediate `h` activations | **SECDED** | Activation. |
| swiglu_expert_fp8.v:217/480 | `h_emax`,`n_h_emax [0:PE_M-1]` | 8 × PE_M | Per-row bf16 exp-max (block scale) | **SECDED** | Row block scale (see `a_emax` rationale). |
| batched_moe.v:191 | `row_gate [0:PE_M-1]` | 16 × PE_M | Per-row gate weight | **SECDED** | Gating data. |
| batched_moe.v:192 | `row_has [0:PE_M-1]` | 1 × PE_M | Per-row valid flag | **PARITY+MBIST** | Control bit. |
| rmsnorm_unit.v:112/126 | `buf_mem [0:NBEATS-1]`, `sq [0:LANES-1]` | `LANES*16`/32 × depth | RMSNorm input beats + per-lane square accum | **SECDED** | Activation scratch. |
| glm_softmax.v:71/72/138 | `xbuf`,`ebuf`,`pbuf [0:LEN-1]` | 32/32/16 × LEN | Softmax logits / exp / probability scratch | **SECDED** | Activation scratch. |
| sampler.v:117/172 | `zbuf [0:VOCAB-1]`, `pprob [0:KEFF-1]` | 32 × depth | Temp-scaled logits + kept-slot probabilities | **SECDED** | Logit/probability data. |
| sampler.v:147/148 | `kidx [0:KEFF-1]`, `kscore [0:KEFF-1]` | `VW`/32 × KEFF | Kept vocab index / kept score | **PARITY+MBIST** (kidx) / **SECDED** (kscore) | `kidx` = index (control); `kscore` = fp32 score (data). |
| topk_select.v:214/232/250 | `score_mem`,`leaf_score`,`t_score [0:N-1]` | `SCORE_W` × N | Captured/tournament **scores** | **SECDED** | Score data. |
| topk_select.v:251 | `t_idx [0:N-1]` | `IDXW` × N | Tournament index tags | **PARITY+MBIST** | Index/control. |
| dsa_indexer.v:162/170/196 | `qbuf [0:IDX_DIM-1]`, `score_mem [0:S_MAX-1]`, `acc_l [0:LANES-1]` | 16/32/32 × depth | Query vector, per-key scores, per-lane dot accum | **SECDED** | Activation/score data. |
| dsa_indexer.v:197/198/212 | `dim_issue_l`,`dim_done_l [0:LANES-1]`, `tagq [0:TQD-1]` | `DIW+1`/`LW` × depth | Per-lane term counters + tag queue | **PARITY+MBIST** | Counters / tag queue (control). |
| glm_act.v:330–455 | per-lane stage regs (`s1_r,s2a_*,s2_pr,s3_d,s4_s,n1_r,n2a_p,n2_pr,n3_d,n4_s,x1..x5,nx1..`) | 32 × `LANES` each | SiLU/sigmoid Horner **activation** pipeline stages | **SECDED**-class (transient) | Live activations in a feed-forward poly pipeline; single-pass, flushes every few cycles. SECDED-class data but lowest priority; MBIST-covered as flops. Listed as one grouped row (see §5). |
| spec_chain_top.v:167 | `h_chain [0:DRAFT_K]` | `MODEL_DIM*16` × DRAFT_K+1 | Draft-chain **hidden states** `h_mtp` | **SECDED** | Model activation carried across draft steps. |
| spec_chain_top.v:168/169 | `draft_id [0:DRAFT_K-1]`, `truth_id [0:DRAFT_K]` | `TOKW` × depth | Draft / verify **token ids** | **PARITY+MBIST** | Token-index queues (control). |
| spec_decode_top.v:267 | `emb_buf [0:MODEL_DIM-1]` | 16 × MODEL_DIM | Embedding vector buffer | **SECDED** | Activation. |
| spec_decode_seq.v:131/233 | `pending_draft [0:DRAFT_K-1]`, `obuf [0:DRAFT_K]` | `TOKW` × depth | Pending / committed **token id** queues | **PARITY+MBIST** | Token-index control queues. |
| swiglu_expert.v:164/165 | `xbuf [0:HIDDEN-1]`, `hbuf [0:INTER-1]` | 16 × depth | Token x + SiLU·up activation (bf16 variant) | **SECDED** | Activation scratch. |
| moe_router.v:138/210/243 | `xbuf`,`gate_f`,`win_gate` | 16/32/32 × depth | Router input + gate values (bf16 variant) | **SECDED** | Activation/gate data. |
| moe_router.v:242 | `win_idx [0:TOPK-1]` | `IDXW` × TOPK | Winning expert indices | **PARITY+MBIST** | Index/control. |

### 2.3 Legacy TPU core + superseded bf16 GLM path (separate — noted per task)

The bf16 GLM modules are **superseded** by their `_fp8` twins (the active
datapath); the TPU tensor units are the legacy v2.0 core. Same classification
rule applies; listed apart so C6/C7 can decide whether the legacy blocks are
taped out at all.

| Module | Array (signal) | Width × Depth | Holds | Class | Rationale |
|--------|----------------|---------------|-------|-------|-----------|
| tile_memory.v:70 | `lines [0:TM_LINES-1]` | `LINE_W`(128) × `TM_LINES`(32) | Tensor **operand tiles** (shared TM) — legacy core | **SECDED** | Model data (every tensor unit's operands/results). If the TPU core ships, this is a top SECDED target. |
| memory.v:50 | `sram [0:DMEM_DEPTH-1]` | `XLEN`(32) × `DMEM_DEPTH`(256) | Scalar **DMEM** data words — legacy core | **SECDED** | Model/scalar data. |
| register_file.v:70 | `registers [0:REGS-1]` | `XLEN`(32) × `RF_REGS`(16) | Architectural scalar registers r0..r15 — legacy core | **PARITY+MBIST** | Tiny (16×32); a flip corrupts a scalar operand. Parity-detect + MBIST; too small to justify SECDED but must be BIST-tested. (r0 is hardwired-zero at the read mux.) |
| mla_attn.v:190–217,418 | `xbuf,qlora,qlora_n,qfull,qrot,ckv_cur,krope_cur,ckv_n,knope_j,v_j,ctx,outbuf,krope_j`, `sel_list` | 16 × depth (+`IDXW` sel) | bf16 MLA activation scratch (**superseded** by `mla_attn_fp8`) | **SECDED** (data) / **PARITY** (`sel_list`) | Same roles as §2.2; older bf16 variant. |
| mtp_head.v:207–213 | `hbuf,ebuf,cbuf,hprime,xcur,xn,lbuf` | 16 × depth | bf16 MTP activation/logit scratch (**superseded**) | **SECDED** | Activations/logits. |
| glm_model.v:229/237/302 | `xcur`,`lbuf`,`xn` | 16 × depth | bf16 model activation/logit scratch (**superseded**) | **SECDED** | Activations/logits. |
| glm_decoder_block.v:185–188,348 | `xbuf,nrm,hbuf,fbuf,facc` | 16/32 × depth | bf16 decoder activation + FFN accum (**superseded**) | **SECDED** | Activations/partial sums. |
| glm_decoder_block.v:285/286 | `sel_e [0:TOPK-1]`, `sel_w [0:TOPK-1]` | `EIDXW`/16 × TOPK | Selected expert ids + weights (**superseded**) | **PARITY** (`sel_e`) / **SECDED** (`sel_w`) | Index vs gate-weight. |
| attention_unit.v:281/293/494 | `wrow [0:S-1]`, `sm_scratch [0:SCR_N-1]`, `sm_xline [0:NSCR_X-1]` | `Q016_W`/`LINE_W` × depth | Legacy TPU attention weight-row + softmax scratch | **SECDED** | Activation scratch (legacy core). |
| softmax_unit.v:247/252 | `ev [0:LEN-1]`, `pv [0:LEN-1]` | `PROD_W`/`Q016_W` × LEN | Legacy TPU softmax exp / prob scratch | **SECDED** | Activation scratch (legacy core). |

### 2.4 Off-die / TB-modeled payloads (OUT OF SCOPE for on-die ECC/MBIST)

Not `reg` arrays in `src/` — these are the external memories reached *through* the
modules above; the testbench models them as latency memories.

| Payload | Reached via | Class | Rationale |
|---------|-------------|-------|-----------|
| **DDR5 payload** (weights/activations resident set) | `ddr5_xbar` `mem_*` ports; `weight_loader` / `boot_loader` `mem_*`/`ddr_*`; `glm_fp8_system(_cdc)` `wl_mem_*` | **OFF-DIE** | On the DRAM die, not ours; protected by device ECC + the `ecc_mem_wrap` boundary where it lands on-die. |
| **Flash payload** (cold KV rows, cold expert weights) | `kv_cache_pager` `flash_*`; `flash_xbar` `flash_*`; `expert_cache_*` `flash_*`; `weight_decomp*` compressed input | **OFF-DIE** | Off-die NAND; SECDED belongs at the on-die controller boundary, not in these routing blocks. |

---

## 3. Per-class work-lists (for C6 / C7)

### 3.1 SECDED (C6 — wrap in `ecc_mem_wrap` / lane-partitioned SECDED)

**Persistent (do these first):**
- `kv_cache_pager.ring` (768b latent KV) — **lane-partition, ragged-lane aware** (§2.1 note).
- `weight_decomp.{count_table,symbol_table}`, `weight_decomp2.{count_table,symbol_table}` — weight-defining decode tables.
- (legacy, iff taped out) `tile_memory.lines`, `memory.sram`.

**Transient activation/accumulator scratch (SECDED-class; prioritize the high-leverage ones):**
- `glm_matmul_fp8.accx` (GEMM accumulator banks) — highest-leverage.
- `mla_attn_fp8.{scores,vstore,probs,ctx_acc,a_emax,a_emax_q, + xbuf/qlora/.../outbuf}`.
- `mtp_head_fp8.{hbuf,ebuf,cbuf,hprime,xcur,xn,lbuf}`; `spec_chain_top.h_chain`; `spec_decode_top.emb_buf`.
- `moe_router_fp8.{tk_score_in,s_reg,rs_reg}`; `glm_decoder_block_fp8.{cur_gate_f,sh_add_a,sh_add_b}`; `batched_moe.row_gate`; `glm_model_fp8.am_best`.
- `swiglu_expert_fp8.{n_hval,h_emax,n_h_emax}`; `rmsnorm_unit.{buf_mem,sq}`; `glm_softmax.{xbuf,ebuf,pbuf}`; `sampler.{zbuf,kscore,pprob}`; `topk_select.{score_mem,leaf_score,t_score}`; `dsa_indexer.{qbuf,score_mem,acc_l}`.
- Block-scale exponent fields (`*_emax`) are cheap, high-leverage SECDED wins (a flip mis-scales a whole row).
- Lowest priority (single-pass feed-forward poly stages): `glm_act` per-lane stage regs, and the legacy/bf16 §2.3 scratch.

### 3.2 PARITY+MBIST (C7 — MBIST wrap; C6 adds parity)

- **Cache directories:** `expert_cache_pf.{valid_arr,tag_arr,rank,pf_flag,freq}`, `expert_cache_ctrl.{valid_arr,tag_arr,rank}` — parity on **tags** matters (wrong-line-return).
- **Predictor tables:** `expert_predictor.{freq,age_ctr}`.
- **FIFOs & pointers:** `cdc_async_fifo.mem` (**CDC — MBIST only, no in-path decode**), `ddr5_xbar.{fifo,head,tail,cnt}`, `flash_xbar.{fifo,head,tail,cnt,outst}`, `boot_loader.{fifo_mem,fbase_q,dbase_q,len_q}`, `glm_fp8_soc.efifo`, `glm_fp8_system.efifo`.
- **Index/queue/counter arrays:** `mla_attn_fp8.sel_list`, `glm_model_fp8.am_arg`, `sampler.kidx`, `topk_select.t_idx`, `dsa_indexer.{dim_issue_l,dim_done_l,tagq}`, `spec_chain_top.{draft_id,truth_id}`, `spec_decode_seq.{pending_draft,obuf}`, `moe_router(_fp8) index arrays`, `batched_moe.row_has`, `glm_decoder_block_fp8.row_active`.
- **Clock/arch control:** `clk_en_ctrl.{en_reg,hold_cnt,gcnt}`, `register_file.registers` (legacy, MBIST the 16 words).

### 3.3 OFF-DIE (no on-die ECC/MBIST; protect at the controller boundary)

- DDR5 payload (via `ddr5_xbar` / loaders).
- Flash payload (via `flash_xbar` / `kv_cache_pager` / `expert_cache_*` / `weight_decomp*`).

---

## 4. Reference mechanism (already in tree — not a target)

- `ecc_secded.v` — SECDED codec (`DATA_W`=64 default; combinational encode+decode).
- `ecc_mem_wrap.v:125` `mem [0:DEPTH-1]` — SECDED RAM wrapper (stores codewords; corrects SBU / flags DBU; back-door port for fault injection & scrub). **This array is already protected** — it is the wrapper C6 instantiates, not a memory to protect.
- `mbist_ctrl.v` — MBIST controller (March-style; registered RAM-facing strobes) that C7 wires to every PARITY+MBIST (and every SECDED) array.

---

## 5. Coverage statement

A broad `reg`-array sweep over `src/*.v` (single- **and** multi-dimensional
declarations) returns **235** matches. This map classifies **every storage-class
array in the active GLM-5.2 FP8 datapath and the entire memory system
individually** (§2.1–§2.2), plus the **legacy TPU core and superseded bf16 GLM
path storage arrays** (§2.3) — i.e. 100% of the C6/C7-relevant memory surface,
including the multi-dimensional `scores`/`vstore`/`probs` MLA arrays that the
single-dimension STEP-1 pattern does not catch. The remaining matches are
**pipeline delay chains and additional legacy-compute stage scratch**, handled as
a group or excluded **by category with reasons** below (the task explicitly
permits excluding "small pipeline regs that are not arrays" and TB-only
memories):

**Grouped (SECDED-class, transient — one representative row instead of ~20):**
- `glm_act.v:330–455` — the per-lane SiLU/sigmoid Horner stage registers
  (`s1_r,s2a_r,s2a_p,s2_pr,s3_d,s4_s,n1_r,n2a_p,n2_pr,n3_d,n4_s,x1..x5,nx1`). All
  are `32×LANES` single-pass activation-pipeline stages (data-independent
  latency, flush every few cycles). Classified SECDED-class/transient, lowest
  C6 priority; MBIST-covered as flops. Enumerated collectively to keep the table
  legible — none is a persistent memory.

**Excluded — pipeline delay/shift chains (not addressable memories, not MBIST SRAM):**
These are `[0:LAT-1]` alignment shift registers whose only job is to delay a
signal by a fixed latency; they hold no addressable state and self-flush every
few cycles. A flip is transient and overwritten. Not ECC/MBIST targets.
- `glm_matmul_fp8.v:422–424` (`dq_v_pipe,dq_slot_pipe,dq_ash_pipe`), `:461–462` (`pd,vd`).
- `glm_fp_pipe.v:607,686–688,718,726,749–751,772–773,793–794,1019,1072`
  (`c_d,y0_d,sp_d,spv_d,xhalf_d,y0_dl,xhalf_chain,sp_chain,spv_chain,xhalf1_d,y1_dl,spv_chain2,sp_chain2,x_dl,r_dl`).
- `glm_matmul_pipe.v:179,185` (`ps,lane_pipe`).

**Excluded — additional legacy-compute stage/accumulator scratch (same SECDED-class-if-taped-out rationale as §2.3; not enumerated per-signal because these blocks are legacy TPU / superseded and hold only single-pass transient activations):**
- `gemm_ml.v`, `gemm_systolic.v`, `glm_matmul.v` — legacy/pre-FP8 systolic GEMM
  partial-sum and operand-staging arrays (superseded by `glm_matmul_fp8`).
- `conv2d_unit.v` — legacy TPU CONV2D line/window scratch.
- Additional per-stage scratch in `attention_unit.v` / `softmax_unit.v` beyond
  the representative rows in §2.3 (legacy TPU tensor path).
  All are SECDED-class-if-taped-out but lowest priority; MBIST-covered as flops.

**Excluded — not `reg` arrays (single wide regs / bus regs caught by the identifier grep):**
- `conv2d_unit.v:227` `obuf` — a single `LINE_W`-wide line-buffer register (not a
  memory array); legacy core.
- `weight_loader.v` — **no** `reg`-array memory: it streams weights through the
  registered `mem_*` interface; `w_scale_q`/`base_q`/counters are scalar/bus
  regs, not arrays. (Confirmed by grep.)
- `tpu_top.v`/`tpu_soc.v`/`tpu_axi.v` `*_mem` names — pipeline-stage / MMIO
  register **suffixes** (`opcode_mem`, `dst_mem`, `result_head`, `REG_*`), not
  memory arrays.

**Excluded — off-die (see §2.4):** external DDR5 / Flash payloads modeled by the
testbench as latency memories; not on our die.
