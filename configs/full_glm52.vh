//============================================================================
// configs/full_glm52.vh  --  REAL GLM-5.2-FP8 full-model configuration macros
//----------------------------------------------------------------------------
// PURPOSE
//   Single source of truth for the *production* GLM-5.2 shape, carried as
//   `define GLM52_* macros alongside the small-but-faithful RTL SLICE values
//   the committed TBs use.  The RTL (glm_model_fp8 and its hierarchy) is fully
//   parameterized; these macros are what an integrator overrides the module
//   parameters WITH to target the real checkpoint.
//
//   Every real value is cited to its source:
//     [cfg] = config.json field of zai-org/GLM-5.2-FP8
//     [doc] = docs/ACCEL_GLM52.md (§2 config table L27-35, §8.1 slice table L315-327)
//
//   NOTE ON q_lora / kv_lora: GLM-5.2 is DeepSeek-MLA-derived.  The DeepSeek-MLA
//   *standard* low-rank sizes are kv_lora_rank=512, q_lora_rank=1536 (docs
//   ACCEL_GLM52.md §2 L42-43).  These two are marked PENDING safetensors
//   confirmation against the published checkpoint tensor shapes; every other
//   value below is a hard config.json / doc citation.
//
//   USAGE (integration, NOT the committed slice TBs):
//     `include "full_glm52.vh"
//     glm_model_fp8 #(.MODEL_DIM(`GLM52_MODEL_DIM), .L(`GLM52_L), ...) u_full (...);
//   USAGE (elaboration study): see build/w2_B4_wrap.v + docs/P12_SCALEUP.md.
//============================================================================
`ifndef FULL_GLM52_VH
`define FULL_GLM52_VH

// ---- top-level model dims ----
`define GLM52_MODEL_DIM   6144      // [cfg] hidden_size            [doc L27]  (slice 128)
`define GLM52_L           78        // [cfg] num_hidden_layers      [doc L28]  (slice 6)
`define GLM52_N_DENSE     3         // [cfg] first_k_dense_replace  [doc L35]  (slice 3)
`define GLM52_VOCAB       154880    // [cfg] vocab_size             [doc L29]  (slice 256)

// ---- MLA attention ----
`define GLM52_H_HEADS     64        // [cfg] num_attention_heads    [doc L32]  (slice 4)
`define GLM52_NOPE        192       // [cfg] qk_nope_head_dim       [doc L32]  (slice 16)
`define GLM52_ROPE        64        // [cfg] qk_rope_head_dim       [doc L32]  (slice 16)
`define GLM52_V_DIM       256       // [cfg] v_head_dim             [doc L32]  (slice 32)
`define GLM52_Q_LORA      1536      // [cfg] q_lora_rank    PENDING safetensors [doc L42-43] (slice 64)
`define GLM52_KV_LORA     512       // [cfg] kv_lora_rank   PENDING safetensors [doc L42-43] (slice 32)
`define GLM52_THETA       8000000   // [cfg] rope_theta = 8e6       [doc L33]  (slice 8000000)
// qk_head_dim = NOPE+ROPE = 256; num_kv_heads=64; attention_bias=false [doc L32]

// ---- DSA sparse attention (attention scale-up is task B7's SWIN decouple) ----
`define GLM52_TOPK_ATTN   2048      // [cfg] index_topk             [doc L34]  (slice 8)
// index_topk_freq=4 ; index_skip_topk_offset=3 (IndexShare)         [doc L34]
// S_MAX (latent-ring / scores scratch) stays SMALL here -- see B7 caveat below.

// ---- MoE / FFN ----
`define GLM52_N_EXPERT    256       // [cfg] n_routed_experts       [doc L35]  (slice 8)
`define GLM52_TOPK        8         // [cfg] num_experts_per_tok    [doc L35]  (slice 2)
// n_shared_experts = 1 (always-on shared expert, handled in decoder block)  [doc L35]
`define GLM52_INTER_MOE   2048      // [cfg] moe_intermediate_size  [doc L35]  (slice 64)
`define GLM52_INTER_DENSE 12288     // [cfg] intermediate_size (dense front L0-2) [doc L35] (slice 256)
`define GLM52_RSCALE      32'h40200000 // routed_scaling_factor 2.5 (fp32)     [doc L35] (slice 2.5)

// ---- context / position width ----
// 1M-token context: position field must cover >= 1,048,576 positions.
// POSW=20 -> 2^20 = 1,048,576 exactly covers 1M.                     (slice 20)
`define GLM52_POSW        20

// ---- FP8 weight quantization ----
`define GLM52_BLK         128       // [cfg] quantization_config.weight_block_size=[128,128] (slice 128)

// ---- hardware tiling knobs (NOT model config -- accelerator microarch) ----
// These are free to retune for the full model; the slice defaults below are
// carried so a full-config build is a pure model-dim override.
`define GLM52_PE_N        4         // attention/matmul output-lane tile width  (slice 4)
`define GLM52_TN          4         // swiglu output-tile width                  (slice 4)
`define GLM52_LM_TN       4         // LM-head GEMV tile width (VOCAB % LM_TN==0) (slice 4)
`define GLM52_PE_M        1         // query-token batch B (1 == committed datapath)

// ============================================================================
// SLICE reference values (what the committed TBs / default params use) --
// mirrored here so a reader has both shapes in ONE place.
// ============================================================================
`define GLM52_SLICE_MODEL_DIM   128
`define GLM52_SLICE_L           6
`define GLM52_SLICE_VOCAB       256
`define GLM52_SLICE_H_HEADS     4
`define GLM52_SLICE_NOPE        16
`define GLM52_SLICE_ROPE        16
`define GLM52_SLICE_V_DIM       32
`define GLM52_SLICE_Q_LORA      64
`define GLM52_SLICE_KV_LORA     32
`define GLM52_SLICE_S_MAX       8
`define GLM52_SLICE_TOPK_ATTN   8
`define GLM52_SLICE_N_EXPERT    8
`define GLM52_SLICE_TOPK        2
`define GLM52_SLICE_INTER_MOE   64
`define GLM52_SLICE_INTER_DENSE 256

// ============================================================================
// STRUCTURAL CAVEAT -- S_MAX (do NOT set to the real 1M context here)
//   mla_attn_fp8 sizes its scores / probs / vstore attention scratch by S_MAX.
//   A full-context S_MAX (1M) would make that scratch (and elaboration) explode.
//   Decoupling attention scratch from S_MAX (a windowed SWIN pass) is task B7,
//   NOT part of this config header.  Full-config integration keeps S_MAX small
//   (the latent-ring depth), independent of the 1M POSW position field.
// ============================================================================
`define GLM52_S_MAX       8         // KEEP SMALL (B7 caveat)  (slice 8)

`endif // FULL_GLM52_VH
