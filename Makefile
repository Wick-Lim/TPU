# TPU v2.0 -- Verilog build / simulation / lint / synth
#
#   make build      -> compile the design + system TB into build/tpu_sim
#   make test       -> build + run the system integration TB (ALL N TESTS PASSED)
#   make hazard     -> build + run the hazard/pipeline TB
#   make unittests  -> build + run EVERY per-unit TB (ALL N TESTS PASSED each)
#   make sim        -> alias for `test`
#   make wave       -> run the system TB and leave ./tpu_waveform.vcd (real VCD)
#   make lint       -> verilator --lint-only -Wall on the whole design (clean)
#   make synth      -> yosys elaborate/synth gate (no error, no inferred latch)
#   make all        -> test + hazard + unittests + lint + synth
#   make clean      -> remove build artifacts and the generated VCDs

IVERILOG  ?= iverilog
VVP       ?= vvp
VERILATOR ?= verilator
YOSYS     ?= yosys

BUILD_DIR  := build
SIM_BIN    := $(BUILD_DIR)/tpu_sim
HAZARD_BIN := $(BUILD_DIR)/hazard_sim
AXI_BIN    := $(BUILD_DIR)/axi_sim
SOC_BIN    := $(BUILD_DIR)/soc_sim

# ---- v2.0 design source set (file name == module name, except top TPU) ----
DESIGN := \
	src/tpu_top.v \
	src/instruction_decoder.v \
	src/register_file.v \
	src/memory.v \
	src/tile_memory.v \
	src/vector_alu.v \
	src/dma_controller.v \
	src/scatter_gather.v \
	src/gemm_systolic.v \
	src/conv2d_unit.v \
	src/softmax_unit.v \
	src/attention_unit.v \
	src/fused_ops_unit.v

# ---- AXI4-Lite wrapper source: the slave wrapper + the full unchanged core. ----
AXI_DESIGN := src/tpu_axi.v $(DESIGN)
# ---- Two-clock SoC: AXI master DMA + async CDC FIFOs + the unchanged core. ----
SOC_DESIGN := src/tpu_soc.v src/axi_master_dma.v src/cdc_async_fifo.v $(DESIGN)

# ---- per-unit TBs.  Each builds against its own module (file==module); the
#      attention TB also needs softmax_unit (it instantiates the real unit). ----
UNITS := instruction_decoder register_file memory tile_memory vector_alu \
         dma_controller scatter_gather gemm_systolic conv2d_unit softmax_unit \
         fused_ops_unit attention_unit

IFLAGS := -g2012 -Wall -I src

.PHONY: all build test hazard axi soc unittests cache-study formal bitacc sim wave lint synth ppa clean

all: test hazard unittests lint synth formal

build: $(SIM_BIN)

$(SIM_BIN): test/tpu_tb.v $(DESIGN) src/tpu_defs.vh
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(IFLAGS) -o $(SIM_BIN) test/tpu_tb.v $(DESIGN)

$(HAZARD_BIN): test/hazard_tb.v $(DESIGN) src/tpu_defs.vh
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(IFLAGS) -o $(HAZARD_BIN) test/hazard_tb.v $(DESIGN)

test sim: $(SIM_BIN)
	$(VVP) $(SIM_BIN)

hazard: $(HAZARD_BIN)
	$(VVP) $(HAZARD_BIN)

# AXI4-Lite BFM testbench: drive the tpu_axi slave wrapper (around the unchanged
# TPU core) over the AW/W/B/AR/R channels and check a real program through the
# bus against independent goldens.  Builds tpu_axi_tb + the wrapper + full core.
$(AXI_BIN): test/tpu_axi_tb.v $(AXI_DESIGN) src/tpu_defs.vh
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(IFLAGS) -o $(AXI_BIN) test/tpu_axi_tb.v $(AXI_DESIGN)

axi: $(AXI_BIN)
	$(VVP) $(AXI_BIN)

# Two-clock SoC end-to-end TB: external-memory BFM -> AXI master DMA -> instr CDC
# -> core (autonomous program execution) -> result CDC -> AXI slave readback, on
# two asynchronous clocks.
$(SOC_BIN): test/tpu_soc_tb.v $(SOC_DESIGN) src/tpu_defs.vh
	@mkdir -p $(BUILD_DIR)
	$(IVERILOG) $(IFLAGS) -o $(SOC_BIN) test/tpu_soc_tb.v $(SOC_DESIGN)

soc: $(SOC_BIN)
	$(VVP) $(SOC_BIN)

# Build + run every per-unit TB.  attention_unit additionally needs softmax_unit.
unittests:
	@mkdir -p $(BUILD_DIR)
	@set -e; for u in $(UNITS); do \
	  extra=""; \
	  if [ "$$u" = "attention_unit" ]; then extra="src/softmax_unit.v"; fi; \
	  $(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/$${u}_sim test/$${u}_tb.v src/$$u.v $$extra; \
	  printf '[%s] ' "$$u"; $(VVP) $(BUILD_DIR)/$${u}_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: $$u"; exit 1; }; \
	done
	@# 2nd-size proof for the parameterized conv2d_unit (IMG_H=IMG_W=6,K=3).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/conv2d_param_sim test/conv2d_param_tb.v src/conv2d_unit.v
	@printf '[%s] ' "conv2d_param"; $(VVP) $(BUILD_DIR)/conv2d_param_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: conv2d_param"; exit 1; }
	@# EVEN-K proof for conv2d_unit (IMG_H=4,IMG_W=6,K=2): guards the pad=1 output-
	@# pixel counter-width fix (NPIX_MAX = OH+2*PAD_MAX, not IMG_H*IMG_W).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/conv2d_evenk_sim test/conv2d_evenk_tb.v src/conv2d_unit.v
	@printf '[%s] ' "conv2d_evenk"; $(VVP) $(BUILD_DIR)/conv2d_evenk_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: conv2d_evenk"; exit 1; }
	@# MULTI-LINE TM proof: gemm_ml at N=8 packs each matrix row across 2 TM lines
	@# (tiles beyond LINE_LANES=4), bit-exact vs an independent golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/gemm_ml_sim test/gemm_ml_tb.v src/gemm_ml.v
	@printf '[%s] ' "gemm_ml"; $(VVP) $(BUILD_DIR)/gemm_ml_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: gemm_ml"; exit 1; }
	@# ---- GLM-5.2 datapath units (bf16/fp32, fp32-golden verified) ----
	@# rmsnorm_unit: bf16 in/out, fp32 reduce + rsqrt; the FP numerics foundation.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rmsnorm_unit_sim test/rmsnorm_unit_tb.v src/rmsnorm_unit.v
	@printf '[%s] ' "rmsnorm_unit"; $(VVP) $(BUILD_DIR)/rmsnorm_unit_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rmsnorm_unit"; exit 1; }
	@# topk_select: top-K of N fp32 scores (DSA top-2048, MoE router top-8), ref-sort golden.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/topk_select_sim test/topk_select_tb.v src/topk_select.v
	@printf '[%s] ' "topk_select"; $(VVP) $(BUILD_DIR)/topk_select_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: topk_select"; exit 1; }
	@# glm_matmul: bf16 x bf16 -> fp32-accum -> bf16 GEMM workhorse (QKV/O/FFN/experts/LM head).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_sim test/glm_matmul_tb.v src/glm_matmul.v
	@printf '[%s] ' "glm_matmul"; $(VVP) $(BUILD_DIR)/glm_matmul_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul"; exit 1; }
	@# glm_act: bf16 sigmoid + silu (MoE router gating + SwiGLU experts), fp32 internal.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_act_sim test/glm_act_tb.v src/glm_act.v
	@printf '[%s] ' "glm_act"; $(VVP) $(BUILD_DIR)/glm_act_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_act"; exit 1; }
	@# rope_interleave_unit: decoupled interleaved RoPE (MLA q_rope/k_rope, DSA indexer),
	@# fp32 angle table theta=8e6 to 1M positions. (slow TB: full position sweep.)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/rope_sim test/rope_interleave_unit_tb.v src/rope_interleave_unit.v
	@printf '[%s] ' "rope_interleave_unit"; $(VVP) $(BUILD_DIR)/rope_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: rope_interleave_unit"; exit 1; }
	@# glm_fp_pipe: pipelined FP modules (mul/add/mac/rsqrt/exp), bit-exact vs glm_fp.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp_pipe_sim test/glm_fp_pipe_tb.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp_pipe"; $(VVP) $(BUILD_DIR)/glm_fp_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp_pipe"; exit 1; }
	@# glm_matmul_pipe: high-fmax bf16 GEMM on pipelined MACs (L-way interleaved accumulate).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_pipe_sim test/glm_matmul_pipe_tb.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_matmul_pipe"; $(VVP) $(BUILD_DIR)/glm_matmul_pipe_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_pipe"; exit 1; }
	@# fp8_e4m3: GLM-5.2-FP8 E4M3 primitives (decode/encode-RNE/mul), exhaustive 256x256 vs fp64.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/fp8_e4m3_sim test/fp8_e4m3_tb.v
	@printf '[%s] ' "fp8_e4m3"; $(VVP) $(BUILD_DIR)/fp8_e4m3_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: fp8_e4m3"; exit 1; }
	@# glm_matmul_fp8: FP8 E4M3 GEMM (4x4 mantissa mult + fp32 accumulate + block scale -> bf16).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_matmul_fp8_sim test/glm_matmul_fp8_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_matmul_fp8"; $(VVP) $(BUILD_DIR)/glm_matmul_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_matmul_fp8"; exit 1; }
	@# swiglu_expert_fp8: FP8 SwiGLU expert (gate/up/down via glm_matmul_fp8 + bf16 silu*up tail).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_fp8_sim test/swiglu_expert_fp8_tb.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert_fp8"; $(VVP) $(BUILD_DIR)/swiglu_expert_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_fp8"; exit 1; }
	@# swiglu_expert_fp8 PE_M batch: B rows per ONE weight fetch (== B single runs, bit-exact) -- the batching keystone.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_fp8_pem_sim test/swiglu_expert_fp8_pem_tb.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert_fp8(PE_M batch)"; $(VVP) $(BUILD_DIR)/swiglu_expert_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert_fp8_pem"; exit 1; }
	@# mla_attn_fp8: MLA attention, 7 weight projections via glm_matmul_fp8, bf16 attention/rope/norm/softmax/dsa.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_sim test/mla_attn_fp8_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8"; exit 1; }
	@# moe_router_fp8: MoE router, W_g GEMV via glm_matmul_fp8, bf16 sigmoid/topk/renorm*2.5 tail.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_fp8_sim test/moe_router_fp8_tb.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_fp8"; $(VVP) $(BUILD_DIR)/moe_router_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_fp8"; exit 1; }
	@# PE_M batch: moe_router_fp8 + mla_attn_fp8 process B rows per ONE weight fetch (== B single runs, bit-exact).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_fp8_pem_sim test/moe_router_fp8_pem_tb.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router_fp8(PE_M)"; $(VVP) $(BUILD_DIR)/moe_router_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router_fp8_pem"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_pem_sim test/mla_attn_fp8_pem_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(PE_M)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_pem"; exit 1; }
	@# glm_decoder_block_fp8: one full GLM-5.2-FP8 decoder layer (mla_attn_fp8 + moe_router_fp8 + swiglu_expert_fp8).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_fp8_sim test/glm_decoder_block_fp8_tb.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block_fp8"; $(VVP) $(BUILD_DIR)/glm_decoder_block_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block_fp8"; exit 1; }
	@# glm_model_fp8: FULL GLM-5.2-FP8 forward pass (embed -> L FP8 layers -> norm -> LM head -> next-token).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_sim test/glm_model_fp8_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8"; $(VVP) $(BUILD_DIR)/glm_model_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8"; exit 1; }
	@# mtp_head_fp8: FP8 multi-token-prediction head (W_proj+decoder_block_fp8 FP8, bf16 LM head + norms).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_fp8_sim test/mtp_head_fp8_tb.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head_fp8"; $(VVP) $(BUILD_DIR)/mtp_head_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head_fp8"; exit 1; }
	@# spec_decode_top: MTP speculative-decode loop (glm_model_fp8 + mtp_head_fp8 + spec_decode_seq); spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_top_sim test/spec_decode_top_tb.v src/spec_decode_top.v src/glm_model_fp8.v src/mtp_head_fp8.v src/spec_decode_seq.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_decode_top"; $(VVP) $(BUILD_DIR)/spec_decode_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_top"; exit 1; }
	@# glm_fp8_soc: TOP-LEVEL SoC -- compute die + expert_cache_pf + kv_cache_pager + Flash arbiter == standalone model.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_soc_sim test/glm_fp8_soc_tb.v src/glm_fp8_soc.v src/glm_model_fp8.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_soc"; $(VVP) $(BUILD_DIR)/glm_fp8_soc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_soc"; exit 1; }
	@# glm_fp8_system: PRODUCTION top -- compute + ddr5_xbar (multichannel) + weight_loader DMA + cache + pager == standalone.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_sim test/glm_fp8_system_tb.v src/glm_fp8_system.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system"; $(VVP) $(BUILD_DIR)/glm_fp8_system_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system"; exit 1; }
	@# glm_fp8_system_cdc: 2-clock wrapper (host/USB <-> compute via cdc_async_fifo) == standalone across async clocks.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_cdc_sim test/glm_fp8_system_cdc_tb.v src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system_cdc"; $(VVP) $(BUILD_DIR)/glm_fp8_system_cdc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system_cdc"; exit 1; }
	@# expert_cache_ctrl: MoE expert-weight HBM cache controller (tag/LRU/miss-DMA), single-package system PoC.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_ctrl_sim test/expert_cache_ctrl_tb.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_ctrl"; $(VVP) $(BUILD_DIR)/expert_cache_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_cache_ctrl"; exit 1; }
	@# expert_predictor: per-(layer,expert) frequency/locality prefetch predictor w/ confidence threshold (fine-grained MoE).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_predictor_sim test/expert_predictor_tb.v src/expert_predictor.v
	@printf '[%s] ' "expert_predictor"; $(VVP) $(BUILD_DIR)/expert_predictor_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_predictor"; exit 1; }
	@# spec_decode_seq: MTP speculative-decode controller (draft/verify/accept-reject; eff tok/pass = 1+alpha).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_sim test/spec_decode_seq_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq"; $(VVP) $(BUILD_DIR)/spec_decode_seq_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq"; exit 1; }
	@# spec_decode_seq K>1: multi-token draft (DRAFT_K=1/2/3), spec==greedy exact + eff-tok/pass vs alpha.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_seq_k_sim test/spec_decode_seq_k_tb.v src/spec_decode_seq.v
	@printf '[%s] ' "spec_decode_seq(K>1)"; $(VVP) $(BUILD_DIR)/spec_decode_seq_k_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_seq_k"; exit 1; }
	@# kv_cache_pager: MLA latent-KV ring cache (append + DSA-gather + Flash overflow), single-module system.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_cache_pager_sim test/kv_cache_pager_tb.v src/kv_cache_pager.v
	@printf '[%s] ' "kv_cache_pager"; $(VVP) $(BUILD_DIR)/kv_cache_pager_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_cache_pager"; exit 1; }
	@# ddr5_xbar: N-channel banked DDR5 read fabric (address striping -> ~Nx aggregate bandwidth).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ddr5_xbar_sim test/ddr5_xbar_tb.v src/ddr5_xbar.v
	@printf '[%s] ' "ddr5_xbar"; $(VVP) $(BUILD_DIR)/ddr5_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ddr5_xbar"; exit 1; }
	@# flash_xbar: N-channel banked Flash read fabric -- deep per-channel outstanding queue hides NAND latency (~QDEPTH x), banking ~N x.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/flash_xbar_sim test/flash_xbar_tb.v src/flash_xbar.v
	@printf '[%s] ' "flash_xbar"; $(VVP) $(BUILD_DIR)/flash_xbar_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: flash_xbar"; exit 1; }
	@# weight_loader: checkpoint FP8+block-scale memory image -> glm_matmul_fp8 pull stream (loader-fed == direct-fed).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_loader_sim test/weight_loader_tb.v src/weight_loader.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "weight_loader"; $(VVP) $(BUILD_DIR)/weight_loader_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: weight_loader"; exit 1; }
	@# boot_loader: power-up Flash->DDR5 resident-set (hot weights) DMA + ready handshake (chip must load model first).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/boot_loader_sim test/boot_loader_tb.v src/boot_loader.v
	@printf '[%s] ' "boot_loader"; $(VVP) $(BUILD_DIR)/boot_loader_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: boot_loader"; exit 1; }
	@# clk_en_ctrl: work-driven clock-enable gating (die idles ~75% Flash-bound -> ~73% idle-power gated; never gates active work).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_en_ctrl_sim test/clk_en_ctrl_tb.v src/clk_en_ctrl.v
	@printf '[%s] ' "clk_en_ctrl"; $(VVP) $(BUILD_DIR)/clk_en_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_en_ctrl"; exit 1; }
	@# swiglu_expert: SwiGLU FFN expert (gate/up/down GEMM + silu*up), dense + MoE modes.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/swiglu_expert_sim test/swiglu_expert_tb.v src/swiglu_expert.v src/glm_matmul_pipe.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "swiglu_expert"; $(VVP) $(BUILD_DIR)/swiglu_expert_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: swiglu_expert"; exit 1; }
	@# moe_router: GEMV + sigmoid + top-K + renormalize-then-scale (GLM-5.2 MoE gating).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/moe_router_sim test/moe_router_tb.v src/moe_router.v src/glm_matmul_pipe.v src/glm_act.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "moe_router"; $(VVP) $(BUILD_DIR)/moe_router_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: moe_router"; exit 1; }
	@# glm_softmax: numerically-stable bf16 softmax (MLA attention), full-denominator sum.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_softmax_sim test/glm_softmax_tb.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_softmax"; $(VVP) $(BUILD_DIR)/glm_softmax_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_softmax"; exit 1; }
	@# dsa_indexer: DSA/IndexShare sparse-attention indexer (index-score + top-K + dense fallback).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/dsa_indexer_sim test/dsa_indexer_tb.v src/dsa_indexer.v src/topk_select.v src/glm_fp_pipe.v
	@printf '[%s] ' "dsa_indexer"; $(VVP) $(BUILD_DIR)/dsa_indexer_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: dsa_indexer"; exit 1; }
	@# mla_attn: MLA latent attention orchestrator (low-rank Q/KV + RoPE + DSA + softmax + *V + O proj).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_sim test/mla_attn_tb.v src/mla_attn.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn"; $(VVP) $(BUILD_DIR)/mla_attn_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn"; exit 1; }
	@# sampler: temperature + top-k/top-p + softmax + multinomial(LFSR) token sampling.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/sampler_sim test/sampler_tb.v src/sampler.v src/topk_select.v src/glm_softmax.v src/glm_fp_pipe.v
	@printf '[%s] ' "sampler"; $(VVP) $(BUILD_DIR)/sampler_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: sampler"; exit 1; }
	@# glm_decoder_block: ONE full GLM-5.2 decoder layer (rmsnorm+mla_attn+residual+rmsnorm+FFN+residual).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_sim test/glm_decoder_block_tb.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block"; $(VVP) $(BUILD_DIR)/glm_decoder_block_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block"; exit 1; }
	@# glm_model: FULL GLM-5.2 forward pass (embed -> 6 layers dense/MoE -> norm -> LM head -> next-token logits).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_sim test/glm_model_tb.v src/glm_model.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model"; $(VVP) $(BUILD_DIR)/glm_model_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model"; exit 1; }
	@# mtp_head: GLM-5.2 multi-token-prediction head (t+2 speculative; num_nextn_predict_layers=1).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_sim test/mtp_head_tb.v src/mtp_head.v src/glm_decoder_block.v src/rmsnorm_unit.v src/mla_attn.v src/rope_interleave_unit.v src/glm_matmul_pipe.v src/dsa_indexer.v src/glm_softmax.v src/topk_select.v src/glm_act.v src/swiglu_expert.v src/moe_router.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head"; $(VVP) $(BUILD_DIR)/mtp_head_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head"; exit 1; }
	@# 2nd-size proof for the parameterized attention_unit (SEQ=2,D=2); it
	@# additionally needs softmax_unit (instantiated at SM_PAD lanes).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/attention_param_sim test/attention_param_tb.v src/attention_unit.v src/softmax_unit.v
	@printf '[%s] ' "attention_param"; $(VVP) $(BUILD_DIR)/attention_param_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: attention_param"; exit 1; }
	@# 2nd-size proof for the parameterized softmax_unit (LEN=4 and LEN=16).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/softmax_param_sim test/softmax_param_tb.v src/softmax_unit.v
	@printf '[%s] ' "softmax_param"; $(VVP) $(BUILD_DIR)/softmax_param_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: softmax_param"; exit 1; }
	@# AXI4-Lite BFM testbench: drive the tpu_axi slave wrapper over the bus.
	@$(IVERILOG) $(IFLAGS) -o $(AXI_BIN) test/tpu_axi_tb.v $(AXI_DESIGN)
	@printf '[%s] ' "tpu_axi"; $(VVP) $(AXI_BIN) | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: tpu_axi"; exit 1; }
	@# AXI4-Lite MASTER DMA engine vs an AXI slave-memory BFM.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/axi_master_dma_sim test/axi_master_dma_tb.v src/axi_master_dma.v
	@printf '[%s] ' "axi_master_dma"; $(VVP) $(BUILD_DIR)/axi_master_dma_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: axi_master_dma"; exit 1; }
	@# Async CDC FIFO across two unrelated clocks (7ns vs 11ns).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/cdc_async_fifo_sim test/cdc_async_fifo_tb.v src/cdc_async_fifo.v
	@printf '[%s] ' "cdc_async_fifo"; $(VVP) $(BUILD_DIR)/cdc_async_fifo_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: cdc_async_fifo"; exit 1; }
	@# Two-clock SoC: autonomous program fetch (AXI master DMA) + CDC + core execution.
	@$(IVERILOG) $(IFLAGS) -o $(SOC_BIN) test/tpu_soc_tb.v $(SOC_DESIGN)
	@printf '[%s] ' "tpu_soc"; $(VVP) $(SOC_BIN) | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: tpu_soc"; exit 1; }
	@echo "unittests: all per-unit TBs passed"

# Single-package system study: regenerate the calibrated GLM-scale routing traces, then run
# the cache hit-rate + batching + prefetch sims through the real RTL (these need the generated
# tools/*.hex, so they are kept out of the self-contained `unittests`). See docs/SYSTEM_SINGLE_PACKAGE.md.
cache-study:
	@python3 tools/route_trace.py --dump >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_cache_batch tools/glm_cache_batch_tb.v src/expert_cache_ctrl.v
	@printf '[%s] ' "cache-batch"; $(VVP) $(BUILD_DIR)/glm_cache_batch | grep -E 'ALL [0-9]+ TESTS PASSED|BATCHING LEVER' \
	    || { echo "FAILED: cache-batch"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf test/expert_cache_pf_tb.v src/expert_cache_pf.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_cache_pf"; $(VVP) $(BUILD_DIR)/expert_cache_pf | grep -E 'ALL [0-9]+ TESTS PASSED|stall cut' \
	    || { echo "FAILED: expert_cache_pf"; exit 1; }
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_cache_pf_policy test/expert_cache_pf_policy_tb.v src/expert_cache_pf.v
	@printf '[%s] ' "cache-policy(LRU vs FREQ)"; $(VVP) $(BUILD_DIR)/expert_cache_pf_policy | grep -E 'ALL [0-9]+ TESTS PASSED|hit-rate|POLICY' \
	    || { echo "FAILED: expert_cache_pf_policy"; exit 1; }
	@# weight_decomp: lossless FP8-expert decompressor (Flash bytes down -> effective Flash_BW + energy). Needs the python vector.
	@python3 tools/fp8_gen.py gen scratchpad/wd_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_decomp test/weight_decomp_tb.v src/weight_decomp.v
	@printf '[%s] ' "weight_decomp"; $(VVP) $(BUILD_DIR)/weight_decomp | grep -E 'ALL [0-9]+ TESTS PASSED|RATIO' \
	    || { echo "FAILED: weight_decomp"; exit 1; }
	@# weight_decomp2: context-modeled (order-1) lossless FP8 decompressor -- 1.42x on locality-bearing weights (1.13x over order-0).
	@python3 tools/fp8_ctxpack.py gen scratchpad/wd2_vec.txt >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/weight_decomp2 test/weight_decomp2_tb.v src/weight_decomp2.v
	@printf '[%s] ' "weight_decomp2(ctx)"; $(VVP) $(BUILD_DIR)/weight_decomp2 | grep -E 'ALL [0-9]+ TESTS PASSED|RATIO' \
	    || { echo "FAILED: weight_decomp2"; exit 1; }
	@# expert_prefetch_top: predictor-driven deep prefetch -- MEASURED no-op at real cache size (see docs/IMPROVEMENT_PLAN.md 2.3).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/expert_prefetch_top test/expert_prefetch_top_tb.v src/expert_prefetch_top.v src/expert_predictor.v src/expert_cache_pf.v src/expert_cache_ctrl.v
	@printf '[%s] ' "expert_prefetch_top"; $(VVP) $(BUILD_DIR)/expert_prefetch_top | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: expert_prefetch_top"; exit 1; }
	@# flash_layout: offline expert->Flash-channel placement so flash_xbar's stripe spreads a token's top-8 (~39%->55% of 8x peak BW).
	@printf '[%s] ' "flash_layout"; python3 tools/flash_layout.py | grep -E 'OPTIMIZED|peakBW' | head -2 \
	    || { echo "FAILED: flash_layout"; exit 1; }
	@echo "cache-study: batching + prefetch + policy + decomp + predictor-prefetch + flash-layout sims passed (see docs/SYSTEM_SINGLE_PACKAGE.md)"

# Formal (bounded model checking) of the memory-system controllers via yosys write_smt2 +
# yosys-smtbmc -s z3.  Each harness test/formal/<dut>_fv.v instantiates the committed controller
# read-only and asserts safety properties.  The mandatory `async2sync; chformal -lower` lowers
# yosys $check cells so the asserts are NOT silently dropped (a vacuous-pass trap); the assert
# count > 0 guard re-checks non-vacuity per model.  See docs/FORMAL.md.  Bounds kept modest for a
# routine run; deeper bounds (e.g. expert_cache_pf K=55) are in docs/FORMAL.md.
FV_DIR := scratchpad
define run_bmc   # $(1)=dut name  $(2)=extra read deps  $(3)=extra yosys (e.g. chparam)  $(4)=bound K
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v $(2) test/formal/$(1)_fv.v; $(3) \
	          prep -top $(1)_fv -flatten; memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(1)_fv.smt2" > $(FV_DIR)/$(1)_fv_build.log 2>&1 \
	    || { echo "FAILED(build): $(1)"; cat $(FV_DIR)/$(1)_fv_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(1)_fv.smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions in smt2): $(1)"; exit 1; }
	@yosys-smtbmc -s z3 -t $(4) $(FV_DIR)/$(1)_fv.smt2 > $(FV_DIR)/$(1)_fv_bmc.log 2>&1 \
	    && printf '[formal %-16s] PASSED  K=%s  (%s asserts)\n' "$(1)" "$(4)" "`grep -ic assert $(FV_DIR)/$(1)_fv.smt2`" \
	    || { echo "FAILED(BMC counterexample): $(1)"; tail -20 $(FV_DIR)/$(1)_fv_bmc.log; exit 1; }
endef

formal:
	$(call run_bmc,ddr5_xbar,,,12)
	$(call run_bmc,flash_xbar,,,12)
	$(call run_bmc,boot_loader,,,16)
	$(call run_bmc,spec_decode_seq,,,20)
	$(call run_bmc,kv_cache_pager,,,16)
	$(call run_bmc,expert_cache_pf,src/expert_cache_ctrl.v,chparam -set PF_ENABLE 0 expert_cache_pf_fv;,20)
	@echo "formal: all 5 controllers BMC-proven (no counterexample); see docs/FORMAL.md for bounds + coverage"

# Real-checkpoint bit-accuracy: prove glm_matmul_fp8 computes the GLM-5.2-FP8 FP8 contract exactly
# as the real inference engine (argmax-preserving vs fp32-accumulate + float64 ground truth), and
# that the HF safetensors -> RTL bridge round-trips.  See docs/BIT_ACCURACY.md (+ the full-model GPU procedure).
bitacc:
	@python3 tools/glm_fp8_ref.py | grep -qE 'SELFTEST PASS' || { echo "FAILED: glm_fp8_ref"; exit 1; }
	@printf '[%s] ' "glm_fp8_ref"; echo "SELFTEST PASS"
	@python3 tools/ckpt_pack.py | grep -qE 'SELFTEST PASS' || { echo "FAILED: ckpt_pack"; exit 1; }
	@printf '[%s] ' "ckpt_pack(HF safetensors->RTL)"; echo "round-trip PASS"
	@python3 test/bit_accuracy_gen.py >/dev/null
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/bit_accuracy test/bit_accuracy_tb.v src/glm_matmul_fp8.v src/glm_fp_pipe.v
	@printf '[%s] ' "bit_accuracy(RTL==real-contract)"; $(VVP) $(BUILD_DIR)/bit_accuracy | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: bit_accuracy"; exit 1; }
	@python3 test/bit_accuracy_check.py | grep -E 'ARGMAX-PRESERVED' || { echo "FAILED: bit_accuracy_check"; exit 1; }
	@echo "bitacc: glm_matmul_fp8 == real GLM-5.2-FP8 FP8 contract, argmax preserved (see docs/BIT_ACCURACY.md)"

wave: $(SIM_BIN)
	$(VVP) $(SIM_BIN)
	@echo "VCD written to ./tpu_waveform.vcd"

lint:
	$(VERILATOR) --lint-only -Wall -Isrc --top-module TPU $(DESIGN)

# Yosys synthesis gate: elaborate the whole hierarchy, run proc/opt, then
# `check -assert` which FAILS (non-zero exit) on any structural problem
# (unresolved hierarchy, combinational loop, multiple drivers).  `stat` prints
# the gate-level cell count.  `-q` keeps the log quiet; a non-zero exit fails make.
synth:
	$(YOSYS) -q -p "read_verilog -sv -I src $(DESIGN); \
	                hierarchy -top TPU -check; proc; opt; check -assert; stat"

# ---- PPA (area + timing) via yosys synth_ecp5 -----------------------------
# For each tensor unit and the full TPU top we map to real Lattice ECP5 FPGA
# primitives with `synth_ecp5` and report:
#   AREA  -- `stat` prints LUT4, TRELLIS_FF (FF), CCU2C (carry), MULT18X18D
#            (DSP), DP16KD (block RAM), L6MUX21/PFUMX (muxes).
#   TIMING-- `ltp` prints "Longest topological path ... length N" = the
#            combinational logic-cell DEPTH (critical-path depth).  Routed fmax
#            needs nextpnr-ecp5 (not installed here) so we report logic depth.
PPA_DIR := $(BUILD_DIR)/ppa

PPA_SRC_gemm_systolic := src/gemm_systolic.v
PPA_SRC_conv2d_unit   := src/conv2d_unit.v
PPA_SRC_softmax_unit  := src/softmax_unit.v
PPA_SRC_attention_unit:= src/attention_unit.v src/softmax_unit.v
PPA_SRC_TPU           := $(DESIGN)

PPA_UNITS := gemm_systolic conv2d_unit softmax_unit attention_unit TPU

# NOTE: `ltp` walks every combinational path; on the full flattened TPU it emits
# millions of "Detected loop" traversal lines (carry-chain artifacts, NOT real
# loops) and runs ~15 min, so we run ltp ONLY on the four tensor units (the
# actionable pipeline candidates) and report AREA-ONLY for the full TPU top.
# Output is FILTERED before being written (grep > log, not tee raw), so each log
# stays a few hundred bytes instead of gigabytes.
ppa:
	@mkdir -p $(PPA_DIR)
	@for m in $(PPA_UNITS); do \
	  case $$m in \
	    gemm_systolic)  src="$(PPA_SRC_gemm_systolic)";  cmds="synth_ecp5 -top $$m; stat; ltp";; \
	    conv2d_unit)    src="$(PPA_SRC_conv2d_unit)";    cmds="synth_ecp5 -top $$m; stat; ltp";; \
	    softmax_unit)   src="$(PPA_SRC_softmax_unit)";   cmds="synth_ecp5 -top $$m; stat; ltp";; \
	    attention_unit) src="$(PPA_SRC_attention_unit)"; cmds="synth_ecp5 -top $$m; stat; ltp";; \
	    TPU)            src="$(PPA_SRC_TPU)";            cmds="synth_ecp5 -top $$m; stat";; \
	  esac; \
	  echo "================================================================"; \
	  echo "== PPA: synth_ecp5 top=$$m"; \
	  echo "================================================================"; \
	  $(YOSYS) -p "read_verilog -sv -I src $$src; $$cmds" 2>&1 \
	    | grep -E '^[[:space:]]*[0-9]+[[:space:]]+(LUT4|TRELLIS_FF|CCU2C|MULT18X18D|DP16KD|L6MUX21|PFUMX|TRELLIS_IO)[[:space:]]*$$|^[[:space:]]*Longest topological path.*length|Number of cells:|^ERROR|Error:' \
	    > $(PPA_DIR)/$$m.log; \
	  if [ ! -s $(PPA_DIR)/$$m.log ]; then echo "PPA FAILED for $$m (no synth output)"; exit 1; fi; \
	  cat $(PPA_DIR)/$$m.log; \
	done
	@echo "ppa: synth_ecp5 area (+ltp for the 4 tensor units) captured (logs in $(PPA_DIR)/)"

clean:
	rm -rf $(BUILD_DIR) tpu_waveform.vcd hazard_waveform.vcd
	rm -f *.vcd
