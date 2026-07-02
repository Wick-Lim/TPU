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

.PHONY: all build test hazard axi soc unittests cache-study formal formal-ind bitacc sim wave lint synth synth-glm cdc ppa clean

all: test hazard unittests lint synth synth-glm formal

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
	@# expert_predictor_tb reads the generated routing trace (tools/glm_trace.hex);
	@# regenerate it so `unittests` is self-contained on a fresh clone (deterministic seed).
	@python3 tools/route_trace.py --dump >/dev/null
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
	@# mla_attn_fp8 per-position: each PE_M row at its OWN query position (per-row RoPE) == its single-token run (product-grade batched decode).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_ppos_sim test/mla_attn_fp8_ppos_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(per-position)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_ppos_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_ppos"; exit 1; }
	@# mla_attn_fp8 per-row s_len: each PE_M row at its OWN causal extent (score mask before softmax) == its single-token run at that s_len (shared KV prefix).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_pslen_sim test/mla_attn_fp8_pslen_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(per-slen)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_pslen_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_pslen"; exit 1; }
	@# mla_attn_fp8 sparse+per-row oracle (task B2): batched row === PE_M=1 reference on that row's
	@# own (x,pos,s_len) with S_MAX>TOPK, + fetch-sharing (one weight/kc fetch per distinct key).
	@# The distinct-x sparse compare is the pending-B6 oracle (guarded by SPARSE_XFAIL, default off).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mla_attn_fp8_sparse_perrow_sim test/mla_attn_fp8_sparse_perrow_tb.v src/mla_attn_fp8.v src/glm_matmul_fp8.v src/glm_matmul_pipe.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "mla_attn_fp8(sparse-perrow)"; $(VVP) $(BUILD_DIR)/mla_attn_fp8_sparse_perrow_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mla_attn_fp8_sparse_perrow"; exit 1; }
	@# glm_decoder_block_fp8: one full GLM-5.2-FP8 decoder layer (mla_attn_fp8 + moe_router_fp8 + swiglu_expert_fp8).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_decoder_block_fp8_sim test/glm_decoder_block_fp8_tb.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_decoder_block_fp8"; $(VVP) $(BUILD_DIR)/glm_decoder_block_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_decoder_block_fp8"; exit 1; }
	@# glm_model_fp8: FULL GLM-5.2-FP8 forward pass (embed -> L FP8 layers -> norm -> LM head -> next-token).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_sim test/glm_model_fp8_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8"; $(VVP) $(BUILD_DIR)/glm_model_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8"; exit 1; }
	@# glm_model_fp8 PE_M batch: the WHOLE forward pass processes B token-rows per ONE weight fetch (== B single runs, per-row argmax).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_model_fp8_pem_sim test/glm_model_fp8_pem_tb.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_model_fp8(PE_M batch)"; $(VVP) $(BUILD_DIR)/glm_model_fp8_pem_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_model_fp8_pem"; exit 1; }
	@# mtp_head_fp8: FP8 multi-token-prediction head (W_proj+decoder_block_fp8 FP8, bf16 LM head + norms).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mtp_head_fp8_sim test/mtp_head_fp8_tb.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/glm_fp_pipe.v
	@printf '[%s] ' "mtp_head_fp8"; $(VVP) $(BUILD_DIR)/mtp_head_fp8_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mtp_head_fp8"; exit 1; }
	@# spec_decode_top: MTP speculative-decode loop (glm_model_fp8 + mtp_head_fp8 + spec_decode_seq); spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_decode_top_sim test/spec_decode_top_tb.v src/spec_decode_top.v src/glm_model_fp8.v src/mtp_head_fp8.v src/spec_decode_seq.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_decode_top"; $(VVP) $(BUILD_DIR)/spec_decode_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_decode_top"; exit 1; }
	@# spec_batched_top: batched-verify -- K+1 draft positions verified in ONE PE_M=K+1 model weight-load (Flash verify /K+1); spec==greedy.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_batched_top_sim test/spec_batched_top_tb.v src/spec_batched_top.v src/glm_model_fp8.v src/spec_decode_seq.v src/mtp_head_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_batched_top"; $(VVP) $(BUILD_DIR)/spec_batched_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_batched_top"; exit 1; }
	@# spec_chain_top (task B8): K-step MTP chain with promoted verify/MTP/embed pull ports --
	@# committed stream == greedy rollout EXACT (spec==greedy safety). (accept-path K_eff coverage: followup.)
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/spec_chain_top_sim test/spec_chain_top_tb.v src/spec_chain_top.v src/mtp_head_fp8.v src/glm_model_fp8.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/spec_decode_seq.v src/glm_fp_pipe.v
	@printf '[%s] ' "spec_chain_top"; $(VVP) $(BUILD_DIR)/spec_chain_top_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: spec_chain_top"; exit 1; }
	@# glm_fp8_soc: TOP-LEVEL SoC -- compute die + expert_cache_pf + kv_cache_pager + Flash arbiter == standalone model.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_soc_sim test/glm_fp8_soc_tb.v src/glm_fp8_soc.v src/glm_model_fp8.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_soc"; $(VVP) $(BUILD_DIR)/glm_fp8_soc_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_soc"; exit 1; }
	@# glm_fp8_system: PRODUCTION top -- compute + ddr5_xbar (multichannel) + weight_loader DMA + cache + pager == standalone.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_sim test/glm_fp8_system_tb.v src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system"; $(VVP) $(BUILD_DIR)/glm_fp8_system_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system"; exit 1; }
	@# glm_fp8_system(DECOMP=1): weight_decomp (order-0 lossless FP8) in the Flash->loader refill path (task C9).
	@# Proves the die consumes DECOMPRESSED FP8 codes from a COMPRESSED backing image, token-identical + per-beat bit-exact.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_decomp_sim test/glm_fp8_system_decomp_tb.v src/glm_fp8_system.v src/weight_decomp.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
	@printf '[%s] ' "glm_fp8_system(decomp)"; $(VVP) $(BUILD_DIR)/glm_fp8_system_decomp_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: glm_fp8_system_decomp"; exit 1; }
	@# glm_fp8_system_cdc: 2-clock wrapper (host/USB <-> compute via cdc_async_fifo) == standalone across async clocks.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/glm_fp8_system_cdc_sim test/glm_fp8_system_cdc_tb.v src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v src/glm_fp_pipe.v
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
	@# ecc_secded: (72,64) SECDED ECC for the DDR5/Flash path -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_secded_sim test/ecc_secded_tb.v src/ecc_secded.v
	@printf '[%s] ' "ecc_secded"; $(VVP) $(BUILD_DIR)/ecc_secded_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_secded"; exit 1; }
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
	@# ---- P2 productization building blocks (ECC / reset-CDC / DFT-MBIST / power-ICG) ----
	@# ecc_mem_wrap: SECDED-protected synchronous RAM (encode on write, decode+correct/detect on read) -- exhaustive single-correct + double-detect.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/ecc_mem_wrap_sim test/ecc_mem_wrap_tb.v src/ecc_mem_wrap.v src/ecc_secded.v
	@printf '[%s] ' "ecc_mem_wrap"; $(VVP) $(BUILD_DIR)/ecc_mem_wrap_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: ecc_mem_wrap"; exit 1; }
	@# reset_sync: async-assert / sync-deassert reset synchronizer (CDC signoff) -- immediate assert, STAGES-edge clean deassert.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/reset_sync_sim test/reset_sync_tb.v src/reset_sync.v
	@printf '[%s] ' "reset_sync"; $(VVP) $(BUILD_DIR)/reset_sync_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: reset_sync"; exit 1; }
	@# mbist_ctrl: March C- memory BIST for a single-port SRAM (good RAM -> pass; stuck-at-0 cell -> fail + fail_addr).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/mbist_ctrl_sim test/mbist_ctrl_tb.v src/mbist_ctrl.v
	@printf '[%s] ' "mbist_ctrl"; $(VVP) $(BUILD_DIR)/mbist_ctrl_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: mbist_ctrl"; exit 1; }
	@# icg_cell: glitch-free integrated clock gate (low-phase enable latch + AND) -- turns clk_en into a real gated clock with no runt pulses.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/icg_cell_sim test/icg_cell_tb.v src/icg_cell.v
	@printf '[%s] ' "icg_cell"; $(VVP) $(BUILD_DIR)/icg_cell_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: icg_cell"; exit 1; }
	@# clk_gate_cluster (task C7): icg_cell + clk_en_ctrl gating a real leaf -- gated == free-running (bit-exact),
	@# idle => clock frozen, scan_enable => transparent, req=1 => enable=1 safety invariant.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/clk_gate_cluster_sim test/clk_gate_cluster_tb.v src/clk_gate_cluster.v src/icg_cell.v src/clk_en_ctrl.v src/register_file.v
	@printf '[%s] ' "clk_gate_cluster"; $(VVP) $(BUILD_DIR)/clk_gate_cluster_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: clk_gate_cluster"; exit 1; }
	@# kv_ecc_ring (task C6): lane-partitioned SECDED ring for wide (ragged, non-64-aligned) KV rows --
	@# single-bit corrected, double-bit detected, across the ragged final lane.
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/kv_ecc_ring_sim test/kv_ecc_ring_tb.v src/kv_ecc_ring.v src/ecc_secded.v
	@printf '[%s] ' "kv_ecc_ring"; $(VVP) $(BUILD_DIR)/kv_ecc_ring_sim | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: kv_ecc_ring"; exit 1; }
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
	@# batched_moe: expert-grouped batched MoE -- B tokens, union of experts fetched once each (aggregate-throughput lever).
	@$(IVERILOG) $(IFLAGS) -o $(BUILD_DIR)/batched_moe test/batched_moe_tb.v src/batched_moe.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
	@printf '[%s] ' "batched_moe"; $(VVP) $(BUILD_DIR)/batched_moe | grep -E 'ALL [0-9]+ TESTS PASSED' \
	    || { echo "FAILED: batched_moe"; exit 1; }
	@printf '[%s] ' "batched_moe(amortize)"; python3 tools/batched_moe_amortize.py | grep -E '256|amort' | head -2 \
	    || { echo "FAILED: batched_moe_amortize"; exit 1; }
	@echo "cache-study: batching + prefetch + policy + decomp + predictor-prefetch + flash-layout + batched-moe sims passed (see docs/SYSTEM_SINGLE_PACKAGE.md)"

# P1.3c: batched_moe FULL B-COVERAGE.  batched_moe_tb.v (in cache-study) proves the
# expert-grouped batch identity at ONE width B=4; this sweeps it across widths
# {1,2,3,5,8} and routing patterns {same,distinct,random,overlap}, re-proving per
# width that batched_moe(PE_M=B) == B independent PE_M=1 expert runs, BIT-EXACT, with
# every union expert fetched exactly once.  B is a compile-time port width, so the
# parametrized TB is recompiled per (B,PATTERN) via +define.  Kept SEPARATE from
# `unittests`/`all` because each iverilog elaboration of the nested MoE datapath is
# minutes-long (esp. B=8).  See test/batched_moe_bcov_tb.v.
BCOV_SRC := src/batched_moe.v src/swiglu_expert_fp8.v src/glm_matmul_fp8.v src/glm_act.v src/glm_fp_pipe.v
bcov:
	@mkdir -p $(BUILD_DIR)
	@set -e; for bp in 1:2:11 2:1:22 3:3:33 5:0:55 8:3:44; do \
	   B=$${bp%%:*}; rest=$${bp#*:}; P=$${rest%%:*}; S=$${rest##*:}; \
	   $(IVERILOG) $(IFLAGS) -DBCOV_B=$$B -DBCOV_PAT=$$P -DBCOV_SEED=$$S \
	     -o $(BUILD_DIR)/bcov_$$B test/batched_moe_bcov_tb.v $(BCOV_SRC); \
	   printf '[batched_moe B=%s] ' "$$B"; \
	   $(VVP) $(BUILD_DIR)/bcov_$$B | grep -E 'ALL [0-9]+ TESTS PASSED' \
	     || { echo "FAILED: batched_moe B=$$B"; exit 1; }; \
	 done
	@echo "bcov: batched_moe B-coverage passed for B in {1,2,3,5,8} (batched == per-row, bit-exact)"

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
	@mkdir -p $(FV_DIR)
	$(call run_bmc,ddr5_xbar,,,12)
	$(call run_bmc,flash_xbar,,,12)
	$(call run_bmc,boot_loader,,,16)
	$(call run_bmc,spec_decode_seq,,,20)
	$(call run_bmc,kv_cache_pager,,,16)
	$(call run_bmc,expert_cache_pf,src/expert_cache_ctrl.v,chparam -set PF_ENABLE 0 expert_cache_pf_fv;,20)
	@echo "formal: all 5 controllers BMC-proven (no counterexample); see docs/FORMAL.md for bounds + coverage"

# UNBOUNDED proof via temporal k-INDUCTION (yosys-smtbmc -i): base case + induction
# step => the asserts hold on ALL reachable states, not just the first K cycles.
# The step needs the design's reachable state space pinned; harnesses add
# STRENGTHENING INVARIANT asserts (over the DUT's primary I/O + harness shadow
# regs -- this yosys build has no internal observability) until the step closes.
define run_kind  # $(1)=dut name  $(2)=ind-harness basename  $(3)=K  $(4)=extra yosys (e.g. connect-bind)
	@yosys -p "read_verilog -sv -formal -I src src/$(1).v test/formal/$(2).v; \
	          prep -top $(2) -flatten; $(4) memory_map; async2sync; chformal -lower; \
	          write_smt2 -wires $(FV_DIR)/$(2).smt2" > $(FV_DIR)/$(2)_build.log 2>&1 \
	    || { echo "FAILED(build): $(2)"; cat $(FV_DIR)/$(2)_build.log; exit 1; }
	@test `grep -ic assert $(FV_DIR)/$(2).smt2` -gt 0 \
	    || { echo "FAILED(vacuous: 0 assertions): $(2)"; exit 1; }
	@yosys-smtbmc -s z3 -i -t $(3) $(FV_DIR)/$(2).smt2 > $(FV_DIR)/$(2)_kind.log 2>&1 \
	    && printf '[k-induction %-20s] PROVEN UNBOUNDED  K=%s  (%s asserts)\n' "$(2)" "$(3)" "`grep -ic assert $(FV_DIR)/$(2).smt2`" \
	    || { echo "FAILED(induction step): $(2)"; tail -20 $(FV_DIR)/$(2)_kind.log; exit 1; }
endef

# flash_xbar response-FIFO / outstanding proof needs the DUT's OWN per-channel counters
# (u_dut.outst[c], u_dut.cnt[c]) in the inductive hypothesis.  yosys 0.66 cannot reference
# them from Verilog (no hierarchical refs), so the harness declares `(* keep *)` UNDRIVEN
# probe wires and we wire them to the flattened DUT registers post-flatten with `connect`.
# The TRAILING SPACE before each `;` is load-bearing: it terminates the escaped bracketed
# id \u_dut.outst[0] (otherwise [0] is parsed as a bit-select of a non-existent wire).
FLASH_IND_CONN := connect -set \dut_outst0 \u_dut.outst[0] ; connect -set \dut_outst1 \u_dut.outst[1] ; connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
# ddr5_xbar response-FIFO proof (task C5) uses the same connect-bind trick to reach
# the DUT's per-channel response-FIFO occupancy counters cnt[0..N_CH-1] (N_CH=2 slice).
DDR5_IND_CONN := connect -set \dut_cnt0 \u_dut.cnt[0] ; connect -set \dut_cnt1 \u_dut.cnt[1] ;
formal-ind:
	@mkdir -p $(FV_DIR)
	$(call run_kind,boot_loader,boot_loader_ind_fv,8)
	$(call run_kind,kv_cache_pager,kv_cache_pager_ind_fv,16)
	$(call run_kind,spec_decode_seq,spec_decode_seq_ind_fv,2)
	$(call run_kind,ddr5_xbar,ddr5_xbar_ind_fv,12,$(DDR5_IND_CONN))
	$(call run_kind,flash_xbar,flash_xbar_ind_fv,3,$(FLASH_IND_CONN))
	@echo "formal-ind: boot_loader done-gate proven UNBOUNDED; kv_cache_pager append/gather in-bounds + window invariants proven UNBOUNDED; spec_decode_seq token-accounting equality + per-cycle modular increment bounds + step-form (non-decreasing-except-wrap) monotonicity proven UNBOUNDED (k-induction K=2); ddr5_xbar request-path routing safety (exclusive one-hot routing / banked-channel selection / ready coherence / payload integrity) proven UNBOUNDED + response-FIFO no-overflow/underflow (cnt[c]<=RESP_QD conservation form, inflight<=CAP) proven UNBOUNDED via connect-bound internal cnt[] counters (task C5) -- tag-issued stays BOUNDED (FIFO-content data-invariant; the FIFO is a 2-D memory cell, not connect-bindable); strict unsigned monotonicity stays BOUNDED (32-bit counter wrap); flash_xbar per-channel-queue no-overflow (cnt[c]<=QDEPTH) + outstanding<=N_CH*QDEPTH (P3) + inflight<=outstanding (P1a/P1b) proven UNBOUNDED via connect-bound internal counters (k-induction K=3) -- tag-issued (P2) stays BOUNDED (needs FIFO-content data-invariant; FIFO is a 2-D memory cell, not connect-bindable) -- see docs/FORMAL.md"

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
	@# glm_fp8_contract: vectorized (torch/numpy/pure-python) our-FP8 contract, bit-exact to glm_fp8_ref -- the GPU-scale model for the Modal P1.1 real-checkpoint gate.
	@python3 tools/glm_fp8_contract.py | grep -qE 'ALL [0-9]+ TESTS PASSED' || { echo "FAILED: glm_fp8_contract"; exit 1; }
	@printf '[%s] ' "glm_fp8_contract"; echo "== glm_fp8_ref (bit-exact, vectorized)"
	@python3 test/modal_validate_test.py | grep -qE 'ALL [0-9]+ TESTS PASSED' || { echo "FAILED: modal_validate compare logic"; exit 1; }
	@printf '[%s] ' "modal_validate(compare logic)"; echo "PASS"
	@echo "bitacc: glm_matmul_fp8 == real GLM-5.2-FP8 FP8 contract, argmax preserved; Modal P1.1 app ready (docs/MODAL_VALIDATE.md)"

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

# ---- Whole-chip structural gate for the GLM-5.2-FP8 product top (task C1) ----
# `synth` above gates only the legacy scalar TPU core.  This gate elaborates the
# ENTIRE product hierarchy -- the 2-clock chip top `glm_fp8_system_cdc` and every
# GLM compute + memory-system + CDC leaf beneath it -- and runs `check -assert`,
# which FAILS (non-zero exit) on any unresolved hierarchy, combinational loop,
# multiple driver, or inferred latch anywhere in the GLM datapath.  This is the
# first structural sign-off of the actual product top (nothing gated it before).
# `stat` prints the flattened gate-level cell count for the whole chip.
GLM_CDC_SRCS := src/glm_fp8_system_cdc.v src/glm_fp8_system.v src/cdc_async_fifo.v \
	src/reset_sync.v src/glm_model_fp8.v src/ddr5_xbar.v src/weight_loader.v \
	src/expert_cache_pf.v src/expert_cache_ctrl.v src/kv_cache_pager.v \
	src/glm_decoder_block_fp8.v src/mla_attn_fp8.v src/swiglu_expert_fp8.v \
	src/moe_router_fp8.v src/glm_matmul_fp8.v src/rmsnorm_unit.v \
	src/rope_interleave_unit.v src/glm_softmax.v src/dsa_indexer.v \
	src/topk_select.v src/glm_act.v src/glm_matmul_pipe.v src/sampler.v \
	src/glm_fp_pipe.v

synth-glm:
	$(YOSYS) -q -p "read_verilog -sv -I src $(GLM_CDC_SRCS); \
	                hierarchy -top glm_fp8_system_cdc -check; proc; opt; check -assert; stat"

# ---- CDC structural sign-off for the 2-clock product top (task C8) ----------
# Asserts every host_clk<->core_clk crossing in glm_fp8_system_cdc flows through a
# recognized synchronizer (async FIFO / 2-FF / reset_sync) and that no raw
# multi-bit register is captured across the boundary.  A targeted structural
# checker (not a commercial CDC tool -- see tools/cdc_check.py header for limits);
# constraints/glm_fp8_system_cdc.sdc carries the matching false-path/async SDC.
cdc:
	@python3 tools/cdc_check.py src/glm_fp8_system_cdc.v \
	    || { echo "FAILED: cdc structural sign-off"; exit 1; }

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
