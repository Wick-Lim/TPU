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

# ---- per-unit TBs.  Each builds against its own module (file==module); the
#      attention TB also needs softmax_unit (it instantiates the real unit). ----
UNITS := instruction_decoder register_file memory tile_memory vector_alu \
         dma_controller scatter_gather gemm_systolic conv2d_unit softmax_unit \
         fused_ops_unit attention_unit

IFLAGS := -g2012 -Wall -I src

.PHONY: all build test hazard axi unittests sim wave lint synth ppa clean

all: test hazard unittests lint synth

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
	@echo "unittests: all per-unit TBs passed"

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
