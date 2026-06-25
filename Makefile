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

# ---- per-unit TBs.  Each builds against its own module (file==module); the
#      attention TB also needs softmax_unit (it instantiates the real unit). ----
UNITS := instruction_decoder register_file memory tile_memory vector_alu \
         dma_controller scatter_gather gemm_systolic conv2d_unit softmax_unit \
         fused_ops_unit attention_unit

IFLAGS := -g2012 -Wall -I src

.PHONY: all build test hazard unittests sim wave lint synth clean

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

clean:
	rm -rf $(BUILD_DIR) tpu_waveform.vcd hazard_waveform.vcd
	rm -f *.vcd
