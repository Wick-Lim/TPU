`ifndef GLM_FP_PIPE_LATS
`define GLM_FP_PIPE_LATS
//============================================================================
// glm_fp_pipe_lat.vh  --  SINGLE SOURCE OF TRUTH for the pipelined-FP latencies.
//----------------------------------------------------------------------------
// Each pipelined FP module in glm_fp_pipe.v sets its `localparam LAT` from the
// matching macro below (structural flop count, valid_in -> valid_out), and
// every CONSUMER (fp32_mac_pipe / fp32_rsqrt_pipe / fp32_exp_pipe inside
// glm_fp_pipe.v, glm_matmul_pipe.v, and the self-checking TBs) reads these same
// macros instead of hardcoding a number.  Changing a base latency (FP_MUL_LAT
// or FP_ADD_LAT) ripples everywhere with no per-site edit.
//
// We use macros rather than cross-module hierarchical `localparam` references
// (which neither iverilog nor verilator accept in a constant expression) so the
// single source of truth is also portable across iverilog / verilator / yosys
// and across single-top compiles.
//
//   FP_MUL_LAT   : fp32_mul_pipe   = 2   (product stage + normalize/round stage)
//   FP_ADD_LAT   : fp32_add_pipe   = 5   (align, add, LZ-count, normalize, round)
//   FP_MAC_LAT   : fp32_mac_pipe   = FP_MUL_LAT + FP_ADD_LAT                 = 7
//   FP_RSQRT_LAT : fp32_rsqrt_pipe = FP_MUL_LAT + 2*(3*FP_MUL_LAT+FP_ADD_LAT)= 24
//   FP_EXP_LAT   : fp32_exp_pipe   = 7*FP_MUL_LAT + 6*FP_ADD_LAT + 2         = 46
//============================================================================
`define FP_MUL_LAT    2
`define FP_ADD_LAT    5
`define FP_MAC_LAT    (`FP_MUL_LAT + `FP_ADD_LAT)
`define FP_RSQRT_LAT  (`FP_MUL_LAT + 2*(3*`FP_MUL_LAT + `FP_ADD_LAT))
`define FP_EXP_LAT    (7*`FP_MUL_LAT + 6*`FP_ADD_LAT + 2)
`endif
