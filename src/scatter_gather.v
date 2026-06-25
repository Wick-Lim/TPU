`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// scatter_gather  --  indexed DMEM access engine (TPU v2.0)      (SPEC §2, §5.5)
//----------------------------------------------------------------------------
// Symmetric indexed (scatter/gather) DMEM access.  v1.5 had GATHER only; v2.0
// adds the SCATTER store path so the unit is direction-symmetric, exactly as
// SPEC §5.5 requires:
//
//     "scatter_gather.v -- KEEP gather, ADD scatter (store path) so the unit
//      is symmetric; effective address base + index*stride, stride from imm."
//
//   EFFECTIVE ADDRESS (combinational, computed in the EX stage):
//
//        eff_addr = base_addr + (index << stride_exp)
//
//   where stride_exp in {0,1,2,3} is the immediate-carried log2(stride)
//   (ISA: imm12[1:0] for OP_GATHER/OP_SCATTER).  A word stride is stride_exp=2.
//
//   DIRECTION (selected by is_scatter):
//     GATHER  (is_scatter=0, OP_GATHER  0x11):  data_out  = DMEM[eff_addr]
//     SCATTER (is_scatter=1, OP_SCATTER 0x12):  DMEM[eff_addr] = data_in
//
// MEMORY OWNERSHIP / PORTS:
//   This unit does NOT instantiate DMEM.  Per the build rules it exposes DMEM
//   ACCESS PORTS and the surrounding datapath (tpu_top MEM stage, or the unit
//   TB) owns the actual memory:
//     * dmem_addr  : effective word address driven to DMEM (both directions)
//     * dmem_rdata : combinational read data coming back from DMEM (gather)
//     * dmem_wdata : write data driven to DMEM (scatter)
//     * dmem_we    : DMEM write-enable (asserted only on a started SCATTER)
//   DMEM is the v2.0 256x32 word memory (combinational read, synchronous write);
//   the unit is purely combinational so the access resolves inside one MEM
//   cycle (SPEC §3.2 latency table: GATHER/SCATTER == 1 cycle, resolves in MEM).
//
// Q-FORMAT: NONE.  DMEM holds raw 32-bit scalar words; this unit performs only
//   integer address math and word movement.  base_addr/index are plain unsigned
//   word addresses; data is an opaque 32-bit word.
//
// LATENCY: COMBINATIONAL (0-cycle).  The unit asserts addr/wdata/we in the same
//   cycle `start` is high; the SCATTER write commits on the NEXT DMEM clock edge
//   (1-cycle visibility, owned by DMEM), and the GATHER read data is valid
//   combinationally in the same cycle.
//
// INTERFACE:
//   start                          enable (issue this indexed access this cycle)
//   is_scatter                     0 => GATHER (load), 1 => SCATTER (store)
//   base_addr   [`XLEN-1:0]        base word address (from rA)
//   index       [`XLEN-1:0]        index   (from rB)
//   stride_exp  [STRIDE_EXP_W-1:0] log2(stride), imm-carried (imm12[1:0])
//   data_in     [`XLEN-1:0]        scatter store data (from rC-source)
//   dmem_rdata  [`XLEN-1:0]        DMEM read data (gather load result)
//   addr_out    [`XLEN-1:0]        full effective address (observability/debug)
//   dmem_addr   [`DMEM_ADDR_W-1:0] DMEM word address (truncated eff_addr)
//   dmem_wdata  [`XLEN-1:0]        DMEM write data (scatter)
//   dmem_we                        DMEM write-enable (scatter & start)
//   data_out    [`XLEN-1:0]        gather load result (= dmem_rdata when gather)
//============================================================================
module scatter_gather #(
    // Width of the stride-exponent field.  ISA carries it in imm12[1:0], so a
    // value of {0,1,2,3} (2 bits) is the committed range; kept as a parameter so
    // the field can widen without editing the body.  index << stride_exp is the
    // (1<<stride_exp)-word stride mandated by SPEC §2/§5.5.
    parameter integer STRIDE_EXP_W = 2
) (
    input  wire                     start,       // issue an indexed access
    input  wire                     is_scatter,  // 0=gather(load) 1=scatter(store)
    input  wire [`XLEN-1:0]         base_addr,   // base word address (rA)
    input  wire [`XLEN-1:0]         index,       // index (rB)
    input  wire [STRIDE_EXP_W-1:0]  stride_exp,  // log2(stride), imm12[1:0]
    input  wire [`XLEN-1:0]         data_in,     // scatter store data (rC-source)
    input  wire [`XLEN-1:0]         dmem_rdata,  // DMEM read data (gather)
    output wire [`XLEN-1:0]         addr_out,    // full effective address (debug)
    output wire [`DMEM_ADDR_W-1:0]  dmem_addr,   // DMEM word address
    output wire [`XLEN-1:0]         dmem_wdata,  // DMEM write data (scatter)
    output wire                     dmem_we,     // DMEM write-enable (scatter)
    output wire [`XLEN-1:0]         data_out     // gather load result
);

    //------------------------------------------------------------------
    // Effective address: base + (index << stride_exp).
    //
    // Computed entirely in XLEN word-address space (matching the v1.5
    // register-width address arithmetic): the shift and the add both wrap at
    // XLEN, so any magnitude beyond the 32-bit word-address range is
    // intentionally discarded (architectural wrap, no out-of-band bits to leave
    // unused).  stride_exp <= 2^STRIDE_EXP_W-1 selects a (1<<stride_exp)-word
    // stride per SPEC §2/§5.5.
    //------------------------------------------------------------------
    wire [`XLEN-1:0] shifted_idx = index << stride_exp;       // XLEN-wrapping shift
    assign addr_out = base_addr + shifted_idx;                // XLEN-wrapping add

    // DMEM physical word address = low DMEM_ADDR_W bits of the effective address
    // (DMEM is 256 words / 8-bit address; addresses wrap within DMEM, SPEC §2).
    assign dmem_addr = addr_out[`DMEM_ADDR_W-1:0];

    //------------------------------------------------------------------
    // Direction control.
    //   SCATTER: drive store data + write-enable (only when started).
    //   GATHER : write-enable low; load result passes the DMEM read data through.
    //------------------------------------------------------------------
    assign dmem_we    = start & is_scatter;
    assign dmem_wdata = data_in;

    // Gather result.  When not performing a started gather the read result is
    // forced to 0 so a disabled/scatter cycle never presents stale load data on
    // the writeback path (every output assigned on every path -> no latch).
    assign data_out = (start & ~is_scatter) ? dmem_rdata : {`XLEN{1'b0}};

endmodule
