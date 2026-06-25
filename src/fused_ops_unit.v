`timescale 1ns/1ps
`include "tpu_defs.vh"
//============================================================================
// fused_ops_unit  --  TPU v2.0 tile-domain elementwise post-op  (SPEC.md §5.5)
//----------------------------------------------------------------------------
// PURPOSE
//   The fused tensor class (OP_FUSE_GEMM_RELU 0x40, OP_FUSE_CONV_RELU 0x41)
//   applies an elementwise RELU to a tensor-compute result tile *in place*,
//   one TM line at a time (SPEC §5.5: "KEEP for scalar fusions ... re-express
//   as tile-domain post-ops; combinational over already-computed tiles").
//   This module is the post-op datapath: it takes ONE 128-bit TM line (the
//   result line just produced by GEMM/CONV2D), applies a per-lane saturating
//   RELU when the opcode is a FUSE_*_RELU, and emits the post-processed line.
//   For any non-fused opcode it is a transparent PASS-THROUGH of the input
//   line, so it can sit unconditionally in the result write-back path.
//
// ALGORITHM  (purely combinational over one tile line)
//   The 128-bit line is interpreted in ONE of TWO packings, selected by the
//   `dense16` input, because GEMM and CONV2D pack their result tiles
//   DIFFERENTLY (this is the fix for the v2.0 FUSE_CONV_RELU granularity bug):
//
//     dense16==0  (GEMM packing): LINE_LANES (4) lanes of LANE_W (32) bits, each
//       holding one Q7.8 element sign-extended into the 32-bit lane (SPEC §1.3:
//       "16-bit value sign-extended into 32-bit lane").  RELU each 32-bit lane:
//           out_lane = ($signed(x) < 0) ? 0 : x
//
//     dense16==1  (CONV2D packing): 8 DENSE 16-bit Q7.8 elements packed
//       [16*e +: 16] (conv2d_unit.v packs output pixels bits[16*(p%4)+:16], 4
//       px in the low 64 bits; the upper 64 are unused/zero, so all 8 sub-fields
//       are RELU'd safely).  RELU each 16-bit element INDEPENDENTLY:
//           out_elem = ($signed(e) < 0) ? 0 : e
//       Treating a 32-bit lane as one signed value here would test bit31 (the
//       ODD pixel's sign) and corrupt the EVEN pixel -- the original bug.
//
//   For any non-FUSE opcode the unit is a transparent PASS-THROUGH of the input
//   line (so it can sit unconditionally in the result write-back path).
//   RELU clamps only the NEGATIVE side (to 0); a non-negative element is already
//   a valid in-range Q7.8 (its producer GEMM/CONV already round+saturated it),
//   so no upper clamp can fire and `sat` is never asserted by this stage.
//
//   NOTE: for sign-extended Q7.8 data (the GEMM packing) the dense16 and 32-bit
//   paths are IDENTICAL -- a positive lane's upper half is 0x0000 (RELU->itself)
//   and a negative lane's both halves clamp to 0 -- so dense16 is purely an
//   element-granularity refinement for genuinely-packed CONV data.
//
// Q-FORMAT
//   In  : Q7.8 elements; either 4 (sign-extended into 32-bit lanes) when
//         dense16==0, or up to 8 (dense 16-bit) when dense16==1.
//   Out : same format/packing; RELU preserves the Q7.8 scale (negatives -> +0).
//
// LATENCY
//   COMBINATIONAL (0 cycles).  `tile_out`/`sat` are valid the same cycle the
//   inputs are valid; there is no clock, no state, hence no reset.
//
// INTERFACE
//   opcode   [OPCODE_W-1:0] (in)   selects RELU (FUSE_*_RELU) vs pass-through
//   dense16                 (in)   1 = dense-16 packing (CONV), 0 = 32-bit lane
//                                   (GEMM).  Ignored on pass-through.
//   tile_in  [LINE_W-1:0]   (in)   one 128-bit input line
//   tile_out [LINE_W-1:0]   (out)  post-processed line (RELU'd or passed thru)
//   sat                     (out)  saturation flag for this stage (always 0)
//
// SYNTHESIZABILITY
//   Pure combinational logic: a single always @(*) that assigns every output on
//   every path (no inferred latch), no clocked state (so no reset needed), no
//   non-synthesizable constructs.  Elements are unrolled with explicit constant
//   bit ranges (no variable part-select) so the result is unambiguously
//   combinational and width-clean.  File name == module name (DECLFILENAME).
//============================================================================
module fused_ops_unit (
    input  wire [`OPCODE_W-1:0] opcode,
    input  wire                 dense16,
    input  wire [`LINE_W-1:0]   tile_in,
    output reg  [`LINE_W-1:0]   tile_out,
    output reg                  sat
);

    // Per-LANE (32-bit) RELU: interpret a 32-bit lane as signed Q7.8 (low 16
    // carry the value, sign-extended) and clamp the negative side to 0.
    function [`LANE_W-1:0] relu_lane;
        input [`LANE_W-1:0] v;
        begin
            relu_lane = $signed(v) < 0 ? {`LANE_W{1'b0}} : v;
        end
    endfunction

    // Per-ELEMENT (16-bit) RELU: clamp one dense Q7.8 element's negative side.
    function [`ELEM_W-1:0] relu_elem;
        input [`ELEM_W-1:0] v;
        begin
            relu_elem = $signed(v) < 0 ? {`ELEM_W{1'b0}} : v;
        end
    endfunction

    // Decode: only the two FUSE_*_RELU opcodes apply RELU; everything else
    // (including unrelated opcodes presented while this stage is idle) is a
    // transparent pass-through of the input line.
    wire do_relu = (opcode == `OP_FUSE_GEMM_RELU) ||
                   (opcode == `OP_FUSE_CONV_RELU);

    integer e;
    always @(*) begin
        if (do_relu && dense16) begin
            // Dense-16 (CONV): RELU all 8 packed 16-bit elements independently.
            for (e = 0; e < (`LINE_W/`ELEM_W); e = e + 1)
                tile_out[ e*`ELEM_W +: `ELEM_W ] =
                    relu_elem(tile_in[ e*`ELEM_W +: `ELEM_W ]);
        end else if (do_relu) begin
            // 32-bit lane (GEMM): RELU the 4 sign-extended lanes.
            tile_out[ 0*`LANE_W +: `LANE_W ] = relu_lane(tile_in[ 0*`LANE_W +: `LANE_W ]);
            tile_out[ 1*`LANE_W +: `LANE_W ] = relu_lane(tile_in[ 1*`LANE_W +: `LANE_W ]);
            tile_out[ 2*`LANE_W +: `LANE_W ] = relu_lane(tile_in[ 2*`LANE_W +: `LANE_W ]);
            tile_out[ 3*`LANE_W +: `LANE_W ] = relu_lane(tile_in[ 3*`LANE_W +: `LANE_W ]);
        end else begin
            // Pass-through: emit the input line unchanged.
            tile_out = tile_in;
        end

        // This post-op never introduces new saturation: RELU only clamps the
        // negative side to exactly 0 (representable), and non-negative elements
        // are already valid Q7.8 from their producer.  Held at 0 for status-path
        // symmetry with the multi-cycle tensor units' `sat` flag.
        sat = 1'b0;
    end
endmodule
