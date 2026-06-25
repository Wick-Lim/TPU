//============================================================================
// tpu_defs.vh  --  TPU v2.0 single source of truth
//----------------------------------------------------------------------------
// This header is the ONE normative place that defines:
//   * datapath sizes / tensor-tile geometry            (SPEC.md  §1.2)
//   * fixed-point Q-formats, accumulator widths, bounds (SPEC.md  §1.3)
//   * the complete 8-bit opcode map  0x00..0x41         (docs/ISA.md §3)
//   * instruction field positions                       (docs/ISA.md §1)
//   * status-register field layout                       (SPEC.md  §4)
//   * shared rounding/saturation/sign-extend EXPRESSION macros so that no two
//     units can diverge on the fixed-point contract      (SPEC.md  §1.3, §5)
//
// CONTRACT: this file contains DEFINES/MACROS ONLY -- no logic, no nets, no
// always blocks.  It is `include`d by every module (compile with `-I src`) so
// the encoding and the arithmetic contract can never drift between the
// decoder, the EX units, the control decoder and the testbenches.
//
// Every module must compile clean under `verilator --lint-only -Wall`; all
// expression macros below are fully parenthesized and width-explicit so a
// caller can drop them into any context without precedence or width warnings.
//
// Instruction encoding (32-bit, two formats selected by opcode):
//   R-format: [31:24] opcode | [23:20] rA | [19:16] rB | [15:12] rC | [11:0] imm12
//   I-format: [31:24] opcode | [19:16] rC | [19:0]  imm20            (LOADI only)
//============================================================================
`ifndef TPU_DEFS_VH
`define TPU_DEFS_VH

//============================================================================
// 1. Datapath / tensor-tile sizes                               (SPEC.md §1.2)
//----------------------------------------------------------------------------
// All sizes are committed defaults.  They are intentionally small (so the unit
// TBs can be near-exhaustive) yet each still exercises the FULL algorithm.
//============================================================================
`define XLEN          32   // scalar word width (bits)
`define LANE_W        32   // tensor lane width (one element slot, bits)
`define LINE_LANES    4    // lanes packed per TM line
`define LINE_W        128  // TM line width = LANE_W * LINE_LANES (bits)
`define TM_LINES      32   // tile-memory depth (number of 128-bit lines)
`define RF_REGS       16   // scalar registers r0..r15 (r0 hardwired zero)

`define GEMM_N        4    // GEMM tile is GEMM_N x GEMM_N (4x4)
`define CONV_H        8    // conv input feature-map height
`define CONV_W        8    // conv input feature-map width
`define CONV_K        3    // conv kernel size (CONV_K x CONV_K = 3x3)
`define CONV_OH       6    // conv output height = CONV_H - CONV_K + 1 (valid pad)
`define CONV_OW       6    // conv output width  = CONV_W - CONV_K + 1 (valid pad)
`define SEQ_LEN       4    // attention sequence length
`define D_MODEL       4    // attention head dimension
`define SM_LEN        8    // softmax vector length

//============================================================================
// 2. Address / index widths and storage depths                  (SPEC.md §2)
//----------------------------------------------------------------------------
// Architecturally distinct, non-aliasing address spaces selected by op class.
//============================================================================
`define RF_IDX_W      4    // scalar register index width  (16 regs)
`define TM_IDX_W      5    // tile-memory line index width  (32 lines)
`define DMEM_DEPTH    256  // data-memory depth (32-bit words)
`define DMEM_ADDR_W   8    // data-memory address width (256 words)

//============================================================================
// 3. Instruction field positions                              (docs/ISA.md §1)
//----------------------------------------------------------------------------
// The decoder extracts every field unconditionally; each op documents which it
// consumes.  Bit ranges are exposed as defines so slicing stays consistent.
//============================================================================
`define OPCODE_HI     31   // opcode field  instr[31:24]
`define OPCODE_LO     24
`define RA_HI         23   // rA field      instr[23:20]
`define RA_LO         20
`define RB_HI         19   // rB field      instr[19:16]
`define RB_LO         16
`define RC_HI         15   // rC field      instr[15:12]
`define RC_LO         12
`define IMM12_HI      11   // imm12 field   instr[11:0]   (R-format)
`define IMM12_LO      0
`define IMM20_HI      19   // imm20 field   instr[19:0]   (I-format, LOADI)
`define IMM20_LO      0

`define OPCODE_W      8    // opcode width (bits)
`define IMM12_W       12   // R-format immediate width
`define IMM20_W       20   // I-format immediate width (LOADI)

//============================================================================
// 4. Fixed-point formats: fractional bits, widths, bounds       (SPEC.md §1.3)
//----------------------------------------------------------------------------
// All tensor data are signed two's-complement fixed-point.  The format is
// per-operation, documented, and enforced by the unit -- never silently
// truncated.  Narrowing is always round-half-up then saturate (see §5).
//----------------------------------------------------------------------------
//   Q7.8   : 16-bit signed, 8 fractional bits     -- GEMM/CONV/ATTN element data
//   Q14.16 : product of two Q7.8  (30-bit signed) -- single MAC product
//   Q15.16 : 48-bit signed accumulator            -- MAC sum / exp accumulation
//   Q1.30  : 32-bit reciprocal (softmax 1/sum)
//   Q0.16  : 16-bit unsigned-range probability [0,1], 0xFFFF ~= 1.0
//============================================================================

// --- Element value (the stored Q7.8 number) ---
`define ELEM_W        16   // element value width (bits) -- the Q7.8 value
`define Q78_FRAC      8    // Q7.8 fractional bits  (FRAC)
`define Q78_INT       7    // Q7.8 integer bits (excl. sign)

// --- Single MAC product width: Q7.8 * Q7.8 = Q14.16 ---
`define Q1416_FRAC    16   // product fractional bits (8 + 8)
`define PROD_W        32   // 16x16 signed product fits in 32 bits (uses 30)

// --- Wide accumulator: Q15.16, 48-bit ---
`define ACC_W         48   // accumulator width (bits)
`define ACC_FRAC      16   // accumulator fractional bits (Q15.16)

// --- Softmax / attention auxiliary formats ---
`define Q016_FRAC     16   // Q0.16 probability fractional bits
`define Q016_W        16   // Q0.16 probability width (bits)
`define Q016_ONE      16'hFFFF // Q0.16 value closest to 1.0
`define Q130_FRAC     30   // Q1.30 reciprocal fractional bits
`define Q130_W        32   // Q1.30 reciprocal width (bits)
`define EXP_LUT_DEPTH 64   // piecewise-linear exp LUT entries (Q15.16 outputs)

// --- Q7.8 saturation bounds (signed 16-bit range) ---
//   smallest representable = -32768 = -128.0 in Q7.8
//   largest  representable = +32767 = ~127.996 in Q7.8
`define Q78_MAX       16'sh7FFF  //  32767
`define Q78_MIN       16'sh8000  // -32768
`define Q78_MAX_VAL   32767      //  plain-int max (for comparisons)
`define Q78_MIN_VAL   (-32768)   //  plain-int min (for comparisons)

//============================================================================
// 5. Status-register field layout                               (SPEC.md §4)
//----------------------------------------------------------------------------
// Small status block, read via RDSTATUS, cleared via CLRSTATUS. `sat` and
// `illegal` are sticky.  Packed into a 32-bit word as below (low bits used).
//   [0]    sat       sticky saturation occurred in last tensor op
//   [3:1]  unit      last tensor unit id
//   [7:4]  argmax    softmax argmax index (4-bit field, 3-bit value 0..7 for
//                    SM_LEN=8; bit [7] is structurally 0 -- unit emits [2:0])
//   [8]    illegal   last decoded illegal opcode (sticky)
//============================================================================
`define ST_SAT_BIT      0
`define ST_UNIT_HI      3
`define ST_UNIT_LO      1
`define ST_ARGMAX_HI    7
`define ST_ARGMAX_LO    4
`define ST_ILLEGAL_BIT  8
`define ST_W            32   // status word width (uses low 9 bits)

// Tensor unit ids (status `unit` field).
`define UNIT_NONE       3'd0
`define UNIT_GEMM       3'd1
`define UNIT_CONV       3'd2
`define UNIT_SOFTMAX    3'd3
`define UNIT_ATTENTION  3'd4

//============================================================================
// 6. Opcode map  (8-bit, exactly per docs/ISA.md §3)
//----------------------------------------------------------------------------
// Scalar / control / memory class
//============================================================================
`define OP_NOP            8'h00   // no operation
`define OP_LOADI          8'h01   // I-fmt: rC = sign_extend(imm20)
`define OP_LOAD           8'h02   // rC = DMEM[rA + imm12]   (8-bit eff addr)
`define OP_STORE          8'h03   // DMEM[rA + imm12] = rB
`define OP_ADD            8'h04   // rC = rA + rB
`define OP_SUB            8'h05   // rC = rA - rB
`define OP_AND            8'h06   // rC = rA & rB
`define OP_OR             8'h07   // rC = rA | rB
`define OP_XOR            8'h08   // rC = rA ^ rB
`define OP_SHL            8'h09   // rC = rA << rB[4:0]
`define OP_SHR            8'h0A   // rC = rA >> rB[4:0]   (logical)
`define OP_RELU           8'h0B   // rC = (signed rA < 0) ? 0 : rA
`define OP_ADDI           8'h0C   // rC = rA + sign_extend(imm12)
`define OP_RDSTATUS       8'h0D   // rC = status_register
`define OP_CLRSTATUS      8'h0E   // clear sticky sat/illegal

// Memory-movement class
`define OP_DMA            8'h10   // for i<imm12: DMEM[rB+i]=DMEM[rA+i]; rC=#words
`define OP_GATHER         8'h11   // rC = DMEM[rA + (rB << imm12[1:0])]
`define OP_SCATTER        8'h12   // DMEM[rA + (rB << imm12[1:0])] = rC-source

// Tile-memory transfer class
`define OP_TLOAD          8'h20   // TM[imm12[4:0]] = {DMEM[rA+3..rA]}
`define OP_TSTORE         8'h21   // {DMEM[rA+3..rA]} = TM[imm12[4:0]]

// Tensor compute class
`define OP_GEMM           8'h30   // C=A.B 4x4 Q7.8; A=rA,B=rB,C=imm12[4:0]; rC=st
`define OP_CONV2D         8'h31   // 8x8*3x3 conv; in=rA,k=rB,out=imm12[9:5],
                                  //   stride=imm12[1:0],pad=imm12[3:2]; rC=status
`define OP_SOFTMAX        8'h32   // len-8 exp-softmax; x=rA,p=imm12[4:0];
                                  //   rC=status_word (argmax in bits [7:4])
`define OP_ATTENTION      8'h33   // softmax(Q.K'/sqrt d).V seq4 d4;
                                  //   Q=rA,K=rB,V=imm12[4:0]; O base = VALUE of rC
                                  //   read at dispatch, then rC OVERWRITTEN with
                                  //   status on retire (rC not preserved)

// Fused tensor class
`define OP_FUSE_GEMM_RELU 8'h40   // GEMM then elementwise RELU on C tile (in place)
`define OP_FUSE_CONV_RELU 8'h41   // CONV2D then RELU on output tile

//============================================================================
// 7. Shared arithmetic EXPRESSION macros                  (SPEC.md §1.3, §5)
//----------------------------------------------------------------------------
// These are the ONLY sanctioned way for any unit to (a) narrow a wide signed
// accumulator back to a Q7.8 element, or (b) sign-extend a 16-bit element to
// 32 bits.  Centralizing them guarantees rounding/saturation cannot diverge
// between GEMM, CONV2D, SOFTMAX and ATTENTION.
//
// Every macro is fully parenthesized (argument- and result-level) and uses
// explicit-width sized literals so `verilator --lint-only -Wall` stays clean in
// any caller context (no UNSIGNED/WIDTH/IMPLICIT warnings introduced here).
//============================================================================

// --- `TPU_SEXT16(v) : sign-extend a 16-bit value to a 32-bit signed value ---
// SEMANTICS: interpret the low 16 bits of `v` as a signed Q7.8 element and
// produce its 32-bit two's-complement sign-extension (Q7.8 occupying the low
// 16 bits, sign replicated through [31:16]).  Result width: 32 bits (XLEN),
// signed.  Implemented with explicit sized casts (not a part-select) so the
// argument may be ANY expression -- a part-select on a parenthesized expression
// is illegal Verilog.  Inner ELEM_W'() narrows to the low 16 bits, $signed
// marks them signed, and the outer XLEN'() sign-extends to 32 bits.
`define TPU_SEXT16(v) \
    ( `XLEN'($signed(`ELEM_W'(v))) )

// --- `TPU_ROUND_SHIFT(acc) : round-half-up then arithmetic >> FRAC ----------
// SEMANTICS: take a wide SIGNED accumulator `acc` (Q15.16, ACC_W bits), add the
// round-half-up bias (1 << (Q78_FRAC-1)) and arithmetic-shift right by Q78_FRAC
// to return to Q7.8 *scale*.  The result is the rounded value BEFORE clamping
// (still ACC_W-bit signed so the subsequent saturate sees true magnitude).
// Rounding mode: round-half-up (ties go toward +inf), per SPEC.md §1.3.
//
// SIGNEDNESS (the fix for the v1.x bug): the rounding bias is built as a
// SIGNED, ACC_W-wide value -- `$signed(`ACC_W'd1 <<< (`Q78_FRAC-1))` is the
// signed accumulator-width constant 1<<(FRAC-1)=128.  Because BOTH operands of
// the `+` are now signed, the add evaluates SIGNED, so the following `>>>` is an
// ARITHMETIC shift and NEGATIVE accumulators narrow correctly (e.g. acc=-12800
// -> -50, not +32767).  The earlier UNSIGNED concatenation bias forced the add
// unsigned and turned `>>>` into a LOGICAL shift, mis-narrowing every negative
// accumulator -- which is exactly why each tensor unit had to carry a local
// signed helper.  `acc` should also be a SIGNED expression at the call site
// (pass a `signed` net/reg or wrap with $signed(...)); $signed(acc) is folded in
// here so the add is signed even if the caller passes an unsigned-typed net.
`define TPU_ROUND_SHIFT(acc) \
    ( ( $signed(acc) + $signed(`ACC_W'd1 <<< (`Q78_FRAC-1)) ) >>> `Q78_FRAC )

// --- `TPU_SAT_Q78(x) : saturate a wide signed value to the Q7.8 range -------
// SEMANTICS: clamp a SIGNED value `x` (any width >= 16, typically the result of
// TPU_ROUND_SHIFT) to the signed 16-bit Q7.8 range [-32768, 32767] and return
// it as a 16-bit signed value.  Clamping is value-comparison based so it works
// for any caller width; the in-range branch narrows to `ELEM_W` via an EXPLICIT
// sized cast (intentional truncation -> no lint WIDTH warning, and a part-select
// is illegal on a parenthesized expression so the cast is also required).
//   x >  32767 -> 16'sh7FFF
//   x < -32768 -> 16'sh8000
//   else       -> ELEM_W'(x)
`define TPU_SAT_Q78(x) \
    ( ( $signed(x) > $signed(`ACC_W'sd32767) )  ? `Q78_MAX : \
      ( $signed(x) < $signed(-`ACC_W'sd32768) ) ? `Q78_MIN : \
                                                  `ELEM_W'($signed(x)) )

// --- `TPU_RND_SAT_Q78(acc) : the full accumulator -> Q7.8 element narrowing --
// SEMANTICS: the single canonical narrowing used by ALL tensor units.  Takes a
// SIGNED Q15.16 accumulator `acc` (ACC_W bits), applies round-half-up
// (add 1<<(FRAC-1)), arithmetic-shifts right FRAC, then saturates to the signed
// Q7.8 range [-32768, 32767].  Result width: 16 bits, signed Q7.8.  Now that
// TPU_ROUND_SHIFT keeps the add SIGNED, this narrows NEGATIVE accumulators
// correctly (e.g. acc=-12800 -> -50), matching the per-unit local helpers it
// is meant to replace.
// Saturation detection (for the sticky `sat` flag) is the caller's job via
// TPU_SAT_HIT below -- this macro returns the clamped value only.
`define TPU_RND_SAT_Q78(acc) \
    ( `TPU_SAT_Q78( `TPU_ROUND_SHIFT(acc) ) )

// --- `TPU_SAT_HIT(acc) : did the canonical narrowing of `acc` saturate? -----
// SEMANTICS: returns 1'b1 iff applying round-half-up+shift to the SIGNED
// accumulator `acc` produces a value outside the Q7.8 range (i.e. the clamp in
// TPU_SAT_Q78 actually fired), else 1'b0.  Units OR this into their sticky
// `sat` status bit so saturation is always observable (SPEC.md §1.3/§4).
// Result width: 1 bit.
`define TPU_SAT_HIT(acc) \
    ( ( $signed( `TPU_ROUND_SHIFT(acc) ) > $signed(`ACC_W'sd32767)  ) || \
      ( $signed( `TPU_ROUND_SHIFT(acc) ) < $signed(-`ACC_W'sd32768) ) )

`endif // TPU_DEFS_VH
