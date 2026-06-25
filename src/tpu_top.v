`include "tpu_defs.vh"
`timescale 1ns/1ps
//============================================================================
// tpu_top.v  --  TPU v2.0 top-level core  (module name MUST stay `TPU`)
//----------------------------------------------------------------------------
// v2.0 INTEGRATION of the independently-verified leaf units into a working
// scalar 5-stage pipeline (IF -> ID -> EX -> MEM -> WB) augmented with a
// variable-length BUSY stall for the multi-cycle tensor units, per SPEC.md
// §3-§4 and docs/ISA.md.  Supersedes the v1.5 top (which wired the now-deleted
// matrix_multiply.v / convolution_unit.v toy units).
//
// PIPELINE & HAZARDS (SPEC §3)
//   * Scalar pipeline kept in spirit from v1.5: combinational RF read (r0=0 via
//     register_file.v); a CONTROL DECODER emitting per-opcode writes_reg /
//     mem_read / mem_write / uses_tensor (NEVER `opcode!=0`); full forwarding
//     (EX / EX-MEM / MEM-WB) into the ID operand path; one-cycle load-use stall
//     for LOAD/GATHER producers.  Illegal opcode -> illegal_opcode flag +
//     sticky status.illegal + safe NOP (no architectural side effects).
//   * Immediate path: LOADI (I-format rC<=sext(imm20)); ADDI; LOAD/STORE
//     displacement (rA+sext(imm12)); GATHER/SCATTER stride exponent imm12[1:0];
//     DMA length imm12; tensor tile-base indices / modifiers.  Software seed
//     path -- NO hierarchical TB pokes for program-visible state.
//   * DMEM (memory.v, 256x32, dual read port) backs LOAD/STORE/DMA/GATHER/
//     SCATTER/TLOAD/TSTORE.  Tile memory (tile_memory.v, 32x128, 2R+1W) is the
//     tensor operand store.
//
// TENSOR OPS (SPEC §3.2)  -- TM->TM, dispatched in EX with a BUSY stall.
//   A tensor op (GEMM/CONV2D/SOFTMAX/ATTENTION + FUSE_*) reaching EX pulses the
//   selected unit's `start`, then the pipeline STALLS IF/ID/EX (freeze upstream,
//   bubble downstream) on a variable-length stall until the unit raises `done`.
//   The unit writes its result tile(s) to TM via its own TM write port.  An
//   explicit DRAIN cycle after `done` honours "result lands in TM the cycle
//   AFTER done" before any following op reads it, and is where the FUSE_*_RELU
//   tile post-op runs (one result line per cycle through fused_ops_unit).
//   The tensor op then retires through MEM/WB writing its STATUS word to rC.
//
// TM PORT ARBITRATION
//   Only one TM consumer is live at a time (the busy stall serializes tensor
//   ops; TLOAD/TSTORE are scalar ops that fully retire), so the shared 2R+1W TM
//   ports are a simple mux: the active tensor unit during its run/drain, else
//   the scalar TLOAD/TSTORE MEM path.
//
// STATUS REGISTER (SPEC §4) {sat[0],unit[3:1],argmax[7:4],illegal[8]}.
//   Tensor `done` updates sat(sticky)/unit/argmax; an illegal decode sets the
//   sticky illegal bit; RDSTATUS reads it to rC; CLRSTATUS clears the sticky
//   bits.  result_out is a debug view of the committed WB word.
//
// SYNTHESIS: synchronous reset on all pipeline state; write-enables gated by
//   control bits; no latches; no comb loops; file name == module name; scoped
//   lint_off only, documented inline.
//============================================================================
// DECLFILENAME is intentionally suppressed for this ONE module: the SPEC/ISA
// mandate the top module be named `TPU` while the file is `tpu_top.v` (the
// conventional top-level filename).  Every OTHER source file is name==module.
/* verilator lint_off DECLFILENAME */
module TPU (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] instruction_in,
    output wire [31:0] result_out,
    output wire        illegal_opcode,

    // Debug observability (pure views of internal state; not arch-required).
    output wire [`ST_W-1:0] dbg_status,      // live status register word
    output wire             dbg_pipe_stall   // pipeline is stalled this cycle
);

    // ======================================================================
    // Forward declarations of the global stall signals (used by IF / ID/EX).
    // ======================================================================
    wire stall;        // load-use stall  (freeze IF/ID, bubble EX)
    wire tstall;       // tensor busy stall (freeze IF/ID/EX, bubble MEM)
    wire mem_stall;    // MEM multi-cycle stall (DMA / TLOAD / TSTORE)
    // Upstream (IF, ID/EX) freezes on ANY of these; the per-stage bubble policy
    // differs and is handled at each register.
    wire hold_front = stall || tstall || mem_stall;

    // ======================================================================
    // 0. Decoder (combinational field extraction, shared leaf unit).
    // ======================================================================
    reg  [31:0] instr_IF_ID;

    wire [`OPCODE_W-1:0] opcode_dec;
    wire [`RF_IDX_W-1:0] rA_dec, rB_dec, rC_dec, rC_i_dec;
    wire [`IMM12_W-1:0]  imm12_dec;
    wire [`XLEN-1:0]     imm12_sext_dec, imm20_sext_dec;
    wire [`IMM20_W-1:0]  imm20_dec;
    wire [`TM_IDX_W-1:0] tile_base_dec, conv_out_base_dec;
    wire [1:0]           conv_stride_dec, conv_pad_dec, gs_stride_exp_dec;
    // The decoder exposes the full 12-bit DMA length field.  A 256-word DMEM
    // bounds a meaningful copy to [0,256]; the full 12-bit value IS consumed by
    // the saturating clamp at the ID/EX latch (§7) so an over-long encoding maps
    // to DMEM_DEPTH instead of wrapping.  (No lint_off needed -- fully used.)
    wire [`IMM12_W-1:0]  dma_len_dec;

    instruction_decoder decoder (
        .instruction   (instr_IF_ID),
        .opcode        (opcode_dec),
        .rA            (rA_dec),
        .rB            (rB_dec),
        .rC            (rC_dec),
        .rC_i          (rC_i_dec),
        .imm12         (imm12_dec),
        .imm12_sext    (imm12_sext_dec),
        .imm20         (imm20_dec),
        .imm20_sext    (imm20_sext_dec),
        .tile_base     (tile_base_dec),
        .conv_out_base (conv_out_base_dec),
        .conv_stride   (conv_stride_dec),
        .conv_pad      (conv_pad_dec),
        .gs_stride_exp (gs_stride_exp_dec),
        .dma_len       (dma_len_dec)
    );

    // imm20[19:16] alias the rC_i bits (consumed via rC_i); imm12 raw and the
    // never-selected conv/gs slices are fully consumed across the opcode set.
    /* verilator lint_off UNUSEDSIGNAL */
    wire [`IMM20_W-1:0] _u_imm20 = imm20_dec;
    wire [`IMM12_W-1:0] _u_imm12 = imm12_dec;
    /* verilator lint_on UNUSEDSIGNAL */

    // ======================================================================
    // 1. CONTROL DECODER  --  per-opcode side-effect classification.
    // ======================================================================
    localparam [2:0] T_NONE=3'd0, T_GEMM=3'd1, T_CONV=3'd2, T_SOFT=3'd3, T_ATTN=3'd4;
    localparam [2:0] M_NONE=3'd0, M_LOAD=3'd1, M_STORE=3'd2, M_DMA=3'd3,
                     M_GATHER=3'd4, M_SCATTER=3'd5, M_TLOAD=3'd6, M_TSTORE=3'd7;

    reg        writes_reg_dec, mem_read_dec, mem_write_dec, uses_tensor_dec;
    reg        illegal_dec, fuse_relu_dec, is_loadi_dec;
    reg  [2:0] tunit_dec, mem_kind_dec;

    always @(*) begin
        writes_reg_dec=1'b0; mem_read_dec=1'b0; mem_write_dec=1'b0;
        uses_tensor_dec=1'b0; illegal_dec=1'b0; fuse_relu_dec=1'b0;
        is_loadi_dec=1'b0; tunit_dec=T_NONE; mem_kind_dec=M_NONE;
        case (opcode_dec)
            `OP_NOP:        ;
            `OP_LOADI:      begin writes_reg_dec=1'b1; is_loadi_dec=1'b1; end
            `OP_LOAD:       begin writes_reg_dec=1'b1; mem_read_dec=1'b1;
                                  mem_kind_dec=M_LOAD; end
            `OP_STORE:      begin mem_write_dec=1'b1; mem_kind_dec=M_STORE; end
            `OP_ADD,`OP_SUB,`OP_AND,`OP_OR,`OP_XOR,`OP_SHL,`OP_SHR,
            `OP_RELU,`OP_ADDI,`OP_RDSTATUS:
                            writes_reg_dec=1'b1;
            `OP_CLRSTATUS:  ;
            `OP_DMA:        begin writes_reg_dec=1'b1; mem_read_dec=1'b1;
                                  mem_write_dec=1'b1; mem_kind_dec=M_DMA; end
            `OP_GATHER:     begin writes_reg_dec=1'b1; mem_read_dec=1'b1;
                                  mem_kind_dec=M_GATHER; end
            `OP_SCATTER:    begin mem_write_dec=1'b1; mem_kind_dec=M_SCATTER; end
            `OP_TLOAD:      begin mem_read_dec=1'b1; mem_kind_dec=M_TLOAD; end
            `OP_TSTORE:     begin mem_write_dec=1'b1; mem_kind_dec=M_TSTORE; end
            `OP_GEMM:       begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_GEMM; end
            `OP_CONV2D:     begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_CONV; end
            `OP_SOFTMAX:    begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_SOFT; end
            `OP_ATTENTION:  begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_ATTN; end
            `OP_FUSE_GEMM_RELU: begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_GEMM; fuse_relu_dec=1'b1; end
            `OP_FUSE_CONV_RELU: begin writes_reg_dec=1'b1; uses_tensor_dec=1'b1;
                                  tunit_dec=T_CONV; fuse_relu_dec=1'b1; end
            default:        illegal_dec=1'b1;
        endcase
        if (illegal_dec) begin
            writes_reg_dec=1'b0; mem_read_dec=1'b0; mem_write_dec=1'b0;
            uses_tensor_dec=1'b0; tunit_dec=T_NONE; mem_kind_dec=M_NONE;
            fuse_relu_dec=1'b0;
        end
    end

    assign illegal_opcode = illegal_dec;

    // Destination register: LOADI uses I-format rC_i; others use R-format rC.
    wire [`RF_IDX_W-1:0] dst_dec = is_loadi_dec ? rC_i_dec : rC_dec;

    // ======================================================================
    // 2. Register file (combinational, 3 ports, r0 hardwired zero).
    //    Port 3 reads rC's current value (SCATTER store data).
    // ======================================================================
    wire [`XLEN-1:0]     rf_data1, rf_data2, rf_data3;
    wire                 wb_we;
    wire [`RF_IDX_W-1:0] wb_addr;
    wire [`XLEN-1:0]     wb_data;

    register_file regfile (
        .clk(clk), .rst(rst),
        .read_addr1(rA_dec), .read_addr2(rB_dec), .read_addr3(rC_dec),
        .write_addr(wb_addr), .write_data(wb_data), .write_enable(wb_we),
        .read_data1(rf_data1), .read_data2(rf_data2), .read_data3(rf_data3)
    );

    // ======================================================================
    // 3. Pipeline registers.
    // ======================================================================
    reg  [`XLEN-1:0] ex_value;       // EX combinational result (scalar)
    reg  [`XLEN-1:0] mem_value;      // MEM combinational result

    // ID/EX
    reg  [`OPCODE_W-1:0] opcode_ex;
    reg  [`RF_IDX_W-1:0] rC_ex, dst_ex;
    reg  [`XLEN-1:0]     opA_ex, opB_ex, opC_ex, imm_ex;
    reg  [`TM_IDX_W-1:0] tile_base_ex, conv_out_base_ex;
    reg  [1:0]           conv_stride_ex, conv_pad_ex, gs_exp_ex;
    reg  [`DMEM_ADDR_W:0] dma_len_ex;    // DMA copy length (9 bits, 0..256)
    reg                  writes_reg_ex, mem_read_ex, mem_write_ex, uses_tensor_ex;
    reg                  fuse_relu_ex;
    reg  [2:0]           tunit_ex, mem_kind_ex;

    // EX/MEM
    reg  [`OPCODE_W-1:0] opcode_mem;
    reg  [`RF_IDX_W-1:0] dst_mem;
    reg  [`XLEN-1:0]     ex_value_mem, opA_mem, opB_mem, opC_mem;
    reg  [1:0]           gs_exp_mem;
    reg  [`DMEM_ADDR_W:0] dma_len_mem;   // DMA copy length (0..DMEM_DEPTH, 9 bits)
    reg  [`TM_IDX_W-1:0] tile_base_mem;
    reg                  writes_reg_mem, mem_write_mem;
    reg  [2:0]           mem_kind_mem;

    // MEM/WB
    reg  [`RF_IDX_W-1:0] dst_wb;
    reg  [`XLEN-1:0]     wb_value;
    reg                  writes_reg_wb;

    // ======================================================================
    // 4. IF stage.
    // ======================================================================
    always @(posedge clk) begin
        if (rst)            instr_IF_ID <= 32'b0;
        else if (!hold_front) instr_IF_ID <= instruction_in;
    end

    // ======================================================================
    // 4b. Tensor-status machinery hoisted ABOVE the forwarding network.
    //   A retiring tensor op writes its STATUS word (sat/unit/argmax) to rC, NOT
    //   the EX-stage `ex_value` (which for a tensor opcode is the unused eff_addr
    //   default).  The forwarding muxes below must therefore forward
    //   `tensor_status_word` -- not `ex_value` -- when an EX-stage tensor producer
    //   matches a directly-dependent (gap=0) scalar consumer of rC.  Declaring the
    //   per-unit handshake wires, the active_* selection, and tensor_status_word
    //   here (before section 5) lets the EX-stage forward arms reference them.
    //   (The FSM that drives tphase/start/done lives in §8c; only the COMBINATIONAL
    //   status word is needed up here.)
    // ----------------------------------------------------------------------
    // Per-unit handshake wires (driven by the tensor unit instances in §8c).
    wire gemm_done, gemm_busy, gemm_sat;
    wire conv_done, conv_busy, conv_sat;
    wire soft_done, soft_busy, soft_sat;  wire [2:0] soft_argmax;
    wire attn_done, attn_busy, attn_sat;

    reg       active_done, active_sat;
    reg [2:0] active_argmax;
    always @(*) begin
        case (tunit_ex)
            T_GEMM: begin active_done=gemm_done; active_sat=gemm_sat; active_argmax=3'd0; end
            T_CONV: begin active_done=conv_done; active_sat=conv_sat; active_argmax=3'd0; end
            T_SOFT: begin active_done=soft_done; active_sat=soft_sat; active_argmax=soft_argmax; end
            T_ATTN: begin active_done=attn_done; active_sat=attn_sat; active_argmax=3'd0; end
            default:begin active_done=1'b1;      active_sat=1'b0;     active_argmax=3'd0; end
        endcase
    end

    // The scalar status word a tensor op retires to rC {sat,unit,argmax,illegal}.
    wire [`ST_W-1:0] tensor_status_word =
        pack_status(active_sat, tunit_to_id(tunit_ex), active_argmax, 1'b0);

    // The value an EX-stage producer forwards: a tensor op contributes its STATUS
    // word (the value it will actually write back to rC), every other op its
    // ordinary scalar ex_value.
    wire [`XLEN-1:0] ex_fwd_value = uses_tensor_ex ? tensor_status_word : ex_value;

    // ======================================================================
    // 5. Forwarding into the ID operand path.
    //   IMPLEMENTED AS EXPLICIT COMBINATIONAL MUXES (not a Verilog function): a
    //   function called from a continuous assign re-evaluates ONLY when its
    //   listed arguments change, so it would MISS updates to the live ex_value /
    //   pipeline-register producers it reads -- exactly the v1.5 pitfall.  These
    //   always @(*) blocks are sensitive to every producer signal.
    //   NOTE: an EX-stage tensor producer forwards `ex_fwd_value` (its status
    //   word), NOT the raw `ex_value` eff_addr default -- see §4b.
    // ======================================================================
    reg [`XLEN-1:0] opA_fwd, opB_fwd, opC_fwd;

    // Forward source A (rA_dec).
    always @(*) begin
        if (rA_dec == {`RF_IDX_W{1'b0}})
            opA_fwd = {`XLEN{1'b0}};                                  // r0 == 0
        else if (writes_reg_ex && !mem_read_ex && (dst_ex == rA_dec))
            opA_fwd = ex_fwd_value;
        else if (writes_reg_mem && (dst_mem == rA_dec))
            opA_fwd = mem_value;
        else if (writes_reg_wb && (dst_wb == rA_dec))
            opA_fwd = wb_value;
        else
            opA_fwd = rf_data1;
    end

    // Forward source B (rB_dec).
    always @(*) begin
        if (rB_dec == {`RF_IDX_W{1'b0}})
            opB_fwd = {`XLEN{1'b0}};
        else if (writes_reg_ex && !mem_read_ex && (dst_ex == rB_dec))
            opB_fwd = ex_fwd_value;
        else if (writes_reg_mem && (dst_mem == rB_dec))
            opB_fwd = mem_value;
        else if (writes_reg_wb && (dst_wb == rB_dec))
            opB_fwd = wb_value;
        else
            opB_fwd = rf_data2;
    end

    // Forward source C (rC_dec) -- SCATTER store data path.
    always @(*) begin
        if (rC_dec == {`RF_IDX_W{1'b0}})
            opC_fwd = {`XLEN{1'b0}};
        else if (writes_reg_ex && !mem_read_ex && (dst_ex == rC_dec))
            opC_fwd = ex_fwd_value;
        else if (writes_reg_mem && (dst_mem == rC_dec))
            opC_fwd = mem_value;
        else if (writes_reg_wb && (dst_wb == rC_dec))
            opC_fwd = wb_value;
        else
            opC_fwd = rf_data3;
    end

    // ======================================================================
    // 6. Load-use stall (LOAD/GATHER producer in EX, consumer in ID).
    //    TLOAD is mem_read but writes TM (not a reg producer: writes_reg_ex=0),
    //    so it never triggers this stall.
    // ======================================================================
    wire id_uses_rA = 1'b1;
    wire id_uses_rB = (opcode_dec==`OP_ADD)||(opcode_dec==`OP_SUB)||
                      (opcode_dec==`OP_AND)||(opcode_dec==`OP_OR)||
                      (opcode_dec==`OP_XOR)||(opcode_dec==`OP_SHL)||
                      (opcode_dec==`OP_SHR)||(opcode_dec==`OP_STORE)||
                      (opcode_dec==`OP_DMA)||(opcode_dec==`OP_GATHER)||
                      (opcode_dec==`OP_SCATTER)||(opcode_dec==`OP_GEMM)||
                      (opcode_dec==`OP_CONV2D)||(opcode_dec==`OP_ATTENTION)||
                      (opcode_dec==`OP_FUSE_GEMM_RELU)||
                      (opcode_dec==`OP_FUSE_CONV_RELU);
    // rC is read as a SOURCE by SCATTER (store data) and by ATTENTION (its
    // o_base = the VALUE of rC, wired to o_base=opC_ex below).  Both must be in
    // the load-use set so a LOAD/GATHER/DMA producer of rC inserts a bubble;
    // otherwise ATTENTION would latch a STALE o_base and write its context lines
    // to the wrong TM region (EX-forward is blocked for a mem_read producer, and
    // MEM/WB has not yet produced the value).
    wire id_uses_rC = (opcode_dec==`OP_SCATTER) || (opcode_dec==`OP_ATTENTION);

    assign stall = mem_read_ex && writes_reg_ex &&
                   ( (id_uses_rA && (rA_dec==dst_ex)) ||
                     (id_uses_rB && (rB_dec==dst_ex)) ||
                     (id_uses_rC && (rC_dec==dst_ex)) );

    // ======================================================================
    // 7. ID/EX register.  Stall priority (highest first):
    //    * rst                 : reset to NOP.
    //    * tstall || mem_stall : FREEZE EX -- HOLD the instruction in EX (a
    //      multi-cycle tensor op in EX, or a multi-cycle MEM op downstream).
    //      The EX instruction must wait, so we hold ALL ID/EX regs unchanged
    //      (NOT a bubble -- bubbling would lose the EX instruction).
    //    * stall (load-use)    : inject a NOP BUBBLE (the EX producer advances
    //      to MEM; the dependent ID instruction is replayed next cycle).
    //    * else                : advance normally.
    // ======================================================================
    always @(posedge clk) begin
        if (rst) begin
            opcode_ex<=`OP_NOP; rC_ex<=0; dst_ex<=0;
            opA_ex<=0; opB_ex<=0; opC_ex<=0; imm_ex<=0;
            tile_base_ex<=0; conv_out_base_ex<=0;
            conv_stride_ex<=0; conv_pad_ex<=0; gs_exp_ex<=0; dma_len_ex<=0;
            writes_reg_ex<=1'b0; mem_read_ex<=1'b0; mem_write_ex<=1'b0;
            uses_tensor_ex<=1'b0; fuse_relu_ex<=1'b0;
            tunit_ex<=T_NONE; mem_kind_ex<=M_NONE;
        end else if (tstall || mem_stall) begin
            // FREEZE: hold the instruction in EX (no change to any ID/EX reg).
            opcode_ex <= opcode_ex;
        end else if (stall) begin
            // Load-use BUBBLE: inject a NOP into EX.
            opcode_ex<=`OP_NOP; rC_ex<=0; dst_ex<=0;
            opA_ex<=0; opB_ex<=0; opC_ex<=0; imm_ex<=0;
            tile_base_ex<=0; conv_out_base_ex<=0;
            conv_stride_ex<=0; conv_pad_ex<=0; gs_exp_ex<=0; dma_len_ex<=0;
            writes_reg_ex<=1'b0; mem_read_ex<=1'b0; mem_write_ex<=1'b0;
            uses_tensor_ex<=1'b0; fuse_relu_ex<=1'b0;
            tunit_ex<=T_NONE; mem_kind_ex<=M_NONE;
        end else begin
            opcode_ex        <= opcode_dec;
            rC_ex            <= rC_dec;
            dst_ex           <= dst_dec;
            opA_ex           <= opA_fwd;
            opB_ex           <= opB_fwd;
            opC_ex           <= opC_fwd;
            imm_ex           <= is_loadi_dec ? imm20_sext_dec : imm12_sext_dec;
            tile_base_ex     <= tile_base_dec;
            conv_out_base_ex <= conv_out_base_dec;
            conv_stride_ex   <= conv_stride_dec;
            conv_pad_ex      <= conv_pad_dec;
            gs_exp_ex        <= gs_stride_exp_dec;
            // DMA length: the ISA exposes a 12-bit imm12 length, but a 256-word
            // DMEM bounds any meaningful copy to [0,256].  SATURATE (not wrap):
            // a len > DMEM_DEPTH clamps to DMEM_DEPTH so an over-long encoding
            // does a full-memory copy instead of silently wrapping mod 512.
            dma_len_ex       <= (dma_len_dec > `IMM12_W'(`DMEM_DEPTH))
                                  ? (`DMEM_ADDR_W+1)'(`DMEM_DEPTH)
                                  : dma_len_dec[`DMEM_ADDR_W:0];
            writes_reg_ex    <= writes_reg_dec;
            mem_read_ex      <= mem_read_dec;
            mem_write_ex     <= mem_write_dec;
            uses_tensor_ex   <= uses_tensor_dec;
            fuse_relu_ex     <= fuse_relu_dec;
            tunit_ex         <= tunit_dec;
            mem_kind_ex      <= mem_kind_dec;
        end
    end

    // ======================================================================
    // 8. EX stage.
    // ======================================================================
    wire [`XLEN-1:0] alu_in2 = (opcode_ex==`OP_ADDI) ? imm_ex : opB_ex;
    wire [`XLEN-1:0] alu_out;
    vector_alu alu (.opcode(opcode_ex), .in1(opA_ex), .in2(alu_in2), .out(alu_out));

    wire [`XLEN-1:0] eff_addr = opA_ex + imm_ex;   // LOAD/STORE displacement

    reg  [`ST_W-1:0] status_reg;     // status register (updated in §12)

    always @(*) begin
        case (opcode_ex)
            `OP_LOADI:    ex_value = imm_ex;
            `OP_RDSTATUS: ex_value = status_reg;
            `OP_ADD,`OP_SUB,`OP_AND,`OP_OR,`OP_XOR,`OP_SHL,`OP_SHR,
            `OP_RELU,`OP_ADDI:
                          ex_value = alu_out;
            default:      ex_value = eff_addr;   // LOAD/STORE/GATHER resolve later
        endcase
    end

    // --------------------------------------------------------------------
    // 8c. TENSOR DISPATCH FSM + busy stall.
    //   PH_GO    : pulse the selected unit's start.
    //   PH_RUN   : wait for done.
    //   PH_DRAIN : 1 cycle so the last synchronous TM write has LANDED, plus the
    //              FUSE_*_RELU line sweep when applicable.
    //   tstall holds IF/ID/EX through GO+RUN+DRAIN.
    // --------------------------------------------------------------------
    localparam [1:0] PH_IDLE=2'd0, PH_GO=2'd1, PH_RUN=2'd2, PH_DRAIN=2'd3;
    reg [1:0] tphase;

    wire tensor_active = uses_tensor_ex;

    // Per-unit handshake wires + the active_* selection mux + tensor_status_word
    // are declared in §4b (hoisted above the forwarding network so an EX-stage
    // tensor producer can forward its status word).

    // FUSE post-op sweep state.  fuse_li/fuse_nlines are 5-bit so they can hold
    // the CONV output-line count (up to 16 for pad1/stride1 8x8); a 3-bit width
    // (max 7) could not address the default 6x6's 9 lines.
    reg                 fuse_run;
    reg [`TM_IDX_W-1:0] fuse_base;
    reg [4:0]           fuse_li, fuse_nlines;

    // PH_DRAIN sub-sequence.  A tensor unit may still be DRIVING its tail TM
    // writes for a couple of cycles after `done` (e.g. GEMM drives its last C
    // row the cycle AFTER done; it lands the following cycle).  We therefore
    // hold a fixed SETTLE window (DRAIN_SETTLE cycles) during which the tensor
    // unit -- not the fuse logic -- owns the TM write port, so every tail write
    // lands.  Only AFTER the settle window does the FUSE_*_RELU sweep run.
    localparam integer DRAIN_SETTLE = 2;   // cycles for the last tensor write to land
    reg [1:0] drain_cnt;                    // counts settle cycles in PH_DRAIN
    wire settle_done = (drain_cnt >= DRAIN_SETTLE[1:0]);

    // Drain is fully complete once the settle window has elapsed AND, for a
    // FUSE_*_RELU op, its tile sweep has both STARTED and FINISHED (fuse_nlines
    // becomes non-zero when the sweep starts and fuse_run drops when it ends).
    // A plain (non-fused) tensor op completes draining at settle_done.
    wire fuse_pending  = uses_tensor_ex && fuse_relu_ex;
    wire fuse_finished = (fuse_nlines != 5'd0) && !fuse_run;
    wire drain_done = settle_done && (!fuse_pending ? 1'b1 : fuse_finished);

    // TM-PORT MUTUAL EXCLUSION (fix for the TLOAD/TSTORE<->tensor TM-port race).
    //   A multi-cycle TLOAD/TSTORE occupies the shared TM port across ALL of its
    //   word cycles INCLUDING its final cycle (TLOAD's TM write at tt_cnt==4,
    //   where mem_stall has already deasserted -- so gating on mem_stall alone is
    //   insufficient).  While such an op is in MEM, the tensor dispatch FSM must
    //   NOT leave PH_IDLE (so it neither pulses `start` nor claims the TM port),
    //   and the TM muxes (below) keep the scalar TLOAD/TSTORE path in ownership.
    //   The tensor op simply waits in EX (tstall stays asserted because
    //   tphase!=PH_DRAIN) until the TM-using MEM op fully retires.  This restores
    //   the documented "only one TM consumer is live at a time" invariant for the
    //   natural "TLOAD an operand tile, then GEMM/CONV/SOFTMAX/ATTENTION" pattern.
    wire tm_scalar_busy = (mem_kind_mem==M_TLOAD) || (mem_kind_mem==M_TSTORE);

    assign tstall = tensor_active && !(tphase==PH_DRAIN && drain_done);

    always @(posedge clk) begin
        if (rst) begin
            tphase <= PH_IDLE; drain_cnt <= 2'd0;
        end else case (tphase)
            // Hold dispatch in IDLE while a TLOAD/TSTORE still owns the TM port.
            PH_IDLE:  begin if (tensor_active && !tm_scalar_busy) tphase <= PH_GO;
                            drain_cnt <= 2'd0; end
            PH_GO:    tphase <= PH_RUN;
            PH_RUN:   if (active_done) begin tphase <= PH_DRAIN; drain_cnt <= 2'd0; end
            PH_DRAIN: begin
                          if (!settle_done) drain_cnt <= drain_cnt + 2'd1;
                          if (drain_done)   tphase    <= PH_IDLE;
                      end
            default:  tphase <= PH_IDLE;
        endcase
    end

    wire tstart = (tphase==PH_GO);
    wire gemm_start = tstart && (tunit_ex==T_GEMM);
    wire conv_start = tstart && (tunit_ex==T_CONV);
    wire soft_start = tstart && (tunit_ex==T_SOFT);
    wire attn_start = tstart && (tunit_ex==T_ATTN);

    /* verilator lint_off UNUSEDSIGNAL */
    wire _u_tbusy = gemm_busy|conv_busy|soft_busy|attn_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    // --------------------------------------------------------------------
    // 8c-i. TM port mux.
    // --------------------------------------------------------------------
    wire [`TM_IDX_W-1:0] gemm_ra1, gemm_ra2, gemm_wa;
    wire [`LINE_W-1:0]   gemm_wd;  wire gemm_we;
    wire [`TM_IDX_W-1:0] conv_ra,  conv_wa;
    wire [`LINE_W-1:0]   conv_wd;  wire conv_we;
    wire [`TM_IDX_W-1:0] soft_ra,  soft_wa;
    wire [`LINE_W-1:0]   soft_wd;  wire soft_we;
    wire [`TM_IDX_W-1:0] attn_ra,  attn_wa;
    wire [`LINE_W-1:0]   attn_wd;  wire attn_we;

    wire [`LINE_W-1:0]   tm_rdata1, tm_rdata2;

    // Scalar TLOAD/TSTORE TM access.
    wire [`TM_IDX_W-1:0] tt_tm_raddr;   // TSTORE TM read line
    wire                 tl_we;         // TLOAD TM write enable
    wire [`TM_IDX_W-1:0] tl_waddr;      // TLOAD TM write line
    wire [`LINE_W-1:0]   tl_wdata;      // assembled TLOAD line

    // FUSE post-op.
    // fuse_li is 5-bit (== TM_IDX_W), so it indexes any TM line directly.
    wire [`TM_IDX_W-1:0] fuse_raddr = fuse_base + fuse_li;
    wire                 fuse_we    = fuse_run;
    wire [`LINE_W-1:0]   fuse_relu_line;

    // TM-port ownership.  A tensor op owns the TM port ONLY when no TLOAD/TSTORE
    // is draining in MEM (tm_scalar_busy).  While a scalar TM op is in MEM, the
    // tensor FSM is held in PH_IDLE (above), so it is not yet reading/writing TM;
    // the scalar TLOAD/TSTORE path keeps ownership for ALL of its word cycles
    // (including the final tt_cnt==4 write).  This is the structural enforcement
    // of the documented one-live-TM-consumer invariant.
    wire tensor_owns_tm = uses_tensor_ex && !tm_scalar_busy;

    // -- TM read port 1 --
    reg [`TM_IDX_W-1:0] tm_raddr1;
    always @(*) begin
        if (tensor_owns_tm) begin
            if (fuse_run) tm_raddr1 = fuse_raddr;
            else case (tunit_ex)
                T_GEMM: tm_raddr1 = gemm_ra1;
                T_CONV: tm_raddr1 = conv_ra;
                T_SOFT: tm_raddr1 = soft_ra;
                T_ATTN: tm_raddr1 = attn_ra;
                default:tm_raddr1 = {`TM_IDX_W{1'b0}};
            endcase
        end else begin
            tm_raddr1 = tt_tm_raddr;     // TSTORE read path
        end
    end

    // -- TM read port 2 (GEMM B row only) --
    wire [`TM_IDX_W-1:0] tm_raddr2 = (tensor_owns_tm && tunit_ex==T_GEMM && !fuse_run)
                                   ? gemm_ra2 : {`TM_IDX_W{1'b0}};

    // -- TM write port --
    reg                 tm_we;
    reg [`TM_IDX_W-1:0] tm_waddr;
    reg [`LINE_W-1:0]   tm_wdata;
    always @(*) begin
        if (tensor_owns_tm) begin
            if (fuse_run) begin
                tm_we=fuse_we; tm_waddr=fuse_raddr; tm_wdata=fuse_relu_line;
            end else case (tunit_ex)
                T_GEMM: begin tm_we=gemm_we; tm_waddr=gemm_wa; tm_wdata=gemm_wd; end
                T_CONV: begin tm_we=conv_we; tm_waddr=conv_wa; tm_wdata=conv_wd; end
                T_SOFT: begin tm_we=soft_we; tm_waddr=soft_wa; tm_wdata=soft_wd; end
                T_ATTN: begin tm_we=attn_we; tm_waddr=attn_wa; tm_wdata=attn_wd; end
                default:begin tm_we=1'b0; tm_waddr=0; tm_wdata=0; end
            endcase
        end else begin
            tm_we=tl_we; tm_waddr=tl_waddr; tm_wdata=tl_wdata;   // TLOAD write
        end
    end

    tile_memory tmem (
        .clk(clk), .rst(rst),
        .raddr1(tm_raddr1), .rdata1(tm_rdata1),
        .raddr2(tm_raddr2), .rdata2(tm_rdata2),
        .we(tm_we), .waddr(tm_waddr), .wdata(tm_wdata)
    );

    // ---- tensor units ----
    gemm_systolic u_gemm (
        .clk(clk), .rst(rst), .start(gemm_start),
        .a_base(opA_ex[`TM_IDX_W-1:0]), .b_base(opB_ex[`TM_IDX_W-1:0]),
        .c_base(tile_base_ex),
        .busy(gemm_busy), .done(gemm_done), .sat(gemm_sat),
        .tm_raddr1(gemm_ra1), .tm_rdata1(tm_rdata1),
        .tm_raddr2(gemm_ra2), .tm_rdata2(tm_rdata2),
        .tm_we(gemm_we), .tm_waddr(gemm_wa), .tm_wdata(gemm_wd)
    );
    conv2d_unit u_conv (
        .clk(clk), .rst(rst), .start(conv_start),
        .in_base(opA_ex[`TM_IDX_W-1:0]), .k_base(opB_ex[`TM_IDX_W-1:0]),
        .out_base(conv_out_base_ex), .stride(conv_stride_ex), .pad(conv_pad_ex),
        .busy(conv_busy), .done(conv_done), .sat(conv_sat),
        .rd_addr(conv_ra), .rd_data(tm_rdata1),
        .tm_we(conv_we), .tm_waddr(conv_wa), .tm_wdata(conv_wd)
    );
    softmax_unit u_soft (
        .clk(clk), .rst(rst), .start(soft_start),
        .x_base(opA_ex[`TM_IDX_W-1:0]), .p_base(tile_base_ex),
        .busy(soft_busy), .done(soft_done), .sat(soft_sat), .argmax(soft_argmax),
        .tm_raddr(soft_ra), .tm_rdata(tm_rdata1),
        .tm_we(soft_we), .tm_waddr(soft_wa), .tm_wdata(soft_wd)
    );
    attention_unit u_attn (
        .clk(clk), .rst(rst), .start(attn_start),
        .q_base(opA_ex[`TM_IDX_W-1:0]), .k_base(opB_ex[`TM_IDX_W-1:0]),
        .v_base(tile_base_ex), .o_base(opC_ex[`TM_IDX_W-1:0]),
        .busy(attn_busy), .done(attn_done), .sat(attn_sat),
        .tm_raddr(attn_ra), .tm_rdata(tm_rdata1),
        .tm_we(attn_we), .tm_waddr(attn_wa), .tm_wdata(attn_wd)
    );

    // ---- FUSE_*_RELU post-op (one line per drain cycle) ----
    // fused_ops_unit's `sat` is documented to be always 0 (RELU never adds new
    // saturation), so its output is intentionally unused here.
    /* verilator lint_off UNUSEDSIGNAL */
    wire fuse_sat_unused;
    /* verilator lint_on UNUSEDSIGNAL */
    // CONV2D packs its output DENSE-16 (2 px per 32-bit lane); GEMM packs one
    // Q7.8 sign-extended per 32-bit lane.  Drive the post-op's packing select and
    // opcode from the active tensor unit so RELU runs at the correct granularity.
    wire fuse_dense16 = (tunit_ex==T_CONV);
    wire [`OPCODE_W-1:0] fuse_opcode = (tunit_ex==T_CONV) ? `OP_FUSE_CONV_RELU
                                                          : `OP_FUSE_GEMM_RELU;
    fused_ops_unit u_fused (
        .opcode(fuse_opcode), .dense16(fuse_dense16), .tile_in(tm_rdata1),
        .tile_out(fuse_relu_line), .sat(fuse_sat_unused)
    );

    // GEMM C base = tile_base_ex (4 lines, 4x4 Q7.8 one row/line).
    // CONV out base = conv_out_base_ex; the conv output occupies ceil(od*od/4)
    // TM lines where od = floor((H+2*pad-K)/stride)+1 (mirrors conv2d_unit.v):
    //   pad0/stride1 -> 6x6 -> 36 px -> 9 lines (the committed default)
    //   pad1/stride1 -> 8x8 -> 64 px -> 16 lines
    //   pad0/stride2 -> 3x3 ->  9 px -> 3 lines
    //   pad1/stride2 -> 4x4 -> 16 px -> 4 lines
    wire [`TM_IDX_W-1:0] fuse_tile_base  = (tunit_ex==T_GEMM) ? tile_base_ex
                                                              : conv_out_base_ex;
    // Derive the CONV output-line count from the same stride/pad fields the conv
    // unit uses.  od in {3,4,6,8}; npix=od*od in {9,16,36,64}; lines=ceil(npix/4).
    wire [3:0] conv_od    = (conv_pad_ex != 2'd0)
                          ? ((conv_stride_ex == 2'd2) ? 4'd4 : 4'd8)   // pad=1
                          : ((conv_stride_ex == 2'd2) ? 4'd3 : 4'd6);  // pad=0
    wire [6:0] conv_npix  = conv_od * conv_od;                          // 9..64
    wire [4:0] conv_lines = conv_npix[6:2] + (|conv_npix[1:0] ? 5'd1 : 5'd0);
    wire [4:0] fuse_tile_lines = (tunit_ex==T_GEMM) ? 5'd4 : conv_lines;

    // Begin the FUSE sweep only AFTER the settle window (so the tensor unit's
    // tail TM writes have all landed), then walk one result line per cycle.
    wire fuse_sweep_begin = uses_tensor_ex && fuse_relu_ex &&
                            (tphase==PH_DRAIN) && settle_done && !fuse_run &&
                            (fuse_nlines == 5'd0);
    always @(posedge clk) begin
        if (rst) begin
            fuse_run<=1'b0; fuse_base<=0; fuse_li<=5'd0; fuse_nlines<=5'd0;
        end else if (!uses_tensor_ex || !fuse_relu_ex) begin
            // Reset the sweep bookkeeping between fused ops.
            fuse_run<=1'b0; fuse_li<=5'd0; fuse_nlines<=5'd0;
        end else if (fuse_sweep_begin) begin
            fuse_run<=1'b1; fuse_base<=fuse_tile_base; fuse_li<=5'd0;
            fuse_nlines<=fuse_tile_lines;
        end else if (fuse_run) begin
            if (fuse_li + 5'd1 >= fuse_nlines) fuse_run<=1'b0;
            fuse_li <= fuse_li + 5'd1;
        end
    end

    // ======================================================================
    // 9. EX/MEM register.
    //    While tstall holds (tensor running) bubble downstream; on the cycle the
    //    tensor op finishes DRAIN it retires carrying its STATUS word to rC.
    // ======================================================================
    wire tensor_retire = uses_tensor_ex && (tphase==PH_DRAIN) && drain_done;
    // tensor_status_word is declared in §4b (hoisted for the EX-stage forward).

    always @(posedge clk) begin
        if (rst) begin
            opcode_mem<=`OP_NOP; dst_mem<=0; ex_value_mem<=0;
            opA_mem<=0; opB_mem<=0; opC_mem<=0;
            gs_exp_mem<=0; dma_len_mem<=0; tile_base_mem<=0;
            writes_reg_mem<=1'b0; mem_write_mem<=1'b0; mem_kind_mem<=M_NONE;
        end else if (mem_stall) begin
            // Hold the EX/MEM op in place while MEM completes a multi-cycle xfer.
            opcode_mem <= opcode_mem;
        end else if (tensor_active && !tensor_retire) begin
            // Bubble while the tensor op is still running.
            opcode_mem<=`OP_NOP; dst_mem<=0; ex_value_mem<=0;
            writes_reg_mem<=1'b0; mem_write_mem<=1'b0; mem_kind_mem<=M_NONE;
        end else if (tensor_retire) begin
            // Retiring tensor op: status writeback to rC only.
            opcode_mem<=opcode_ex; dst_mem<=rC_ex; ex_value_mem<=tensor_status_word;
            opA_mem<=0; opB_mem<=0; opC_mem<=0;
            gs_exp_mem<=0; dma_len_mem<=0; tile_base_mem<=0;
            writes_reg_mem<=writes_reg_ex; mem_write_mem<=1'b0; mem_kind_mem<=M_NONE;
        end else begin
            opcode_mem<=opcode_ex; dst_mem<=dst_ex; ex_value_mem<=ex_value;
            opA_mem<=opA_ex; opB_mem<=opB_ex; opC_mem<=opC_ex;
            gs_exp_mem<=gs_exp_ex; dma_len_mem<=dma_len_ex; tile_base_mem<=tile_base_ex;
            writes_reg_mem<=writes_reg_ex; mem_write_mem<=mem_write_ex;
            mem_kind_mem<=mem_kind_ex;
        end
    end

    // ======================================================================
    // 10. MEM stage.
    // ======================================================================
    wire [`XLEN-1:0]        dmem_rdata1, dmem_rdata2;

    // ---- DMA engine ----
    wire                    dma_busy, dma_done;
    wire [`DMEM_ADDR_W:0]   dma_words;
    wire [`DMEM_ADDR_W-1:0] dma_rd_addr, dma_wr_addr;
    wire                    dma_wr_en;
    wire [`XLEN-1:0]        dma_wr_data;

    reg  dma_launched;
    wire is_dma_mem = (mem_kind_mem==M_DMA);
    wire dma_start  = is_dma_mem && !dma_launched && !dma_busy && !dma_done;
    always @(posedge clk) begin
        if (rst)            dma_launched<=1'b0;
        else if (dma_done)  dma_launched<=1'b0;
        else if (dma_start) dma_launched<=1'b1;
    end

    dma_controller u_dma (
        .clk(clk), .rst(rst), .start(dma_start),
        .src_addr(opA_mem[`DMEM_ADDR_W-1:0]),
        .dst_addr(opB_mem[`DMEM_ADDR_W-1:0]),
        .len(dma_len_mem),
        .busy(dma_busy), .done(dma_done), .words(dma_words),
        .rd_addr(dma_rd_addr), .rd_data(dmem_rdata2),
        .wr_en(dma_wr_en), .wr_addr(dma_wr_addr), .wr_data(dma_wr_data)
    );
    wire dma_stall = is_dma_mem && !dma_done;

    // ---- scatter_gather (combinational) ----
    wire [`XLEN-1:0]        sg_addr_out, sg_data_out, sg_wdata;
    wire [`DMEM_ADDR_W-1:0] sg_dmem_addr;
    wire                    sg_we;
    wire sg_is_scatter = (mem_kind_mem==M_SCATTER);
    wire sg_active     = (mem_kind_mem==M_GATHER)||(mem_kind_mem==M_SCATTER);
    scatter_gather u_sg (
        .start(sg_active), .is_scatter(sg_is_scatter),
        .base_addr(opA_mem), .index(opB_mem), .stride_exp(gs_exp_mem),
        .data_in(opC_mem), .dmem_rdata(dmem_rdata1),
        .addr_out(sg_addr_out), .dmem_addr(sg_dmem_addr),
        .dmem_wdata(sg_wdata), .dmem_we(sg_we), .data_out(sg_data_out)
    );

    /* verilator lint_off UNUSEDSIGNAL */
    wire [`XLEN-1:0]      _u_sg_addr   = sg_addr_out;
    wire [`DMEM_ADDR_W:0] _u_dma_words = dma_words;
    wire                  _u_dma_busy  = dma_busy;
    /* verilator lint_on UNUSEDSIGNAL */

    // ---- TLOAD / TSTORE 4-word line transfer (MEM-stage FSM) ----
    //   tt_cnt counts the word being processed THIS cycle: 0,1,2,3 then 4=done.
    //   The op is held in MEM (mem_stall) until tt_cnt reaches 4.
    //     TSTORE: each cycle tt_cnt in {0..3} writes DMEM[base+tt_cnt] from TM
    //             lane tt_cnt; at tt_cnt==4 the transfer is complete.
    //     TLOAD : each cycle tt_cnt in {0..3} captures DMEM[base+tt_cnt] into
    //             tt_line lane tt_cnt; at tt_cnt==4 the assembled line is
    //             written to TM (tl_we) and the transfer completes.
    reg  [2:0]         tt_cnt;     // 0..4 word index (this-cycle word)
    reg  [`LINE_W-1:0] tt_line;    // assembled TLOAD line
    wire is_tload_mem  = (mem_kind_mem==M_TLOAD);
    wire is_tstore_mem = (mem_kind_mem==M_TSTORE);
    wire is_tt_mem     = is_tload_mem || is_tstore_mem;

    wire [`DMEM_ADDR_W-1:0] tt_base = opA_mem[`DMEM_ADDR_W-1:0];   // DMEM base
    wire [`DMEM_ADDR_W-1:0] tt_addr = tt_base +
                                      {{(`DMEM_ADDR_W-3){1'b0}}, tt_cnt};

    // TSTORE reads the TM line named by imm12[4:0]=tile_base_mem (port 1).
    assign tt_tm_raddr = tile_base_mem;
    // TLOAD writes the assembled line to TM line tile_base_mem at tt_cnt==4.
    assign tl_waddr = tile_base_mem;
    assign tl_wdata = tt_line;
    assign tl_we    = is_tload_mem && (tt_cnt==3'd4);

    // Stall while the transfer has words left (tt_cnt < 4).
    wire tt_stall = is_tt_mem && (tt_cnt != 3'd4);

    assign mem_stall = dma_stall || tt_stall;

    // ---- DMEM port mux ----
    wire [`XLEN-1:0] tstore_word = tm_rdata1[(tt_cnt[1:0])*`LANE_W +: `LANE_W];

    reg  [`DMEM_ADDR_W-1:0] dmem_addr1;
    reg  [`XLEN-1:0]        dmem_wdata1;
    reg                     dmem_we1;
    always @(*) begin
        dmem_addr1=ex_value_mem[`DMEM_ADDR_W-1:0]; dmem_wdata1=opB_mem; dmem_we1=1'b0;
        if (is_dma_mem) begin
            dmem_addr1=dma_wr_addr; dmem_wdata1=dma_wr_data; dmem_we1=dma_wr_en;
        end else if (sg_active) begin
            dmem_addr1=sg_dmem_addr; dmem_wdata1=sg_wdata; dmem_we1=sg_we;
        end else if (is_tstore_mem) begin
            // Write the current TM lane to DMEM[base+tt_cnt] each word cycle.
            dmem_addr1=tt_addr; dmem_wdata1=tstore_word;
            dmem_we1=(tt_cnt < 3'd4);
        end else if (is_tload_mem) begin
            dmem_addr1=tt_addr; dmem_wdata1={`XLEN{1'b0}}; dmem_we1=1'b0;
        end else if (mem_kind_mem==M_STORE) begin
            dmem_addr1=ex_value_mem[`DMEM_ADDR_W-1:0]; dmem_wdata1=opB_mem;
            dmem_we1=mem_write_mem;
        end else begin
            dmem_addr1=ex_value_mem[`DMEM_ADDR_W-1:0];
            dmem_wdata1={`XLEN{1'b0}}; dmem_we1=1'b0;
        end
    end

    wire [`DMEM_ADDR_W-1:0] dmem_addr2 = dma_rd_addr;

    memory u_dmem (
        .clk(clk), .rst(rst),
        .addr(dmem_addr1), .data_in(dmem_wdata1), .write_enable(dmem_we1),
        .data_out(dmem_rdata1),
        .raddr2(dmem_addr2), .rdata2(dmem_rdata2)
    );

    // TLOAD/TSTORE sequencer.  tt_cnt advances 0->1->2->3->4 while the op is in
    // MEM; on tt_cnt==4 the op is done (released next cycle) so reset tt_cnt.
    always @(posedge clk) begin
        if (rst) begin
            tt_cnt<=3'd0; tt_line<={`LINE_W{1'b0}};
        end else if (is_tt_mem) begin
            if (tt_cnt < 3'd4) begin
                if (is_tload_mem)
                    tt_line[tt_cnt[1:0]*`LANE_W +: `LANE_W] <= dmem_rdata1;
                tt_cnt <= tt_cnt + 3'd1;
            end else begin
                // tt_cnt==4: this op completes and leaves MEM next cycle; reset
                // the counter so a BACK-TO-BACK TLOAD/TSTORE starts fresh at 0.
                tt_cnt <= 3'd0;
            end
        end else begin
            tt_cnt<=3'd0;
        end
    end

    // MEM-stage result mux.
    always @(*) begin
        case (mem_kind_mem)
            M_LOAD:   mem_value = dmem_rdata1;
            M_GATHER: mem_value = sg_data_out;
            M_DMA:    mem_value = {{(`XLEN-(`DMEM_ADDR_W+1)){1'b0}}, dma_words};
            default:  mem_value = ex_value_mem;
        endcase
    end

    // ======================================================================
    // 11. MEM/WB register.
    // ======================================================================
    always @(posedge clk) begin
        if (rst) begin
            dst_wb<=0; wb_value<=0; writes_reg_wb<=1'b0;
        end else if (mem_stall) begin
            dst_wb<=0; wb_value<=0; writes_reg_wb<=1'b0;   // bubble during xfer
        end else begin
            dst_wb<=dst_mem; wb_value<=mem_value; writes_reg_wb<=writes_reg_mem;
        end
    end

    // ======================================================================
    // 12. WB stage + STATUS register update.
    // ======================================================================
    assign wb_we   = writes_reg_wb;
    assign wb_addr = dst_wb;
    assign wb_data = wb_value;
    assign result_out = wb_value;
    assign dbg_pipe_stall = hold_front;

    wire tensor_status_update = uses_tensor_ex && (tphase==PH_RUN) && active_done;
    wire clrstatus_now = (opcode_ex==`OP_CLRSTATUS) && !uses_tensor_ex;

    always @(posedge clk) begin
        if (rst) begin
            status_reg <= {`ST_W{1'b0}};
        end else begin
            if (illegal_dec)
                status_reg[`ST_ILLEGAL_BIT] <= 1'b1;
            if (tensor_status_update) begin
                status_reg[`ST_SAT_BIT] <= status_reg[`ST_SAT_BIT] | active_sat;
                status_reg[`ST_UNIT_HI:`ST_UNIT_LO] <= tunit_to_id(tunit_ex);
                status_reg[`ST_ARGMAX_HI:`ST_ARGMAX_LO] <= {1'b0, active_argmax};
            end
            if (clrstatus_now) begin
                status_reg[`ST_SAT_BIT]     <= 1'b0;
                status_reg[`ST_ILLEGAL_BIT] <= 1'b0;
            end
        end
    end

    assign dbg_status = status_reg;

    // ======================================================================
    // Helper functions.
    // ======================================================================
    function [`ST_W-1:0] pack_status;
        input s_sat; input [2:0] s_unit; input [2:0] s_argmax; input s_illegal;
        reg [`ST_W-1:0] w;
        begin
            w = {`ST_W{1'b0}};
            w[`ST_SAT_BIT]                 = s_sat;
            w[`ST_UNIT_HI:`ST_UNIT_LO]     = s_unit;
            w[`ST_ARGMAX_HI:`ST_ARGMAX_LO] = {1'b0, s_argmax};
            w[`ST_ILLEGAL_BIT]             = s_illegal;
            pack_status = w;
        end
    endfunction

    function [2:0] tunit_to_id;
        input [2:0] t;
        begin
            case (t)
                T_GEMM: tunit_to_id=`UNIT_GEMM;
                T_CONV: tunit_to_id=`UNIT_CONV;
                T_SOFT: tunit_to_id=`UNIT_SOFTMAX;
                T_ATTN: tunit_to_id=`UNIT_ATTENTION;
                default:tunit_to_id=`UNIT_NONE;
            endcase
        end
    endfunction

endmodule
/* verilator lint_on DECLFILENAME */
