module TPU (
    input wire clk,
    input wire rst,
    input wire [31:0] instruction_in, // 외부에서 들어오는 명령어
    output wire [31:0] result_out     // 최종 결과 출력
);
    //--------------------------------------------------------------------------
    // 파이프라인 레지스터 선언
    //--------------------------------------------------------------------------
    // IF/ID
    reg [31:0] instr_IF_ID;

    // ID/EX
    reg [7:0]  opcode_ID_EX;
    reg [3:0]  regA_ID_EX;
    reg [3:0]  regB_ID_EX;
    reg [3:0]  regC_ID_EX;
    reg [31:0] regA_data_ID_EX;
    reg [31:0] regB_data_ID_EX;

    // EX/MEM
    reg [7:0]  opcode_EX_MEM;
    reg [3:0]  regC_EX_MEM;
    reg [31:0] ex_result_EX_MEM;

    // MEM/WB
    reg [3:0]  regC_MEM_WB;
    reg [31:0] mem_result_MEM_WB;

    //--------------------------------------------------------------------------
    // 모듈 간 연결선
    //--------------------------------------------------------------------------
    wire [7:0]  opcode_dec;
    wire [3:0]  regA_dec, regB_dec, regC_dec;
    wire [31:0] regA_data, regB_data;
    wire [31:0] mmu_result, conv_result, att_result, alu_result, fused_result;
    wire [31:0] dma_result, mem_out;
    wire [31:0] gather_scatter_result;
    
    //--------------------------------------------------------------------------
    // 1. IF 단계
    //--------------------------------------------------------------------------
    // 간단히 외부에서 들어온 instruction_in을 IF 단계에서 읽고,
    // IF/ID 파이프라인 레지스터에 저장
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            instr_IF_ID <= 32'b0;
        end else begin
            instr_IF_ID <= instruction_in; 
        end
    end

    //--------------------------------------------------------------------------
    // 2. ID 단계 (Instruction Decode)
    //--------------------------------------------------------------------------
    InstructionDecoder decoder (
        .instruction(instr_IF_ID),
        .opcode(opcode_dec),
        .regA(regA_dec),
        .regB(regB_dec),
        .regC(regC_dec)
    );

    // 레지스터 파일: ID 단계에서 읽기
    wire write_enable_wb;
    wire [31:0] write_data_wb;
    RegisterFile regfile (
        .clk(clk),
        .read_addr1(regA_dec),
        .read_addr2(regB_dec),
        .write_addr(regC_MEM_WB),
        .write_data(write_data_wb),
        .write_enable(write_enable_wb),
        .read_data1(regA_data),
        .read_data2(regB_data)
    );

    // ID/EX 레지스터
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            opcode_ID_EX      <= 8'b0;
            regA_ID_EX        <= 4'b0;
            regB_ID_EX        <= 4'b0;
            regC_ID_EX        <= 4'b0;
            regA_data_ID_EX   <= 32'b0;
            regB_data_ID_EX   <= 32'b0;
        end else begin
            opcode_ID_EX      <= opcode_dec;
            regA_ID_EX        <= regA_dec;
            regB_ID_EX        <= regB_dec;
            regC_ID_EX        <= regC_dec;
            regA_data_ID_EX   <= regA_data;
            regB_data_ID_EX   <= regB_data;
        end
    end

    //--------------------------------------------------------------------------
    // 3. EX 단계
    //--------------------------------------------------------------------------
    // EX 단계에서 다양한 연산(MMU, ALU, Convolution, Attention, DMA 등)을 수행
    // opcode 에 따라 결과를 선택

    // (1) 행렬 연산
    MatrixMultiplyUnit mmu (
        .clk(clk),
        .opcode(opcode_ID_EX),
        .A(regA_data_ID_EX),
        .B(regB_data_ID_EX),
        .result(mmu_result)
    );

    // (2) 벡터 ALU (ReLU, Add, Softmax 등)
    VectorALU alu (
        .clk(clk),
        .opcode(opcode_ID_EX),
        .in1(regA_data_ID_EX),
        .in2(regB_data_ID_EX),
        .out(alu_result)
    );

    // (3) 컨볼루션 유닛
    ConvolutionUnit conv (
        .clk(clk),
        .start(opcode_ID_EX == 8'h20 || opcode_ID_EX == 8'h31), // FUSE_CONV_RELU 예시
        .stride(2'd1),     // 예시: stride 1
        .padding(2'd0),    // 예시: padding 0
        .input_data(regA_data_ID_EX),
        .filter_data(regB_data_ID_EX),
        .output_data(conv_result)
    );

    // (4) Attention 유닛 (멀티헤드 확장 가능 예시)
    AttentionUnit att (
        .clk(clk),
        .start(opcode_ID_EX == 8'h21),
        .query(regA_data_ID_EX),
        .key(regB_data_ID_EX),
        .value(32'h00010001),  // 예: 고정 값 or 레지스터에서 입력받도록 확장 가능
        .output_data(att_result)
    );

    // (5) DMA & Scatter/Gather
    DMAController dma (
        .clk(clk),
        .src_addr(regA_data_ID_EX),
        .dst_addr(regB_data_ID_EX),
        .start(opcode_ID_EX == 8'h10),
        .data_out(dma_result)
    );

    ScatterGatherUnit sg_unit (
        .clk(clk),
        .start(opcode_ID_EX == 8'h11),
        .base_addr(regA_data_ID_EX),
        .index(regB_data_ID_EX),
        .data_out(gather_scatter_result)
    );

    // (6) Fused Ops (간단 예: FUSE_MMA_RELU, FUSE_CONV_RELU 등)
    FusedOpsUnit fused_ops (
        .clk(clk),
        .opcode(opcode_ID_EX),
        .inA(regA_data_ID_EX),
        .inB(regB_data_ID_EX),
        .mmu_out(mmu_result),
        .conv_out(conv_result),
        .alu_out(alu_result),
        .fused_out(fused_result)
    );

    // EX 단계 결과 선택 (우선순위 예시)
    reg [31:0] ex_stage_result;
    always @(*) begin
        case (opcode_ID_EX)
            8'h03, 8'h04: ex_stage_result = mmu_result;          // MatrixMultiply (FP16, INT8)
            8'h05, 8'h06, 8'h07: ex_stage_result = alu_result;   // ReLU, ADD, Softmax
            8'h20: ex_stage_result = conv_result;                // CONV2D
            8'h21: ex_stage_result = att_result;                 // ATTENTION
            8'h10: ex_stage_result = dma_result;                 // DMA
            8'h11: ex_stage_result = gather_scatter_result;      // Scatter/Gather
            8'h30, 8'h31: ex_stage_result = fused_result;        // FUSED 연산들
            default: ex_stage_result = 32'b0;
        endcase
    end

    // EX/MEM 파이프라인 레지스터
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            opcode_EX_MEM     <= 8'b0;
            regC_EX_MEM       <= 4'b0;
            ex_result_EX_MEM  <= 32'b0;
        end else begin
            opcode_EX_MEM     <= opcode_ID_EX;
            regC_EX_MEM       <= regC_ID_EX;
            ex_result_EX_MEM  <= ex_stage_result;
        end
    end

    //--------------------------------------------------------------------------
    // 4. MEM 단계 (SRAM, Prefetch, etc.)
    //--------------------------------------------------------------------------
    wire mem_read_enable  = (opcode_EX_MEM == 8'h01);
    wire mem_write_enable = (opcode_EX_MEM == 8'h02);

    Memory memory (
        .clk(clk),
        .addr(ex_result_EX_MEM[3:0]),   // 예: 4비트 주소
        .data_in(ex_result_EX_MEM),     // 쓰기 시
        .data_out(mem_out),
        .write_enable(mem_write_enable),
        .read_enable(mem_read_enable)
    );

    // MEM/WB 파이프라인 레지스터
    always @(posedge clk or posedge rst) begin
        if(rst) begin
            regC_MEM_WB         <= 4'b0;
            mem_result_MEM_WB   <= 32'b0;
        end else begin
            regC_MEM_WB         <= regC_EX_MEM;
            // 메모리 동작 결과 / 그대로 전달된 EX 결과 선택
            // opcode가 메모리 읽기/쓰기가 아닐 경우, ex_result_EX_MEM 그대로
            if(mem_read_enable)
                mem_result_MEM_WB <= mem_out;
            else
                mem_result_MEM_WB <= ex_result_EX_MEM;
        end
    end

    //--------------------------------------------------------------------------
    // 5. WB 단계
    //--------------------------------------------------------------------------
    // Write Back: 레지스터 파일에 결과 쓰기
    assign write_enable_wb = (opcode_EX_MEM != 8'h00); // 예: NOP(0x00)이 아니면 일단 쓰기
    assign write_data_wb   = mem_result_MEM_WB;

    //--------------------------------------------------------------------------
    // 최종 결과 출력 (디버깅/모니터링 용도)
    //--------------------------------------------------------------------------
    assign result_out = mem_result_MEM_WB;

endmodule