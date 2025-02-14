module FusedOpsUnit (
    input wire clk,
    input wire [7:0] opcode,
    input wire [31:0] inA,   // 일반적으로는 더 많은 입력이 필요할 수 있음
    input wire [31:0] inB,
    input wire [31:0] mmu_out,
    input wire [31:0] conv_out,
    input wire [31:0] alu_out,
    output reg [31:0] fused_out
);
    always @(posedge clk) begin
        case (opcode)
            8'h30: begin
                // 예시: FUSE_MMA_RELU
                // mmu_out => ReLU
                if(mmu_out[31] == 1'b1)
                    fused_out <= 32'b0;
                else
                    fused_out <= mmu_out;
            end
            8'h31: begin
                // 예시: FUSE_CONV_RELU
                if(conv_out[31] == 1'b1)
                    fused_out <= 32'b0;
                else
                    fused_out <= conv_out;
            end
            8'h32: begin
                // 예시: FUSE_MMA_ADD (mmu_out + inB)
                fused_out <= mmu_out + inB;
            end
            default: begin
                // 기타 (확장 가능)
                fused_out <= alu_out;
            end
        endcase
    end
endmodule