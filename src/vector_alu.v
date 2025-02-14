module VectorALU (
    input wire clk,
    input wire [7:0] opcode,
    input wire [31:0] in1,
    input wire [31:0] in2,
    output reg [31:0] out
);
    reg [31:0] temp;

    always @(posedge clk) begin
        case (opcode)
            8'h05: begin
                // ReLU (단순: 음수면 0, 양수면 그대로)
                if(in1[31] == 1'b1) out <= 32'b0; 
                else out <= in1;
            end
            8'h06: begin
                // Add
                out <= in1 + in2;
            end
            8'h07: begin
                // Softmax (아주 간단히)
                // 실제론 여러 요소를 exp 후 총합으로 나누어야 함
                // 여기서는 단일 스칼라 x에 대해 x/(x+1)
                if(in1 == 32'b0) out <= 32'b0;
                else begin
                    // 정수 단순화된 softmax
                    temp = in1 + 32'd1;
                    out <= in1 / temp;
                end
            end
            default: out <= 32'b0;
        endcase
    end
endmodule