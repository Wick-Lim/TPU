module AttentionUnit (
    input wire clk,
    input wire start,
    input wire [31:0] query,
    input wire [31:0] key,
    input wire [31:0] value,
    output reg [31:0] output_data
);
    reg [31:0] score;
    always @(posedge clk) begin
        if (start) begin
            // Scaled Dot-Product (단순화)
            score = (query * key) >> 8; 
            // 실제론 softmax 필요
            output_data = score * value;
        end
    end
endmodule