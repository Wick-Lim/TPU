module ConvolutionUnit (
    input wire clk,
    input wire start,
    input wire [1:0] stride,
    input wire [1:0] padding,
    input wire [31:0] input_data,
    input wire [31:0] filter_data,
    output reg [31:0] output_data
);
    integer i;
    reg [31:0] acc;

    always @(posedge clk) begin
        if (start) begin
            acc = 0;
            // 예시: input_data와 filter_data를 9개(3x3)라고 가정
            // 실제론 별도 메모리에서 가져올 것
            for (i = 0; i < 9; i = i + 1) begin
                acc = acc + (input_data[i*4 +: 4] * filter_data[i*4 +: 4]);
            end
            // stride, padding 고려 로직은 생략 (개념 표시용)
            // ...
            output_data <= acc;
        end
    end
endmodule