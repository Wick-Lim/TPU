module ScatterGatherUnit (
    input wire clk,
    input wire start,
    input wire [31:0] base_addr,
    input wire [31:0] index,
    output reg [31:0] data_out
);
    reg [31:0] sg_buffer;

    always @(posedge clk) begin
        if (start) begin
            // index를 활용해 base_addr에서 분산/수집
            // 실제론 메모리 접근 로직 필요
            sg_buffer = base_addr + (index << 2); // 예: 4바이트 단위
            data_out <= sg_buffer;
        end
    end
endmodule