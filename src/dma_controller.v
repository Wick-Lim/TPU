module DMAController (
    input wire clk,
    input wire [31:0] src_addr,
    input wire [31:0] dst_addr,
    input wire start,
    output reg [31:0] data_out
);
    reg [31:0] dma_buffer;
    always @(posedge clk) begin
        if (start) begin
            // 단순히 src_addr를 data_out에 전달하는 예시
            // 실제론 메모리에서 읽어 dst_addr로 쓰는 로직 필요
            dma_buffer = src_addr;  
            data_out <= dma_buffer;
        end
    end
endmodule