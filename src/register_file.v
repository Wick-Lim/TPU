module RegisterFile (
    input wire clk,
    input wire [3:0] read_addr1,
    input wire [3:0] read_addr2,
    input wire [3:0] write_addr,
    input wire [31:0] write_data,
    input wire write_enable,
    output reg [31:0] read_data1,
    output reg [31:0] read_data2
);
    reg [31:0] registers [15:0];
    integer i;

    // 초기화 (시뮬레이션 용도)
    initial begin
        for(i = 0; i < 16; i = i + 1) begin
            registers[i] = 32'b0;
        end
    end

    always @(posedge clk) begin
        // Write
        if (write_enable) begin
            registers[write_addr] <= write_data;
        end
        // Read
        read_data1 <= registers[read_addr1];
        read_data2 <= registers[read_addr2];
    end
endmodule