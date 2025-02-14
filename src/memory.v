module Memory (
    input wire clk,
    input wire [3:0] addr,
    input wire [31:0] data_in,
    output reg [31:0] data_out,
    input wire write_enable,
    input wire read_enable
);
    reg [31:0] sram [15:0];
    integer i;

    initial begin
        for(i = 0; i < 16; i = i + 1) begin
            sram[i] = 32'b0;
        end
    end

    always @(posedge clk) begin
        if(write_enable) begin
            sram[addr] <= data_in;
        end
        if(read_enable) begin
            data_out <= sram[addr];
        end
    end
endmodule