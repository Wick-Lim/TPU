module InstructionDecoder (
    input wire [31:0] instruction,
    output wire [7:0] opcode,
    output wire [3:0] regA,
    output wire [3:0] regB,
    output wire [3:0] regC
);
    assign opcode = instruction[31:24];
    assign regA   = instruction[23:20];
    assign regB   = instruction[19:16];
    assign regC   = instruction[15:12];
endmodule