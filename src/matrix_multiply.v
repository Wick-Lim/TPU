module MatrixMultiplyUnit (
    input wire clk,
    input wire [7:0] opcode,
    input wire [31:0] A,
    input wire [31:0] B,
    output reg [31:0] result
);
    // ✅ 블록 밖에서 모든 변수 선언
    integer i;
    reg [31:0] acc;
    reg [7:0] a_slice, b_slice;
    reg [15:0] A_low16, A_high16, B_low16, B_high16;

    always @(posedge clk) begin
        acc = 32'd0;

        case (opcode)
            // ✅ 전통 Verilog 규칙 준수: 블록 외부 변수 사용
            8'h03: begin
                A_low16  = A[15:0];
                A_high16 = A[31:16];
                B_low16  = B[15:0];
                B_high16 = B[31:16];

                acc = (A_low16 * B_low16) + (A_high16 * B_high16);
            end

            8'h04: begin
                for (i = 0; i < 4; i = i + 1) begin
                    case (i)
                        0: begin a_slice = A[7:0];   b_slice = B[7:0];   end
                        1: begin a_slice = A[15:8];  b_slice = B[15:8];  end
                        2: begin a_slice = A[23:16]; b_slice = B[23:16]; end
                        3: begin a_slice = A[31:24]; b_slice = B[31:24]; end
                    endcase

                    acc = acc + ({{24{a_slice[7]}}, a_slice} *
                                 {{24{b_slice[7]}}, b_slice});
                end
            end

            default: acc = 32'd0;
        endcase

        result <= acc;
    end
endmodule