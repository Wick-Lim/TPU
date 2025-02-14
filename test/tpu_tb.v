module tpu_tb;
    reg clk;
    reg rst;
    reg [31:0] instruction_in;
    wire [31:0] result_out;

    // TPU 인스턴스
    TPU uut (
        .clk(clk),
        .rst(rst),
        .instruction_in(instruction_in),
        .result_out(result_out)
    );

    // 클록 생성
    always #5 clk = ~clk;

    initial begin
        $dumpfile("tpu_waveform.vcd");
        $dumpvars(0, tpu_tb);

        clk = 0; 
        rst = 1; 
        #10 rst = 0;

        // 1. FP16 행렬 곱 (WMMA)
        instruction_in = 32'h0301_2000; // opcode=0x03, regA=1, regB=2, regC=0
        #20;
        $display("[Test1] FP16 MMA Result: %h", result_out);

        // 2. INT8 행렬 곱 (INT8_DP)
        instruction_in = 32'h0401_2000; // opcode=0x04
        #20;
        $display("[Test2] INT8 MMA Result: %h", result_out);

        // 3. FUSED_MMA_RELU (opcode=0x30)
        instruction_in = 32'h3001_2000;
        #20;
        $display("[Test3] FUSE_MMA_RELU Result: %h", result_out);

        // 4. 컨볼루션 (CONV2D)
        instruction_in = 32'h2001_2000; // opcode=0x20
        #20;
        $display("[Test4] CONV2D Result: %h", result_out);

        // 5. Attention
        instruction_in = 32'h2101_2000; // opcode=0x21
        #20;
        $display("[Test5] ATTENTION Result: %h", result_out);

        // 6. Scatter/Gather
        instruction_in = 32'h1101_2000; // opcode=0x11
        #20;
        $display("[Test6] Scatter/Gather Result: %h", result_out);

        // 7. DMA
        instruction_in = 32'h1001_2000; // opcode=0x10
        #20;
        $display("[Test7] DMA Result: %h", result_out);

        #50;
        $finish;
    end
endmodule