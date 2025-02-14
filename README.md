
```
// Icarus Verilog(iverilog)를 설치
brew install icarus-verilog

// Icarus Verilog 컴파일 및 시뮬레이션
iverilog -o tpu_sim \
    tpu_tb.v \
    tpu_top.v \
    instruction_decoder.v \
    register_file.v \
    memory.v \
    matrix_multiply.v \
    vector_alu.v \
    convolution_unit.v \
    attention_unit.v \
    dma_controller.v \
    scatter_gather.v \
    fused_ops_unit.v

vvp tpu_sim

// 파형 분석(GTKWave)
gtkwave tpu_waveform.vcd
```