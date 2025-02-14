
```
// Icarus Verilog(iverilog)를 설치
brew install icarus-verilog

// Icarus Verilog 컴파일 및 시뮬레이션
iverilog -o tpu_waveform.vcd \
    test/tpu_tb.v \
    src/tpu_top.v \
    src/instruction_decoder.v \
    src/register_file.v \
    src/memory.v \
    src/matrix_multiply.v \
    src/vector_alu.v \
    src/convolution_unit.v \
    src/attention_unit.v \
    src/dma_controller.v \
    src/scatter_gather.v \
    src/fused_ops_unit.v

vvp tpu_sim

// 파형 분석기 설치
brew install gtkwave

// 파형 분석(GTKWave)
gtkwave tpu_waveform.vcd
```