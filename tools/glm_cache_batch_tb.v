`timescale 1ns/1ps
//============================================================================
// glm_cache_batch_tb.v  --  PART A: BATCHING is the lever (NO RTL change).
//----------------------------------------------------------------------------
// Feeds three GLM-scale routing traces -- decode (batch=1), batch8, batch32 --
// through the COMMITTED src/expert_cache_ctrl at SLOTS=900 (~34 GB HBM) and
// asserts hit_count/miss_count match the python EXACT-LRU reference bit-for-bit.
// Same router picks in all three (route_trace.py rebuilds an identical top-k
// `sel` from the same seed); ONLY the access ORDER differs, so the rising hit
// rate decode->batch8->batch32 is the pure effect of batching.
//
//   Regenerate traces+refs first:  python3 tools/route_trace.py --dump
//   iverilog -g2012 -Wall -I src -o /tmp/glmbatch \
//            tools/glm_cache_batch_tb.v src/expert_cache_ctrl.v && vvp /tmp/glmbatch
//
// PYREF @ slots=900 :  decode 4771/13229 (26.51%)
//                      batch8 5341/12659 (29.67%)
//                      batch32 9090/8910 (50.50%)
//============================================================================
module glm_cache_batch_tb;
  localparam integer SLOTS=900, NE=19200, FL=2;
  localparam integer IDW=15;           // clog2(19200)
  localparam integer SLW=10;           // clog2(900)
  localparam integer N=18000;

  reg clk=1'b0, rst=1'b1, req_valid=1'b0, flash_done=1'b0;
  reg [IDW-1:0] req_expert_id=0;
  wire resp_valid, hit, busy, flash_req;
  wire [IDW-1:0] flash_expert_id;
  wire [SLW-1:0] resp_slot;
  wire [31:0] hit_count, miss_count;

  expert_cache_ctrl #(.SLOTS(SLOTS), .N_EXPERT(NE), .FLASH_LAT(FL)) dut (
    .clk(clk), .rst(rst), .req_valid(req_valid), .req_expert_id(req_expert_id),
    .resp_valid(resp_valid), .hit(hit), .resp_slot(resp_slot), .busy(busy),
    .flash_req(flash_req), .flash_expert_id(flash_expert_id), .flash_done(flash_done),
    .hit_count(hit_count), .miss_count(miss_count));

  always #5 clk = ~clk;

  // flash responder: FL cycles after flash_req asserted, pulse flash_done 1 cycle
  integer fcnt; reg fpend;
  always @(posedge clk) begin
    if (rst) begin flash_done<=1'b0; fpend<=1'b0; fcnt<=0; end
    else begin
      flash_done <= 1'b0;
      if (flash_req && !fpend) begin fpend<=1'b1; fcnt<=FL; end
      else if (fpend) begin
        if (fcnt<=1) begin flash_done<=1'b1; fpend<=1'b0; end
        else fcnt<=fcnt-1;
      end
    end
  end

  reg [IDW-1:0] tr_dec [0:N-1];
  reg [IDW-1:0] tr_b8  [0:N-1];
  reg [IDW-1:0] tr_b32 [0:N-1];

  integer i, pass;

  // run all N accesses of `trace` through the (freshly-reset) cache, then
  // compare against the python reference (exp_hit/exp_miss).  +1 test on match.
  task run_pass(input [127:0] name, input integer exp_hit, input integer exp_miss);
    integer m;
    begin
      // synchronous reset to clear the directory + counters
      rst <= 1'b1; req_valid <= 1'b0;
      repeat (4) @(posedge clk);
      rst <= 1'b0; @(posedge clk);
      for (m=0; m<N; m=m+1) begin
        while (busy) @(posedge clk);
        case (name)
          "decode" : req_expert_id <= tr_dec[m];
          "batch8" : req_expert_id <= tr_b8[m];
          "batch32": req_expert_id <= tr_b32[m];
          default  : req_expert_id <= tr_dec[m];
        endcase
        req_valid <= 1'b1;
        @(posedge clk);
        req_valid <= 1'b0;
        while (!resp_valid) @(posedge clk);
      end
      @(posedge clk);
      $display("RTL %0s slots=%0d hits=%0d miss=%0d total=%0d  (%.2f%% hit)",
               name, SLOTS, hit_count, miss_count, hit_count+miss_count,
               100.0*hit_count/(hit_count+miss_count));
      if (hit_count+miss_count !== N)
        begin $display("FAIL: %0s total %0d != %0d", name, hit_count+miss_count, N); $fatal; end
      if (hit_count !== exp_hit || miss_count !== exp_miss)
        begin $display("FAIL: %0s RTL %0d/%0d != python %0d/%0d",
                       name, hit_count, miss_count, exp_hit, exp_miss); $fatal; end
      pass = pass + 1;
    end
  endtask

  initial begin
    pass = 0;
    $readmemh("tools/glm_trace.hex",     tr_dec);
    $readmemh("tools/glm_trace_b8.hex",  tr_b8);
    $readmemh("tools/glm_trace_b32.hex", tr_b32);

    run_pass("decode",  4771, 13229);   // batch=1
    run_pass("batch8",  5341, 12659);   // batch=8
    run_pass("batch32", 9090,  8910);   // batch=32

    $display("----------------------------------------------------------------");
    $display("BATCHING LEVER @ SLOTS=900 (~34GB):  batch1=26.51%%  batch8=29.67%%  batch32=50.50%%");
    $display("ALL %0d TESTS PASSED  (RTL expert_cache_ctrl == python EXACT-LRU, all patterns)", pass);
    $finish;
  end

  initial begin #500000000; $display("FAIL: timeout"); $fatal; end
endmodule
