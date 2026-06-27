`timescale 1ns/1ps
// Confirm: run the calibrated GLM-scale routing trace through the REAL expert_cache_ctrl
// RTL and check hit_count/miss_count match the python LRU reference (slots=900 -> 4771/13229
// = 26.51%). Regenerate first:  python3 tools/route_trace.py --dump
//   iverilog -g2012 -Wall -I src -o /tmp/glmcache tools/glm_cache_confirm_tb.v src/expert_cache_ctrl.v && vvp /tmp/glmcache
module glm_cache_confirm_tb;
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

  // flash responder: FL cycles after flash_req asserted, pulse flash_done for 1 cycle
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

  reg [IDW-1:0] trace [0:N-1];
  integer i;
  initial begin
    $readmemh("tools/glm_trace.hex", trace);
    repeat (4) @(posedge clk);
    rst <= 1'b0;
    @(posedge clk);
    for (i=0; i<N; i=i+1) begin
      while (busy) @(posedge clk);
      req_expert_id <= trace[i]; req_valid <= 1'b1;
      @(posedge clk);
      req_valid <= 1'b0;
      while (!resp_valid) @(posedge clk);
    end
    @(posedge clk);
    $display("RTL slots=%0d hits=%0d miss=%0d total=%0d", SLOTS, hit_count, miss_count, hit_count+miss_count);
    if (hit_count+miss_count !== N)
      $display("FAIL: total %0d != %0d", hit_count+miss_count, N);
    else if (hit_count === 32'd4771 && miss_count === 32'd13229)
      $display("ALL 1 TESTS PASSED  (RTL == python LRU: 4771 hits / 13229 miss / 26.51%%)");
    else
      $display("MISMATCH vs python (expected 4771/13229), got %0d/%0d", hit_count, miss_count);
    $finish;
  end
endmodule
