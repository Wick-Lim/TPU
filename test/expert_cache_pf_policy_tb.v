`timescale 1ns/1ps
//============================================================================
// expert_cache_pf_policy_tb.v  --  LRU vs FREQUENCY-AWARE replacement on the
//                                  GLM-5.2 decode trace (REPL_POLICY 0 vs 1).
//----------------------------------------------------------------------------
// Drives the FULL GLM decode routing trace (tools/glm_trace.hex, 18000 accesses,
// expert id space 19200) through TWO otherwise-identical prefetching caches at
// the GLM cache size (SLOTS=900 ~= 34 GB HBM, N_EXPERT=19200):
//
//     dutL : REPL_POLICY=0  -- EXACT move-to-front LRU (the committed policy)
//     dutF : REPL_POLICY=1  -- frequency-aware (LFU + LRU tie-break + aging)
//
// Both are demand-only here (pf_valid tied low) so this isolates the eviction
// policy as the single variable.  We REPORT hit_count/miss_count + hit-rate for
// each + the delta, and INDEPENDENTLY VERIFY functional correctness of BOTH:
//
//   A TB-side shadow directory per cache is built PURELY from each cache's own
//   reported resp_slot on a miss-install (the only event that changes a slot's
//   contents).  Before every access we PREDICT hit/miss + the resident slot from
//   that shadow, then assert the cache agrees: it must report HIT iff the expert
//   is currently resident, and a HIT must return the slot where the expert
//   actually lives.  This holds for ANY replacement policy -- the policy only
//   chooses WHICH slot is evicted (which the shadow simply follows), never which
//   data is returned.  So a wrong-data / phantom-hit / lost-expert bug in either
//   cache makes the shadow diverge and trips an assertion.
//
// dutL's totals are additionally locked to the proven python EXACT-LRU reference
// (4771/13229 @ slots=900).  dutF's totals are locked to the RTL freq-aware result
// (a regression lock; the headline LRU-vs-freq delta is printed at the end).
//
//   Regenerate the trace first:  python3 tools/route_trace.py --dump
//   iverilog -g2012 -Wall -I src -o /tmp/pol test/expert_cache_pf_policy_tb.v \
//            src/expert_cache_pf.v && vvp /tmp/pol
//============================================================================
module expert_cache_pf_policy_tb;
  localparam integer SLOTS = 900, NE = 19200, FL = 2;
  localparam integer IDW = 15;    // clog2(19200)
  localparam integer SLW = 10;    // clog2(900)
  localparam integer N   = 18000;

  // expected references @ slots=900
  localparam integer EXP_LRU_HITS  = 4771;   // python EXACT-LRU (proven, bit-exact)
  localparam integer EXP_LRU_MISS  = 13229;
  localparam integer EXP_FREQ_HITS = 4889;   // RTL freq-aware (FW=4, AGE=8192) regression lock
  localparam integer EXP_FREQ_MISS = 13111;

  reg clk = 1'b0, rst = 1'b1;
  reg req_valid = 1'b0;  reg [IDW-1:0] req_expert_id = 0;

  // --- dutL : LRU (REPL_POLICY=0) ---
  wire resp_validL, hitL, busyL, pf_readyL, flash_reqL;
  wire [SLW-1:0] resp_slotL;  wire [IDW-1:0] flash_expert_idL;
  reg  flash_doneL = 1'b0;
  wire [31:0] hit_countL, miss_countL, demand_stallL, pf_issuedL, pf_hitL;
  expert_cache_pf #(.SLOTS(SLOTS), .N_EXPERT(NE), .FLASH_LAT(FL), .REPL_POLICY(0)) dutL (
    .clk(clk), .rst(rst),
    .req_valid(req_valid), .req_expert_id(req_expert_id),
    .resp_valid(resp_validL), .hit(hitL), .resp_slot(resp_slotL), .busy(busyL),
    .pf_valid(1'b0), .pf_expert_id({IDW{1'b0}}), .pf_ready(pf_readyL),
    .flash_req(flash_reqL), .flash_expert_id(flash_expert_idL), .flash_done(flash_doneL),
    .hit_count(hit_countL), .miss_count(miss_countL),
    .demand_stall_cycles(demand_stallL), .pf_issued(pf_issuedL), .pf_hit(pf_hitL));

  // --- dutF : frequency-aware (REPL_POLICY=1) ---
  wire resp_validF, hitF, busyF, pf_readyF, flash_reqF;
  wire [SLW-1:0] resp_slotF;  wire [IDW-1:0] flash_expert_idF;
  reg  flash_doneF = 1'b0;
  wire [31:0] hit_countF, miss_countF, demand_stallF, pf_issuedF, pf_hitF;
  expert_cache_pf #(.SLOTS(SLOTS), .N_EXPERT(NE), .FLASH_LAT(FL), .REPL_POLICY(1)) dutF (
    .clk(clk), .rst(rst),
    .req_valid(req_valid), .req_expert_id(req_expert_id),
    .resp_valid(resp_validF), .hit(hitF), .resp_slot(resp_slotF), .busy(busyF),
    .pf_valid(1'b0), .pf_expert_id({IDW{1'b0}}), .pf_ready(pf_readyF),
    .flash_req(flash_reqF), .flash_expert_id(flash_expert_idF), .flash_done(flash_doneF),
    .hit_count(hit_countF), .miss_count(miss_countF),
    .demand_stall_cycles(demand_stallF), .pf_issued(pf_issuedF), .pf_hit(pf_hitF));

  always #5 clk = ~clk;

  // two independent Flash responders (FL-cycle latency each)
  integer fcntL; reg fpendL;
  always @(posedge clk) begin
    if (rst) begin flash_doneL<=1'b0; fpendL<=1'b0; fcntL<=0; end
    else begin
      flash_doneL <= 1'b0;
      if (flash_reqL && !fpendL) begin fpendL<=1'b1; fcntL<=FL; end
      else if (fpendL) begin
        if (fcntL<=1) begin flash_doneL<=1'b1; fpendL<=1'b0; end
        else fcntL<=fcntL-1;
      end
    end
  end
  integer fcntF; reg fpendF;
  always @(posedge clk) begin
    if (rst) begin flash_doneF<=1'b0; fpendF<=1'b0; fcntF<=0; end
    else begin
      flash_doneF <= 1'b0;
      if (flash_reqF && !fpendF) begin fpendF<=1'b1; fcntF<=FL; end
      else if (fpendF) begin
        if (fcntF<=1) begin flash_doneF<=1'b1; fpendF<=1'b0; end
        else fcntF<=fcntF-1;
      end
    end
  end

  // trace + TB-side shadow directories
  reg [IDW-1:0] tr [0:N-1];
  reg [IDW-1:0] slt_tagL [0:SLOTS-1];  reg slt_valL [0:SLOTS-1];
  reg [IDW-1:0] slt_tagF [0:SLOTS-1];  reg slt_valF [0:SLOTS-1];

  integer m, k, errors;
  integer pL_hit, pL_slot, pF_hit, pF_slot;
  integer gotL, gotF, guard;
  integer tb_hL, tb_hF;

  task fail(input [255:0] msg);
    begin $display("FAIL @ m=%0d : %0s", m, msg); errors = errors + 1; $fatal; end
  endtask

  initial begin
    errors = 0;
    $readmemh("tools/glm_trace.hex", tr);

    // synchronous reset of both caches + clear shadows
    rst <= 1'b1; req_valid <= 1'b0;
    for (k=0;k<SLOTS;k=k+1) begin
      slt_valL[k]=1'b0; slt_tagL[k]={IDW{1'b0}};
      slt_valF[k]=1'b0; slt_tagF[k]={IDW{1'b0}};
    end
    repeat (4) @(posedge clk);
    rst <= 1'b0; @(posedge clk);
    tb_hL = 0; tb_hF = 0;

    for (m=0; m<N; m=m+1) begin
      // ---- predict hit/miss + resident slot from each shadow ----
      pL_hit=0; pL_slot=0;
      for (k=0;k<SLOTS;k=k+1)
        if (slt_valL[k] && slt_tagL[k]==tr[m]) begin pL_hit=1; pL_slot=k; end
      pF_hit=0; pF_slot=0;
      for (k=0;k<SLOTS;k=k+1)
        if (slt_valF[k] && slt_tagF[k]==tr[m]) begin pF_hit=1; pF_slot=k; end

      // ---- issue the demand to both (drain any in-flight miss first) ----
      while (busyL || busyF) @(posedge clk);
      req_expert_id <= tr[m]; req_valid <= 1'b1;
      @(posedge clk); req_valid <= 1'b0;

      // ---- collect both responses; verify against the prediction ----
      gotL=0; gotF=0; guard=0;
      while (!(gotL && gotF)) begin
        @(posedge clk);
        if (resp_validL && !gotL) begin
          gotL = 1;
          if (hitL !== pL_hit[0]) fail("dutL hit/miss != shadow");
          if (hitL) begin
            if (resp_slotL !== pL_slot[SLW-1:0]) fail("dutL hit returned wrong slot");
            tb_hL = tb_hL + 1;
          end else begin
            slt_valL[resp_slotL] = 1'b1;       // follow the cache's install
            slt_tagL[resp_slotL] = tr[m];
          end
        end
        if (resp_validF && !gotF) begin
          gotF = 1;
          if (hitF !== pF_hit[0]) fail("dutF hit/miss != shadow");
          if (hitF) begin
            if (resp_slotF !== pF_slot[SLW-1:0]) fail("dutF hit returned wrong slot");
            tb_hF = tb_hF + 1;
          end else begin
            slt_valF[resp_slotF] = 1'b1;
            slt_tagF[resp_slotF] = tr[m];
          end
        end
        guard = guard + 1;
        if (guard > 1000) fail("response timeout");
      end
    end
    @(posedge clk);

    // ---- cross-checks: DUT counters vs TB-independent tally ----
    if (hit_countL+miss_countL !== N) fail("dutL total != N");
    if (hit_countF+miss_countF !== N) fail("dutF total != N");
    if (hit_countL !== tb_hL) fail("dutL hit_count != TB tally");
    if (hit_countF !== tb_hF) fail("dutF hit_count != TB tally");

    $display("================================================================");
    $display("GLM-5.2 decode trace  N=%0d  SLOTS=%0d  N_EXPERT=%0d", N, SLOTS, NE);
    $display("  LRU  (REPL_POLICY=0): hits=%0d miss=%0d  hit_rate=%.2f%%",
             hit_countL, miss_countL, 100.0*hit_countL/N);
    $display("  FREQ (REPL_POLICY=1): hits=%0d miss=%0d  hit_rate=%.2f%%",
             hit_countF, miss_countF, 100.0*hit_countF/N);
    $display("  DELTA  freq - lru   : %0d more hits  = %+.2f pp  (%+.1f%% relative)",
             hit_countF-hit_countL,
             100.0*(hit_countF-hit_countL)/N,
             100.0*(hit_countF-hit_countL)/hit_countL);
    $display("----------------------------------------------------------------");

    // ---- reference locks ----
    if (hit_countL !== EXP_LRU_HITS || miss_countL !== EXP_LRU_MISS)
      fail("dutL != python EXACT-LRU reference");
    $display("  dutL == python EXACT-LRU reference (%0d/%0d)  OK", EXP_LRU_HITS, EXP_LRU_MISS);
    if (hit_countF !== EXP_FREQ_HITS || miss_countF !== EXP_FREQ_MISS)
      fail("dutF != freq-aware regression lock");
    $display("  dutF == freq-aware regression lock     (%0d/%0d)  OK", EXP_FREQ_HITS, EXP_FREQ_MISS);

    $display("----------------------------------------------------------------");
    $display("CORRECTNESS: both caches returned the right slot for every gather");
    $display("             (%0d accesses, independent shadow-directory check).", N);
    $display("ALL POLICY TESTS PASSED  (errors=%0d)", errors);
    $finish;
  end

  initial begin #2000000000; $display("FAIL: timeout"); $fatal; end
endmodule
