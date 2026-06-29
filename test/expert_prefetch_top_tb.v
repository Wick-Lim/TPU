`timescale 1ns/1ps
//============================================================================
// expert_prefetch_top_tb.v
//   PREDICTIVE-PREFETCH MEASUREMENT vs the committed demand-only LRU baseline,
//   on the CALIBRATED GLM-5.2 decode routing trace, at TWO cache regimes.
//----------------------------------------------------------------------------
// TRACE  tools/glm_trace.hex  (regenerate: `python3 tools/route_trace.py --dump`)
//   ids = layer*256+expert ; 75 layers x 256 experts, top-8, token-major decode.
//   Token-major => the reuse distance of any (layer,expert) is ~one token =
//   75*8 = ~600 accesses, so a demand LRU needs >=~600 slots to score ANY hit.
//
// TWO REGIMES (both at full GLM id space, 19200 experts):
//   * SLOTS=900 (~34 GB, the GLM HBM cache size, > reuse distance) -- the
//     popular experts a popularity-predictor would hint are ALREADY resident,
//     so this exposes the honest truth that predictive prefetch is a NO-OP here.
//   * SLOTS=550 (just BELOW the ~600 reuse distance) -- the demand LRU evicts
//     experts ~just before they recur (baseline hit rate ~0%); this is the
//     regime where bringing an expert back EARLY can actually help, so it
//     exercises + quantifies the prefetch engine (issued / hit / over-prefetch /
//     the Flash-channel contention cost on demand-stall).
//
// Against each regime's BASELINE (expert_cache_pf, prefetch port tied OFF == the
// committed EXACT-LRU demand cache; we ASSERT bit-exact hit/miss vs the python
// EXACT-LRU reference) we measure expert_prefetch_top at LOOKAHEAD in {1,2,3}
// and CONF_THRESH in {2,4}: hit/miss (hit rate), demand-stall cycles, prefetches
// issued (pf_issued), prefetches that paid off (pf_hit), and the OVER-PREFETCH
// cost = issued-but-never-demanded.
//
// FUNCTIONAL CORRECTNESS (load-bearing): a WHITEBOX shadow check on EVERY demand
// response of EVERY instance -- the HBM slot the cache returns MUST actually hold
// the demanded GLOBAL expert id (peek that instance's own tag_arr/valid_arr).
// This proves prefetch only changes WHICH experts are resident early; it NEVER
// returns wrong data.  Any mismatch -> $fatal.
//============================================================================
module expert_prefetch_top_tb;
  // ---- GLM-5.2 dimensions ----
  localparam integer NE_PER   = 256;            // experts per layer
  localparam integer N_LAYER  = 75;             // MoE layers
  localparam integer NE_TOTAL = N_LAYER*NE_PER; // 19200 global ids
  localparam integer EID_W    = 8;              // clog2(256)
  localparam integer LAY_W    = 7;              // clog2(75)
  localparam integer ID_W     = 15;             // clog2(19200)
  localparam integer SLOT_W   = 10;             // clog2(900)==clog2(550)==10 (uniform)

  // ---- two cache regimes ----
  localparam integer SLOTS_BIG = 900;           // GLM size, > reuse distance
  localparam integer SLOTS_SML = 550;           // just below reuse distance (stress)

  // ---- experiment knobs ----
  localparam integer TRACE_N  = 4800;           // trace slice (8 decode tokens of ~600)
  localparam integer FL       = 8;              // Flash miss latency (cycles)
  localparam integer GAP      = 3;              // idle cycles after a demand (prefetch room)
  localparam integer NP       = 8;              // configs: 0..3 -> SLOTS_BIG, 4..7 -> SLOTS_SML

  // python EXACT-LRU reference for this slice (route_trace.py), per regime.
  localparam integer PY_HIT_BIG=1147, PY_MISS_BIG=3653;   // slots=900
  localparam integer PY_HIT_SML=0,    PY_MISS_SML=4800;   // slots=550

  reg clk = 1'b0, rst = 1'b1;
  always #5 clk = ~clk;

  reg [ID_W-1:0] trace [0:TRACE_N-1];

  //==========================================================================
  // BASELINES : expert_cache_pf, prefetch OFF == committed EXACT-LRU (per regime)
  //==========================================================================
  // generic baseline helper macro-by-hand: two explicit instances.
  // ---- baseline @ SLOTS_BIG ----
  reg              bb_reqv=1'b0; reg [ID_W-1:0] bb_reqid=0, bb_infl=0; reg bb_fdone=1'b0;
  wire             bb_respv,bb_hit,bb_busy,bb_pfrdy,bb_flreq;
  wire [SLOT_W-1:0]bb_slot; wire [ID_W-1:0] bb_flid;
  wire [31:0]      bb_hitc,bb_missc,bb_stall,bb_pfiss,bb_pfhit;
  integer          bb_wberr=0;
  expert_cache_pf #(.SLOTS(SLOTS_BIG), .N_EXPERT(NE_TOTAL), .FLASH_LAT(FL)) bdutB (
    .clk(clk),.rst(rst),.req_valid(bb_reqv),.req_expert_id(bb_reqid),
    .resp_valid(bb_respv),.hit(bb_hit),.resp_slot(bb_slot),.busy(bb_busy),
    .pf_valid(1'b0),.pf_expert_id({ID_W{1'b0}}),.pf_ready(bb_pfrdy),
    .flash_req(bb_flreq),.flash_expert_id(bb_flid),.flash_done(bb_fdone),
    .hit_count(bb_hitc),.miss_count(bb_missc),.demand_stall_cycles(bb_stall),
    .pf_issued(bb_pfiss),.pf_hit(bb_pfhit));
  integer bb_fcnt; reg bb_fpend;
  always @(posedge clk) begin
    if (rst) begin bb_fdone<=0;bb_fpend<=0;bb_fcnt<=0; end
    else begin bb_fdone<=1'b0;
      if (bb_flreq&&!bb_fpend&&!bb_fdone) begin bb_fpend<=1;bb_fcnt<=FL; end
      else if (bb_fpend) begin if(bb_fcnt<=1)begin bb_fdone<=1;bb_fpend<=0;end else bb_fcnt<=bb_fcnt-1; end
    end
  end
  always @(posedge clk) if (!rst && bb_respv)
    if (!(bdutB.valid_arr[bb_slot]===1'b1 && bdutB.tag_arr[bb_slot]===bb_infl)) begin
      $display("FAIL[wb baseBIG] slot=%0d tag=%0d valid=%b demanded=%0d",
               bb_slot,bdutB.tag_arr[bb_slot],bdutB.valid_arr[bb_slot],bb_infl);
      bb_wberr=bb_wberr+1; end

  // ---- baseline @ SLOTS_SML ----
  reg              bs_reqv=1'b0; reg [ID_W-1:0] bs_reqid=0, bs_infl=0; reg bs_fdone=1'b0;
  wire             bs_respv,bs_hit,bs_busy,bs_pfrdy,bs_flreq;
  wire [SLOT_W-1:0]bs_slot; wire [ID_W-1:0] bs_flid;
  wire [31:0]      bs_hitc,bs_missc,bs_stall,bs_pfiss,bs_pfhit;
  integer          bs_wberr=0;
  expert_cache_pf #(.SLOTS(SLOTS_SML), .N_EXPERT(NE_TOTAL), .FLASH_LAT(FL)) bdutS (
    .clk(clk),.rst(rst),.req_valid(bs_reqv),.req_expert_id(bs_reqid),
    .resp_valid(bs_respv),.hit(bs_hit),.resp_slot(bs_slot),.busy(bs_busy),
    .pf_valid(1'b0),.pf_expert_id({ID_W{1'b0}}),.pf_ready(bs_pfrdy),
    .flash_req(bs_flreq),.flash_expert_id(bs_flid),.flash_done(bs_fdone),
    .hit_count(bs_hitc),.miss_count(bs_missc),.demand_stall_cycles(bs_stall),
    .pf_issued(bs_pfiss),.pf_hit(bs_pfhit));
  integer bs_fcnt; reg bs_fpend;
  always @(posedge clk) begin
    if (rst) begin bs_fdone<=0;bs_fpend<=0;bs_fcnt<=0; end
    else begin bs_fdone<=1'b0;
      if (bs_flreq&&!bs_fpend&&!bs_fdone) begin bs_fpend<=1;bs_fcnt<=FL; end
      else if (bs_fpend) begin if(bs_fcnt<=1)begin bs_fdone<=1;bs_fpend<=0;end else bs_fcnt<=bs_fcnt-1; end
    end
  end
  always @(posedge clk) if (!rst && bs_respv)
    if (!(bdutS.valid_arr[bs_slot]===1'b1 && bdutS.tag_arr[bs_slot]===bs_infl)) begin
      $display("FAIL[wb baseSML] slot=%0d tag=%0d valid=%b demanded=%0d",
               bs_slot,bdutS.tag_arr[bs_slot],bdutS.valid_arr[bs_slot],bs_infl);
      bs_wberr=bs_wberr+1; end

  //==========================================================================
  // PREFETCH configs : expert_prefetch_top (predictor LOOKAHEAD ahead -> pf hint)
  //   idx 0..3 : SLOTS=900   {LA1C2, LA2C2, LA3C2, LA2C4}
  //   idx 4..7 : SLOTS=550   {LA1C2, LA2C2, LA3C2, LA2C4}
  //==========================================================================
  reg  [NP-1:0]    p_reqv = {NP{1'b0}};
  reg  [LAY_W-1:0] p_lay  [0:NP-1];
  reg  [EID_W-1:0] p_exp  [0:NP-1];
  reg  [ID_W-1:0]  p_infl [0:NP-1];
  reg  [NP-1:0]    p_fdone = {NP{1'b0}};
  wire [NP-1:0]    p_respv,p_hit,p_busy,p_pfrdy,p_pfhv,p_flreq;
  wire [SLOT_W-1:0]p_slot [0:NP-1];
  wire [ID_W-1:0]  p_flid [0:NP-1], p_pfhe [0:NP-1];
  wire [31:0]      p_hitc [0:NP-1], p_missc[0:NP-1], p_stall[0:NP-1];
  wire [31:0]      p_pfiss[0:NP-1], p_pfhit[0:NP-1];

  genvar i;
  generate
    for (i = 0; i < NP; i = i + 1) begin : gpf
      localparam integer SL = (i<4) ? SLOTS_BIG : SLOTS_SML;
      localparam integer LA = (i%4==0) ? 1 : (i%4==1) ? 2 : (i%4==2) ? 3 : 2;
      localparam integer CT = (i%4==3) ? 4 : 2;
      integer wb_err; initial wb_err = 0;

      expert_prefetch_top #(
        .N_EXPERT(NE_PER), .N_LAYER(N_LAYER), .SLOTS(SL),
        .LOOKAHEAD(LA), .CONF_THRESH(CT),
        .TOPK(2),                 // only the registered top-1 hint is used
        .HIST_DEPTH(16), .FREQ_W(4), .FLASH_LAT(FL)
      ) dut (
        .clk(clk), .rst(rst),
        .in_valid(p_reqv[i]), .in_layer(p_lay[i]), .in_expert(p_exp[i]),
        .resp_valid(p_respv[i]), .hit(p_hit[i]), .resp_slot(p_slot[i]), .busy(p_busy[i]),
        .flash_req(p_flreq[i]), .flash_expert_id(p_flid[i]), .flash_done(p_fdone[i]),
        .pf_ready(p_pfrdy[i]), .pf_hint_valid(p_pfhv[i]), .pf_hint_expert(p_pfhe[i]),
        .hit_count(p_hitc[i]), .miss_count(p_missc[i]),
        .demand_stall_cycles(p_stall[i]), .pf_issued(p_pfiss[i]), .pf_hit(p_pfhit[i])
      );

      integer fcnt; reg fpend;
      always @(posedge clk) begin
        if (rst) begin p_fdone[i]<=1'b0; fpend<=1'b0; fcnt<=0; end
        else begin p_fdone[i] <= 1'b0;
          if (p_flreq[i] && !fpend && !p_fdone[i]) begin fpend<=1'b1; fcnt<=FL; end
          else if (fpend) begin
            if (fcnt<=1) begin p_fdone[i]<=1'b1; fpend<=1'b0; end
            else fcnt<=fcnt-1;
          end
        end
      end

      // WHITEBOX: the returned demand slot must hold the demanded global id.
      always @(posedge clk) begin
        if (!rst && p_respv[i]) begin
          if (!(dut.u_cache.valid_arr[p_slot[i]]===1'b1 &&
                dut.u_cache.tag_arr[p_slot[i]]===p_infl[i])) begin
            $display("FAIL[wb gpf%0d] slot=%0d tag=%0d valid=%b demanded=%0d",
                     i, p_slot[i], dut.u_cache.tag_arr[p_slot[i]],
                     dut.u_cache.valid_arr[p_slot[i]], p_infl[i]);
            wb_err = wb_err + 1;
          end
        end
      end
    end
  endgenerate

  //==========================================================================
  // measured snapshots
  //==========================================================================
  integer m_hit [0:NP-1], m_miss[0:NP-1], m_stall[0:NP-1];
  integer m_pfiss[0:NP-1], m_pfhit[0:NP-1];
  integer mbB_hit,mbB_miss,mbB_stall, mbS_hit,mbS_miss,mbS_stall;

  //==========================================================================
  // drivers
  //==========================================================================
  task pulse_reset;
    begin
      rst<=1'b1; bb_reqv<=1'b0; bs_reqv<=1'b0; p_reqv<={NP{1'b0}};
      repeat (4) @(negedge clk); rst<=1'b0; @(negedge clk);
    end
  endtask

  task run_pf(input integer c);
    integer m; reg [ID_W-1:0] gid;
    begin
      pulse_reset;
      for (m=0;m<TRACE_N;m=m+1) begin
        while (p_busy[c]) @(negedge clk);
        gid = trace[m];
        @(negedge clk);
        p_infl[c] <= gid;
        p_lay[c]  <= gid[ID_W-1:EID_W];
        p_exp[c]  <= gid[EID_W-1:0];
        p_reqv[c] <= 1'b1;
        @(negedge clk);
        p_reqv[c] <= 1'b0;
        while (p_respv[c]!==1'b1) @(negedge clk);
        repeat (GAP) @(negedge clk);
      end
      repeat (40) @(negedge clk);
      m_hit[c]=p_hitc[c]; m_miss[c]=p_missc[c]; m_stall[c]=p_stall[c];
      m_pfiss[c]=p_pfiss[c]; m_pfhit[c]=p_pfhit[c];
    end
  endtask

  // baseline driver (which=0 -> BIG, 1 -> SML)
  task run_base(input integer which);
    integer m; reg [ID_W-1:0] gid;
    begin
      pulse_reset;
      for (m=0;m<TRACE_N;m=m+1) begin
        if (which==0) while (bb_busy) @(negedge clk);
        else          while (bs_busy) @(negedge clk);
        gid = trace[m];
        @(negedge clk);
        if (which==0) begin bb_infl<=gid; bb_reqid<=gid; bb_reqv<=1'b1; end
        else          begin bs_infl<=gid; bs_reqid<=gid; bs_reqv<=1'b1; end
        @(negedge clk);
        if (which==0) bb_reqv<=1'b0; else bs_reqv<=1'b0;
        if (which==0) while (bb_respv!==1'b1) @(negedge clk);
        else          while (bs_respv!==1'b1) @(negedge clk);
      end
      repeat (8) @(negedge clk);
      if (which==0) begin mbB_hit=bb_hitc; mbB_miss=bb_missc; mbB_stall=bb_stall; end
      else          begin mbS_hit=bs_hitc; mbS_miss=bs_missc; mbS_stall=bs_stall; end
    end
  endtask

  //==========================================================================
  // run + report
  //==========================================================================
  integer errors=0, tests=0, c, total_wb;
  real    brB, brS, pr, dimp, simp;

  task report_section(input integer base_lo, input [127:0] tag, input integer slots,
                      input integer bhit, input integer bmiss, input integer bstall);
    integer c2; real br2;
    begin
      br2 = 100.0*bhit/(bhit+bmiss);
      $display("----- regime %0s : SLOTS=%0d (%s reuse distance) -----",
               tag, slots, (slots>=600)?">":"<");
      $display(" config            | hits  miss  hit%%   d-stall  pf_iss pf_hit overPF  dHit%%   dStall%%");
      $display(" demand-only base  | %5d %5d %6.2f %8d      -      -      -       -        -",
               bhit, bmiss, br2, bstall);
      for (c2=base_lo; c2<base_lo+4; c2=c2+1) begin
        pr   = 100.0*m_hit[c2]/(m_hit[c2]+m_miss[c2]);
        dimp = pr - br2;
        simp = (bstall==0)?0.0:100.0*(bstall - m_stall[c2])/bstall;
        $display(" LA=%0d CONF>=%0d        | %5d %5d %6.2f %8d %6d %6d %6d %+7.2f %+8.2f",
                 (c2%4==0)?1:(c2%4==1)?2:(c2%4==2)?3:2, (c2%4==3)?4:2,
                 m_hit[c2], m_miss[c2], pr, m_stall[c2],
                 m_pfiss[c2], m_pfhit[c2], m_pfiss[c2]-m_pfhit[c2], dimp, simp);
      end
    end
  endtask

  initial begin
    for (c=0;c<NP;c=c+1) begin p_lay[c]=0; p_exp[c]=0; p_infl[c]=0; end
    $readmemh("tools/glm_trace.hex", trace);

    run_base(0);                       // SLOTS=900 baseline
    run_base(1);                       // SLOTS=550 baseline
    for (c=0;c<NP;c=c+1) run_pf(c);    // 8 prefetch configs

    // ===================== correctness gates =====================
    if (mbB_hit===PY_HIT_BIG && mbB_miss===PY_MISS_BIG) tests=tests+1;
    else begin $display("FAIL: base900 %0d/%0d != python %0d/%0d",
                        mbB_hit,mbB_miss,PY_HIT_BIG,PY_MISS_BIG); errors=errors+1; end
    if (mbS_hit===PY_HIT_SML && mbS_miss===PY_MISS_SML) tests=tests+1;
    else begin $display("FAIL: base550 %0d/%0d != python %0d/%0d",
                        mbS_hit,mbS_miss,PY_HIT_SML,PY_MISS_SML); errors=errors+1; end

    if ((mbB_hit+mbB_miss)===TRACE_N) tests=tests+1;
    else begin $display("FAIL: base900 total %0d", mbB_hit+mbB_miss); errors=errors+1; end
    if ((mbS_hit+mbS_miss)===TRACE_N) tests=tests+1;
    else begin $display("FAIL: base550 total %0d", mbS_hit+mbS_miss); errors=errors+1; end
    for (c=0;c<NP;c=c+1) begin
      if ((m_hit[c]+m_miss[c])===TRACE_N) tests=tests+1;
      else begin $display("FAIL: cfg%0d total %0d != %0d",c,m_hit[c]+m_miss[c],TRACE_N);
                 errors=errors+1; end
    end

    total_wb = bb_wberr + bs_wberr + gpf[0].wb_err + gpf[1].wb_err + gpf[2].wb_err
               + gpf[3].wb_err + gpf[4].wb_err + gpf[5].wb_err + gpf[6].wb_err + gpf[7].wb_err;
    if (total_wb===0) tests=tests+1;
    else begin $display("FAIL: %0d whitebox shadow mismatch(es)", total_wb); errors=errors+1; end

    if ((^mbB_hit!==1'bx)&&(^mbS_stall!==1'bx)&&(^m_hit[4]!==1'bx)&&
        (^m_pfiss[4]!==1'bx)&&(^m_stall[0]!==1'bx)) tests=tests+1;
    else begin $display("FAIL: X in a measured counter"); errors=errors+1; end

    // ===================== measurement report =====================
    brB = 100.0*mbB_hit/(mbB_hit+mbB_miss);
    brS = 100.0*mbS_hit/(mbS_hit+mbS_miss);
    $display("");
    $display("=========== PREDICTIVE-PREFETCH MEASUREMENT (GLM-5.2 decode trace) ===========");
    $display(" slice=%0d accesses (token-major decode, reuse distance ~600)  FlashLat=%0d  gap=%0d",
             TRACE_N, FL, GAP);
    $display(" demand pool: 75 layers x 256 experts top-8 (fine-grained MoE, weak locality)");
    $display("-----------------------------------------------------------------------------");
    report_section(0, "BIG", SLOTS_BIG, mbB_hit, mbB_miss, mbB_stall);
    $display("-----------------------------------------------------------------------------");
    report_section(4, "SML", SLOTS_SML, mbS_hit, mbS_miss, mbS_stall);
    $display("-----------------------------------------------------------------------------");
    $display(" dHit%%=hit-rate gain over baseline (pp); dStall%%=demand-stall reduction (+=fewer waits)");
    $display(" overPF=prefetches issued but never demanded-while-resident (over-prefetch cost)");
    $display("=============================================================================");

    if (errors==0) $display("ALL %0d TESTS PASSED", tests);
    else begin $display("%0d CHECK(S) FAILED", errors); $fatal(1,"measurement failed"); end
    $finish;
  end

  initial begin #4000000000; $display("FAIL: timeout"); $fatal(1,"timeout"); end
endmodule
