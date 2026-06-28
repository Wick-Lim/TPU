`timescale 1ns/1ps
//============================================================================
// expert_cache_pf_tb.v  --  smoke + invariant TB for src/expert_cache_pf.v
//----------------------------------------------------------------------------
//  T1  pf OFF : a random demand sequence reproduces an INDEPENDENT behavioral
//               move-to-front LRU bit-for-bit (every hit/miss + final counts)
//               -> proves pf-off == expert_cache_ctrl policy.
//  T2  prefetch an id, then demand it -> HIT with ZERO demand-stall (latency
//               hidden), pf_issued==1, pf_hit==1.
//  T3  cold demand of a NEVER-prefetched id -> MISS with NON-ZERO demand-stall
//               (the un-hidden case, for contrast).
//  T4  prefetch an already-resident id -> SKIP (pf_issued unchanged), still hits.
//----------------------------------------------------------------------------
//  build: iverilog -g2012 -Wall -I src -o /tmp/pf test/expert_cache_pf_tb.v \
//                   src/expert_cache_pf.v && vvp /tmp/pf
//============================================================================
module expert_cache_pf_tb;
  localparam integer SLOTS = 8, NE = 64, FL = 6;
  localparam integer IDW = 6;    // clog2(64)
  localparam integer SLW = 3;    // clog2(8)

  reg clk=1'b0, rst=1'b1;
  reg req_valid=1'b0; reg [IDW-1:0] req_expert_id=0;
  reg pf_valid=1'b0;  reg [IDW-1:0] pf_expert_id=0;
  reg flash_done=1'b0;

  wire resp_valid, hit, busy, pf_ready, flash_req;
  wire [SLW-1:0] resp_slot;
  wire [IDW-1:0] flash_expert_id;
  wire [31:0] hit_count, miss_count, demand_stall_cycles, pf_issued, pf_hit;

  expert_cache_pf #(.SLOTS(SLOTS), .N_EXPERT(NE), .FLASH_LAT(FL)) dut (
    .clk(clk), .rst(rst),
    .req_valid(req_valid), .req_expert_id(req_expert_id),
    .resp_valid(resp_valid), .hit(hit), .resp_slot(resp_slot), .busy(busy),
    .pf_valid(pf_valid), .pf_expert_id(pf_expert_id), .pf_ready(pf_ready),
    .flash_req(flash_req), .flash_expert_id(flash_expert_id), .flash_done(flash_done),
    .hit_count(hit_count), .miss_count(miss_count),
    .demand_stall_cycles(demand_stall_cycles), .pf_issued(pf_issued), .pf_hit(pf_hit));

  always #5 clk = ~clk;

  // Flash responder: FL cycles after flash_req asserted, pulse flash_done 1 cycle.
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

  //==========================================================================
  // PART B harness: GLM-scale config so the REAL decode trace (tools/glm_trace.hex,
  // ids up to 19199) fits.  dutB = expert_cache_pf (prefetching cache under test);
  // refB = the COMMITTED expert_cache_ctrl (the demand-only reference).  A scheduler
  // runs a COMPUTE WINDOW of TC cycles after every demand response and PREFETCHES the
  // NEXT demand's expert id during that window.  pf-OFF run must equal refB exactly;
  // pf-ON run hides the miss latency (lower demand_stall_cycles) without changing the
  // expert each demand resolves to.
  //==========================================================================
  localparam integer SLOTS2=900, NE2=19200, FL2=20;
  localparam integer IDW2=15, SLW2=10;
  localparam integer NREQ=200;            // first 200 decode accesses (all distinct/cold)

  reg                 rstB=1'b1;
  reg                 req_validB=1'b0;  reg [IDW2-1:0] req_expert_idB=0;
  reg                 pf_validB=1'b0;   reg [IDW2-1:0] pf_expert_idB=0;
  // dutB (prefetching cache)
  wire                resp_validB, hitB, busyB, pf_readyB, flash_reqB;
  wire [SLW2-1:0]     resp_slotB;
  wire [IDW2-1:0]     flash_expert_idB;
  reg                 flash_doneB=1'b0;
  wire [31:0]         hit_countB, miss_countB, demand_stallB, pf_issuedB, pf_hitB;
  // refB (committed expert_cache_ctrl, demand-only)
  reg                 req_validR=1'b0;  reg [IDW2-1:0] req_expert_idR=0;
  wire                resp_validR, hitR, busyR, flash_reqR;
  wire [SLW2-1:0]     resp_slotR;
  wire [IDW2-1:0]     flash_expert_idR;
  reg                 flash_doneR=1'b0;
  wire [31:0]         hit_countR, miss_countR;

  expert_cache_pf #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2)) dutB (
    .clk(clk), .rst(rstB),
    .req_valid(req_validB), .req_expert_id(req_expert_idB),
    .resp_valid(resp_validB), .hit(hitB), .resp_slot(resp_slotB), .busy(busyB),
    .pf_valid(pf_validB), .pf_expert_id(pf_expert_idB), .pf_ready(pf_readyB),
    .flash_req(flash_reqB), .flash_expert_id(flash_expert_idB), .flash_done(flash_doneB),
    .hit_count(hit_countB), .miss_count(miss_countB),
    .demand_stall_cycles(demand_stallB), .pf_issued(pf_issuedB), .pf_hit(pf_hitB));

  expert_cache_ctrl #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2)) refB (
    .clk(clk), .rst(rstB),
    .req_valid(req_validR), .req_expert_id(req_expert_idR),
    .resp_valid(resp_validR), .hit(hitR), .resp_slot(resp_slotR), .busy(busyR),
    .flash_req(flash_reqR), .flash_expert_id(flash_expert_idR), .flash_done(flash_doneR),
    .hit_count(hit_countR), .miss_count(miss_countR));

  // flash responders for dutB and refB (FL2 latency each, independent channels)
  integer fcntB; reg fpendB;
  always @(posedge clk) begin
    if (rstB) begin flash_doneB<=1'b0; fpendB<=1'b0; fcntB<=0; end
    else begin
      flash_doneB <= 1'b0;
      if (flash_reqB && !fpendB) begin fpendB<=1'b1; fcntB<=FL2; end
      else if (fpendB) begin
        if (fcntB<=1) begin flash_doneB<=1'b1; fpendB<=1'b0; end
        else fcntB<=fcntB-1;
      end
    end
  end
  integer fcntR; reg fpendR;
  always @(posedge clk) begin
    if (rstB) begin flash_doneR<=1'b0; fpendR<=1'b0; fcntR<=0; end
    else begin
      flash_doneR <= 1'b0;
      if (flash_reqR && !fpendR) begin fpendR<=1'b1; fcntR<=FL2; end
      else if (fpendR) begin
        if (fcntR<=1) begin flash_doneR<=1'b1; fpendR<=1'b0; end
        else fcntR<=fcntR-1;
      end
    end
  end

  //==========================================================================
  // PART C harness: GDDR6 hit-latency study.  FOUR identical GLM-scale
  // prefetching caches that differ ONLY in CACHE_HIT_LAT (0/2/4/8 GDDR6 read
  // cycles for a RESIDENT expert), driven in LOCKSTEP off ONE shared demand/
  // prefetch bus so OFF and ON are measured with IDENTICAL inter-demand spacing
  // (the per-miss Flash cost M is then the SAME constant across every run, so
  // the stall-cut % is an apples-to-apples comparison and not a spacing artifact).
  //   dutH0=LAT0 (idealized), dutH1=LAT2, dutH2=LAT4 (GDDR6-realistic), dutH3=LAT8.
  // Each keeps its OWN directory/counters and OWN Flash channel; with the trace
  // cold+distinct and SLOTS2 >> NREQ there are no evictions so all four evolve
  // their directories identically and differ only in hit-delivery TIMING.
  //==========================================================================
  reg               rstH=1'b1;
  reg               req_validH=1'b0;  reg [IDW2-1:0] req_expert_idH=0;  // shared demand bus
  reg               pf_validH=1'b0;   reg [IDW2-1:0] pf_expert_idH=0;   // shared prefetch bus

  // ---- dutH0 : CACHE_HIT_LAT = 0 (idealized: a resident hit delivers free) ----
  wire resp_validH0, hitH0, busyH0, pf_readyH0, flash_reqH0;
  wire [SLW2-1:0] resp_slotH0;  wire [IDW2-1:0] flash_expert_idH0;
  reg  flash_doneH0=1'b0;
  wire [31:0] hit_countH0, miss_countH0, demand_stallH0, pf_issuedH0, pf_hitH0;
  expert_cache_pf #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2), .CACHE_HIT_LAT(0)) dutH0 (
    .clk(clk), .rst(rstH),
    .req_valid(req_validH), .req_expert_id(req_expert_idH),
    .resp_valid(resp_validH0), .hit(hitH0), .resp_slot(resp_slotH0), .busy(busyH0),
    .pf_valid(pf_validH), .pf_expert_id(pf_expert_idH), .pf_ready(pf_readyH0),
    .flash_req(flash_reqH0), .flash_expert_id(flash_expert_idH0), .flash_done(flash_doneH0),
    .hit_count(hit_countH0), .miss_count(miss_countH0),
    .demand_stall_cycles(demand_stallH0), .pf_issued(pf_issuedH0), .pf_hit(pf_hitH0));

  // ---- dutH1 : CACHE_HIT_LAT = 2 ----
  wire resp_validH1, hitH1, busyH1, pf_readyH1, flash_reqH1;
  wire [SLW2-1:0] resp_slotH1;  wire [IDW2-1:0] flash_expert_idH1;
  reg  flash_doneH1=1'b0;
  wire [31:0] hit_countH1, miss_countH1, demand_stallH1, pf_issuedH1, pf_hitH1;
  expert_cache_pf #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2), .CACHE_HIT_LAT(2)) dutH1 (
    .clk(clk), .rst(rstH),
    .req_valid(req_validH), .req_expert_id(req_expert_idH),
    .resp_valid(resp_validH1), .hit(hitH1), .resp_slot(resp_slotH1), .busy(busyH1),
    .pf_valid(pf_validH), .pf_expert_id(pf_expert_idH), .pf_ready(pf_readyH1),
    .flash_req(flash_reqH1), .flash_expert_id(flash_expert_idH1), .flash_done(flash_doneH1),
    .hit_count(hit_countH1), .miss_count(miss_countH1),
    .demand_stall_cycles(demand_stallH1), .pf_issued(pf_issuedH1), .pf_hit(pf_hitH1));

  // ---- dutH2 : CACHE_HIT_LAT = 4 (GDDR6-realistic) ----
  wire resp_validH2, hitH2, busyH2, pf_readyH2, flash_reqH2;
  wire [SLW2-1:0] resp_slotH2;  wire [IDW2-1:0] flash_expert_idH2;
  reg  flash_doneH2=1'b0;
  wire [31:0] hit_countH2, miss_countH2, demand_stallH2, pf_issuedH2, pf_hitH2;
  expert_cache_pf #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2), .CACHE_HIT_LAT(4)) dutH2 (
    .clk(clk), .rst(rstH),
    .req_valid(req_validH), .req_expert_id(req_expert_idH),
    .resp_valid(resp_validH2), .hit(hitH2), .resp_slot(resp_slotH2), .busy(busyH2),
    .pf_valid(pf_validH), .pf_expert_id(pf_expert_idH), .pf_ready(pf_readyH2),
    .flash_req(flash_reqH2), .flash_expert_id(flash_expert_idH2), .flash_done(flash_doneH2),
    .hit_count(hit_countH2), .miss_count(miss_countH2),
    .demand_stall_cycles(demand_stallH2), .pf_issued(pf_issuedH2), .pf_hit(pf_hitH2));

  // ---- dutH3 : CACHE_HIT_LAT = 8 ----
  wire resp_validH3, hitH3, busyH3, pf_readyH3, flash_reqH3;
  wire [SLW2-1:0] resp_slotH3;  wire [IDW2-1:0] flash_expert_idH3;
  reg  flash_doneH3=1'b0;
  wire [31:0] hit_countH3, miss_countH3, demand_stallH3, pf_issuedH3, pf_hitH3;
  expert_cache_pf #(.SLOTS(SLOTS2), .N_EXPERT(NE2), .FLASH_LAT(FL2), .CACHE_HIT_LAT(8)) dutH3 (
    .clk(clk), .rst(rstH),
    .req_valid(req_validH), .req_expert_id(req_expert_idH),
    .resp_valid(resp_validH3), .hit(hitH3), .resp_slot(resp_slotH3), .busy(busyH3),
    .pf_valid(pf_validH), .pf_expert_id(pf_expert_idH), .pf_ready(pf_readyH3),
    .flash_req(flash_reqH3), .flash_expert_id(flash_expert_idH3), .flash_done(flash_doneH3),
    .hit_count(hit_countH3), .miss_count(miss_countH3),
    .demand_stall_cycles(demand_stallH3), .pf_issued(pf_issuedH3), .pf_hit(pf_hitH3));

  // four independent Flash responders (FL2 latency each), identical model
  integer fcntH0; reg fpendH0;
  always @(posedge clk) begin
    if (rstH) begin flash_doneH0<=1'b0; fpendH0<=1'b0; fcntH0<=0; end
    else begin
      flash_doneH0 <= 1'b0;
      if (flash_reqH0 && !fpendH0) begin fpendH0<=1'b1; fcntH0<=FL2; end
      else if (fpendH0) begin
        if (fcntH0<=1) begin flash_doneH0<=1'b1; fpendH0<=1'b0; end
        else fcntH0<=fcntH0-1;
      end
    end
  end
  integer fcntH1; reg fpendH1;
  always @(posedge clk) begin
    if (rstH) begin flash_doneH1<=1'b0; fpendH1<=1'b0; fcntH1<=0; end
    else begin
      flash_doneH1 <= 1'b0;
      if (flash_reqH1 && !fpendH1) begin fpendH1<=1'b1; fcntH1<=FL2; end
      else if (fpendH1) begin
        if (fcntH1<=1) begin flash_doneH1<=1'b1; fpendH1<=1'b0; end
        else fcntH1<=fcntH1-1;
      end
    end
  end
  integer fcntH2; reg fpendH2;
  always @(posedge clk) begin
    if (rstH) begin flash_doneH2<=1'b0; fpendH2<=1'b0; fcntH2<=0; end
    else begin
      flash_doneH2 <= 1'b0;
      if (flash_reqH2 && !fpendH2) begin fpendH2<=1'b1; fcntH2<=FL2; end
      else if (fpendH2) begin
        if (fcntH2<=1) begin flash_doneH2<=1'b1; fpendH2<=1'b0; end
        else fcntH2<=fcntH2-1;
      end
    end
  end
  integer fcntH3; reg fpendH3;
  always @(posedge clk) begin
    if (rstH) begin flash_doneH3<=1'b0; fpendH3<=1'b0; fcntH3<=0; end
    else begin
      flash_doneH3 <= 1'b0;
      if (flash_reqH3 && !fpendH3) begin fpendH3<=1'b1; fcntH3<=FL2; end
      else if (fpendH3) begin
        if (fcntH3<=1) begin flash_doneH3<=1'b1; fpendH3<=1'b0; end
        else fcntH3<=fcntH3-1;
      end
    end
  end

  reg [IDW2-1:0] decode_tr [0:NREQ-1];    // first NREQ decode accesses

  //--------------------------------------------------------------------------
  // INDEPENDENT behavioral move-to-front LRU golden (the reference policy)
  //--------------------------------------------------------------------------
  reg              gvalid [0:SLOTS-1];
  reg [IDW-1:0]    gtag   [0:SLOTS-1];
  reg [SLW-1:0]    grank  [0:SLOTS-1];
  integer gj;

  task golden_reset; begin
    for (gj=0; gj<SLOTS; gj=gj+1) begin
      gvalid[gj]=1'b0; gtag[gj]={IDW{1'b0}}; grank[gj]=gj[SLW-1:0];
    end
  end endtask

  task golden_demand(input [IDW-1:0] id, output pred_hit);
    integer j; reg found; reg [SLW-1:0] hs, vic, vlru; reg hv;
    begin
      found=1'b0; hs={SLW{1'b0}};
      for (j=0;j<SLOTS;j=j+1) if (gvalid[j] && gtag[j]==id) begin found=1'b1; hs=j[SLW-1:0]; end
      if (found) begin
        pred_hit=1'b1;
        for (j=0;j<SLOTS;j=j+1) if (grank[j]<grank[hs]) grank[j]=grank[j]+1'b1;
        grank[hs]={SLW{1'b0}};
      end else begin
        pred_hit=1'b0;
        hv=1'b0; vic={SLW{1'b0}};
        for (j=SLOTS-1;j>=0;j=j-1) if (!gvalid[j]) begin hv=1'b1; vic=j[SLW-1:0]; end
        if (!hv) begin
          vlru={SLW{1'b0}};
          for (j=0;j<SLOTS;j=j+1) if (grank[j]==(SLOTS-1)) vlru=j[SLW-1:0];
          vic=vlru;
        end
        gvalid[vic]=1'b1; gtag[vic]=id;
        for (j=0;j<SLOTS;j=j+1) if (grank[j]<grank[vic]) grank[j]=grank[j]+1'b1;
        grank[vic]={SLW{1'b0}};
      end
    end
  endtask

  //--------------------------------------------------------------------------
  // driver helpers
  //--------------------------------------------------------------------------
  integer pass=0;
  task fail(input [255:0] msg); begin
    $display("FAIL: %0s", msg); $fatal;
  end endtask

  task do_reset; begin
    rst<=1'b1; req_valid<=1'b0; pf_valid<=1'b0;
    repeat (4) @(posedge clk);
    rst<=1'b0; @(posedge clk);
    golden_reset;
  end endtask

  // issue one DEMAND request; report hit, slot, busy-seen, and stall delta
  task do_demand(input [IDW-1:0] id, output got_hit, output [SLW-1:0] got_slot,
                 output saw_busy, output integer stall_delta);
    integer s0;
    begin
      saw_busy=1'b0;
      while (busy) @(posedge clk);
      s0 = demand_stall_cycles;
      req_expert_id<=id; req_valid<=1'b1;
      @(posedge clk);
      req_valid<=1'b0;
      while (!resp_valid) begin if (busy) saw_busy=1'b1; @(posedge clk); end
      got_hit=hit; got_slot=resp_slot;
      stall_delta = demand_stall_cycles - s0;
    end
  endtask

  // issue one PREFETCH hint from idle (waits for pf_ready accept)
  task do_prefetch(input [IDW-1:0] id);
    begin
      while (busy || !pf_ready) @(posedge clk);
      pf_expert_id<=id; pf_valid<=1'b1;
      @(posedge clk);          // accepted at this edge (idle, no demand)
      pf_valid<=1'b0;
      repeat (FL+4) @(posedge clk);   // let the background fetch install
    end
  endtask

  //--------------------------------------------------------------------------
  // PART B helpers (GLM-scale dutB / refB)
  //--------------------------------------------------------------------------
  task resetB; begin
    rstB<=1'b1; req_validB<=1'b0; pf_validB<=1'b0; req_validR<=1'b0;
    repeat (4) @(posedge clk);
    rstB<=1'b0; @(posedge clk);
  end endtask

  // one demand on dutB (the prefetching cache)
  task demandB(input [IDW2-1:0] id, output got_h); begin
    while (busyB) @(posedge clk);            // wait any DEMAND fetch to drain (prefetch keeps busy low)
    req_expert_idB<=id; req_validB<=1'b1;
    @(posedge clk); req_validB<=1'b0;
    while (!resp_validB) @(posedge clk);
    got_h = hitB;
  end endtask

  // one demand on refB (the committed expert_cache_ctrl)
  task demandR(input [IDW2-1:0] id, output got_h); begin
    while (busyR) @(posedge clk);
    req_expert_idR<=id; req_validR<=1'b1;
    @(posedge clk); req_validR<=1'b0;
    while (!resp_validR) @(posedge clk);
    got_h = hitR;
  end endtask

  //--------------------------------------------------------------------------
  // PART C lockstep driver: drive the FOUR hit-latency caches (dutH0..H3) off
  // the SHARED demand/prefetch bus through the whole decode window under the
  // compute-window model.  pf_on selects prefetch ON vs OFF; TC is the compute
  // window after each demand (the cycles the prefetch has to land).  All four
  // see the same stimulus and the same spacing -- so the only thing that varies
  // run-to-run is CACHE_HIT_LAT, isolating its stall cost.  Results are left in
  // each cache's demand_stall_cycles / hit_count / miss_count.
  //--------------------------------------------------------------------------
  task pc_run(input pf_on, input integer TC);
    integer i;
    begin
      rstH<=1'b1; req_validH<=1'b0; pf_validH<=1'b0;
      repeat (4) @(posedge clk);
      rstH<=1'b0; @(posedge clk);
      for (i=0;i<NREQ;i=i+1) begin
        // all four idle before issuing the demand (any miss/hit-read drained)
        while (busyH0 || busyH1 || busyH2 || busyH3) @(posedge clk);
        req_expert_idH<=decode_tr[i]; req_validH<=1'b1;
        @(posedge clk); req_validH<=1'b0;     // demand accepted this edge
        @(posedge clk);                        // busy asserts for miss / GDDR6 hit-read
        while (busyH0 || busyH1 || busyH2 || busyH3) @(posedge clk);  // wait the slowest
        // compute window: optionally PREFETCH the next demand's expert, then
        // compute TC cycles (long enough for the Flash prefetch to land).
        if (pf_on && i<NREQ-1) begin
          pf_expert_idH<=decode_tr[i+1]; pf_validH<=1'b1;
        end
        @(posedge clk); pf_validH<=1'b0;       // hint accepted from idle
        repeat (TC) @(posedge clk);
      end
    end
  endtask

  //--------------------------------------------------------------------------
  integer i;
  reg got_hit, saw_busy;
  reg [SLW-1:0] got_slot;
  integer stall_delta;
  reg pred_hit;
  reg [IDW-1:0] xid, yid;
  integer pf_issued_snap;

  // Part B state
  reg          ref_seq [0:NREQ-1];
  reg          off_seq [0:NREQ-1];
  reg          on_seq  [0:NREQ-1];
  integer      ref_h, ref_m, off_h, on_h;
  integer      stall_off, stall_on, red, TC, tci;
  integer      stall_at [0:2];
  reg          gh;

  // Part C state (GDDR6 hit-latency study)
  integer      c_off, c_on0, c_on2, c_on4, c_on8;   // demand_stall: OFF + ON{LAT0,2,4,8}
  integer      cut0, cut4, TCa, TCb;
  integer      hl [0:3];                              // CACHE_HIT_LAT of dutH0..H3

  initial begin
    // ============================ T1: pf OFF == LRU ====================
    do_reset;
    pf_valid<=1'b0;
    for (i=0;i<300;i=i+1) begin
      xid = ($random % NE); if (xid[IDW-1]) xid = xid; // keep in range via mask below
      xid = xid & {IDW{1'b1}};
      golden_demand(xid, pred_hit);
      do_demand(xid, got_hit, got_slot, saw_busy, stall_delta);
      if (got_hit !== pred_hit)
        fail("T1 hit/miss disagrees with behavioral LRU");
      pass = pass + 1;
      // sanity: a hit must never stall; a miss must stall
      if (got_hit && (saw_busy || stall_delta!=0)) fail("T1 hit stalled");
      if (!got_hit && stall_delta==0) fail("T1 miss did not stall");
      pass = pass + 1;
    end
    if (pf_issued !== 32'd0 || pf_hit !== 32'd0)
      fail("T1 prefetch stats nonzero with pf off");
    pass = pass + 1;
    if (hit_count+miss_count !== 32'd300)
      fail("T1 total count mismatch");
    pass = pass + 1;
    $display("T1 ok: %0d demands, hits=%0d miss=%0d (== independent LRU), stall=%0d",
             hit_count+miss_count, hit_count, miss_count, demand_stall_cycles);

    // ============================ T2: prefetch hides latency ===========
    do_reset;
    xid = 6'd21;
    do_prefetch(xid);
    if (pf_issued !== 32'd1) fail("T2 pf_issued != 1");
    pass = pass + 1;
    do_demand(xid, got_hit, got_slot, saw_busy, stall_delta);
    if (!got_hit)                       fail("T2 prefetched id did not HIT");
    pass = pass + 1;
    if (saw_busy || stall_delta != 0)   fail("T2 prefetched demand stalled (latency not hidden)");
    pass = pass + 1;
    if (pf_hit !== 32'd1)               fail("T2 pf_hit != 1");
    pass = pass + 1;
    $display("T2 ok: prefetched id %0d -> HIT, demand-stall delta=0, pf_hit=%0d", xid, pf_hit);

    // ============================ T3: cold (un-hidden) miss ============
    yid = 6'd42;   // never prefetched, not resident
    do_demand(yid, got_hit, got_slot, saw_busy, stall_delta);
    if (got_hit)             fail("T3 cold id unexpectedly HIT");
    pass = pass + 1;
    if (stall_delta == 0)    fail("T3 cold miss did not stall");
    pass = pass + 1;
    $display("T3 ok: cold id %0d -> MISS, demand-stall delta=%0d (>0, un-hidden)", yid, stall_delta);

    // ============================ T4: prefetch resident -> SKIP =======
    pf_issued_snap = pf_issued;
    do_prefetch(xid);        // xid (21) is still resident from T2
    if (pf_issued !== pf_issued_snap) fail("T4 resident prefetch issued a Flash fetch");
    pass = pass + 1;
    do_demand(xid, got_hit, got_slot, saw_busy, stall_delta);
    if (!got_hit || stall_delta != 0) fail("T4 resident id did not HIT/0-stall");
    pass = pass + 1;
    $display("T4 ok: prefetch of resident id %0d skipped (pf_issued stays %0d)", xid, pf_issued);

    // =================== T5: PART B -- prefetch latency-hiding ==============
    // COMPUTE WINDOW model on the REAL decode trace through dutB (expert_cache_pf)
    // and refB (committed expert_cache_ctrl).  After each demand the compute die
    // works TC cycles; the scheduler PREFETCHES the next demand's expert during
    // that window.  Run OFF then ON; report demand_stall_cycles + reduction vs TC.
    $readmemh("tools/glm_trace.hex", decode_tr);   // first NREQ decode accesses

    // (5a) committed expert_cache_ctrl reference on the trace (demand-only).
    //      First NREQ decode accesses are all DISTINCT/cold -> every demand misses.
    resetB;
    ref_h=0; ref_m=0;
    for (i=0;i<NREQ;i=i+1) begin
      demandR(decode_tr[i], gh);
      ref_seq[i]=gh; if (gh) ref_h=ref_h+1; else ref_m=ref_m+1;
    end
    if (ref_h!==0) fail("T5 committed ctrl not all-miss on cold decode window");
    pass = pass + 1;

    // (5b) pf-OFF run of dutB -> MUST equal the committed ctrl demand-by-demand.
    resetB;
    pf_validB<=1'b0;
    off_h=0;
    for (i=0;i<NREQ;i=i+1) begin
      demandB(decode_tr[i], gh);
      off_seq[i]=gh; if (gh) off_h=off_h+1;
      if (off_seq[i]!==ref_seq[i]) fail("T5 pf-OFF demand outcome differs from committed ctrl");
      repeat (4) @(posedge clk);            // compute window (pf-off stall is gap-independent)
    end
    if (hit_countB!==ref_h[31:0] || miss_countB!==ref_m[31:0])
      fail("T5 pf-OFF counts != committed expert_cache_ctrl");
    pass = pass + 1;
    stall_off = demand_stallB;
    $display("T5 pf-OFF: %0d demands hits=%0d miss=%0d  == committed expert_cache_ctrl  demand_stall=%0d",
             NREQ, hit_countB, miss_countB, stall_off);

    // (5c) pf-ON sweep over a few compute-window lengths TC.
    //      Independent prediction (cold/distinct window, no eviction): demand 0 cold-
    //      misses; every later demand was prefetched in the prior window -> HIT.
    for (tci=0; tci<3; tci=tci+1) begin
      case (tci) 0:TC=2; 1:TC=12; default:TC=28; endcase
      resetB;
      on_h=0;
      for (i=0;i<NREQ;i=i+1) begin
        demandB(decode_tr[i], gh);
        on_seq[i]=gh; if (gh) on_h=on_h+1;
        if (i==0 && gh)  fail("T5 pf-ON demand 0 should be a cold MISS");
        if (i>0  && !gh) fail("T5 pf-ON prefetched demand should HIT");
        // compute window: PREFETCH the next demand's expert, then compute TC cycles
        if (i<NREQ-1) begin pf_expert_idB<=decode_tr[i+1]; pf_validB<=1'b1; end
        @(posedge clk); pf_validB<=1'b0;       // hint accepted from idle
        repeat (TC) @(posedge clk);            // remainder of the compute window
      end
      pass = pass + 1;                          // exact pf-ON outcome sequence held
      // CORRECTNESS-PRESERVED guard: prefetch only converts demand MISSES into
      // zero-stall HITS -- it must NEVER turn a pf-OFF hit into a miss (and every
      // demand still resolves its requested expert).  (Note: literal hit/miss-bit
      // identity is the OPPOSITE of latency-hiding -- a hidden miss IS a hit -- so
      // the right invariant is this Pareto check, not on_seq==off_seq.)
      for (i=0;i<NREQ;i=i+1)
        if (off_seq[i] && !on_seq[i]) fail("T5 prefetch demoted a pf-OFF hit to a MISS");
      pass = pass + 1;
      stall_on = demand_stallB;
      stall_at[tci] = stall_on;
      red = (stall_off==0) ? 0 : (100*(stall_off-stall_on))/stall_off;
      $display("T5 pf-ON  TC=%0d: hits=%0d miss=%0d demand_stall=%0d  -> %0d%% stall cut vs OFF  (pf_hit=%0d)",
               TC, hit_countB, miss_countB, stall_on, red, pf_hitB);
    end
    // small TC may not hide the fetch (compute too short); the LARGEST TC must.
    if (stall_at[2] >= stall_off)
      fail("T5 large-TC prefetch failed to reduce demand stall");
    pass = pass + 1;
    $display("T5 ok: pf-OFF == committed expert_cache_ctrl; prefetch turns cold misses into");
    $display("       zero-stall hits as the compute window TC grows -- latency hidden, outcome correct.");

    // =================== PART C : GDDR6 hit-latency on prefetch =============
    // The decode trace is cold/all-distinct.  Under the compute-window model:
    //   * prefetch OFF -> every demand is a Flash MISS (~FLASH_LAT stall each).
    //   * prefetch ON  -> every demand after the cold first one was prefetched
    //     during the prior window, so it is a HIT.  But with GDDR6 a HIT is no
    //     longer free: it pays CACHE_HIT_LAT cycles for the GDDR6 read.  So
    //     prefetch CONVERTS the big FLASH_LAT stall into the small CACHE_HIT_LAT
    //     stall -- it does NOT drive stall to ~0.  The residual is the per-access
    //     GDDR6 read floor.  We measure OFF, ON@LAT0 (ideal), ON@LAT4 (realistic)
    //     and sweep CACHE_HIT_LAT in {0,2,4,8}, all on dutH0..H3 with IDENTICAL
    //     spacing (FLASH_LAT=FL2=%0d).
    hl[0]=0; hl[1]=2; hl[2]=4; hl[3]=8;
    TCa = 26; TCb = 40;   // both >= FLASH_LAT so the Flash prefetch fully lands

    // (C0) prefetch OFF baseline -- every demand misses on all four caches
    //      (CACHE_HIT_LAT is irrelevant when nothing is resident -> all equal).
    pc_run(1'b0, TCa);
    if (miss_countH0!==32'd200 || hit_countH0!==32'd0) fail("C0 OFF not all-miss");
    if (demand_stallH1!==demand_stallH0 || demand_stallH2!==demand_stallH0 ||
        demand_stallH3!==demand_stallH0)
      fail("C0 OFF stall differs across CACHE_HIT_LAT (should be hit-lat independent)");
    pass = pass + 1;
    c_off = demand_stallH0;
    $display("C0 pf-OFF (all Flash miss): demand_stall=%0d  (~%0d cyc/miss over %0d demands)",
             c_off, c_off/NREQ, NREQ);

    // (C-ON @ TCa) prefetch ON -- the per-TC OFF-vs-ON@LAT0-vs-ON@LAT4 comparison.
    pc_run(1'b1, TCa);
    // outcome correctness: demand 0 cold-misses, the other 199 are prefetched hits,
    // identically on all four caches (latency changes timing, never the outcome).
    if (miss_countH0!==32'd1 || hit_countH0!==32'd199) fail("C ON: expected 1 cold miss + 199 hits");
    if (hit_countH1!==hit_countH0 || hit_countH2!==hit_countH0 || hit_countH3!==hit_countH0 ||
        miss_countH1!==miss_countH0 || miss_countH2!==miss_countH0 || miss_countH3!==miss_countH0)
      fail("C ON: hit/miss outcome differs across CACHE_HIT_LAT");
    pass = pass + 1;
    // SANITY: a larger GDDR6 hit latency can only ADD stall -> monotone, and
    // LAT0 is the floor.  None may exceed the prefetch-OFF Flash stall.
    if (!(demand_stallH0 <= demand_stallH1 && demand_stallH1 <= demand_stallH2 &&
          demand_stallH2 <= demand_stallH3))
      fail("C ON: demand_stall not monotone in CACHE_HIT_LAT");
    if (demand_stallH3 >= c_off) fail("C ON: GDDR6 prefetch failed to beat Flash-miss OFF stall");
    pass = pass + 1;
    c_on0 = demand_stallH0; c_on4 = demand_stallH2;
    cut0  = (c_off==0)?0:(100*(c_off-c_on0))/c_off;
    cut4  = (c_off==0)?0:(100*(c_off-c_on4))/c_off;
    $display("C  TC=%0d  prefetch comparison (cut %% vs OFF):", TCa);
    $display("     OFF            demand_stall=%0d", c_off);
    $display("     ON CACHE_HIT_LAT=0 (ideal)     demand_stall=%0d  -> %0d%% cut", c_on0, cut0);
    $display("     ON CACHE_HIT_LAT=4 (GDDR6)     demand_stall=%0d  -> %0d%% cut", c_on4, cut4);
    $display("     => prefetch trades the big FLASH_LAT(%0d) miss for the small CACHE_HIT_LAT(4)",
             FL2);
    $display("        GDDR6 read: stall is cut ~%0d%%, NOT ~99%%; the residual is the GDDR6 floor.",
             cut4);

    // (C-sweep @ TCb) fixed large compute window: residual stall FLOOR vs CACHE_HIT_LAT.
    pc_run(1'b1, TCb);
    if (miss_countH0!==32'd1 || hit_countH0!==32'd199) fail("C sweep: expected 1 cold miss + 199 hits");
    if (!(demand_stallH0 <= demand_stallH1 && demand_stallH1 <= demand_stallH2 &&
          demand_stallH2 <= demand_stallH3))
      fail("C sweep: demand_stall not monotone in CACHE_HIT_LAT");
    pass = pass + 1;
    c_on0=demand_stallH0; c_on2=demand_stallH1; c_on4=demand_stallH2; c_on8=demand_stallH3;
    $display("C  TC=%0d  CACHE_HIT_LAT sweep -- residual stall floor (1 cold miss + 199 GDDR6 hits):",
             TCb);
    $display("     CACHE_HIT_LAT=%0d : demand_stall=%0d", hl[0], c_on0);
    $display("     CACHE_HIT_LAT=%0d : demand_stall=%0d  (+%0d vs LAT0)", hl[1], c_on2, c_on2-c_on0);
    $display("     CACHE_HIT_LAT=%0d : demand_stall=%0d  (+%0d vs LAT0)", hl[2], c_on4, c_on4-c_on0);
    $display("     CACHE_HIT_LAT=%0d : demand_stall=%0d  (+%0d vs LAT0)", hl[3], c_on8, c_on8-c_on0);
    $display("     floor grows ~linearly: +CACHE_HIT_LAT per resident hit (199 hidden hits).");
    $display("C ok: prefetch hides FLASH_LAT but leaves the irreducible GDDR6 read floor;");
    $display("      FUTURE WORK: a 2nd-level GDDR6->die read-ahead could hide CACHE_HIT_LAT too.");

    $display("ALL %0d TESTS PASSED", pass);
    $finish;
  end

  // global watchdog
  initial begin #50000000; $display("FAIL: timeout"); $fatal; end
endmodule
