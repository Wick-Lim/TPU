`timescale 1ns/1ps
//============================================================================
// expert_predictor_tb.v -- VERIFY + MEASURE the GLM-5.2 expert prefetch predictor
//----------------------------------------------------------------------------
// Drives the GLM-scale routing trace (tools/glm_trace.hex from route_trace.py:
// decode pattern, token-major, 30 tokens x 75 layers x 8 experts = 18000 ids,
// id = layer*256 + expert) through expert_predictor and measures, against the
// ACTUAL next-token selections, the research metrics for a prefetch predictor:
//
//   * PRECISION  : of the experts we emit as prefetch HINTS, how many were
//                  really demanded that turn (the over-prefetch waste).
//   * RECALL     : of the 8 experts actually demanded, how many we hinted.
//   * AT-LEAST-1 : fraction of (token,layer) turns where >=1 hinted expert was
//                  truly demanded -- the binding fine-grained target (>90%).
//   * HINTS/TOKEN: prefetch hints emitted per token (the bandwidth cost).
//
// PROTOCOL per token t (1..29), per layer L (0..74):
//   1. QUERY the predictor for layer L using history of tokens 0..t-1
//      (combinational top-P read of L's freq row).  Score its HINTS vs the
//      experts layer L actually selects at token t (the future, from the trace).
//   2. UPDATE the predictor with token t's actual layer-L selections.
//   Each layer's freq row is independent, so query-before-update per (t,L) keeps
//   the prediction strictly causal (only the past has been observed).
//
// Plus SANITY + X-AWARE asserts on prediction/confidence/threshold behaviour.
// Prints "ALL <N> TESTS PASSED" iff every assert holds.
//
//   iverilog -g2012 -Wall -I src -o sim test/expert_predictor_tb.v src/expert_predictor.v
//   vvp sim
//============================================================================
module expert_predictor_tb;
  // GLM-5.2 MoE scale
  localparam integer NE   = 256;   // experts/layer
  localparam integer NK   = 8;     // top-k per layer
  localparam integer NL   = 75;    // MoE layers
  localparam integer TP   = 8;     // predictions/query (== top-k footprint per layer)
  localparam integer FW   = 4;     // freq counter width
  localparam integer HD   = 64;    // aging window (updates/layer) = 8 tokens
  localparam integer EIDW = 8;     // clog2(256)
  localparam integer LAYW = 7;     // clog2(75)
  localparam integer GIDW = 15;    // clog2(75*256=19200)
  localparam integer PCW  = 4;     // clog2(TP+1)

  localparam integer NTRACE = 18000;
  localparam integer TOKENS = NTRACE/(NL*NK);   // 30
  localparam integer PERTOK = NL*NK;            // 600

  // CONF_THRESH is a top-module override per instance below; we instantiate two
  // predictors (low + high threshold) to show the precision/over-prefetch knob.
  reg clk=1'b0, rst=1'b1;
  always #5 clk = ~clk;

  // shared stimulus
  reg                upd_valid;
  reg  [LAYW-1:0]    upd_layer;
  reg  [EIDW-1:0]    upd_expert;
  reg  [LAYW-1:0]    pred_layer;

  // --- instance A: low threshold (aggressive prefetch) ---
  wire [TP*GIDW-1:0] a_id;   wire [TP*FW-1:0] a_conf;
  wire [TP-1:0]      a_vmask, a_hmask;  wire [PCW-1:0] a_hn;
  wire               a_pfv;  wire [GIDW-1:0] a_pfe;
  expert_predictor #(.N_EXPERT(NE), .TOPK(NK), .N_LAYER(NL), .HIST_DEPTH(HD),
                     .CONF_THRESH(1), .TOP_P(TP), .FREQ_W(FW)) dutA (
    .clk(clk), .rst(rst), .upd_valid(upd_valid), .upd_layer(upd_layer), .upd_expert(upd_expert),
    .pred_layer(pred_layer), .pred_id_flat(a_id), .pred_conf_flat(a_conf),
    .pred_valid_mask(a_vmask), .pred_hint_mask(a_hmask), .pred_hint_n(a_hn),
    .pf_hint_valid(a_pfv), .pf_hint_expert(a_pfe));

  // --- instance B: high threshold (selective prefetch) ---
  wire [TP*GIDW-1:0] b_id;   wire [TP*FW-1:0] b_conf;
  wire [TP-1:0]      b_vmask, b_hmask;  wire [PCW-1:0] b_hn;
  wire               b_pfv;  wire [GIDW-1:0] b_pfe;
  expert_predictor #(.N_EXPERT(NE), .TOPK(NK), .N_LAYER(NL), .HIST_DEPTH(HD),
                     .CONF_THRESH(4), .TOP_P(TP), .FREQ_W(FW)) dutB (
    .clk(clk), .rst(rst), .upd_valid(upd_valid), .upd_layer(upd_layer), .upd_expert(upd_expert),
    .pred_layer(pred_layer), .pred_id_flat(b_id), .pred_conf_flat(b_conf),
    .pred_valid_mask(b_vmask), .pred_hint_mask(b_hmask), .pred_hint_n(b_hn),
    .pf_hint_valid(b_pfv), .pf_hint_expert(b_pfe));

  // ---- trace + derived per-(token,layer) expert sets ----
  reg [GIDW-1:0] tr [0:NTRACE-1];
  // actual[t][l][j] = j-th selected expert (within-layer id) at token t, layer l
  reg [EIDW-1:0] actual [0:TOKENS*NL*NK-1];
  function integer AIDX(input integer t, input integer l, input integer j);
    AIDX = ((t*NL)+l)*NK + j;
  endfunction

  integer pass, fail;
  task chk(input cond, input [255:0] name);
    begin
      if (cond) pass = pass + 1;
      else begin fail = fail + 1; $display("  FAIL: %0s", name); end
    end
  endtask

  // measurement accumulators (per instance A and B)
  integer hintsA, correctA, atleast1A, demandedA, queriesA;
  integer hintsB, correctB, atleast1B, demandedB, queriesB;

  // helpers to test membership of a within-layer expert id in actual[t][l]
  function is_demanded(input integer t, input integer l, input [EIDW-1:0] e);
    integer j; begin
      is_demanded = 1'b0;
      for (j=0;j<NK;j=j+1) if (actual[AIDX(t,l,j)]==e) is_demanded=1'b1;
    end
  endfunction

  integer t, l, j, p;
  reg [EIDW-1:0] e_local;
  reg [GIDW-1:0] gid;
  integer turn_hits, turn_hint_cnt;
  reg     x_clean;

  // score the currently-presented prediction (combinational) of instance A or B
  // for (t,l): id/conf/hint buses passed in flattened.  Returns nothing; updates
  // the right accumulators via the `which` selector (0=A,1=B).
  task score(input integer which, input integer t, input integer l,
             input [TP*GIDW-1:0] idf, input [TP-1:0] hmask);
    integer pp; reg [EIDW-1:0] el; reg [GIDW-1:0] g; integer th, thc;
    begin
      th = 0; thc = 0;
      for (pp=0; pp<TP; pp=pp+1) begin
        if (hmask[pp]) begin
          thc = thc + 1;
          g  = idf[pp*GIDW +: GIDW];
          el = g % NE;                 // recover within-layer id
          if (is_demanded(t,l,el)) th = th + 1;
        end
      end
      if (which==0) begin
        hintsA = hintsA + thc; correctA = correctA + th;
        if (th>0) atleast1A = atleast1A + 1;
        demandedA = demandedA + NK; queriesA = queriesA + 1;
      end else begin
        hintsB = hintsB + thc; correctB = correctB + th;
        if (th>0) atleast1B = atleast1B + 1;
        demandedB = demandedB + NK; queriesB = queriesB + 1;
      end
    end
  endtask

  // apply token t's layer-l selections as updates (NK cycles)
  task update_layer(input integer t, input integer l);
    integer j2;
    begin
      for (j2=0;j2<NK;j2=j2+1) begin
        @(negedge clk);
        upd_valid  <= 1'b1;
        upd_layer  <= l[LAYW-1:0];
        upd_expert <= actual[AIDX(t,l,j2)];
        @(posedge clk);
        @(negedge clk);
        upd_valid  <= 1'b0;
      end
    end
  endtask

  integer warm_e;

  initial begin
    pass=0; fail=0;
    hintsA=0; correctA=0; atleast1A=0; demandedA=0; queriesA=0;
    hintsB=0; correctB=0; atleast1B=0; demandedB=0; queriesB=0;

    $readmemh("tools/glm_trace.hex", tr);
    // de-interleave the trace into per-(token,layer) within-layer expert sets.
    for (t=0;t<TOKENS;t=t+1)
      for (l=0;l<NL;l=l+1)
        for (j=0;j<NK;j=j+1) begin
          gid = tr[t*PERTOK + l*NK + j];
          // sanity: the layer of this id must equal l (trace is layer-sequential)
          actual[AIDX(t,l,j)] = gid % NE;
        end

    // reset
    upd_valid=1'b0; upd_layer=0; upd_expert=0; pred_layer=0;
    rst=1'b1; repeat(4) @(posedge clk); @(negedge clk); rst=1'b0; @(negedge clk);

    //========================================================================
    // (1) SANITY + X-AWARE asserts (before the trace run, clean state)
    //========================================================================
    // empty table -> no valid predictions, no hints, top-1 hint low.
    pred_layer = 3; #1;
    chk(a_vmask==={TP{1'b0}}, "empty: no valid predictions");
    chk(a_hmask==={TP{1'b0}}, "empty: no hints");
    chk(a_hn  ==={PCW{1'b0}}, "empty: hint count 0");
    @(posedge clk); #1;
    chk(a_pfv===1'b0, "empty: registered pf_hint_valid low");
    // X-aware: a defined pred_layer must yield fully-defined (non-X) outputs.
    pred_layer = 5; #1;
    x_clean = (^a_id !== 1'bx) && (^a_conf !== 1'bx) &&
              (^a_vmask !== 1'bx) && (^a_hmask !== 1'bx) && (^a_hn !== 1'bx);
    chk(x_clean, "X-aware: outputs defined for defined pred_layer");

    // teach layer 5 a clear winner: hammer within-layer expert 200 HD/2 times.
    for (warm_e=0; warm_e<HD/2; warm_e=warm_e+1) begin
      @(negedge clk); upd_valid<=1'b1; upd_layer<=5; upd_expert<=8'd200;
      @(posedge clk); @(negedge clk); upd_valid<=1'b0;
    end
    pred_layer = 5; #1;
    chk(a_vmask[0]===1'b1, "learned: top-1 prediction valid");
    chk((a_id[0 +: GIDW] % NE)===8'd200, "learned: top-1 is the hammered expert 200");
    chk(a_hmask[0]===1'b1, "learned: top-1 emits a hint (conf>=thresh)");
    chk((a_conf[0 +: FW]) >= 4'd4, "learned: top-1 confidence saturated/high");
    @(posedge clk); #1;
    chk(a_pfv===1'b1 && (a_pfe % NE)===8'd200, "learned: registered pf hint = expert 200");
    // threshold gating: a single touch to layer 6 expert 7 -> conf 1.
    @(negedge clk); upd_valid<=1'b1; upd_layer<=6; upd_expert<=8'd7;
    @(posedge clk); @(negedge clk); upd_valid<=1'b0;
    pred_layer = 6; #1;
    chk(a_vmask[0]===1'b1, "thresh: layer6 has a valid prediction (conf=1)");
    chk(a_hmask[0]===1'b1, "thresh: conf=1 >= CONF_THRESH(1) -> A hints");
    chk(b_hmask[0]===1'b0, "thresh: conf=1 <  CONF_THRESH(4) -> B suppresses hint");
    chk(b_vmask[0]===1'b1, "thresh: B still reports the prediction as valid");

    // fresh reset before the causal trace measurement
    rst=1'b1; repeat(4) @(posedge clk); @(negedge clk); rst=1'b0; @(negedge clk);

    //========================================================================
    // (2) CAUSAL TRACE MEASUREMENT
    //========================================================================
    // token 0: warm-up only (no future-from-history yet) -- just update.
    for (l=0;l<NL;l=l+1) update_layer(0, l);
    // tokens 1..: query (history 0..t-1) then update with token t.
    for (t=1;t<TOKENS;t=t+1) begin
      for (l=0;l<NL;l=l+1) begin
        pred_layer = l[LAYW-1:0]; #1;     // combinational top-P settles
        score(0, t, l, a_id, a_hmask);
        score(1, t, l, b_id, b_hmask);
        update_layer(t, l);
      end
    end

    //========================================================================
    // (3) REPORT
    //========================================================================
    $display("========================================================================");
    $display("GLM-5.2 expert PREFETCH PREDICTOR -- per-layer freq/locality table");
    $display("  trace=tools/glm_trace.hex  N_EXPERT=%0d top-k=%0d N_LAYER=%0d  TOP_P=%0d FREQ_W=%0d HIST_DEPTH=%0d",
             NE, NK, NL, TP, FW, HD);
    $display("  causal turns scored = %0d  (tokens 1..%0d x %0d layers)", queriesA, TOKENS-1, NL);
    $display("------------------------------------------------------------------------");
    $display("  CONF_THRESH=1 (aggressive):");
    $display("    hints/turn=%0.2f  hints/token=%0.1f", 1.0*hintsA/queriesA, 1.0*hintsA/(TOKENS-1));
    $display("    precision = %0d/%0d = %0.1f%%   (hinted experts that were demanded)",
             correctA, hintsA, hintsA? 100.0*correctA/hintsA : 0.0);
    $display("    recall    = %0d/%0d = %0.1f%%   (of 8 demanded/turn, fraction hinted)",
             correctA, demandedA, 100.0*correctA/demandedA);
    $display("    at-least-1= %0d/%0d = %0.1f%%   (turns with >=1 correct hint)",
             atleast1A, queriesA, 100.0*atleast1A/queriesA);
    $display("  CONF_THRESH=4 (selective):");
    $display("    hints/turn=%0.2f  hints/token=%0.1f", 1.0*hintsB/queriesB, 1.0*hintsB/(TOKENS-1));
    $display("    precision = %0d/%0d = %0.1f%%",
             correctB, hintsB, hintsB? 100.0*correctB/hintsB : 0.0);
    $display("    recall    = %0d/%0d = %0.1f%%", correctB, demandedB, 100.0*correctB/demandedB);
    $display("    at-least-1= %0d/%0d = %0.1f%%", atleast1B, queriesB,
             queriesB? 100.0*atleast1B/queriesB : 0.0);
    $display("------------------------------------------------------------------------");

    // measurement asserts: the binding research target + sanity bounds.
    chk(hintsA > 0,                              "meas: A emits hints");
    chk(correctA <= hintsA,                      "meas: A correct <= hints (precision<=100%)");
    chk(correctA <= demandedA,                   "meas: A correct <= demanded (recall<=100%)");
    chk(100.0*atleast1A/queriesA >= 90.0,        "meas: A at-least-1 recall >= 90% (binding target)");
    chk(hintsB <= hintsA,                        "meas: higher threshold emits <= hints (selectivity)");
    chk(hintsB==0 || (100.0*correctB/hintsB) >= (100.0*correctA/hintsA) - 0.001,
                                                 "meas: higher threshold precision >= aggressive");

    $display("========================================================================");
    if (fail==0) $display("ALL %0d TESTS PASSED", pass);
    else         $display("%0d TEST(S) FAILED (%0d passed)", fail, pass);
    $finish;
  end

  initial begin #500000000; $display("FAIL: timeout"); $fatal; end
endmodule
