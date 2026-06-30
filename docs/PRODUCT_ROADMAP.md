# Product Roadmap — GLM-5.2-FP8 accelerator (product, not research)

The `prototype` branch (frozen at `fee8501`) holds the **research prototype**: the full FP8
datapath + memory system + ultra-perf batching stack, **bit-exact and mechanism-proven at a
small-but-faithful slice**, with honest gaps documented. It answers *"does the architecture work,
and how fast can it go?"* — **yes**, and the levers are measured.

`main` now develops the **product**: a manufacturable accelerator that **runs the published
`zai-org/GLM-5.2-FP8` reliably**. The mindset shifts from *demonstrate + measure a mechanism* to
*run the real model correctly, at full scale, robustly, and ship it.*

---

## The research → product gap (what must change)

| Dimension | Prototype (have) | Product (need) |
|---|---|---|
| Correctness scope | operator-level bit-exact vs the FP8 contract on **synthetic** weights; argmax 4/31/20 on the slice | the **real 753 GB checkpoint** produces the **real model's tokens** end-to-end |
| Scale | small faithful slice (MODEL_DIM 128, 6 layers, 8 experts) | full config (6144, 78 layers, 256 experts, vocab 154880, 1M ctx) |
| Batching/KV | PE_M batch shares pos/s_len/KV (dense-decode regime); ÷K is TB-driven; batched_moe covers B=4 | per-position causal KV at production widths; real draft chaining; full B coverage |
| Memory | DDR5/Flash/USB-C **stubbed** (TB) | licensed **PHY IP** integrated + signed off |
| Verification | bounded BMC + directed TBs at slice | coverage closure, constrained-random regression, gate-level sim, k-induction, production-width formal |
| Reliability | none | ECC, error recovery, CDC sign-off, reset/init hardening, DFT/scan |
| Physical | slice-scale yosys estimates | full synth + P&R + timing/power sign-off + DRC/LVS + tapeout/bitstream |
| Software | weight-pack tools (ckpt_pack/flash_layout) | host driver, tokenizer, runtime, quant-layout pipeline |
| Manufacturing | — | PCB, BOM, assembly, qualification |

---

## The #1 product gate (do this FIRST)

**Real-checkpoint full-model fidelity.** Until the actual GLM-5.2-FP8 weights produce the actual
model's next tokens through our datapath, there is no product. The bridge is built
(`tools/ckpt_pack.py` + `docs/BIT_ACCURACY.md` §C). Execute it on a GPU host:
1. Download `zai-org/GLM-5.2-FP8`; run a transformers/vLLM reference forward on a prompt → golden
   next-token logits/argmax.
2. Run our **bit-accurate software model** (`tools/glm_fp8_ref.py`, scaled to full config) on the
   same weights + prompt → argmax.
3. Assert token match over a corpus. Any divergence = a per-Linear scale-orientation / bf16-tail /
   RoPE/KV/MoE plumbing bug to fix in the RTL contract.

This is gating: it validates the *assembly* of hundreds of operators on *trained* weights — the
one thing the slice cannot.

---

## Product phases (main)

### P1 — Full-scale + real-model correctness  *(gates everything)*
- P1.1 Real-checkpoint validation (above). **Blocking.**
- P1.2 Scale the RTL/params to the full config; verify via the bit-accurate software model + a
  full-config RTL elaboration (synth-only where sim is intractable).
- P1.3 Close the prototype correctness gaps for product: **per-position causal KV** (replace the
  PE_M shared-pos decode-batch regime with a real per-row position/KV), real draft chaining for
  batched-verify, full B-coverage for batched_moe.

### P2 — Productize the RTL (robustness)
- P2.1 ECC on DDR5 + Flash; error detection / correction / retry / recovery paths.
- P2.2 Full CDC sign-off across USB / memory / compute clock domains; reset/init/boot-load hardening.
- P2.3 DFT: scan-chain insertion, MBIST for the SRAMs/caches, boundary scan.
- P2.4 Power: real ICG clock-gating cells, power domains, DVFS hooks, thermal budget.
- P2.5 Verification closure: functional + code coverage targets, constrained-random regression,
  gate-level (post-synth) sim, production-width controller formal + k-induction for unboundedness.

### P3 — Vendor IP + physical implementation
- P3.1 License + integrate the PHYs: DDR5 multi-channel controller+PHY, NVMe/Flash, USB-C device.
- P3.2 Target choice (the product fork — decide early):
  - **FPGA-card product** (faster to market): a data-center FPGA + on-board DDR + NVMe runs the
    real model streamed; bitstream via the vendor flow. Lower NRE, higher unit cost, slower.
  - **ASIC product** (cost/power/size at volume): standard-cell synth → floorplan → P&R → clock
    tree → timing/power/IR sign-off → DRC/LVS → tapeout → packaging. High NRE, months–years.
- P3.3 Full-scale STA (SDC), power sign-off, signal/power integrity.

### P4 — System, software, manufacturing
- P4.1 PCB: multi-layer controlled-impedance board (die/FPGA + DDR5 + Flash + USB-C), BOM, assembly.
- P4.2 Software stack: USB-C host driver, tokenizer, the checkpoint→Flash quant-layout pipeline
  (productionize `ckpt_pack.py`/`flash_layout.py`), inference runtime + continuous-batch scheduler.
- P4.3 Qualification: reliability (temp/voltage/aging), compliance (USB-C, EMI), yield/binning.

---

## What stays / what changes from the prototype

- **Keep (the core IP):** the FP8 datapath, MLA/DSA/MoE, the memory-system controllers
  (expert_cache_pf, kv_cache_pager, ddr5_xbar, flash_xbar, weight_loader, boot_loader), the
  batching stack (PE_M wrappers + model + batched_moe + spec_batched_top), the optimizations
  (BFP accumulator, parallel indexer, decompressor), the formal harnesses, the bit-accuracy kit.
  These are the hard, validated parts — they carry forward.
- **Change (the mindset):** every prototype "demonstrates the mechanism / honest gap / TB-driven"
  becomes a closed, full-scale, covered, signed-off product feature. Correctness is now *the real
  model on real weights*, not a slice.
- **Out of pure RTL (but on the product critical path):** PHY IP, physical implementation,
  software, PCB, manufacturing. These dominate product cost/time and are mostly vendor/EDA/board
  work, not algorithm design.

## Immediate next step (product)

**P1.1 + P1.3a**: run the real-checkpoint validation procedure (needs a GPU host) and, in parallel
on `main`, start closing the per-position causal-KV gap (the prototype's PE_M shared-pos
limitation) so batched decode is position-accurate for product. Everything else (P2–P4) sequences
behind a green P1.
