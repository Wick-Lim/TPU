# 다음 단계 계획 — GLM-5.2-FP8 가속기 (P1.1 실체크포인트 검증 제외)

> **범위 결정:** 실제 753B 모델을 GPU로 돌려 대조하는 P1.1 트랙은 **완전 제외**.
> 검증은 기존 **모듈 단위 유닛테스트(각 TB의 독립 fp64/fp8 golden, iverilog/CPU, GPU 0)** 방식을 신뢰 기준으로 삼는다.
> 아래 모든 작업은 **GPU 불필요**. file:line 참조는 이번 감사에서 실제 코드로 검증됨.

## 이번 감사에서 검증된 "지금 고쳐야 할 실제 갭" (리드)

1. **전체칩 synth 게이트가 없다.** `make synth`는 `hierarchy -top TPU`(레거시 스칼라 코어)만 검사한다(Makefile:505). GLM top `glm_fp8_system_cdc`는 **whole-chip 구조 게이트가 전무**. → **C1**
2. **sparse-DSA 마스킹 버그.** `mla_attn_fp8.v:1068-1073`이 실제 키 인덱스 `sel_list[sf_feed_i]`가 아니라 **선택 슬롯 `sf_feed_i`로 마스크**한다. dense fallback에선 `sel_list[s]=s`라 no-op(그래서 테스트 통과)이지만, sparse+per-row extent에선 틀림. line 81이 sparse PE_M>1을 out-of-scope로 선언 중. → **B1 + B2**
3. **P2 신뢰성 유닛 + weight_decomp가 어떤 product top에도 인스턴스화 안 됨.** `reset_sync`, `ecc_mem_wrap`, `mbist_ctrl`, `icg_cell`, `clk_en_ctrl`, `weight_decomp/2` 전부 유닛검증만 되고 배선된 곳 없음. → **C3, C9, C7**
4. **`weight_decomp`은 tok/s를 실제로 움직이는 유일한 die-side 레버**(Flash-BW 바운드, 1.34×→~1.42× 실제 Flash 바이트 절감)인데 datapath에 안 붙어 있음. → **C9**
5. **`spec_chain_top`은 "syntax-checked 스켈레톤"보다 더 미완성.** TB 없음, pull 포트 전부 hard-zero(spec_chain_top.v:217-345), `mtp_emb` placeholder-zero, FSM C_IDLE→C_DONE에 DRAIN 없음 + multi-pass 커서 깨짐. → **B3 + B8**
6. **CI 전무** — `.github/` 없음. → quick win
7. **P1.2 "파라미터만 올리면 됨"은 한 가지 구조 변경을 과소평가.** `mla_attn_fp8`이 `scores`/`probs`/`vstore`와 `glm_softmax` LEN을 `S_MAX`(=1M 캐시주소 범위)로 잡아서, 어텐션 스케일업은 SWIN-vs-S_MAX 디커플(B7)이 필요. 풀config 기능 sim은 비현실적(LM head ~238M cyc/token). → **B4/B5/B7 + 정직한 스코핑**

---

## Track B — RTL 정확성 & 스케일 (no GPU)

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **B1** | `mla_attn_fp8.v:1068-1073` — 선택 슬롯이 아니라 실제 키 인덱스 `sel_list[sf_feed_i] < slen_r[r]`로 마스크 | dense에서 no-op 증명(`sel_list[s]=s`); `mla_attn_fp8_pslen_tb` byte-identical 통과 | S |
| **B2** | `test/mla_attn_fp8_sparse_perrow_tb.v` 작성 (S_MAX=8, TOPK=4, row별 상이한 `x`, PER_ROW_POS/SLEN 변형) — 각 배치 행 `===` 그 행의 `(x_r,pos_r,s_len_r)`로 돌린 PE_M=1 모듈; `kc_req`/`W_uk`/`W_uv` fetch 수 == distinct keys | **현 RTL에서 rows>0 실패**(sparse 갭 고정) / all-equal-x·PE_M=1 fold는 통과. B6의 pass-gate 오라클이 됨 | S |
| **B3** | `spec_chain_top.v` 값싼 수정 — `spec_batched_top.v:294-324`에서 accepted-prefix 커서 전진 포팅, `spec_batched_top` B_DRAIN(422-425) 미러하는 DRAIN 상태 추가, seed 불일치(step-0 post-final-norm `h_state` vs step≥1 pre-norm `db_y`)를 헤더에 문서화 | multi-pass가 커밋된 토큰 안에서 재시작 안 함; `done`이 drain beat와 레이스 안 함 | S |
| **B4** | P1.2 elaboration — `configs/full_glm52.vh`(MODEL_DIM=6144,L=78,N_EXPERT=256,TOPK=8,VOCAB=154880,KV_LORA=512,NOPE=192/ROPE=64,TOPK_ATTN=2048,POSW=20,THETA=8e6,BLK=128); `yosys hierarchy -top glm_model_fp8; check`(elaborate만) + `verilator -Wall`; `glm_matmul_fp8` leaf-synth @KMAX=16384 | 미해결 param/zero-width/포트 불일치 0; lint clean (256-expert/VOCAB 버스가 OOM이면 서브모듈 개별 elaborate) | M |
| **B5** | P1.2 중간크기 기능검증 — FFN/MoE/vocab param만 올리고(어텐션은 slice) `glm_model_fp8` 1토큰 vs in-TB fp64 golden; 비-/128 out-dim(`W_kr` out=64, NOPE=192)에서 `[128,128]` 블록스케일 부기 검증; 사이클수 노트로 **구조+중간크기 P1.2 계약** 확립 | 1토큰 FFN/vocab sim argmax 일치·X-clean; 블록스케일 TB가 `glm_fp8_contract` 레이아웃과 일치; 풀config 기능 sim은 비시도로 명시 | L |
| **B6** | `mla_attn_fp8` sparse per-row **union** 데이터패스 — row별 DSA 선택(각 행 `qrot[r]`/`slen_r`), distinct 키당 1회 `kc`+`W_uk`/`W_uv` fetch, row별 score/softmax/context를 각 행의 descending-score 순서로 재인덱싱; 먼저 param-gated serialize 스톱갭 옵션 | B2가 모든 행에서 bit-exact 통과(3-row distinct-extent 포함); fetch 수 == distinct union keys; dense TB 전부 byte-identical; line 81 caveat 제거 | XL |
| **B7** | SWIN 디커플 — `scores`/`probs`/`vstore`·`glm_softmax #(.LEN())`를 `SWIN=TOPK_ATTN=2048`로 재범위, `IDXW`/`kc_idx`는 full S_MAX(1M) 유지; default S_MAX=SWIN. **B6 이후 순서**(둘 다 `scores`/`vstore`/`sel_list` 재범위) | 기존 TB byte-identical @S_MAX=SWIN; S_MAX=64/SWIN=8 sparse TB가 fp64 golden 일치. 경고: SWIN=2048 `vstore`≈4.3 Gbit — scratch를 BRAM/pager로 옮기는 **1단계**일 뿐 | XL |
| **B8** | `spec_chain_top` 완전 승격 — `mn_*/tn_*/vn_*` pull 포트 승격(verify는 `spec_batched_top.v:165-207` 재사용, MTP는 `mtp_head_fp8` pull set 추가), `em_*` embed pull로 `mtp_emb=embed(prev_tok)`, seed 규약을 numpy/fp64 MTP-chain 레퍼런스로 확정, `test/spec_chain_top_tb.v`(committed==greedy, X-free, K∈{2,3}), K_eff 테스트, `make unittests` 편입 | `make unittests`가 `spec_chain_top` green; committed stream == 독립 greedy 레퍼런스(K∈{2,3}); seed 결정 헤더 기록 | L |

## Track C — 제품화 / DFT / formal (no GPU)

| # | 작업 | 수락 기준 | 노력 |
|---|------|-----------|------|
| **C1** | `make synth-glm` 추가 — `glm_fp8_system_cdc` set을 `hierarchy -top glm_fp8_system_cdc -check; proc; opt; check -assert; stat`; `make all`에 편입 | **최초 전체칩 구조 게이트**(현재 synth는 `-top TPU`만); exit 0, `check -assert` clean, `stat`에 leaf cell 전부 resolved | S |
| **C2** | `docs/P2_MEMORY_MAP.md` — 모든 비-TB `reg [] arr[]`(kv_cache_pager 768b ring, ddr5/flash_xbar 응답 FIFO, cdc_async_fifo mem, boot/weight 버퍼 vs `expert_cache_pf` directory)를 SECDED / parity-MBIST / off-die로 분류 | grep된 reg array 100% 커버 + 근거 | S |
| **C3** | `reset_sync`를 `glm_fp8_system_cdc`의 host_clk/core_clk 양 경계에 배선(현재 `host_rst`/`core_rst`는 pre-synchronized 가정; reset_sync는 검증됐지만 어디에도 미인스턴스) | `glm_fp8_system_cdc_tb` 통과 유지; 도메인별 STAGES-edge 동기 deassert directed case | S |
| **C4** | `ecc_mem_wrap` scrub-write-back + sticky `serr`/`derr` + ack(현재 read시 정정만 — 썩은 비트가 남아 double error로 누적 가능; P2.1은 retry/recovery 요구) | 새 `ecc_mem_wrap_tb`: `bd_we` 주입 → read(serr=1, 정정) → 재read ⇒ serr=0(scrub) | M |
| **C5** | `ddr5_xbar` 응답-FIFO no-overflow/underflow를 **unbounded k-induction**으로 승격 — `cnt[0:N_CH-1]`(ddr5_xbar.v:159) connect-bind, `test/formal/flash_xbar_ind_fv.v` 템플릿 미러 | `make formal-ind`에 통과하는 ddr5 run(base+step, 비-vacuity 재보증); `docs/FORMAL.md` 행 BOUNDED→UNBOUNDED | M |
| **C6** | DDR5/Flash payload 경로(weight 바이트 운반) + `kv_cache_pager` ring에 ECC; **위젠 워드에 대해 6개 committed BMC 증명 재파라미터/재검증** | fault-injection TB: single-bit 정정 / double-bit `derr`; 기존 유닛+formal 전부 green (ROW_BITS=768은 /64 아님 — lane 분할이 pager read latency 이동 가능) | L |
| **C7** | MBIST 래퍼(SRAM별 functional/BIST mux + daisy-chain + `bist_mode/done/fail`, `mbist_ctrl`용 registered-read 어댑터) + `clk_en_ctrl`/`icg_cell`을 실제 compute cluster에 + top `scan_enable`→모든 `icg_cell.test_en` | 주입 stuck-at에 MBIST `bist_fail=1`(macro id 정확), `bist_mode=0`서 bit-identical; gated-clock TB가 free-running과 bit-identical·runt 없음; `scan_enable`시 전 도메인 `gated_clk==clk` | L/XL |
| **C8** | CDC 사인오프 — 모든 async crossing에 SDC `set_false_path`/`set_max_delay` + `make cdc` 구조 체커; **"returned bytes not fed into die" loopback 폐쇄**(glm_fp8_system.v:82-89) — `xbar_resp_data`를 die의 weight/KV 소비로 valid/stall 핸드셰이크 뒤 되먹임, default-off(검증된 combinational 경로 불변) | `make cdc` unguarded crossing 0; loopback 모드가 combinational-stub와 동일 next token, `synth-glm check -assert` clean | M/XL |
| **C9** | `weight_decomp`(order-0)를 `glm_fp8_system.v` Flash→DDR5 refill 경로에 배선(`weight_decomp2` order-1은 빌드 옵션) + raw-vs-decompressed FP8 코드 byte-identical 증명 system TB | **tok/s를 움직이는 유일한 die-side 레버**(실제 Flash 바이트 1.34×→~1.42× 절감); 토큰 출력 불변, `make unittests` green | L |
| **C10** | P2 클로저 — `make all`에 `synth-glm` + ECC/MBIST/gated-clock system TB; 각 PRODUCT_ROADMAP P2 항목을 증명 TB에 링크; unit-proven vs system-proven 문서화 | `make all`이 P2 system TB green; 각 `ALL N TESTS PASSED` | S |

## Quick wins — 이번 주 시작 (no GPU)

- [ ] **전체칩 게이트:** `make synth-glm` 추가 → *Makefile*, *src/glm_fp8_system_cdc.v* (**C1**)
- [ ] **sparse 갭 고정:** 마스크 1줄 수정 + 회귀 오라클 → *src/mla_attn_fp8.v:1068-1073*, *test/mla_attn_fp8_sparse_perrow_tb.v* (**B1+B2**)
- [ ] **spec_chain 값싼 수정:** 커서 전진 + DRAIN + seed 헤더 노트 → *src/spec_chain_top.v* (port from *spec_batched_top.v:294-324, 422-425*) (**B3**)
- [ ] **reset 하드닝:** `reset_sync`를 CDC top에 배선 → *src/glm_fp8_system_cdc.v* (**C3**)
- [ ] **ECC/MBIST 작업 언블록:** 모든 reg-array 분류 → *docs/P2_MEMORY_MAP.md* (**C2**)
- [ ] **unbounded ddr5 증명:** connect-bind lift → *test/formal/ddr5_xbar_ind_fv.v* (템플릿 *flash_xbar_ind_fv.v:210-234*), *docs/FORMAL.md* (**C5**)
- [ ] **CI 부트스트랩:** `.github/workflows/ci.yml`(iverilog/yosys **0.66 핀**/z3 apt) + `make all + formal + formal-ind + bitacc + cache-study`
- [ ] **문서 정합화:** `make all` 스코프 명시, tok/s ~27 vs ~30+ (README:125), `q_lora/kv_lora` 값 통일(docs 1536/512 vs slice 64/32), `PROJECT_BRIEFING.md`/`NEXT_STEPS_PLAN.md` track 또는 gitignore

## 재조준 타임라인 (P1.1 제거)

```
WEEK 0 — Enabler (모든 no-GPU 검증의 게이트)
  make synth-glm + CI + 문서 정합화                         [C1, quick wins]

WEEKS 1-2 — 값싼 자체완결 수정 (병렬)
  Track B:  B1 mla 마스크 + B2 sparse 오라클 TB ; B3 spec_chain 커서/drain
  Track C:  C2 메모리 맵 ; C3 reset_sync ; C4 ecc scrub ; C5 ddr5 formal lift

WEEKS 3-5 — 중량급 (CI/오라클/synth-glm 존재 후)
  Track B:  B4 풀config elaborate → B5 중간크기 기능검증 + 사이클 계약
  Track C:  C9 weight_decomp 통합 (tok/s 레버) ; C8 CDC 사인오프 + loopback 폐쇄

WEEKS 4-8 — XL 구조 작업 (오라클 게이트)
  Track B:  B8 spec_chain 완전승격 → B6 sparse union 데이터패스(B2 게이트) → B7 SWIN 디커플(B6 후)
  Track C:  C6 payload/KV ECC + BMC 재검증 ; C7 MBIST+ICG+scan ; C10 P2 클로저
```

**게이트 관계:** Week-0 CI/`synth-glm`이 모든 B/C 검증 게이트 · **B2가 B6 게이트** · **B6는 B7보다 먼저**(shared `scores`/`vstore`/`sel_list`) · **C1+C2가 C6/C7 게이트**.

## 리스크 & 미지수 (no-GPU 범위)

- **B6 sparse union 순서 민감성 (진짜 XL).** serial fp32 softmax/context 체인은 순서 의존적. 잘못된 per-row gather 순서는 저비트 mismatch만 내서 fp8 노이즈로 오독하기 쉬움 — DSA emit 순서가 정확한 계약.
- **B7 SWIN 디커플이 메모리 재구조화를 과소평가.** `vstore`가 SWIN=2048서 ~4.3 Gbit — flop으로 비현실적. "elaborate clean" ≠ "스케일서 realizable". 풀config **기능** sim은 불가(LM head ~238M cyc/token) — P1.2를 "param 올리고 TB 돌리기"로 잡으면 조용히 안 끝남.
- **C6 formal 결합.** ECC check bit로 워드 확장 시 6개 committed BMC가 도는 datapath가 바뀜 → 재파라미터/재검증. ROW_BITS=768(/64 아님)이 pager read latency 이동 가능.
- **B8 spec_chain seed 규약은 배선이 아니라 설계 결정.** single-MTP-layer 자기회귀 체이닝은 1-layer 체크포인트 밖 외삽 — "정답"을 문서화된 수치 레퍼런스로 고정해야. K_eff에 영향(spec==greedy 안전성은 무관).
- **툴링 상한(정직한 경계).** CI yosys/iverilog는 로컬 **0.66** 베이스라인과 일치해야(connect-bind formal 트릭 의존). 실제 scan stitching·JTAG TAP·ATPG·static CDC 사인오프·풀config STA/power는 이 OSS 플로에 없는 상용툴 필요 — **P2는 hooks+harness+문서화된 hand-off를 제공하지, 측정된 coverage가 아님.** P3(PHY/FPGA-vs-ASIC/STA/power)·P4(PCB/driver/tokenizer/qual)는 설계상 out-of-scope.

## 브리핑 정정 (RTL/시스템 관련 — 코드로 검증됨)

3. **`make synth`는 product 계층을 게이트하지 않음.** `synth:`은 `hierarchy -top TPU`(Makefile:505, 레거시 스칼라). GLM top은 전체칩 구조 게이트 전무(C1로 폐쇄).
6. **`h_mtp`는 FP8 전용.** `src/mtp_head_fp8.v`만 포트 있음 — `src/mtp_head.v`(bf16)엔 없음. bf16 체인 레퍼런스는 추가 필요.
8. **`spec_chain_top`은 스켈레톤보다 더 미완성** (TB 없음, pull 포트 hard-zero:217-345, `mtp_emb` zero, DRAIN 없음, multi-pass 커서 깨짐).
9. **`reset_sync` 및 P2 프리미티브·`weight_decomp/2`가 어떤 product top에도 미인스턴스.**
10. **"모든 dim은 param bump"는 한 구조 변경 과소평가** — `mla_attn_fp8`이 scratch를 S_MAX(1M)로 사이징 → SWIN 디커플 필요(B7). RTL default(`Q_LORA=64`,`KV_LORA=32`,`POSW=20`,`S_MAX=8`)는 slice 값.
11. **`ecc_mem_wrap`은 read시 정정만** — scrub/retry 없음, P2.1 "retry/recovery" 미충족(C4로 폐쇄).
12. **문서 불일치:** README "~3→~30+ tok/s"(README:125) vs ~27; `make all` = `test hazard unittests lint synth formal`(Makefile:56)이고 `bitacc`/`cache-study`/`bcov`/`formal-ind`는 별도.
