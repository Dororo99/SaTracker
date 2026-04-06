# SaTracker 적용 플랜 (260406_01 기반, 수정본)

## 0) 목표/성공 기준
- **목표**: Stage-1 semantic mIoU(특히 `divider`, `boundary`) 개선 + `ped_crossing` 비열화.
- **핵심 KPI**
  - primary: class-wise mIoU (`divider`, `boundary`, `ped_crossing`) + mean mIoU
  - secondary: boundary F-score, thin-structure connectivity 지표
  - 안정성: seed 3회 평균/표준편차

> 주의: 현재 evaluator(`plugin/datasets/evaluation/raster_eval.py`)는 IoU/mIoU만 계산하므로, boundary/connectivity 지표는 먼저 평가 도구를 추가해야 함.

---

## 1) 사전 고정 (코드/재현성/평가)

### 1.1 기준 코드/실험 고정
- 기준 config: `plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py`
- 기준 스크립트: `scripts/train_satmaptracker_stage1_skeleton.sh`
- 관련 모듈
  - sat encoder: `plugin/models/backbones/satmae_encoder.py`, `plugin/models/backbones/satellite_encoder.py`
  - sat fusion: `plugin/models/backbones/bevformer/sat_encoder.py`, `plugin/models/backbones/conv_fusion.py`
  - mapper: `plugin/models/mapers/SatMapTracker.py`
  - seg/loss: `plugin/models/heads/MapSegHead.py`, `plugin/models/losses/seg_loss.py`

### 1.2 재현성 러너 먼저 추가
- `scripts/train_satmaptracker_stage1_skeleton.sh`에 seed loop(0/1/2) + `--seed` + `--deterministic` 전달 옵션 추가
- `tools/train.py`가 seed 인자를 이미 지원하므로 이를 활용
- 각 seed 결과를 JSON/CSV로 저장하고 mean/std 자동 집계

### 1.3 평가 지표 확장 먼저 완료
- boundary F-score + connectivity metric 계산 스크립트 추가 (기존 IoU evaluator와 병행)
- 단계별 승격/롤백 판단은 **확장 지표 + mIoU**를 함께 사용

---

## 2) 적용 순서 (한 번에 한 축만 변경)

## Phase A — Satellite를 semantic prior로 전환

### A1. Prior-only residual logit correction (정합/게이트 제외)
- 목표: raw feature fusion 제거 후 prior 기반 최소 구조 검증
- 핵심 구현
  - `P_sat(divider,boundary,ped_crossing)` + `U_sat(confidence)` 예측
  - `S_final = S_cam + U_sat * ΔS_sat` (거리/클래스 gate는 아직 미적용)
- **필수 격리 조건**
  - legacy sat cross-attn OFF
  - `conv_fusion` OFF
  - 즉, 문서(260406_01.md) 권고대로 첫 실험은 residual-logit 경로만 검증
- 완료 기준
  - baseline 대비 mIoU 비열화 없음 + gate/보정값 로그 정상

### A2. class-aware + distance-aware gate 추가
- 핵심 구현
  - `α(r,c)` 도입: 거리/클래스별 gate
  - `ped_crossing` 초기 gate는 낮게 시작
- 완료 기준
  - `divider/boundary` 개선 또는 안정성 개선(분산 감소)
  - `ped_crossing` 과보정 억제

### A3. alignment-aware warp head 추가
- 핵심 구현
  - `[F_cam || F_sat] -> Conv -> (Δx, Δy)`
  - `grid_sample`로 sat prior warp 후 fusion
  - offset regularization + clipping
- 완료 기준
  - A2 대비 mIoU/안정성 추가 개선

---

## Phase B — Loss를 직접 최적화 (구현 게이트 포함)

### B0. loss 모듈 구현/등록 먼저
- 현재 registry는 focal/dice/skeleton 중심이므로, 아래 순서로 구현:
  1) Lovasz
  2) soft-clDice
  3) ABL
- 파일
  - `plugin/models/losses/seg_loss.py` 분리/확장
  - `plugin/models/losses/__init__.py` 등록
  - `plugin/models/heads/MapSegHead.py`에서 loss 조합 입력 가능하게 확장
- 검증
  - shape/numeric unit check(간단 torch 스모크)

### B1. loss 조합 점진 적용
- 조합 목표
  - `divider`: focal + dice + `0.5*lovasz` + `0.1~0.2*soft-clDice`
  - `boundary`: focal + dice + `0.5*lovasz` + `0.05~0.1*ABL`
  - `ped_crossing`: focal + dice + `0.5*lovasz`
- 적용 순서
  - B1-1: Lovasz only
  - B1-2: + divider clDice
  - B1-3: + boundary ABL
- 완료 기준
  - `divider/boundary` 개선, `ped_crossing` 비열화

---

## Phase C — Trainable satellite branch 안정화

### C1. normalization/LR/schedule
- SatelliteEncoder BN -> GN/LN 또는 BN stats freeze
- sat branch lr = camera의 0.1x부터 시작
- warmup 구간 sat encoder freeze 후 점진 unfreeze
- training 시 satellite prior dropout 추가

---

## Phase D — Late fusion 의존 축소

### D1. projected PV auxiliary supervision
- satellite prior를 camera view로 projection
- PV segmentation auxiliary(가능하면 depth auxiliary 병행)

### D2. training-only semantic guidance
- training에서만 sat alignment/consistency 사용
- inference graph에서 satellite 경로 제거 옵션 유지

---

## 3) 실험 매트릭스 (엄격 순차)
1. Baseline(3 seeds)
2. A1 (prior-only, legacy sat fusion OFF)
3. A2 (A1 + class/range gate)
4. A3 (A2 + warp)
5. B1-1 (A3 + Lovasz)
6. B1-2 (B1-1 + clDice(divider))
7. B1-3 (B1-2 + ABL(boundary))
8. C1
9. D1
10. D2

> 각 단계는 바로 이전 단계 best checkpoint 기준 누적.  
> 단계 승격은 3-seed mean/std 기준으로만 결정.

---

## 4) 승격/롤백 규칙
- 승격: `divider/boundary` mIoU 개선 + `ped_crossing` 비열화 + 분산 악화 없음
- 롤백: 2회 연속 평균 mIoU 하락 + 분산 증가
- 기록: 단계별
  - class-wise mIoU, mean mIoU
  - boundary F-score/connectivity
  - gate/offset 통계

---

## 5) 즉시 실행 TODO
- [ ] seed runner(0/1/2) + deterministic + 집계 스크립트 추가
- [ ] boundary/connectivity evaluator 추가
- [ ] A1 prior-only 구현 (legacy sat cross-attn/conv_fusion OFF)
- [ ] A2 gate 도입(클래스/거리)
- [ ] A3 warp head + offset regularization
- [ ] B0 loss 모듈 등록(Lovasz/clDice/ABL)

---

## 6) 주요 리스크/대응
- 정합 실패: warp offset clipping + regularization
- ped_crossing 붕괴: class-aware 저게이트 + 별도 임계치
- 학습 불안정: sat lr 축소 + warmup freeze + normalization 안정화
- 복잡도 증가: D2(training-only guidance)로 inference 경량 유지
