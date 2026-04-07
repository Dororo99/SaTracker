# Skeleton Loss 통합 계획 (Stage1, Main Loss Only)

## 목표
`F_decide = DecidingModule(F_sensor, F_sat, Memory)` 구조에서,
- **Aux thin branch는 추가하지 않고**,
- 기존 **Main branch의 seg loss**에 skeleton loss만 추가한다.

요청 고정값:
- 대상 클래스: `divider`, `boundary`
- `num_dilation = 1`

---

## 1. 브랜치 구조

### Main branch (유지)
- `MapDecoder(F_decide) -> original map losses`
- 여기에 사용되는 segmentation loss를 아래처럼 확장:
  \[
  L_{seg}^{main} = L_{focal} + L_{dice} + \lambda_{skel} L_{skeleton(divider,boundary)}
  \]

### Aux branch
- **추가하지 않음**

### 간단 아키텍처(요약)
```text
F_sensor + F_sat + Memory
          │
          ▼
      DecidingModule
          │
          ▼
        F_decide
          │
          ▼
      MapDecoder
          │
          ▼
    MapSegHead(logits C=3)
      ├─ L_focal (all classes)
      ├─ L_dice  (all classes)
      └─ L_skeleton (divider,boundary only, num_dilation=1)
```

### Skeleton loss 부착 위치(명시)
- **부착 함수**: `plugin/models/heads/MapSegHead.py::forward_train`
- 기존
  - `seg_loss = self.loss_seg(preds, gts)`
  - `dice_loss = self.loss_dice(preds, gts)`
- 변경
  - 위 두 loss 계산 뒤, 같은 `preds/gts`에서 `loss_skeleton(preds, gts)`를 추가 계산
  - 단, `class_indices=[divider, boundary]`만 사용
  - 최종적으로 `seg + dice + skeleton`을 main seg loss에 합산

---

## 2. Loss 설계

최종 손실:
\[
L_{total} = L_{main,original} + \lambda_{skel} L_{skeleton(divider,boundary)}
\]

### 클래스 제한 (필수)
- Skeleton loss는 전체 클래스가 아니라
  - `divider`
  - `boundary`
  두 클래스 채널에서만 계산

### Skeleton 설정
- `num_dilation = 1` 고정

### 구현 포인트
- Focal/Dice는 기존 전체 클래스 로직 유지
- Skeleton만 class-mask로 부분 적용
- Skeleton 내부 클래스 기여도는 `divider:boundary = 2:1`로 가중

---

## 3. 코드 변경 위치 (repo 기준)

1) `plugin/models/losses/seg_loss.py`
- `MaskSkeletonLoss` 신규 추가
- 옵션:
  - `num_dilation=1`
  - `loss_weight` (현재 설정: `1.0`)
  - `class_indices` (예: `[1, 2]` = divider, boundary)
  - `class_weights` (현재 설정: `[2.0, 1.0]` = divider 2배)

2) `plugin/models/losses/__init__.py`
- `MaskSkeletonLoss` export 추가

3) `plugin/models/heads/MapSegHead.py`
- `loss_skeleton`(optional) 지원 추가
- `forward_train`에서
  - 기존 `seg_loss`, `dice_loss` 계산 유지
  - `loss_skeleton`이 설정된 경우만 skeleton loss 계산해 합산
  - 반환 시 skeleton loss를 log/외부합산에 노출

4) `plugin/models/mapers/MapTracker.py` / `plugin/models/mapers/SatMapTracker.py`
- `MapSegHead` 반환값 확장에 맞춰 log_vars 반영
  - 예: `seg_skeleton`, `seg_skeleton_t*`, (필요 시) `seg_skeleton_fused`, `seg_skeleton_sat`
- **새 branch/head 추가는 없음**

> 참고: SatMapTracker는 `seg_decoder`, `seg_decoder_fused`, `seg_decoder_sat` 모두 `MapSegHead`를 사용하므로,  
`seg_cfg`에 `loss_skeleton`을 넣으면 해당 decoder들에 동일 로직이 적용된다.

5) Config
- 우선 실험 타깃: `plugin/configs/satmaptracker/nuscenes_newsplit/satmaptracker_stage1_third.py`
- `model.seg_cfg`(필요 시 fused/sat seg_cfg도)에 skeleton loss 설정 추가
  - `loss_skeleton=dict(type='MaskSkeletonLoss', loss_weight=1.0, num_dilation=1, class_indices=[1,2], class_weights=[2.0,1.0])`
- 기존 `loss_seg`/`loss_dice`는 유지

---

## 4. 안전장치 / 구현 규칙

1. 클래스 인덱스 검증
- `class_indices=[1,2]`가 실제 `cat2id`와 일치하는지 assert/log로 검증

2. 빈 타깃 처리
- 배치에 divider/boundary GT가 없는 경우 skeleton loss는 0으로 안전 처리

3. 최소 침습 변경
- 기존 main pipeline/decoder/head 개수는 변경하지 않음
- skeleton은 기존 loss에 additive term으로만 주입

4. 가중치 보수적 시작
- 현재 `loss_weight=1.0`로 적용 (skeleton 총 계수, config 기준 10:1:1)
- 클래스 내부 가중은 `class_weights=[2.0,1.0]` 적용 (divider가 boundary 대비 2배)

---

## 5. 검증 계획 (Ablation)

1) Baseline: 기존(main only, Focal + Dice)
2) + Main Skeleton (Focal + Dice + Skeleton on divider/boundary, `num_dilation=1`)

비교 지표:
- divider/boundary IoU
- 연결성(끊김 감소) 관련 정성/정량 지표
- 전체 클래스 성능 저하 여부

---

## 6. 구현 순서

1. `MaskSkeletonLoss` 구현 (`num_dilation=1`, class mask, empty-target 안전 처리)
2. `MapSegHead`에 optional skeleton loss 경로 추가
3. Mapper log_vars 연결
4. stage1_third config에 skeleton loss 항목 연결
5. baseline vs +skeleton ablation 수행
