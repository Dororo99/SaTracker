# SatMapTracker 설계 분석: Sumin ConvFuser vs 이전 버전 비교

## 1. 성능 기준점

| 모델 | mIoU | 비고 |
|------|------|------|
| Sumin ConvFuser (SatDistillMapTracker) | **0.514** | Stage 1, teacher_mode |
| Kyungmin v1 (SatMapTracker, cross-attn) | - | Stage 2부터 fusion |
| Kyungmin v5 (SatMapTracker, addition) | - | Stage 1부터 fusion |

---

## 2. 세 모델 구조 비교

### 2.1 공통점
- 모두 `MapTracker` 기반 (동일한 backbone/decoder/seg_decoder 구조)
- Stage 1에서 `skip_vector_head=True` (seg만 학습)
- `neck = nn.Identity()` (neck이 실질적으로 없음)
- BEV 크기: 50H x 100W, embed_dims: 256

### 2.2 핵심 차이 요약

| | Sumin ConvFuser | Kyungmin v1 | Kyungmin v5 |
|---|---|---|---|
| **모델 클래스** | `SatDistillMapTracker` | `SatMapTracker` | `SatMapTracker` |
| **위성 인코더** | `SatelliteEncoder` (ResNet50+FPN) | `SatelliteEncoder` (ResNet50+FPN) | `SimpleSatEncoder` |
| **Fusion 방식** | `SatelliteConvFuser` (concat+conv, learnable) | Decoder-level cross-attention | element-wise addition (파라미터 없음) |
| **Fusion 시점** | Stage 1부터 | Stage 2부터 | Stage 1부터 |
| **Fusion 위치** | 별도 branch (pipeline 밖) | Decoder 내부 | `_post_backbone_hook` (pipeline 안) |
| **History buffer** | cam_bev only | cam_bev only | fused_bev (cam+sat) |
| **Seg supervision** | 이중 (cam_bev + fused_bev) | 단일 (cam_bev) | 단일 (fused_bev) |

---

## 3. 상세 구조 비교 (그림)

### Sumin ConvFuser (Stage 1) — mIoU 0.514

```
                                       history_bev_feats (cam only)
                                              ↑ append
                                              │
Camera ─→ BEVFormerBackbone ──────→ cam_bev ──┤
             ↑ temporal                       │
             │ self-attn                      ├──→ seg_decoder ──→ seg_loss ①  (cam supervision)
             │                                │
          history_bev ←───────────────────────┘
          (cam only)                          │
                                              ├──→ ConvFuser ──→ fused_bev
                                              │       ↑                │
AID4AD ─→ SatelliteEncoder ──→ sat_bev ──────┘       │                │
          (ResNet50+FPN)                    concat+conv          seg_decoder ──→ seg_loss ②  (fused supervision)
                                            (learnable)
                                                          total_loss = ① + ②
```

- history에는 cam_bev만 저장 → temporal pipeline 보존
- ConvFuser는 별도 branch → seg_decoder를 2번 호출 (cam, fused)
- 평가 시 fused_bev의 seg 결과 사용

### Kyungmin v1 (Stage 1 → Stage 2)

```
[Stage 1: fusion 없음, cam/sat 독립 학습]

Camera ─→ BEVFormerBackbone ──→ cam_bev ──→ seg_decoder ──→ seg_loss  (cam only)

AID4AD ─→ SatelliteEncoder ──→ sat_bev ──→ sat_seg_head ──→ seg_loss  (sat 독립)
          (ResNet50+FPN)                    (별도 head)


[Stage 2: cross-attention fusion, 뒤늦게 합류]

Camera ─→ BEVFormerBackbone ──→ cam_bev ──→ Transformer Decoder ──→ vector head
                                                    ↑                    │
                                          deformable cross-attn    seg_decoder
                                                    │
AID4AD ─→ SatelliteEncoder ──→ sat_bev ────────────┘
          (ResNet50+FPN)                  (key/value로 간접 참조)
```

- Stage 1에서 따로 학습 → Stage 2에서 만남 → feature space mismatch
- Cross-attention은 query 레벨 간접 참조 → BEV spatial 정보 활용이 우회적

### Kyungmin v5 (Stage 1)

```
                                      history_bev_feats (fused!)
                                              ↑ append
                                              │
Camera ─→ BEVFormerBackbone ──→ cam_bev       │
             ↑ temporal                │      │
             │ self-attn               (+) ──→ fused_bev ──→ seg_decoder ──→ seg_loss
             │                         ↑             │
          history_bev ←────────────────│─────────────┘
          (fused!!)                    │
                                       │
AID4AD ─→ SimpleSatEncoder ──→ sat_bev─┘
                                (파라미터 없는 단순 덧셈)
```

- fusion이 pipeline 안 (_post_backbone_hook) → fused_bev가 history에 들어감
- backbone이 다음 frame에서 fused history를 temporal reference로 받음
  → camera가 만든 적 없는 distribution → mismatch 발생
- 단순 addition → scale/distribution 차이 학습 불가

### Kyungmin v7 (Stage 1, 완료) — history_mode='fused', 공유 decoder

```
                                      history_bev_feats (fused)
                                              ↑ append
                                              │
Camera ─→ BEVFormerBackbone ──→ cam_bev ──────┤
             ↑ temporal                       │
             │ self-attn                      │
             │                          ConvFuser ──→ fused_bev
          history_bev ←───────────────────→   ↑            │
          (fused)                              │            ├──→ seg_decoder ──→ seg_loss (fused, parent)
                                               │            │
AID4AD ─→ SatelliteEncoder ──→ sat_bev ───────┘            │
          (ResNet50+FPN)                                    │
                                                            │
          추가 branch (current frame only):                 │
          cam_bev ──→ seg_decoder ──→ seg_loss ② (cam supervision)  ← 공유!
          sat_bev ──→ seg_decoder ──→ seg_loss ③ (sat supervision)  ← 공유!

                              total_loss = fused(parent) + ② + ③
```

- ConvFuser: residual 없음, sat backbone full lr (1.0x), cam backbone 0.1x
- **seg_decoder를 cam/sat/fused 3개가 공유** → gradient 간섭 문제

#### v7 실험 결과 (third split, 18 epoch)

| epoch | fused mIoU | cam mIoU | sat mIoU |
|-------|-----------|---------|---------|
| 3     | **0.492** | **0.482** | 0.383 |
| 6     | 0.480     | 0.471   | 0.394 |
| 9     | 0.464     | 0.456   | 0.400 |
| 12    | 0.452     | 0.450   | 0.407 |
| 15    | 0.438     | 0.439   | 0.404 |
| 18    | 0.429     | 0.432   | 0.404 |

**문제**: epoch 3이 피크, 이후 cam/fused 지속 하락. sat만 완만히 상승.

**원인 분석**:
1. 공유 seg_decoder가 fused/sat 분포 쪽으로 shift → cam-only 평가 성능 하락
2. fusion에 residual 없음 → cam_bev가 "sat과 합쳐졌을 때 좋은 feature"로 변질
   → cam 단독 품질 저하

### Kyungmin v8 (Stage 1, 실패) — residual + sat lr 0.1x

v7의 문제를 해결하기 위해 residual + sat backbone lr 0.1x를 시도.

- ConvFuser: **use_residual=True**, sat backbone **0.1x**, cam backbone 0.1x
- seg_decoder: 공유 (v7과 동일)

| epoch | fused mIoU | cam mIoU | sat mIoU |
|-------|-----------|---------|---------|
| 3     | 0.473     | 0.462   | 0.332 |
| 6     | 0.455     | 0.445   | 0.326 |

**결론**: sat backbone lr 0.1x가 주범. sat 학습이 너무 느려서 adaptation이 안 됨.
- satonly 실험에서도 sat backbone은 paramwise_cfg 없이 full lr로 학습했음
- v7에서도 sat 1.0x로 sat mIoU가 정상적으로 올라갔음
- → sat backbone을 0.1x로 줄인 것은 불필요한 제약

### Kyungmin v9 (Stage 1, 현재) — 분리 decoder + residual + sat lr 1.0x

v7/v8의 두 문제를 동시에 해결:

```
                                      history_bev_feats (fused)
                                              ↑ append
                                              │
Camera ─→ BEVFormerBackbone ──→ cam_bev ──────┤
             ↑ temporal                       │
             │ self-attn                      │
             │                    ConvFuser(+residual) ──→ fused_bev
          history_bev ←──────────────→  ↑                    │
          (fused)                        │                    ├──→ seg_decoder     ──→ seg_loss (fused)
                                         │                    │
AID4AD ─→ SatelliteEncoder ──→ sat_bev──┘                    │
          (ResNet50+FPN, lr 1.0x)                             │
                                                              │
          추가 branch (current frame only):                   │
          cam_bev ──→ seg_decoder_cam ──→ seg_loss ② (cam 전용)
          sat_bev ──→ seg_decoder_sat ──→ seg_loss ③ (sat 전용)

                              total_loss = fused(parent) + ② + ③
```

**변경점 (v7 대비)**:
1. **분리 decoder**: cam/sat 각각 전용 seg_decoder. fused는 parent seg_decoder 사용
   - gradient가 깨끗: cam_decoder → cam backbone만, sat_decoder → sat backbone만
   - fused_decoder → fused에만 최적화
   - 진단 지표 오염 해결: cam mIoU 하락이 feature 문제인지 decoder shift인지 구분 가능
   - MapSegHead가 conv 3개 수준으로 가벼워 3개로 늘려도 파라미터 부담 거의 없음
2. **ConvFuser residual**: `fused = Conv(cat(cam, sat)) + cam_bev`
   - cam 분포 구조적 보존, fusion이 additive delta만 학습
3. **sat backbone full lr (1.0x)**: v7/satonly와 동일. ImageNet pretrained → driving 도메인 adaptation에 충분한 lr 필요

**LR 설정**:
- img_backbone: 0.1x (detectron2 pretrained, BEVFormer 원본 설정 유지)
- sat_encoder.backbone: 1.0x (ImageNet pretrained, adaptation 필요)
- fusion, decoder, 기타: 1.0x

---

## 4. 성능 차이 분석

### 4.1 Sumin ConvFuser vs v1 (동일 인코더)

인코더가 동일(ResNet50+FPN)하므로 차이는 순수 구조적:

1. **Fusion 시점**: Sumin은 Stage 1부터, v1은 Stage 2부터
   - v1은 Stage 1에서 cam/sat 독립 학습 → Stage 2에서 뒤늦게 합류 → 이미 다른 방향으로 수렴한 feature space끼리 만남
2. **Fusion 방식**: ConvFuser(concat+conv) vs Decoder cross-attention
   - Satellite은 이미 BEV-aligned → BEV 공간에서 직접 합치는 게 자연스러움
   - Cross-attention은 query 레벨에서 간접 참조 → 불필요하게 복잡

### 4.2 Sumin ConvFuser vs v5

1. **Fusion 방식**: ConvFuser(learnable) vs Addition(파라미터 없음)
   - Addition은 두 feature space의 scale/distribution이 완벽히 맞아야 효과적
   - ConvFuser는 projection + conv로 이를 학습
2. **인코더 품질**: ResNet50+FPN vs SimpleSatEncoder (별개 문제)
3. **History buffer**: Sumin은 cam only, v5는 fused_bev → 아래 4.3에서 상세 분석

### 4.3 History Buffer에 fused_bev를 넣는 문제 (v5의 구조적 약점)

MapTracker의 temporal 구조:
```
backbone(camera_images, history_bev_feats) → _bev_feats
```

BEVFormer backbone이 `history_bev_feats`를 temporal self-attention의 reference로 사용. 이 구조는 **camera feature끼리의 temporal consistency**를 전제로 설계됨.

**v5의 문제**: `history_bev_feats`에 `cam+sat` fused feature가 들어감
- Backbone은 camera 이미지만 보고 feature를 생성
- Reference로 받는 history는 cam+sat 혼합 feature
- Backbone 입장에서 자기가 만든 적 없는 distribution을 reference로 받는 셈
- Stage 2/3에서 tracking 들어오면 이 mismatch가 vector head에서 문제됨

**Sumin이 cam only history를 유지한 이유**: camera pipeline을 건드리지 않으니 MapTracker의 temporal 구조가 원래대로 동작. Fusion은 최종 prediction에서만 적용.

---

## 5. Sumin ConvFuser의 Seg Decoder 공유 구조

### 이중 Supervision 메커니즘

```python
# ① 부모 MapTracker forward → cam_bev에 대한 seg loss
loss, log_vars, num_sample = super().forward_train(...)

# ② cam_bev 꺼내서 ConvFuser로 fusion → 같은 seg_decoder에 fused_bev 넣어서 추가 seg loss
cam_bev = self._last_bev_feats
sat_feats = self.sat_encoder(sat_img)
fused_bev, fusion_info = self.sat_fusion(cam_bev, sat_feats)
seg_preds_fused, _, seg_loss_fused, seg_dice_fused = \
    self.seg_decoder(fused_bev, gt_semantic, None, return_loss=True)

loss = loss + seg_loss_fused + seg_dice_fused  # ① + ②
```

- **같은 `seg_decoder`를 두 번 호출**: cam_bev 한 번, fused_bev 한 번
- seg_decoder는 두 입력 모두에서 잘 작동하도록 학습됨
- 최종 평가(0.514)는 fused_bev 결과 → 공정한 비교
- ①이 있어서 seg_decoder가 cam_bev에서도 잘 동작하는 상태 유지 → Phase 1b distillation에 유리

---

## 6. Sumin에 구현된 Fusion 방식 3가지

파일: `/workspace/sumin/.../plugin/models/necks/satellite_fusion.py`

### 6.1 SatelliteConvFuser (concat + conv)
```
Conv1x1(cam_bev) ─┐
                    ├─ concat [B,512,H,W] → Conv3x3→BN→ReLU→Conv3x3→BN→ReLU → fused
Conv1x1(sat_bev) ─┘
```
- 대칭 fusion: 두 modality를 동등하게 취급
- BEVFusion/SatMap 스타일
- **0.514 mIoU 달성**

### 6.2 SatelliteAdaptiveFusion (spatial gating)
```
quality_map = sigmoid(gate(cam_bev))     # [B,1,H,W]
fused = quality_map * cam_bev + (1 - quality_map) * sat_bev
```
- 픽셀별 cam/sat 비율 학습
- 해석 가능 (quality_map 시각화)
- 한계: cam_bev에서만 gate 생성 → sat_bev 품질 정보 미반영

### 6.3 SatelliteConditionedModulation (FiLM/SPADE)
```
gamma = gamma_conv(shared(sat_bev))      # [B,C,H,W], init ≈ 1
beta = beta_conv(shared(sat_bev))        # [B,C,H,W], init ≈ 0
fused = gamma * BN(cam_bev) + beta + cam_bev
```
- 비대칭: satellite이 camera를 조건부 보정
- 초반 cam_bev 그대로 통과 → 점진적으로 satellite 영향 증가
- 비대칭 구조라 triple supervision과는 궁합이 안 맞음 (fused ≈ cam이므로 ①②가 너무 유사)

---

## 7. 설계 방향 (v9 기준)

### 핵심 설계 원칙

1. **분리 decoder**: cam/sat/fused 각각 전용 seg_decoder → gradient 간섭 제거, 진단 정확도 향상
2. **ConvFuser + residual**: fusion output = Conv(cat(cam, sat)) + cam_bev → cam 분포 보존
3. **sat backbone full lr**: ImageNet pretrained → driving 도메인 adaptation에 1.0x 필요
4. **History buffer는 fused_bev**: per-frame aligned satellite 정보가 temporal에 반영

### 분리 decoder의 이점

- cam_decoder gradient → cam backbone만, sat_decoder gradient → sat backbone만
- fused_decoder → fusion conv + 양쪽 backbone으로 gradient 전달
- cam mIoU 하락 시 원인 특정 가능 (feature 품질 vs decoder 편향)
- MapSegHead가 conv 3개 수준으로 가벼워 파라미터 부담 거의 없음

### 향후 확장: Feature Selection 모듈

분리 decoder가 있으면 cam/sat/fused 각각의 prediction confidence를 독립적으로 측정 가능.
이를 기반으로 step마다 best feature를 선택하여 memory에 저장하는 구조로 확장 가능:
- 각 decoder의 sigmoid confidence (max or entropy) 기반
- 또는 별도 learned quality estimator

### 미결 사항

- **history_mode='cam' ablation**: fused_bev를 history에 넣는 것의 temporal mismatch 문제 재검증
- **aux loss weight**: cam/sat aux loss가 fused와 동일 weight(1:1:1)인데, 조절 필요할 수 있음
- **Stage 2/3 ���환**: 분리 decoder 구조가 vector head와 어떻게 연동되는지 설계 필요
