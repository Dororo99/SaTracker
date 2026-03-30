# SatMapTracker Architecture

AID4AD 위성 이미지를 활용한 MapTracker satellite-camera BEV fusion.

**핵심 전략**: Stage 1에서 camera/satellite 각각 독립적으로 BEV feature를 학습시키고,
Stage 3에서 fusion하여 joint finetuning.

---

## 1. 전체 아키텍처 개요

### 1.1 Stage 1 — Independent BEV Pretraining

Camera와 satellite 각각 자기 seg head를 가지고 **독립적으로** 학습.
Fusion 없음. 각 encoder가 스스로 유의미한 BEV feature를 만들도록 강제.

```
                    ┌─────────────────────────────────────────────────┐
                    │              Stage 1: Independent BEV Pretrain  │
                    │                                                 │
  Camera Images     │  ┌───────────────────────────┐                  │
  [B,6,3,480,800]  ─┼─→│  BEVFormerBackbone         │                  │
                    │  │  ResNet50(Caffe) + FPN     │                  │
                    │  │  + BEVFormer Encoder       │                  │
                    │  │  (TemporalSA + SpatialCA)  │                  │
                    │  └─────────┬─────────────────┘                  │
                    │            ↓                                    │
                    │      cam_bev [B, 256, 50, 100]                  │
                    │            │                                    │
                    │            ↓                                    │
                    │  ┌──────────────────┐                           │
                    │  │  cam_seg_head     │ ← MapSegHead             │
                    │  │  Conv + 4x Up     │   (cam 전용)             │
                    │  │  → [B,3,200,100]  │                          │
                    │  └──────┬───────────┘                           │
                    │         ↓                                       │
                    │    cam_seg_loss  ─────────┐                     │
                    │                           │                     │
                    │                      total_loss                 │
                    │                           │                     │
                    │    sat_seg_loss  ─────────┘                     │
                    │         ↑                                       │
                    │  ┌──────┴───────────┐                           │
                    │  │  sat_seg_head     │ ← MapSegHead             │
                    │  │  Conv + 4x Up     │   (satellite 전용)       │
                    │  │  → [B,3,200,100]  │                          │
                    │  └──────────────────┘                           │
                    │            ↑                                    │
                    │      sat_bev [B, 256, 50, 100]                  │
                    │            │                                    │
                    │  ┌─────────┴─────────────────┐                  │
  AID4AD Satellite  │  │  SatelliteEncoder          │                  │
  [B,3,200,400]    ─┼─→│  ResNet50(PyTorch) + FPN   │                  │
                    │  │  Multi-scale → interpolate │                  │
                    │  └───────────────────────────┘                  │
                    │                                                 │
                    └─────────────────────────────────────────────────┘

  학습 대상: BEVFormerBackbone 전체, cam_seg_head, SatelliteEncoder 전체, sat_seg_head
  Freeze:    없음
  Loss:      cam_seg_loss + cam_dice + sat_seg_loss + sat_dice
  Epochs:    18 (2 GPU, batch 3)
```

### 1.2 Stage 2 — Tracking Warmup (Camera Only)

기존 MapTracker Stage 2와 동일. Satellite encoder는 idle.

```
                    ┌─────────────────────────────────────────────────┐
                    │              Stage 2: Vector Head Warmup        │
                    │                                                 │
  Camera Images    ─┼─→  BEVFormerBackbone (FROZEN)                   │
                    │            ↓                                    │
                    │      cam_bev [B, 256, 50, 100]                  │
                    │            │                                    │
                    │            ├──────────────────────┐             │
                    │            ↓                      ↓             │
                    │     cam_seg_head          MapDetectorHead       │
                    │     (seg loss)            (vector + tracking)   │
                    │                                                 │
                    │  SatelliteEncoder (FROZEN, idle)                 │
                    │                                                 │
                    └─────────────────────────────────────────────────┘

  학습 대상: MapDetectorHead, MotionMLP, seg_decoder
  Freeze:    BEVFormerBackbone, SatelliteEncoder
  Loss:      seg + cls(focal) + reg(L1) + transition
  Epochs:    4 (4 GPU, batch 8)
  Load from: Stage 1 checkpoint
```

### 1.3 Stage 3 — Joint Finetuning with Fusion

Fusion 모듈 도입. Camera/Satellite BEV를 각각 projection 후 concat+conv로 fusion.
Stage 1에서 독립 head로 학습된 두 feature space가 다를 수 있으므로,
1x1 projection으로 common space에 align 후 fusion하는 ConvFusion을 사용.

```
                    ┌──────────────────────────────────────────────────────────────┐
                    │              Stage 3: Joint Finetuning with Fusion           │
                    │                                                              │
  Camera Images    ─┼─→  BEVFormerBackbone (lr×0.1~0.5)                            │
                    │            ↓                                                │
                    │      cam_bev [B, 256, 50, 100]                              │
                    │            │                                                │
                    │            ↓                                                │
                    │  ┌─────────────────────────────────────────────┐            │
                    │  │           _FusionNeckWrapper                 │            │
                    │  │                                             │            │
                    │  │   cam_bev → proj(1×1) ──┐                   │            │
                    │  │                          ├── cat+conv       │            │
                    │  │   sat_bev → proj(1×1) ──┘   → fused_bev   │            │
                    │  │           (ConvFusion)                      │            │
                    │  └──────────────────┬──────────────────────────┘            │
                    │                     │                                       │
                    │               fused_bev [B, 256, 50, 100]                   │
                    │                     │                                       │
                    │                     ├──────────────────────┐                │
                    │                     ↓                      ↓                │
                    │              cam_seg_head          MapDetectorHead          │
                    │              (seg loss)            (vector + tracking)      │
                    │                                   + MotionMLP              │
                    │                                   + VectorInstanceMemory   │
                    │                     ↑                                       │
                    │               sat_bev                                       │
                    │                     │                                       │
  AID4AD Satellite ─┼─→  SatelliteEncoder (lr×0.1)                                │
                    │                                                              │
                    └──────────────────────────────────────────────────────────────┘

  학습 대상: 전부 (differentiated LR)
  LR 전략:  Camera backbone 0.1x, BEV transformer 0.5x,
            SatelliteEncoder 0.1x, Fusion module 1.0x (full),
            Vector head 1.0x, seg_decoder 0.5x
  Loss:     seg + cls + reg + transition
  Epochs:   36 (4 GPU, batch 4)
  Load from: Stage 2 checkpoint
```

---

## 2. 모듈 상세

### 2.1 SatelliteEncoder

AID4AD 위성 이미지를 BEV-aligned feature map으로 인코딩.

```
AID4AD Satellite Image [B, 3, 200, 400]     ← 원본 해상도 (0.15 m/px)
       │
       ↓
┌──────────────────────────────────────┐
│  ResNet50 (ImageNet pretrained)       │
│  4 stages, out_indices=(0,1,2,3)     │
│                                      │
│  Stage 0: [B, 256, 50, 100]         │
│  Stage 1: [B, 512, 25, 50]          │
│  Stage 2: [B, 1024, 13, 25]         │
│  Stage 3: [B, 2048, 7, 13]          │
└───┬──────┬──────┬──────┬─────────────┘
    │      │      │      │
    ↓      ↓      ↓      ↓
┌──────────────────────────────────────┐
│  FPN-like Multi-Scale Aggregation    │
│                                      │
│  lateral_conv(256→256)  ─────────┐   │
│  lateral_conv(512→256)  → upsamp ┤   │
│  lateral_conv(1024→256) → upsamp ┤ + │  (element-wise sum)
│  lateral_conv(2048→256) → upsamp ┘   │
│                                      │
│  → output_conv (3×3 + BN + ReLU)    │
└──────────────┬───────────────────────┘
               ↓
        [B, 256, 50, 100]              ← bev_h × bev_w 로 interpolate
               │
          sat_bev features
```

**원본 해상도(400×200) 사용 이유**:
- AID4AD 해상도 0.15 m/px → lane divider (~10cm) 도 최소 1px로 표현
- 200×100으로 downsample하면 thin structure 정보 손실
- ResNet50 첫 stage에서 stride=4 → 100×50 feature map이므로 fine detail 보존
- FPN이 multi-scale feature를 aggregation → 최종 50×100으로 interpolate

### 2.2 Fusion Modules (Stage 3)

Config에서 선택 가능한 4가지 fusion 방식. 기본값: **SatCamConvFusion**.

> **SatCamModulation(FiLM) → ConvFusion 전환 근거 (2026-03-29)**:
> Stage 1에서 camera/satellite을 독립 head로 학습했기 때문에 두 BEV feature space가
> 서로 다르게 형성됨. FiLM은 satellite feature로 camera feature를 직접 scale/shift하므로
> feature space가 aligned되어 있다고 가정하지만, 실제로는 gamma가 음수로 발산하며
> total loss가 40→71로 증가. ConvFusion은 각 modality를 1x1 projection으로 common space에
> 먼저 매핑한 후 concat+conv하므로 feature space 불일치에 robust함.

#### (A) SatCamDeformAttnFusion (Class-Aware Dual-Path, 기본 선택)

Stage 1 독립 평가 결과에서 드러난 complementary pattern을 활용하는 fusion:

```
Stage 1 Evaluation (third val, single-frame):
              ped_crossing   divider   boundary   mIoU
  Camera        0.4423       0.3586    0.4108    0.4039
  Satellite     0.5283       0.3112    0.3486    0.3960
→ ped_crossing은 satellite 우위 (+8.6%p), divider/boundary는 camera 우위
→ 각 class별 modality 신뢰도가 다름 → class-aware gating 필요
```

```
cam_bev [B, 256, 50, 100]    sat_bev [B, 256, 50, 100]
       │                            │
       │    ┌────────────────────────┘
       │    │
       ↓    ↓
  ┌─────────────────────────────────────────────────────────────┐
  │              Dual-Path Deformable Cross-Attention            │
  │                                                              │
  │  Path 1 (cam→sat):                                          │
  │    cam_bev [Q] × sat_bev [K,V]                              │
  │    DeformableAttn(num_heads=8, num_points=4)                │
  │    ref_point = identity grid + learned offsets               │
  │    → cam_attended = cam_bev + Attn(cam, sat)                │
  │    → LayerNorm                                               │
  │                                                              │
  │  Path 2 (sat→cam):                                          │
  │    sat_bev [Q] × cam_bev [K,V]                              │
  │    DeformableAttn(num_heads=8, num_points=4)                │
  │    → sat_attended = sat_bev + Attn(sat, cam)                │
  │    → LayerNorm                                               │
  │                                                              │
  └─────────┬──────────────────────────────────┬────────────────┘
            │                                  │
       cam_attended                       sat_attended
            │                                  │
            └──────────┬───────────────────────┘
                       │
                       ↓
  ┌─────────────────────────────────────────────────────────────┐
  │              Class-Aware Spatial Gate                        │
  │                                                              │
  │  cat(cam_attended, sat_attended) [B, 512, 50, 100]          │
  │       ↓                                                      │
  │  Conv(512→128, 3×3) + BN + ReLU                             │
  │       ↓                                                      │
  │  Conv(128→3, 1×1) + sigmoid                                 │
  │       ↓                                                      │
  │  class_gates [B, 3, H, W]  ← per-class spatial gate         │
  │    gate[0] = ped_crossing  (학습으로 sat 쪽으로 이동 기대)    │
  │    gate[1] = divider       (학습으로 cam 쪽 유지 기대)       │
  │    gate[2] = boundary      (학습으로 cam 쪽 유지 기대)       │
  │       ↓                                                      │
  │  Conv(3→256, 1×1) → channel_gate [B, 256, H, W]            │
  │       ↓                                                      │
  │  fused = channel_gate * cam_attended                         │
  │        + (1 - channel_gate) * sat_attended                   │
  │                                                              │
  └──────────────────────┬──────────────────────────────────────┘
                         │
                         ↓
  ┌─────────────────────────────────────────────────────────────┐
  │  FFN (256→512→256) + LayerNorm                              │
  │       ↓                                                      │
  │  + cam_bev   ← final residual (camera passthrough 보장)     │
  └──────────────────────┬──────────────────────────────────────┘
                         ↓
                  fused_bev [B, 256, 50, 100]
```

**설계 근거**:
- **Deformable attention**: cam_bev/sat_bev가 같은 ego BEV 좌표에 정렬 → ref point = identity grid, K=4 offset만 학습. Full MHA(25M) 대비 1250배 효율적(20K)
- **Dual-path**: cam이 sat에서 (ped_crossing 등), sat이 cam에서 (divider 등) 각각 필요한 정보를 선택적으로 가져옴
- **Class-aware gate**: per-class gate가 각 클래스에 최적인 modality 비중을 pixel별로 학습
- **Camera residual**: 최종 출력에 cam_bev를 더해서 downstream (Stage 2에서 학습된 vector head)과의 호환성 보장
- **Params**: 1.17M (전체 92.1M의 1.3%)

**Wandb 모니터링**:
```
fusion/gate_cam_mean              ← 전체 평균 camera 비중
fusion/gate_sat_mean              ← 전체 평균 satellite 비중
fusion/gate_ped_cam, gate_ped_sat ← ped_crossing class의 cam/sat 비중
fusion/gate_div_cam, gate_div_sat ← divider class의 cam/sat 비중
fusion/gate_bound_cam, gate_bound_sat ← boundary class의 cam/sat 비중
```

#### (B) SatCamModulation (FiLM/SPADE-style)

```
sat_bev [B, 256, 50, 100]
       │
       ↓
  shared_conv (3×3, 256→128, BN, ReLU)
       │
       ├─→ gamma_conv (3×3, 128→256)  → gamma  [B, 256, 50, 100]
       │                                 init: weight=0, bias=1 (≈ passthrough)
       │
       └─→ beta_conv  (3×3, 128→256)  → beta   [B, 256, 50, 100]
                                         init: weight=0, bias=0

cam_bev [B, 256, 50, 100]
       │
       ↓
  BatchNorm2d(cam_bev) = cam_normed
       │
       ↓
  modulated = gamma * cam_normed + beta
       │
       ↓
  fused = modulated + cam_bev          ← residual connection
       │
       ↓
  fused_bev [B, 256, 50, 100]
```

#### (C) SatCamGatedFusion

```
cam_bev ──┐
           ├── cat [B, 512, 50, 100]
sat_bev ──┘
           ↓
  Conv(512→64, 3×3) + BN + ReLU → Conv(64→1, 1×1) → sigmoid
           ↓
  gate [B, 1, 50, 100]    (bias init = 1.0 → camera 선호)
           ↓
  fused = gate * cam_bev + (1-gate) * sat_bev
```

#### (D) SatCamConvFusion (BEVFusion-style)

```
cam_bev → proj_cam (1×1, 256→256) ──┐
                                      ├── cat [B, 512, 50, 100]
sat_bev → proj_sat (1×1, 256→256) ──┘
                                      ↓
                           Conv(512→256, 3×3, BN, ReLU)
                           Conv(256→256, 3×3, BN, ReLU)
                                      ↓
                               fused_bev [B, 256, 50, 100]
```

### 2.3 _FusionNeckWrapper (투명한 Fusion 주입)

MapTracker의 `self.neck()` 호출을 가로채어 satellite fusion을 주입.
기존 MapTracker 코드를 수정하지 않고 fusion을 모든 프레임에 적용.

```
                 기존 MapTracker 내부
                 ───────────────────
  _bev_feats = backbone(img, ...)
       │
       ↓
  bev_feats = self.neck(_bev_feats)    ← 이 호출을 가로챔
       │
       ↓
  seg_decoder / vector_head

                 _FusionNeckWrapper 내부
                 ──────────────────────
  self.neck.forward(x):
       │
       ↓
  out = original_neck(x)               ← Identity 또는 기존 neck
       │
       ↓
  if sat_feats is not None:
       out, info = fusion_module(out, sat_feats)
       │
       ↓
  return out                            ← fused BEV features
```

**작동 방식 (Stage 3 forward_train)**:
1. `sat_feats = sat_encoder(sat_img)` — satellite BEV 한 번 인코딩
2. `self.neck.sat_feats = sat_feats` — wrapper에 satellite feature 세팅
3. `super().forward_train(...)` — MapTracker 원본 forward 실행
   - 내부에서 매 프레임마다 `self.neck(_bev_feats)` 호출 시 fusion 자동 적용
4. `self.neck.sat_feats = None` — 정리

---

## 3. Data Pipeline

### 3.1 AID4AD Satellite 이미지

```
AID4AD Dataset (0.15 m/px, ego-aligned, per-frame pre-cropped)
    │
    ├── boston-seaport/
    ├── singapore-onenorth/
    ├── singapore-hollandvillage/
    └── singapore-queenstown/
         └── {counter}_{token}.png     ← 400×200 RGB

LoadAID4ADSatelliteImage Pipeline:
    1. Token으로 파일 lookup (초기화 시 전체 인덱싱)
    2. cv2.imread → RGB 변환
    3. np.flipud (BEV convention 맞춤)
    4. canvas_size=None → 원본 400×200 유지
    5. ImageNet normalize: (img - mean) / std
    6. HWC → CHW → results['sat_img']
```

### 3.2 Train Pipeline (Stage 1)

```python
train_pipeline = [
    VectorizeMap(...)           # GT polyline 추출
    RasterizeMap(...)           # GT → semantic mask [3, 200, 100]
    LoadMultiViewImagesFromFiles(to_float32=True)
    LoadAID4ADSatelliteImage(data_root=..., canvas_size=None)   # ★ 원본 400×200
    PhotoMetricDistortionMultiViewImage()
    ResizeMultiViewImages(size=(480, 800))
    Normalize3D(mean=..., std=...)
    PadMultiViewImages(size_divisor=32)
    FormatBundleMap()           # sat_img → tensor
    Collect3D(keys=['img', 'vectors', 'semantic_mask', 'sat_img'])
]
```

### 3.3 Test Pipeline

```python
# Stage 1 & 2: camera only (satellite 없음)
test_pipeline = [LoadMultiViewImages, Resize, Normalize, Pad, Format, Collect(['img'])]

# Stage 3: satellite 포함 (fusion 활성)
test_pipeline = [LoadMultiViewImages, LoadAID4AD, Resize, Normalize, Pad, Format,
                 Collect(['img', 'sat_img'])]
```

---

## 4. 학습 설정 상세

### 4.1 Stage별 Config 비교

```
                    Stage 1              Stage 2              Stage 3
                    ─────────────       ─────────────       ─────────────
Model Type          SatMapTracker        SatMapTracker        SatMapTracker
skip_vector_head    True                 False                False
freeze_bev          False                True                 False
use_sat_fusion      False                False                True ★
freeze_sat_encoder  False                True                 False
sat_seg_cfg         MapSegHead (있음)    없음                  없음
sat_fusion_cfg      없음                  없음                  SatCamConvFusion ★

GPUs                2                    4                    4
Batch/GPU           3                    8                    4
Epochs              18                   4                    36
Base LR             5e-4                 5e-4                 5e-4
LR Schedule         CosineAnnealing      CosineAnnealing      CosineAnnealing
Satellite Input     원본 400×200         없음                  원본 400×200
Eval Metric         mIoU (seg only)      mIoU + mAP           mIoU + mAP
Load From           -                    Stage 1              Stage 2
```

### 4.2 Stage 3 Differentiated Learning Rates

```
Component               │ lr_mult │ Base LR=5e-4 → Effective LR
────────────────────────┼─────────┼──────────────────────────────
backbone.img_backbone    │  0.1    │  5e-5    (이미 수렴)
backbone.img_neck        │  0.5    │  2.5e-4  (미세 조정)
backbone.transformer     │  0.5    │  2.5e-4  (미세 조정)
backbone.positional_enc  │  0.5    │  2.5e-4
sat_encoder              │  0.1    │  5e-5    (이미 수렴, Stage 1에서)
seg_decoder              │  0.5    │  2.5e-4
neck.fusion_module ★     │  1.0    │  5e-4    (새 모듈, full LR)
head (vector+tracking)   │  1.0    │  5e-4
query_propagate          │  1.0    │  5e-4
```

### 4.3 Loss 구성

**Stage 1:**
```
L_total = L_cam_seg + L_cam_dice + L_sat_seg + L_sat_dice

  L_cam_seg:  MaskFocalLoss(weight=10.0)   on cam_bev
  L_cam_dice: MaskDiceLoss(weight=1.0)     on cam_bev
  L_sat_seg:  MaskFocalLoss(weight=10.0)   on sat_bev
  L_sat_dice: MaskDiceLoss(weight=1.0)     on sat_bev
```

**Stage 2:**
```
L_total = L_seg + L_dice + L_cls + L_reg + L_trans

  L_seg:   MaskFocalLoss(10.0)  on cam_bev
  L_dice:  MaskDiceLoss(1.0)
  L_cls:   FocalLoss(5.0)
  L_reg:   LinesL1Loss(50.0)
  L_trans:  TransitionLoss(0.1)  (forward + backward)
```

**Stage 3:**
```
L_total = L_seg + L_dice + L_cls + L_reg + L_trans

  (Stage 2와 동일하지만, fused_bev 사용)
  L_seg/L_dice는 fused_bev에서 계산
```

---

## 5. 로깅 (Stage 1)

### 5.1 Scalar Metrics (매 50 iter)

```
# Loss
sat_seg, sat_seg_dice             ← satellite seg loss
seg, seg_dice                     ← camera seg loss (from parent)
total                             ← 전체 합

# BEV Feature Statistics
cam_bev_norm, sat_bev_norm        ← 각 feature의 L2 norm
cam_sat_cosine_sim                ← cam/sat feature 유사도
cam_bev_far_norm, cam_bev_near_norm   ← spatial: 원거리 vs 근거리
sat_bev_far_norm, sat_bev_near_norm
```

### 5.2 Wandb Images (매 500 iter)

```
bev/cam_bev               ← Camera BEV feature norm map
bev/sat_bev               ← Satellite BEV feature norm map
bev/cam_sat_diff           ← Camera vs Satellite L2 차이맵
input/sat_image            ← AID4AD 원본 이미지 (denormalized)
input/seg_gt               ← Seg GT (R=ped, G=div, B=bound)
pred/seg_cam               ← Camera seg prediction
pred/seg_sat               ← Satellite seg prediction
pred/overlay_sat_seg_on_sat ← Satellite seg pred를 위성 이미지에 overlay
```

### 5.3 Eval Metrics (매 3 epoch)

```
boundary, divider, ped_crossing, mIoU    ← semantic segmentation IoU
```

---

## 6. 파일 구조

```
plugin/
├── configs/
│   └── satmaptracker/
│       └── nuscenes_newsplit/
│           ├── satmaptracker_stage1_bev_pretrain.py
│           ├── satmaptracker_stage2_warmup.py
│           ├── satmaptracker_stage3_joint_finetune.py
│           └── satmaptracker_stage3_deform_fusion_third.py  ← DeformAttn fusion
│
├── datasets/
│   └── pipelines/
│       ├── loading_sat.py              ← LoadAID4ADSatelliteImage
│       └── formating.py                ← sat_img tensor 변환 추가
│
├── models/
│   ├── mapers/
│   │   ├── MapTracker.py               ← _last_bev_feats 1줄 추가
│   │   └── SatMapTracker.py            ← 메인 모델 (SD map 비활성화)
│   │
│   ├── backbones/
│   │   └── satellite_encoder.py        ← ResNet50 + FPN → BEV
│   │
│   └── necks/
│       ├── satellite_fusion.py         ← SatCamModulation / Gated / ConvFusion
│       └── deformable_fusion.py        ← SatCamDeformAttnFusion (Class-Aware Dual-Path)
│
└── docs/
    └── satmap/
        └── architecture.md             ← 이 문서
```

---

## 7. 설계 근거: 왜 독립 Pretraining 후 Fusion인가?

### 7.1 수민 실험의 문제점 (Stage 1에서 바로 fusion)

수민은 3가지 fusion 방식 (Gating, Modulation, ConvFuser)을 Stage 1에서 바로 적용.
결과: baseline 대비 mIoU +0.02 수준 (미미).

**근본 원인: Chicken-and-Egg Problem**

```
Satellite encoder 초기화 (random)
        ↓
  Satellite feature 품질 낮음 (norm ≈ 8~9, camera ≈ 16~20)
        ↓
  Fusion module이 satellite 무시하도록 학습
  (gating: qmap → 0.7+, modulation: gamma → residual 지배)
        ↓
  Satellite encoder에 gradient 부족
        ↓
  영원히 개선 불가 → 성능 정체
```

실제 로그 분석:
- Gating: `qmap_mean = 0.79 → 0.70` (항상 camera 편향)
- Modulation: `gamma_mean = -0.35 → -0.21` (음수! residual이 지배)
- ConvFuser: `cam_fused_cosine_sim = 0.25` (가장 큰 변환이지만 +0.02만)
- 공통: Epoch 6 이후 성능 수렴 (cam backbone이 이미 포화)

### 7.2 독립 Pretraining의 해결책

```
Stage 1 (독립):
  Camera encoder ← seg GT 직접 supervision     → cam_bev 품질 우수
  Satellite encoder ← seg GT 직접 supervision   → sat_bev 품질 우수 ★
        ↓
Stage 3 (fusion):
  이미 좋은 cam_bev + 이미 좋은 sat_bev → Fusion module만 학습
  → Chicken-and-egg 문제 없음
```

**장점**:
1. Satellite encoder가 **직접** seg GT를 보고 학습 → "위성 이미지에서 도로 읽는 법" 확실히 학습
2. Stage 1에서 satellite-only mIoU를 **독립 측정** 가능 (fusion 의미 검증)
3. Fusion module이 **이미 의미있는 두 feature를 결합**하는 것만 학습
4. Differentiated LR로 encoder 고착화 방지 (0.1x로 미세 조정)

### 7.3 Feature 고착화 우려에 대해

"독립적으로 학습된 feature를 fusion하면 feature space 불일치로 손해?"

**fusion module 선택이 중요하다**:
- Stage 3에서 encoder lr=0.1x (0이 아님) → feature space가 fusion에 맞게 **적응**
- Fusion module lr=1.0x → 두 feature space를 **bridging**하는 것이 주 역할
- ~~Modulation(FiLM)은 feature space가 aligned되어 있다고 가정 → 실제로 발산~~ (실험 확인)
- **ConvFusion**: 1x1 proj로 각 feature를 common space에 매핑 → space 불일치에 robust
- Concat+conv 구조가 두 feature 간 관계를 자유롭게 학습 가능

---

## 8. AID4AD Dataset 특성

```
                   AID4AD
해상도              0.15 m/px
정합 정확도          ~0.16m mean error (≈ 1px)
Coverage           NuScenes 전체 (34,149 tiles)
도시               Boston-Seaport, Singapore (3 regions)
크기               400×200 px per tile (60m × 30m)
촬영 시점 차이      수개월 (NuScenes 촬영 대비)
Coordinate         Ego-centric, pre-cropped, pre-aligned
```

**관찰 (notebook 시각화 결과)**:
- 정합 품질 우수: GT polyline이 위성 이미지의 도로 구조와 잘 일치
- Lane divider, road boundary, crosswalk이 사람 눈으로도 식별 가능
- Satellite-only segmentation이 충분히 학습 가능할 것으로 기대

---

## 9. Stage 1 독립 평가 결과

### 9.1 Dataset-level mIoU (Third Val Set, Single-Frame)

Camera seg head와 satellite seg head를 각각 독립적으로 평가.
평가 방식은 학습 시 `RasterEvaluate`와 동일 (전체 dataset intersection/union 합산, threshold=0.4).

```
+-----------+--------------+---------+----------+--------+
|           | ped_crossing | divider | boundary |  mIoU  |
+-----------+--------------+---------+----------+--------+
|  Camera   |    0.4423    |  0.3586 |  0.4108  | 0.4039 |
| Satellite |    0.5283    |  0.3112 |  0.3486  | 0.3960 |
+-----------+--------------+---------+----------+--------+

참고 (학습 로그, cam multi-frame): mIoU = 0.4571
→ Single-frame vs multi-frame 차이 (~5%p)는 temporal history에 의한 것.
  Satellite은 history를 사용하지 않으므로 single/multi 결과 동일.
```

### 9.2 Near vs Far Region 분석

BEV 상반부 (far, ego에서 먼 곳)와 하반부 (near) 각각의 IoU.

```
+------------+--------------+---------+----------+--------+
|            | ped_crossing | divider | boundary |  mIoU  |
+------------+--------------+---------+----------+--------+
| Cam (near) |    0.4685    |  0.3354 |  0.4437  | 0.4159 |
| Sat (near) |    0.5381    |  0.3022 |  0.3526  | 0.3976 |
| Cam (far)  |    0.4192    |  0.3778 |  0.3765  | 0.3912 |
| Sat (far)  |    0.5191    |  0.3188 |  0.3444  | 0.3941 |
+------------+--------------+---------+----------+--------+
```

**관찰**:
- Camera near→far 성능 저하: mIoU 0.4159 → 0.3912 (Δ -2.5%p)
- Satellite near→far 성능 저하: mIoU 0.3976 → 0.3941 (Δ -0.4%p)
- "Satellite이 장거리에서 보완" 가설은 미미. Near/far 차이가 예상보다 작음
- **핵심 차이점은 class별**: ped_crossing에서 satellite +8.6%p 우위 (near/far 모두)

### 9.3 Complementary Pattern 요약

```
                 Camera 우위              Satellite 우위
  ped_crossing                              ★★★ (+8.6%p)
  divider         ★★ (+4.7%p)
  boundary        ★★ (+6.2%p)
```

→ **Fusion 전략의 핵심**: class-aware gating으로 각 modality의 강점을 활용.
  단순 near/far 기반 weighting보다 semantic-level gating이 더 효과적.

---

## Appendix A. Uncertainty-Guided Adaptive Fusion (후보 기법)

### A.1 개요

각 modality가 pixel별로 자신의 불확실성(uncertainty)을 예측하고,
더 확신하는 modality를 더 신뢰하는 Bayesian precision-weighted fusion.

Occlusion, 역광, 센서 고장 등 camera가 불확실한 상황에서
자동으로 satellite 의존도를 높일 수 있다는 장점이 있음.

### A.2 수학적 원리

두 modality의 예측을 Gaussian으로 모델링:
```
Camera:    N(cam_bev, σ²_cam)
Satellite: N(sat_bev, σ²_sat)

Optimal fusion (Kalman filter):
  precision_cam = 1/σ²_cam = exp(-log_var_cam)
  precision_sat = 1/σ²_sat = exp(-log_var_sat)

  w_cam = precision_cam / (precision_cam + precision_sat)
  w_sat = precision_sat / (precision_cam + precision_sat)

  fused = w_cam * cam_bev + w_sat * sat_bev
```

### A.3 Architecture

```
cam_bev ─→ UncertaintyHead ─→ cam_log_var [B, 1, H, W]
           Conv(256→64, 3×3, BN, ReLU) + Conv(64→1, 1×1)
           init: all zeros → σ² = 1 → equal weight at start

sat_bev ─→ UncertaintyHead ─→ sat_log_var [B, 1, H, W]
           (same architecture)

Precision weighting:
  cam_prec = exp(-cam_log_var)
  sat_prec = exp(-sat_log_var)
  w_cam = cam_prec / (cam_prec + sat_prec)    [B, 1, H, W]
  w_sat = sat_prec / (cam_prec + sat_prec)

  fused_bev = w_cam * cam_bev + w_sat * sat_bev
```

파라미터: Conv(256→64→1) × 2 = ~66K params (전체 모델의 0.07%)

### A.4 Uncertainty Calibration Loss

Kendall et al. (CVPR 2018) homoscedastic uncertainty loss:
```
L_unc = exp(-log_var) * ||pred - GT||² + log_var

→ 틀린 곳 (||pred-GT||² 큼): log_var를 올려야 loss 감소 → 높은 uncertainty
→ 맞는 곳 (||pred-GT||² 작음): log_var를 내려야 loss 감소 → 낮은 uncertainty
→ 자동으로 calibrate됨
```

Stage 1의 frozen seg head를 `pred` 생성에 재활용:
```
cam_pred = frozen_cam_seg_head(cam_bev)     # no grad
sat_pred = frozen_sat_seg_head(sat_bev)     # no grad
L_cam_unc = mean(exp(-cam_log_var) * BCE(cam_pred, GT) + cam_log_var)
L_sat_unc = mean(exp(-sat_log_var) * BCE(sat_pred, GT) + sat_log_var)
L_total += 0.1 * (L_cam_unc + L_sat_unc)
```

### A.5 기대 동작

```
학습 초기:
  log_var ≈ 0 → w_cam ≈ 0.5, w_sat ≈ 0.5 (균등)

학습 수렴 후:
  Camera 잘 맞추는 pixel (divider 근거리):
    cam_err 작음 → cam_log_var ↓ → cam_prec ↑ → w_cam ↑

  Camera 못 맞추는 pixel (ped_crossing, occlusion):
    cam_err 큼 → cam_log_var ↑ → cam_prec ↓ → w_sat ↑
```

### A.6 DeformAttnFusion과의 관계

- DeformAttnFusion: class-aware gate로 **semantic level** adaptive fusion
- UncertaintyFusion: pixel-level uncertainty로 **confidence level** adaptive fusion
- 두 기법은 조합 가능: DeformAttn의 output에 uncertainty weighting을 추가 적용

### A.7 구현 시 고려사항

- log_var를 clamp (-10, 10) 해서 numerical stability 보장
- Uncertainty loss weight (0.1)은 main loss 대비 보조적으로 설정
- Wandb 모니터링: `w_cam_mean`, `w_sat_mean`, `cam_unc_mean`, `sat_unc_mean`
- Near/far, per-class 별 weight 분포 시각화로 fusion 동작 검증
