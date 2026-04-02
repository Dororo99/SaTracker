# SatOnlyMapTracker Architecture Document

## 1. Overview

SatOnlyMapTracker는 MapTracker의 카메라 파이프라인(BEVFormerBackbone)을 완전히 제거하고, AID4AD 위성 이미지만으로 BEV feature를 생성하는 모델이다. MapTracker의 tracking, memory, query propagation 로직은 그대로 유지된다.

### 핵심 변경점
- `BEVFormerBackbone` (ResNet50-DCN + FPN + BEVFormer Transformer) → `SatBEVProvider` (SatelliteEncoder wrapper)
- 카메라 6장 처리 제거 → 위성 이미지 1장만 처리
- BEV temporal fusion 무의미 (같은 위성 이미지 = 같은 BEV)
- Query-level temporal (tracking, memory bank)은 유지

---

## 2. Model Architecture

### 2.1 SatOnlyMapTracker (`plugin/models/mapers/SatOnlyMapTracker.py`)

```
SatOnlyMapTracker(MapTracker)
├── backbone: SatBEVProvider          ← BEVFormerBackbone 대체
│   └── sat_encoder: SatelliteEncoder
│       ├── backbone: ResNet50 (pretrained, ImageNet style)
│       ├── lateral_convs: 4× Conv2d(Ci, 256, 1×1)  [i=0..3]
│       └── output_conv: Conv2d(256, 256, 3×3) + BN + ReLU
├── neck: nn.Identity()
├── seg_decoder: MapSegHead
│   ├── Conv layers (256 → num_classes=3)
│   ├── loss_seg: MaskFocalLoss (weight=10.0)
│   └── loss_dice: MaskDiceLoss (weight=1.0)
├── head: MapDetectorHead
│   ├── 100 learnable queries (embed_dims=512)
│   ├── transformer: MapTransformer
│   │   └── decoder: MapTransformerDecoder_new (6 layers)
│   │       └── MapTransformerLayer ×6
│   │           ├── self_attn: MultiheadAttention(512, 8 heads)
│   │           ├── cross_attn: CustomMSDeformableAttention(512, 8 heads, 20 pts)
│   │           ├── cross_attn: MultiheadAttention(512, 8 heads)  ← memory cross-attn
│   │           └── ffn: FFN(512 → 1024 → 512)
│   ├── loss_cls: FocalLoss (weight=5.0)
│   ├── loss_reg: LinesL1Loss (weight=50.0)
│   └── assigner: HungarianLinesAssigner
├── query_propagate: MotionMLP (c_dim=7, f_dim=512)
└── memory_bank: VectorInstanceMemory (dim=512, 100 instances)
```

### 2.2 SatBEVProvider (`plugin/models/mapers/SatOnlyMapTracker.py`)

BEVFormerBackbone의 call signature를 그대로 따르는 wrapper. 실제로는 satellite encoder만 사용.

```python
class SatBEVProvider(nn.Module):
    def set_sat_img(self, sat_img):
        """위성 이미지 encode + H flip (image→BEV convention) → 캐싱"""
        feats = self.sat_encoder(sat_img)        # (B, 256, 50, 100)
        self._sat_bev = torch.flip(feats, [2,])  # y-down → y-up

    def forward(self, img, img_metas, timestep, history_bev, ...):
        """캐싱된 satellite BEV 반환. 카메라 입력 전부 무시."""
        return self._sat_bev, None
```

**중요**: `forward()`가 매 frame마다 호출되지만, 실제 연산은 `set_sat_img()`에서 1번만 수행. 나머지는 캐시 반환.

### 2.3 SatelliteEncoder (`plugin/models/backbones/satellite_encoder.py`)

```
Input: sat_img (B, 3, 400, 200) — AID4AD normalized
  ↓
ResNet50 backbone → 4 stage features
  stage0: (B, 256, 100, 50)
  stage1: (B, 512, 50, 25)
  stage2: (B, 1024, 25, 13)
  stage3: (B, 2048, 13, 7)
  ↓
FPN-like aggregation:
  lateral_conv[i](stage_i) → all upsampled to stage0 size → sum
  ↓
output_conv: Conv2d(256, 256, 3×3) + BN + ReLU
  ↓
Resize to (50, 100) via bilinear interpolation
  ↓
Output: (B, 256, 50, 100) — matches BEVFormer output shape
```

---

## 3. MapTracker vs SatOnlyMapTracker 비교

### 3.1 Component 비교

| Component | MapTracker | SatOnlyMapTracker | 상태 |
|-----------|-----------|-------------------|------|
| Camera ResNet50-DCN + FPN | 6 views × 480×800 | **없음** | 제거 |
| BEVFormer Transformer | 2 layers, TemporalSelfAttn + SpatialCrossAttn | **없음** | 제거 |
| SatelliteEncoder | 없음 | ResNet50 + FPN agg, 1 img × 400×200 | 신규 |
| BEV Feature Shape | (B, 256, 50, 100) | (B, 256, 50, 100) | **동일** |
| Neck | nn.Identity() | nn.Identity() | 동일 |
| Seg Decoder (MapSegHead) | 3-class seg, focal+dice loss | 3-class seg, focal+dice loss | 동일 |
| Vector Head (MapDetectorHead) | 6-layer decoder, 100 queries | 6-layer decoder, 100 queries | 동일 |
| Query Propagation (MotionMLP) | ego-pose 기반 query warping | ego-pose 기반 query warping | 동일 |
| Memory Bank | instance-level, bank_size=4 | instance-level, bank_size=4 | 동일 |
| BEV History Buffer | 매 frame 다른 BEV 저장 | **같은 BEV 반복 저장 (redundant)** | 무의미 |

### 3.2 Forward 흐름 비교

**MapTracker forward_train (multi_frame=5):**
```
for t in range(4):  # prev frames
    bev_t = BEVFormerBackbone(cam_img_t, history_bev)  ← 매번 다른 BEV
    neck(bev_t) → seg_loss_t
    head(bev_t, track_queries) → vector_loss_t, track_query_update
    history_bev.append(bev_t)

bev_curr = BEVFormerBackbone(cam_img_curr, history_bev)  ← 매번 다른 BEV
neck(bev_curr) → seg_loss
head(bev_curr, track_queries) → vector_loss
```

**SatOnlyMapTracker forward_train (multi_frame=5):**
```
backbone.set_sat_img(sat_img)  ← 1번만 encode

for t in range(4):  # prev frames
    bev_t = backbone(...)  ← 캐싱된 같은 BEV 반환 (연산 없음)
    neck(bev_t) → seg_loss_t
    head(bev_t, track_queries) → vector_loss_t, track_query_update
    history_bev.append(bev_t)  ← 같은 tensor 반복 저장

bev_curr = backbone(...)  ← 캐싱된 같은 BEV 반환
neck(bev_curr) → seg_loss
head(bev_curr, track_queries) → vector_loss
```

### 3.3 Temporal 분석

| 요소 | MapTracker | SatOnlyMapTracker | 의미 |
|------|-----------|-------------------|------|
| BEV History (TemporalSelfAttn) | 매 frame 다른 BEV → temporal fusion 유의미 | 매 frame 같은 BEV → self-attend to identical = no-op | **무의미** |
| Query Propagation | ego-pose로 query warp → 다음 frame에 전달 | 동일 | **유의미** — query가 detection state 전달 |
| Memory Bank | 이전 frame instance embedding 저장 | 동일 | **유의미** — tracking continuity |
| GT Matching | 두 frame 간 instance 매칭 | 동일 | **유의미** — track supervision |

---

## 4. Stage별 Training 설계

### 4.1 Stage 1: Segmentation Pretrain

**목적**: SatelliteEncoder + SegHead 학습. BEV feature가 map structure를 잘 표현하도록.

```
Config: satonlymaptracker_stage1_seg_pretrain_third.py

Model settings:
  skip_vector_head = True      # Head 비활성
  freeze_bev = False           # SatEncoder trainable
  use_memory = False           # Memory 비활성
  multi_frame = 1              # Single frame only

Training:
  batch_size = 16 per GPU × 4 GPU = 64 effective
  epochs = 18
  iters/epoch = 9274 // 64 = 144 (third dataset)
  total_iters = 18 × 144 = 2,592
  lr = 5e-4 (AdamW)
  warmup = linear, 125 iters, ratio=1/3
  lr_schedule = CosineAnnealing, min_lr_ratio=5e-2
  checkpoint = every 3 epochs (432 iters)

Loss:
  seg_loss = MaskFocalLoss(weight=10.0)
  seg_dice_loss = MaskDiceLoss(weight=1.0)
  total = seg_loss + seg_dice_loss

Forward flow:
  sat_img → SatelliteEncoder → flip H → BEV (256, 50, 100)
  BEV → Identity neck → MapSegHead → seg predictions
  seg predictions vs GT (flipped semantic_mask) → loss

Trainable parameters:
  - backbone.sat_encoder (ResNet50 + FPN agg + output_conv)
  - seg_decoder (MapSegHead)

Frozen parameters:
  - head (MapDetectorHead) — not used
  - query_propagate — not used
```

### 4.2 Stage 2: Vector Head Warmup

**목적**: Vector head 학습. SatEncoder는 freeze. Temporal/tracking 없이 single-frame detection.

```
Config: satonlymaptracker_stage2_warmup_third.py
Load from: work_dirs/satonlymaptracker_stage1_seg_pretrain_third/latest.pth

Model settings:
  skip_vector_head = False     # Head 활성
  freeze_bev = True            # SatEncoder + SegHead frozen
  use_memory = False           # Memory 비활성 (no tracking)
  multi_frame = 1              # Single frame (no temporal)

Training:
  batch_size = 16 per GPU × 4 GPU = 64 effective
  epochs = 4
  iters/epoch = 9274 // 64 = 144 (third dataset)
  total_iters = 4 × 144 = 576
  lr = 5e-4 (AdamW)
  warmup = linear, 125 iters, ratio=1/3
  lr_schedule = CosineAnnealing, min_lr_ratio=3e-3
  checkpoint = every 4 epochs (576 iters = end)

Loss (single frame):
  seg_loss = MaskFocalLoss(weight=10.0)          # SegHead frozen, still computed
  seg_dice_loss = MaskDiceLoss(weight=1.0)
  cls_loss = FocalLoss(weight=5.0)               # vector classification
  reg_loss = LinesL1Loss(weight=50.0)            # vector regression
  total = seg_loss + seg_dice + cls_loss + reg_loss

Forward flow:
  sat_img → SatelliteEncoder → flip H → BEV (256, 50, 100)
  BEV → neck → seg_loss
  BEV → head(queries) → vector_loss

Trainable parameters:
  - head (MapDetectorHead: 6-layer decoder, query embeddings)
  - query_propagate (MotionMLP) — exists but unused (multi_frame=1)

Frozen parameters:
  - backbone.sat_encoder (SatelliteEncoder)
  - seg_decoder (MapSegHead)
```

### 4.3 Stage 3: Joint Finetuning

**목적**: 전체 모델 joint training. Differentiated LR. Single-frame detection.

```
Config: satonlymaptracker_stage3_joint_finetune_third.py
Load from: work_dirs/satonlymaptracker_stage2_warmup_third/latest.pth

Model settings:
  skip_vector_head = False     # Head 활성
  freeze_bev = False           # 전부 trainable
  use_memory = False           # Memory 비활성 (no tracking)
  multi_frame = 1              # Single frame (no temporal)

Training:
  batch_size = 16 per GPU × 4 GPU = 64 effective
  epochs = 36
  iters/epoch = 9274 // 64 = 144 (third dataset)
  total_iters = 36 × 144 = 5,184
  lr = 5e-4 (AdamW)
  warmup = linear, 125 iters, ratio=1/3
  lr_schedule = CosineAnnealing, min_lr_ratio=3e-3
  checkpoint = every 6 epochs (864 iters)
  eval = every 6 epochs (864 iters)

Differentiated LR:
  backbone.sat_encoder: 0.1× = 5e-5   # already converged from Stage 1
  seg_decoder:          0.5× = 2.5e-4  # fine-tune
  head:                 1.0× = 5e-4    # full LR
  query_propagate:      1.0× = 5e-4    # full LR (unused but trainable)

Loss: Stage 2와 동일

Forward flow: Stage 2와 동일 (단, SatEncoder도 gradient 흐름)

Trainable parameters:
  - backbone.sat_encoder (0.1× LR)
  - seg_decoder (0.5× LR)
  - head (1.0× LR)
  - query_propagate (1.0× LR, unused)
```

---

## 5. Data Pipeline

### 5.1 Train Pipeline

```
VectorizeMap          → GT vectors normalized, 20 points, permuted
RasterizeMap          → GT semantic mask (200×100, 3-class)
LoadMultiViewImages   → 6 camera views (pipeline 호환용, 모델에서 무시)
LoadAID4ADSatelliteImage → sat_img (3, 400, 200), ImageNet normalized, y-flipped
PhotoMetricDistortion → camera augmentation (sat에는 미적용)
ResizeMultiViewImages → camera resize to 480×800
Normalize3D           → camera normalize
PadMultiViewImages    → camera pad to divisor 32
FormatBundleMap       → tensor 변환
Collect3D             → keys: [img, vectors, semantic_mask, sat_img]
```

**Note**: Camera images는 pipeline 호환성을 위해 로드되지만, `SatBEVProvider.forward()`에서 완전히 무시됨. DataLoader가 camera 로딩에 시간을 쓰므로 향후 최적화 가능.

### 5.2 Satellite Image Convention

```
AID4AD on disk: image convention (y-down at H=0)
  ↓ LoadAID4ADSatelliteImage: flipud (y-up)
  ↓ ImageNet normalize
sat_img tensor: (B, 3, 400, 200), y-up convention
  ↓ SatelliteEncoder: ResNet50 + FPN → (B, 256, 50, 100)
  ↓ torch.flip(feats, [2,]): y-up → BEVFormer convention (y-down at H=0)
BEV features: (B, 256, 50, 100), BEVFormer convention
```

---

## 6. Compute Profile

| Component | MapTracker (per training step) | SatOnlyMapTracker (per training step) |
|-----------|-------------------------------|---------------------------------------|
| Camera ResNet50-DCN + FPN | 6 views × 480×800 × 5 frames = **30 forward passes** | **0** |
| BEVFormer Encoder (2 layers) | 5 frames × deformable attn | **0** |
| SatelliteEncoder | 0 | 1 image × 400×200 × **1 forward** (cached) |
| Seg Decoder (MapSegHead) | 5× (per frame) | 5× (per frame) |
| Vector Head (6-layer decoder) | 5× (per frame, stage 2/3) | 5× (per frame, stage 2/3) |
| Query Propagation | 4× (between frames) | 4× (between frames) |
| **Estimated iter time** | ~3-5s (4090, batch=4) | ~1-2s (4090, batch=16) |

---

## 7. File Structure

```
plugin/
├── models/mapers/
│   ├── MapTracker.py              # Base model (1354 lines)
│   ├── SatMapTracker.py           # Camera + Satellite fusion variant
│   ├── SatOnlyMapTracker.py       # Satellite-only variant (this)
│   └── __init__.py                # Registers all three
├── models/backbones/
│   └── satellite_encoder.py       # SatelliteEncoder (ResNet50 + FPN agg)
├── configs/satonlymaptracker/nuscenes_newsplit/
│   ├── satonlymaptracker_stage1_seg_pretrain.py        # Stage 1 base (full dataset)
│   ├── satonlymaptracker_stage1_seg_pretrain_third.py  # Stage 1 (1/3 dataset)
│   ├── satonlymaptracker_stage2_warmup.py              # Stage 2 base
│   ├── satonlymaptracker_stage2_warmup_third.py        # Stage 2 (1/3 dataset)
│   ├── satonlymaptracker_stage3_joint_finetune.py      # Stage 3 base
│   └── satonlymaptracker_stage3_joint_finetune_third.py # Stage 3 (1/3 dataset)
└── datasets/pipelines/
    └── loading_sat.py             # LoadAID4ADSatelliteImage

tools/
├── train_satonlymaptracker_stage1.sh  # Stage 1 launch (4 GPU, CUDA 4,5,6,7)
├── train_satonlymaptracker_stage2.sh  # Stage 2 launch
└── train_satonlymaptracker_stage3.sh  # Stage 3 launch

docs/satonly/
└── sat_only_architecture.md           # This document
```

---

## 8. Training Commands

```bash
# Stage 1: Satellite Seg Pretrain (third dataset, 4 GPU)
bash tools/train_satonlymaptracker_stage1.sh

# Stage 2: Tracking Warmup (loads Stage 1 checkpoint)
bash tools/train_satonlymaptracker_stage2.sh

# Stage 3: Joint Finetuning (loads Stage 2 checkpoint)
bash tools/train_satonlymaptracker_stage3.sh
```

Checkpoint chain:
```
(none) → Stage 1 → work_dirs/satonlymaptracker_stage1_seg_pretrain_third/latest.pth
                  → Stage 2 → work_dirs/satonlymaptracker_stage2_warmup_third/latest.pth
                            → Stage 3 → work_dirs/satonlymaptracker_stage3_joint_finetune_third/latest.pth
```
