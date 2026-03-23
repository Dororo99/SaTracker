# SDMapTracker 아키텍처

---

## 1. 기존 MapTracker 아키텍처 (변경 없는 부분)

### 1.1 전체 구조

```
Camera Images [B, 6, 3, 480, 800]
       ↓
┌──────────────────────────────────┐
│  BEVFormerBackbone               │
│  ResNet50 + FPN + BEVFormer      │
│  (temporal self-attn + spatial   │
│   cross-attn, 2 encoder layers)  │
└──────────┬───────────────────────┘
           ↓ BEV features [B, 256, 50, 100]
           ├────────────────────────────┐
           ↓                            ↓
┌──────────────────┐         ┌──────────────────┐
│  MapSegHead      │         │  MapDetectorHead │
│  Conv → 4x Up    │         │  Transformer     │
│  → seg mask      │         │  Decoder 6 layers│
│  [B,3,200,100]   │         │  → vector map    │
└──────────────────┘         └──────────────────┘
```

- **BEV 좌표계**: ego-centric, 60m×30m → 50×100 grid (1px ≈ 0.6m×0.3m)
- **예측 클래스 3개**: ped_crossing(0), divider(1), boundary(2)
- **각 instance**: 20개 점의 polyline, [0,1] normalized 좌표

### 1.2 MapDetectorHead: Query → Prediction

```
Learnable Queries
  query_embedding: Embedding(100, 512)
  reference_points_embed: Linear(512, 40) → sigmoid → [B, 100, 20, 2]

       ↓ query [B, 100, 512], ref_pts [B, 100, 20, 2]

Transformer Decoder (6 layers, 각 layer):
  ├── Self-Attention: query ↔ query (MultiheadAttn, 8 heads, 512dim)
  ├── Cross-Attention: query ↔ BEV feature (Deformable Attn, 20 sampling pts)
  ├── Cross-Attention: query ↔ Memory Bank (track queries만, use_memory=True일 때)
  └── FFN: 512→1024→512

  좌표 예측 (predict_refine=False):
    reg_branches[l](query).sigmoid() → absolute 좌표 [B, 100, 20, 2]

       ↓ 각 layer마다 prediction

cls_branches[l](query) → class score [B, 100, 3]
reg_points              → polyline 좌표 [B, 100, 40]
```

### 1.3 Temporal Tracking (multi-frame)

```
Frame t:
  Head → prediction → Hungarian matching → positive queries 선별 (score > 0.4)
                                                    ↓
                                            hs_embeds + lines 저장
                                                    ↓
Frame t+1:
  1. Relative pose 계산 (ego2global transforms)
  2. MotionMLP로 embedding 전파:
     [hs_embeds(512) + pose_fourier(147)] → MLP(659→1024→1024→512) + residual
  3. Lines 좌표: pose transform으로 직접 변환
                                                    ↓
  Query layout: [track_queries(가변), dummy_queries(100), pad]
  Head → prediction (track은 forced matching)
```

### 1.4 Memory Bank

```
mem_bank: [bank_size=4, B, max_instances=300, 512]

Track query에 대해서만 memory cross-attention 수행:
  query[valid_track_idx] ← Attn(query, memory_embeds, temporal_pos_enc)

Dummy query는 memory 참조 없음 → BEV cross-attention만으로 예측
```

### 1.5 Loss

```
Classification: FocalLoss (weight=5.0, gamma=2.0, alpha=0.25)
Regression:     LinesL1Loss (weight=50.0, beta=0.01, permutation-aware)
Segmentation:   MaskFocalLoss(10.0) + MaskDiceLoss(1.0)
Temporal:       L1 propagation loss (weight=0.1, forward + backward)
```

### 1.6 기존 3-Stage 학습

| Stage | 내용 | Backbone | Vector Head | Memory | Epochs |
|-------|------|----------|-------------|--------|--------|
| **Stage 1** | BEV pretrain | 학습 | skip (seg만) | off | 18 |
| **Stage 2** | Vector warmup | freeze | 학습 | off (warmup 500 iter 후 on) | 4 |
| **Stage 3** | Joint finetune | 학습 | 학습 | on | 36 |

---

## 2. SDMapTracker: 변경 사항

### 2.1 변경 개요

```
기존 MapTracker                      SDMapTracker
─────────────────                    ─────────────────
Query = [track, dummy(100), pad]  →  Query = [track, sd_anchor(≤25), free(100), pad]
dummy = learnable embedding       →  sd_anchor = SD map prior에서 생성 (순수 추가)
                                     free = learnable embedding (기존 100개 유지)
ref_pts = learned linear          →  sd_ref = SD prior polyline 좌표 그대로
                                     free_ref = learned linear (기존과 동일)
```

**설계 원칙:** SD query는 기존 query를 대체하지 않고 순수하게 추가.
Free query 100개를 유지하여 기존 MapTracker 대비 detection capacity 손실 없음.

**변경되지 않는 것:**
- BEVFormerBackbone 전체
- MapSegHead 전체
- Transformer Decoder 구조 (6 layers, attention, FFN)
- predict_refine=False (absolute decode)
- MotionMLP, Memory Bank 구조
- Loss 함수 종류 (FocalLoss, LinesL1, Seg loss)
- Temporal propagation 로직

### 2.2 추가되는 모듈: SD Query Encoder

```python
sd_query_encoder = nn.Sequential(
    nn.Linear(40 + C_attr + K + 1, 512),
    nn.ReLU(),
    nn.Linear(512, 512),
)
```

| 입력 | 차원 | 설명 |
|------|------|------|
| polyline | 40 | SD prior의 20개 점 × 2 좌표 (flatten) |
| attrs | C_attr | lanes, width, oneway, highway_class 등 |
| has_tag_mask | K | 각 태그의 실제 존재 여부 (bool) |
| reliability | 1 | prior 신뢰도 (0~1) |

출력: (N_sd, 512) → query embedding으로 사용

### 2.3 변경되는 Query Layout

**기존 MapDetectorHead.forward_train (line 196~224):**
```
query = [track_embeds,   dummy_embeds(100),   pad_embeds]
ref   = [track_ref,      dummy_ref(100),      pad_ref]
mask  = [track_mask,     False(100),          True(pad)]
```

**변경 후:**
```
query = [track_embeds,   sd_embeds(≤25),      free_embeds(100),    pad_embeds]
ref   = [track_ref,      sd_prior_polyline,   free_ref,            pad_ref]
mask  = [track_mask,     sd_pad_mask,         False(100),          True(pad)]

+ query_meta = {
    num_track, num_sd, num_free,
    query_type: [0,0,..,1,1,..,2,2,..,3,3,..],  # track=0, sd=1, free=2, pad=3
    reliability: [1.0,..,r_i,..,1.0,..,0.0,..],  # track/free=1.0, pad=0.0
  }
```

**SD padding (query_type==3) 처리:** loss에서 완전 제외 (label_weight=0, num_total_neg 미포함)

**각 query type의 초기화 비교:**

| | embedding | reference point | 역할 |
|---|---|---|---|
| **track** (기존) | 이전 frame hs_embeds + MotionMLP | 이전 frame lines + pose transform | 이미 추적 중인 instance |
| **sd_anchor** (신규) | sd_query_encoder(polyline+attrs+reliability) | **SD prior polyline 좌표 그대로** | 새로 보이는 도로 (SD map 기반) |
| **free** (기존 dummy 유지) | learnable embedding (100개, 기존과 동일) | learned linear → sigmoid | SD map에 없는 도로 대비 |

### 2.4 SD-Track 매칭 (Head 호출 직전)

SD anchor와 기존 track이 같은 도로를 중복 표현하지 않도록:

```
MapTracker.forward에서, head 호출 직전:
  for each sd_prior:
    if same_class track query와 Chamfer < 0.05 (normalized ≈ 3m):
      → 해당 sd_prior 제거
  남은 sd_prior만 sd_anchor query로 사용
```

### 2.5 Assigner 변경

기존 Hungarian matching에 SD query용 prior cost 추가:

```
기존:  C = 5.0 * C_cls + 50.0 * C_reg

변경:  C = 5.0 * C_cls + 50.0 * C_reg + 5.0 * reliability * C_sd_init  (SD query만)

C_sd_init = L1(sd_prior_polyline, gt_polyline)  ← same-class GT에만 적용
```

Track query: forced matching 유지 (기존 그대로)
Free query: 기존 그대로 (prior cost 없음)

**전달 경로:**
```
Head.forward_train
  → pred_dict에 query_meta, sd_init_lines 포함
  → get_targets(preds, gts, query_meta, sd_init_lines)
    → _get_target_single()
      → assigner.assign(preds, gts, query_meta, sd_init_lines)
```

### 2.6 Loss 변경: SD Negative Down-weighting

SD query 중 GT와 매칭 안 된 것(background)의 cls loss weight를 낮춤:

```
기존: 모든 unmatched query에 동일한 background loss

변경: SD query의 unmatched에 대해
  label_weight *= reliability * 0.3

위치: MapDetectorHead._get_target_single() 내부
```

### 2.7 Post-process 변경

```
기존: props[-100:] = 0 (newborn), props[:-100] = 1 (tracked)
변경: query_meta['query_type']으로 판별
  - type=0 (track) → props=1
  - type=1,2 (sd, free) → props=0

추가: Dedup
  같은 class + Chamfer < threshold → 높은 score 유지
  우선순위: track > sd > free
```

### 2.8 Training Augmentation (신규)

```
SD Prior Dropout: 15% 확률로 발동, SD prior 중 30% random drop
SD Polyline Jitter: 항상, ±0.01 normalized noise (≈0.6m)
```

### 2.9 `num_queries` 하드코딩 제거

기존에 `self.head.num_queries`(=100)를 직접 참조하던 모든 곳을 `query_meta`로 대체:

| 위치 | 기존 | 변경 |
|------|------|------|
| `_batchify_tracks` | `tracked_len = lengths - num_queries` | `tracked_len = lengths - (max_sd + num_free)` |
| `prepare_track_queries_and_targets` | `pad_bound = tracked + num_queries` | `pad_bound = tracked + max_sd + num_free` |
| `post_process` | `props[-100:] = 0` | `query_type != 0` |
| `prepare_temporal_propagation` | `scores[:-100]` = track | `query_type == 0` = track |

---

## 3. 학습 구조 (3-Stage)

기존 MapTracker의 3-stage를 그대로 유지하되, Stage 2/3에서 SD query 관련 모듈 추가.

### Stage 0: Offline SD Prior Cache 생성 (1회)

```
tools/generate_sd_prior_cache.py 실행

Input:  OSM 캐시 (.osm) + nuScenes annotation pkl
Output: datasets/nuscenes/sd_prior_cache_{train|val}_newsplit.pkl

각 sample_token → {polylines, labels, attrs, has_tag_mask, reliability}
Phase 1 변환 로직 + GT-style 정제 (junction suppress, fragment 제거 등)
```

### Stage 1: BEV Pretrain (변경 없음)

```
목적: BEV feature 추출 학습 (backbone + seg head)
Vector head: skip (skip_vector_head=True)
SD query: 사용 안 함
Memory: off

학습 대상: BEVFormerBackbone + MapSegHead
Loss: seg_focal(10.0) + seg_dice(1.0)

기존 checkpoint 그대로 사용 가능.
```

| 항목 | 값 |
|------|---|
| Epochs | 18 |
| Backbone | 학습 |
| Vector Head | skip |
| SD Query | 없음 |
| Memory | off |
| Load from | ImageNet pretrained ResNet50 |

### Stage 2: SD+Free Warmup (변경)

```
목적: SD query encoder + vector head 학습
BEV backbone: freeze
Memory: off → warmup 후 on (mem_warmup_iters=500)

Query layout: [sd_anchor(≤25), free(100)]  (track 없음, single-frame이므로)
SD prior dropout/jitter 적용

학습 대상: sd_query_encoder, cls_branches, reg_branches, transformer decoder
```

| 항목 | 값 |
|------|---|
| Epochs | 4 |
| Backbone | **freeze** |
| Vector Head | 학습 |
| SD Query | **활성** |
| Memory | off → on (500 iter 후) |
| Load from | Stage 1 checkpoint (`strict=False`) |
| 새 모듈 초기화 | sd_query_encoder: random init |
| | query_embedding: (100,512) shape 일치 → 기존 weight 사용 가능 |

### Stage 3: Joint Finetune (변경)

```
목적: 전체 모델 joint 학습 (backbone 포함)
Multi-frame (5 frames, span 10)
Memory: on

Query layout: [track(가변), sd_anchor(≤25), free(100), pad]
SD-Track 매칭 + SDPriorCost + neg down-weight 전부 활성
SD prior dropout/jitter 적용
Temporal propagation loss 활성

학습 대상: 전체 (backbone + neck + seg + head + sd_encoder)
```

| 항목 | 값 |
|------|---|
| Epochs | 36 |
| Backbone | **학습** |
| Vector Head | 학습 |
| SD Query | **활성** |
| Memory | **on** |
| Track propagation | **활성** (MotionMLP + trans_loss) |
| Load from | Stage 2 checkpoint |

### Stage별 비교 요약

```
           Stage 1          Stage 2              Stage 3
           ─────────        ──────────           ──────────
Backbone   학습              freeze               학습
Seg Head   학습              (freeze)             학습
Vec Head   skip              학습                  학습
SD Query   없음              sd+free               track+sd+free
Memory     off               off→on(500)           on
Frames     1                 1 (→multi)            5 (span 10)
Trans Loss 없음              없음(→있음)            있음
Load       ImageNet          Stage 1               Stage 2

         ┌──────────┐    ┌──────────┐    ┌──────────┐
         │ Stage 1  │───→│ Stage 2  │───→│ Stage 3  │
         │ BEV만    │    │ SD+Vec   │    │ 전체     │
         │ 학습     │    │ 학습     │    │ 학습     │
         └──────────┘    └──────────┘    └──────────┘
              18ep            4ep             36ep
```

---

## 4. 전체 데이터 흐름 (Stage 3, 2 frames)

```
══════════════════════════════════════════════════
Frame t=0
══════════════════════════════════════════════════

Images [B,6,3,480,800] → BEVFormer → BEV [B,256,50,100]
                                          │
                          ┌───────────────┤
                          ↓               ↓
                     MapSegHead     SD Prior Cache 로드
                     → seg loss     → sd_query_encoder
                                    → sd_embed [B,≤50,512]
                                      sd_ref [B,≤25,20,2]
                                          │
                          free_embed [B,100,512] (learnable)
                          free_ref [B,100,20,2]  (learned linear)
                                          │
                     Q = [sd(≤25), free(100)]   ← 첫 프레임: track 없음
                                          │
                     Decoder 6 layers ────┤
                     (self-attn, BEV cross-attn, FFN)
                     (memory cross-attn: 없음, 첫 프레임)
                                          │
                     Hungarian Matching ───┤
                     (SDPriorCost for SD queries)
                                          │
                     Loss: cls + reg ──────┤
                     (SD neg down-weight)  │
                                          │
                     Positive queries → track_query_info
                     (score > 0.4, hs_embeds + lines 저장)

══════════════════════════════════════════════════
Frame t=1
══════════════════════════════════════════════════

Images → BEVFormer (+ history BEV warping) → BEV [B,256,50,100]
                                                   │
                          ┌────────────────────────┤
                          ↓                        ↓
                     MapSegHead              Track Propagation
                     → seg loss              MotionMLP(embeds + pose)
                                             Lines: pose transform
                                             → trans_loss(0.1)
                                                   │
                                             SD Prior Cache 로드
                                             SD-Track 매칭 (중복 제거)
                                             → sd_query_encoder
                                                   │
                     Q = [track(가변), sd_anchor(가변), free(100), pad]
                         query_meta: {type, reliability}
                                                   │
                     Decoder 6 layers ─────────────┤
                     (self-attn, BEV cross-attn,   │
                      memory cross-attn: track만,  │
                      FFN)                         │
                                                   │
                     Hungarian Matching ───────────┤
                     track: forced match           │
                     sd: cls + reg + prior_cost     │
                     free: cls + reg               │
                                                   │
                     Loss: cls + reg ──────────────┤
                     (SD pad excluded from loss)   │
                     (SD neg down-weight)          │
                                                   │
                     Memory Bank update
                     Post-process + Dedup

Total Loss = Σ_frames (cls + reg + seg + dice) + trans_loss
══════════════════════════════════════════════════
```
