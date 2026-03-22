# SDMapTracker: SD Map Prior를 활용한 MapTracker 성능 향상 계획

## 목표

OSM(OpenStreetMap) SD map 정보를 MapTracker에 주입하여:
1. **Occlusion 상황**에서 SD map prior fallback으로 성능 유지
2. **장거리 detection** 성능 향상 (15-30m 영역)
3. 기존 MapTracker 구조를 최대한 유지하면서 점진적 확장

---

## Phase 1 실험 결과 요약 (완료)

**실험**: 모델 없이 OSM SD map만으로 vector map을 생성하여 nuScenes GT와 비교

### 사용한 OSM 정보
- Way geometry (도로 전체의 중심선 좌표)
- `lanes` 태그 → divider 개수 결정
- `width` 태그 → boundary offset 거리
- `oneway` 태그 → 양방향/단방향 divider 배치
- `highway` 등급 → 기본값 결정 및 vehicle way 필터링
- `footway=crossing` → ped_crossing 식별

### 변환 로직
- Way centerline ±(width/2) offset → road boundary
- Way를 lanes 수로 등분 → lane divider 위치
- Crossing way → ped_crossing polyline
- 좌표 변환: WGS84 → nuScenes city coords (Haversine, SDTagNet 동일 방식) → ego-centric BEV

### 정량 결과 (nuScenes newsplit val, 500 samples)

| 클래스 | Mean CD | Median CD | n |
|--------|---------|-----------|---|
| divider | 3.84m | 3.52m | 489/500 |
| boundary | 2.89m | 2.87m | 500/500 |
| ped_crossing | 3.36m | 2.24m | 208/500 |

#### 도시별

| 도시 | divider | boundary | ped_crossing |
|------|---------|----------|-------------|
| boston-seaport | 4.44m (n=292) | 3.28m (n=302) | 3.52m (n=167) |
| singapore-hollandvillage | 2.88m (n=58) | 2.34m (n=58) | - |
| singapore-onenorth | 2.82m (n=83) | 2.31m (n=84) | 3.25m (n=27) |
| singapore-queenstown | 3.17m (n=56) | 2.26m (n=56) | 1.75m (n=14) |

- Best case: 0.19m (직선 도로, lanes 태그 있음)
- Worst case: 9.15m (복잡한 교차로, instance 과다 생성)

### 핵심 결론
- **SD map만으로 GT 대비 평균 3m 이내** — 모델이 ~3m만 refine하면 됨
- 오차 원인: width 태그 부재 시 기본값 사용(~2-3m), 교차로 과생성, instance 분절 불일치
- **lanes 태그가 핵심** — vehicle way 중 거의 전부에 존재하며 instance 개수를 직접 결정

### 관련 파일
- 실험 노트북: `/home/kyungmin/min_ws/mapping/phase1_sdmap_vectormap.ipynb`
- OSM 캐시: `/home/kyungmin/min_ws/mapping/{city}_roads.osm`
- 좌표 변환 참조: `SDTagNet/tools/sdtagnet/nusc_to_wgs_conversion.py`

---

## 설계 방향: SD Map을 Query Prior로 직접 활용

### 왜 BEV Rasterize가 아닌 Query Prior인가

| | BEV Rasterize + Add | Query Prior (채택) |
|---|---|---|
| SD map 정보 보존 | 0.6m grid로 뭉개짐 | 원본 좌표 그대로 |
| lanes=3 활용 | 스칼라 채널 하나 | 3개 polyline instance로 직접 변환 |
| 모델이 할 일 | BEV에서 도로 다시 찾기 | ~3m 보정만 하면 됨 |
| Occlusion fallback | 별도 학습 필요 | SD prior 자체가 초기값으로 작동 |

### 왜 BERT 텍스트 인코딩이 아닌 Structured Encoding인가 (SDTagNet 대비)

SDTagNet은 `"highway: residential, lanes: 2, ..."` 문자열을 BERT로 인코딩.
- `lanes: 2` vs `lanes: 4`가 NLP 임베딩에서 명확히 구분 안 됨
- Heavyweight BERT가 도메인 특화 정보 인코딩에 과함
- → **숫자는 숫자로, 범주는 범주로** structured feature vector + MLP

### Track Query는 SD Prior를 받지 않는다

Track query는 이전 프레임에서 GT에 매칭된 prediction으로, 위치가 이미 ~1m 이내로 정확.
SD prior는 ~3m 오차이므로 track보다 부정확 → track을 SD로 당기면 오히려 성능 하락.

**SD prior가 힘을 발휘하는 상황:**
- 첫 프레임 (track 전파 자체가 없음)
- 새로 ROI에 들어온 도로 (이전에 안 보이던 영역)
- Occlusion으로 track이 끊긴 경우 (score < threshold → track 탈락 → 다음 프레임에서 SD anchor가 다시 잡아줌)

SD anchor가 positive로 매칭되면 다음 프레임부터 **일반 track query**로 전파됨.
SD prior의 reliability/attrs 등은 전파하지 않음 (track은 hs_embeds + lines 좌표만 전파하는 구조).
다음 프레임에서 해당 track이 다시 SD prior와 매칭되면 dedup 용도로만 사용.

**향후 확장 가능성**: 장기 시퀀스에서 track drift가 문제되면, "low-confidence track에 한해 nearby high-reliability SD prior로 보정"하는 옵션 추가 가능. 1차에서는 빼고 실험 결과를 보고 판단.

---

## 구현 계획

### 핵심 설계 결정

#### 1. `predict_refine=False` 유지
- 현재 sdmaptracker의 모든 config, 모든 decode 경로가 absolute decode 전제
- trans_loss, _process_track_query_info, prepare_temporal_propagation 전부 `reg_branches(embed).sigmoid()`
- predict_refine=True로 바꾸면 이 모든 경로를 동시에 수정해야 하고, 기존 checkpoint 호환도 깨짐
- **SD map 효과를 보는 실험에서 decoder semantics까지 바꾸면 변수가 너무 많음**
- SD prior 좌표는 query의 reference point 초기값으로 주입, decoder는 기존 absolute decode 수행

#### 2. Query Layout: `[track, sd_anchor, free, pad]`
- 기존: `[track_queries, dummy_queries(100개), pad]`
- 변경: `[track_queries, sd_anchor_queries, free_queries(50개), pad]`
- track: 이전 프레임에서 전파된 query (기존 그대로)
- sd_anchor: 현재 ROI의 SD prior 중 track에 매칭 안 된 것 (newborn candidate)
- free: learnable query (기존 dummy의 축소 버전, OSM에 없는 도로 대비)

#### 3. `num_queries` 하드코딩 제거
- 현재 `self.head.num_queries`가 pad_bound, tracked_query_length, post_process, temporal_propagation에서 사용
- `query_meta` dict로 대체: `{num_track, num_sd, num_free, query_type}`
- newborn 판별: `query_type != 0` (기존 "마지막 100개" 가정 제거)

#### 4. Checkpoint 호환성
- query_embedding (100, 512) → (50, 512)로 크기 변경되므로 기존 checkpoint과 shape 불일치 발생
- `strict=False`로 warm start: 매칭되는 weight는 로드하고, sd_query_encoder 등 새 모듈은 랜덤 초기화
- "완전한 호환"은 아님. BEV backbone + neck + seg_decoder + cls/reg_branches는 재사용 가능

### 구현 항목

#### Stage 0: Offline SD Prior Cache 생성

**파일**: `tools/generate_sd_prior_cache.py` (신규)

Phase 1의 `generate_sdmap_vectors` 로직을 재사용하되 GT-style 정제 추가:

```
각 sample_token → sd_priors dict:
  polylines: [N, 20, 2]  # normalized [0,1], MapTracker 좌표계
  labels: [N]             # 0=ped, 1=div, 2=bnd
  attrs: [N, C]           # lanes, width, oneway, highway_class (숫자 그대로)
  has_tag_mask: [N, K]    # 실제 태그 존재 여부 (default 사용 여부 구분)
  reliability: [N]        # 0~1
```

GT-style 정제:
- Junction 내부 boundary/divider suppress
- 3m 미만 fragment 제거
- 20점 uniform sampling + [0,1] normalize
- Reliability = f(has_lanes, has_width, is_junction, distance_to_ego)
  - has_lanes & has_width → 1.0
  - has_lanes only → 0.7
  - default값만 → 0.4
  - junction 내부 → 0.1

출력: `datasets/nuscenes/sd_prior_cache_{split}.pkl`

#### A. 데이터 파이프라인 수정 (sd_priors를 모델까지 전달)

sd_priors가 모델에 도달하려면 dataset → formatting → collate → forward까지 전체 경로를 수정해야 함.

**수정 파일과 변경 내용:**

1. **`plugin/datasets/nusc_dataset.py`**
   - `__init__`에서 sd_prior_cache 로드
   - `get_sample()`에서 `results['sd_priors'] = cache[token]`

2. **`plugin/datasets/pipelines/formating.py`** (`FormatBundleMap`)
   - 현재: img, vectors, semantic_mask만 포맷
   - 변경: `sd_priors`를 `DataContainer(stack=False, cpu_only=True)`로 추가
   - sd_priors는 sample마다 N이 다르므로 stack 불가

3. **`plugin/datasets/base_dataset.py`** (`__getitem__` / collate 관련)
   - `get_sample()` 반환값에 sd_priors 포함 확인
   - multi-frame 학습 시 `all_prev_data`에도 각 프레임의 sd_priors가 포함되도록 수정
   - `__getitem__`에서 prev frame data 수집 시 sd_priors도 같이 수집

4. **`plugin/models/mapers/MapTracker.py`** (`forward_train`, `forward_test`)
   - `forward_train` 시그니처에 `sd_priors` 인자 추가
   - `batch_data()`에서 sd_priors 추출
   - `all_prev_data` 순회 시 각 prev frame의 sd_priors도 같이 전달
   - `forward_test`에서도 img_metas 또는 별도 인자로 sd_priors 전달

5. **Config 파일** (`stage2_warmup.py`, `stage3_joint_finetune.py`)
   - train/test pipeline에 sd_priors 관련 설정 추가

#### B. SD-Track 매칭 (중복 방지)

SD anchor query를 만들기 전에, propagated track query와 SD prior 간 중복을 제거해야 함.

**호출 위치**: `MapTracker.forward_train/test`에서 **head 호출 직전**에 수행.
```python
# MapTracker.forward_train, head 호출 직전:
sd_priors_filtered = self.filter_sd_priors_by_tracks(
    sd_priors, track_query_info)
sd_query_info = self.build_sd_query_info(sd_priors_filtered)
self.head(..., sd_query_info=sd_query_info, ...)
```

**매칭 로직:**
```python
def filter_sd_priors_by_tracks(self, sd_priors, track_query_info):
    """track과 매칭된 SD prior를 제거하여 sd_anchor 후보만 남김"""
    if track_query_info is None:
        return sd_priors  # 첫 프레임: track 없음, 모든 SD prior 사용

    unmatched_mask = torch.ones(len(sd_priors['polylines']), dtype=torch.bool)

    for b_i in range(bs):
        track_ref_pts = track_query_info[b_i]['trans_track_query_boxes']  # (N_track, 40)
        track_labels = track_query_info[b_i]['track_query_labels']

        for i, (sd_poly, sd_label) in enumerate(zip(
                sd_priors[b_i]['polylines'], sd_priors[b_i]['labels'])):
            same_cls = (track_labels == sd_label)
            if same_cls.any():
                dists = chamfer_distance(sd_poly, track_ref_pts[same_cls])
                if dists.min() < SD_TRACK_MATCH_THRESHOLD:  # ~0.05 normalized ≈ 3m
                    unmatched_mask[i] = False

    return {k: v[unmatched_mask] for k, v in sd_priors[b_i].items()}
```

- threshold ≈ 0.05 (normalized 좌표 기준, 실제 ~3m)
- 첫 프레임에서는 track 없음 → 모든 SD prior가 sd_anchor로 들어감
- same-class만 비교 (divider ↔ boundary는 매칭 안 함)

#### C. Variable-length SD Query 배치화

sample마다 N_sd가 다르므로 batch 차원에서 padding 필요.

**규약:**
- `max_sd_queries` (config, default 50)를 상한으로 설정
- N_sd > max_sd_queries인 경우: reliability 순으로 상위 max_sd_queries개만 사용
- N_sd < max_sd_queries인 경우: zero-padding + query_padding_mask=True

**배치화 흐름** (기존 `_batchify_tracks` 패턴과 동일):
```python
# 각 batch item에 대해:
# query = [track(가변), sd(가변, max 50), free(고정 50), pad]
# 전체를 batch 내 max_len으로 padding
# query_padding_mask: pad 위치 = True

# _batchify_tracks에서:
for b_i in range(bs):
    total_len = n_track[b_i] + n_sd[b_i] + num_free_queries
    # pad to max_len across batch
    target['query_meta'] = {
        'num_track': n_track[b_i],
        'num_sd': n_sd[b_i],
        'num_free': num_free_queries,
        'query_type': LongTensor,  # 0=track, 1=sd, 2=free, 3=pad
    }
```

#### D. MapDetectorHead 수정

**파일**: `plugin/models/heads/MapDetectorHead.py`

새로 추가할 모듈:
```python
self.sd_query_encoder = nn.Sequential(
    nn.Linear(20*2 + C_attr + K_tag_mask + 1, embed_dims),
    nn.ReLU(),
    nn.Linear(embed_dims, embed_dims),
)
# 입력: polyline(40) + attrs(C) + has_tag_mask(K) + reliability(1)
# 출력: (N_sd, embed_dims)
```

`forward_train/test` 변경 (line 196~241):
```python
# 현재:
# query = [track, dummy(100), pad]

# 변경:
# 1) SD query 생성
sd_embed = self.sd_query_encoder(sd_features)  # (N_sd, embed_dims)
sd_ref_pts = sd_priors['polylines']             # (N_sd, 20, 2), 이미 normalized

# 2) Free query
free_embed = self.query_embedding.weight[:self.num_free_queries]  # (N_free, embed_dims)
free_ref = self.reference_points_embed(free_embed).sigmoid()      # (N_free, 20, 2)

# 3) 조립
if track_query_info is not None:
    query = cat([track_embed, sd_embed, free_embed, pad_embed])
    ref   = cat([track_ref,   sd_ref,   free_ref,   pad_ref])
else:
    query = cat([sd_embed, free_embed])
    ref   = cat([sd_ref,   free_ref])
```

Free query 크기:
- `num_free_queries = 50` (기존 100에서 축소, train split 통계로 최적화 가능)
- `max_sd_queries = 50` (초과 시 reliability 순으로 자르되, class별 최소 5개 보장)
  - 예: boundary 30개, divider 25개, ped 3개 → ped는 min 5 보장, 나머지 reliability 순 cut

`query_meta`를 pred_dict에 포함:
```python
pred_dict['query_meta'] = {
    'num_track': ..., 'num_sd': ..., 'num_free': ...,
    'query_type': LongTensor,  # 0=track, 1=sd, 2=free, 3=pad
    'reliability': FloatTensor,  # per-query reliability (track/free는 1.0, sd는 cache 값, pad는 0.0)
}
```

**reliability 전달 규약**: `query_meta['reliability']`가 유일한 source.
- Stage 0 cache에서 생성 → dataset에서 로드 → head에서 query_meta에 패딩하여 포함
- assigner(SDPriorCost)와 loss(negative down-weight) 모두 `query_meta['reliability']`를 참조
- 별도 `pred_dict['query_reliability']`는 만들지 않음 (중복 방지)

#### E. MapTracker 수정

**파일**: `plugin/models/mapers/MapTracker.py`

변경해야 하는 함수 목록과 구체적 변경 내용:

1. **`forward_train` (line 344~584)**
   - sd_priors를 인자로 받음
   - backbone → bev_feats 이후, head 호출 전에 `sd_query_info` 생성
   - 매 prev frame 순회 시에도 해당 frame의 sd_priors 전달
   - head 호출: `self.head(..., sd_query_info=sd_query_info, ...)`

2. **`forward_test` (line 586~692)**
   - sd_priors를 인자로 받음 (img_metas에 포함 또는 별도 인자)
   - SD-Track 매칭 후 sd_query_info 생성
   - head 호출 시 전달

3. **`prepare_track_queries_and_targets` (line 793~907)**
   - `pad_bound` 계산: `tracked_length + num_sd + num_free` (기존 `+ num_queries`)
   - `track_queries_mask`, `track_queries_fal_pos_mask` 길이: `+ num_sd + num_free` (기존 `+ num_queries`)
   - `target['query_meta']` 추가
   - SD anchor에 대한 mask: track처럼 forced matching 아님, 그냥 newborn candidate

4. **`_batchify_tracks` (line 909~924)**
   - `tracked_query_length = lengths - (num_sd + num_free)` (기존 `- num_queries`)
   - pad_hs_embeds, pad_query_boxes 크기 조정
   - query_meta도 같이 padding

5. **`_process_track_query_info` (line 970~985)**
   - query_type 정보가 있으면 track/sd/free 구분하여 line decode
   - 기존 로직은 track query에 대해서만 수행하므로 큰 변경 없음

6. **`post_process` (MapDetectorHead 내)**
   - 기존: `props[-100:] = 0` (newborn)
   - 변경: `query_type`으로 판별. track=propagated, sd+free=newborn
   - `origin` 필드 추가: track/sd/free 구분 (dedup, 분석용)

7. **`prepare_temporal_propagation` (MapDetectorHead 내)**
   - 기존: `scores[:-100]` = track, `scores[-100:]` = detection
   - 변경: `query_type`으로 구분

#### F. Assigner 수정

**파일**: `plugin/models/assigner/match_cost.py`, `plugin/models/assigner/assigner.py`, `plugin/models/heads/MapDetectorHead.py`

현재 matching cost 인터페이스: `preds['scores'], preds['lines'], gts['labels'], gts['lines']`만 받음.

변경:

1. **`match_cost.py`**: `SDPriorCost` 클래스 추가
   ```python
   class SDPriorCost:
       """SD query의 init polyline과 GT 간 L1 distance"""
       def __call__(self, sd_init_lines, gt_lines, sd_labels, gt_labels):
           # same-class만 비교, 다른 class는 cost=0 (bonus 없음)
           cost = L1(sd_init_lines, gt_lines)  # bounded
           class_mask = (sd_labels[:, None] == gt_labels[None, :])
           return cost * class_mask
   ```

2. **`assigner.py`**: `assign()` 메서드에 `query_meta`, `sd_init_lines` 인자 추가
   - track query: 기존 forced matching 유지 (cost matrix inf/-1 조작)
   - SD query: `C_total = C_cls + C_reg + λ_prior * reliability * C_sd_init`
   - free query: 기존 그대로 (`C_cls + C_reg`)
   - `λ_prior ≈ 5.0` (C_reg=50.0 대비 10%)
   - reliability는 `query_meta['reliability']`에서 가져옴

3. **Head → Assigner 연결 경로** (이 부분이 핵심):

   현재 호출 체인:
   ```
   MapDetectorHead.forward_train()
     → self.get_targets()        # line ~397
       → self._get_target_single()  # line ~485, 여기서 assigner.assign() 호출
   ```

   변경 필요 사항:
   - **`get_targets()`**: 인자에 `query_meta`, `sd_init_lines` 추가하여 `_get_target_single`로 전달
   - **`_get_target_single()`**: assigner.assign() 호출 시 `query_meta`, `sd_init_lines` 전달
   - **`forward_train()`**: get_targets() 호출 시 pred_dict에서 query_meta, sd_init_lines 추출하여 전달

   ```python
   # forward_train에서:
   targets = self.get_targets(
       preds_list, gts, query_meta=pred_dict['query_meta'],
       sd_init_lines=pred_dict.get('sd_init_lines'))

   # _get_target_single에서:
   assign_result = self.assigner.assign(
       preds, gts, query_meta=query_meta,
       sd_init_lines=sd_init_lines)
   ```

4. **`pred_dict['sd_init_lines']`**: SD query의 원본 prior polyline (refine 전)
   - head의 forward에서 SD ref_pts를 저장해두고 pred_dict에 포함
   - track/free query 위치에는 None 또는 dummy 값

#### G. Loss: SD Negative Down-weighting

**파일**: `plugin/models/heads/MapDetectorHead.py` (loss 계산 부분)

현재: `_get_target_single()` → `label_weights` 생성 → `loss_cls` 계산
변경: SD query 중 unmatched(background)의 `label_weights`를 reliability에 따라 낮춤

```python
# _get_target_single 또는 loss_single에서:
if query_meta is not None:
    sd_mask = (query_meta['query_type'] == 1)  # SD query
    neg_mask = (labels == self.num_classes)      # background
    sd_neg_mask = sd_mask & neg_mask

    # reliability-aware down-weighting
    label_weights[sd_neg_mask] *= reliability[sd_neg_mask] * sd_neg_weight_factor  # factor ≈ 0.3
```

Junction/default-width prior가 매칭 안 되는 건 당연 → hard negative로 때리면 SD branch 죽음.

#### H. Training Augmentation: SD Prior Dropout / Jitter

Dataset 또는 MapTracker.forward_train에서 적용:

```python
# SD prior dropout (15% 확률로 전체 적용, 적용 시 30% drop)
if self.training and random.random() < 0.15:
    drop_mask = torch.rand(N_sd) < 0.3
    sd_priors = {k: v[~drop_mask] for k, v in sd_priors.items()}

# SD polyline jitter (항상 적용)
sd_priors['polylines'] += torch.randn_like(...) * 0.01  # normalized 좌표 기준
```

목적:
- 모델이 SD prior를 맹신하지 않도록
- Free query가 계속 학습되도록 (OSM missing/outdated case 대비)

#### I. Inference Post-process: Dedup

**파일**: `plugin/models/heads/MapDetectorHead.py` (post_process 내)

SD query와 free query가 같은 map element을 동시에 예측할 수 있음:
```python
# 같은 class + Chamfer < dedup_threshold → 높은 score 하나만 남기기
# 우선순위: track > sd > free (동일 score 시)
```

#### J. vector_memory 수정

**파일**: `plugin/models/mapers/vector_memory.py`

현재: `max_number_ins = 3 * number_ins`, `number_ins = head_cfg.num_queries`
변경: `number_ins`를 별도 config `memory_num_instances`로 분리하거나 `num_free_queries + max_sd_queries`로 설정

#### K. WandB Logging 추가

Config에 WandbLoggerHook 추가:
```python
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook'),
        dict(type='WandbLoggerHook',
             init_kwargs=dict(
                 entity='IRCV_Mapping',
                 project='sdmaptracker',
                 name='{experiment_name}',
             )),
    ])
```

Val 시 GT vs Pred 정성적 결과를 WandB에 scene별로 로깅.

---

## 학습 순서

| Stage | 내용 | 비고 |
|-------|------|------|
| **Stage 0** | Offline SD prior cache 생성 | 전처리 스크립트 1회 실행 |
| **Stage 1** | BEV pretrain | 변경 없음, 기존 checkpoint 사용 |
| **Stage 2** | SD+free warmup | 새 config 작성 (기존 stage2와 별도 분기), SD query 포함, sd_query_encoder 학습 |
| **Stage 3** | Joint finetune (multi-frame) | Track+SD+free, `use_memory=True`, 전체 학습 |

Stage 1 checkpoint는 기존 것 재사용 (`strict=False` warm start).
Stage 2에서 SD query 관련 모듈(sd_query_encoder 등)만 새로 랜덤 초기화.
Stage 2/3는 기존 repo의 stage2/3 config와 독립된 새 config로 분기.

**Stage 전환 기준**: 기존 MapTracker와 동일하게 epoch 수로 결정.
- Stage 2: 24 epoch (기존 stage2 warmup과 동일 규모)
- Stage 3: 36 epoch (기존 stage3 joint finetune과 동일 규모)
- 학습 중 val mAP가 plateau하면 조기 종료 가능

---

## Ablation 실험 계획

| 실험 | SD query | Free query | 목적 |
|------|----------|------------|------|
| Baseline | 0 | 100 | 기존 MapTracker (비교 기준) |
| SD only | 50 | 0 | SD prior만으로 어디까지 가는지 |
| SD + Free | 50 | 50 | **메인 실험** |
| Free only (축소) | 0 | 50 | free 축소 자체의 영향 분리 |
| No dropout | 50 | 50 | SD prior dropout/jitter 효과 |
| No SDPriorCost | 50 | 50 | assigner prior bonus 효과 |
| No neg down-weight | 50 | 50 | SD negative down-weighting 효과 |

---

## 변경 파일 요약

| 파일 | 변경 | 난이도 |
|------|------|--------|
| `tools/generate_sd_prior_cache.py` | **신규**. offline cache 생성 | 중 |
| `plugin/datasets/nusc_dataset.py` | cache 로드 + results 추가 | 낮음 |
| `plugin/datasets/pipelines/formating.py` | FormatBundleMap에 sd_priors DC 추가 | 낮음 |
| `plugin/datasets/base_dataset.py` | get_sample/collate에서 sd_priors 수집, multi-frame 시 prev data에도 포함 | 중 |
| `plugin/models/heads/MapDetectorHead.py` | sd_query_encoder, query 조립, query_meta, loss down-weight, post_process, prepare_temporal_propagation 수정 | 높음 |
| `plugin/models/mapers/MapTracker.py` | sd_priors 인자 추가, SD-Track 매칭, sd_query_info 생성, prepare_track/batchify 수정 (6개 함수) | 높음 |
| `plugin/models/assigner/match_cost.py` | SDPriorCost 추가 | 낮음 |
| `plugin/models/assigner/assigner.py` | SD cost 적용, sd_query_info 인자 추가 | 중 |
| `plugin/models/mapers/vector_memory.py` | max_number_ins 상한 조정 | 낮음 |
| `plugin/configs/sdmaptracker/` | 신규 config 파일들 (stage2, stage3) | 낮음 |

---

## 실험 분석 계획

단순 mAP 외에 반드시 분석할 항목:

- **거리별 성능** (0-15m / 15-30m) — 장거리 개선 확인
- **Class별 성능** — boundary가 가장 올라야 함
- **GT 매칭 source 비율** — track/sd/free 중 어디에서 매칭됐는지
- **도시별 성능** — OSM 품질 차이 반영
- **has_tag vs default 성능** — SD prior 품질 영향
- **First-frame recall** — SD prior 덕분에 첫 프레임에서도 높아야 함
- **Duplicate rate** — dedup 전후 비교
- **Junction vs non-junction** — GT-style 정제 효과 확인
- **OSM missing/outdated case에서 free query recall** — free query가 살아있는지
