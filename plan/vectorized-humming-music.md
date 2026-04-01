# Skeleton-Recall Loss Integration Plan (Revised)

## Scope / Target

이 문서는 **`/workspace/dohyun/sat_maptracker`** 기준 수정 계획이다.
목표는 SatMapTracker Stage1 BEV pretrain에 **클래스 선택적 Skeleton-Recall Loss**를 추가해,
`divider(class 1)`와 `boundary(class 2)`의 선형 연결성을 강화하는 것이다.
`ped_crossing(class 0)`은 면적형 구조이므로 skeleton 대상에서 제외한다.

Skeleton GT는 기존 `semantic_mask`에서 **iteration마다 on-the-fly 생성**한다.
따라서 **pkl 재생성은 하지 않는다.**

---

## 이번 수정본에서 확정한 결정 사항

1. **대상 repo를 명시한다.**
   - 기준 경로: `/workspace/dohyun/sat_maptracker`
   - 다른 fork / clone은 이번 계획의 기준이 아니다.

2. **새 의존성은 추가하지 않는다.**
   - 기존 안의 `skimage.morphology.skeletonize()` 사용안은 제거한다.
   - 대신 이미 repo가 사용 중인 `cv2` 기반 morphology 연산으로 skeleton을 만든다.
   - 만약 runtime에서 `cv2`가 없다면, 그것은 새 문제라기보다 기존 `RasterizeMap` 실행 환경 문제다.

3. **`MapSegHead`는 shared component이므로 `SatMapTracker`만 고치면 안 된다.**
   - `plugin/models/mapers/MapTracker.py`도 함께 수정해 shared return contract를 맞춘다.

4. **backward compatibility는 `loss_skel=None` 경로로 유지한다.**
   - skeleton loss를 쓰지 않는 config / batch에서는 `seg_skel == 0`으로 안전하게 동작해야 한다.

5. **변경하지 않는 범위도 명시한다.**
   - `test_pipeline`, `eval_config`, `base_dataset.py`, dataset pkl 포맷은 변경하지 않는다.
   - `base_dataset.py`는 이미 `all_prev_data`에 pipeline 결과를 넣고 있으므로 추가 수정이 필요 없다.

---

## 수정 파일 목록

| # | 파일 | 액션 | 설명 |
|---|------|------|------|
| 1 | `plugin/datasets/pipelines/skeletonize.py` | 신규 | `semantic_mask -> skeleton_mask` 생성 |
| 2 | `plugin/datasets/pipelines/__init__.py` | 수정 | `SkeletonizeMap` 등록 |
| 3 | `plugin/datasets/pipelines/formating.py` | 수정 | `skeleton_mask`를 tensor / DC로 변환 |
| 4 | `plugin/models/losses/seg_loss.py` | 수정 | `SkelRecallLoss` 추가 |
| 5 | `plugin/models/losses/__init__.py` | 수정 | loss import 등록 |
| 6 | `plugin/models/heads/MapSegHead.py` | 수정 | `loss_skel` 추가, train return 5-tuple로 확장 |
| 7 | `plugin/models/mapers/MapTracker.py` | 수정 | shared caller도 `seg_skel`를 수용 |
| 8 | `plugin/models/mapers/SatMapTracker.py` | 수정 | `skeleton_mask` 전달 + `seg_skel` 로깅 |
| 9 | `plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py` | 수정 | train pipeline / `Collect3D` / `loss_skel` 설정 |

---

## 구현 계획

### Step 1: `SkeletonizeMap` pipeline transform 추가

**파일:** `plugin/datasets/pipelines/skeletonize.py`

`RasterizeMap` 다음 단계에서 `semantic_mask`를 읽어 `skeleton_mask`를 생성한다.

핵심 구현:
- `@PIPELINES.register_module(force=True)` 사용
- 입력: `input_dict['semantic_mask']` (`np.uint8`, shape `(C, H, W)`)
- 출력: `input_dict['skeleton_mask']` (`np.uint8`, shape `(C, H, W)`, 값은 `{0,1}`)
- 기본 파라미터:
  - `skel_classes=(1, 2)`
  - `num_dilations=2`
  - `kernel_size=3`

구현 방식:
- `semantic_mask`와 동일 shape의 zero mask를 먼저 만든다.
- `skel_classes`에 포함된 채널만 처리한다.
- 각 채널은 아래 절차로 skeleton화한다.

```python
mask = (semantic_mask[class_idx] > 0).astype(np.uint8) * 255
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
skel = np.zeros_like(mask)
current = mask.copy()

while cv2.countNonZero(current) > 0:
    eroded = cv2.erode(current, kernel)
    opened = cv2.dilate(eroded, kernel)
    residue = cv2.subtract(current, opened)
    skel = cv2.bitwise_or(skel, residue)
    current = eroded

if num_dilations > 0:
    skel = cv2.dilate(skel, kernel, iterations=num_dilations)

skeleton_mask[class_idx] = (skel > 0).astype(np.uint8)
```

세부 규칙:
- class 0(`ped_crossing`)은 항상 zero로 둔다.
- skeletonize 전/후 dtype 변환 규칙을 고정한다.
  - morphology 연산용 내부 표현: `{0,255}` `uint8`
  - 최종 저장: `{0,1}` `uint8`
- 빈 mask(`sum == 0`)는 그대로 zero 채널을 유지한다.

**왜 이렇게 하는가:**
- `cv2`는 이미 `plugin/datasets/pipelines/rasterize.py`에서 사용 중이므로 새 의존성이 아니다.
- skeleton 생성 로직을 pipeline으로 두면 현재 프레임 / 이전 프레임 모두 동일 경로로 자동 전파된다.

---

### Step 2: pipeline registry 등록

**파일:** `plugin/datasets/pipelines/__init__.py`

추가:
```python
from .skeletonize import SkeletonizeMap
```

`__all__`에 `'SkeletonizeMap'` 추가.

---

### Step 3: `FormatBundleMap`에 `skeleton_mask` 처리 추가

**파일:** `plugin/datasets/pipelines/formating.py`

기존 `semantic_mask` 처리 블록 바로 아래에 동일 패턴을 추가한다.

```python
if 'skeleton_mask' in results:
    if isinstance(results['skeleton_mask'], np.ndarray):
        results['skeleton_mask'] = DC(to_tensor(results['skeleton_mask']), stack=True,
                                      pad_dims=None)
    else:
        assert isinstance(results['skeleton_mask'], list)
        results['skeleton_mask'] = DC(results['skeleton_mask'], stack=False)
```

주의:
- `FormatBundleMap.__init__`의 `keys` 기본값은 실제 분기 로직을 강제하지 않으므로, 여기서는 `__call__` 처리 추가만 있으면 충분하다.

---

### Step 4: `SkelRecallLoss` 추가

**파일:** `plugin/models/losses/seg_loss.py`

`MaskDiceLoss` 아래에 새 loss를 추가한다.

```python
@LOSSES.register_module()
class SkelRecallLoss(nn.Module):
    def __init__(self, loss_weight=1.0, skel_classes=(1, 2), eps=1e-6):
        ...

    def forward(self, pred, skel_gt):
        # pred: (B, C, H, W) logits
        # skel_gt: (B, C, H, W) binary {0,1}
        ...
```

계산 규칙:
1. `pred.sigmoid()` 적용
2. `skel_classes` 채널만 선택
3. `(B, C, H, W) -> (B, C, HW)` flatten
4. per-sample, per-class recall 계산
   - `intersection = (pred * skel_gt).sum(dim=2)`
   - `gt_sum = skel_gt.sum(dim=2)`
   - `valid = gt_sum > 0`
   - `recall = intersection / (gt_sum + eps)`
5. `valid`인 항목만 평균
6. 유효 skeleton이 하나도 없으면 `pred.new_tensor(0.0)` 반환
7. 최종 loss는 `loss_weight * (1 - mean_recall)`

중요:
- empty skeleton 채널은 loss에 기여하지 않아야 한다.
- reduction 기준은 **per-sample, per-class valid mask**다. 배치 전체를 한 번에 합치지 않는다.

---

### Step 5: loss registry 등록

**파일:** `plugin/models/losses/__init__.py`

```python
from .seg_loss import MaskFocalLoss, MaskDiceLoss, SkelRecallLoss
```

---

### Step 6: `MapSegHead` 수정

**파일:** `plugin/models/heads/MapSegHead.py`

#### 6a. `__init__`에 `loss_skel=None` 추가

```python
def __init__(self,
             num_classes=3,
             in_channels=256,
             embed_dims=256,
             bev_size=(100, 50),
             canvas_size=(200, 100),
             loss_seg=dict(),
             loss_dice=dict(),
             loss_skel=None):
```

초기화 규칙:
```python
self.loss_seg = build_loss(loss_seg)
self.loss_dice = build_loss(loss_dice)
self.loss_skel = build_loss(loss_skel) if loss_skel is not None else None
self.use_skel_loss = self.loss_skel is not None
```

#### 6b. `forward_train` 시그니처 확장

```python
def forward_train(self, bev_features, gts, history_coords, skel_gts=None):
```

#### 6c. skeleton loss 계산 추가

```python
seg_loss = self.loss_seg(preds, gts)
dice_loss = self.loss_dice(preds, gts)

if self.use_skel_loss and skel_gts is not None:
    skel_loss = self.loss_skel(preds, skel_gts)
else:
    skel_loss = preds.new_tensor(0.0)
```

#### 6d. return contract 확장

기존:
```python
return preds, seg_feats, seg_loss, dice_loss
```

변경:
```python
return preds, seg_feats, seg_loss, dice_loss, skel_loss
```

주의:
- `forward_test()`는 **그대로 2-tuple 유지**한다.
- 이 변경을 안전하게 만들기 위해 **Step 7과 Step 8에서 모든 train caller를 함께 수정**한다.

---

### Step 7: `MapTracker.forward_train`도 함께 수정

**파일:** `plugin/models/mapers/MapTracker.py`

`MapSegHead`의 shared caller이므로 반드시 같이 수정한다.

#### 7a. 시그니처 확장

```python
def forward_train(self, img, vectors, semantic_mask, skeleton_mask=None, points=None,
                  img_metas=None, all_prev_data=None, all_local2global_info=None, **kwargs):
```

#### 7b. prev frame skeleton 수집

`all_prev_data` 처리부에서 다음 리스트를 추가한다.

```python
all_skeleton_mask_prev = []
...
all_skeleton_mask_prev.append(prev_data.get('skeleton_mask', None))
```

#### 7c. current frame skeleton 준비

```python
gt_semantic = torch.flip(semantic_mask, [2,])
gt_skeleton = torch.flip(skeleton_mask, [2,]) if skeleton_mask is not None else None
```

#### 7d. prev frame seg decoder 호출 변경

```python
gts_semantic_prev = torch.flip(all_semantic_mask_prev[t], [2,])
gts_skeleton_prev = (
    torch.flip(all_skeleton_mask_prev[t], [2,])
    if all_skeleton_mask_prev[t] is not None else None
)

seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
    bev_feats, gts_semantic_prev, all_history_coord,
    skel_gts=gts_skeleton_prev, return_loss=True)
```

#### 7e. current frame seg decoder 호출 변경

```python
seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
    bev_feats, gt_semantic, all_history_coord,
    skel_gts=gt_skeleton, return_loss=True)
```

#### 7f. loss dict 반영

prev/current 모두 다음 키를 추가한다.

```python
loss_dict_prev['seg_skel'] = seg_skel_loss
loss_dict['seg_skel'] = seg_skel_loss
```

효과:
- 기존 aggregation / `log_vars` 루프는 dict 순회 기반이므로 `seg_skel`을 자동 합산/로깅한다.

---

### Step 8: `SatMapTracker.forward_train` 수정

**파일:** `plugin/models/mapers/SatMapTracker.py`

`MapTracker`와 동일한 skeleton data plumbing을 satellite path에도 반영한다.

#### 8a. 시그니처 확장

```python
def forward_train(self, img, vectors, semantic_mask, skeleton_mask=None, points=None,
                  img_metas=None, all_prev_data=None, all_local2global_info=None,
                  sat_img=None, **kwargs):
```

#### 8b. prev frame skeleton 수집

```python
all_skeleton_mask_prev = []
...
all_skeleton_mask_prev.append(prev_data.get('skeleton_mask', None))
```

#### 8c. current frame skeleton 준비

```python
gt_semantic = torch.flip(semantic_mask, [2,])
gt_skeleton = torch.flip(skeleton_mask, [2,]) if skeleton_mask is not None else None
```

#### 8d. prev frame / current frame seg decoder 호출을 5-tuple로 변경

```python
seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
    bev_feats, gts_semantic_prev, all_history_coord,
    skel_gts=gts_skeleton_prev, return_loss=True)
```

```python
seg_preds, seg_feats, seg_loss, seg_dice_loss, seg_skel_loss = self.seg_decoder(
    bev_feats, gt_semantic, all_history_coord,
    skel_gts=gt_skeleton, return_loss=True)
```

#### 8e. loss dict 반영

```python
loss_dict_prev['seg_skel'] = seg_skel_loss
loss_dict['seg_skel'] = seg_skel_loss
```

---

### Step 9: sat config 수정

**파일:** `plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py`

#### 9a. train pipeline에 `SkeletonizeMap` 추가

`RasterizeMap` 바로 뒤, `LoadMultiViewImagesFromFiles` 앞에 삽입:

```python
dict(
    type='SkeletonizeMap',
    skel_classes=(1, 2),
    num_dilations=2,
    kernel_size=3,
),
```

#### 9b. `Collect3D` keys에 `skeleton_mask` 추가

기존:
```python
keys=['img', 'vectors', 'semantic_mask', 'sat_img']
```

변경:
```python
keys=['img', 'vectors', 'semantic_mask', 'skeleton_mask', 'sat_img']
```

#### 9c. `seg_cfg`에 `loss_skel` 추가

```python
loss_skel=dict(
    type='SkelRecallLoss',
    loss_weight=1.0,
    skel_classes=(1, 2),
    eps=1e-6,
),
```

#### 9d. test pipeline / eval config는 변경하지 않음

이유:
- `skeleton_mask`는 학습 loss용이다.
- `forward_test()`는 segmentation prediction만 사용하므로 skeleton GT가 필요 없다.

---

## 변경하지 않는 파일 / 범위

다음은 **이번 작업에서 수정하지 않는다**:
- `plugin/datasets/base_dataset.py`
  - 이유: `all_prev_data`는 이미 같은 pipeline 결과를 재사용하므로 `Collect3D`만 바꾸면 `skeleton_mask`가 따라간다.
- `test_pipeline`
- `eval_config`
- dataset pkl 생성 로직
- satellite encoder / vector head 관련 로직

---

## 실행 순서

1. **Step 1~3**: skeleton data 생성/등록/포맷 경로 완성
2. **Step 4~5**: `SkelRecallLoss` 추가 및 registry 등록
3. **Step 6**: `MapSegHead` 수정
4. **Step 7~8**: `MapTracker` + `SatMapTracker` caller 동시 수정
5. **Step 9**: target sat config에만 `loss_skel` / `skeleton_mask` 활성화
6. **검증**: unit smoke -> target sat config 1 iter -> non-sat config 1 iter

주의:
- Step 6만 먼저 반영하고 caller를 나중에 두면 train path가 깨질 수 있으므로,
  **MapSegHead 변경과 caller 수정은 같은 작업 묶음으로 처리**한다.

---

## 검증 계획

### A. import / transform smoke

1. transform import
```bash
python -c "from plugin.datasets.pipelines.skeletonize import SkeletonizeMap; print('OK')"
```

2. loss import
```bash
python -c "from plugin.models.losses.seg_loss import SkelRecallLoss; print('OK')"
```

3. dummy mask transform smoke
- class 1/2 선형 mask를 만든 뒤 `SkeletonizeMap` 적용
- 출력 shape가 `(3, H, W)` 유지되는지 확인
- 출력 dtype이 `uint8`인지 확인
- class 0 채널이 zero인지 확인

### B. loss unit smoke

4. `SkelRecallLoss` empty-skeleton guard 확인
- 모든 `skel_gt == 0`일 때 loss가 crash 없이 `0.0`인지 확인

5. `loss_skel=None` 경로 확인
- `MapSegHead(loss_skel=None)`로 forward_train 호출 시 `seg_skel == 0`이고 기존 loss 계산이 유지되는지 확인

### C. integration smoke

6. target sat config 1-iter train
- 대상 config:
  - `plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py`
- 확인 항목:
  - dataloader가 `skeleton_mask`를 current / prev frame 모두 전달하는지
  - `log_vars`에 `seg_skel`, `seg_skel_t0`, ... 가 찍히는지
  - crash 없이 1 iteration 완료되는지

7. non-sat config 1-iter train
- 대상 config 예시:
  - `plugin/configs/maptracker/nuscenes_newsplit/maptracker_nusc_newsplit_5frame_span10_stage1_bev_pretrain.py`
- 확인 항목:
  - `MapTracker` train path가 5-tuple 변경 이후에도 정상 동작하는지
  - `loss_skel`이 없는 config에서 `seg_skel == 0`으로 안전하게 처리되는지

### D. sanity checks

8. value range sanity
- `seg_skel`이 음수가 아닌지 확인
- 초기 학습에서 NaN / inf가 없는지 확인

9. regression sanity
- `loss_skel` 블록을 config에서 제거했을 때 기존 BEV pretrain 경로가 깨지지 않는지 확인

---

## 구현 시 주의사항

1. `MapSegHead.forward_test()` return shape는 바꾸지 않는다.
2. `seg_skel`는 별도 키로 loss dict에 넣어 log 분리를 유지한다.
3. `skeleton_mask`가 없는 batch / config는 반드시 zero-loss fallback으로 처리한다.
4. prev frame에서도 `skeleton_mask`를 `semantic_mask`와 동일하게 `torch.flip(..., [2,])` 처리한다.
5. skeleton loss는 class 1, 2만 대상으로 한다. class 0은 zero 채널로 유지한다.

---

## 비목표 (Non-goals)

- skeleton supervision을 inference에 사용하기
- test/eval pipeline에 skeleton GT를 넣기
- dataset serialization 포맷 변경
- 새로운 third-party package 추가

---

## 기대 결과

이 계획대로 구현되면:
- sat pretrain에서 `divider / boundary` 중심선 recall을 직접 supervision할 수 있고,
- 기존 `MapTracker` / `SatMapTracker` train path 모두 shared head 변경과 함께 안전하게 유지되며,
- `loss_skel=None` 경로를 통해 기존 config와의 호환성도 보존된다.
