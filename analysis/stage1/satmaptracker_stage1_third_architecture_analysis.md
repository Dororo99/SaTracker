# SatMapTracker Stage1(Third Split) 학습 원리/아키텍처 분석

## 1) 대상 설정과 상속 구조
- 분석 대상: `plugin/configs/satmaptracker/nuscenes_newsplit/satmaptracker_stage1_third.py`
- 이 파일은 `_base_ = ['./satmaptracker_stage1.py']`로 **대부분의 모델/학습 로직을 base config에서 상속**하고, third split용 데이터/스케줄/로깅만 override한다.
  - 근거: `satmaptracker_stage1_third.py:1, 14-31, 33-37, 42-53`

즉, 실제 학습 원리와 아키텍처는 `satmaptracker_stage1.py` + `SatMapTracker/MapTracker` 구현을 같이 봐야 정확하다.

---

## 2) 핵심 한 줄 요약
이 설정은 **카메라 BEV를 시간축(history)으로 안정적으로 학습**하고, **현재 프레임에서만 위성 BEV를 융합**해서 성능을 올리는 구조다. 학습 시에는
1) 카메라 분기(dense temporal supervision),
2) 융합 분기(main),
3) 위성 단독 분기(aux)
를 동시에 supervision하는 **triple supervision**을 사용한다.

---

## 3) 학습 원리 (Training Principle)

### 3-1. Stage1 목적: 벡터 헤드 스킵 + BEV segmentation pretrain
- `skip_vector_head=True`로 설정되어, 벡터 검출/트래킹 헤드 손실은 Stage1에서 사실상 비활성화된다.
  - 근거: `satmaptracker_stage1.py:79`
  - 코드상 `skip_vector_head=True`이면 `MapTracker.forward_train`에서 `loss_dict = {}`로 두고 seg loss만 유지: `MapTracker.py:745-778`

### 3-2. Temporal history는 `history_mode='cam'`
- SatMapTracker는 `history_mode='cam'`만 허용(assert)한다.
- 의미: 시간 메모리에는 **순수 cam_bev만** 쌓고, satellite fusion은 현재 프레임에서만 적용한다.
  - 근거: `satmaptracker_stage1.py:86-90`, `SatMapTracker.py:71-77, 100-104`
- 의도: fused feature를 history에 넣으면 TemporalSelfAttention 경로로 fusion drift가 카메라 인코더에 역전파되는 문제를 피하려는 설계.

### 3-3. Triple supervision 구성
SatMapTracker 학습은 2단으로 동작:

1. **부모(MapTracker) 학습 루프 실행**
   - 이전 프레임들 + 현재 프레임의 cam_bev에 대해 `self.seg_decoder`로 seg loss를 누적.
   - 현재 프레임 cam BEV는 `self._last_bev_feats`에 저장.
   - 근거: `MapTracker.py:579-633, 739-743, 780-810`

2. **현재 프레임 추가 supervision (SatMapTracker 확장)**
   - `sat_encoder(sat_img) -> sat_bev`
   - `sat_fusion(cam_bev, sat_bev) -> fused_bev -> seg_decoder_fused` (메인 분기)
   - `sat_bev -> seg_decoder_sat` (보조 분기)
   - 두 분기 loss를 부모 loss에 더한다.
   - 근거: `SatMapTracker.py:106-160`

정리하면 전체 loss는 개념적으로 아래 형태:

\[
L_{total} = L_{cam\_temporal} + L_{fused\_seg} + L_{fused\_dice} + L_{sat\_seg} + L_{sat\_dice}
\]

> 참고: `MapTracker.py` 주석은 “average”라고 쓰지만 실제 코드는 프레임별 loss를 합산(sum)한다 (`MapTracker.py:780-789`).

### 3-4. Inference 시 메인 출력
- 부모가 만든 cam 분기 결과를 `semantic_mask_cam`으로 보존하고,
- fused 분기 결과를 `semantic_mask`에 overwrite하여 메인 평가 마스크로 사용,
- sat 단독 분기는 `semantic_mask_sat`으로 별도 저장.
- 근거: `SatMapTracker.py:188-230`

---

## 4) 아키텍처 (Architecture)

## 4-1. 전체 흐름
1. 멀티뷰 카메라(6대) -> BEVFormerBackbone -> cam BEV (`256 x 50 x 100`)
2. AID4AD 위성 이미지 -> SatelliteEncoder -> sat BEV (`256 x 50 x 100`)
3. SatelliteConvFuser로 cam/sat 융합 -> fused BEV
4. Seg head 3개가 각 branch를 감독
   - cam: `self.seg_decoder` (부모)
   - fused: `self.seg_decoder_fused`
   - sat: `self.seg_decoder_sat`

### 4-2. Camera branch (BEVFormer)
- 백본: ResNet50(caffe) + FPN + PerceptionTransformer(BEVFormerEncoder)
- Encoder attention: TemporalSelfAttention + SpatialCrossAttention
- 근거: `satmaptracker_stage1.py:116-176`

### 4-3. Satellite branch (`SatelliteEncoder`)
- ResNet50 feature pyramid를 받아 lateral 1x1 conv 후 upsample/sum하는 FPN-like 집계
- 최종 conv 후 target BEV size로 resize
- 근거: `satellite_encoder.py:55-107`

### 4-4. Fusion (`SatelliteConvFuser`)
- cam/sat 각각 1x1 projection -> concat -> conv block
- `use_residual=True`면 최종 fused에 cam residual 더함
- 근거: `satmaptracker_stage1.py:109-114`, `satellite_fusion.py:140-163`

### 4-5. Segmentation head (`MapSegHead`)
- 입력 BEV에 conv + upsample block을 거쳐 class logits 생성
- loss는 `MaskFocalLoss` + `MaskDiceLoss`
- 근거: `satmaptracker_stage1.py:229-238`, `MapSegHead.py:35-45, 71-79`

---

## 5) 데이터/학습 설정 (stage1_third 기준)

### 5-1. Third split override
- train/val/test ann_file이 `_third.pkl`로 바뀜
- 근거: `satmaptracker_stage1_third.py:16-29`

### 5-2. 배치/스케줄
- `num_gpus=4`, `batch_size=3` -> global batch 12
- `num_iters_per_epoch = 9274 // 12 = 772`
- `num_epochs=18` -> `total_iters=13896`
- eval/checkpoint interval: `3 epoch`마다 (`2316 iters`)
- 근거: `satmaptracker_stage1_third.py:7-12, 33-37`

### 5-3. 입력 파이프라인
- 카메라: `LoadMultiViewImagesFromFiles` + photometric/resize/normalize/pad
- 위성: `LoadAID4ADSatelliteImage`
  - token 기반 lookup 테이블 생성/조회 후 RGB 변환, 로더 단계에서 세로축 flip, normalize 수행
  - 추가로 인코더 출력에서도 `torch.flip`을 한 번 더 적용해 BEV 축 정렬을 맞춤
- 근거: `satmaptracker_stage1.py:243-260`, `loading_sat.py:46-56, 61-63, 66-68, 82-89`, `SatMapTracker.py:90-99`

### 5-4. Optimizer/LR (base 상속)
- AdamW, lr=5e-4, weight_decay=1e-2
- `img_backbone`에만 `lr_mult=0.1`
- CosineAnnealing + warmup 500 iters
- 근거: `satmaptracker_stage1.py:349-364`

---

## 6) 주석과 실제 설정 간 체크 포인트
1. `satmaptracker_stage1.py` 상단 문자열에는 “no residual” 문구가 있으나, 실제 config는 `use_residual=True`다.
   - 근거: `satmaptracker_stage1.py:4` vs `satmaptracker_stage1.py:109-114`
2. `satmaptracker_stage1_third.py` 주석에는 “both backbones at lr_mult=0.1”라고 되어 있지만, base optimizer에는 `img_backbone`만 명시돼 있다.
   - 근거: `satmaptracker_stage1_third.py:40` vs `satmaptracker_stage1.py:353-355`
3. base 상단 문자열에는 “shared seg_decoder”라고 적혀 있지만, 실제 구현은 `seg_decoder`(cam) + `seg_decoder_fused` + `seg_decoder_sat`로 분리되어 있다.
   - 근거: `satmaptracker_stage1.py:5` vs `SatMapTracker.py:83-85, 145-154`
4. base 상단 문자열에는 “History: fused_bev”라고 적혀 있지만, 실제 동작은 `history_mode='cam'`으로 cam history만 유지한다.
   - 근거: `satmaptracker_stage1.py:6, 90` vs `SatMapTracker.py:71-77`

---

## 7) 결론
`satmaptracker_stage1_third.py`는 **base(Stage1) 설계를 그대로 사용하되 데이터 split만 third로 바꾼 실행 config**다. 실제 핵심은
- `history_mode='cam'`로 temporal stability 확보,
- 현재 프레임에서만 satellite fusion 적용,
- cam/fused/sat 3분기 segmentation supervision,
- Stage1에서는 vector head를 끄고 BEV pretrain에 집중
하는 데 있다.

따라서 이 실험의 본질은 “**카메라 기반 시간 문맥을 보존하면서 위성 정보를 현재 프레임 보정(delta)으로 학습시키는 BEV 사전학습(stage1)**”으로 해석할 수 있다.
