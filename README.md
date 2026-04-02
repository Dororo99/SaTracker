<div align="center">
<h2 align="center"> MapTracker: Tracking with Strided Memory Fusion for <br/> Consistent Vector HD Mapping </h1>

<h4 align="center"> ECCV 2024 (Oral) </h4>


[Jiacheng Chen*<sup>1</sup>](https://jcchen.me) , [Yuefan Wu*<sup>1</sup>](https://ivenwu.com/) , [Jiaqi Tan*<sup>1</sup>](https://www.linkedin.com/in/jiaqi-christina-tan-800697158/), [Hang Ma<sup>1</sup>](https://www.cs.sfu.ca/~hangma/), [Yasutaka Furukawa<sup>1,2</sup>](https://www2.cs.sfu.ca/~furukawa/)

<sup>1</sup> Simon Fraser University <sup>2</sup> Wayve


([arXiv](https://arxiv.org/abs/2403.15951), [Project page](https://map-tracker.github.io/))

</div>



https://github.com/woodfrog/maptracker/assets/13405255/1c0e072a-cb77-4000-b81b-5b9fd40f8f39




This repository provides the official implementation of the paper [MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping](https://arxiv.org/abs/2403.15951). MapTracker reconstructs temporally consistent vector HD maps, and the local maps can be progressively merged into a global reconstruction.

This repository is built upon [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet). 


## Table of Contents
- [Introduction](#introduction)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Data preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)

## Introduction
This paper presents a vector HD-mapping algorithm that formulates the mapping as a tracking task and uses a history of memory latents to ensure consistent reconstructions over time.

Our method, MapTracker, accumulates a sensor stream into memory buffers of two latent representations: 1) Raster latents in the bird's-eye-view (BEV) space and 2) Vector latents over the road elements (i.e., pedestrian-crossings, lane-dividers, and road-boundaries). The approach borrows the query propagation paradigm from the tracking literature that explicitly associates tracked road elements from the previous frame to the current, while fusing a subset of memory latents selected with distance strides to further enhance temporal consistency. A vector latent is decoded to reconstruct the geometry of a road element.

The paper further makes benchmark contributions by 1) Improving processing code for existing datasets to produce consistent ground truth with temporal alignments and 2) Augmenting existing mAP metrics with consistency checks. MapTracker significantly outperforms existing methods on both nuScenes and Agroverse2 datasets by over 8% and 19% on the conventional and the new consistency-aware metrics, respectively.


## Model Architecture

![visualization](docs/fig/arch.png)

(Top) The architecture of MapTracker, consistsing of the BEV and VEC Modules and their memory buffers. (Bottom) The close-up views of the BEV and the vector fusion layers.

The **BEV Module** takes ConvNet features of onboard perspective images, the BEV memory buffer ${M_{\text{BEV}}(t-1), M_{\text{BEV}}(t-2),\ ... }$ and vehicle motions ${P^t_{t-1}, P^t_{t-2},\ ... }$ as input. It propagates the previous BEV memory $M_{\text{BEV}}(t-1)$ based on vehicle motion to initialize $M_{\text{BEV}}(t)$. In the BEV Memory Fusion layer, $M_{\text{BEV}}(t)$ is integrated with selected history BEV memories $\{M_{\text{BEV}}^{*}(t'), t'\in \pi(t)\}$, which is used for semantic segmentation and passed to the VEC Module.

The **VEC Module** propagates the previous latent vector memory $M_{\text{VEC}}(t-1)$ with a PropMLP to initialize the vector queries $M_{\text{VEC}}(t)$. In Vector Memory Fusion layer, each propagated $M_{\text{VEC}}(t)$ is fused with its selected history vector memories $\{M_{\text{VEC}}^{*}(t'), t' \in \pi(t)\}$. The final vector latents are decoded to reconstruct the road elements.


## Installation

Please refer to the [installation guide](docs/installation.md) to set up the environment.


## Data preparation

For how to download and prepare data for the nuScenes and Argoverse2 datasets, as well as downloading our checkpoints, please see the [data preparation guide](docs/data_preparation.md). 


## Getting Started

For instructions on how to run training, inference, evaluation, and visualization, please follow [getting started guide](docs/getting_started.md).


## Satellite-Augmented MapTracker with Skeleton-Recall Loss

This fork extends MapTracker with two key additions:

### 1. SatMAE Satellite Fusion

BEVFormer encoder에 위성 영상(AID4AD) cross-attention을 추가하여 camera-only 대비 segmentation 성능을 향상시킵니다. SatMAE(ViT-L) pretrained encoder로 위성 feature를 추출하고, learnable sigmoid gate로 위성 정보 반영 비율을 제어합니다.

### 2. Skeleton-Recall Loss (Class-Selective)

Divider와 boundary는 가늘고 긴 선형 구조이므로, 모델이 선의 중심축(skeleton)을 놓치지 않도록 강제하는 auxiliary loss를 추가했습니다.

**핵심 아이디어:**
- GT segmentation mask에서 `skimage.morphology.skeletonize()`로 중심선을 추출하고 2px dilation 적용
- 모델 prediction이 해당 skeleton 위치를 빠뜨리지 않도록 recall을 최대화
- **Divider(class 1)와 boundary(class 2)에만 적용**, ped_crossing(class 0)은 면적형 구조이므로 제외

**Loss 수식:**
```
SkelRecallLoss = 1 - (1/|valid_classes|) * sum_{c in {1,2}} [ sum(sigmoid(pred_c) * skel_c) / (sum(skel_c) + eps) ]
```

**전체 Loss 구성 (Stage1 BEV Pretrain):**
```
Total = FocalLoss(all classes) + DiceLoss(all classes) + SkelRecallLoss(divider, boundary only)
```

| Loss | 역할 | 적용 클래스 |
|------|------|------------|
| MaskFocalLoss | 픽셀 분류 정확도 (hard example 집중) | 전체 |
| MaskDiceLoss | 영역 overlap 최대화 | 전체 |
| SkelRecallLoss | 선의 중심축 recall 강제 (연결성 보존) | divider, boundary만 |

### Training

```bash
# 기본 실행 (Skeleton Loss ON, weight=1.0)
bash scripts/train_satmaptracker_stage1_skeleton.sh

# Skeleton Loss OFF (baseline)
bash scripts/train_satmaptracker_stage1_skeleton.sh --no-skeleton

# Skeleton Loss weight 조절
bash scripts/train_satmaptracker_stage1_skeleton.sh --skel-weight 2.0
```

### Shell Script Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--no-skeleton` | (skeleton ON) | Skeleton-Recall Loss 비활성화 |
| `--skel-weight` | `1.0` | Skeleton loss weight (0.5, 1.0, 2.0 등) |
| `--gpus` | `2,3` | 사용할 GPU (e.g., `0,1,2,3`) |
| `--port` | `29570` | 분산 학습 master port |
| `--wandb-name` | `...-dohyun` | WandB run 이름 |
| `--wandb-project` | `Third-SatMAE_...` | WandB 프로젝트 이름 |
| `--bev-vis-interval` | `500` | BEV 시각화 WandB 로깅 주기 (iter) |
| `--config` | `satmaptracker_stage1_...` | Config 파일 경로 |

### WandB Logging

학습 중 아래 항목이 WandB에 자동 로깅됩니다:

**Scalar Metrics:**
- `seg`, `seg_dice`, `seg_skel` : 현재 프레임 loss
- `seg_t0`, `seg_dice_t0`, `seg_skel_t0` : temporal frame별 loss
- `sat_gate_L0`, `sat_gate_L1` : 위성 gate 값 (sigmoid, 0=닫힘, 1=열림)

**BEV Images (500 iter 간격):**
- `BEV/pred` : BEV segmentation prediction
- `BEV/gt` : BEV segmentation ground truth
- `BEV/skeleton_gt` : Skeleton ground truth (divider, boundary)

### Config에서 Skeleton Loss 직접 제어

Shell script 대신 config 파일에서 직접 on/off 가능합니다:

```python
# ON (seg_cfg 내부에 추가)
loss_skel=dict(
    type='SkelRecallLoss',
    loss_weight=1.0,
    skel_classes=[1, 2],
),

# OFF (loss_skel 제거하면 자동으로 비활성화)
```

### 추가된 파일 목록

| 파일 | 설명 |
|------|------|
| `plugin/datasets/pipelines/skeletonize.py` | SkeletonizeMap transform (skeleton GT 생성) |
| `plugin/models/losses/seg_loss.py` | SkelRecallLoss 클래스 |
| `plugin/core/hooks/wandb_bev_vis_hook.py` | BEV Pred/GT WandB 시각화 hook |
| `scripts/train_satmaptracker_stage1_skeleton.sh` | 학습 실행 스크립트 |


## Acknowledgements

We're grateful to the open-source projects below, their great work made our project possible:

* BEV perception: [BEVFormer](https://github.com/fundamentalvision/BEVFormer) ![GitHub stars](https://img.shields.io/github/stars/fundamentalvision/BEVFormer.svg?style=flat&label=Star)
* Vector HD mapping: [StreamMapNet](https://github.com/yuantianyuan01/StreamMapNet) ![GitHub stars](https://img.shields.io/github/stars/yuantianyuan01/StreamMapNet.svg?style=flat&label=Star), [MapTR](https://github.com/hustvl/MapTR) ![GitHub stars](https://img.shields.io/github/stars/hustvl/MapTR.svg?style=flat&label=Star)


## Citation

If you find MapTracker useful in your research or applications, please consider citing:

```
@inproceedings{chen2024maptrakcer,
  author  = {Chen, Jiacheng and Wu, Yuefan and Tan, Jiaqi and Ma, Hang and Furukawa, Yasutaka},
  title   = {MapTracker: Tracking with Strided Memory Fusion for Consistent Vector HD Mapping},
  journal = {arXiv preprint arXiv:2403.15951},
  year    = {2024}
}
```

## License

This project is licensed under GPL, see the [license file](LICENSE) for details.
