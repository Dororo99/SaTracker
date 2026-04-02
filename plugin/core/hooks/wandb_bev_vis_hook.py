import numpy as np
import torch
from mmcv.runner import HOOKS, Hook

try:
    import wandb
except ImportError:
    wandb = None


# BGR colors for each class
CLASS_COLORS = {
    0: np.array([255, 100, 100]),   # ped_crossing - blue
    1: np.array([100, 255, 100]),   # divider - green
    2: np.array([100, 100, 255]),   # boundary - red
}


def _bev_feat_to_heatmap(feat_tensor):
    """Convert (C, H, W) BEV feature to (H, W, 3) heatmap via L2 norm.

    Args:
        feat_tensor: (C, H, W) feature tensor.
    Returns:
        heatmap: (H, W, 3) uint8 numpy array (blue=low, red=high).
    """
    norm = feat_tensor.norm(dim=0).cpu().numpy()  # (H, W)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    norm = (norm * 255).astype(np.uint8)

    # Apply colormap: blue(low) -> green(mid) -> red(high)
    heatmap = np.zeros((*norm.shape, 3), dtype=np.uint8)
    heatmap[..., 0] = norm            # R
    heatmap[..., 1] = 255 - norm      # G (inverse)
    heatmap[..., 2] = 255 - norm      # B (inverse)
    return heatmap


def _seg_to_rgb(seg_tensor, threshold=0.5):
    """Convert (C, H, W) segmentation tensor to (H, W, 3) RGB image.

    Args:
        seg_tensor: (C, H, W) tensor (sigmoid probabilities or binary mask).
        threshold: threshold for converting probabilities to binary.
    Returns:
        rgb: (H, W, 3) uint8 numpy array.
    """
    seg = seg_tensor.cpu().numpy()
    C, H, W = seg.shape
    rgb = np.ones((H, W, 3), dtype=np.uint8) * 40  # dark gray background

    for cls_idx in range(min(C, 3)):
        mask = seg[cls_idx] > threshold
        if mask.any():
            rgb[mask] = CLASS_COLORS.get(cls_idx, np.array([200, 200, 200]))

    return rgb


@HOOKS.register_module()
class WandbBEVVisHook(Hook):
    """Periodically log BEV segmentation Pred vs GT to WandB.

    Args:
        interval (int): Logging interval in iterations. Default: 500.
        vis_skeleton (bool): Also visualize skeleton GT. Default: True.
    """

    def __init__(self, interval=500, vis_skeleton=True):
        self.interval = interval
        self.vis_skeleton = vis_skeleton

    def after_train_iter(self, runner):
        if wandb is None or wandb.run is None:
            return
        if not self.every_n_iters(runner, self.interval):
            return

        model = runner.model
        if hasattr(model, 'module'):
            model = model.module

        if not hasattr(model, '_vis_seg_preds'):
            return

        seg_preds = model._vis_seg_preds  # (B, C, H, W) logits
        gt_semantic = model._vis_gt_semantic  # (B, C, H, W) binary

        # Take first sample in batch
        pred_prob = seg_preds[0].sigmoid()  # (C, H, W)
        gt = gt_semantic[0].float()         # (C, H, W)

        pred_rgb = _seg_to_rgb(pred_prob)
        gt_rgb = _seg_to_rgb(gt)

        images = {
            'BEV/pred': wandb.Image(pred_rgb, caption='Prediction'),
            'BEV/gt': wandb.Image(gt_rgb, caption='Ground Truth'),
        }

        if self.vis_skeleton and hasattr(model, '_vis_gt_skeleton'):
            gt_skel = model._vis_gt_skeleton
            if gt_skel is not None:
                skel_rgb = _seg_to_rgb(gt_skel[0].float())
                images['BEV/skeleton_gt'] = wandb.Image(skel_rgb, caption='Skeleton GT')

        # BEV feature before/after ConvFusion
        if hasattr(model, '_vis_bev_pre_fusion') and hasattr(model, '_vis_bev_post_fusion'):
            pre = model._vis_bev_pre_fusion[0]   # (C, H, W)
            post = model._vis_bev_post_fusion[0]  # (C, H, W)
            images['BEV/feat_pre_fusion'] = wandb.Image(
                _bev_feat_to_heatmap(pre), caption='Before ConvFusion')
            images['BEV/feat_post_fusion'] = wandb.Image(
                _bev_feat_to_heatmap(post), caption='After ConvFusion')

            # Difference heatmap (where fusion changed the most)
            diff = (post - pre).norm(dim=0).cpu().numpy()
            diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
            diff = (diff * 255).astype(np.uint8)
            diff_rgb = np.stack([diff, diff, diff], axis=-1)
            images['BEV/fusion_diff'] = wandb.Image(
                diff_rgb, caption='Fusion Difference (bright=large change)')

        wandb.log(images, step=runner.iter)
