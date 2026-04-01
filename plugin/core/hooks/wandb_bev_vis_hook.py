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

        wandb.log(images, step=runner.iter)
