"""
WandB Visualization Hook for MapTracker evaluation.

After each validation epoch, logs GT vs Pred BEV visualizations to wandb.
Samples one scene per city and renders GT (solid) vs Pred (dashed) overlaid.
"""
import os
import os.path as osp
import numpy as np
import mmcv
import torch.distributed as dist
from mmcv.runner import Hook, HOOKS

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


CLASS_COLORS = {
    0: (255, 107, 107),   # ped_crossing - red
    1: (78, 205, 196),    # divider - teal
    2: (255, 215, 0),     # boundary - yellow
}

CLASS_NAMES = {0: 'ped_crossing', 1: 'divider', 2: 'boundary'}


def render_bev_comparison(gt_vectors, pred_dict, roi_size=(60, 30),
                          canvas_size=(800, 400), score_thr=0.3):
    """Render GT vs Pred on a single BEV canvas.

    Args:
        gt_vectors: dict {label_id: [list of (N,2) arrays]}
        pred_dict: dict with 'vectors', 'scores', 'labels'
        roi_size: (x_range, y_range) in meters
        canvas_size: (width, height) in pixels

    Returns:
        np.ndarray: RGB image (H, W, 3)
    """
    import cv2

    W, H = canvas_size
    canvas = np.zeros((H, W, 3), dtype=np.uint8) + 40  # dark background

    def world_to_pixel(pts):
        """Convert BEV world coords to pixel coords."""
        px = (pts[:, 0] + roi_size[0] / 2) / roi_size[0] * W
        py = (1.0 - (pts[:, 1] + roi_size[1] / 2) / roi_size[1]) * H
        return np.stack([px, py], axis=1).astype(np.int32)

    # Draw GT (solid lines, thicker)
    for label_id, vectors in gt_vectors.items():
        color = CLASS_COLORS.get(label_id, (200, 200, 200))
        for vec in vectors:
            if len(vec) < 2:
                continue
            pts = world_to_pixel(np.array(vec))
            cv2.polylines(canvas, [pts], isClosed=False, color=color,
                          thickness=3, lineType=cv2.LINE_AA)

    # Draw Pred (dashed lines, thinner)
    if pred_dict and 'vectors' in pred_dict:
        for i, vec in enumerate(pred_dict['vectors']):
            score = pred_dict['scores'][i]
            if score < score_thr:
                continue
            label = pred_dict['labels'][i]
            color = CLASS_COLORS.get(label, (200, 200, 200))
            # Make pred color slightly brighter
            bright = tuple(min(255, c + 60) for c in color)

            vec = np.array(vec)
            if len(vec) < 2:
                continue
            pts = world_to_pixel(vec)

            # Draw dashed line
            for j in range(len(pts) - 1):
                if j % 2 == 0:  # dash pattern
                    cv2.line(canvas, tuple(pts[j]), tuple(pts[j + 1]),
                             bright, 2, cv2.LINE_AA)

    # Draw ego position (center)
    cx, cy = W // 2, H // 2
    cv2.drawMarker(canvas, (cx, cy), (255, 255, 255),
                   cv2.MARKER_STAR, 15, 2)

    # Legend
    y_off = 20
    for lid, name in CLASS_NAMES.items():
        color = CLASS_COLORS[lid]
        cv2.rectangle(canvas, (10, y_off - 12), (25, y_off + 2), color, -1)
        cv2.putText(canvas, name, (30, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        y_off += 20

    cv2.putText(canvas, "solid=GT, dash=Pred", (10, y_off + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

    return canvas


@HOOKS.register_module()
class WandbVisHook(Hook):
    """Log GT vs Pred BEV visualizations to wandb after each eval.

    Args:
        eval_config (dict): dataset config for loading GT
        num_vis_samples (int): number of samples to visualize per eval
        score_thr (float): minimum score for pred visualization
        roi_size (tuple): BEV range
    """

    def __init__(self, eval_config, num_vis_samples=8, score_thr=0.3,
                 roi_size=(60, 30)):
        self.eval_config = eval_config
        self.num_vis_samples = num_vis_samples
        self.score_thr = score_thr
        self.roi_size = roi_size
        self._evaluator = None

    @property
    def evaluator(self):
        if self._evaluator is None:
            from plugin.datasets.evaluation.vector_eval import VectorEvaluate
            self._evaluator = VectorEvaluate(self.eval_config, n_workers=0)
        return self._evaluator

    def after_val_epoch(self, runner):
        if not HAS_WANDB or wandb.run is None:
            return

        rank, _ = dist.get_rank(), dist.get_world_size()
        if rank != 0:
            return

        # Find the latest submission file
        result_path = osp.join(runner.work_dir, 'submission_vector.json')
        if not osp.exists(result_path):
            return

        results = mmcv.load(result_path)
        results = results.get('results', {})
        if not results:
            return

        gts = self.evaluator.gts

        # Sample tokens to visualize
        tokens = list(results.keys())
        if len(tokens) > self.num_vis_samples:
            indices = np.linspace(0, len(tokens) - 1,
                                  self.num_vis_samples, dtype=int)
            tokens = [tokens[i] for i in indices]

        images = []
        for token in tokens:
            gt = gts.get(token, {})
            pred = results.get(token, {})

            canvas = render_bev_comparison(
                gt, pred,
                roi_size=self.roi_size,
                score_thr=self.score_thr)

            images.append(wandb.Image(
                canvas,
                caption=f"iter={runner.iter} | {token[:16]}..."
            ))

        wandb.log({
            "val/gt_vs_pred": images,
            "global_step": runner.iter,
        })
