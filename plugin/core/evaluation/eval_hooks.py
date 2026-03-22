
# Note: Considering that MMCV's EvalHook updated its interface in V1.3.16,
# in order to avoid strong version dependency, we did not directly
# inherit EvalHook but BaseDistEvalHook.

import bisect
import os.path as osp

import mmcv
import numpy as np
import torch.distributed as dist
from mmcv.runner import DistEvalHook as BaseDistEvalHook
from mmcv.runner import EvalHook as BaseEvalHook
from torch.nn.modules.batchnorm import _BatchNorm
from mmdet.core.evaluation.eval_hooks import DistEvalHook

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def _calc_dynamic_intervals(start_interval, dynamic_interval_list):
    assert mmcv.is_list_of(dynamic_interval_list, tuple)

    dynamic_milestones = [0]
    dynamic_milestones.extend(
        [dynamic_interval[0] for dynamic_interval in dynamic_interval_list])
    dynamic_intervals = [start_interval]
    dynamic_intervals.extend(
        [dynamic_interval[1] for dynamic_interval in dynamic_interval_list])
    return dynamic_milestones, dynamic_intervals


class CustomDistEvalHook(BaseDistEvalHook):

    def __init__(self, *args, dynamic_intervals=None,
                 wandb_vis_samples=8, wandb_vis_score_thr=0.3, **kwargs):
        super(CustomDistEvalHook, self).__init__(*args, **kwargs)
        self.use_dynamic_intervals = dynamic_intervals is not None
        if self.use_dynamic_intervals:
            self.dynamic_milestones, self.dynamic_intervals = \
                _calc_dynamic_intervals(self.interval, dynamic_intervals)
        self.wandb_vis_samples = wandb_vis_samples
        self.wandb_vis_score_thr = wandb_vis_score_thr

    def _decide_interval(self, runner):
        if self.use_dynamic_intervals:
            progress = runner.epoch if self.by_epoch else runner.iter
            step = bisect.bisect(self.dynamic_milestones, (progress + 1))
            # Dynamically modify the evaluation interval
            self.interval = self.dynamic_intervals[step - 1]

    def before_train_epoch(self, runner):
        """Evaluate the model only at the start of training by epoch."""
        self._decide_interval(runner)
        super().before_train_epoch(runner)

    def before_train_iter(self, runner):
        self._decide_interval(runner)
        super().before_train_iter(runner)

    def _do_evaluate(self, runner):
        """perform evaluation and save ckpt."""
        # Synchronization of BatchNorm's buffer (running_mean
        # and running_var) is not supported in the DDP of pytorch,
        # which may cause the inconsistent performance of models in
        # different ranks, so we broadcast BatchNorm's buffers
        # of rank 0 to other ranks to avoid this.
        if self.broadcast_bn_buffer:
            model = runner.model
            for name, module in model.named_modules():
                if isinstance(module,
                              _BatchNorm) and module.track_running_stats:
                    dist.broadcast(module.running_var, 0)
                    dist.broadcast(module.running_mean, 0)

        if not self._should_evaluate(runner):
            return

        tmpdir = self.tmpdir
        if tmpdir is None:
            tmpdir = osp.join(runner.work_dir, '.eval_hook')

        from ..apis.test import custom_multi_gpu_test # to solve circlur  import

        results = custom_multi_gpu_test(
            runner.model,
            self.dataloader,
            tmpdir=tmpdir,
            gpu_collect=self.gpu_collect)
        
        if runner.rank == 0:
            print('\n')
            runner.log_buffer.output['eval_iter_num'] = len(self.dataloader)

            key_score = self.evaluate(runner, results)

            # Log GT vs Pred visualizations to wandb
            self._log_wandb_vis(runner)

            if self.save_best:
                self._save_ckpt(runner, key_score)

    def _log_wandb_vis(self, runner):
        """Log GT vs Pred BEV visualizations to wandb."""
        if not HAS_WANDB or wandb.run is None:
            return

        from .wandb_vis_hook import render_bev_comparison
        from plugin.datasets.evaluation.vector_eval import VectorEvaluate

        result_path = osp.join(runner.work_dir, 'submission_vector.json')
        if not osp.exists(result_path):
            return

        results = mmcv.load(result_path)
        results = results.get('results', {})
        if not results:
            return

        # Build evaluator to access GT (uses cache after first call)
        eval_cfg = self.dataloader.dataset.eval_config
        evaluator = VectorEvaluate(eval_cfg, n_workers=0)
        gts = evaluator.gts

        # Sample tokens evenly
        tokens = list(results.keys())
        n = min(self.wandb_vis_samples, len(tokens))
        indices = np.linspace(0, len(tokens) - 1, n, dtype=int)
        sampled_tokens = [tokens[i] for i in indices]

        roi_size = self.dataloader.dataset.roi_size

        images = []
        for token in sampled_tokens:
            gt = gts.get(token, {})
            pred = results.get(token, {})
            canvas = render_bev_comparison(
                gt, pred,
                roi_size=roi_size,
                score_thr=self.wandb_vis_score_thr)
            images.append(wandb.Image(
                canvas,
                caption=f"iter={runner.iter} | {token[:20]}"
            ))

        wandb.log({
            "val/gt_vs_pred": images,
            "global_step": runner.iter,
        })
        runner.logger.info(
            f"Logged {len(images)} GT vs Pred visualizations to wandb")

