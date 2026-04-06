import torch
import torch.nn.functional as F
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from functools import cached_property
import prettytable
import numpy as np
from numpy.typing import NDArray
from typing import Dict, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy

try:
    import cv2
except Exception:
    cv2 = None

N_WORKERS = 16

class RasterEvaluate(object):
    """Evaluator for rasterized map.

    Args:
        dataset_cfg (Config): dataset cfg for gt
        n_workers (int): num workers to parallel
    """

    def __init__(self, dataset_cfg: Config, n_workers: int=N_WORKERS):
        self.dataset = build_dataset(dataset_cfg)
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=n_workers, shuffle=False, dist=False)
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers

    @cached_property
    def gts(self) -> Dict[str, NDArray]:
        print('collecting gts...')
        gts = {}
        for data in mmcv.track_iter_progress(self.dataloader):
            token = deepcopy(data['img_metas'].data[0][0]['token'])
            gt = deepcopy(data['semantic_mask'].data[0][0])
            gts[token] = gt
            del data # avoid dataloader memory crash
        
        return gts

    def evaluate(self, 
                 result_path: str, 
                 logger: Optional[Logger]=None) -> Dict[str, float]:
        ''' Do evaluation for a submission file and print evalution results to `logger` if specified.
        The submission will be aligned by tokens before evaluation. 
        
        Args:
            result_path (str): path to submission file
            logger (Logger): logger to print evaluation result, Default: None
        
        Returns:
            result_dict (Dict): evaluation results. IoU by categories.
        '''
        
        results = mmcv.load(result_path)
        meta = results['meta']
        results = results['results']

        result_dict = {}

        gts = []
        preds = []
        for token, gt in self.gts.items():
            gts.append(gt)
            pred = torch.zeros((len(self.cat2id), gt.shape[1], gt.shape[2])).bool()
            if token in results:
                semantic_mask = torch.tensor(results[token]['semantic_mask'])
                for label_i in range(gt.shape[0]):
                    pred[label_i] = (semantic_mask == label_i+1)
            preds.append(pred)
        
        preds = torch.stack(preds).bool()
        gts = torch.stack(gts).bool()

        # TODO: flip the gt
        gts = torch.flip(gts, [2,])

        # for every label
        total = 0
        total_boundary_f1 = 0
        total_connectivity = 0
        connectivity_valid_count = 0
        for i in range(gts.shape[1]):
            category = self.id2cat[i]
            pred = preds[:, i]
            gt = gts[:, i]
            intersect = (pred & gt).sum().float().item()
            union = (pred | gt).sum().float().item()
            result_dict[category] = intersect / (union + 1e-7)
            total += result_dict[category]

            # Boundary F1
            pred_boundary = self._extract_boundary(pred)
            gt_boundary = self._extract_boundary(gt)
            boundary_f1 = self._binary_f1(pred_boundary, gt_boundary)
            result_dict[f'{category}_boundary_f1'] = boundary_f1
            total_boundary_f1 += boundary_f1

            # Connectivity
            connectivity = self._connectivity_score(pred, gt)
            result_dict[f'{category}_connectivity'] = connectivity
            if not np.isnan(connectivity):
                total_connectivity += connectivity
                connectivity_valid_count += 1
        
        mIoU = total / gts.shape[1]
        result_dict['mIoU'] = mIoU
        result_dict['boundary_F1'] = total_boundary_f1 / gts.shape[1]
        result_dict['connectivity'] = (
            total_connectivity / connectivity_valid_count
            if connectivity_valid_count > 0 else float('nan'))
        
        categories = list(self.cat2id.keys())
        table = prettytable.PrettyTable([' ', *categories, 'mean'])
        table.add_row(['IoU', 
            *[round(result_dict[cat], 4) for cat in categories], 
            round(mIoU, 4)])
        table.add_row(['BoundaryF1',
            *[round(result_dict[f'{cat}_boundary_f1'], 4) for cat in categories],
            round(result_dict['boundary_F1'], 4)])
        table.add_row(['Connectivity',
            *[round(result_dict[f'{cat}_connectivity'], 4) for cat in categories],
            round(result_dict['connectivity'], 4)])
        
        if logger:
            from mmcv.utils import print_log
            print_log('\n'+str(table), logger=logger)
            print_log(
                f"mIoU = {mIoU:.4f}, boundary_F1 = {result_dict['boundary_F1']:.4f}, "
                f"connectivity = {result_dict['connectivity']:.4f}\n",
                logger=logger)

        return result_dict

    @staticmethod
    def _extract_boundary(mask: torch.Tensor) -> torch.Tensor:
        """Extract binary boundaries using morphological gradient."""
        x = mask.float().unsqueeze(1)
        dil = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        ero = -F.max_pool2d(-x, kernel_size=3, stride=1, padding=1)
        boundary = (dil - ero) > 0
        return boundary.squeeze(1)

    @staticmethod
    def _binary_f1(pred: torch.Tensor, gt: torch.Tensor) -> float:
        tp = (pred & gt).sum().float().item()
        fp = (pred & (~gt)).sum().float().item()
        fn = ((~pred) & gt).sum().float().item()
        if (tp + fp + fn) == 0:
            return 1.0
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        return 2 * precision * recall / (precision + recall + 1e-7)

    @staticmethod
    def _connectivity_score(pred: torch.Tensor, gt: torch.Tensor) -> float:
        """Approximate connectivity consistency using connected component counts."""
        if cv2 is None:
            return float('nan')

        pred_np = pred.detach().cpu().numpy().astype(np.uint8)
        gt_np = gt.detach().cpu().numpy().astype(np.uint8)
        scores = []
        for b in range(pred_np.shape[0]):
            pred_cc = cv2.connectedComponents(pred_np[b], connectivity=8)[0] - 1
            gt_cc = cv2.connectedComponents(gt_np[b], connectivity=8)[0] - 1
            denom = max(gt_cc, 1)
            score = 1.0 - (abs(pred_cc - gt_cc) / float(denom))
            score = max(0.0, min(1.0, score))
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0
