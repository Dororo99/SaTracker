import torch
from mmdet3d.datasets import build_dataset, build_dataloader
import mmcv
from functools import cached_property
import prettytable
from numpy.typing import NDArray
from typing import Dict, Optional
from logging import Logger
from mmcv import Config
from copy import deepcopy

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

    def _compute_iou(self, results, gts_stacked, mask_key='semantic_mask'):
        """Compute per-category IoU for a given mask key."""
        preds = []
        for token, gt in self.gts.items():
            pred = torch.zeros((len(self.cat2id), gt.shape[1], gt.shape[2])).bool()
            if token in results and mask_key in results[token]:
                semantic_mask = torch.tensor(results[token][mask_key])
                for label_i in range(gt.shape[0]):
                    pred[label_i] = (semantic_mask == label_i+1)
            preds.append(pred)

        preds = torch.stack(preds).bool()
        result_dict = {}
        total = 0
        for i in range(gts_stacked.shape[1]):
            category = self.id2cat[i]
            pred = preds[:, i]
            gt = gts_stacked[:, i]
            intersect = (pred & gt).sum().float().item()
            union = (pred | gt).sum().float().item()
            result_dict[category] = intersect / (union + 1e-7)
            total += result_dict[category]
        result_dict['mIoU'] = total / gts_stacked.shape[1]
        return result_dict

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

        # Prepare GT (once)
        gts = []
        for token, gt in self.gts.items():
            gts.append(gt)
        gts_stacked = torch.stack(gts).bool()
        gts_stacked = torch.flip(gts_stacked, [2,])

        # Main eval (fused)
        result_dict = self._compute_iou(results, gts_stacked, 'semantic_mask')
        categories = list(self.cat2id.keys())
        table = prettytable.PrettyTable([' ', *categories, 'mean'])
        table.add_row(['IoU',
            *[round(result_dict[cat], 4) for cat in categories],
            round(result_dict['mIoU'], 4)])

        if logger:
            from mmcv.utils import print_log
            print_log('\n'+str(table), logger=logger)
            print_log(f'mIoU = {result_dict["mIoU"]:.4f}', logger=logger)

        # Cam/Sat variants (if present)
        has_cam = any('semantic_mask_cam' in v for v in results.values())
        has_sat = any('semantic_mask_sat' in v for v in results.values())

        if has_cam or has_sat:
            rows = []
            if has_cam:
                cam_dict = self._compute_iou(results, gts_stacked, 'semantic_mask_cam')
                result_dict['mIoU_cam'] = cam_dict['mIoU']
                for cat in categories:
                    result_dict[f'{cat}_cam'] = cam_dict[cat]
                rows.append(['IoU_cam',
                    *[round(cam_dict[cat], 4) for cat in categories],
                    round(cam_dict['mIoU'], 4)])
            if has_sat:
                sat_dict = self._compute_iou(results, gts_stacked, 'semantic_mask_sat')
                result_dict['mIoU_sat'] = sat_dict['mIoU']
                for cat in categories:
                    result_dict[f'{cat}_sat'] = sat_dict[cat]
                rows.append(['IoU_sat',
                    *[round(sat_dict[cat], 4) for cat in categories],
                    round(sat_dict['mIoU'], 4)])

            if logger and rows:
                table2 = prettytable.PrettyTable([' ', *categories, 'mean'])
                for row in rows:
                    table2.add_row(row)
                print_log('\n'+str(table2), logger=logger)
                if has_cam:
                    print_log(f'mIoU_cam = {cam_dict["mIoU"]:.4f}', logger=logger)
                if has_sat:
                    print_log(f'mIoU_sat = {sat_dict["mIoU"]:.4f}\n', logger=logger)

        return result_dict
