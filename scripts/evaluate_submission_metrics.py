#!/usr/bin/env python3
"""Evaluate a submission json with dataset evaluator and export metrics.

Example:
  python scripts/evaluate_submission_metrics.py \
      --config plugin/configs/maptracker/nuscenes_newsplit/satmaptracker_stage1_bev_pretrain.py \
      --result-path work_dirs/exp/submission_vector.json \
      --output work_dirs/exp/metrics.json
"""
import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset


def _import_plugins(cfg, config_path: str):
    sys.path.append(os.path.abspath('.'))
    if not getattr(cfg, 'plugin', False):
        return

    plugin_dirs = getattr(cfg, 'plugin_dir', None)
    if plugin_dirs is None:
        plugin_dirs = [os.path.dirname(config_path)]
    elif not isinstance(plugin_dirs, list):
        plugin_dirs = [plugin_dirs]

    for plugin_dir in plugin_dirs:
        module_parts = os.path.dirname(plugin_dir).split('/')
        module_path = module_parts[0]
        for m in module_parts[1:]:
            module_path += f'.{m}'
        importlib.import_module(module_path)


def _sanitize_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _sanitize_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_jsonable(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate submission json and export metrics.')
    parser.add_argument('--config', required=True, help='Config path')
    parser.add_argument('--result-path', required=True, help='submission_vector.json path')
    parser.add_argument('--output', default=None, help='Optional output metrics json')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='Override config options (same format as tools/train.py)')
    parser.add_argument(
        '--eval-semantic',
        action='store_true',
        help='Force semantic evaluator path')
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    _import_plugins(cfg, args.config)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    eval_semantic = args.eval_semantic or bool(getattr(dataset, 'eval_semantic', False))
    metrics = dataset._evaluate(args.result_path, eval_semantic=eval_semantic)
    metrics = _sanitize_jsonable(metrics)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open('w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
