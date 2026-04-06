#!/usr/bin/env python3
"""Aggregate per-seed metric json files into mean/std summary.

Usage:
  python scripts/aggregate_multiseed_metrics.py \
      --inputs seed0.json seed1.json seed2.json \
      --output summary.json
"""
import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


def _is_number(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float('nan')


def _std(values: List[float]) -> float:
    if not values:
        return float('nan')
    m = _mean(values)
    return math.sqrt(sum((v - m) ** 2 for v in values) / len(values))


def aggregate(records: List[Dict]) -> Dict:
    keys = sorted({k for rec in records for k, v in rec.items() if _is_number(v)})
    out = {
        'num_seeds': len(records),
        'metrics': {},
    }
    for k in keys:
        vals = [float(rec[k]) for rec in records if k in rec and _is_number(rec[k]) and not math.isnan(float(rec[k]))]
        if len(vals) == 0:
            continue
        out['metrics'][k] = {
            'mean': _mean(vals),
            'std': _std(vals),
            'values': vals,
        }
    return out


def main():
    parser = argparse.ArgumentParser(description='Aggregate per-seed metric json files.')
    parser.add_argument('--inputs', nargs='+', required=True, help='Per-seed metric json paths')
    parser.add_argument('--output', type=str, default=None, help='Optional output summary json')
    args = parser.parse_args()

    records = []
    for p in args.inputs:
        path = Path(p)
        with path.open('r', encoding='utf-8') as f:
            records.append(json.load(f))

    summary = aggregate(records)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
