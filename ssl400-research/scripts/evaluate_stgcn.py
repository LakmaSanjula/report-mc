#!/usr/bin/env python
"""Evaluate a trained ST-GCN checkpoint on a frozen split.

Usage (from ssl400-research/):
  python scripts/evaluate_stgcn.py --checkpoint checkpoints/baseline_stgcn_seed42_best.pt
  python scripts/evaluate_stgcn.py --checkpoint checkpoints/baseline_stgcn_seed42_best.pt --split val
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.evaluate import evaluate_checkpoint
from src.utils.config import load_yaml, resolve_path, project_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ST-GCN checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    ckpt = resolve_path(args.checkpoint, project_root())
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    evaluate_checkpoint(ckpt, data_cfg, model_cfg, split=args.split)


if __name__ == "__main__":
    main()
