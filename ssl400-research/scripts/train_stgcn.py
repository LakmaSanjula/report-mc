#!/usr/bin/env python
"""Train the SSL400 ST-GCN baseline.

Usage (from ssl400-research/):
  python scripts/train_stgcn.py
  python scripts/train_stgcn.py --data-config configs/data.yaml --model-config configs/model.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training.train import train_model
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ST-GCN on prepared SSL400 data")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    train_model(data_cfg, model_cfg)


if __name__ == "__main__":
    main()
