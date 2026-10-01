#!/usr/bin/env python
"""Prepare SSL400 Pose CSVs into ST-GCN tensors and stratified splits.

Usage (from ssl400-research/):
  python scripts/prepare_dataset.py
  python scripts/prepare_dataset.py --data-config configs/data.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.prepare import prepare_dataset
from src.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SSL400 dataset for ST-GCN")
    parser.add_argument("--data-config", default="configs/data.yaml")
    args = parser.parse_args()
    data_cfg = load_yaml(args.data_config)
    prepare_dataset(data_cfg)


if __name__ == "__main__":
    main()
