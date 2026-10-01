#!/usr/bin/env python
"""Export a trained ST-GCN .pt checkpoint to HDF5 (.h5).

This stores PyTorch weights + metadata in HDF5 format.
It is NOT a Keras/TensorFlow SavedModel; reload with the companion
loader or read arrays with h5py.

Usage (from ssl400-research/):
  python scripts/export_h5.py --checkpoint checkpoints/baseline_stgcn_seed42_best.pt
  python scripts/export_h5.py --checkpoint checkpoints/baseline_stgcn_seed42_best.pt --output checkpoints/baseline_stgcn_seed42_best.h5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import project_root, resolve_path


def export_checkpoint_to_h5(checkpoint_path: Path, output_path: Path) -> Path:
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in ckpt:
        raise KeyError(f"No model_state_dict in checkpoint: {checkpoint_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = ckpt["model_state_dict"]

    with h5py.File(output_path, "w") as f:
        weights = f.create_group("model_weights")
        for name, tensor in state.items():
            arr = tensor.detach().cpu().numpy()
            weights.create_dataset(name, data=arr, compression="gzip")

        meta = f.create_group("metadata")
        meta.attrs["framework"] = "pytorch"
        meta.attrs["model"] = "STGCN"
        meta.attrs["source_checkpoint"] = str(checkpoint_path)
        meta.attrs["num_classes"] = int(ckpt.get("num_classes", -1))
        meta.attrs["in_channels"] = int(ckpt.get("in_channels", 3))
        meta.attrs["epoch"] = int(ckpt.get("epoch", -1))

        model_config = ckpt.get("model_config", {})
        data_config = ckpt.get("data_config", {})
        val_metrics = ckpt.get("val_metrics", {})
        meta.attrs["model_config_json"] = json.dumps(model_config)
        meta.attrs["data_config_json"] = json.dumps(data_config)
        meta.attrs["val_metrics_json"] = json.dumps(
            {k: v for k, v in val_metrics.items() if k not in ("y_true", "y_pred", "sample_ids")}
        )

        class_map_path = ckpt.get("class_to_idx_path")
        if class_map_path and Path(class_map_path).exists():
            class_to_idx = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
            meta.attrs["class_to_idx_json"] = json.dumps(class_to_idx)

        # Convenience: flat list of parameter names
        names = np.array(list(state.keys()), dtype=object)
        dt = h5py.string_dtype(encoding="utf-8")
        weights.create_dataset("param_names", data=names.astype(object), dtype=dt)

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ST-GCN .pt checkpoint to .h5")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument(
        "--output",
        default=None,
        help="Output .h5 path (default: same name as checkpoint with .h5)",
    )
    args = parser.parse_args()

    root = project_root()
    ckpt = resolve_path(args.checkpoint, root)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if args.output:
        out = resolve_path(args.output, root)
    else:
        out = ckpt.with_suffix(".h5")

    path = export_checkpoint_to_h5(ckpt, out)
    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"Exported HDF5 weights → {path}")
    print(f"Size: {size_mb:.2f} MB")


if __name__ == "__main__":
    main()
