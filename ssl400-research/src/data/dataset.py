"""PyTorch Dataset for prepared SSL400 Pose sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from src.utils.config import project_root, resolve_path


def augment_pose_tensor(x: torch.Tensor) -> torch.Tensor:
    """Mild train-time augmentation that keeps the sign label valid.

    Used transforms (small, label-preserving):
      - slight scale
      - small coordinate noise
      - tiny in-plane rotation
      - short temporal shift

    Not used: left-right mirror (changes many signs) and joint dropout
    (the previous strong version hurt accuracy).
    """
    out = x.clone()
    _, t, _ = out.shape

    if torch.rand(1).item() < 0.7:
        scale = float(torch.empty(1).uniform_(0.95, 1.05).item())
        out = out * scale

    if torch.rand(1).item() < 0.7:
        out = out + torch.randn_like(out) * 0.005

    # Small rotation in the image plane (x, y). z is left unchanged.
    if out.size(0) >= 2 and torch.rand(1).item() < 0.5:
        angle = float(torch.empty(1).uniform_(-0.08, 0.08).item())  # ~±5 degrees
        cos_a = float(np.cos(angle))
        sin_a = float(np.sin(angle))
        x_coord = out[0].clone()
        y_coord = out[1].clone()
        out[0] = cos_a * x_coord - sin_a * y_coord
        out[1] = sin_a * x_coord + cos_a * y_coord

    if t > 4 and torch.rand(1).item() < 0.4:
        shift = int(torch.randint(-2, 3, (1,)).item())
        if shift != 0:
            out = torch.roll(out, shifts=shift, dims=1)

    # Signing-speed change: resample time, then restore length T.
    # This is label-preserving and matches the guide's signing-speed condition.
    if t > 8 and torch.rand(1).item() < 0.5:
        rate = float(torch.empty(1).uniform_(0.85, 1.15).item())
        new_t = max(8, int(round(t * rate)))
        src_idx = torch.linspace(0, t - 1, new_t).round().long().clamp(0, t - 1)
        sampled = out[:, src_idx, :]
        back_idx = torch.linspace(0, new_t - 1, t).round().long().clamp(0, new_t - 1)
        out = sampled[:, back_idx, :]

    return out


class SSL400PoseDataset(Dataset):
    """Loads cached `(C, T, V)` tensors from the preparation manifest."""

    def __init__(
        self,
        processed_dir: str | Path,
        split: str = "train",
        augment: bool = False,
    ) -> None:
        processed_dir = resolve_path(processed_dir, project_root())
        manifest_path = processed_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Missing {manifest_path}. Run: python scripts/prepare_dataset.py"
            )
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest: dict[str, list[dict[str, Any]]] = json.load(f)
        if split not in manifest:
            raise KeyError(f"Split '{split}' not in manifest keys: {list(manifest)}")
        self.records = manifest[split]
        self.split = split
        self.augment = bool(augment) and split == "train"

        class_map_path = processed_dir / "class_to_idx.json"
        with class_map_path.open("r", encoding="utf-8") as f:
            self.class_to_idx = json.load(f)
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        tensor = np.load(record["tensor_path"]).astype(np.float32)
        x = torch.from_numpy(tensor)  # (C, T, V)
        if self.augment:
            x = augment_pose_tensor(x)
        y = torch.tensor(record["label"], dtype=torch.long)
        return {
            "x": x,
            "y": y,
            "sample_id": record["sample_id"],
            "class_name": record["class_name"],
        }

    def label_counts(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for r in self.records:
            label = int(r["label"])
            counts[label] = counts.get(label, 0) + 1
        return counts

    def sample_weights(self) -> list[float]:
        """Inverse-frequency weights for WeightedRandomSampler."""
        counts = self.label_counts()
        return [1.0 / float(counts[int(r["label"])]) for r in self.records]


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    x = torch.stack([b["x"] for b in batch], dim=0)
    y = torch.stack([b["y"] for b in batch], dim=0)
    return {
        "x": x,
        "y": y,
        "sample_id": [b["sample_id"] for b in batch],
        "class_name": [b["class_name"] for b in batch],
    }
