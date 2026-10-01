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
    """Light train-time augmentation on (C, T, V) landmark tensors."""
    out = x.clone()
    c, t, v = out.shape

    # Random isotropic scale
    if torch.rand(1).item() < 0.8:
        scale = float(torch.empty(1).uniform_(0.9, 1.1).item())
        out = out * scale

    # Gaussian noise on coordinates
    if torch.rand(1).item() < 0.8:
        noise = torch.randn_like(out) * 0.01
        out = out + noise

    # Temporal shift (circular)
    if t > 1 and torch.rand(1).item() < 0.5:
        shift = int(torch.randint(-max(1, t // 10), max(2, t // 10 + 1), (1,)).item())
        out = torch.roll(out, shifts=shift, dims=1)

    # Randomly drop a few joints (simulate occlusion), keep major torso joints
    if torch.rand(1).item() < 0.3:
        drop_n = int(torch.randint(1, 4, (1,)).item())
        idxs = torch.randperm(v)[:drop_n]
        out[:, :, idxs] = 0.0

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
