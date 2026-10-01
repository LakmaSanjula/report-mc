"""Prepare SSL400 MediaPipe Pose CSVs into tensors, splits, and class maps.

CSV schema (inspected):
  - Headerless
  - Each row = one frame
  - 33 columns = MediaPipe Pose landmarks
  - Each cell = string "[x, y, z, visibility]"
  - Label = parent folder name (Category/Class)
"""

from __future__ import annotations

import ast
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.utils.config import project_root, resolve_path


def _parse_cell(cell: str) -> list[float]:
    value = ast.literal_eval(cell)
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError(f"Expected [x,y,z,visibility], got: {cell!r}")
    return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]


def load_csv_sequence(
    csv_path: Path,
    use_z: bool = True,
    visibility_threshold: float = 0.5,
    mask_low_visibility: bool = True,
) -> np.ndarray:
    """Load one sample as array shaped (T, V, C) where C is 2 or 3 (no visibility channel)."""
    frames: list[np.ndarray] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) != 33:
                raise ValueError(f"{csv_path}: expected 33 landmarks, found {len(row)}")
            joints = []
            for cell in row:
                x, y, z, vis = _parse_cell(cell)
                coords = [x, y, z] if use_z else [x, y]
                if mask_low_visibility and vis < visibility_threshold:
                    coords = [0.0] * len(coords)
                joints.append(coords)
            frames.append(np.asarray(joints, dtype=np.float32))
    if not frames:
        raise ValueError(f"Empty CSV: {csv_path}")
    return np.stack(frames, axis=0)  # (T, 33, C)


def temporal_resize(seq: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or center-crop/truncate to fixed length T."""
    t = seq.shape[0]
    if t == target_len:
        return seq
    if t > target_len:
        start = (t - target_len) // 2
        return seq[start : start + target_len]
    pad = target_len - t
    pad_before = pad // 2
    pad_after = pad - pad_before
    return np.pad(seq, ((pad_before, pad_after), (0, 0), (0, 0)), mode="constant")


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Translate by mid-hip and scale by shoulder width (training-safe, per-sample)."""
    # MediaPipe: left_shoulder=11, right_shoulder=12, left_hip=23, right_hip=24
    out = seq.copy()
    mid_hip = 0.5 * (out[:, 23, :2] + out[:, 24, :2])
    out[..., 0] = out[..., 0] - mid_hip[:, 0:1]
    out[..., 1] = out[..., 1] - mid_hip[:, 1:2]

    shoulder = out[:, 11, :2] - out[:, 12, :2]
    scale = np.linalg.norm(shoulder, axis=1)
    scale = np.median(scale[scale > 1e-6]) if np.any(scale > 1e-6) else 1.0
    scale = max(float(scale), 1e-6)
    out[..., :2] = out[..., :2] / scale
    if out.shape[-1] == 3:
        out[..., 2] = out[..., 2] / scale
    return out.astype(np.float32)


def discover_samples(raw_root: Path) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for category_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        for class_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            class_name = class_dir.name
            for csv_path in sorted(class_dir.glob("*.csv")):
                sample_id = f"{category_dir.name}/{class_name}/{csv_path.stem}"
                samples.append(
                    {
                        "sample_id": sample_id,
                        "path": str(csv_path.resolve()),
                        "category": category_dir.name,
                        "class_name": class_name,
                    }
                )
    return samples


def filter_by_min_count(
    samples: list[dict[str, Any]], min_samples: int
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    counts = Counter(s["class_name"] for s in samples)
    keep = {c for c, n in counts.items() if n >= min_samples}
    filtered = [s for s in samples if s["class_name"] in keep]
    kept_counts = {c: counts[c] for c in sorted(keep)}
    return filtered, kept_counts


def build_class_map(class_names: list[str]) -> dict[str, int]:
    return {name: idx for idx, name in enumerate(sorted(class_names))}


def stratified_split(
    samples: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    labels = [s["class_name"] for s in samples]
    train_val, test = train_test_split(
        samples,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    relative_val = val_ratio / (train_ratio + val_ratio)
    train_labels = [s["class_name"] for s in train_val]
    train, val = train_test_split(
        train_val,
        test_size=relative_val,
        random_state=seed,
        stratify=train_labels,
    )
    return {"train": train, "val": val, "test": test}


def prepare_dataset(data_cfg: dict[str, Any]) -> dict[str, Any]:
    """Discover CSVs, filter classes, split, preprocess, and cache tensors."""
    root = project_root()
    raw_root = resolve_path(data_cfg["raw_csv_root"], root)
    processed_dir = resolve_path(data_cfg["processed_dir"], root)
    inspected_dir = resolve_path(data_cfg["inspected_dir"], root)
    processed_dir.mkdir(parents=True, exist_ok=True)
    inspected_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(
            f"Raw CSV root not found: {raw_root}\n"
            "Update configs/data.yaml raw_csv_root to your Dataset - MP - CSV path."
        )

    all_samples = discover_samples(raw_root)
    filtered, kept_counts = filter_by_min_count(
        all_samples, int(data_cfg["min_samples_per_class"])
    )
    if not filtered:
        raise RuntimeError("No classes left after min_samples_per_class filter.")

    class_names = sorted({s["class_name"] for s in filtered})
    class_to_idx = build_class_map(class_names)
    idx_to_class = {idx: name for name, idx in class_to_idx.items()}

    splits = stratified_split(
        filtered,
        float(data_cfg["train_ratio"]),
        float(data_cfg["val_ratio"]),
        float(data_cfg["test_ratio"]),
        int(data_cfg["split_seed"]),
    )

    use_z = bool(data_cfg.get("use_z", True))
    seq_len = int(data_cfg["sequence_length"])
    vis_thr = float(data_cfg.get("visibility_threshold", 0.5))
    mask_vis = bool(data_cfg.get("mask_low_visibility", True))
    save_npz = bool(data_cfg.get("save_npz", True))

    channels = 3 if use_z else 2
    sequences_dir = processed_dir / "sequences"
    if save_npz:
        sequences_dir.mkdir(parents=True, exist_ok=True)

    frame_lengths: list[int] = []
    manifest: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for split_name, split_samples in splits.items():
        for sample in tqdm(split_samples, desc=f"Preparing {split_name}"):
            csv_path = Path(sample["path"])
            seq = load_csv_sequence(
                csv_path,
                use_z=use_z,
                visibility_threshold=vis_thr,
                mask_low_visibility=mask_vis,
            )
            frame_lengths.append(int(seq.shape[0]))
            seq = normalize_sequence(seq)
            seq = temporal_resize(seq, seq_len)  # (T, V, C)
            # ST-GCN convention: (C, T, V)
            tensor = np.transpose(seq, (2, 0, 1)).astype(np.float32)

            label = class_to_idx[sample["class_name"]]
            record = {
                "sample_id": sample["sample_id"],
                "class_name": sample["class_name"],
                "label": label,
                "category": sample["category"],
                "source_csv": sample["path"],
                "split": split_name,
                "shape": list(tensor.shape),
            }
            if save_npz:
                out_name = sample["sample_id"].replace("/", "__") + ".npy"
                out_path = sequences_dir / out_name
                np.save(out_path, tensor)
                record["tensor_path"] = str(out_path.resolve())
            else:
                record["tensor"] = tensor
            manifest[split_name].append(record)

    inventory = {
        "raw_root": str(raw_root),
        "total_raw_samples": len(all_samples),
        "filtered_samples": len(filtered),
        "num_classes": len(class_to_idx),
        "min_samples_per_class": int(data_cfg["min_samples_per_class"]),
        "class_counts": kept_counts,
        "split_counts": {k: len(v) for k, v in manifest.items()},
        "sequence_length": seq_len,
        "channels": channels,
        "num_joints": 33,
        "frame_length_stats": {
            "min": int(min(frame_lengths)),
            "max": int(max(frame_lengths)),
            "mean": float(np.mean(frame_lengths)),
            "median": float(np.median(frame_lengths)),
        },
        "notes": [
            "Pose-only MediaPipe landmarks (33 joints); no hand/face landmarks in CSVs.",
            "No signer IDs available; split is stratified by class only.",
            "Visibility masked below threshold; coordinates normalized by mid-hip / shoulder width.",
        ],
    }

    with (processed_dir / "class_to_idx.json").open("w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, indent=2, ensure_ascii=False)
    with (processed_dir / "idx_to_class.json").open("w", encoding="utf-8") as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2, ensure_ascii=False)
    with (processed_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with (inspected_dir / "dataset_inventory.json").open("w", encoding="utf-8") as f:
        json.dump(inventory, f, indent=2, ensure_ascii=False)
    with (inspected_dir / "data_representation_decision.md").open("w", encoding="utf-8") as f:
        f.write(_decision_markdown(inventory, data_cfg))

    print(f"Prepared {inventory['filtered_samples']} samples, {inventory['num_classes']} classes.")
    print(f"Splits: {inventory['split_counts']}")
    print(f"Artifacts written under: {processed_dir}")
    return inventory


def _decision_markdown(inventory: dict[str, Any], data_cfg: dict[str, Any]) -> str:
    return f"""# Data representation decision

## Representation
- Source: `{inventory['raw_root']}`
- Format: headerless MediaPipe Pose CSV (`T` rows × 33 `[x,y,z,visibility]` cells)
- Training tensor: `(C, T, V)` with C={inventory['channels']}, T={inventory['sequence_length']}, V=33

## Why this matches the plan
- Supplied MediaPipe CSVs are the only on-disk landmark representation.
- Labels come from folder names (`Category/Class`).
- Pose graph nodes match CSV column order 0..32.

## Filters and split
- `min_samples_per_class`: {data_cfg['min_samples_per_class']}
- Kept classes: {inventory['num_classes']}
- Split seed: {data_cfg['split_seed']}
- Ratios: train={data_cfg['train_ratio']}, val={data_cfg['val_ratio']}, test={data_cfg['test_ratio']}
- Split counts: {inventory['split_counts']}

## Limitations
- Pose-only (no hands/face) limits fine-grained SSL discrimination.
- No signer IDs → cannot claim unseen-signer evaluation from this split.
- Variable native frame lengths (min={inventory['frame_length_stats']['min']}, max={inventory['frame_length_stats']['max']}, median={inventory['frame_length_stats']['median']}) resized to fixed T.
"""
