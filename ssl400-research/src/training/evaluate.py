"""Evaluation helpers for ST-GCN."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SSL400PoseDataset, collate_batch
from src.models.stgcn import STGCN
from src.training.metrics import classification_report_dict
from src.utils.config import project_root, resolve_path
from src.utils.seed import resolve_device


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> dict[str, Any]:
    model.eval()
    losses = []
    y_true: list[int] = []
    y_pred: list[int] = []
    sample_ids: list[str] = []

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        logits = model(x)
        if criterion is not None:
            losses.append(float(criterion(logits, y).item()))
        pred = logits.argmax(dim=1)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
        sample_ids.extend(batch["sample_id"])

    metrics = classification_report_dict(y_true, y_pred)
    metrics["loss"] = float(sum(losses) / max(len(losses), 1)) if losses else None
    metrics["num_samples"] = len(y_true)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    metrics["sample_ids"] = sample_ids
    return metrics


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_cfg: dict[str, Any],
    model_cfg: dict[str, Any],
    split: str = "test",
) -> dict[str, Any]:
    root = project_root()
    processed_dir = resolve_path(data_cfg["processed_dir"], root)
    results_dir = resolve_path(model_cfg["paths"]["results_dir"], root)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(str(model_cfg["train"].get("device", "auto")))
    ckpt = torch.load(checkpoint_path, map_location=device)
    in_channels = int(ckpt.get("in_channels", 3 if data_cfg.get("use_z", True) else 2))
    mcfg = ckpt.get("model_config", model_cfg["model"])

    dataset = SSL400PoseDataset(processed_dir, split=split)
    loader = DataLoader(
        dataset,
        batch_size=int(model_cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(model_cfg["train"].get("num_workers", 0)),
        collate_fn=collate_batch,
    )

    model = STGCN(
        num_classes=int(ckpt["num_classes"]),
        in_channels=in_channels,
        num_joints=int(mcfg.get("num_joints", 33)),
        channels=list(mcfg.get("channels", [64, 64, 128, 256])),
        temporal_kernel=int(mcfg.get("temporal_kernel", 9)),
        dropout=float(mcfg.get("dropout", 0.5)),
        graph_strategy=str(mcfg.get("graph_strategy", "spatial")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    metrics = evaluate_model(model, loader, device, criterion=nn.CrossEntropyLoss())
    # Strip large arrays from printed summary; keep them in saved file.
    out = {k: v for k, v in metrics.items()}
    experiment = model_cfg["paths"].get("experiment_name", "baseline_stgcn")
    out_path = results_dir / f"{experiment}_{split}_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        f"{split} | acc={metrics['accuracy']:.4f} | "
        f"f1_macro={metrics['f1_macro']:.4f} | "
        f"f1_weighted={metrics['f1_weighted']:.4f}"
    )
    print(f"Wrote {out_path}")
    return out
