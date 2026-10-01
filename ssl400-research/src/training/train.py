"""ST-GCN training loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.dataset import SSL400PoseDataset, collate_batch
from src.models.stgcn import STGCN
from src.training.evaluate import evaluate_model
from src.utils.config import project_root, resolve_path
from src.utils.seed import resolve_device, set_seed


def _class_weights(dataset: SSL400PoseDataset, device: torch.device) -> torch.Tensor:
    counts = dataset.label_counts()
    weights = torch.ones(dataset.num_classes, dtype=torch.float32)
    total = sum(counts.values())
    for label, count in counts.items():
        weights[label] = total / (dataset.num_classes * max(count, 1))
    return weights.to(device)


def _build_optimizer(model: nn.Module, train_cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_cfg.get("optimizer", "adam")).lower()
    lr = float(train_cfg["learning_rate"])
    wd = float(train_cfg.get("weight_decay", 0.0))
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    raise ValueError(f"Unsupported optimizer: {name}")


def _build_scheduler(optimizer: torch.optim.Optimizer, train_cfg: dict[str, Any]):
    name = str(train_cfg.get("scheduler", "cosine")).lower()
    epochs = int(train_cfg["epochs"])
    if name == "none":
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("step_size", 30)),
            gamma=float(train_cfg.get("gamma", 0.1)),
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def train_model(data_cfg: dict[str, Any], model_cfg: dict[str, Any]) -> dict[str, Any]:
    train_cfg = model_cfg["train"]
    paths_cfg = model_cfg["paths"]
    mcfg = model_cfg["model"]

    set_seed(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg.get("device", "auto")))
    root = project_root()
    processed_dir = resolve_path(data_cfg["processed_dir"], root)
    ckpt_dir = resolve_path(paths_cfg["checkpoint_dir"], root)
    results_dir = resolve_path(paths_cfg["results_dir"], root)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SSL400PoseDataset(processed_dir, split="train")
    val_ds = SSL400PoseDataset(processed_dir, split="val")

    in_channels = 3 if data_cfg.get("use_z", True) else 2
    if int(mcfg.get("in_channels", in_channels)) != in_channels:
        print(
            f"Warning: model.in_channels={mcfg.get('in_channels')} "
            f"but data channels={in_channels}. Using {in_channels}."
        )

    model = STGCN(
        num_classes=train_ds.num_classes,
        in_channels=in_channels,
        num_joints=int(mcfg.get("num_joints", 33)),
        channels=list(mcfg.get("channels", [64, 64, 128, 256])),
        temporal_kernel=int(mcfg.get("temporal_kernel", 9)),
        dropout=float(mcfg.get("dropout", 0.5)),
        graph_strategy=str(mcfg.get("graph_strategy", "spatial")),
    ).to(device)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_batch,
        pin_memory=device.type == "cuda",
    )

    if bool(train_cfg.get("class_weights", True)):
        criterion = nn.CrossEntropyLoss(weight=_class_weights(train_ds, device))
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = _build_optimizer(model, train_cfg)
    scheduler = _build_scheduler(optimizer, train_cfg)
    use_amp = bool(train_cfg.get("amp", False)) and device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_val_f1 = -1.0
    best_epoch = -1
    patience = int(train_cfg.get("early_stopping_patience", 15))
    patience_left = patience
    history: list[dict[str, Any]] = []
    experiment = str(paths_cfg.get("experiment_name", "baseline_stgcn"))
    best_path = ckpt_dir / f"{experiment}_best.pt"
    last_path = ckpt_dir / f"{experiment}_last.pt"

    print(f"Device: {device}")
    print(f"Classes: {train_ds.num_classes}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"MC Dropout p={mcfg.get('dropout', 0.5)} (active during training)")

    epochs = int(train_cfg["epochs"])
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} train", leave=False):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            try:
                autocast_ctx = torch.amp.autocast("cuda", enabled=use_amp)
            except (AttributeError, TypeError):
                autocast_ctx = torch.cuda.amp.autocast(enabled=use_amp)
            with autocast_ctx:
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.item()) * x.size(0)
            n_seen += x.size(0)

        train_loss = running_loss / max(n_seen, 1)
        val_metrics = evaluate_model(model, val_loader, device, criterion)
        if scheduler is not None:
            scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": float(optimizer.param_groups[0]["lr"]),
            "seconds": round(time.time() - t0, 2),
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_f1_macro={val_metrics['f1_macro']:.4f}"
        )

        # Keep checkpoint small (no per-sample arrays).
        val_metrics_compact = {
            k: v
            for k, v in val_metrics.items()
            if k not in ("y_true", "y_pred", "sample_ids")
        }
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "num_classes": train_ds.num_classes,
            "in_channels": in_channels,
            "model_config": mcfg,
            "data_config": {
                "sequence_length": data_cfg["sequence_length"],
                "use_z": data_cfg.get("use_z", True),
                "min_samples_per_class": data_cfg["min_samples_per_class"],
                "split_seed": data_cfg["split_seed"],
            },
            "val_metrics": val_metrics_compact,
            "class_to_idx_path": str((processed_dir / "class_to_idx.json").resolve()),
        }
        torch.save(checkpoint, last_path)

        if val_metrics["f1_macro"] > best_val_f1:
            best_val_f1 = val_metrics["f1_macro"]
            best_epoch = epoch
            patience_left = patience
            torch.save(checkpoint, best_path)
            print(f"  Saved best checkpoint → {best_path}")
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    history_path = results_dir / f"{experiment}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    summary = {
        "best_epoch": best_epoch,
        "best_val_f1_macro": best_val_f1,
        "best_checkpoint": str(best_path.resolve()),
        "last_checkpoint": str(last_path.resolve()),
        "history_path": str(history_path.resolve()),
        "device": str(device),
        "num_classes": train_ds.num_classes,
    }
    with (results_dir / f"{experiment}_train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Training complete. Best val F1-macro={best_val_f1:.4f} @ epoch {best_epoch}")
    return summary
