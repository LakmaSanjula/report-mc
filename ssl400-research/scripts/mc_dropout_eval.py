#!/usr/bin/env python
"""Monte Carlo Dropout uncertainty for a trained ST-GCN.

Low-resource technique from the development guide (Phase 7): one backbone,
repeated stochastic passes, no second model.

Usage (from ssl400-research/):
  python scripts/mc_dropout_eval.py --checkpoint checkpoints/stgcn_lowres_seed42_best.pt --split val
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset import SSL400PoseDataset, collate_batch
from src.models.stgcn import STGCN, enable_mc_dropout
from src.utils.config import load_yaml, project_root, resolve_path
from src.utils.seed import resolve_device


@torch.no_grad()
def mc_predict(model, loader, device, passes: int) -> dict:
    enable_mc_dropout(model)
    all_true = []
    all_pred = []
    all_conf = []
    all_entropy = []

    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        probs = []
        for _ in range(passes):
            logits = model(x)
            probs.append(F.softmax(logits, dim=1))
        mean_prob = torch.stack(probs, dim=0).mean(dim=0)
        pred = mean_prob.argmax(dim=1)
        conf = mean_prob.max(dim=1).values
        entropy = -(mean_prob * (mean_prob.clamp_min(1e-8).log())).sum(dim=1)
        all_true.extend(y.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_conf.extend(conf.cpu().tolist())
        all_entropy.extend(entropy.cpu().tolist())

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)
    correct = y_true == y_pred
    return {
        "accuracy": float(correct.mean()) if len(y_true) else 0.0,
        "mean_confidence": float(np.mean(all_conf)) if all_conf else 0.0,
        "mean_entropy": float(np.mean(all_entropy)) if all_entropy else 0.0,
        "confidence": all_conf,
        "entropy": all_entropy,
        "correct": correct.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="MC Dropout evaluation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    args = parser.parse_args()

    root = project_root()
    data_cfg = load_yaml(args.data_config)
    model_cfg = load_yaml(args.model_config)
    device = resolve_device(str(model_cfg["train"].get("device", "auto")))
    ckpt_path = resolve_path(args.checkpoint, root)
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)

    mcfg = ckpt.get("model_config", model_cfg["model"])
    processed = resolve_path(data_cfg["processed_dir"], root)
    dataset = SSL400PoseDataset(processed, split=args.split, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=int(model_cfg["train"]["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_batch,
    )
    model = STGCN(
        num_classes=int(ckpt["num_classes"]),
        in_channels=int(ckpt.get("in_channels", 3)),
        num_joints=int(mcfg.get("num_joints", 33)),
        channels=list(mcfg.get("channels", [64, 64, 128, 256])),
        temporal_kernel=int(mcfg.get("temporal_kernel", 9)),
        dropout=float(mcfg.get("dropout", 0.5)),
        graph_strategy=str(mcfg.get("graph_strategy", "spatial")),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    stats = mc_predict(model, loader, device, args.passes)
    experiment = model_cfg["paths"].get("experiment_name", "stgcn")
    results_dir = resolve_path(model_cfg["paths"]["results_dir"], root)
    plots_dir = resolve_path(model_cfg["paths"].get("plots_dir", "results/plots"), root)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "split": args.split,
        "passes": args.passes,
        "accuracy": stats["accuracy"],
        "mean_confidence": stats["mean_confidence"],
        "mean_entropy": stats["mean_entropy"],
        "num_samples": len(stats["confidence"]),
    }
    out_json = results_dir / f"{experiment}_{args.split}_mc_dropout.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    correct = np.asarray(stats["correct"], dtype=bool)
    conf = np.asarray(stats["confidence"])
    ent = np.asarray(stats["entropy"])
    axes[0].hist(conf[correct], bins=20, alpha=0.7, label="correct")
    if (~correct).any():
        axes[0].hist(conf[~correct], bins=20, alpha=0.7, label="wrong")
    axes[0].set_title("MC confidence")
    axes[0].legend()
    axes[1].hist(ent[correct], bins=20, alpha=0.7, label="correct")
    if (~correct).any():
        axes[1].hist(ent[~correct], bins=20, alpha=0.7, label="wrong")
    axes[1].set_title("MC predictive entropy")
    axes[1].legend()
    fig.tight_layout()
    plot_path = plots_dir / f"{experiment}_{args.split}_mc_uncertainty.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)

    print(
        f"MC Dropout | split={args.split} | passes={args.passes} | "
        f"acc={stats['accuracy']:.4f} | mean_entropy={stats['mean_entropy']:.4f}"
    )
    print(f"Wrote {out_json}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
