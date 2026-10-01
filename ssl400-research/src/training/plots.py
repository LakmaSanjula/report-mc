"""Training / evaluation plots for SSL400 ST-GCN."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_distribution(
    manifest: dict[str, list[dict[str, Any]]],
    out_path: Path,
    top_k: int = 40,
) -> None:
    """Bar chart of per-split sample counts for the most frequent classes."""
    all_names = []
    for split_records in manifest.values():
        all_names.extend(r["class_name"] for r in split_records)
    top = [n for n, _ in Counter(all_names).most_common(top_k)]

    splits = [s for s in ("train", "val", "test") if s in manifest]
    x = np.arange(len(top))
    width = 0.25
    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.35), 5))
    for i, split in enumerate(splits):
        counts = Counter(r["class_name"] for r in manifest[split])
        vals = [counts.get(n, 0) for n in top]
        ax.bar(x + (i - 1) * width, vals, width=width, label=split)
    ax.set_xticks(x)
    ax.set_xticklabels(top, rotation=75, ha="right", fontsize=8)
    ax.set_ylabel("Samples")
    ax.set_title(f"Class distribution (top {top_k} by total count)")
    ax.legend()
    _savefig(out_path)


def plot_split_sizes(manifest: dict[str, list[dict[str, Any]]], out_path: Path) -> None:
    splits = [s for s in ("train", "val", "test") if s in manifest]
    sizes = [len(manifest[s]) for s in splits]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(splits, sizes, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(sizes):
        ax.text(i, v, str(v), ha="center", va="bottom")
    ax.set_ylabel("Samples")
    ax.set_title("Split sizes")
    _savefig(out_path)


def plot_training_curves(history: list[dict[str, Any]], out_dir: Path, prefix: str) -> None:
    if not history:
        return
    epochs = [h["epoch"] for h in history]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, [h["train_loss"] for h in history], label="train_loss")
    ax.plot(epochs, [h["val_loss"] for h in history], label="val_loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training / validation loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(out_dir / f"{prefix}_loss_curves.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    if "train_accuracy" in history[0]:
        ax.plot(epochs, [h["train_accuracy"] for h in history], label="train_acc")
    ax.plot(epochs, [h["val_accuracy"] for h in history], label="val_top1_acc")
    if "val_top5_accuracy" in history[0]:
        ax.plot(epochs, [h["val_top5_accuracy"] for h in history], label="val_top5_acc")
    ax.plot(epochs, [h["val_f1_macro"] for h in history], label="val_f1_macro")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score")
    ax.set_title("Top-1 / Top-5 accuracy and macro-F1")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _savefig(out_dir / f"{prefix}_metric_curves.png")

    if "train_accuracy" in history[0] and "val_accuracy" in history[0]:
        fig, ax = plt.subplots(figsize=(7, 4))
        gap = [h["train_accuracy"] - h["val_accuracy"] for h in history]
        ax.plot(epochs, gap, color="#C44E52")
        ax.axhline(0.0, color="gray", linewidth=1)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Train acc − val acc")
        ax.set_title("Overfit gap")
        ax.grid(True, alpha=0.3)
        _savefig(out_dir / f"{prefix}_overfit_gap.png")

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(epochs, [h["lr"] for h in history], color="#8172B3")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate schedule")
    ax.grid(True, alpha=0.3)
    _savefig(out_dir / f"{prefix}_lr_curve.png")


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    out_path: Path,
    class_names: Optional[list[str]] = None,
    top_k: int = 30,
) -> Path:
    """Plot confusion for the top-K most frequent true classes (full CM saved as .npy)."""
    cm_full = confusion_matrix(y_true, y_pred)
    np.save(out_path.with_suffix(".npy"), cm_full)

    counts = Counter(y_true)
    top_labels = [i for i, _ in counts.most_common(min(top_k, len(counts)))]
    if not top_labels:
        return out_path

    label_index = {lab: i for i, lab in enumerate(top_labels)}
    cm = np.zeros((len(top_labels), len(top_labels)), dtype=np.int64)
    for yt, yp in zip(y_true, y_pred):
        if yt in label_index and yp in label_index:
            cm[label_index[yt], label_index[yp]] += 1

    # Row-normalize for readability
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(row_sum, 1))

    tick_labels = (
        [class_names[i] if class_names and i < len(class_names) else str(i) for i in top_labels]
        if class_names is not None
        else [str(i) for i in top_labels]
    )

    fig, ax = plt.subplots(figsize=(max(8, top_k * 0.35), max(7, top_k * 0.35)))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_xticks(range(len(top_labels)))
    ax.set_yticks(range(len(top_labels)))
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticklabels(tick_labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Normalized confusion (top {len(top_labels)} frequent classes)")
    _savefig(out_path)
    return out_path


def plot_per_class_f1(
    y_true: list[int],
    y_pred: list[int],
    out_path: Path,
    class_names: Optional[list[str]] = None,
    top_k: int = 30,
) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    f1s = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    pairs = sorted(zip(labels, f1s), key=lambda t: t[1], reverse=True)
    top = pairs[:top_k]
    bottom = pairs[-min(top_k, len(pairs)) :]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(4, top_k * 0.22)))
    for ax, subset, title in (
        (axes[0], top, f"Top {len(top)} F1"),
        (axes[1], list(reversed(bottom)), f"Bottom {len(bottom)} F1"),
    ):
        names = []
        vals = []
        for lab, score in subset:
            name = class_names[lab] if class_names and lab < len(class_names) else str(lab)
            names.append(name)
            vals.append(float(score))
        ax.barh(range(len(names)), vals, color="#4C72B0")
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_xlabel("F1")
        ax.set_title(title)
    _savefig(out_path)


def save_all_training_plots(
    history: list[dict[str, Any]],
    manifest_path: Path,
    plots_dir: Path,
    experiment: str,
    y_true: Optional[list[int]] = None,
    y_pred: Optional[list[int]] = None,
    class_names: Optional[list[str]] = None,
) -> list[str]:
    """Generate standard training graphs; return list of written file paths."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        p1 = plots_dir / f"{experiment}_split_sizes.png"
        plot_split_sizes(manifest, p1)
        written.append(str(p1))
        p2 = plots_dir / f"{experiment}_class_distribution.png"
        plot_class_distribution(manifest, p2)
        written.append(str(p2))

    plot_training_curves(history, plots_dir, experiment)
    if history:
        written.extend(
            [
            str(plots_dir / f"{experiment}_loss_curves.png"),
            str(plots_dir / f"{experiment}_metric_curves.png"),
            str(plots_dir / f"{experiment}_overfit_gap.png"),
            str(plots_dir / f"{experiment}_lr_curve.png"),
        ]
        )

    if y_true is not None and y_pred is not None and len(y_true) > 0:
        cm_path = plots_dir / f"{experiment}_confusion_top30.png"
        plot_confusion_matrix(y_true, y_pred, cm_path, class_names=class_names, top_k=30)
        written.append(str(cm_path))
        written.append(str(cm_path.with_suffix(".npy")))
        f1_path = plots_dir / f"{experiment}_per_class_f1.png"
        plot_per_class_f1(y_true, y_pred, f1_path, class_names=class_names, top_k=30)
        written.append(str(f1_path))

    return written
