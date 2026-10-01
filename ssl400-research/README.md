# SSL400 ST-GCN — Model & Training

Word-level **Sinhala Sign Language** recognition using an **ST-GCN** (Spatial-Temporal Graph Convolutional Network) on the SSL400 MediaPipe Pose CSV landmarks.

This package contains **only** the model and the code needed to prepare data, train, and evaluate it. Uncertainty, human-in-the-loop, continual learning, webcam, and LLM features are not included here.

---

## What we did here

1. Inspected SSL400 `Dataset - MP - CSV` and confirmed the format:
   - One CSV per sample, **headerless**
   - Each row = one frame
   - **33 columns** = MediaPipe **Pose** joints
   - Each cell = `[x, y, z, visibility]`
   - Label = parent folder name (`Category/Class/...`)
2. Chose this CSV set as the **only** training representation available in the project folder (no hand/face landmarks, no official splits, no signer IDs).
3. Built a training pipeline that:
   - Keeps classes with **≥ 10 samples** (severe imbalance otherwise)
   - Creates a stratified **70% / 15% / 15%** train/val/test split
   - Converts each CSV to an ST-GCN tensor `(C, T, V) = (3, 64, 33)`
   - Defines a **33-node Pose graph** matching MediaPipe joint order
   - Implements **ST-GCN** with **Dropout(0.5)** after pooling (MC Dropout–ready later)
4. Provides scripts to **prepare → train → evaluate** and save checkpoints under `checkpoints/`.

---

## Requirements

| Item | Detail |
|---|---|
| Python | 3.11 or 3.12 |
| OS | Windows / Linux / Colab |
| GPU | Optional but recommended (Colab Free GPU is fine) |
| Dataset | Sibling folder `Dataset - MP - CSV` (see path below) |

### Python packages (`requirements.txt`)

- `torch` — ST-GCN training
- `numpy`, `pandas` — arrays / CSV handling
- `PyYAML` — configs
- `scikit-learn` — stratified split + metrics
- `tqdm` — progress bars
- `matplotlib` — optional plots later

Install:

```powershell
cd D:\SSL-RE\ssl400-research
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Expected dataset layout:

```text
D:\SSL-RE\
├── Dataset - MP - CSV\     ← raw CSVs (Category / Class / Class_NNN.csv)
└── ssl400-research\        ← this code
```

If your CSV path differs, edit `configs/data.yaml` → `raw_csv_root`.

---

## Folder structure

```text
ssl400-research/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   ├── data.yaml             ← dataset path, filter, T, split
│   └── model.yaml            ← ST-GCN + training hyperparameters
├── scripts/
│   ├── prepare_dataset.py    ← CSV → tensors + splits
│   ├── train_stgcn.py        ← train model
│   └── evaluate_stgcn.py     ← evaluate checkpoint
└── src/
    ├── models/stgcn.py       ← ST-GCN model (+ dropout before classifier)
    ├── graph/pose_graph.py   ← MediaPipe Pose adjacency graph
    ├── data/                 ← load / prepare CSVs
    ├── training/             ← train / eval loops
    └── utils/                ← config + seed helpers
```

Running prepare/train automatically creates `data/`, `checkpoints/`, and `results/` when needed.
---

## What happens in the model

### Input

Each sample becomes a tensor:

```text
(C, T, V) = (3, 64, 33)
```

- **C = 3** → `x, y, z` (visibility is used only as a mask, not as a channel)
- **T = 64** → fixed frames (pad or center-crop)
- **V = 33** → MediaPipe Pose joints

Coordinates are normalized per sample (mid-hip translation, shoulder-width scale). Low-visibility joints (`vis < 0.5`) are zeroed.

### Graph

Joints are nodes. Anatomical Pose links are edges (shoulders–elbows–wrists, torso, legs, etc.). ST-GCN uses a **spatial partition** adjacency (self / centripetal / centrifugal), same idea as the original ST-GCN paper.

### Network flow

```text
Input (N, C, T, V)
  → BatchNorm
  → ST-GCN blocks (64 → 64 → 128 → 256)
       • spatial graph convolution over joints
       • temporal convolution over frames
  → Global average pool (over time + joints)
  → Dropout(0.5)          ← for regularization now; MC Dropout later
  → Linear classifier
  → logits (one score per sign class)
```

Training uses cross-entropy (optionally class-weighted for imbalance). The best checkpoint is chosen by **validation macro-F1**.

### Important limitation

These CSVs are **Pose-only** (body). Finger/handshape detail is missing, so recognition accuracy has a natural ceiling for many SSL signs. That is a dataset limit, not a training bug.

---

## How to train

Run all commands from `ssl400-research/`.

### 1) Prepare data (once)

```powershell
python scripts/prepare_dataset.py
```

Creates tensors, class map, and train/val/test lists under `data/processed/`.

### 2) Train ST-GCN

```powershell
python scripts/train_stgcn.py
```

Writes:

- `checkpoints/baseline_stgcn_seed42_best.pt`
- `checkpoints/baseline_stgcn_seed42_last.pt`
- `results/baseline_stgcn_seed42_history.json`

### 3) Evaluate on the frozen test set

```powershell
python scripts/evaluate_stgcn.py --checkpoint checkpoints/baseline_stgcn_seed42_best.pt --split test
```

### Useful config changes

| File | Key | Meaning |
|---|---|---|
| `configs/data.yaml` | `min_samples_per_class` | Drop rare classes (default 10) |
| `configs/data.yaml` | `sequence_length` | Fixed frames `T` (default 64) |
| `configs/model.yaml` | `train.batch_size` | Lower if out of memory |
| `configs/model.yaml` | `train.device` | `auto` / `cpu` / `cuda` |
| `configs/model.yaml` | `train.amp` | `true` on GPU for faster training |
| `configs/model.yaml` | `model.dropout` | Dropout before classifier (default 0.5) |

After changing data settings, run **prepare** again, then train.

### Colab tip

Mount Drive, `cd` into this folder, `pip install -r requirements.txt`, set `raw_csv_root` if needed, then run the same three scripts. Prefer GPU + `train.amp: true`.

---

## What this package does **not** do

- Does not train automatically when you open the project (you must run the scripts)
- Does not include uncertainty comparison, Top-5 verification, replay memory, or LLM demos
- Does not claim unseen-signer evaluation (no signer IDs in the CSVs)
