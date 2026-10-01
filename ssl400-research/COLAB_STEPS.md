# Colab run guide — SSL400 ST-GCN

Verified for the current code in `ssl400-research/`.  
**Runtime:** Runtime → Change runtime type → **GPU → T4** → Save.

---

## A. Upload to Google Drive first (on your PC)

Put both folders on Drive like this:

```text
MyDrive/SSL400/
  ssl400-research/          ← code (configs, scripts, src, notebook)
  Dataset - MP - CSV/       ← all landmark CSVs
```

Your local paths today:

- Code: `D:\SSL-RE\report-mc\ssl400-research`
- Data: `D:\SSL-RE\Dataset - MP - CSV`

Upload the whole folders (or zip → upload → unzip on Drive).

---

## B. Code check (what was verified)

| Check | Status |
|---|---|
| **All classes** (`min_samples_per_class: 1`) | Enabled |
| Rare-class-safe split (1-sample → train only) | Enabled |
| Augmentation + weighted sampler + label smoothing | Enabled |
| Training graphs under `results/plots/` | Enabled |
| Scripts add project root to `sys.path` | OK |
| Absolute Drive paths work in `data.yaml` | OK |
| AMP + `torch.load` compatibility | OK |
| `.h5` export | OK |

**Important:** Using all classes (including 1-sample signs) usually **does not raise** overall accuracy vs the old ≥10 filter. Expect more classes covered, but overall % may drop. The new training tweaks aim to recover as much accuracy as possible on Colab.

---

## C. Option 1 — Use the notebook (easiest)

1. Open [Google Colab](https://colab.research.google.com)
2. **File → Upload notebook** → choose `colab_train_stgcn.ipynb`  
   (or open it from Drive)
3. Set **GPU (T4)**
4. Run all cells in order

If your Drive path is not `MyDrive/SSL400/...`, edit `PROJECT_DIR` and `DATASET_DIR` in cell 2.

---

## D. Option 2 — Step-by-step Colab commands

Create a **new Colab notebook**, set **GPU (T4)**, then run each block.

### Step 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 2 — Enter project + check folders

```python
import os
from pathlib import Path

PROJECT_DIR = Path('/content/drive/MyDrive/SSL400/ssl400-research')
DATASET_DIR = Path('/content/drive/MyDrive/SSL400/Dataset - MP - CSV')

assert PROJECT_DIR.exists(), f'Missing: {PROJECT_DIR}'
assert DATASET_DIR.exists(), f'Missing: {DATASET_DIR}'

os.chdir(PROJECT_DIR)
print('CWD:', Path.cwd())
!ls
!ls "{DATASET_DIR}" | head
```

### Step 3 — Confirm GPU

```python
import torch
print(torch.__version__)
print('CUDA:', torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')
```

You must see `CUDA: True`. If not, fix the runtime type.

### Step 4 — Install packages

```bash
!pip install -q -r requirements.txt
```

### Step 5 — Write Colab configs

```python
from pathlib import Path

Path('configs/data.yaml').write_text(f'''
raw_csv_root: "{DATASET_DIR.as_posix()}"
processed_dir: "data/processed"
inspected_dir: "data/inspected"
min_samples_per_class: 1
sequence_length: 64
use_z: true
visibility_threshold: 0.5
mask_low_visibility: true
mask_lower_body: true
train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15
split_seed: 42
save_npz: true
'''.strip() + '\n', encoding='utf-8')

Path('configs/model.yaml').write_text('''
model:
  in_channels: 3
  num_joints: 33
  num_classes: null
  channels: [64, 64, 128, 256]
  temporal_kernel: 9
  dropout: 0.3
  graph_strategy: spatial

train:
  epochs: 100
  batch_size: 32
  learning_rate: 0.0008
  weight_decay: 0.0005
  optimizer: adamw
  scheduler: cosine
  step_size: 30
  gamma: 0.1
  early_stopping_patience: 20
  num_workers: 0
  amp: true
  class_weights: true
  weighted_sampler: true
  label_smoothing: 0.05
  augment: true
  seed: 42
  device: auto

paths:
  checkpoint_dir: "checkpoints"
  results_dir: "results"
  plots_dir: "results/plots"
  experiment_name: "stgcn_allclasses_seed42"
'''.strip() + '\n', encoding='utf-8')

print('Configs ready')
```

### Step 6 — Quick import smoke test

```python
from src.models.stgcn import STGCN
import torch

m = STGCN(num_classes=10)
x = torch.randn(2, 3, 64, 33)
y = m(x)
print('Forward OK:', y.shape)  # expect torch.Size([2, 10])
```

### Step 7 — Prepare dataset (once, can take a while)

```bash
!python scripts/prepare_dataset.py --data-config configs/data.yaml
```

Verify:

```python
import json
from pathlib import Path
import numpy as np

class_map = json.loads(Path('data/processed/class_to_idx.json').read_text())
manifest = json.loads(Path('data/processed/manifest.json').read_text())
t = np.load(manifest['train'][0]['tensor_path'])
print('classes', len(class_map))
print('splits', {k: len(v) for k, v in manifest.items()})
print('tensor', t.shape, t.dtype)  # expect (3, 64, 33) float32
```

### Step 8 — Train

```bash
!python scripts/train_stgcn.py --data-config configs/data.yaml --model-config configs/model.yaml
```

Outputs:

- `checkpoints/stgcn_allclasses_seed42_best.pt`
- `checkpoints/stgcn_allclasses_seed42_last.pt`
- `results/stgcn_allclasses_seed42_history.json`
- `results/plots/*.png` (loss, metrics, LR, class distribution, confusion, F1)

### Step 9 — Evaluate test set

```bash
!python scripts/evaluate_stgcn.py --checkpoint checkpoints/stgcn_allclasses_seed42_best.pt --split test --data-config configs/data.yaml --model-config configs/model.yaml
```

### Step 10 — Export model to `.h5`

```bash
!python scripts/export_h5.py --checkpoint checkpoints/stgcn_allclasses_seed42_best.pt --output checkpoints/stgcn_allclasses_seed42_best.h5
```

Creates `checkpoints/stgcn_allclasses_seed42_best.h5` (PyTorch weights in HDF5, not a Keras model file).

---

## E. If something fails

| Error | Fix |
|---|---|
| `Missing dataset` / `raw_csv_root not found` | Fix `DATASET_DIR` path in Step 2/5 |
| `CUDA: False` | Runtime → GPU (T4) |
| `CUDA out of memory` | In model.yaml set `batch_size: 16` or `8`, re-run Step 5 + 8 |
| `No module named src` | You are not in `ssl400-research` (`os.chdir` failed) |
| Prepare very slow | Normal on Drive with ~4k CSVs; wait |
| Session disconnected | Remount Drive, `chdir`, skip prepare if `data/processed` exists, train again |

---

## F. What success looks like

1. Smoke test prints `Forward OK: torch.Size([2, 10])`
2. Prepare prints class/split counts and finishes without error
3. Train prints epoch lines with `val_acc` / `val_f1_macro`
4. Best checkpoint file appears under `checkpoints/`
5. Evaluate prints test accuracy / F1
