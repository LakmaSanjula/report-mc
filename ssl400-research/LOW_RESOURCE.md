# Low-resource SSL — what is done and what is still required

This project follows [SSL400_Development_Guide.md](../SSL400_Development_Guide.md). Sinhala Sign Language here is a **low-resource** setting: few examples per sign, strong class imbalance, pose-only landmarks, and no signer IDs.

## Already in the training code

| Technique | Why it fits low-resource SSL | Where |
|---|---|---|
| MediaPipe Pose sequences instead of raw video | Much less compute; trainable on Colab | `src/data/prepare.py` |
| Keep classes with at least 10 samples | 1-sample classes cannot be learned and dropped accuracy to ~25% | `configs/data.yaml` |
| Rare-class-safe split | Very small classes do not break the split | `src/data/prepare.py` |
| Mild landmark augmentation | Extra views without collecting new signers | `src/data/dataset.py` |
| Signing-speed resample | Same sign, different speed; also named in the guide's robustness section | `src/data/dataset.py` |
| Focal loss (gamma 1) | Pays more attention to hard/scarce classes without the accuracy collapse of full rebalancing | `src/training/losses.py` |
| Dropout in the ST-GCN | Regularizes a small dataset and enables MC Dropout | `src/models/stgcn.py` |
| Top-5 accuracy | The guide's human check only corrects inside the Top-5 | training logs and plots |
| MC Dropout evaluation | Uncertainty from the **same** model; no second network | `scripts/mc_dropout_eval.py` |

Not used, because they hurt this dataset: training all 383 classes, weighted sampling, joint dropout, and left-right mirroring.

## Still required by the research guide (not in this training package yet)

These are the next phases. They should stay separate from this Colab training run:

1. Softmax confidence and predictive entropy (single forward pass).
2. Temperature scaling on the **validation** set only.
3. Accept / verify policy and Top-5 human correction.
4. Replay memory of verified samples only.
5. Balanced replay and periodic continual learning.
6. Forgetting and ablation comparisons.
7. Webcam path using the **same** 33 pose landmarks.

## Colab

Open `colab_train_stgcn.ipynb`, select GPU T4, and run the cells in order. The notebook rebuilds data, trains, evaluates, runs MC Dropout on validation, exports `.h5`, and shows the graphs.
