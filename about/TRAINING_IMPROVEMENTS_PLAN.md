# Training Improvements Implementation Plan

This document describes the plan to improve model accuracy without hurting performance. All changes preserve checkpoint format so inference and the web app keep working.

---

## 1. Train / Validation / Test Split

- **Current:** 80% train, 20% test; "best" model chosen by test F1 (leaks test into model selection).
- **Change:** 70% train, 15% validation, 15% test (stratified).
- **Scaler:** Fit `StandardScaler` on **train only**, then transform train, val, and test. Prevents information leakage from val/test into normalization.
- **Result:** Validation set used only for model selection and early stopping; test used only for final reporting.

---

## 2. Model Selection and Early Stopping

- **Best model:** Save checkpoint when **validation F1** improves (not test F1).
- **Early stopping:** If validation F1 does not improve for `patience` epochs (default 10), stop training. The best checkpoint is already saved on disk.
- **Final metrics:** At the end, optionally evaluate the saved best model on the test set once and print test metrics (so we don't overfit to test).

---

## 3. Learning Rate Scheduler

- **Scheduler:** `ReduceLROnPlateau` on **validation F1** (mode='max').
- **Defaults:** factor=0.5, patience=5 (reduce LR when val F1 has not improved for 5 epochs).
- **Min LR:** Stop reducing below a floor (e.g. 1e-6) to avoid vanishing updates.
- **Effect:** Refines training in later epochs without large, unstable steps.

---

## 4. Class Weights (Optional)

- **Problem:** Imbalanced crop classes can cause underfitting on rare crops.
- **Change:** Compute balanced class weights from training labels (`sklearn.utils.class_weight.compute_class_weight('balanced', ...)`) and pass to `CrossEntropyLoss(weight=...)`.
- **CLI:** Add flag `--no_class_weights` to disable (default: use class weights).
- **Effect:** Model pays more attention to minority classes without changing checkpoint format.

---

## 5. What Stays the Same (No Regression Risk)

- **Model architecture defaults:** d_token=64, n_layers=3, n_heads=4, d_ff=128, dropout=0.1. Unchanged so current behavior is preserved.
- **Checkpoint format:** Same keys (model_state_dict, scaler, label_encoder, num_classes, feature_columns). Inference and app load it as before.
- **Data pipeline:** Merge, preprocessing, and feature columns unchanged. Only the split ratios and scaler fitting change.

---

## 6. Implementation Order

1. Add train/val/test split in `load_and_preprocess`; fit scaler on train only.
2. Update `get_dataloaders` to build and return a validation DataLoader.
3. In `main`: use validation F1 for saving best model; add early stopping loop.
4. Add `ReduceLROnPlateau` and call `scheduler.step(val_f1)` after each epoch.
5. Compute class weights from y_train and pass to CrossEntropyLoss (with --no_class_weights to disable).
6. Add CLI args (--val_ratio, --patience, --scheduler_patience, --min_lr, --no_class_weights); store train/val/test metrics in history.json.

---

## 7. New CLI Arguments (Optional Tuning)

| Argument | Default | Description |
|----------|---------|-------------|
| `--val_ratio` | 0.15 | Fraction of data for validation (test is 0.20, train is the rest). |
| `--patience` | 10 | Early stopping: stop if val F1 does not improve for this many epochs. |
| `--scheduler_patience` | 5 | ReduceLROnPlateau: reduce LR after this many epochs without val F1 gain. |
| `--min_lr` | 1e-6 | Minimum learning rate; scheduler will not go below this. |
| `--no_class_weights` | False | Disable balanced class weights in loss. |

All existing CLI arguments remain unchanged; defaults keep current behavior where possible.
