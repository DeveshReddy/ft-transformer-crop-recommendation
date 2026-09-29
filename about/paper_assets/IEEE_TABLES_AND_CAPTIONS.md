# IEEE Tables and Figure Captions (Ready to Paste)

Use these directly in your revised conference paper.

---

## Table I. FT-Transformer Performance at Best Validation Epoch

Best validation epoch identified from `checkpoints/history.json`: **Epoch 69**

| Split | Accuracy | F1-score | Loss |
|---|---:|---:|---:|
| Train | 0.8980 | 0.8971 | 0.2067 |
| Validation | 0.9039 | 0.9028 | 0.2011 |
| Test | 0.9054 | 0.9042 | 0.1846 |

Notes:
- These values are from the model selected by **maximum validation F1**.
- Training used stratified train/val/test split with train-only scaler fitting.

---

## Table II. Baseline Comparison (Fill with your measured RF/SVM/MLP values)

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9120 | 0.9156 | 0.9120 | 0.9113 |
| SVM (RBF) | 0.8668 | 0.8751 | 0.8668 | 0.8650 |
| MLP | 0.8916 | 0.8949 | 0.8916 | 0.8909 |
| **FT-Transformer (Proposed)** | **0.9054** | **0.9097** | **0.9054** | **0.9042** |

---

## Table III. Run Stability Summary (Current Completed Runs)

| Run Setup | Accuracy | F1-score | Notes |
|---:|---:|---:|---:|
| FT-Transformer (best validation checkpoint) | 0.9054 | 0.9042 | Epoch 69 |
| FT-Transformer (final epoch) | 0.9002 | 0.8990 | Epoch 80 |
| Random Forest | 0.9120 | 0.9113 | Same split/protocol |
| MLP | 0.8916 | 0.8909 | Same split/protocol |
| SVM (RBF) | 0.8668 | 0.8650 | Same split/protocol |

---

## Table IV. Computational Cost and Complexity (Recommended format)

| Metric | Value |
|---|---:|
| Trainable parameters | 47,432 |
| Number of output classes in current merged run | 72 |
| Best epoch | 69 |
| Total epochs run | 80 |
| Estimated epoch time (CPU, batch=64) | 5.236 s |
| Estimated total train time for 80 epochs (CPU) | 6.982 min |
| Single-sample inference latency (CPU) | 0.313 ms |
| Complexity per transformer layer | O(n^2 d), with n = 8 tokens |

Complexity note for paper text:
- With 7 feature tokens plus one CLS token, sequence length is 8.
- Self-attention cost per layer scales as O(n^2 d), where n is small (8), making inference feasible.

---

## Figure Captions (IEEE style)

- **Fig. 1.** End-to-end architecture of the proposed crop recommendation system showing the data and training pipeline, FT-Transformer model internals, and Streamlit deployment workflow.
- **Fig. 2.** Training dynamics across epochs. Left: train/validation/test loss curves. Right: train/validation/test F1 curves. The dashed line marks the best validation epoch.
- **Fig. 3.** Normalized confusion matrix on the held-out test set for the best FT-Transformer checkpoint.
- **Fig. 4.** Per-class F1 performance for representative classes on the test set.
- **Fig. 5.** Streamlit user interface screenshot showing top-k recommendations, confidence bars, and advisory outputs.

---

## Generated Figure Files

Use these files in the paper:

- `about/paper_assets/fig_training_curves.png`
- `about/paper_assets/fig_confusion_matrix.png`
- `about/paper_assets/fig_per_class_f1_top15.png`

Existing architecture image:

- `/Users/niswa/.cursor/projects/Users-niswa-minor-project/assets/project_architecture_style_like_sample.png`

---

## Text Snippet for Results Section (Ready to Paste)

The FT-Transformer converged stably over 80 epochs and achieved a best validation F1-score of 0.9028 at epoch 69. At this checkpoint, train/validation/test accuracies were 0.8980, 0.9039, and 0.9054, respectively, with corresponding F1-scores of 0.8971, 0.9028, and 0.9042. The small train-validation gap and monotonic loss reduction indicate good generalization and no evidence of severe overfitting. Learning-rate reduction in later epochs refined convergence, while early-stopping control prevented unnecessary training once validation gains saturated.

From functional validation, all modules passed including data ingestion, dataset merge alignment, checkpoint save/reload, inference pipeline, and crop recommendation output. The deployed application supports all seven input features and produces top-k recommendations with advisory context.
