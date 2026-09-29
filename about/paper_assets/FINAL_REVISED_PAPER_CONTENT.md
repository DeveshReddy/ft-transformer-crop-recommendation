# AI-Powered Crop Recommendation System Using FT-Transformer for Precision Agriculture

**Mrs. M. Ramya**, **Nisvan Mohamed**, **M. Devesh Reddy**  
Department of Computing Technologies, SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu, India  
ramyam5@srmist.edu.in, Nm8383@srmist.edu.in, Dm1092@srmist.edu.in

---

## Abstract
This paper presents an AI-powered crop recommendation system using a Feature Tokenizer Transformer (FT-Transformer) for tabular agronomic prediction. The model takes seven inputs, including soil nitrogen, phosphorus, potassium, pH, temperature, humidity, and rainfall, and predicts crop suitability with top-k confidence scores. A dual-source data pipeline integrates Kaggle and Mendeley datasets through schema mapping, label normalization, and rainfall imputation where required. The revised training protocol uses stratified train-validation-test splitting, train-only feature scaling, validation-guided checkpointing, adaptive learning-rate reduction, and early stopping to reduce leakage and improve generalization reliability. The final system is deployed through a Streamlit web application with optional OpenWeatherMap-based weather integration and agronomy-aware advisory outputs such as NPK gap guidance, seasonal suitability, and crop calendar recommendations. On the held-out test set, FT-Transformer achieves 0.9054 accuracy and 0.9042 weighted F1-score. Comparative experiments show strong competitiveness against classical baselines, with superior performance to SVM and MLP and performance close to Random Forest. The paper also analyzes data bias risks in merged datasets, computational cost, deployment scalability, and real-world validation strategy.

**Keywords—** crop recommendation, FT-Transformer, precision agriculture, tabular deep learning, model deployment, Streamlit, OpenWeatherMap

---

## I. INTRODUCTION
Crop selection is a critical decision in agriculture, especially under variable soil fertility and climate conditions. Incorrect selection reduces yield, increases input costs, and amplifies production risk. Data-driven recommendation systems can provide scalable decision support, particularly where agronomic advisory access is limited.

Traditional machine learning models such as Random Forest and Support Vector Machines have been widely applied to crop recommendation. However, their ability to represent complex interactions among soil chemistry and weather factors can be limited in multi-class, mixed-source tabular settings. In this context, transformer-based tabular architectures offer a useful alternative by explicitly learning feature-level interactions through self-attention.

This work proposes an end-to-end crop recommendation system built on FT-Transformer and deployed as an interactive Streamlit application. The manuscript has been revised to address reviewer feedback on methodology clarity, robustness, computational analysis, scalability, and result reporting quality.

### Main Contributions
1. A practical two-source preprocessing pipeline for Kaggle and Mendeley datasets with schema alignment and controlled rainfall imputation.  
2. A leakage-safe training protocol with stratified train-validation-test split and train-only scaling.  
3. Validation-driven model selection with adaptive learning-rate scheduling and early stopping.  
4. End-to-end deployment with live weather option, top-k prediction, and agronomy advisory modules.  
5. Expanded evaluation with baseline comparison, training dynamics, confusion matrix analysis, and computational-cost reporting.

---

## II. RELATED WORK

### A. Machine Learning for Crop and Soil–Climate Recommendation
Precision agriculture increasingly relies on supervised learning to map soil chemistry, weather, and management variables to crop outcomes or suitability labels. Ensemble tree methods—especially Random Forest and gradient-boosted decision trees—are common because they handle mixed-scale numeric inputs, tolerate mild non-linearity, and provide competitive accuracy with modest tuning on medium-sized tabular datasets [14]. Support Vector Machines with non-linear kernels remain in use for multi-class problems but can become costly as the number of classes grows, because many implementations rely on pairwise coupling among classes. Shallow multilayer perceptrons offer additional non-linear capacity but, in standard form, do not explicitly structure how individual input dimensions interact. Surveys and application studies in agricultural informatics emphasize that performance depends strongly on data quality, label consistency, and evaluation protocol (train–test leakage, class imbalance, and regional shift), not only on the choice of classifier [10].

### B. Tabular Deep Learning and Explicit Feature Interaction
Recent work revisits deep networks for tabular data, motivated by the need to model feature interactions beyond additive tree ensembles. TabTransformer contextualizes categorical features using transformer blocks while numeric features are handled in parallel or fused [12]. SAINT introduces row-level and column-level attention mechanisms to capture interactions both within a sample and across samples in a batch [4]. The Feature Tokenizer Transformer (FT-Transformer) tokenizes each numeric (or categorical) field into a learned embedding and applies stacked transformer encoder layers with self-attention over the resulting token sequence [1]. This design is well aligned with agronomic settings where interactions such as soil pH versus nutrient availability or rainfall versus humidity can be decisive: attention weights provide a flexible mechanism to re-weight feature contributions depending on context. The self-attention formulation follows the scaled dot-product framework popularized for sequence modeling [2]. Large-scale empirical comparisons on tabular benchmarks show that strong tree ensembles often remain competitive with deep models on default settings, while tuned deep tabular architectures—including transformer variants—can close or reverse that gap depending on dataset and protocol [11], [13].

### C. Optimization, Regularization, and Training Stability
Training deep tabular models typically relies on adaptive optimizers. AdamW decouples weight decay from adaptive gradient updates and is widely used as a stable default for transformer training [3]. Practical pipelines also employ learning-rate schedules (e.g., reduction on plateau of a validation metric) and early stopping to limit overfitting when capacity is high relative to sample size. Class imbalance—common when merging regional datasets or when some crops are rare—can be mitigated through stratified splitting, class-weighted loss, or resampling; these choices affect both accuracy and fairness across classes and should be reported alongside headline metrics.

### D. Multi-Source Agricultural Data and Generalization Risk
Agricultural datasets collected from different institutions, sensors, or regions often differ in schema, units, missingness, and label definitions. Integrating such sources can increase sample diversity and coverage but also introduces covariate shift and label noise if harmonization is incomplete. Imputation strategies (e.g., filling missing rainfall using crop-level statistics from a reference corpus) preserve feature dimensionality but encode assumptions that should be acknowledged in interpretation and uncertainty analysis. Best practice is to fit preprocessing transforms on training data only, hold out validation data for model selection, and report test performance without repeated tuning on the test set—principles increasingly emphasized in applied ML reporting [10], [11].

### E. From Models to Deployed Decision Support
Many crop recommendation papers stop at offline accuracy tables. Deployed systems must additionally address input acquisition (manual entry versus APIs), credential management, latency, versioning of model artifacts, and user-facing explanation or advisory layers that are not necessarily identical to the model’s internal reasoning. Lightweight web frameworks and API-based weather services are common building blocks for prototypes; reproducible packaging of scalers, encoders, and model weights is essential so that inference matches training normalization [8], [9]. This work differs from purely offline studies by documenting a full pipeline: merged data ingestion, validation-governed training, checkpointed artifacts, inference with optional live weather, and Streamlit-based interaction with agronomy-oriented post-processing.

### F. Positioning of This Work
Relative to classical crop classifiers, we adopt FT-Transformer to target explicit cross-feature interaction modeling while retaining a compact architecture suitable for modest tabular width (seven agronomic inputs). Relative to prior tabular transformer proposals, we emphasize leakage-safe preprocessing, validation-based checkpointing, merged-source bias awareness, and a deployable application layer with top-k outputs and advisory modules. Baseline comparisons against Random Forest, RBF-SVM, and MLP on the same split protocol situate the proposed model against standard practice.

---

## III. DATASETS AND PREPROCESSING

### A. Data Sources
- **Primary:** Kaggle Crop Recommendation dataset.  
- **Secondary (optional):** Mendeley crop dataset (`Mendley_crop.csv`), merged using column mapping rules.

### B. Schema Harmonization
A mapping dictionary converts varying source column names into canonical fields:
`N, P, K, temperature, humidity, rainfall, ph, label`.

### C. Rainfall Imputation
When rainfall is absent in secondary data, rainfall is imputed using:
- class-wise median rainfall from the primary dataset, or
- global median fallback.

### D. Label and Feature Processing
- Labels normalized to lowercase and encoded with `LabelEncoder`.  
- Features standardized using `StandardScaler` fitted on train split only.  
- Missing rows in required columns are removed.

### E. Split Protocol
Current implementation uses stratified splitting:
- Train: ~65%  
- Validation: ~15%  
- Test: ~20%

This split design supports fair model selection and leakage control.

### F. Data Bias Discussion
Merging independent datasets can introduce:
- source-specific covariate shift,
- class-frequency imbalance,
- imputation-induced uncertainty.

Mitigation used:
- strict schema mapping,
- stratified splitting,
- optional class-weighted loss,
- validation-based checkpointing,
- explicit reporting of train/val/test behavior.

---

## IV. PROPOSED METHODOLOGY

### A. FT-Transformer Architecture
Given input \(x \in \mathbb{R}^7\), each feature is projected into token space:
\[
t_i = W_i x_i + b_i
\]
A learnable CLS token is prepended, then transformer blocks process the sequence.

Each block contains:
- multi-head self-attention,
- residual connection + LayerNorm,
- GEGLU feed-forward sublayer,
- residual connection + LayerNorm.

Attention is computed as:
\[
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Final prediction uses CLS representation through a linear classification head.

### B. Training Objective and Optimization
- Loss: CrossEntropyLoss (class-weighted option enabled by default unless disabled).  
- Optimizer: AdamW.  
- LR scheduler: ReduceLROnPlateau on validation F1.  
- Early stopping: triggered by validation F1 stagnation.

### C. Checkpointing
Best checkpoint is selected by **maximum validation F1**, saving:
- `model_state_dict`
- `scaler`
- `label_encoder`
- `num_classes`
- `feature_columns`

---

## V. SYSTEM IMPLEMENTATION AND DEPLOYMENT

### A. Inference Pipeline
`inference.py` supports:
1. checkpoint load,  
2. feature vector construction,  
3. optional live weather fetch from OpenWeatherMap,  
4. scaler transform,  
5. top-k probability prediction.

### B. Streamlit Application
The app provides:
- soil and weather input modes,
- crop-type filtering,
- planned month logic,
- top-k confidence output,
- NPK gap advisory,
- season/sowing/harvest guidance,
- CSV export and history summary.

### C. Architecture
End-to-end architecture is documented in `about/ARCHITECTURE.md` and in the generated architecture figure.

---

## VI. EXPERIMENTAL SETUP

### A. Environment
- Frameworks: PyTorch, scikit-learn, Streamlit  
- Training history used from `checkpoints/history.json`  
- Best model from `checkpoints/best_model.pt`  
- Epochs run: 80  
- Best validation epoch: 69

### B. Baselines
Compared under same split/protocol:
- Random Forest (200 trees)
- SVM (RBF)
- MLP
- FT-Transformer (proposed)

### C. Metrics
Weighted:
- Accuracy
- Precision
- Recall
- F1-score  
plus loss for FT training curves.

---

## VII. RESULTS AND DISCUSSION

### A. FT-Transformer Best Checkpoint Performance

**Table I. FT-Transformer Performance at Best Validation Epoch**

| Split | Accuracy | F1-score | Loss |
|---|---:|---:|---:|
| Train | 0.8980 | 0.8971 | 0.2067 |
| Validation | 0.9039 | 0.9028 | 0.2011 |
| Test | 0.9054 | 0.9042 | 0.1846 |

The train-validation-test gap is small, indicating stable generalization with no severe overfitting.

### B. Baseline Comparison

**Table II. Baseline Comparison**

| Model | Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| Random Forest | 0.9120 | 0.9156 | 0.9120 | 0.9113 |
| SVM (RBF) | 0.8668 | 0.8751 | 0.8668 | 0.8650 |
| MLP | 0.8916 | 0.8949 | 0.8916 | 0.8909 |
| FT-Transformer (Proposed) | 0.9054 | 0.9119 | 0.9054 | 0.9042 |

Interpretation:
- FT-Transformer clearly outperforms SVM and MLP.
- Random Forest is slightly higher in this split, while FT remains competitive and offers better extensibility for richer feature interaction modeling and deep-learning deployment workflows.

### C. Run Stability Snapshot

**Table III. Run Stability Summary (Current Completed Runs)**

| Run Setup | Accuracy | F1-score | Notes |
|---|---:|---:|---|
| FT-Transformer (best validation checkpoint) | 0.9054 | 0.9042 | Epoch 69 |
| FT-Transformer (final epoch) | 0.9002 | 0.8990 | Epoch 80 |
| Random Forest | 0.9120 | 0.9113 | Same split/protocol |
| MLP | 0.8916 | 0.8909 | Same split/protocol |
| SVM (RBF) | 0.8668 | 0.8650 | Same split/protocol |

### D. Computational Cost and Complexity

**Table IV. Computational Cost and Complexity**

| Metric | Value |
|---|---:|
| Trainable parameters | 47,432 |
| Output classes in current merged run | 72 |
| Best epoch | 69 |
| Total epochs run | 80 |
| Estimated epoch time (CPU, batch=64) | 5.236 s |
| Estimated total train time for 80 epochs (CPU) | 6.982 min |
| Single-sample inference latency (CPU) | 0.313 ms |
| Attention complexity per layer | \(O(n^2 d)\), with \(n=8\) tokens |

Given short token length (7 features + CLS), inference remains computationally lightweight.

---

## VIII. FIGURES (INSERT IN PAPER)

Use these generated files:

1. **Fig. 1** Architecture diagram  
   `/Users/niswa/.cursor/projects/Users-niswa-minor-project/assets/project_architecture_style_like_sample.png`

2. **Fig. 2** Training dynamics (loss + F1 vs epoch)  
   `about/paper_assets/fig_training_curves.png`

3. **Fig. 3** Normalized confusion matrix (test set)  
   `about/paper_assets/fig_confusion_matrix.png`

4. **Fig. 4** Per-class F1 chart (top classes)  
   `about/paper_assets/fig_per_class_f1_top15.png`

**Figure captions:**
- Fig. 1. End-to-end architecture of the crop recommendation system showing data pipeline, FT-Transformer internals, and Streamlit deployment flow.  
- Fig. 2. Training dynamics for train, validation, and test sets across epochs. Dashed line marks best validation epoch.  
- Fig. 3. Normalized confusion matrix on held-out test data for the selected best checkpoint.  
- Fig. 4. Per-class F1 distribution for representative classes on test set.

---

## IX. SCALABILITY AND DEPLOYMENT ANALYSIS
Current Streamlit deployment is suitable for prototype and moderate usage. For large-scale agricultural deployment, recommended enhancements are:
- decoupled inference API service,
- weather-response caching and retries,
- queue-based request handling,
- horizontal scaling for concurrent users,
- monitoring for data drift and periodic retraining.

---

## X. REAL-WORLD VALIDATION PLAN
A structured field validation phase is required for production-level trust:
1. collect real farm season data across regions,
2. compare recommendation outcomes with observed yields,
3. gather farmer feedback on recommendation usability,
4. refine model and advisory logic iteratively.

This study demonstrates controlled-data performance and deployment readiness, while field validation is identified as the next step.

---

## XI. CONCLUSION AND FUTURE WORK
This work delivers a full AI-powered crop recommendation system using FT-Transformer with robust preprocessing, validation-safe training, and deployment-ready inference. The revised pipeline improves methodological correctness through train-only scaling, validation-based checkpoint selection, early stopping, and adaptive LR scheduling. The final model achieved 0.9054 test accuracy and 0.9042 weighted F1-score, outperforming SVM and MLP and remaining competitive with Random Forest on the current split. The integrated Streamlit application provides practical advisory outputs beyond classification, including NPK guidance and seasonal planning. Future work will focus on multi-seed robustness expansion, field-level validation with farmer feedback, and scalable inference architecture for large agricultural ecosystems.

---

## ACKNOWLEDGMENT
The authors thank Mrs. M. Ramya for guidance and supervision. The authors also acknowledge the Kaggle and Mendeley dataset contributors and open-source software ecosystems used in this work.

---

## REFERENCES
[1] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, “Revisiting deep learning models for tabular data,” *NeurIPS*, 2021.  
[2] A. Vaswani et al., “Attention is all you need,” *NeurIPS*, 2017.  
[3] I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” *ICLR*, 2019.  
[4] S. Somepalli et al., “SAINT: Improved neural networks for tabular data via row attention and contrastive pre-training,” arXiv:2106.01342, 2021.  
[5] A. Ingle, “Crop Recommendation Dataset,” Kaggle.  
[6] A. Thangatamilan and C. Sagana, “Crop Recommendation Dataset,” Mendeley Data, doi:10.17632/vynxnppr7j.1, 2025.  
[7] OpenWeatherMap, “Current Weather Data API,” https://openweathermap.org/api  
[8] Streamlit Inc., “Streamlit Documentation,” https://streamlit.io  
[9] F. Pedregosa et al., “Scikit-learn: Machine learning in Python,” *JMLR*, 2011.  
[10] K. Liakos et al., “Machine learning in agriculture: A review,” *Sensors*, vol. 18, no. 8, p. 2674, 2018.  
[11] L. Grinsztajn, E. Oyallon, and G. Varoquaux, “Why do tree-based models still outperform deep learning on typical tabular data?” *NeurIPS*, 2022.  
[12] X. Huang et al., “TabTransformer: Tabular data modeling using contextual embeddings,” arXiv:2012.06678, 2020.  
[13] Y. Gorishniy et al., “On embeddings for numerical features in tabular deep learning,” *NeurIPS*, 2022.  
[14] T. Chen and C. Guestrin, “XGBoost: A scalable tree boosting system,” *KDD*, 2016.
