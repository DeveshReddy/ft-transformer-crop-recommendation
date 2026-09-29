# Full Project Analysis: Crop Recommendation with FT-Transformer

This document provides a detailed text analysis of the Crop Prediction project: its purpose, architecture, data flow, components, strengths, and limitations.

---

## 1. Project overview and objectives

The project is an **AI-powered crop recommendation system** that suggests the best crop(s) to grow based on **seven input factors**: soil nutrients (Nitrogen N, Phosphorus P, Potassium K, and pH) and weather-related variables (temperature, humidity, and rainfall). It replaces traditional machine learning models (e.g. Random Forest, SVM) with a **Feature Tokenizer Transformer (FT-Transformer)** to capture complex, non-linear interactions between soil fertility and climate. The system is designed for both **offline training** and **live use**: users can enter soil and weather manually or fetch current weather by city via the OpenWeatherMap API, and receive a ranked list of recommended crops plus auxiliary information (NPK fertilizer guidance, water needs, sowing and harvest calendar, season suitability).

**Primary deliverables:** (1) A PyTorch FT-Transformer model and training pipeline, (2) Inference logic with optional live weather integration, and (3) A Streamlit web application that exposes the full workflow with a modern UI and rich per-crop metadata.

---

## 2. Data pipeline and datasets

**Primary dataset:** The system expects a CSV with seven numerical features and a crop label. The default source is the **Kaggle Crop Recommendation Dataset** (e.g. `Crop_recommendation.csv`), with columns: N, P, K, temperature, humidity, rainfall, ph, and label (or crop). The training script can **auto-download** this dataset via the Kaggle API when the file is missing, storing it under a `data/` directory.

**Optional second dataset (merge):** A second CSV (e.g. the Mendeley “Crop recommendation” dataset by Thangatamilan, Sagana 2025) can be **merged** with the primary one to increase sample size and diversity. The merge logic in `train.py`:

- Maps secondary column names to the canonical set (N, P, K, temperature, humidity, ph, rainfall, label) using a configurable dictionary (`MENDELEY_STYLE_MAP`). For example, CROPS or TYPE_OF_CROP map to label; TEMP to temperature; RELATIVE_HUMIDITY to humidity; SOIL_PH to ph.
- If the secondary dataset has no rainfall column, **rainfall is imputed** from the primary dataset: for each row, the median rainfall for that crop in the primary data is used when available, otherwise the global median. This keeps the model’s input dimension fixed at seven.
- Both dataframes are concatenated; labels are normalized (e.g. lowercase) so that “Rice” and “rice” are treated as the same class. Rows with missing values in the seven features or label are dropped.
- The merged CSV is saved to a user-specified path (e.g. `data/Merged_crop_recommendation.csv`).

**Preprocessing (single or merged CSV):** After loading (and optionally merging), the pipeline:

- Ensures a unique target column (label or crop) and normalizes column names (e.g. pH → ph where needed).
- Extracts the seven feature columns and the target. Features are cast to float32; target to string for encoding.
- Fits a **StandardScaler** on the feature matrix and transforms it so that all inputs are zero-mean, unit-variance. This scaler is saved with the checkpoint and reused at inference so that user or API-sourced inputs are scaled identically.
- Fits a **LabelEncoder** on crop names, producing integer class indices. The number of classes is determined by the unique labels in the dataset (e.g. 22 for the original Kaggle set; more if the merged set introduces new crops).
- Splits the data **80/20** into train and test sets with **stratified sampling** so that class proportions are preserved. This avoids biased metrics when classes are imbalanced.

The pipeline is deterministic for a given random seed (default 42), so the same CSV and seed yield the same splits and training behaviour.

---

## 3. Model architecture (FT-Transformer)

The core predictor is implemented in `model.py` and consists of the following building blocks.

**Feature tokenizer:** Each of the seven numerical inputs is treated as a separate “token.” A linear projection maps each scalar to a `d_token`-dimensional vector (default 64). Formally, for feature index \(f\), the output is \(x_f \cdot w_f + b_f\), where \(w_f\) and \(b_f\) are learned. The result is a tensor of shape (batch, 7, d_token), i.e. a sequence of seven tokens per sample.

**CLS token:** A learnable vector of dimension `d_token` is prepended to the sequence (similar to BERT/ViT). After prepending, the sequence length is 8 (1 CLS + 7 feature tokens). The CLS token is used to aggregate information from all tokens via self-attention and is the only position used for the final classification.

**Transformer backbone:** A stack of identical layers (default 3). Each layer comprises:

- **Multi-head self-attention (MHSA)** over the 8 positions, with residual connection and pre-norm LayerNorm. This allows the model to learn interactions between any pair of features (e.g. nitrogen and rainfall, or pH and temperature).
- A **position-wise feed-forward network (FFN)** with **GEGLU** activation: the hidden dimension is expanded (e.g. to 2×d_ff), split into two halves, one half is passed through GELU and multiplied by the other, then projected back to d_token. Dropout is applied. Again, residual connection and LayerNorm are used.

Default hyperparameters: d_token=64, n_heads=4, n_layers=3, d_ff=128, dropout=0.1. The model is flexible to different numbers of classes because the final linear layer projects from d_token to `num_classes`, which is inferred from the training set.

**Classification head:** The representation at the CLS position (after all transformer layers) is passed through dropout and a linear layer to produce logits of shape (batch, num_classes). Training uses cross-entropy loss; at inference, softmax is applied and the top-k classes are returned with their probabilities.

This design is well-suited to tabular data because it does not assume a fixed grid or sequence order beyond the seven “positions”; the attention mechanism learns which feature combinations matter for each crop.

---

## 4. Training pipeline

Training is orchestrated in `train.py`. After data loading and preprocessing (and optional merge), the script:

- Builds PyTorch **TensorDatasets** and **DataLoaders** for train and test sets. Batch size is configurable (default 64). On CUDA, `pin_memory` is enabled for faster transfer; on MPS (Apple Silicon) it is disabled to avoid warnings.
- Instantiates the **FTTransformer** with the inferred `num_classes` and the chosen hyperparameters (d_token, n_heads, n_layers, d_ff, dropout).
- Uses **CrossEntropyLoss** and **AdamW** optimizer with configurable learning rate and weight decay (default 1e-3 and 0.01). This encourages generalization and stable training.
- Runs a fixed number of **epochs** (default 50). Each epoch: one full pass over the training set with gradient updates; then evaluation on the test set. For both train and test, the script computes **loss**, **accuracy**, **precision**, **recall**, and **F1-score** (weighted over classes).
- Saves the **best** model by test F1: whenever the current epoch’s test F1 exceeds the previous best, the script writes to `checkpoints/best_model.pt` the model state dict, the fitted StandardScaler, the LabelEncoder, num_classes, and feature column names. Training history (train and test metrics per epoch) is written to `checkpoints/history.json` for later analysis or plotting.

Random seeds for PyTorch and NumPy are set so that runs are reproducible. The same data and arguments yield the same training curve and best checkpoint (up to hardware non-determinism).

---

## 5. Inference and external APIs

Inference is implemented in `inference.py` and is used by both the CLI and the Streamlit app.

**Checkpoint loading:** `load_checkpoint` reads `best_model.pt`, reconstructs the FTTransformer with the saved num_classes and fixed architecture (7 features, d_token=64, etc.), loads the state dict, and returns the model, scaler, label_encoder, and device. The model is set to eval mode. Dropout is effectively disabled at inference.

**Feature construction:** Two entry points exist. `build_features_from_user` takes soil (N, P, K, ph) and optional weather (temperature, humidity, rainfall). If any of the weather values are missing, it can call the OpenWeatherMap API (when an API key and city or lat/lon are provided) to fetch current temperature (°C), humidity (%), and rainfall (mm), and then builds a single row of seven features in the order expected by the model. `build_features_and_weather` does the same but also returns a small dictionary describing the weather used (temperature, humidity, rainfall, and city or None for manual entry), so the UI can display “Weather used: Chennai — 28°C, 78%, 0 mm.”

**OpenWeatherMap:** The script uses the REST API (e.g. `api.openweathermap.org/data/2.5/weather`) with query by city name or by latitude/longitude. Units are metric. Rainfall is taken from the current conditions (e.g. rain.1h or rain.3h if present; otherwise 0). The same scaler fitted at training is applied to the raw feature vector before passing it to the model, so the recommendation is consistent with the training distribution.

**Top-k prediction:** `predict_top_k` accepts the loaded model, scaler, label_encoder, a (1, 7) or (7,) feature array, device, and k (default 3). It scales the features, runs the model, applies softmax, and returns a list of (crop_name, probability) for the top-k classes, sorted by probability descending. Crop names come from `label_encoder.classes_`.

The CLI mode (`run_interactive`) prompts the user for N, P, K, pH, then either a city name or manual temperature, humidity, and rainfall, and prints the top-three crops. This is useful for quick tests without the web UI.

---

## 6. Web application (Streamlit)

The Streamlit app in `app.py` provides the main user-facing interface. It loads the checkpoint once (cached with `@st.cache_resource`) and then renders a single-page form and results.

**Layout and UX:** The app uses a gradient header banner, optional “Planned start month” dropdown (defaulting to the current month) for season and timing messages, and an optional **crop type filter** (multiselect: Cereal, Pulses, Fruit, etc.). When the filter is set, the model returns a larger top-k (e.g. 10), results are filtered by the selected types, and the first three are shown. Soil inputs are implemented as **sliders** (N, P, K, pH) with sensible ranges and hints. Weather can be supplied either by **city name** (triggering OpenWeatherMap) or by **manual** temperature, humidity, and rainfall. A single primary button triggers “Get detailed crop recommendation.”

**Results presentation:** After a successful run, the app displays:

- A **blue weather card** summarizing the weather used (location and three values).
- A **hero “Recommended crop” card** (dark green) for the top-ranked crop, with name, type, and confidence.
- A **grid of six detail cards** for the top crop: type, sow-in months, duration, current season, harvest months, water source. Data for these comes from `crop_metadata.py`.
- An **NPK fertilizer guide** for the top crop: ideal N/P/K ranges (from metadata) and an actionable line per nutrient (add X kg/ha, adequate, or reduce), derived by comparing user inputs to the ideal ranges.
- A **timing alert bar**: “Great timing!” if the chosen month is in the crop’s suitable seasons, or a suggestion to consider sowing in the crop’s typical sowing months.
- A **sowing advisory bar**: “If you sow in [month], estimated harvest: [harvest months].”
- A **Top 3 crop confidence** section with horizontal bars and percentages for all three recommendations.
- **Expandable “Details for #2 / #3”** sections with type, water, sowing/harvest, duration, NPK summary, and season suitability for the second and third crops.

**Export and history:** The user can download the current recommendation as a CSV (rank, crop, score, type, water, duration, sowing, harvest, NPK note). The last run is stored in `st.session_state` and can be viewed in a collapsible “View last recommendation summary” section.

**Error handling:** Missing or invalid OpenWeatherMap API key, missing checkpoint, or API failures are caught and shown in the UI with clear messages (e.g. suggest adding the key to `.env` or using manual weather). Paths for the checkpoint and .env are resolved relative to the project directory so the app works regardless of the terminal’s current working directory.

---

## 7. Crop metadata and auxiliary logic

The file `crop_metadata.py` holds **static metadata** for each crop (keyed by lowercase crop name as in the dataset). For each crop, the metadata includes: crop type (Cereal, Pulses, Fruit, etc.), water requirement in mm and a category (Low/Medium/High), duration in days, sowing and harvest months (1–12), ideal N/P/K ranges (min–max), water source (Rainfed/Irrigated/Both), and a list of suitable seasons (Winter, Summer, Monsoon, Post-Monsoon). The module also defines a month-to-season mapping (India-style) and helpers: `get_metadata(crop_name)` (with defaults for unknown crops), `get_current_season(month)`, `npk_gaps(user_N, user_P, user_K, meta)` (returns short messages like “N: add ~20 kg/ha” or “P: adequate”), and `month_names` / `month_name` for readable month labels.

This metadata **does not** affect the model’s predictions; it only enriches the UI with interpretable advice (NPK gap, water advisor, sowing/harvest calendar, season suitability). The FT-Transformer is trained purely on the seven numerical features and the crop label. The app uses the metadata only after the model has produced the top-k crops.

---

## 8. Configuration and environment

The project uses a `.env` file (template `.env.example`) for secrets and configuration. Expected variables include: **OPENWEATHERMAP_API_KEY** (for live weather in the app and inference), and optionally **KAGGLE_USERNAME** and **KAGGLE_KEY** (for auto-download of the primary dataset). The app and inference load `.env` from the project root via `python-dotenv`, so the same keys work whether the app is run from the project directory or elsewhere. The training script disables `pin_memory` on non-CUDA devices to avoid MPS warnings on Apple Silicon.

Dependencies are listed in `requirements.txt`: PyTorch, NumPy, Pandas, scikit-learn, requests, kaggle, python-dotenv, and Streamlit. The codebase is structured so that training and inference can run without Streamlit if only the model and CLI are needed.

---

## 9. Strengths of the project

- **Modern model choice:** The FT-Transformer is a strong fit for tabular crop data: it models feature interactions via self-attention and avoids hand-crafted feature engineering. The feature tokenizer and CLS-based pooling are standard and interpretable.
- **Unified preprocessing:** A single StandardScaler and LabelEncoder are fitted on the training (or merged) data and saved with the checkpoint. All inference paths (CLI, app, manual or API weather) use the same scaler, so recommendations are consistent and avoid train–test skew.
- **Flexible data sources:** Support for merging a second dataset with column mapping and rainfall imputation allows combining Kaggle and Mendeley (or similar) data without changing the model or inference code. New crops from the secondary set are naturally included via the label encoder.
- **Live weather integration:** OpenWeatherMap integration lets users get recommendations based on current local conditions (by city or coordinates), which is useful for real-world deployment. The design clearly separates “build feature vector” from “call model,” so the same logic supports both API and manual weather.
- **Rich application layer:** The Streamlit app goes beyond “top-3 crops” by adding crop type filter, NPK fertilizer guide, water requirement and source, sowing/harvest calendar, and season suitability. This makes the system more actionable for farmers or advisors.
- **Reproducibility:** Fixed seeds, stratified split, and saved artifacts (scaler, label encoder, history) make experiments and reports reproducible. The architecture is documented in `ARCHITECTURE.md` with block diagrams.

---

## 10. Limitations and considerations

- **Metadata coverage:** Crop metadata (NPK ideals, water, calendar, seasons) is hand-defined for the 22 Kaggle crops and may not match every new crop introduced by a merged dataset. Unknown crops fall back to generic defaults, which might be less accurate for NPK or calendar advice.
- **Rainfall imputation:** When the secondary dataset has no rainfall column, imputation from the primary set (by crop or global median) is a simplification. It can bias the merged distribution and may not reflect true regional variation. For production, having real rainfall (or a dedicated rainfall model) for the secondary source would be preferable where possible.
- **Weather API limitations:** OpenWeatherMap’s current weather may not include rainfall (often 0), so “live” recommendations can underweight rainfall. Seasonal or historical weather is not integrated; the system is best seen as “current conditions + soil” rather than full seasonal forecasting.
- **Model capacity and data scale:** The default FT-Transformer is relatively small (e.g. 3 layers, 4 heads). For much larger merged datasets, tuning depth, width, or regularization might improve performance. The current design does not implement early stopping or learning-rate scheduling; training runs for a fixed number of epochs.
- **Deployment:** The app is served via Streamlit’s default server. For production, one would typically deploy behind a reverse proxy, use a production ASGI server, or wrap the inference in a REST API and keep Streamlit as an optional front end. The analysis here does not cover scaling or security hardening.

---

## 11. Summary

The project is a **complete crop recommendation pipeline**: from multi-source data (Kaggle ± Mendeley) and merge/preprocess, through an FT-Transformer trained with standard metrics and saved with scaler and label encoder, to inference that supports manual or API-sourced weather and a Streamlit UI with detailed per-crop metadata (NPK gap, water, calendar, season suitability). The architecture is modular (model, train, inference, metadata, app), uses a single preprocessing and checkpoint format, and is documented with architecture diagrams and this analysis. It is well-suited for learning, prototyping, and extension (e.g. more datasets, more features, or deployment as an API or production web app).
