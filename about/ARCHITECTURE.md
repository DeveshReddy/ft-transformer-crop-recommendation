# Crop Recommendation Project - Updated Architecture

This is the refreshed architecture for your current codebase. It covers data ingestion, optional dataset merge, FT-Transformer training with validation-driven model selection, checkpointing, inference, and Streamlit app presentation.

---

## 1) End-to-end architecture

```mermaid
flowchart TB
    subgraph DATA["Data Sources"]
        KAG["Kaggle: Crop_recommendation.csv"]
        MEN["Optional: Mendeley CSV"]
    end

    subgraph TRAIN["Training System (train.py)"]
        RES["ensure_data_path / optional Kaggle download"]
        MRG["merge_crop_datasets (optional)"]
        PRE["normalize columns + encode labels"]
        SPL["Stratified split: Train 65% / Val 15% / Test 20%"]
        SCL["StandardScaler fit on Train only"]
        LDR["DataLoaders: train, val, test"]
        LOOP["Train loop (AdamW + CE loss)"]
        SEL["Model selection by best Validation F1"]
        STOP["Early stopping + ReduceLROnPlateau"]
        CKPT["Save best_model.pt + history.json"]
    end

    subgraph INF["Inference Layer (inference.py)"]
        LOAD["load_checkpoint"]
        BUILD["build_features_* (manual or API weather)"]
        SCALE["scaler.transform"]
        PRED["predict_top_k (softmax + top-k)"]
    end

    subgraph APP["Application Layer (app.py + crop_metadata.py)"]
        UI["Streamlit UI inputs + filters"]
        META["Metadata enrichment (water, NPK gap, season, calendar)"]
        OUT["Top-3 recommendations + advisory cards + CSV export"]
    end

    KAG --> RES
    MEN --> MRG
    RES --> MRG
    MRG --> PRE
    PRE --> SPL
    SPL --> SCL
    SCL --> LDR
    LDR --> LOOP
    LOOP --> STOP
    STOP --> SEL
    SEL --> CKPT
    CKPT --> LOAD
    UI --> BUILD
    LOAD --> SCALE
    BUILD --> SCALE
    SCALE --> PRED
    PRED --> META
    META --> OUT
```

---

## 2) Data pipeline and merge logic

```mermaid
flowchart LR
    A["Primary CSV"] --> B["Normalize feature names (ph/pH)"]
    C["Secondary CSV (optional)"] --> D["Column mapping (MENDELEY_STYLE_MAP)"]
    D --> E{"Rainfall present?"}
    E -- "No" --> F["Impute rainfall from primary median (per crop, else global)"]
    E -- "Yes" --> G["Use provided rainfall"]
    F --> H["Keep columns: N,P,K,temp,humidity,rainfall,ph,label"]
    G --> H
    B --> I["Validate required 7 features + target"]
    I --> J["Lowercase/clean labels"]
    H --> J
    J --> K["Merged/clean dataframe"]
    K --> L["LabelEncoder fit"]
    K --> M["Stratified split"]
    M --> N["Train+Val / Test"]
    N --> O["Train / Val"]
    O --> P["Scaler fit on train only"]
    P --> Q["Transform train/val/test"]
```

**Current behavior in code:**
- Primary data can auto-download from Kaggle if not found.
- Secondary dataset is optional via `--merge_secondary`.
- Final feature order is fixed: `N, P, K, temperature, humidity, rainfall, ph`.
- Split is stratified and leakage-safe (scaler fitted only on training subset).

---

## 3) FT-Transformer internals (`model.py`)

```mermaid
flowchart TB
    X["Input X: (B, 7)"] --> TOK["FeatureTokenizer -> (B, 7, d_token)"]
    CLS["Learnable CLS token: (B, 1, d)"] --> CAT["Concat -> (B, 8, d)"]
    TOK --> CAT
    CAT --> BLK1["Transformer Block x n_layers"]
    BLK1 --> CLSP["Take CLS output (B, d)"]
    CLSP --> HEAD["Linear head -> logits (B, num_classes)"]
```

```mermaid
flowchart LR
    IN["Block input"] --> ATTN["Multi-head self-attention"]
    ATTN --> ADD1["Residual + LayerNorm"]
    ADD1 --> FFN["FFN (GEGLU/GELU + Dropout)"]
    FFN --> ADD2["Residual + LayerNorm"]
    ADD2 --> OUT["Block output"]
```

**Why this fits your task:**
- Each numerical feature is treated as a token, so attention can learn feature interactions (e.g., nutrient-weather coupling).
- CLS token acts as a global summary for classification.

---

## 4) Training control flow (`train.py`)

```mermaid
flowchart TB
    START["Parse CLI args"] --> DATA["Load / merge / preprocess"]
    DATA --> MODEL["Build FTTransformer"]
    MODEL --> LOSS["CrossEntropyLoss (optional class weights)"]
    LOSS --> OPT["AdamW optimizer"]
    OPT --> SCH["ReduceLROnPlateau on Validation F1"]
    SCH --> EPOCH["For each epoch"]
    EPOCH --> TR["train_epoch()"]
    TR --> VL["evaluate() on val"]
    VL --> TE["evaluate() on test (logged)"]
    TE --> BEST{"val F1 improved?"}
    BEST -- "Yes" --> SAVE["Save best_model.pt"]
    BEST -- "No" --> PAT["Increase no-improve counter"]
    SAVE --> ES{"counter >= patience?"}
    PAT --> ES
    ES -- "No" --> EPOCH
    ES -- "Yes" --> END["Stop training + write history.json"]
```

**Artifacts produced:**
- `checkpoints/best_model.pt`
  - `model_state_dict`
  - `scaler`
  - `label_encoder`
  - `num_classes`
  - `feature_columns`
- `checkpoints/history.json`
  - per-epoch metrics for `train`, `val`, `test`

---

## 5) Inference and weather integration (`inference.py`)

```mermaid
flowchart LR
    U1["User soil: N,P,K,pH"] --> FEAT["Feature builder"]
    U2["Weather input"] --> MODE{"City/API or Manual?"}
    MODE -- "City/API" --> OWM["OpenWeatherMap fetch"]
    MODE -- "Manual" --> MAN["Use manual temp/humidity/rainfall"]
    OWM --> FEAT
    MAN --> FEAT
    CK["Load checkpoint"] --> SC["Use saved scaler"]
    FEAT --> SC
    SC --> M["FT-Transformer forward pass"]
    M --> SM["Softmax probabilities"]
    SM --> TOP["Top-k crops + confidence"]
```

**Inference contract:**
- Input shape to model is always `(1, 7)` in fixed feature order.
- Same scaler from training checkpoint is always used.
- Output is ranked crop recommendations with probabilities.

---

## 6) Streamlit app flow (`app.py`)

```mermaid
flowchart TB
    A["App startup"] --> B["Load .env + checkpoint (cached)"]
    B --> C["Render UI: soil sliders, weather mode, month, crop-type filters"]
    C --> D["User clicks recommendation button"]
    D --> E["build_features_and_weather"]
    E --> F["predict_top_k"]
    F --> G["Apply crop type filter logic"]
    G --> H["Get metadata for top crops"]
    H --> I["Generate NPK gaps, season suitability, sow/harvest guidance"]
    I --> J["Render hero card + details + confidence bars"]
    J --> K["Export CSV + store last-result summary"]
```

**UI enrichment sources:**
- `crop_metadata.py` provides crop type, water requirement, duration, sow/harvest months, ideal NPK ranges, water source, and suitable seasons.

---

## 7) File-to-component map

| File | Responsibility |
|---|---|
| `train.py` | Data loading, optional merge, preprocessing, split, training loop, checkpoint/history save |
| `model.py` | FT-Transformer model (FeatureTokenizer, TransformerBlock, GEGLU, classifier head) |
| `inference.py` | Checkpoint loading, weather API fetch, feature building, top-k prediction |
| `app.py` | Streamlit UI, input collection, model invocation, result rendering/export |
| `crop_metadata.py` | Static agronomy metadata + helper utilities (season, NPK gaps, month labels) |
| `.env` / `.env.example` | API credentials and local environment configuration |
| `data/*.csv` | Primary/secondary/merged datasets |
| `checkpoints/*` | Model checkpoints and training history |

---

## 8) Deployment/runtime view

```text
Developer/CLI path:
  python train.py [--merge_secondary ...] -> checkpoints/best_model.pt
  python inference.py -> terminal recommendations

End-user path:
  streamlit run app.py -> browser UI -> checkpoint inference -> advisory output
```

---

## 9) Notes on correctness and maintainability

- Validation-driven checkpointing avoids selecting model by test performance.
- Scaler fit on train-only prevents leakage.
- Checkpoint schema is stable for app/inference compatibility.
- Data merge is resilient to alternate column names and missing rainfall.

You can render these Mermaid diagrams in Cursor, GitHub, or [mermaid.live](https://mermaid.live).

---

## 10) One-page compact architecture (presentation view)

```mermaid
flowchart LR
    subgraph DATA["Data Layer"]
        D1["Kaggle CSV"]
        D2["Mendeley CSV (optional)"]
        D3["Merge + map columns + impute rainfall"]
    end

    subgraph TRAIN["Model Training Layer"]
        T1["Preprocess: LabelEncoder + train-only StandardScaler"]
        T2["Split: Train 65 / Val 15 / Test 20"]
        T3["FT-Transformer training (AdamW)"]
        T4["Validation F1 selection + Early stopping + LR scheduler"]
        T5["Artifacts: best_model.pt + history.json"]
    end

    subgraph SERVE["Inference/Serving Layer"]
        S1["Load checkpoint: model + scaler + label encoder"]
        S2["Build 7-feature vector from soil + weather"]
        S3["Weather source: manual or OpenWeatherMap API"]
        S4["Predict top-k crops + confidence"]
    end

    subgraph APP["Application Layer (Streamlit)"]
        A1["User inputs + crop-type filter + planned month"]
        A2["Metadata enrichment (NPK gap, water, season, sow/harvest)"]
        A3["UI output: hero crop, top-3 bars, advisory cards, CSV export"]
    end

    D1 --> D3
    D2 --> D3
    D3 --> T1 --> T2 --> T3 --> T4 --> T5
    T5 --> S1
    A1 --> S2
    S3 --> S2
    S1 --> S4
    S2 --> S4
    S4 --> A2 --> A3
```

This compact diagram is the high-level "single slide" version. Use Sections 1-9 for full technical detail.
