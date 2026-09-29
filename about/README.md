# Crop Prediction with FT-Transformer

High-performance **FT-Transformer** (Feature Tokenizer Transformer) for recommending the best crop from 7 inputs: soil nutrients (N, P, K, pH) and weather (Temperature, Humidity, Rainfall). Uses the [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset).

## Setup (step by step)

### Step 1: Dataset — manual or auto-download from Kaggle

You can get the data in either way:

**Option A – Auto-download (recommended)**  
When you run training, if **`Crop_recommendation.csv`** is not in the project folder, the script will try to download the dataset from Kaggle. For that you need **Kaggle API** credentials in one of these ways:

1. **Using a `.env` file (easiest):** add **KAGGLE_USERNAME** and **KAGGLE_KEY** to a `.env` file in the project root (see Step 3).
2. **Using `kaggle.json`:** Log in to [Kaggle](https://www.kaggle.com) → **Settings** → **API** → **Create New Token** (downloads `kaggle.json`). Then:
   - **Mac/Linux:** `mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`
   - **Windows:** create folder `%USERPROFILE%\.kaggle`, move `kaggle.json` there.

When you run `python train.py`, if the CSV is missing, it will download the [Crop Recommendation Dataset](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset) into a **`data/`** folder and use it.

**Option B – Manual download**  
- Download the dataset from [Kaggle](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset), then put the CSV in the project root and name it **`Crop_recommendation.csv`** (or pass its path with `--data path/to/file.csv`).

The CSV must have these columns (the code accepts **ph** or **pH**, and **label** or **crop**):
- **N**, **P**, **K** – soil nutrients  
- **temperature** (°C), **humidity** (%), **rainfall** (mm)  
- **ph** – soil pH  
- **label** or **crop** – crop name (target)

### Step 2: Install Python dependencies

- Open a terminal and go to the project folder, for example:
  ```bash
  cd "/Users/niswa/minor project"
  ```
- (Recommended) Create and activate a virtual environment:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate   # On Windows: .venv\Scripts\activate
  ```
- Install all required packages:
  ```bash
  pip install -r requirements.txt
  ```
  This installs: PyTorch, NumPy, Pandas, scikit-learn, `requests` (weather API), **`kaggle`** (dataset auto-download), and **`python-dotenv`** (to load `.env`).

### Step 3: API keys — use a `.env` file (recommended)

All API keys can go in a **`.env`** file in the **project root** (same folder as `model.py`). The app loads it automatically.

1. Copy the example file and edit it:
   ```bash
   cp .env.example .env
   ```
2. Open **`.env`** and add your keys:

   | Variable | Where to get it | Used by |
   |----------|-----------------|--------|
   | **OPENWEATHERMAP_API_KEY** | [OpenWeatherMap](https://openweathermap.org/api) → Sign up → API keys | `inference.py` (live weather by city/coordinates) |
   | **KAGGLE_USERNAME** | Your Kaggle profile username | `train.py` (auto-download dataset if CSV missing) |
   | **KAGGLE_KEY** | Kaggle → Settings → API → Create New Token → open `kaggle.json`, copy `key` | `train.py` (auto-download) |

   Example `.env`:
   ```bash
   OPENWEATHERMAP_API_KEY=abc123yourkey
   KAGGLE_USERNAME=yourusername
   KAGGLE_KEY=your_kaggle_api_key_from_kaggle_json
   ```

3. **Do not commit `.env`** — it’s in `.gitignore`. Only commit `.env.example` (no real keys).

**Alternative to `.env` for Kaggle:** you can skip `KAGGLE_USERNAME` and `KAGGLE_KEY` and instead put **`kaggle.json`** in `~/.kaggle/` (see Step 1). The code checks environment variables first, then Kaggle’s default config.

---

## Project layout (what each file does)

| File | Purpose |
|------|--------|
| **model.py** | Defines the FT-Transformer: feature tokenizer (7 inputs → tokens), learnable [CLS] token, transformer layers (multi-head self-attention + GEGLU feed-forward), and a linear head that outputs crop class logits. |
| **train.py** | Loads the CSV, preprocesses with StandardScaler and LabelEncoder, builds an 80/20 train–test split, trains the model with CrossEntropyLoss and AdamW, and saves the best model plus scaler and label encoder under `checkpoints/`. |
| **inference.py** | Loads the saved model and scaler; can fetch live weather via OpenWeatherMap; provides `predict_top_k()` for top-3 crop recommendations and an interactive CLI that asks for soil + weather (or city). |
| **app.py** | Streamlit web UI: soil + weather input, top crop recommendations with crop type filter, water advisor, sowing/harvest calendar, NPK gap, duration, season suitability, weather summary, export CSV, and last-result history. |
| **crop_metadata.py** | Metadata for each crop: type, water requirement, duration, sowing/harvest months, ideal NPK ranges, water source, suitable seasons. Used by the app for filters and detail blocks. |

---

## Run the web UI

After training, you can use the **web interface**:

```bash
cd "/Users/niswa/minor project"
source .venv/bin/activate   # if you use a venv
pip install streamlit       # if not already installed
streamlit run app.py
```

**Web UI features:**
- **Auto season detection** — Shows current season (Winter/Summer/Monsoon/Post-Monsoon).
- **Crop type filter** — Optionally restrict recommendations to Cereal, Pulses, Fruit, Vegetable, etc.
- **Soil & weather** — Enter N, P, K, pH; weather by city (live API) or manual values. Input hints show typical ranges.
- **Top recommendations** — Each crop shows: **Water requirement** (mm and category), **Water source advisor** (Rainfed/Irrigated/Both), **Sowing & harvest calendar** (months), **Crop duration** (days), **NPK fertilizer gap** (add/reduce N, P, K), **Season suitability** (suitable or not for current season).
- **Weather summary** — When using city, displays the fetched temperature, humidity, and rainfall used for the recommendation.
- **Export** — Download results as CSV.
- **Last recommendation** — Expandable summary of the previous run.

---

## Merging a second dataset (e.g. Mendeley)

You can merge the **Mendeley Crop recommendation** dataset (Thangatamilan, Sagana 2025, DOI: 10.17632/vynxnppr7j.1) with your current Kaggle data to get more samples and potentially better results.

**Dataset comparison**

| Aspect | Your current (Kaggle) | Mendeley (vynxnppr7j.1) |
|--------|------------------------|--------------------------|
| **Columns** | N, P, K, temperature, humidity, ph, rainfall, label | Soil (pH), climate (temperature, relative humidity, season), N, P, K; may include crop name/label, sown, harvested, water source. **Rainfall** may be missing. |
| **Compatibility** | 7 features + label | Same nutrients (N, P, K) and climate (temp, humidity, pH). Column names may differ (e.g. "Relative Humidity", "Soil pH"). |
| **Difference** | Has **rainfall** | May not have rainfall; the code **imputes** it from the primary dataset (median per crop, or global median) so the merged set still has 7 features. |

**Is merging possible?** Yes, if the Mendeley CSV has (or can be mapped to) **N, P, K, temperature, humidity, ph**, and a **crop/label** column. Rainfall is optional: if missing, it is filled from your Kaggle data so the model keeps 7 inputs.

**How to merge and train**

1. Download the Mendeley dataset from [Mendeley Data](https://data.mendeley.com/datasets/vynxnppr7j/1) (Download All → extract the CSV).
2. Put the CSV in your project (e.g. in the `data/` folder). The filename can be anything (e.g. `Mendeley_crop.csv` or `Crop recommendation dataset.csv`).
3. Run training with merge. Use the **exact path** to your primary and secondary CSVs (use quotes if the path has spaces):
   ```bash
   python train.py --data "data/Crop_recommendation.csv" --merge_secondary "data/Crop recommendation dataset.csv" --merged_output data/Merged_crop_recommendation.csv
   ```
   If your secondary file is named `data/Mendeley_crop.csv`, use that path instead of `"data/Crop recommendation dataset.csv"`.
   The script will: load the primary dataset (or download from Kaggle if missing), map the Mendeley columns to N, P, K, temperature, humidity, ph, rainfall, label, impute rainfall for Mendeley rows if needed, save the merged CSV, then train on it.
4. If the Mendeley file has different column names, the code maps common variants (e.g. "Relative Humidity" → humidity, "Soil pH" → ph, "Crop" / "Type of crop" → label). If your file has a column not in the map, you may need to rename it in the CSV to one of the supported names, or add it to `MENDELEY_STYLE_MAP` in `train.py`.

**Without errors:** Use the same units as your primary data (e.g. °C, mm, %). Crop names are normalized to lowercase so "Rice" and "rice" are treated as the same class.

---

## Train (step by step)

1. Either have **`Crop_recommendation.csv`** in the project root, or set up the Kaggle API (Setup, Step 1) so the script can download it automatically when missing.
2. From the project folder (with your venv activated if you use one), run:
   ```bash
   python train.py --data Crop_recommendation.csv --epochs 50 --batch_size 64
   ```
3. What this does:
   - **`--data`** – Path to the CSV (default is `Crop_recommendation.csv` in the current folder).
   - **`--epochs`** – How many full passes over the training data (default 50).
   - **`--batch_size`** – Number of samples per gradient update (default 64).
4. While training, you’ll see loss and metrics (Accuracy, Precision, Recall, F1) for both train and test. The **best** model (by test F1) is saved automatically.
5. After training, check the **`checkpoints/`** folder:
   - **`best_model.pt`** – Model weights, scaler, label encoder, and metadata (needed for inference).
   - **`history.json`** – Train/test metrics per epoch for plotting or analysis.

---

## Inference (step by step)

You can get crop recommendations in two ways: **from code** (e.g. in a script) or **interactively** in the terminal.

### Option A: Use the model from your own code (top-k crops)

1. Train first so that **`checkpoints/best_model.pt`** exists.
2. In your Python script or notebook:
   - Load the checkpoint (model + scaler + label encoder).
   - Build a feature vector (soil N, P, K, pH and weather temp, humidity, rainfall).
   - Call **`predict_top_k`** to get the top 3 (or any k) recommended crops with scores.

   Example:
   ```python
   from inference import load_checkpoint, predict_top_k, build_features_from_user

   # Load the trained model and preprocessing artifacts
   model, scaler, label_encoder, device = load_checkpoint("checkpoints/best_model.pt")

   # Build one row of features: N, P, K, temperature, humidity, rainfall, ph
   features = build_features_from_user(
       N=90, P=42, K=43, ph=6.5,
       temperature=20, humidity=82, rainfall=202
   )

   # Get top 3 recommended crops (list of (crop_name, probability))
   top3 = predict_top_k(model, scaler, label_encoder, features, device, k=3)
   for name, prob in top3:
       print(f"  {name}: {prob:.2%}")
   ```
3. **Important:** Any new data (e.g. from a weather API) must be combined with soil into the same 7 features and passed through **`build_features_from_user`** (or scaled with the same scaler); the code uses the scaler saved in the checkpoint so live weather is scaled the same way as in training.

### Option B: Interactive mode (terminal prompts)

1. Train first so that **`checkpoints/best_model.pt`** exists.
2. Run the inference script:
   ```bash
   python inference.py
   ```
3. The script will ask you, in order:
   - **Nitrogen (N), Phosphorus (P), Potassium (K), pH** – enter numbers.
   - **City for live weather** – either type a city name (e.g. `London`) to fetch live weather from OpenWeatherMap, or leave it blank.
   - If you left city blank: it will ask for **Temperature (°C), Humidity (%), Rainfall (mm)** – enter them manually.
   - If you entered a city: it will use the API (requires `OPENWEATHERMAP_API_KEY`) to get temperature, humidity, and rainfall for that city.
4. The script then prints the **top 3 recommended crops** with probabilities.
5. To run without using the weather API at all (always type weather by hand), use:
   ```bash
   python inference.py --no-api
   ```
   Then when prompted, leave city blank and enter temperature, humidity, and rainfall when asked.

**Note:** Live weather from the API is scaled with the **same StandardScaler** that was fitted during training (stored in the checkpoint), so the model sees weather in the same scale as the training data.
