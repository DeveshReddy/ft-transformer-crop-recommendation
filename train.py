"""
Training script for FT-Transformer crop prediction.
Data pipeline: load CSV, StandardScaler (fit on train only) + LabelEncoder,
train/val/test split (default 70/15/15). Model selection by validation F1;
early stopping and ReduceLROnPlateau on validation F1. Optional class weights.
If CSV is missing, downloads dataset from Kaggle via API.
Metrics: Accuracy, Precision, Recall, F1-Score.
"""

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import pandas as pd

# Kaggle dataset (auto-download if CSV not found)
KAGGLE_DATASET = "atharvaingle/crop-recommendation-dataset"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DATA_DOWNLOAD_DIR = DEFAULT_DATA_DIR / "data"
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from model import FTTransformer

# Expected column names (Kaggle Crop Recommendation Dataset)
FEATURE_COLUMNS = [
    "N",           # Nitrogen
    "P",           # Phosphorus
    "K",           # Potassium
    "temperature",
    "humidity",
    "rainfall",
    "ph",          # pH (dataset often uses lowercase)
]
TARGET_COLUMN_ALIASES = ["label", "crop", "Label", "Crop"]


def download_dataset_from_kaggle(download_dir: Path) -> str:
    """
    Download Crop Recommendation Dataset from Kaggle and return path to the CSV.
    Requires: pip install kaggle, and ~/.kaggle/kaggle.json with your API key.
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        raise ImportError(
            "Kaggle API is required for auto-download. Run: pip install kaggle"
        )
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(
        KAGGLE_DATASET,
        path=str(download_dir),
        unzip=True,
    )
    # Find the CSV (dataset may name it Crop_recommendation.csv or similar)
    csv_files = list(download_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV found in {download_dir} after downloading {KAGGLE_DATASET}"
        )
    # Prefer one that looks like the crop dataset (has N, P, K columns)
    for f in csv_files:
        try:
            df = pd.read_csv(f, nrows=1)
            if "N" in df.columns and "P" in df.columns and "K" in df.columns:
                return str(f)
        except Exception:
            continue
    return str(csv_files[0])


def ensure_data_path(data_path: str) -> str:
    """
    If data_path exists, return it. Otherwise try to download from Kaggle and return CSV path.
    """
    path = Path(data_path)
    if path.is_file():
        return data_path
    # Resolve default path relative to project root
    if not path.is_absolute():
        path = DEFAULT_DATA_DIR / path
    if path.is_file():
        return str(path)
    # Try default filename in project root
    default_file = DEFAULT_DATA_DIR / "Crop_recommendation.csv"
    if default_file.is_file():
        return str(default_file)
    print(f"Data file not found at {data_path}. Attempting to download from Kaggle...")
    csv_path = download_dataset_from_kaggle(DATA_DOWNLOAD_DIR)
    print(f"Downloaded dataset to {csv_path}")
    return csv_path


def find_target_column(df: pd.DataFrame) -> str:
    for name in TARGET_COLUMN_ALIASES:
        if name in df.columns:
            return name
    raise KeyError(f"Target column not found. Has columns: {list(df.columns)}")


def normalize_feature_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure feature column names match (e.g. 'ph' vs 'pH')."""
    mapping = {}
    for c in df.columns:
        if c.lower() == "ph" and c != "ph":
            mapping[c] = "ph"
    return df.rename(columns=mapping)


# Optional second dataset (e.g. Mendeley): map common column names to our 7 + label
MENDELEY_STYLE_MAP = {
    "N": ["N", "nitrogen", "Nitrogen", "N (nitrogen)"],
    "P": ["P", "phosphorus", "Phosphorus", "P (phosphorus)"],
    "K": ["K", "potassium", "Potassium", "K (potassium)"],
    "temperature": ["temperature", "Temperature", "temp", "Temp", "TEMP"],
    "humidity": ["humidity", "Humidity", "relative humidity", "Relative Humidity", "relative_humidity", "RELATIVE_HUMIDITY"],
    "ph": ["ph", "pH", "soil ph", "Soil pH", "soil_ph", "SOIL_PH"],
    "rainfall": ["rainfall", "Rainfall", "rain", "Rain", "precipitation", "Precipitation"],
    "label": ["label", "crop", "Crop", "Label", "crop name", "Crop Name", "type of crop", "crop_type", "CROPS", "TYPE_OF_CROP"],
}


def _map_columns(df: pd.DataFrame, column_map: dict) -> pd.DataFrame:
    """Rename columns to standard names using a list of possible names per standard name."""
    rename = {}
    cand_lower = {k: [x.strip().lower() for x in v] for k, v in column_map.items()}
    for std_name, candidates in column_map.items():
        for c in df.columns:
            cn = c.strip().lower()
            if c in candidates or cn in cand_lower.get(std_name, []) or cn == std_name.lower():
                rename[c] = std_name
                break
    return df.rename(columns=rename)


def merge_crop_datasets(
    primary_path: str,
    secondary_path: str,
    output_path: str,
    impute_missing_rainfall: bool = True,
) -> str:
    """
    Merge two crop recommendation CSVs into one with columns N, P, K, temperature, humidity, ph, rainfall, label.
    - primary_path: e.g. Crop_recommendation.csv (Kaggle) — must have all 7 features + label.
    - secondary_path: e.g. Mendeley dataset — will be mapped to same columns; if 'rainfall' is missing, imputed from primary when impute_missing_rainfall=True.
    - output_path: where to save the merged CSV.
    Returns path to merged file.
    """
    df1 = pd.read_csv(primary_path)
    df1 = normalize_feature_columns(df1)
    target1 = find_target_column(df1)
    if "pH" in df1.columns and "ph" not in df1.columns:
        df1["ph"] = df1["pH"]
    required = FEATURE_COLUMNS + [target1]
    if not all(c in df1.columns for c in FEATURE_COLUMNS):
        raise ValueError(f"Primary dataset must have columns {FEATURE_COLUMNS}. Found: {list(df1.columns)}")
    df1 = df1[FEATURE_COLUMNS + [target1]].copy()
    df1 = df1.rename(columns={target1: "label"})

    df2 = pd.read_csv(secondary_path)
    df2 = normalize_feature_columns(df2)
    df2 = _map_columns(df2, MENDELEY_STYLE_MAP)

    # Require at least N, P, K, temperature, humidity, ph + label for secondary
    for col in ["N", "P", "K", "temperature", "humidity", "ph", "label"]:
        if col not in df2.columns:
            raise ValueError(
                f"Secondary dataset must have mappable columns for N, P, K, temperature, humidity, ph, label. "
                f"After mapping, got: {list(df2.columns)}. Original: {list(pd.read_csv(secondary_path).columns)}"
            )

    if "rainfall" not in df2.columns:
        if not impute_missing_rainfall:
            raise ValueError(
                "Secondary dataset has no 'rainfall' column. Use impute_missing_rainfall=True to fill from primary median (per crop if possible)."
            )
        # Impute: median rainfall from primary, per crop if label matches, else global median
        df1_lab = df1["label"].astype(str).str.strip().str.lower()
        global_median = float(df1["rainfall"].median())
        crop_median = df1.groupby(df1_lab)["rainfall"].median().to_dict()
        df2_labels = df2["label"].astype(str).str.strip().str.lower()
        df2["rainfall"] = df2_labels.map(lambda c: crop_median.get(c, global_median)).fillna(global_median)

    df2 = df2[[c for c in FEATURE_COLUMNS + ["label"] if c in df2.columns]]
    if list(df2.columns) != FEATURE_COLUMNS + ["label"]:
        raise ValueError(f"Secondary columns after merge: {list(df2.columns)}")

    # Normalize label to string, lowercase for consistency
    df1["label"] = df1["label"].astype(str).str.strip().str.lower()
    df2["label"] = df2["label"].astype(str).str.strip().str.lower()

    # Drop any row with NaN in features or label
    df1 = df1.dropna(subset=FEATURE_COLUMNS + ["label"])
    df2 = df2.dropna(subset=FEATURE_COLUMNS + ["label"])

    merged = pd.concat([df1, df2], ignore_index=True)
    merged = merged.dropna(subset=FEATURE_COLUMNS + ["label"])
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    return str(out)


def load_and_preprocess(data_path: str, val_ratio: float = 0.15, test_size: float = 0.2, random_state: int = 42):
    """
    Load Crop_recommendation.csv, scale numerical features, encode labels.
    Split: train (1 - test_size - val_ratio), validation (val_ratio), test (test_size).
    Scaler is fitted on train only, then applied to train/val/test.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test, scaler, label_encoder, num_classes
    """
    df = pd.read_csv(data_path)
    df = normalize_feature_columns(df)
    target_col = find_target_column(df)

    # Handle possible 'pH' column name (Kaggle often uses 'ph')
    if "pH" in df.columns and "ph" not in df.columns:
        df["ph"] = df["pH"]
    cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    if len(cols) != 7:
        raise ValueError(
            f"Expected 7 feature columns from {FEATURE_COLUMNS}, got {cols}. "
            f"DataFrame columns: {list(df.columns)}"
        )
    X = df[FEATURE_COLUMNS].astype(np.float32).values
    y_raw = df[target_col].astype(str).values

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)
    num_classes = len(label_encoder.classes_)

    # First split: (train+val) vs test (stratified)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    # Second split: train vs val from (train+val). val_ratio is fraction of full data.
    # So from train_val we take test_size = val_ratio / (1 - test_size)
    val_ratio_of_train_val = val_ratio / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_ratio_of_train_val,
        random_state=random_state,
        stratify=y_train_val,
    )

    # Fit scaler on train only to avoid leakage
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    return (
        X_train,
        X_val,
        X_test,
        y_train.astype(np.int64),
        y_val.astype(np.int64),
        y_test.astype(np.int64),
        scaler,
        label_encoder,
        num_classes,
    )


def get_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64, pin_memory=False):
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )
    test_ds = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test),
    )
    # pin_memory only helps on CUDA; MPS (Apple Silicon) doesn't support it and warns
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=0
    )
    return train_loader, val_loader, test_loader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, average: str = "weighted"):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        logits = model(X_b)
        loss = criterion(logits, y_b)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_b.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_b.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logits = model(X_b)
        loss = criterion(logits, y_b)
        total_loss += loss.item() * X_b.size(0)
        all_preds.append(logits.argmax(dim=1).cpu().numpy())
        all_labels.append(y_b.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    metrics = compute_metrics(y_true, y_pred)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train FT-Transformer for crop prediction")
    parser.add_argument(
        "--data",
        type=str,
        default="Crop_recommendation.csv",
        help="Path to primary dataset (e.g. Kaggle Crop_recommendation.csv)",
    )
    parser.add_argument(
        "--merge_secondary",
        type=str,
        default=None,
        help="Path to second dataset (e.g. Mendeley) to merge with --data. Merged file saved to --merged_output.",
    )
    parser.add_argument(
        "--merged_output",
        type=str,
        default="data/Merged_crop_recommendation.csv",
        help="Path to save merged CSV when using --merge_secondary",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_token", type=int, default=64)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--out_dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    # Train/val/test and early stopping
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Fraction of data for validation (train=rest, test=0.2)")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping: stop if val F1 does not improve for this many epochs")
    parser.add_argument("--scheduler_patience", type=int, default=5, help="ReduceLROnPlateau: reduce LR after this many epochs without val F1 gain")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate for scheduler")
    parser.add_argument("--no_class_weights", action="store_true", help="Disable balanced class weights in CrossEntropyLoss")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading and preprocessing data...")
    data_path = ensure_data_path(args.data)
    if args.merge_secondary:
        # Resolve relative paths from project root so it works regardless of terminal cwd
        secondary_path = Path(args.merge_secondary)
        if not secondary_path.is_absolute():
            secondary_path = DEFAULT_DATA_DIR / secondary_path
        if not secondary_path.is_file():
            raise FileNotFoundError(
                f"Secondary dataset not found: {secondary_path}. "
                f"Run from project folder and use path like 'data/Mendley_crop.csv' (no @)."
            )
        merged_out = Path(args.merged_output)
        if not merged_out.is_absolute():
            merged_out = DEFAULT_DATA_DIR / merged_out
        print(f"Merging primary ({data_path}) with secondary ({secondary_path})...")
        data_path = merge_crop_datasets(
            data_path,
            str(secondary_path),
            str(merged_out),
            impute_missing_rainfall=True,
        )
        print(f"Merged dataset saved to {data_path}. Training on merged data.")
    (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        scaler, label_encoder, num_classes,
    ) = load_and_preprocess(data_path, val_ratio=args.val_ratio, random_state=args.seed)

    train_loader, val_loader, test_loader = get_dataloaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
    )

    # Optional class weights for imbalanced classes (default: use weights)
    criterion = nn.CrossEntropyLoss()
    if not args.no_class_weights:
        classes = np.unique(y_train)
        cw = compute_class_weight("balanced", classes=classes, y=y_train)
        weight_tensor = torch.tensor(cw, dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        print("Using balanced class weights in loss.")

    model = FTTransformer(
        num_features=7,
        d_token=args.d_token,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        num_classes=num_classes,
        dropout=args.dropout,
        use_geglu=True,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=args.scheduler_patience, min_lr=args.min_lr
    )

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    history = {"train": [], "val": [], "test": []}

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        test_metrics = evaluate(model, test_loader, criterion, device)
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)
        history["test"].append(test_metrics)

        scheduler.step(val_metrics["f1"])

        if val_metrics["f1"] > best_val_f1:
            best_val_f1 = val_metrics["f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "scaler": scaler,
                    "label_encoder": label_encoder,
                    "num_classes": num_classes,
                    "feature_columns": FEATURE_COLUMNS,
                },
                os.path.join(args.out_dir, "best_model.pt"),
            )
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d} | Train Loss: {train_metrics['loss']:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} F1: {train_metrics['f1']:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} F1: {val_metrics['f1']:.4f} | "
            f"Test Acc: {test_metrics['accuracy']:.4f} F1: {test_metrics['f1']:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping: no val F1 improvement for {args.patience} epochs.")
            break

    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nBest validation F1: {best_val_f1:.4f}. Model and artifacts saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
