"""
Inference with FT-Transformer: load model, fetch live weather via OpenWeatherMap,
and recommend top-k crops from user input (soil + weather).
"""

import os
from typing import List, Tuple

from dotenv import load_dotenv
from pathlib import Path

# Load .env from project folder so it works regardless of cwd
load_dotenv(Path(__file__).resolve().parent / ".env")

import numpy as np
import requests
import torch

from model import FTTransformer
from train import FEATURE_COLUMNS

# OpenWeatherMap: pass same feature order as training
FEATURE_ORDER = ["N", "P", "K", "temperature", "humidity", "rainfall", "ph"]


def fetch_weather_openweathermap(
    api_key: str,
    city: str = None,
    lat: float = None,
    lon: float = None,
) -> Tuple[float, float, float]:
    """
    Fetch current temperature (°C), humidity (%), and rainfall (mm) from OpenWeatherMap.
    Provide either (city) or (lat, lon). Rainfall is from current weather (rain.1h or 0).
    Returns: (temperature, humidity, rainfall).
    """
    if not api_key:
        raise ValueError("OPENWEATHERMAP_API_KEY must be set for live weather.")
    if city:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"q": city, "appid": api_key, "units": "metric"}
    elif lat is not None and lon is not None:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    else:
        raise ValueError("Provide either city or (lat, lon).")

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    temp = float(data["main"]["temp"])
    humidity = float(data["main"]["humidity"])
    # OpenWeatherMap current weather may include "rain":{"1h": x}; else 0
    rain = 0.0
    if "rain" in data and isinstance(data["rain"], dict):
        rain = float(data["rain"].get("1h", data["rain"].get("3h", 0)))
    return temp, humidity, rain


def build_features_from_user(
    N: float,
    P: float,
    K: float,
    ph: float,
    temperature: float = None,
    humidity: float = None,
    rainfall: float = None,
    api_key: str = None,
    city: str = None,
    lat: float = None,
    lon: float = None,
) -> np.ndarray:
    """
    Build 7-dim feature vector. If temperature/humidity/rainfall are None,
    fetch them from OpenWeatherMap using api_key and city or (lat, lon).
    """
    if temperature is None or humidity is None or rainfall is None:
        if not api_key:
            api_key = os.environ.get("OPENWEATHERMAP_API_KEY")
        t, h, r = fetch_weather_openweathermap(api_key, city=city, lat=lat, lon=lon)
        temperature = temperature if temperature is not None else t
        humidity = humidity if humidity is not None else h
        rainfall = rainfall if rainfall is not None else r
    return np.array(
        [[N, P, K, temperature, humidity, rainfall, ph]],
        dtype=np.float32,
    )


def build_features_and_weather(
    N: float,
    P: float,
    K: float,
    ph: float,
    temperature: float = None,
    humidity: float = None,
    rainfall: float = None,
    api_key: str = None,
    city: str = None,
    lat: float = None,
    lon: float = None,
):
    """
    Build feature vector and return (features, weather_used).
    weather_used: dict with keys temperature, humidity, rainfall, city (str or None for manual).
    """
    weather_used = None
    if temperature is not None and humidity is not None and rainfall is not None:
        weather_used = {"temperature": temperature, "humidity": humidity, "rainfall": rainfall, "city": None}
    elif city or (lat is not None and lon is not None):
        if not api_key or str(api_key).strip() == "your_openweathermap_api_key_here":
            raise ValueError("Invalid or missing OpenWeatherMap API key.")
        t, h, r = fetch_weather_openweathermap(api_key, city=city, lat=lat, lon=lon)
        temperature, humidity, rainfall = t, h, r
        weather_used = {"temperature": t, "humidity": h, "rainfall": r, "city": city or "Current location"}
    else:
        raise ValueError("Provide temperature, humidity, rainfall or use city/API.")
    features = np.array(
        [[N, P, K, temperature, humidity, rainfall, ph]],
        dtype=np.float32,
    )
    return features, weather_used


def load_checkpoint(checkpoint_path: str, device: torch.device = None):
    """Load model, scaler, and label encoder from training checkpoint."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    scaler = ckpt["scaler"]
    label_encoder = ckpt["label_encoder"]
    num_classes = ckpt["num_classes"]
    model = FTTransformer(
        num_features=7,
        d_token=64,
        n_heads=4,
        n_layers=3,
        d_ff=128,
        num_classes=num_classes,
        dropout=0.0,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model, scaler, label_encoder, device


def predict_top_k(
    model: torch.nn.Module,
    scaler,
    label_encoder,
    features_raw: np.ndarray,
    device: torch.device,
    k: int = 3,
) -> List[Tuple[str, float]]:
    """
    Scale features with training scaler, run model, return top-k (crop_name, score) sorted by score.
    features_raw: (1, 7) or (7,) in order [N, P, K, temperature, humidity, rainfall, ph].
    """
    if features_raw.ndim == 1:
        features_raw = features_raw.reshape(1, -1)
    assert features_raw.shape[1] == 7
    X = scaler.transform(features_raw).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X).to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_indices = np.argsort(probs)[::-1][:k]
    return [
        (label_encoder.classes_[i], float(probs[i]))
        for i in top_indices
    ]


def run_interactive(
    checkpoint_path: str = "checkpoints/best_model.pt",
    use_weather_api: bool = True,
):
    """
    Interactive: prompt for soil (N, P, K, pH), optionally city/coords,
    then print top-3 recommended crops.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, scaler, label_encoder, _ = load_checkpoint(checkpoint_path, device)
    api_key = os.environ.get("OPENWEATHERMAP_API_KEY") if use_weather_api else None

    print("Crop recommendation (FT-Transformer). Enter soil and optional weather.")
    try:
        N = float(input("Nitrogen (N): "))
        P = float(input("Phosphorus (P): "))
        K = float(input("Potassium (K): "))
        ph = float(input("pH: "))
    except (ValueError, EOFError):
        print("Invalid input.")
        return

    city = input("City for live weather (or leave blank to type Temp, Humidity, Rainfall): ").strip()
    temp = humidity = rainfall = None
    if not city:
        try:
            temp = float(input("Temperature (°C): "))
            humidity = float(input("Humidity (%): "))
            rainfall = float(input("Rainfall (mm): "))
        except (ValueError, EOFError):
            if api_key:
                city = input("Fallback: city name for OpenWeatherMap: ").strip()

    # If city provided, fetch weather (temp/humidity/rainfall left None); else use typed values
    features = build_features_from_user(
        N=N, P=P, K=K, ph=ph,
        temperature=temp, humidity=humidity, rainfall=rainfall,
        api_key=api_key,
        city=city if city else None,
    )
    top3 = predict_top_k(model, scaler, label_encoder, features, device, k=3)
    print("\nTop 3 recommended crops:")
    for name, score in top3:
        print(f"  {name}: {score:.2%}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best_model.pt")
    p.add_argument("--no-api", action="store_true", help="Do not use OpenWeatherMap")
    a = p.parse_args()
    run_interactive(checkpoint_path=a.checkpoint, use_weather_api=not a.no_api)
