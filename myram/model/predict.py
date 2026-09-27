"""Prediction and recommendation helpers for Flask API."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"

CROP_PROFILES = {
    "rice": {"temp": (22, 32), "rainfall": (120, 300), "humidity": (70, 95)},
    "wheat": {"temp": (12, 25), "rainfall": (40, 120), "humidity": (45, 70)},
    "maize": {"temp": (18, 30), "rainfall": (60, 180), "humidity": (50, 80)},
    "sugarcane": {"temp": (21, 35), "rainfall": (100, 250), "humidity": (55, 85)},
    "cotton": {"temp": (20, 34), "rainfall": (50, 150), "humidity": (40, 70)},
}


def _load_json(path: Path, default: dict) -> dict:
    """Load JSON safely with default fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def load_model():
    """Load persisted best model pipeline."""
    model_path = ARTIFACT_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model artifact not found. Run: python model/train_model.py")
    return joblib.load(model_path)


def predict_yield(model, payload: Dict[str, float | str]) -> float:
    """Predict crop yield in ton/hectare from user inputs."""
    input_df = pd.DataFrame(
        [
            {
                "temperature": float(payload["temperature"]),
                "rainfall": float(payload["rainfall"]),
                "humidity": float(payload["humidity"]),
                "soil_type": str(payload["soil_type"]).strip().lower(),
            }
        ]
    )
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 3)


def recommend_crop(payload: Dict[str, float | str]) -> Dict[str, float | str]:
    """Recommend best crop based on weather, soil, and historical performance."""
    stats = _load_json(ARTIFACT_DIR / "recommendation_stats.json", {"soil_crop_mean": {}, "overall_crop_mean": {}})
    soil = str(payload["soil_type"]).strip().lower()
    t = float(payload["temperature"])
    r = float(payload["rainfall"])
    h = float(payload["humidity"])

    scores: Dict[str, float] = {}
    soil_crop = stats.get("soil_crop_mean", {}).get(soil, {})
    overall_crop = stats.get("overall_crop_mean", {})

    for crop, profile in CROP_PROFILES.items():
        score = 0.0

        temp_min, temp_max = profile["temp"]
        rain_min, rain_max = profile["rainfall"]
        hum_min, hum_max = profile["humidity"]

        score += 1.0 if temp_min <= t <= temp_max else -0.4
        score += 1.0 if rain_min <= r <= rain_max else -0.4
        score += 1.0 if hum_min <= h <= hum_max else -0.4

        score += float(soil_crop.get(crop, 0.0)) * 0.4
        score += float(overall_crop.get(crop, 0.0)) * 0.2

        scores[crop] = score

    best_crop = max(scores, key=scores.get)
    return {"crop": best_crop, "score": round(scores[best_crop], 3)}


def dashboard_data() -> dict:
    """Provide dashboard data for UI visualizations."""
    metrics = _load_json(ARTIFACT_DIR / "metrics.json", {"best_model": "N/A", "metrics": {}})
    feature_importance = _load_json(ARTIFACT_DIR / "feature_importance.json", {"features": [], "importances": []})
    return {"metrics": metrics, "feature_importance": feature_importance}

