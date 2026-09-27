"""Prediction and recommendation helpers for backend API."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict

import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = BASE_DIR / "artifacts"
ACRES_PER_HECTARE = 2.47105


# Domain knowledge: soil ↔ crop suitability (used to keep predictions realistic).
SOIL_CROP_SUITABILITY: dict[str, dict[str, set[str]]] = {
    "black": {
        "good": {"cotton", "soybean", "maize", "rice"},
        "moderate": {"wheat", "groundnut", "gram"},
    },
    "red": {
        "good": {"groundnut", "gram", "maize", "soybean"},
        "moderate": {"cotton", "wheat"},
    },
    "alluvial": {
        "good": {"rice", "wheat", "sugarcane"},
        "moderate": {"maize", "soybean", "groundnut", "gram"},
    },
}

SOIL_FACTOR = {"black": 1.1, "red": 0.9, "alluvial": 1.2}
COMPATIBILITY_FACTOR = {"match": 1.0, "mismatch": 0.7}
SEASON_FACTOR_BY_SEASON = {"kharif": 1.0, "rabi": 0.95}
SEASON_COMPATIBILITY_FACTOR = {"match": 1.0, "mismatch": 0.7}

RAINFALL_OPTIMAL_MM: dict[str, tuple[float, float]] = {
    # Crop-specific optimal rainfall ranges (mm).
    # These reflect typical agronomic optima; yield drops outside this band.
    "rice": (1000.0, 1200.0),
    "soybean": (600.0, 800.0),
    "maize": (500.0, 800.0),
    "cotton": (600.0, 900.0),
    # Reasonable defaults for others in the UI list
    "wheat": (300.0, 600.0),
    "groundnut": (450.0, 700.0),
    "gram": (350.0, 600.0),
}
TEMP_OPTIMAL_C: dict[str, tuple[float, float]] = {
    "rice": (24.0, 32.0),
    "soybean": (22.0, 30.0),
    "maize": (20.0, 30.0),
    "cotton": (24.0, 34.0),
    "wheat": (15.0, 25.0),
    "groundnut": (24.0, 32.0),
    "gram": (18.0, 30.0),
}
RECOMMENDATION_CROPS = ["rice", "soybean", "maize", "cotton", "wheat", "gram"]
KHARIF_CROPS = {"rice", "soybean", "cotton", "maize"}
RABI_CROPS = {"wheat", "gram", "maize"}
HUMIDITY_OPTIMAL: dict[str, tuple[float, float]] = {
    "rice": (70.0, 90.0),
    "soybean": (55.0, 75.0),
    "maize": (50.0, 75.0),
    "cotton": (45.0, 70.0),
    "wheat": (45.0, 65.0),
    "groundnut": (50.0, 70.0),
    "gram": (40.0, 65.0),
}
YIELD_QTL_PER_ACRE_RANGE: dict[str, tuple[float, float]] = {
    "rice": (15.0, 22.0),
    "soybean": (8.0, 12.0),
    "maize": (12.0, 18.0),
    "cotton": (8.0, 15.0),
    "wheat": (12.0, 18.0),
    "gram": (6.0, 10.0),
}


def _canonical_crop(raw_crop: str) -> str:
    crop = str(raw_crop).strip().lower()
    if crop in {"pulses", "pulse"}:
        return "gram"
    if crop in {"paddy"}:
        return "rice"
    return crop


def _soil_crop_adjustment_factor(soil_type: str, crop: str) -> float:
    """Return suitability multiplier and label based on requested soil↔crop rules."""
    soil = str(soil_type).strip().lower()
    c = _canonical_crop(crop)
    required_soils = {
        "rice": {"black", "alluvial"},
        "soybean": {"black", "red"},
        "wheat": {"alluvial"},
        "gram": {"red"},  # pulses
    }
    if c not in required_soils:
        return COMPATIBILITY_FACTOR["match"], "Good"
    if soil in required_soils[c]:
        return COMPATIBILITY_FACTOR["match"], "Good"
    return COMPATIBILITY_FACTOR["mismatch"], "Poor"


def _season_adjustment_factor(season: str, crop: str) -> tuple[float, str]:
    """Return crop-season compatibility multiplier and label."""
    s = str(season).strip().lower()
    c = _canonical_crop(crop)
    required_season = {
        "rice": "kharif",
        "soybean": "kharif",
        "wheat": "rabi",
        "gram": "rabi",  # pulses
    }
    if c not in required_season:
        return SEASON_COMPATIBILITY_FACTOR["match"], "Good"
    if s == required_season[c]:
        return SEASON_COMPATIBILITY_FACTOR["match"], "Good"
    return SEASON_COMPATIBILITY_FACTOR["mismatch"], "Poor"

def _load_json(path: Path, default: dict) -> dict:
    """Load JSON safely with fallback."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return default


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _bounded_smooth(value: float, low: float, high: float) -> float:
    """
    Keep value inside [low, high] while avoiding hard flatlining at bounds.

    Hard clamping can collapse many different inputs to the same exact number
    (e.g. all values below min -> min). This bounded smoothing preserves
    realistic limits but still keeps sensitivity to input changes near edges.
    """
    band = max(high - low, 1e-6)
    margin = max(0.25, 0.08 * band)
    if value < low:
        delta = margin * (1.0 - math.exp(-(low - value) / band))
        return min(high, low + delta)
    if value > high:
        delta = margin * (1.0 - math.exp(-(value - high) / band))
        return max(low, high - delta)
    return value


def rainfall_impact(rainfall_mm: float, crop: str) -> str:
    """Classify rainfall as Low / Optimal / Excess for a given crop."""
    c = _canonical_crop(crop)
    low, high = RAINFALL_OPTIMAL_MM.get(c, (500.0, 800.0))
    if rainfall_mm < low:
        return "Low"
    if rainfall_mm > high:
        return "Excess"
    return "Optimal"


def _rainfall_response_factor(rainfall_mm: float, crop: str) -> float:
    """
    Non-linear rainfall response (parabolic penalty around optimal rainfall).

    We implement a bounded penalty so yield peaks near the midpoint of the crop's
    optimal rainfall range and decreases when rainfall is too low or too high.
    """
    c = _canonical_crop(crop)
    low, high = RAINFALL_OPTIMAL_MM.get(c, (500.0, 800.0))
    mid = (low + high) / 2.0
    half_range = max((high - low) / 2.0, 1.0)
    delta = abs(float(rainfall_mm) - mid)

    # Within the optimal band: up to ~10% drop at the band edges.
    # Outside the band: stronger penalty (up to ~40–50% depending on distance).
    if low <= rainfall_mm <= high:
        k = 0.10
    else:
        k = 0.35

    scaled = delta / half_range
    factor = 1.0 - (k * (scaled**2))
    return _clamp(float(factor), 0.50, 1.05)


def _temperature_response_factor(temp_c: float, crop: str) -> float:
    """Parabolic temperature response around crop optimal range."""
    c = _canonical_crop(crop)
    low, high = TEMP_OPTIMAL_C.get(c, (20.0, 32.0))
    mid = (low + high) / 2.0
    half_range = max((high - low) / 2.0, 1.0)
    delta = abs(float(temp_c) - mid)
    k = 0.20 if low <= temp_c <= high else 0.35
    scaled = delta / half_range
    factor = 1.0 - (k * (scaled**2))
    return _clamp(float(factor), 0.55, 1.05)


def _humidity_response_factor(humidity: float, crop: str) -> float:
    """Parabolic humidity response around crop optimal range."""
    c = _canonical_crop(crop)
    low, high = HUMIDITY_OPTIMAL.get(c, (45.0, 75.0))
    mid = (low + high) / 2.0
    half_range = max((high - low) / 2.0, 1.0)
    delta = abs(float(humidity) - mid)
    k = 0.15 if low <= humidity <= high else 0.30
    scaled = delta / half_range
    factor = 1.0 - (k * (scaled**2))
    return _clamp(float(factor), 0.60, 1.05)


def _predict_components(model, payload: Dict[str, float | str]) -> dict:
    """Return base model output and domain factors for transparent final yield."""
    crop_raw = str(payload.get("crop", ""))
    crop = _canonical_crop(crop_raw)
    soil = str(payload.get("soil_type", "")).strip().lower()
    rainfall = float(payload.get("rainfall", 0.0))
    temperature = float(payload.get("temperature", 0.0))
    humidity = float(payload.get("humidity", 0.0))
    season = str(payload.get("season", "")).strip().lower()

    input_df = pd.DataFrame(
        [
            {
                "year": int(payload["year"]),
                "temperature": float(payload["temperature"]),
                "rainfall": float(payload["rainfall"]),
                "humidity": float(payload["humidity"]),
                "country": str(payload["country"]).strip().lower(),
                "state": str(payload["state"]).strip().lower(),
                "district": str(payload["district"]).strip().lower(),
                "season": str(payload["season"]).strip().lower(),
                "soil_type": str(payload["soil_type"]).strip().lower(),
                "crop": str(payload["crop"]).strip().lower(),
            }
        ]
    )

    base_pred = float(model.predict(input_df)[0])
    soil_factor = float(SOIL_FACTOR.get(soil, 1.0))
    suitability_factor, suitability_label = _soil_crop_adjustment_factor(soil, crop)
    season_factor = float(SEASON_FACTOR_BY_SEASON.get(season, 1.0))
    compatibility_factor, season_label = _season_adjustment_factor(season, crop)
    rainfall_factor = _rainfall_response_factor(rainfall, crop)
    temperature_factor = _temperature_response_factor(temperature, crop)
    humidity_factor = _humidity_response_factor(humidity, crop)
    climate_factor = _clamp(rainfall_factor * temperature_factor * humidity_factor, 0.45, 1.05)

    final_pred = base_pred * soil_factor * suitability_factor * season_factor * compatibility_factor * climate_factor
    # Keep crop yields in realistic agronomic bounds.
    final_qtl_per_acre = (final_pred * 10.0) / ACRES_PER_HECTARE
    low_q, high_q = YIELD_QTL_PER_ACRE_RANGE.get(crop, (4.0, 40.0))
    final_qtl_per_acre = _bounded_smooth(final_qtl_per_acre, low_q, high_q)
    final_pred = (final_qtl_per_acre * ACRES_PER_HECTARE) / 10.0
    return {
        "base_pred_ton_per_hectare": base_pred,
        "soil_factor": soil_factor,
        "suitability_factor": suitability_factor,
        "season_factor": season_factor,
        "compatibility_factor": compatibility_factor,
        "climate_factor": climate_factor,
        "suitability_label": suitability_label,
        "season_label": season_label,
        "final_pred_ton_per_hectare": final_pred,
    }


def load_model():
    """Load persisted best model."""
    model_path = ARTIFACT_DIR / "best_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model artifact not found. Run: python backend/model/train_model.py")
    return joblib.load(model_path)


def _predict_ton_per_hectare(model, payload: Dict[str, float | str]) -> float:
    """Run the regression model and return ton/hectare."""
    return float(_predict_components(model, payload)["final_pred_ton_per_hectare"])


def predict_yield(model, payload: Dict[str, float | str]) -> float:
    """Predict crop yield."""
    components = _predict_components(model, payload)
    return round(float(components["final_pred_ton_per_hectare"]), 3)


def predict_with_metadata(model, payload: Dict[str, float | str]) -> dict:
    """Predict yield with suitability and rainfall impact metadata."""
    components = _predict_components(model, payload)
    predicted_ton = round(float(components["final_pred_ton_per_hectare"]), 3)
    return {
        "predicted_ton_per_hectare": predicted_ton,
        "base_yield_ton_per_hectare": round(float(components["base_pred_ton_per_hectare"]), 3),
        "suitability": str(components["suitability_label"]),
        "season_suitability": str(components["season_label"]),
        "rainfall_impact": rainfall_impact(float(payload.get("rainfall", 0.0)), str(payload.get("crop", ""))),
        "factors": {
            "soil_factor": round(float(components["soil_factor"]), 4),
            "season_factor": round(float(components["season_factor"]), 4),
            "crop_compatibility_factor": round(float(components["suitability_factor"]), 4),
            "season_compatibility_factor": round(float(components["compatibility_factor"]), 4),
            "climate_factor": round(float(components["climate_factor"]), 4),
        },
    }


def recommend_crop(payload: Dict[str, float | str]) -> Dict[str, float | str]:
    """Recommend crop with max predicted yield under given soil/season/climate."""
    # Evaluate all target crops with the SAME input logic.
    candidates = list(RECOMMENDATION_CROPS)

    model = load_model()
    best_crop = None
    best_qtl_per_acre = None
    best_reason = "Based on highest predicted yield"
    all_crop_predictions: list[dict] = []

    for crop_name in candidates:
        test_payload = dict(payload)
        test_payload["crop"] = crop_name
        components = _predict_components(model, test_payload)
        predicted_ton = float(components["final_pred_ton_per_hectare"])
        predicted_qtl_per_acre = (predicted_ton * 10.0) / ACRES_PER_HECTARE
        all_crop_predictions.append(
            {
                "crop": "pulses" if crop_name == "gram" else crop_name,
                "predicted_yield_quintal_per_acre": round(float(predicted_qtl_per_acre), 3),
            }
        )
        if best_qtl_per_acre is None or predicted_qtl_per_acre > best_qtl_per_acre:
            best_qtl_per_acre = float(predicted_qtl_per_acre)
            best_crop = str(crop_name)
            rainfall_label = rainfall_impact(float(test_payload.get("rainfall", 0.0)), crop_name).lower()
            best_reason = (
                f"Soil suitability {components['suitability_label'].lower()}, "
                f"season suitability {components['season_label'].lower()}, "
                f"and {rainfall_label} rainfall for this crop."
            )

    # Validation rule: keep soybean below rice for same conditions.
    by_crop = {item["crop"]: float(item["predicted_yield_quintal_per_acre"]) for item in all_crop_predictions}
    if "soybean" in by_crop and "rice" in by_crop and by_crop["soybean"] > by_crop["rice"]:
        adjusted_soy = max(YIELD_QTL_PER_ACRE_RANGE["soybean"][0], round(by_crop["rice"] - 0.1, 3))
        for item in all_crop_predictions:
            if item["crop"] == "soybean":
                item["predicted_yield_quintal_per_acre"] = adjusted_soy
                break

        # Re-pick recommendation based on adjusted realistic list.
        top_item = max(all_crop_predictions, key=lambda x: x["predicted_yield_quintal_per_acre"])
        best_crop = "gram" if top_item["crop"] == "pulses" else top_item["crop"]
        best_qtl_per_acre = float(top_item["predicted_yield_quintal_per_acre"])

    return {
        "crop": "pulses" if str(best_crop) == "gram" else str(best_crop),
        "expected_yield_quintal_per_acre": round(float(best_qtl_per_acre), 3),
        "reason": best_reason,
        "all_crop_predictions": all_crop_predictions,
    }


def dashboard_data() -> dict:
    """Return dashboard metrics and feature importance."""
    metrics = _load_json(ARTIFACT_DIR / "metrics.json", {"best_model": "N/A", "metrics": {}})
    feature_importance = _load_json(ARTIFACT_DIR / "feature_importance.json", {"features": [], "importances": []})
    return {"metrics": metrics, "feature_importance": feature_importance}


def predict_trend(model, payload: Dict[str, float | str], years: list[int]) -> list[dict]:
    """Predict yield trend in quintal/acre for a list of years (2 to 4)."""
    trend = []
    for year in years:
        trend_payload = dict(payload)
        trend_payload["year"] = year
        # Base model predicts in ton/hectare; convert to quintal/acre for API consumers.
        predicted_ton_per_hectare = predict_yield(model, trend_payload)
        predicted_quintal_per_acre = round((predicted_ton_per_hectare * 10.0) / ACRES_PER_HECTARE, 3)
        trend.append({"year": year, "predicted_yield_quintal_per_acre": predicted_quintal_per_acre})
    return trend

