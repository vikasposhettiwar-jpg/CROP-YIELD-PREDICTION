"""Backend Flask API server."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request
from flask_cors import CORS

from model.predict import (
    dashboard_data,
    load_model,
    predict_trend,
    predict_with_metadata,
    predict_yield,
    recommend_crop,
)
from model.train_model import DATA_PATH
from model.train_model import train_from_dataframe
from utils.data_store import init_db, insert_entry, training_dataframe_actual_only
from utils.preprocessing import load_and_clean_dataset

app = Flask(__name__)
CORS(app)
model = None
ACRES_PER_HECTARE = 2.47105
_districts_by_state: dict[str, list[str]] | None = None
OFFICIAL_DISTRICTS_PATH = Path(__file__).resolve().parent / "data" / "official_districts_by_state.json"


def ton_per_hectare_to_quintal_per_acre(value: float) -> float:
    """Convert ton/hectare to quintal/acre."""
    return round((float(value) * 10.0) / ACRES_PER_HECTARE, 3)


def quintal_per_acre_to_ton_per_hectare(value: float) -> float:
    """Convert quintal/acre to ton/hectare."""
    return float(value) * ACRES_PER_HECTARE / 10.0


def _load_districts_by_state() -> dict[str, list[str]]:
    """Load state -> districts map from a curated JSON file (cached)."""
    global _districts_by_state  # noqa: PLW0603
    if _districts_by_state is not None:
        return _districts_by_state

    try:
        raw = json.loads(OFFICIAL_DISTRICTS_PATH.read_text(encoding="utf-8"))
    except FileNotFoundError:
        _districts_by_state = {}
        return _districts_by_state

    mapping: dict[str, list[str]] = {}
    for state_key, districts in (raw or {}).items():
        sk = str(state_key).strip().lower()
        district_set = {str(d).strip() for d in (districts or []) if str(d).strip()}
        # Sort for stable UI ordering, but keep the original district casing.
        normalized = sorted(district_set, key=lambda s: s.lower())
        mapping[sk] = normalized

    _districts_by_state = mapping
    return _districts_by_state


@app.get("/api/districts")
def get_districts():
    """Return districts for a given state (used by frontend dropdown)."""
    state = request.args.get("state", "").strip().lower()
    if not state:
        return jsonify({"error": "Query param `state` is required."}), 400

    mapping = _load_districts_by_state()
    return jsonify({"districts": mapping.get(state, [])})


def validate_payload(payload: dict) -> tuple[bool, str]:
    """Validate request payload fields."""
    required = [
        "year",
        "temperature",
        "rainfall",
        "humidity",
        "country",
        "state",
        "district",
        "season",
        "soil_type",
        "crop",
    ]
    for field in required:
        if field not in payload:
            return False, f"Missing required field: {field}"
    try:
        year = int(payload["year"])
        temperature = float(payload["temperature"])
        rainfall = float(payload["rainfall"])
        humidity = float(payload["humidity"])
        country = str(payload["country"]).strip().lower()
        state = str(payload["state"]).strip()
        district = str(payload["district"]).strip()
        season = str(payload["season"]).strip().lower()
        soil_type = str(payload["soil_type"]).strip()
        crop = str(payload["crop"]).strip()
    except (TypeError, ValueError):
        return False, "temperature, rainfall, and humidity must be numeric."

    if not (2000 <= year <= 2100):
        return False, "year must be between 2000 and 2100."
    if not (0 <= temperature <= 55):
        return False, "temperature must be between 0 and 55 C."
    if not (0 <= rainfall <= 1500):
        return False, "rainfall must be between 0 and 1500 mm."
    if not (0 <= humidity <= 100):
        return False, "humidity must be between 0 and 100."
    if not soil_type:
        return False, "soil_type cannot be empty."
    if not country:
        return False, "country cannot be empty."
    if country != "india":
        return False, "country must be india."
    if not state:
        return False, "state cannot be empty."
    if not district:
        return False, "district cannot be empty."
    if season not in {"kharif", "rabi", "zaid", "whole year"}:
        return False, "season must be one of: kharif, rabi, zaid, whole year."
    if not crop:
        return False, "crop cannot be empty."
    return True, "ok"


def aggregate_recent_conditions(entries: list[dict]) -> dict:
    """Average recent 3-4 year weather conditions for recommendation input."""
    if not entries:
        return {}
    ordered = sorted(entries, key=lambda item: int(item["year"]))
    recent = ordered[-4:]
    base = dict(recent[-1])
    base["temperature"] = round(sum(float(item["temperature"]) for item in recent) / len(recent), 3)
    base["rainfall"] = round(sum(float(item["rainfall"]) for item in recent) / len(recent), 3)
    base["humidity"] = round(sum(float(item["humidity"]) for item in recent) / len(recent), 3)
    return base


def evaluate_prediction_accuracy(predicted_q_per_acre: float, actual_q_per_acre: float) -> dict:
    """Return error metrics and qualitative accuracy status."""
    error = round(float(predicted_q_per_acre) - float(actual_q_per_acre), 3)
    absolute_error = round(abs(error), 3)
    if absolute_error <= 2.0:
        status = "Highly Accurate"
    elif absolute_error <= 3.0:
        status = "Acceptable Accuracy"
    else:
        status = "Low Accuracy"
    return {"error_q_per_acre": error, "absolute_error_q_per_acre": absolute_error, "accuracy_status": status}


@app.get("/api/health")
def health():
    """Health endpoint."""
    return jsonify({"status": "ok"})


@app.get("/api/dashboard")
def get_dashboard():
    """Return model comparison and explainability data."""
    return jsonify(dashboard_data())


@app.post("/api/predict")
def predict():
    """Predict crop yield."""
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400
    try:
        # Base model predicts in ton/hectare; expose API in quintal/acre.
        details = predict_with_metadata(model, payload)
        predicted_ton_per_hectare = details["predicted_ton_per_hectare"]
        predicted_quintal_per_acre = ton_per_hectare_to_quintal_per_acre(predicted_ton_per_hectare)
        base_quintal_per_acre = ton_per_hectare_to_quintal_per_acre(details["base_yield_ton_per_hectare"])
        return jsonify(
            {
                "predicted_yield_quintal_per_acre": predicted_quintal_per_acre,
                "base_yield_quintal_per_acre": base_quintal_per_acre,
                "rainfall_impact": details["rainfall_impact"],
                "soil_type": str(payload.get("soil_type", "")),
                "crop": str(payload.get("crop", "")),
                "suitability": details["suitability"],
                "factor_breakdown": details["factors"],
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.post("/api/recommend")
def recommend():
    """Recommend best crop."""
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400
    try:
        result = recommend_crop(payload)
        max_from_list = max((item["predicted_yield_quintal_per_acre"] for item in result["all_crop_predictions"]), default=None)
        expected = result["expected_yield_quintal_per_acre"]
        if max_from_list is None or abs(float(max_from_list) - float(expected)) > 1e-6:
            return jsonify({"error": "Recommendation logic mismatch"}), 500
        return jsonify(
            {
                "recommended_crop": result["crop"],
                "expected_yield_quintal_per_acre": result["expected_yield_quintal_per_acre"],
                "reason": result["reason"],
                "all_crop_predictions": result["all_crop_predictions"],
            }
        )
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Recommendation failed: {exc}"}), 500


@app.post("/api/predict-trend")
def predict_year_trend():
    """Predict yield trend for 2 to 4 years."""
    payload = request.get_json(silent=True) or {}
    years = payload.get("years", [])
    if not isinstance(years, list) or not all(isinstance(y, int) for y in years):
        return jsonify({"error": "years must be a list of integers."}), 400
    if len(years) < 2 or len(years) > 4:
        return jsonify({"error": "Provide 2 to 4 years for trend prediction."}), 400

    base_payload = {k: v for k, v in payload.items() if k != "years"}
    valid, message = validate_payload({**base_payload, "year": years[0]})
    if not valid:
        return jsonify({"error": message}), 400

    try:
        trend = predict_trend(model, base_payload, years)
        return jsonify({"trend": trend})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": f"Trend prediction failed: {exc}"}), 500


@app.post("/api/predict-recommend")
def predict_recommend_multi_year():
    """Predict yields for 3-4 yearly entries and return one crop recommendation."""
    payload = request.get_json(silent=True) or {}
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        return jsonify({"error": "entries must be a list of yearly records."}), 400
    if len(entries) < 3 or len(entries) > 4:
        return jsonify({"error": "Please provide 3 to 4 yearly entries."}), 400

    validated_entries = []
    yearly_predictions = []
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            return jsonify({"error": f"Entry {idx + 1} must be an object."}), 400
        valid, message = validate_payload(entry)
        if not valid:
            return jsonify({"error": f"Entry {idx + 1}: {message}"}), 400
        validated_entries.append(entry)

        try:
            # Base model predicts in ton/hectare; convert to quintal/acre for response.
            details = predict_with_metadata(model, entry)
            predicted_ton_per_hectare = details["predicted_ton_per_hectare"]
            predicted_quintal_per_acre = ton_per_hectare_to_quintal_per_acre(predicted_ton_per_hectare)
        except Exception as exc:  # pylint: disable=broad-except
            return jsonify({"error": f"Prediction failed for entry {idx + 1}: {exc}"}), 500

        # Accept optional actual yield from client in q/acre (preferred), plus legacy q/ha or ton/ha.
        actual_quintal_per_acre = None
        raw_actual_quintal_per_acre = entry.get("actual_yield_quintal_per_acre")
        raw_actual_quintal_per_hectare = (
            entry.get("actual_yield_quintal_per_hectare") if raw_actual_quintal_per_acre in (None, "") else None
        )
        raw_actual_ton = (
            entry.get("actual_yield_ton_per_hectare")
            if raw_actual_quintal_per_acre in (None, "") and raw_actual_quintal_per_hectare in (None, "")
            else None
        )

        if raw_actual_quintal_per_acre not in (None, ""):
            try:
                actual_quintal_per_acre = float(raw_actual_quintal_per_acre)
            except (TypeError, ValueError):
                return jsonify({"error": f"Entry {idx + 1}: actual_yield_quintal_per_acre must be numeric."}), 400
        elif raw_actual_quintal_per_hectare not in (None, ""):
            try:
                actual_quintal_per_acre = round(float(raw_actual_quintal_per_hectare) / ACRES_PER_HECTARE, 3)
            except (TypeError, ValueError):
                return jsonify(
                    {"error": f"Entry {idx + 1}: actual_yield_quintal_per_hectare must be numeric."}
                ), 400
        elif raw_actual_ton not in (None, ""):
            try:
                actual_quintal_per_acre = ton_per_hectare_to_quintal_per_acre(float(raw_actual_ton))
            except (TypeError, ValueError):
                return jsonify({"error": f"Entry {idx + 1}: actual_yield_ton_per_hectare must be numeric."}), 400

        yearly_predictions.append(
            {
                "year": int(entry["year"]),
                "predicted_yield_quintal_per_acre": predicted_quintal_per_acre,
                "base_yield_quintal_per_acre": ton_per_hectare_to_quintal_per_acre(details["base_yield_ton_per_hectare"]),
                "actual_yield_quintal_per_acre": actual_quintal_per_acre,
                "soil_type": str(entry.get("soil_type", "")),
                "crop": str(entry.get("crop", "")),
                "rainfall_impact": details["rainfall_impact"],
                "suitability": details["suitability"],
                "factor_breakdown": details["factors"],
            }
        )
        if actual_quintal_per_acre is not None:
            yearly_predictions[-1].update(
                evaluate_prediction_accuracy(
                    predicted_q_per_acre=predicted_quintal_per_acre,
                    actual_q_per_acre=actual_quintal_per_acre,
                )
            )

    recommended_input = aggregate_recent_conditions(validated_entries)
    recommendation = recommend_crop(recommended_input)
    max_from_list = max((item["predicted_yield_quintal_per_acre"] for item in recommendation["all_crop_predictions"]), default=None)
    expected = recommendation["expected_yield_quintal_per_acre"]
    mismatch_warning = None
    if max_from_list is None or abs(float(max_from_list) - float(expected)) > 1e-6:
        mismatch_warning = "Recommendation logic mismatch"

    # Validation: warn if model outputs are constant across years.
    predicted_values = [float(item["predicted_yield_quintal_per_acre"]) for item in yearly_predictions]
    warning = None
    if predicted_values and len({round(v, 3) for v in predicted_values}) == 1:
        warning = "Model not trained properly"

    # Validation: if inputs differ but output is same, flag model-input usage issue.
    input_signatures = {
        (
            round(float(entry["temperature"]), 4),
            round(float(entry["rainfall"]), 4),
            round(float(entry["humidity"]), 4),
            str(entry.get("soil_type", "")).strip().lower(),
            str(entry.get("crop", "")).strip().lower(),
            str(entry.get("season", "")).strip().lower(),
        )
        for entry in validated_entries
    }
    if len(input_signatures) > 1 and predicted_values and len({round(v, 3) for v in predicted_values}) == 1:
        msg = "Model not using inputs properly"
        warning = f"{warning}; {msg}" if warning else msg

    # Validation: detect suspiciously identical predicted/actual pairs.
    accuracy_rows = [row for row in yearly_predictions if row.get("actual_yield_quintal_per_acre") is not None]
    if len(accuracy_rows) >= 2:
        near_zero_abs_errors = [
            float(row.get("absolute_error_q_per_acre", 999.0)) <= 0.1 for row in accuracy_rows
        ]
        if all(near_zero_abs_errors):
            msg = "Model may be overfitting or using actual values incorrectly"
            warning = f"{warning}; {msg}" if warning else msg

    # Validation: ensure soil/crop influence is visible.
    # Same climate + crop but different soil should change yield; same climate + soil but different crop should change yield.
    sensitivity_warning = None
    try:
        probe = dict(recommended_input)
        probe_year = int(probe.get("year", validated_entries[-1]["year"]))
        probe["year"] = probe_year
        base_crop = str(probe.get("crop", "")).strip().lower() or str(validated_entries[-1]["crop"]).strip().lower()
        base_soil = str(probe.get("soil_type", "")).strip().lower() or str(validated_entries[-1]["soil_type"]).strip().lower()

        soil_variants = ["black", "red", "alluvial"]
        crop_variants = ["rice", "cotton", "maize", "soybean", "wheat", "groundnut", "gram"]

        soil_preds = []
        for soil in soil_variants:
            soil_payload = dict(probe)
            soil_payload["crop"] = base_crop
            soil_payload["soil_type"] = soil
            ton = predict_yield(model, soil_payload)
            soil_preds.append(ton_per_hectare_to_quintal_per_acre(ton))

        crop_preds = []
        for crop_name in crop_variants:
            crop_payload = dict(probe)
            crop_payload["soil_type"] = base_soil
            crop_payload["crop"] = crop_name
            ton = predict_yield(model, crop_payload)
            crop_preds.append(ton_per_hectare_to_quintal_per_acre(ton))

        if len({round(v, 3) for v in soil_preds}) == 1 or len({round(v, 3) for v in crop_preds}) == 1:
            sensitivity_warning = "Model is not considering soil/crop properly"
    except Exception:  # pylint: disable=broad-except
        sensitivity_warning = None

    # Validation: ensure season affects yield.
    season_warning = None
    try:
        probe = dict(validated_entries[-1])
        probe_k = dict(probe)
        probe_r = dict(probe)
        probe_k["season"] = "kharif"
        probe_r["season"] = "rabi"
        pred_k = ton_per_hectare_to_quintal_per_acre(predict_yield(model, probe_k))
        pred_r = ton_per_hectare_to_quintal_per_acre(predict_yield(model, probe_r))
        if abs(float(pred_k) - float(pred_r)) < 1e-6:
            season_warning = "Season not affecting prediction"
    except Exception:  # pylint: disable=broad-except
        season_warning = None

    # Comparison data for visualization (use the most recent year inputs).
    comparison = {}
    try:
        latest = dict(validated_entries[-1])
        latest_year = int(latest["year"])
        latest["year"] = latest_year
        chosen_crop = str(latest["crop"]).strip().lower()
        chosen_soil = str(latest["soil_type"]).strip().lower()

        crop_compare = []
        for crop_name in ["rice", "cotton", "maize", "soybean", "wheat", "groundnut", "gram"]:
            p = dict(latest)
            p["crop"] = crop_name
            ton = predict_yield(model, p)
            crop_compare.append({"crop": crop_name, "predicted_yield_quintal_per_acre": ton_per_hectare_to_quintal_per_acre(ton)})

        soil_compare = []
        for soil_name in ["black", "red", "alluvial"]:
            p = dict(latest)
            p["crop"] = chosen_crop
            p["soil_type"] = soil_name
            ton = predict_yield(model, p)
            soil_compare.append({"soil_type": soil_name, "predicted_yield_quintal_per_acre": ton_per_hectare_to_quintal_per_acre(ton)})

        # Rainfall curve (same crop/soil/climate; vary rainfall to show non-linear behavior)
        rainfall_curve = []
        for r in range(0, 1501, 50):
            p = dict(latest)
            p["crop"] = chosen_crop
            p["soil_type"] = chosen_soil
            p["rainfall"] = r
            ton = predict_yield(model, p)
            rainfall_curve.append(
                {
                    "rainfall": r,
                    "predicted_yield_quintal_per_acre": ton_per_hectare_to_quintal_per_acre(ton),
                }
            )

        # Validation: if yield increases continuously with rainfall, warn.
        monotonic_warning = None
        curve_vals = [float(item["predicted_yield_quintal_per_acre"]) for item in rainfall_curve]
        if curve_vals and all(curve_vals[i + 1] >= curve_vals[i] - 1e-6 for i in range(len(curve_vals) - 1)):
            monotonic_warning = "Model is not handling excess rainfall correctly"

        comparison = {
            "yield_by_crop": crop_compare,
            "yield_by_soil": soil_compare,
            "rainfall_curve": rainfall_curve,
        }
        if monotonic_warning and not (warning or sensitivity_warning):
            sensitivity_warning = monotonic_warning
    except Exception:  # pylint: disable=broad-except
        comparison = {}
    return jsonify(
        {
            "predictions": sorted(yearly_predictions, key=lambda item: item["year"]),
            "recommended_crop": recommendation["crop"],
            "expected_yield_quintal_per_acre": recommendation["expected_yield_quintal_per_acre"],
            "recommendation_reason": recommendation["reason"],
            "warning": warning or sensitivity_warning or season_warning or mismatch_warning,
            "comparison": comparison,
            "normalization_note": "Model uses StandardScaler + one-hot encoding inside training pipeline.",
            "unit_consistency_note": "Inputs are validated in mm (rainfall), C (temperature), and % (humidity).",
        }
    )


def initialize_model():
    """Load trained model artifact."""
    global model  # noqa: PLW0603
    init_db()
    model = load_model()


@app.post("/api/save-entry")
def save_entry():
    """Persist a user-entered record into SQLite for incremental training."""
    payload = request.get_json(silent=True) or {}
    valid, message = validate_payload(payload)
    if not valid:
        return jsonify({"error": message}), 400

    # Accept actual yield from client in q/acre (preferred), plus legacy q/ha or ton/ha.
    raw_actual_quintal_per_acre = payload.get("actual_yield_quintal_per_acre")
    raw_actual_quintal_per_hectare = (
        payload.get("actual_yield_quintal_per_hectare") if raw_actual_quintal_per_acre in (None, "") else None
    )
    raw_actual_ton = (
        payload.get("actual_yield_ton_per_hectare")
        if raw_actual_quintal_per_acre in (None, "") and raw_actual_quintal_per_hectare in (None, "")
        else None
    )

    label_source = "actual"
    if raw_actual_quintal_per_acre in (None, "") and raw_actual_quintal_per_hectare in (None, "") and raw_actual_ton in (None, ""):
        # No actual provided: fall back to model prediction in ton/hectare.
        predicted_ton_per_hectare = predict_yield(model, payload)
        actual_ton_per_hectare = float(predicted_ton_per_hectare)
        label_source = "predicted_proxy"
    else:
        if raw_actual_quintal_per_acre not in (None, ""):
            try:
                actual_quintal_per_acre = float(raw_actual_quintal_per_acre)
            except (TypeError, ValueError):
                return jsonify({"error": "actual_yield_quintal_per_acre must be numeric when provided."}), 400
            actual_ton_per_hectare = quintal_per_acre_to_ton_per_hectare(actual_quintal_per_acre)
        elif raw_actual_quintal_per_hectare not in (None, ""):
            try:
                actual_quintal_per_hectare = float(raw_actual_quintal_per_hectare)
            except (TypeError, ValueError):
                return jsonify({"error": "actual_yield_quintal_per_hectare must be numeric when provided."}), 400
            actual_ton_per_hectare = actual_quintal_per_hectare / 10.0
        else:
            try:
                actual_ton_per_hectare = float(raw_actual_ton)
            except (TypeError, ValueError):
                return jsonify({"error": "actual_yield_ton_per_hectare must be numeric when provided."}), 400

    row_id = insert_entry(payload, float(actual_ton_per_hectare), label_source)
    return jsonify(
        {
            "message": "Entry saved for model retraining.",
            "entry_id": row_id,
            "label_source": label_source,
            "stored_yield_ton_per_hectare": round(float(actual_ton_per_hectare), 3),
            "stored_yield_quintal_per_acre": ton_per_hectare_to_quintal_per_acre(actual_ton_per_hectare),
        }
    )


@app.post("/api/retrain-from-db")
def retrain_from_db():
    """Retrain model directly from user-entered database records."""
    global model  # noqa: PLW0603
    payload = request.get_json(silent=True) or {}
    min_records = int(payload.get("min_records", 20))
    include_base_dataset = bool(payload.get("include_base_dataset", True))

    df = training_dataframe_actual_only()
    if df.empty or len(df) < min_records:
        return jsonify({"error": f"Need at least {min_records} saved records with actual yield to retrain."}), 400

    required_columns = [
        "year",
        "temperature",
        "rainfall",
        "humidity",
        "country",
        "state",
        "district",
        "season",
        "soil_type",
        "crop",
        "yield_ton_per_hectare",
    ]
    for col in required_columns:
        if col not in df.columns:
            return jsonify({"error": f"DB training data missing required column: {col}"}), 500

    for col in ["year", "temperature", "rainfall", "humidity", "yield_ton_per_hectare"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required_columns).reset_index(drop=True)
    if len(df) < min_records:
        return jsonify({"error": "Not enough clean records after validation for retraining."}), 400

    training_df = df.copy()
    source = "sqlite_actual_entries"
    if include_base_dataset:
        base_df = load_and_clean_dataset(DATA_PATH).dataframe
        training_df = pd.concat([base_df, df], ignore_index=True)
        source = "csv_plus_sqlite_actual_entries"

    best_name = train_from_dataframe(training_df, source=source)
    model = load_model()
    return jsonify(
        {
            "message": "Model retrained successfully.",
            "best_model": best_name,
            "actual_records_used": len(df),
            "total_training_rows": len(training_df),
            "include_base_dataset": include_base_dataset,
        }
    )


if __name__ == "__main__":
    initialize_model()
    app.run(host="0.0.0.0", port=5002, debug=True)

