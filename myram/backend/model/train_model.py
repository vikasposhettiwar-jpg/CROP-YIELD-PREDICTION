"""Training script for backend model artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import build_preprocessor, load_and_clean_dataset

DATA_PATH = BASE_DIR / "data" / "crop_yield_india.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
FRONTEND_IMG_DIR = BASE_DIR.parent / "frontend" / "images"


def evaluate_model(model: Pipeline, x_test, y_test) -> Dict[str, float]:
    """Compute RMSE, MAE and R2."""
    predictions = model.predict(x_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "mae": float(mean_absolute_error(y_test, predictions)),
        "r2": float(r2_score(y_test, predictions)),
    }


def create_model_comparison_chart(metrics: Dict[str, Dict[str, float]]) -> None:
    """Create model comparison dashboard chart."""
    model_names = list(metrics.keys())
    rmse_values = [metrics[m]["rmse"] for m in model_names]
    mae_values = [metrics[m]["mae"] for m in model_names]
    r2_values = [metrics[m]["r2"] for m in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(model_names, rmse_values, color="#ff6b6b")
    axes[0].set_title("RMSE (Lower is Better)")
    axes[1].bar(model_names, mae_values, color="#feca57")
    axes[1].set_title("MAE (Lower is Better)")
    axes[2].bar(model_names, r2_values, color="#1dd1a1")
    axes[2].set_title("R2 (Higher is Better)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(FRONTEND_IMG_DIR / "model_comparison.png", dpi=160)
    plt.close()


def create_feature_importance_chart(model: Pipeline, numeric_features: list[str], categorical_features: list[str]):
    """Create feature importance chart if estimator supports it."""
    estimator = model.named_steps["regressor"]
    if not hasattr(estimator, "feature_importances_"):
        return [], []

    preprocessor = model.named_steps["preprocessor"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    feature_names = numeric_features + list(onehot.get_feature_names_out(categorical_features))
    importances = estimator.feature_importances_

    sorted_idx = np.argsort(importances)[::-1][:10]
    top_features = [feature_names[i] for i in sorted_idx]
    top_importances = [float(importances[i]) for i in sorted_idx]

    plt.figure(figsize=(9, 5))
    plt.barh(top_features[::-1], top_importances[::-1], color="#54a0ff")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(FRONTEND_IMG_DIR / "feature_importance.png", dpi=160)
    plt.close()
    return top_features, top_importances


def build_recommendation_stats(df):
    """Build soil and overall crop yield means for recommendation."""
    grouped = (
        df.groupby(["soil_type", "crop"], as_index=False)["yield_ton_per_hectare"]
        .mean()
        .sort_values(by="yield_ton_per_hectare", ascending=False)
    )
    soil_crop_map = {}
    for _, row in grouped.iterrows():
        soil_crop_map.setdefault(str(row["soil_type"]), {})[str(row["crop"])] = float(row["yield_ton_per_hectare"])

    overall = (
        df.groupby("crop", as_index=False)["yield_ton_per_hectare"]
        .mean()
        .sort_values(by="yield_ton_per_hectare", ascending=False)
    )
    overall_map = {row["crop"]: float(row["yield_ton_per_hectare"]) for _, row in overall.iterrows()}
    return {"soil_crop_mean": soil_crop_map, "overall_crop_mean": overall_map}


def build_climate_stats(df: pd.DataFrame) -> dict:
    """Persist crop-level medians for climate sensitivity adjustments."""
    stats = {}
    for crop, group in df.groupby("crop"):
        stats[str(crop)] = {
            "temperature_median": float(group["temperature"].median()),
            "rainfall_median": float(group["rainfall"].median()),
            "humidity_median": float(group["humidity"].median()),
        }
    overall = {
        "temperature_median": float(df["temperature"].median()),
        "rainfall_median": float(df["rainfall"].median()),
        "humidity_median": float(df["humidity"].median()),
    }
    return {"by_crop": stats, "overall": overall}


def train_from_dataframe(df: pd.DataFrame, source: str = "dataset") -> str:
    """Train and persist artifacts from an already prepared dataframe."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    FRONTEND_IMG_DIR.mkdir(parents=True, exist_ok=True)

    feature_columns = [
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
    target_column = "yield_ton_per_hectare"

    x = df[feature_columns]
    y = df[target_column]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    preprocessor, numeric_features, categorical_features = build_preprocessor(feature_columns)
    # Prefer tree-based models so one-hot encoded soil/crop features are utilized strongly
    # and feature importances can be reported.
    candidates = {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=600,
            random_state=42,
            min_samples_leaf=2,
            n_jobs=-1,
        )
    }

    trained_models, metrics = {}, {}
    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        metrics[name] = evaluate_model(pipeline, x_test, y_test)

    best_name = min(metrics, key=lambda k: (metrics[k]["rmse"], -metrics[k]["r2"]))
    best_model = trained_models[best_name]
    top_features, top_importances = create_feature_importance_chart(best_model, numeric_features, categorical_features)
    create_model_comparison_chart(metrics)

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(
        json.dumps({"best_model": best_name, "metrics": metrics, "training_source": source}, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "feature_importance.json").write_text(
        json.dumps({"features": top_features, "importances": top_importances}, indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "recommendation_stats.json").write_text(
        json.dumps(build_recommendation_stats(df), indent=2), encoding="utf-8"
    )
    (ARTIFACT_DIR / "climate_stats.json").write_text(
        json.dumps(build_climate_stats(df), indent=2), encoding="utf-8"
    )
    return best_name


def main() -> None:
    """Run complete training pipeline from base CSV dataset."""
    bundle = load_and_clean_dataset(DATA_PATH)
    df = bundle.dataframe
    best_name = train_from_dataframe(df, source="csv_dataset")
    print(f"Training complete. Best model: {best_name}")


if __name__ == "__main__":
    main()

