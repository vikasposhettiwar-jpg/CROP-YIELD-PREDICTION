"""Training script for smart crop yield prediction system."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from utils.preprocessing import build_preprocessor, load_and_clean_dataset

DATA_PATH = BASE_DIR / "data" / "crop_yield_india.csv"
ARTIFACT_DIR = BASE_DIR / "artifacts"
STATIC_IMG_DIR = BASE_DIR / "static" / "images"


def evaluate_model(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """Compute evaluation metrics for a trained model."""
    predictions = model.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, predictions)))
    mae = float(mean_absolute_error(y_test, predictions))
    r2 = float(r2_score(y_test, predictions))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def create_model_comparison_chart(metrics: Dict[str, Dict[str, float]]) -> None:
    """Generate and save model comparison visualization."""
    model_names = list(metrics.keys())
    rmse_values = [metrics[name]["rmse"] for name in model_names]
    mae_values = [metrics[name]["mae"] for name in model_names]
    r2_values = [metrics[name]["r2"] for name in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].bar(model_names, rmse_values, color="#ff6b6b")
    axes[0].set_title("RMSE (Lower is Better)")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(model_names, mae_values, color="#feca57")
    axes[1].set_title("MAE (Lower is Better)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].bar(model_names, r2_values, color="#1dd1a1")
    axes[2].set_title("R2 (Higher is Better)")
    axes[2].tick_params(axis="x", rotation=20)

    plt.tight_layout()
    plt.savefig(STATIC_IMG_DIR / "model_comparison.png", dpi=160)
    plt.close()


def create_feature_importance_chart(
    model: Pipeline, numeric_features: list[str], categorical_features: list[str]
) -> Tuple[list[str], list[float]]:
    """Generate feature importance chart when best model supports it."""
    estimator = model.named_steps["regressor"]
    if not hasattr(estimator, "feature_importances_"):
        return [], []

    preprocessor = model.named_steps["preprocessor"]
    onehot = preprocessor.named_transformers_["cat"].named_steps["onehot"]
    onehot_names = list(onehot.get_feature_names_out(categorical_features))
    feature_names = numeric_features + onehot_names
    importances = estimator.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    top_idx = sorted_idx[:10]

    top_features = [feature_names[i] for i in top_idx]
    top_importances = [float(importances[i]) for i in top_idx]

    plt.figure(figsize=(9, 5))
    plt.barh(top_features[::-1], top_importances[::-1], color="#54a0ff")
    plt.title("Top Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(STATIC_IMG_DIR / "feature_importance.png", dpi=160)
    plt.close()

    return top_features, top_importances


def build_recommendation_stats(df: pd.DataFrame) -> Dict[str, dict]:
    """Prepare crop recommendation statistics from historical data."""
    grouped = (
        df.groupby(["soil_type", "crop"], as_index=False)["yield_ton_per_hectare"]
        .mean()
        .sort_values(by="yield_ton_per_hectare", ascending=False)
    )
    soil_crop_map: Dict[str, Dict[str, float]] = {}
    for _, row in grouped.iterrows():
        soil = str(row["soil_type"])
        crop = str(row["crop"])
        soil_crop_map.setdefault(soil, {})[crop] = float(row["yield_ton_per_hectare"])

    overall_crop_mean = (
        df.groupby("crop", as_index=False)["yield_ton_per_hectare"].mean().sort_values(
            by="yield_ton_per_hectare", ascending=False
        )
    )
    overall_map = {row["crop"]: float(row["yield_ton_per_hectare"]) for _, row in overall_crop_mean.iterrows()}
    return {"soil_crop_mean": soil_crop_map, "overall_crop_mean": overall_map}


def main() -> None:
    """Run end-to-end training and persist model artifacts."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    STATIC_IMG_DIR.mkdir(parents=True, exist_ok=True)

    bundle = load_and_clean_dataset(DATA_PATH)
    df = bundle.dataframe
    x = df[bundle.feature_columns]
    y = df[bundle.target_column]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    preprocessor, numeric_features, categorical_features = build_preprocessor(bundle.feature_columns)

    candidates = {
        "RandomForestRegressor": RandomForestRegressor(n_estimators=250, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=42),
        "LinearRegression": LinearRegression(),
    }

    trained_models: Dict[str, Pipeline] = {}
    metrics: Dict[str, Dict[str, float]] = {}

    for name, estimator in candidates.items():
        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", estimator)])
        pipeline.fit(x_train, y_train)
        trained_models[name] = pipeline
        metrics[name] = evaluate_model(pipeline, x_test, y_test)

    best_name = min(metrics, key=lambda k: (metrics[k]["rmse"], -metrics[k]["r2"]))
    best_model = trained_models[best_name]

    top_features, top_importances = create_feature_importance_chart(
        best_model, numeric_features, categorical_features
    )
    create_model_comparison_chart(metrics)

    recommendation_stats = build_recommendation_stats(df)

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")
    with open(ARTIFACT_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"best_model": best_name, "metrics": metrics}, f, indent=2)
    with open(ARTIFACT_DIR / "feature_importance.json", "w", encoding="utf-8") as f:
        json.dump({"features": top_features, "importances": top_importances}, f, indent=2)
    with open(ARTIFACT_DIR / "recommendation_stats.json", "w", encoding="utf-8") as f:
        json.dump(recommendation_stats, f, indent=2)

    print("Model training complete.")
    print(json.dumps({"best_model": best_name, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()

