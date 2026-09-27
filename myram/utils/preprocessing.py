"""Data preprocessing utilities for crop yield project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DatasetBundle:
    """Container for cleaned data and selected model columns."""

    dataframe: pd.DataFrame
    feature_columns: List[str]
    target_column: str


def load_and_clean_dataset(csv_path: str | Path) -> DatasetBundle:
    """Load dataset from CSV, normalize schema, and clean missing values."""
    df = pd.read_csv(csv_path)
    df.columns = [col.strip().lower() for col in df.columns]

    required_columns = [
        "crop",
        "soil_type",
        "temperature",
        "rainfall",
        "humidity",
        "yield_ton_per_hectare",
    ]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Dataset missing required columns: {missing_columns}")

    for col in ["temperature", "rainfall", "humidity", "yield_ton_per_hectare"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["soil_type"] = df["soil_type"].astype(str).str.strip().str.lower()
    df["crop"] = df["crop"].astype(str).str.strip().str.lower()

    # Drop rows where target is missing.
    df = df.dropna(subset=["yield_ton_per_hectare"]).reset_index(drop=True)

    # Fill categorical missing values with mode.
    for col in ["soil_type", "crop"]:
        mode_val = df[col].mode(dropna=True)
        fallback = "unknown" if mode_val.empty else mode_val.iloc[0]
        df[col] = df[col].replace({"nan": None, "none": None})
        df[col] = df[col].fillna(fallback)

    feature_columns = ["temperature", "rainfall", "humidity", "soil_type"]
    target_column = "yield_ton_per_hectare"
    return DatasetBundle(dataframe=df, feature_columns=feature_columns, target_column=target_column)


def build_preprocessor(feature_columns: List[str]) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Create preprocessing pipeline with imputation, encoding, and scaling."""
    numeric_features = [f for f in feature_columns if f in ["temperature", "rainfall", "humidity"]]
    categorical_features = [f for f in feature_columns if f == "soil_type"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor, numeric_features, categorical_features

