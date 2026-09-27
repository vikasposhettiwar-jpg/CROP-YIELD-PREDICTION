"""SQLite data store for user-entered crop records."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = BASE_DIR / "data" / "user_training_data.db"


def get_connection() -> sqlite3.Connection:
    """Return a SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create table for user-entered records if it does not exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_entries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                country TEXT NOT NULL,
                state TEXT NOT NULL,
                district TEXT NOT NULL,
                season TEXT NOT NULL,
                crop TEXT NOT NULL,
                soil_type TEXT NOT NULL,
                year INTEGER NOT NULL,
                temperature REAL NOT NULL,
                rainfall REAL NOT NULL,
                humidity REAL NOT NULL,
                yield_ton_per_hectare REAL NOT NULL,
                label_source TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()


def insert_entry(payload: dict, target_yield: float, label_source: str) -> int:
    """Insert a user entry and return the row id."""
    with get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO user_entries (
                country, state, district, season, crop, soil_type,
                year, temperature, rainfall, humidity, yield_ton_per_hectare, label_source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(payload["country"]).strip().lower(),
                str(payload["state"]).strip().lower(),
                str(payload["district"]).strip().lower(),
                str(payload["season"]).strip().lower(),
                str(payload["crop"]).strip().lower(),
                str(payload["soil_type"]).strip().lower(),
                int(payload["year"]),
                float(payload["temperature"]),
                float(payload["rainfall"]),
                float(payload["humidity"]),
                float(target_yield),
                label_source,
            ),
        )
        conn.commit()
        return int(cursor.lastrowid)


def training_dataframe() -> pd.DataFrame:
    """Return user-entered records in model-training schema."""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                country, state, district, season, crop, soil_type,
                year, temperature, rainfall, humidity, yield_ton_per_hectare
            FROM user_entries
            ORDER BY id ASC
            """,
            conn,
        )


def training_dataframe_actual_only() -> pd.DataFrame:
    """Return only user records that contain true (non-proxy) labels."""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                country, state, district, season, crop, soil_type,
                year, temperature, rainfall, humidity, yield_ton_per_hectare
            FROM user_entries
            WHERE label_source = 'actual'
            ORDER BY id ASC
            """,
            conn,
        )

