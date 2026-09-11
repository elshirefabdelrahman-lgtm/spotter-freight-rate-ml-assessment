"""Shared loading and schema validation for the freight-rate assessment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TARGET_COLUMN = "posted_rate"
ID_COLUMN = "load_id"
DATE_COLUMN = "date"

FEATURE_COLUMNS = [
    "load_id",
    "pickup",
    "delivery",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "equipment",
    "weight",
    "date",
    "market_index",
    "quote_signal",
]
TRAIN_COLUMNS = FEATURE_COLUMNS + [TARGET_COLUMN]


def _read_csv(name: str, expected_columns: list[str]) -> pd.DataFrame:
    """Read one project CSV and fail early if its schema has changed."""
    path = DATA_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Required data file not found: {path}")
    frame = pd.read_csv(path)
    if frame.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected columns in {path.name}. "
            f"Expected {expected_columns}, found {frame.columns.tolist()}"
        )
    return frame


def load_development(*, parse_dates: bool = True) -> pd.DataFrame:
    """Load labeled development data only."""
    frame = _read_csv("train-test.csv", TRAIN_COLUMNS)
    if parse_dates:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    return frame


def load_unlabeled_validation(*, parse_dates: bool = True) -> pd.DataFrame:
    """Load the unlabeled submission set; never use this frame for fitting."""
    frame = _read_csv("validation.csv", FEATURE_COLUMNS)
    if parse_dates:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    return frame


def load_prediction_template() -> pd.DataFrame:
    return _read_csv(
        "validation-predictions-template.csv", [ID_COLUMN, "predicted_rate"]
    )


def load_december_scenarios(*, parse_dates: bool = True) -> pd.DataFrame:
    columns = [
        "pickup",
        "delivery",
        "distance",
        "equipment",
        "weight",
        "date",
        "predicted_rate",
    ]
    frame = _read_csv("december-chart-inputs.csv", columns)
    if parse_dates:
        frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN], errors="raise")
    return frame

