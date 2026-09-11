"""Leakage-safe, deterministic feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


REFERENCE_DATE = pd.Timestamp("2025-01-01")

RAW_NUMERIC_FEATURES = [
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "distance",
    "weight_clean",
    "market_index",
    "quote_signal",
]

ENGINEERED_NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + [
    "day_index",
    "day_of_week",
    "month",
    "week_of_year",
    "annual_sin",
    "annual_cos",
    "weekday_sin",
    "weekday_cos",
    "abs_lat_diff",
    "abs_lon_diff",
    "haversine_miles",
    "route_detour_ratio",
    "log_distance",
    "log_weight",
    "distance_x_market_index",
    "distance_x_quote_signal",
]

RISK_SIGNAL_COLUMNS = [
    "market_index",
    "quote_signal",
    "distance_x_market_index",
    "distance_x_quote_signal",
]


def _haversine_miles(frame: pd.DataFrame) -> np.ndarray:
    lat1 = np.radians(frame["pickup_lat"].to_numpy(dtype=float))
    lon1 = np.radians(frame["pickup_lon"].to_numpy(dtype=float))
    lat2 = np.radians(frame["delivery_lat"].to_numpy(dtype=float))
    lon2 = np.radians(frame["delivery_lon"].to_numpy(dtype=float))
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = np.sin(delta_lat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(
        delta_lon / 2.0
    ) ** 2
    return 3958.7613 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


class FreightFeatureEngineer(TransformerMixin, BaseEstimator):
    """Create only row-local features; no target or population statistics are used."""

    def __init__(self, feature_set: str = "engineered") -> None:
        self.feature_set = feature_set

    def fit(self, X: pd.DataFrame, y: object = None) -> "FreightFeatureEngineer":
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        frame = X.copy()
        dates = pd.to_datetime(frame["date"], errors="raise")

        # Negative physical weights are invalid and enter fold-fitted imputation as missing.
        frame["weight_clean"] = frame["weight"].mask(frame["weight"] < 0)
        frame["route"] = frame["pickup"].astype(str) + "__TO__" + frame[
            "delivery"
        ].astype(str)

        if self.feature_set == "raw":
            return frame

        frame["day_index"] = (dates - REFERENCE_DATE).dt.days.astype(float)
        frame["day_of_week"] = dates.dt.dayofweek.astype(float)
        frame["month"] = dates.dt.month.astype(float)
        frame["week_of_year"] = dates.dt.isocalendar().week.astype(float)
        frame["annual_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofyear / 365.25)
        frame["annual_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofyear / 365.25)
        frame["weekday_sin"] = np.sin(2.0 * np.pi * dates.dt.dayofweek / 7.0)
        frame["weekday_cos"] = np.cos(2.0 * np.pi * dates.dt.dayofweek / 7.0)

        frame["abs_lat_diff"] = (frame["delivery_lat"] - frame["pickup_lat"]).abs()
        frame["abs_lon_diff"] = (frame["delivery_lon"] - frame["pickup_lon"]).abs()
        frame["haversine_miles"] = _haversine_miles(frame)
        frame["route_detour_ratio"] = frame["distance"] / frame[
            "haversine_miles"
        ].replace(0.0, np.nan)
        frame["log_distance"] = np.log1p(frame["distance"].clip(lower=0.0))
        frame["log_weight"] = np.log1p(frame["weight_clean"].clip(lower=0.0))
        if self.feature_set == "engineered":
            frame["distance_x_market_index"] = frame["distance"] * frame["market_index"]
            frame["distance_x_quote_signal"] = frame["distance"] * frame["quote_signal"]
        return frame


def numeric_features(feature_set: str) -> list[str]:
    if feature_set == "raw":
        return RAW_NUMERIC_FEATURES.copy()
    if feature_set == "engineered":
        return ENGINEERED_NUMERIC_FEATURES.copy()
    if feature_set == "no_market_quote":
        return [c for c in ENGINEERED_NUMERIC_FEATURES if c not in RISK_SIGNAL_COLUMNS]
    raise ValueError(f"Unknown feature set: {feature_set}")


def categorical_features(*, include_route: bool) -> list[str]:
    columns = ["pickup", "delivery", "equipment"]
    if include_route:
        columns.append("route")
    return columns
