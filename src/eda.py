"""Deterministic, read-only data audit for Phase 1."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from data_utils import (
    DATE_COLUMN,
    FEATURE_COLUMNS,
    ID_COLUMN,
    TARGET_COLUMN,
    load_december_scenarios,
    load_development,
    load_prediction_template,
    load_unlabeled_validation,
)


def _native(value: object) -> object:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def frame_summary(frame: pd.DataFrame) -> dict[str, object]:
    dates = frame[DATE_COLUMN]
    return {
        "shape": list(frame.shape),
        "columns": frame.columns.tolist(),
        "dtypes": frame.dtypes.astype(str).to_dict(),
        "missing": frame.isna().sum().astype(int).to_dict(),
        "duplicate_rows": int(frame.duplicated().sum()),
        "unique_counts": frame.nunique(dropna=False).astype(int).to_dict(),
        "date_min": dates.min().date().isoformat(),
        "date_max": dates.max().date().isoformat(),
        "unique_dates": int(dates.nunique()),
    }


def build_audit() -> dict[str, object]:
    development = load_development()
    validation = load_unlabeled_validation()
    template = load_prediction_template()
    december = load_december_scenarios()

    comparable_features = [c for c in FEATURE_COLUMNS if c != ID_COLUMN]
    exact_cross_matches = development.merge(
        validation, on=comparable_features, how="inner", suffixes=("_dev", "_val")
    )
    target = development[TARGET_COLUMN]
    target_stats = target.describe(
        percentiles=[0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999]
    ).to_dict()

    id_checks: dict[str, object] = {}
    for label, frame, prefix in [
        ("development", development, "TR"),
        ("validation", validation, "TE"),
    ]:
        ids = frame[ID_COLUMN].astype(str)
        id_checks[label] = {
            "duplicate_ids": int(ids.duplicated().sum()),
            "bad_format": int((~ids.str.fullmatch(rf"{prefix}-\d{{6}}")).sum()),
        }

    categorical_checks = {}
    for column in ["pickup", "delivery", "equipment"]:
        categorical_checks[column] = {
            "development_unique": int(development[column].nunique()),
            "validation_unique": int(validation[column].nunique()),
            "unseen_in_validation": sorted(
                set(validation[column].dropna()) - set(development[column].dropna())
            ),
        }

    numeric_ranges = {}
    for label, frame in [("development", development), ("validation", validation)]:
        numeric_ranges[label] = {
            column: {
                "min": _native(frame[column].min()),
                "median": _native(frame[column].median()),
                "max": _native(frame[column].max()),
            }
            for column in frame.select_dtypes(include="number").columns
        }

    return {
        "files": {
            "train-test.csv": frame_summary(development),
            "validation.csv": frame_summary(validation),
            "validation-predictions-template.csv": {
                "shape": list(template.shape),
                "columns": template.columns.tolist(),
                "missing": template.isna().sum().astype(int).to_dict(),
            },
            "december-chart-inputs.csv": frame_summary(december),
        },
        "schema": {
            "development_features_match_validation": development.drop(
                columns=TARGET_COLUMN
            ).columns.tolist()
            == validation.columns.tolist(),
            "template_ids_match_validation_in_order": bool(
                template[ID_COLUMN].equals(validation[ID_COLUMN])
            ),
        },
        "ids": id_checks,
        "categoricals": categorical_checks,
        "target_statistics": {key: _native(value) for key, value in target_stats.items()},
        "quality_flags": {
            "development_negative_weight": int((development["weight"] < 0).sum()),
            "validation_negative_weight": int((validation["weight"] < 0).sum()),
            "development_nonpositive_target": int((target <= 0).sum()),
            "development_bad_latitude": int(
                ((development["pickup_lat"].abs() > 90) | (development["delivery_lat"].abs() > 90)).sum()
            ),
            "development_bad_longitude": int(
                ((development["pickup_lon"].abs() > 180) | (development["delivery_lon"].abs() > 180)).sum()
            ),
            "exact_feature_matches_across_development_and_validation": len(exact_cross_matches),
        },
        "numeric_ranges": numeric_ranges,
    }


if __name__ == "__main__":
    print(json.dumps(build_audit(), indent=2, default=_native, sort_keys=True))
