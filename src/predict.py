"""Fit the fixed Phase 2 model and create validated local Phase 3 outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import (
    DATA_DIR,
    ID_COLUMN,
    PROJECT_ROOT,
    TARGET_COLUMN,
    load_december_scenarios,
    load_development,
    load_prediction_template,
    load_unlabeled_validation,
)
from train import build_final_pipeline


VALIDATION_OUTPUT = PROJECT_ROOT / "validation_predictions.csv"
DECEMBER_OUTPUT = PROJECT_ROOT / "december_predictions.csv"
MANIFEST_OUTPUT = PROJECT_ROOT / "scorer_results" / "phase3_prediction_manifest.json"


def _model_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.drop(columns=[TARGET_COLUMN, ID_COLUMN], errors="ignore")


def predict_checked(model, features: pd.DataFrame) -> np.ndarray:
    predictions = np.asarray(model.predict(features), dtype=float)
    if predictions.shape != (len(features),):
        raise ValueError("Model returned an unexpected prediction shape")
    if not np.isfinite(predictions).all():
        raise ValueError("Model returned missing or non-finite predictions")
    if not (predictions > 0).all():
        raise ValueError("Model returned nonpositive predictions")
    return predictions


def _city_coordinates(development: pd.DataFrame) -> dict[str, tuple[float, float]]:
    pairs = pd.concat(
        [
            development[["pickup", "pickup_lat", "pickup_lon"]].rename(
                columns={"pickup": "city", "pickup_lat": "lat", "pickup_lon": "lon"}
            ),
            development[["delivery", "delivery_lat", "delivery_lon"]].rename(
                columns={"delivery": "city", "delivery_lat": "lat", "delivery_lon": "lon"}
            ),
        ],
        ignore_index=True,
    ).drop_duplicates()
    variants = pairs.groupby("city").size()
    ambiguous = variants[variants != 1]
    if not ambiguous.empty:
        raise ValueError(f"Cities have ambiguous coordinates: {ambiguous.to_dict()}")
    return {row.city: (float(row.lat), float(row.lon)) for row in pairs.itertuples(index=False)}


def prepare_december_features(
    december: pd.DataFrame, development: pd.DataFrame
) -> pd.DataFrame:
    """Supply absent model inputs using development-only mappings and missing values."""
    coords = _city_coordinates(development)
    required_cities = set(december["pickup"]) | set(december["delivery"])
    missing_cities = sorted(required_cities - set(coords))
    if missing_cities:
        raise ValueError(f"December cities absent from development: {missing_cities}")

    features = december.drop(columns="predicted_rate").copy()
    features["pickup_lat"] = features["pickup"].map(lambda city: coords[city][0])
    features["pickup_lon"] = features["pickup"].map(lambda city: coords[city][1])
    features["delivery_lat"] = features["delivery"].map(lambda city: coords[city][0])
    features["delivery_lon"] = features["delivery"].map(lambda city: coords[city][1])
    # The scenario file omits both signals. NaN invokes the same medians learned
    # solely from development data by the fitted final pipeline.
    features["market_index"] = np.nan
    features["quote_signal"] = np.nan
    return features


def _prediction_stats(values: np.ndarray) -> dict[str, float | int]:
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.size),
        "missing": int(series.isna().sum()),
        "nonfinite": int((~np.isfinite(series)).sum()),
        "minimum": float(series.min()),
        "maximum": float(series.max()),
        "mean": float(series.mean()),
        "median": float(series.median()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_validation_output(
    output: pd.DataFrame, validation: pd.DataFrame, template: pd.DataFrame
) -> None:
    if output.columns.tolist() != [ID_COLUMN, "predicted_rate"]:
        raise AssertionError("Validation output columns or order are incorrect")
    if len(output) != 12_000:
        raise AssertionError(f"Expected 12000 validation rows, found {len(output)}")
    if not output[ID_COLUMN].equals(template[ID_COLUMN]):
        raise AssertionError("Output IDs do not match the template in order")
    if not output[ID_COLUMN].equals(validation[ID_COLUMN]):
        raise AssertionError("Output IDs do not match validation.csv in order")
    if output[ID_COLUMN].duplicated().any():
        raise AssertionError("Duplicate validation IDs found")
    values = output["predicted_rate"].to_numpy(dtype=float)
    if output["predicted_rate"].isna().any() or not np.isfinite(values).all():
        raise AssertionError("Missing or non-finite validation predictions found")
    if not (values > 0).all():
        raise AssertionError("Nonpositive validation predictions found")


def validate_december_output(output: pd.DataFrame, original: pd.DataFrame) -> None:
    if output.columns.tolist() != original.columns.tolist():
        raise AssertionError("December columns or order changed")
    if len(output) != 31:
        raise AssertionError(f"Expected 31 December rows, found {len(output)}")
    expected_dates = pd.date_range("2025-12-01", "2025-12-31", freq="D")
    if not output["date"].reset_index(drop=True).equals(pd.Series(expected_dates)):
        raise AssertionError("December dates are incomplete or out of order")
    if not output.drop(columns="predicted_rate").reset_index(drop=True).equals(
        original.drop(columns="predicted_rate").reset_index(drop=True)
    ):
        raise AssertionError("Original December input values changed")
    values = output["predicted_rate"].to_numpy(dtype=float)
    if output["predicted_rate"].isna().any() or not np.isfinite(values).all():
        raise AssertionError("Missing or non-finite December predictions found")
    if not (values > 0).all():
        raise AssertionError("Nonpositive December predictions found")


def run_final_predictions(*, verify_reproducibility: bool = False) -> dict[str, object]:
    development = load_development()
    X_development = _model_inputs(development)
    y_development = development[TARGET_COLUMN].to_numpy(dtype=float)

    # Fit before loading prediction datasets. Only labeled development data can
    # influence model or preprocessing parameters.
    model = build_final_pipeline()
    model.fit(X_development, y_development)

    validation = load_unlabeled_validation()
    template = load_prediction_template()
    validation_predictions = predict_checked(model, _model_inputs(validation))
    validation_output = template.copy()
    validation_output["predicted_rate"] = validation_predictions
    validate_validation_output(validation_output, validation, template)

    december = load_december_scenarios()
    december_features = prepare_december_features(december, development)
    december_predictions = predict_checked(model, december_features)
    december_output = december.copy()
    december_output["predicted_rate"] = december_predictions
    validate_december_output(december_output, december)

    max_reproduction_difference = None
    if verify_reproducibility:
        repeated_model = build_final_pipeline()
        repeated_model.fit(X_development, y_development)
        repeated = predict_checked(repeated_model, _model_inputs(validation))
        max_reproduction_difference = float(np.max(np.abs(validation_predictions - repeated)))
        if max_reproduction_difference != 0.0:
            raise AssertionError(f"Repeated predictions differ: {max_reproduction_difference}")

    validation_output.to_csv(VALIDATION_OUTPUT, index=False)
    december_output.to_csv(DECEMBER_OUTPUT, index=False, date_format="%Y-%m-%d")

    validate_validation_output(
        pd.read_csv(VALIDATION_OUTPUT),
        load_unlabeled_validation(parse_dates=False),
        load_prediction_template(),
    )
    validate_december_output(
        pd.read_csv(DECEMBER_OUTPUT, parse_dates=["date"]),
        load_december_scenarios(),
    )

    estimator = model.named_steps["model"]
    manifest: dict[str, object] = {
        "training_file": str(DATA_DIR / "train-test.csv"),
        "training_rows": len(development),
        "training_date_min": development["date"].min().date().isoformat(),
        "training_date_max": development["date"].max().date().isoformat(),
        "model": type(estimator).__name__,
        "actual_iterations": int(estimator.n_iter_),
        "validation_output": str(VALIDATION_OUTPUT),
        "validation_statistics": _prediction_stats(validation_predictions),
        "december_output": str(DECEMBER_OUTPUT),
        "december_statistics": _prediction_stats(december_predictions),
        "december_fallback": {
            "coordinates": "exact city mappings learned from development data",
            "market_index": "missing; training-fitted median imputation",
            "quote_signal": "missing; training-fitted median imputation",
        },
        "reproducibility_max_absolute_difference": max_reproduction_difference,
        "validation_output_sha256": _sha256(VALIDATION_OUTPUT),
        "december_output_sha256": _sha256(DECEMBER_OUTPUT),
    }
    MANIFEST_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_OUTPUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-reproducibility", action="store_true")
    args = parser.parse_args()
    run_final_predictions(verify_reproducibility=args.verify_reproducibility)


if __name__ == "__main__":
    main()
