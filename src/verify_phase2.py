"""Runtime reproducibility and leakage-boundary checks for the fixed Phase 2 model."""

from __future__ import annotations

import json

import numpy as np

from baseline import regression_metrics
from data_utils import ID_COLUMN, TARGET_COLUMN, load_development
from features import RAW_NUMERIC_FEATURES
from predict import predict_checked
from train import RANDOM_SEED, build_final_pipeline
from validation import HOLDOUT_START, chronological_holdout


def _xy(frame):
    X = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    assert TARGET_COLUMN not in X.columns
    assert ID_COLUMN not in X.columns
    return X, y


def run_verification() -> dict[str, object]:
    development = load_development()
    fit, october = chronological_holdout(development)
    X_fit, y_fit = _xy(fit)
    X_october, y_october = _xy(october)

    assert fit["date"].max() < HOLDOUT_START
    assert october["date"].min() >= HOLDOUT_START

    predictions = []
    models = []
    for _ in range(2):
        model = build_final_pipeline()
        model.fit(X_fit, y_fit)
        predictions.append(predict_checked(model, X_october))
        models.append(model)

    max_difference = float(np.max(np.abs(predictions[0] - predictions[1])))
    if max_difference != 0.0:
        raise AssertionError(f"Repeated predictions differ by {max_difference}")

    engineered_fit = models[0].named_steps["features"].transform(X_fit)
    imputer = models[0].named_steps["preprocess"].named_transformers_[
        "numeric"
    ].named_steps["impute"]
    expected_medians = engineered_fit[RAW_NUMERIC_FEATURES].median().to_numpy()
    if not np.allclose(imputer.statistics_, expected_medians, equal_nan=True):
        raise AssertionError("Numeric imputer statistics do not match the fit partition")

    encoder = models[0].named_steps["preprocess"].named_transformers_[
        "categorical"
    ].named_steps["encode"]
    categorical_columns = ["pickup", "delivery", "equipment"]
    for column, learned in zip(categorical_columns, encoder.categories_, strict=True):
        fit_values = set(engineered_fit[column].astype(str).unique())
        if not set(learned).issubset(fit_values):
            raise AssertionError(f"Encoder learned non-training values for {column}")

    train_predictions = predict_checked(models[0], X_fit)
    train_metrics = regression_metrics(y_fit, train_predictions)
    october_metrics = regression_metrics(y_october, predictions[0])
    fitted_estimator = models[0].named_steps["model"]
    result = {
        "random_seed": RANDOM_SEED,
        "configured_max_iterations": fitted_estimator.max_iter,
        "actual_iterations": fitted_estimator.n_iter_,
        "early_stopping": fitted_estimator.early_stopping,
        "fit_rows": len(fit),
        "october_rows": len(october),
        "repeated_fits": 2,
        "max_prediction_difference": max_difference,
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "october_mae": october_metrics["mae"],
        "october_rmse": october_metrics["rmse"],
        "mae_gap": october_metrics["mae"] - train_metrics["mae"],
        "rmse_gap": october_metrics["rmse"] - train_metrics["rmse"],
        "numeric_features": RAW_NUMERIC_FEATURES,
        "categorical_features": categorical_columns,
        "preprocessing_fit_partition_verified": True,
        "target_and_id_excluded": True,
    }
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_verification()
