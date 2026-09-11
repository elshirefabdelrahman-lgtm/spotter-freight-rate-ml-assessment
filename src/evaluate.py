"""Common temporal evaluation for Phase 2 model comparison."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline import regression_metrics
from data_utils import ID_COLUMN, TARGET_COLUMN, load_development
from train import build_pipeline
from validation import chronological_holdout, expanding_month_folds


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    X = frame.drop(columns=[TARGET_COLUMN, ID_COLUMN])
    y = frame[TARGET_COLUMN].to_numpy(dtype=float)
    return X, y


def _score_fitted(model, fit: pd.DataFrame, validation: pd.DataFrame) -> dict[str, float]:
    X_fit, y_fit = _xy(fit)
    X_validation, y_validation = _xy(validation)
    train_metrics = regression_metrics(y_fit, model.predict(X_fit))
    validation_metrics = regression_metrics(y_validation, model.predict(X_validation))
    return {
        "train_mae": train_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "mae_gap": validation_metrics["mae"] - train_metrics["mae"],
        "rmse_gap": validation_metrics["rmse"] - train_metrics["rmse"],
    }


def backtest_model_spec(
    development_fit: pd.DataFrame, spec: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate one non-baseline specification without touching October."""
    folds = []
    start = time.perf_counter()
    for fold_name, fit, validation in expanding_month_folds(development_fit):
        model = build_pipeline(
            spec["name"],
            feature_set=spec.get("feature_set", "engineered"),
            include_route=spec.get("include_route"),
            overrides=spec.get("overrides"),
        )
        X_fit, y_fit = _xy(fit)
        model.fit(X_fit, y_fit)
        folds.append({"fold": fold_name, **_score_fitted(model, fit, validation)})
    fold_frame = pd.DataFrame(folds)
    return {
        "label": spec["label"],
        "model": spec["name"],
        "feature_set": spec.get("feature_set", "engineered"),
        "include_route": spec.get("include_route"),
        "overrides": spec.get("overrides", {}),
        "backtest_folds": folds,
        "backtest_mean_validation_mae": float(fold_frame["validation_mae"].mean()),
        "backtest_std_validation_mae": float(fold_frame["validation_mae"].std(ddof=0)),
        "backtest_mean_validation_rmse": float(fold_frame["validation_rmse"].mean()),
        "elapsed_seconds": time.perf_counter() - start,
    }


def evaluate_spec(
    development_fit: pd.DataFrame,
    october: pd.DataFrame,
    spec: dict[str, Any],
) -> dict[str, Any]:
    name = spec["name"]
    if name == "median_baseline":
        folds = []
        for fold_name, fit, validation in expanding_month_folds(development_fit):
            value = float(fit[TARGET_COLUMN].median())
            train_pred = np.full(len(fit), value)
            validation_pred = np.full(len(validation), value)
            train_metrics = regression_metrics(fit[TARGET_COLUMN].to_numpy(), train_pred)
            val_metrics = regression_metrics(validation[TARGET_COLUMN].to_numpy(), validation_pred)
            folds.append(
                {
                    "fold": fold_name,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "validation_mae": val_metrics["mae"],
                    "validation_rmse": val_metrics["rmse"],
                    "mae_gap": val_metrics["mae"] - train_metrics["mae"],
                    "rmse_gap": val_metrics["rmse"] - train_metrics["rmse"],
                }
            )
        value = float(development_fit[TARGET_COLUMN].median())
        october_metrics = {
            **regression_metrics(
                development_fit[TARGET_COLUMN].to_numpy(),
                np.full(len(development_fit), value),
            ),
            **{
                f"validation_{key}": val
                for key, val in regression_metrics(
                    october[TARGET_COLUMN].to_numpy(), np.full(len(october), value)
                ).items()
            },
        }
        october_scores = {
            "train_mae": october_metrics["mae"],
            "train_rmse": october_metrics["rmse"],
            "validation_mae": october_metrics["validation_mae"],
            "validation_rmse": october_metrics["validation_rmse"],
        }
        october_scores["mae_gap"] = october_scores["validation_mae"] - october_scores["train_mae"]
        october_scores["rmse_gap"] = october_scores["validation_rmse"] - october_scores["train_rmse"]
        elapsed = 0.0
    else:
        folds = []
        start = time.perf_counter()
        for fold_name, fit, validation in expanding_month_folds(development_fit):
            model = build_pipeline(
                name,
                feature_set=spec.get("feature_set", "engineered"),
                include_route=spec.get("include_route"),
                overrides=spec.get("overrides"),
            )
            X_fit, y_fit = _xy(fit)
            model.fit(X_fit, y_fit)
            folds.append({"fold": fold_name, **_score_fitted(model, fit, validation)})

        final_model = build_pipeline(
            name,
            feature_set=spec.get("feature_set", "engineered"),
            include_route=spec.get("include_route"),
            overrides=spec.get("overrides"),
        )
        X_fit, y_fit = _xy(development_fit)
        final_model.fit(X_fit, y_fit)
        october_scores = _score_fitted(final_model, development_fit, october)
        elapsed = time.perf_counter() - start

    fold_frame = pd.DataFrame(folds)
    return {
        "label": spec["label"],
        "model": name,
        "feature_set": spec.get("feature_set", "none"),
        "include_route": spec.get("include_route"),
        "overrides": spec.get("overrides", {}),
        "backtest_folds": folds,
        "backtest_mean_validation_mae": float(fold_frame["validation_mae"].mean()),
        "backtest_std_validation_mae": float(fold_frame["validation_mae"].std(ddof=0)),
        "backtest_mean_validation_rmse": float(fold_frame["validation_rmse"].mean()),
        "october": october_scores,
        "elapsed_seconds": elapsed,
    }


INITIAL_SPECS = [
    {"label": "Median baseline", "name": "median_baseline"},
    {"label": "Ridge engineered + route", "name": "ridge", "feature_set": "engineered", "include_route": True},
    {"label": "Random forest engineered", "name": "random_forest", "feature_set": "engineered", "include_route": False},
    {
        "label": "Histogram gradient boosting engineered reference",
        "name": "hist_gradient_boosting",
        "feature_set": "engineered",
        "include_route": False,
        "overrides": {
            "learning_rate": 0.05,
            "max_iter": 250,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 50,
            "l2_regularization": 5.0,
        },
    },
]


def run_evaluation(specs: list[dict[str, Any]]) -> dict[str, Any]:
    development = load_development()
    development_fit, october = chronological_holdout(development)
    results = []
    for spec in specs:
        print(f"Evaluating: {spec['label']}", flush=True)
        result = evaluate_spec(development_fit, october, spec)
        results.append(result)
        print(
            f"  backtest MAE={result['backtest_mean_validation_mae']:.6f}; "
            f"October MAE={result['october']['validation_mae']:.6f}",
            flush=True,
        )
    return {
        "selection_method": "July-September expanding monthly backtests; October final holdout",
        "development_fit_rows": len(development_fit),
        "october_rows": len(october),
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "scorer_results" / "phase2_initial_benchmark.json",
    )
    args = parser.parse_args()
    payload = run_evaluation(INITIAL_SPECS)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
