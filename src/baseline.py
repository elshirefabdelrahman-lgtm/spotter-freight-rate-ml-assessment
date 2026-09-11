"""Training-partition-only median baseline for Phase 1."""

from __future__ import annotations

import math

import numpy as np

from data_utils import TARGET_COLUMN, load_development
from validation import HOLDOUT_START, chronological_holdout


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    errors = np.asarray(actual, dtype=float) - np.asarray(predicted, dtype=float)
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(math.sqrt(float(np.mean(np.square(errors))))),
    }


def run_baseline() -> dict[str, object]:
    development = load_development()
    fit, holdout = chronological_holdout(development)

    # The sole fitted quantity is calculated from the earlier partition only.
    fitted_median = float(fit[TARGET_COLUMN].median())
    predictions = np.full(len(holdout), fitted_median, dtype=float)
    metrics = regression_metrics(holdout[TARGET_COLUMN].to_numpy(), predictions)

    return {
        "strategy": "training-partition median",
        "holdout_start": HOLDOUT_START.date().isoformat(),
        "fit_rows": len(fit),
        "holdout_rows": len(holdout),
        "fitted_median": fitted_median,
        **metrics,
    }


if __name__ == "__main__":
    result = run_baseline()
    for key, value in result.items():
        print(f"{key}: {value}")

