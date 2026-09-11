"""Leakage-safe temporal validation utilities."""

from __future__ import annotations

import pandas as pd

from data_utils import DATE_COLUMN, TARGET_COLUMN


# The cutoff is an observed calendar boundary: the final full month in development.
HOLDOUT_START = pd.Timestamp("2025-10-01")
BACKTEST_MONTHS = (
    pd.Timestamp("2025-07-01"),
    pd.Timestamp("2025-08-01"),
    pd.Timestamp("2025-09-01"),
)


def chronological_holdout(
    frame: pd.DataFrame,
    *,
    holdout_start: pd.Timestamp = HOLDOUT_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split development rows into earlier training and later holdout periods.

    The function rejects malformed ordering and missing targets rather than allowing
    a split that could accidentally mix future observations into training.
    """
    required = {DATE_COLUMN, TARGET_COLUMN}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if not pd.api.types.is_datetime64_any_dtype(frame[DATE_COLUMN]):
        raise TypeError(f"{DATE_COLUMN} must be parsed as datetime before splitting")
    if frame[DATE_COLUMN].isna().any() or frame[TARGET_COLUMN].isna().any():
        raise ValueError("Date and target must be complete before splitting")

    fit = frame.loc[frame[DATE_COLUMN] < holdout_start].copy()
    holdout = frame.loc[frame[DATE_COLUMN] >= holdout_start].copy()
    if fit.empty or holdout.empty:
        raise ValueError("Temporal split produced an empty partition")
    if fit[DATE_COLUMN].max() >= holdout[DATE_COLUMN].min():
        raise AssertionError("Temporal partitions overlap")
    return fit, holdout


def expanding_month_folds(
    frame: pd.DataFrame,
    *,
    validation_months: tuple[pd.Timestamp, ...] = BACKTEST_MONTHS,
):
    """Yield expanding training data and one full subsequent validation month."""
    if not pd.api.types.is_datetime64_any_dtype(frame[DATE_COLUMN]):
        raise TypeError(f"{DATE_COLUMN} must be parsed as datetime before splitting")
    for month_start in validation_months:
        month_end = month_start + pd.offsets.MonthBegin(1)
        fit = frame.loc[frame[DATE_COLUMN] < month_start].copy()
        validation = frame.loc[
            (frame[DATE_COLUMN] >= month_start) & (frame[DATE_COLUMN] < month_end)
        ].copy()
        if fit.empty or validation.empty:
            raise ValueError(f"Empty temporal fold for {month_start:%Y-%m}")
        if fit[DATE_COLUMN].max() >= validation[DATE_COLUMN].min():
            raise AssertionError("Temporal fold overlaps")
        yield month_start.strftime("%Y-%m"), fit, validation

