"""Reproducible model pipelines for freight-rate regression."""

from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FreightFeatureEngineer, categorical_features, numeric_features


RANDOM_SEED = 42


MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "ridge": {"alpha": 100.0},
    "random_forest": {
        "n_estimators": 160,
        "max_depth": 18,
        "min_samples_leaf": 8,
        "max_features": 0.8,
    },
    "hist_gradient_boosting": {
        "learning_rate": 0.05,
        "max_iter": 220,
        "max_leaf_nodes": 15,
        "min_samples_leaf": 100,
        "l2_regularization": 10.0,
    },
}


def build_pipeline(
    model_name: str,
    *,
    feature_set: str = "engineered",
    include_route: bool | None = None,
    overrides: dict[str, Any] | None = None,
) -> Pipeline:
    """Construct a model with all learned preprocessing inside the pipeline."""
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model: {model_name}")
    if include_route is None:
        include_route = model_name == "ridge"

    numeric = numeric_features(feature_set)
    categoricals = categorical_features(include_route=include_route)
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median", add_indicator=True))]
    )

    if model_name == "ridge":
        numeric_pipe.steps.append(("scale", StandardScaler()))
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=10,
                    ),
                ),
            ]
        )
        preprocessing = ColumnTransformer(
            [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categoricals)]
        )
        params = {**MODEL_CONFIGS[model_name], **(overrides or {})}
        estimator = Ridge(**params)
    else:
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                (
                    "encode",
                    OneHotEncoder(
                        handle_unknown="infrequent_if_exist",
                        min_frequency=10,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        preprocessing = ColumnTransformer(
            [("numeric", numeric_pipe, numeric), ("categorical", categorical_pipe, categoricals)],
            sparse_threshold=0.0,
        )
        params = {**MODEL_CONFIGS[model_name], **(overrides or {})}
        if model_name == "random_forest":
            estimator = RandomForestRegressor(
                **params, random_state=RANDOM_SEED, n_jobs=-1
            )
        else:
            estimator = HistGradientBoostingRegressor(
                **params, random_state=RANDOM_SEED
            )

    return Pipeline(
        [
            ("features", FreightFeatureEngineer(feature_set=feature_set)),
            ("preprocess", preprocessing),
            ("model", estimator),
        ]
    )


def fit_model(model_name: str, X, y, **pipeline_kwargs) -> Pipeline:
    pipeline = build_pipeline(model_name, **pipeline_kwargs)
    pipeline.fit(X, y)
    return pipeline


def build_final_pipeline() -> Pipeline:
    """Return the Phase 2 selection; it is not fitted until the caller supplies data."""
    return build_pipeline(
        "hist_gradient_boosting",
        feature_set="raw",
        include_route=False,
    )
