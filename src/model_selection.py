"""Reproduce the limited Phase 2 tuning, ablation, and final evaluation."""

from __future__ import annotations

import argparse
import json

from data_utils import load_development
from evaluate import PROJECT_ROOT, backtest_model_spec, evaluate_spec
from validation import chronological_holdout


COMPACT_PARAMS = {
    "learning_rate": 0.05,
    "max_iter": 220,
    "max_leaf_nodes": 15,
    "min_samples_leaf": 100,
    "l2_regularization": 10.0,
}

TUNING_SPECS = [
    {
        "label": "HGB compact",
        "name": "hist_gradient_boosting",
        "feature_set": "engineered",
        "include_route": False,
        "overrides": COMPACT_PARAMS,
    },
    {
        "label": "HGB reference",
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
    {
        "label": "HGB regularized",
        "name": "hist_gradient_boosting",
        "feature_set": "engineered",
        "include_route": False,
        "overrides": {
            "learning_rate": 0.05,
            "max_iter": 300,
            "max_leaf_nodes": 31,
            "min_samples_leaf": 100,
            "l2_regularization": 10.0,
        },
    },
]

ABLATION_SPECS = [
    {
        "label": "HGB compact raw features",
        "name": "hist_gradient_boosting",
        "feature_set": "raw",
        "include_route": False,
        "overrides": COMPACT_PARAMS,
    },
    {
        "label": "HGB compact engineered",
        "name": "hist_gradient_boosting",
        "feature_set": "engineered",
        "include_route": False,
        "overrides": COMPACT_PARAMS,
    },
    {
        "label": "HGB compact without market/quote signals",
        "name": "hist_gradient_boosting",
        "feature_set": "no_market_quote",
        "include_route": False,
        "overrides": COMPACT_PARAMS,
    },
]

FINAL_SPEC = ABLATION_SPECS[0] | {"label": "Selected HGB compact raw features"}


def _write(name: str, payload: dict) -> None:
    path = PROJECT_ROOT / "scorer_results" / name
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["tuning", "ablation", "final", "all"], default="all", nargs="?"
    )
    args = parser.parse_args()
    development_fit, october = chronological_holdout(load_development())

    if args.stage in {"tuning", "all"}:
        results = [backtest_model_spec(development_fit, spec) for spec in TUNING_SPECS]
        _write(
            "phase2_hgb_tuning.json",
            {"selection_scope": "July-September only; October not evaluated", "results": results},
        )
    if args.stage in {"ablation", "all"}:
        results = [backtest_model_spec(development_fit, spec) for spec in ABLATION_SPECS]
        _write(
            "phase2_feature_ablation.json",
            {"selection_scope": "July-September only; October not evaluated", "results": results},
        )
    if args.stage in {"final", "all"}:
        result = evaluate_spec(development_fit, october, FINAL_SPEC)
        _write(
            "phase2_final_evaluation.json",
            {
                "selection_rule": "Selected on July-September backtests before October evaluation",
                "result": result,
            },
        )


if __name__ == "__main__":
    main()
