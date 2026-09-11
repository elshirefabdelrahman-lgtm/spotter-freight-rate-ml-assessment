# Phase 2 model selection report

## Objective

Build a reproducible regression pipeline for `posted_rate`, compare a small set of
appropriate models without using the unlabeled submission data, and select a
candidate based primarily on forward temporal performance and generalization.

## Protocol

The target is `posted_rate`. The established final internal holdout contains all
4,853 October 2025 rows. Candidate choice, limited tuning, and feature selection
used three identical expanding-window folds within the earlier data: July, August,
and September, with all preceding months used for fitting. No code in the
evaluation or training pipeline loads `data/validation.csv`.

Audit limitation: the initial benchmark artifact computed October diagnostics for
all four broad model families before the later tuning and ablation steps. The
actual selection does not follow those October results: random forest had the best
initial October MAE, while histogram boosting was chosen for its better mean
July–September MAE and lower overfitting, and its compact/raw configuration was
chosen using July–September only. Nevertheless, October cannot truthfully be called
completely unseen from the start of Phase 2. The fixed final model was not changed
after its October score was obtained, and October targets were never used for
fitting or preprocessing.

Every learned imputer, scaler, and encoder is inside its model pipeline and is fit
only on the applicable training partition. The target and `load_id` are removed
before transformation.

## Features investigated

The minimal feature set contains pickup/delivery coordinates, distance, cleaned
weight, `market_index`, and `quote_signal`, plus pickup, delivery, and equipment
categoricals. Negative weights are converted to missing values before a median and
missingness-indicator transformer is fit on each training fold.

The expanded feature set tested a route categorical, calendar fields and cyclic
calendar encodings, absolute coordinate differences, Haversine distance, a route
detour ratio, log distance/weight, and distance interactions with the two signals.
All are row-local and target-free. Standalone population frequency features and
target encoding were rejected because they were unnecessary and would add leakage
or stability risk. The one-hot encoder's `min_frequency=10` grouping is learned
independently on each training fold; it does not use targets or held-out category
counts. `load_id` was rejected as an identifier.

## Initial model comparison

All values below were produced by `src/evaluate.py`. “Validation” is October.

| Model | Train MAE | Validation MAE | MAE gap | Train RMSE | Validation RMSE | RMSE gap |
|---|---:|---:|---:|---:|---:|---:|
| Median baseline | 1127.486 | 1146.794 | 19.308 | 1521.013 | 1567.970 | 46.958 |
| Ridge, engineered + route | 132.274 | 167.646 | 35.372 | 585.540 | 653.741 | 68.200 |
| Random forest, engineered | 90.040 | 148.883 | 58.844 | 521.885 | 660.831 | 138.945 |
| Histogram boosting, engineered reference | 113.911 | 154.854 | 40.943 | 557.267 | 655.206 | 97.939 |

The random forest had the largest gap and took about 123 seconds for the full
comparison versus about 18 seconds for histogram boosting. Ridge was unstable in
the July fold: its July validation MAE was 269.649 versus 136.576 and 149.563 in
August and September.

## Limited tuning and ablation

Only histogram boosting was tuned. Three settings were compared on July–September,
without evaluating October:

| Setting | Mean validation MAE | MAE standard deviation | Mean validation RMSE |
|---|---:|---:|---:|
| Compact: 220 iterations, 15 leaves, minimum leaf 100, L2 10 | 139.289 | 16.203 | 628.040 |
| Reference: 250 iterations, 31 leaves, minimum leaf 50, L2 5 | 149.190 | 19.539 | 633.952 |
| Regularized: 300 iterations, 31 leaves, minimum leaf 100, L2 10 | 148.407 | 19.916 | 630.616 |

Using the compact setting, the feature ablation was:

| Feature set | Mean validation MAE | MAE standard deviation | Mean validation RMSE |
|---|---:|---:|---:|
| Minimal raw features | 118.996 | 8.399 | 623.881 |
| Expanded engineered features | 139.289 | 16.203 | 628.040 |
| Expanded features without market/quote signals | 158.242 | 32.449 | 629.914 |

The expanded feature set was rejected because it worsened both MAE and stability.
Removing `market_index` and `quote_signal` worsened all three aggregate measures,
so they were retained for the validation-load model. Their availability at quote
time is still a business-provenance assumption that the supplied files cannot
verify.

## Final candidate

The selected model is `HistGradientBoostingRegressor` with the minimal feature set.
It uses learning rate 0.05, a maximum of 220 iterations, at most 15 leaves per tree,
minimum leaf size 100, L2 regularization 10, and random seed 42. Scikit-learn's
default `early_stopping="auto"` remains enabled and uses only a seeded subset of the
training partition; the final pre-October fit stopped after 128 iterations.
Categorical values are one-hot encoded with unknown-category
handling; missing numeric values are median-imputed with missingness indicators.

Final October results, after selection:

| Partition | MAE | RMSE |
|---|---:|---:|
| Training through September | 113.379 | 574.210 |
| October holdout | 121.891 | 652.345 |
| Holdout minus training | 8.512 | 78.135 |

The July–September validation MAEs for the final candidate were 112.157, 114.005,
and 130.825. The modest October MAE gap and consistent backtests do not show the
route-memorization behavior seen in the forest. RMSE remains much larger than MAE,
which indicates that a small number of large errors still matter; no target rows
were removed or winsorized to improve these scores.

The exact retained inputs are:

- numeric: `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon`, `distance`,
  cleaned `weight`, `market_index`, and `quote_signal`;
- categorical: `pickup`, `delivery`, and `equipment`.

`date`, route, `load_id`, calendar expansions, geographic ratios, log transforms,
and interaction features are not used by the final pipeline.

## Reproducibility and leakage verification

`src/verify_phase2.py` fit the final pipeline twice from scratch with seed 42. The
maximum absolute difference between the two 4,853-row October prediction vectors
was exactly 0.0. Both runs reproduced training MAE 113.37914167324652, training RMSE
574.2104679464239, October MAE 121.89127460633962, and October RMSE
652.3450442547527.

The runtime audit confirmed that the target and `load_id` are removed before model
input, imputer medians equal medians calculated from the pre-October fit partition,
and encoder category vocabularies are subsets of that same fit partition. All
learned preprocessing lives inside the pipeline. There are no standalone frequency
features, target encoders, or future-derived aggregates. The final raw feature path
does not construct or consume date features. Training and evaluation modules load
only `train-test.csv`; `validation.csv` and December data are referenced only by the
read-only Phase 1 EDA loader. No external/private data was introduced.

## Remaining risks

- The files do not document whether `market_index` and `quote_signal` are available
  at the real decision timestamp. The model currently treats provided values as
  prediction-time inputs because they are supplied for each validation load and
  removing them worsened July–September mean MAE from 139.289 to 158.242 in the
  tested expanded-feature ablation. This evidence supports predictive usefulness,
  not operational availability.
- October was exposed in the initial broad benchmark as described under Protocol.
  It was not used for the subsequent hyperparameter or feature decisions, but it is
  not a pristine once-only holdout.
- The December scenario file omits both signals and coordinates, so it cannot be
  passed directly to the final candidate. Phase 3 now handles this with exact
  development-only city coordinate mappings and training-fitted median imputation
  for the two absent signals. This produces no date-to-date variation because date
  is intentionally excluded from the selected model.
- Validation introduces eight unseen cities. The encoder will not fail, but unseen
  category indicators become inactive; coordinates still provide geographic input.
- Freight-rate outliers drive a sizable RMSE even when MAE is stable. They were kept
  because there is no evidence that they are data errors.
