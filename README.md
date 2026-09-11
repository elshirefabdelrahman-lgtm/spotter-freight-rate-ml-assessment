# Freight Rate ML Assessment

This repository predicts `posted_rate` for freight loads. It contains a
reproducible, leakage-aware data audit, temporal model selection, final training
pipeline, local prediction files, December chart, and submission-review report.
Nothing has been submitted or published.

## Data

- `data/train-test.csv`: 48,000 labeled development rows dated 2025-01-01 through
  2025-10-31.
- `data/validation.csv`: 12,000 unlabeled rows dated 2025-11-01 through 2025-12-31.
- `data/validation-predictions-template.csv`: submission IDs and an empty
  `predicted_rate` column.
- `data/december-chart-inputs.csv`: 31 daily December scenarios with an empty
  `predicted_rate` column.

The assessment folder did not contain an upstream README or `requirements.txt`
when Phase 1 was performed, so these project foundations were added locally. The
official `score.py` was supplied later, preserved at the repository root, and run
successfully against the final output files.

## Features and roles

| Column | Role and observed type | Phase 1 treatment |
|---|---|---|
| `load_id` | Unique string identifier (`TR-######` or `TE-######`) | Excluded from prediction; sequential numbering is not assumed to carry business meaning. |
| `pickup`, `delivery` | City categoricals | Candidate route/geography features; unseen cities require leakage-safe handling. |
| `pickup_lat`, `pickup_lon`, `delivery_lat`, `delivery_lon` | Numeric coordinates | Candidate geographic features. Coordinates are valid by latitude/longitude bounds and are consistent per observed city. |
| `distance` | Positive numeric route distance | Candidate numeric feature. Units are not stated in the supplied files. |
| `equipment` | Three-level categorical (`Dry Van`, `Reefer`, `Flatbed`) | Candidate categorical feature. |
| `weight` | Numeric load weight | Candidate numeric feature, but missing and negative values require explicit preprocessing. Units are not stated. |
| `date` | ISO calendar date | Used to enforce chronological validation; future-safe calendar features may be derived later. |
| `market_index`, `quote_signal` | Numeric signals | Candidate features only if confirmed available at quote time; provenance cannot be verified from the files alone. |
| `posted_rate` | Continuous numeric target, development only | Never used to construct predictors. |

The December scenario file intentionally has a reduced schema: it omits IDs,
coordinates, `market_index`, and `quote_signal`. For the local scenario output,
coordinates are recovered from unique city mappings in labeled development data;
the absent signals remain missing and use medians learned by the final pipeline
from development data only.

## Validation philosophy

The labeled data ends on 2025-10-31 and the unlabeled prediction period begins the
next day. The primary holdout is therefore the complete final development month:

- fit: 43,147 rows dated 2025-01-01 through 2025-09-30;
- holdout: 4,853 rows dated 2025-10-01 through 2025-10-31.

This forward split is closer to the real forecasting task than a random split and
prevents later observations from informing earlier predictions. The cutoff is the
observed month boundary, not a tuned percentage. Model selection in a later phase
should use earlier rolling/expanding time folds within the fit period and reserve
October as a final internal check. `validation.csv` must not be used for fitting,
preprocessing, target encoding, feature selection, or hyperparameter selection.

## Leakage precautions and data-quality findings

- No feature column directly duplicates `posted_rate`; simple numeric correlations
  were inspected, but feature provenance is unavailable. In particular, whether
  `market_index` and `quote_signal` are known at prediction time must be confirmed.
- There are no exact duplicate rows, duplicate load IDs, or exact feature matches
  crossing development and validation. IDs are unique, sequential, correctly
  formatted, and disjoint.
- Development has 300 missing `weight` values and 374 missing `market_index`
  values. Validation has 165 and 249 respectively. Imputation must be fitted on
  each training fold only.
- Negative weights are invalid: 292 development rows and 145 validation rows.
  These should be treated as missing or otherwise handled inside the fitted
  preprocessing pipeline, not deleted using knowledge of validation outcomes.
- Validation contains eight pickup/delivery city labels unseen in development:
  Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk, and San Diego.
  Encoding must support unknown categories.
- No invalid dates, nonpositive distances, nonpositive targets, or out-of-range
  latitude/longitude values were observed. High target values exist; they were not
  automatically removed because the files provide no evidence that they are errors.

## Baseline

`src/baseline.py` fits one value—the median `posted_rate` of the earlier training
partition (2029.70)—and applies it to the October holdout. On the 4,853-row holdout:

- MAE: 1146.7937069853701
- RMSE: 1567.9704854490194

These values are a sanity-check baseline, not a claim about a final model. They are
calculated at runtime exclusively from the earlier labeled partition.

## Installation

From the project root, create a local environment and install the pinned dependency
set:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Then reproduce the Phase 1 checks with:

```powershell
.\.venv\Scripts\python.exe src\eda.py
.\.venv\Scripts\python.exe src\baseline.py
```

`notebooks/01_eda.ipynb` is a thin interactive entry point; reusable logic remains
in `src/`. Both scripts are read-only and do not generate predictions.

## Phase 2 model selection

Phase 2 uses July–September expanding monthly backtests for model/configuration
selection and uses October as the final internal evaluation period. The selected
candidate is a compact histogram gradient-boosting model using the minimal feature
set. Its measured October MAE is 121.89127460633962 and RMSE is
652.3450442547527. The initial broad benchmark had already computed October
diagnostics for each model family, so October was not a pristine once-only holdout;
the later tuning and feature decisions used July–September only. See
`PHASE2_REPORT.md` for the comparison, gaps, ablations, reproducibility test,
hyperparameters, and this limitation.

Reproduce the benchmark and selection stages from the project root:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py
.\.venv\Scripts\python.exe src\model_selection.py all
.\.venv\Scripts\python.exe src\verify_phase2.py
```

The second command reruns the deliberately small tuning and ablation study. None of
these commands reads `data/validation.csv` or writes submission predictions.

These are development-data results only. They are not the final Spotter evaluation,
do not estimate leaderboard performance, and do not guarantee submission results.

## Final training and predictions

The fixed Phase 2 `HistGradientBoostingRegressor` is trained on all 48,000 labeled
rows dated 2025-01-01 through 2025-10-31. Neither `validation.csv` nor the December
scenarios participate in fitting. Run:

```powershell
.\.venv\Scripts\python.exe src\predict.py --verify-reproducibility
```

This creates and programmatically validates:

- `validation_predictions.csv`: 12,000 template-aligned rows with columns
  `load_id,predicted_rate`;
- `december_predictions.csv`: the 31 original scenario rows with only
  `predicted_rate` populated;
- `scorer_results/phase3_prediction_manifest.json`: prediction statistics, hashes,
  training scope, fitted iterations, and reproducibility result.

Run the official supplied scorer with:

```powershell
.\.venv\Scripts\python.exe score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv --output-dir scorer_results
```

It validates both output CSVs and creates
`scorer_results/candidate_december.png`. It does not calculate Spotter's private
post-submission model-quality metric. Then build the PDF report with:

```powershell
.\.venv\Scripts\python.exe src\build_report.py
```

## Official scorer status

The official supplied `score.py` completed successfully: it validated 12,000 final
predictions, validated all 31 fixed December predictions, and created
`scorer_results/candidate_december.png`. Final validation metrics remain calculated
by Spotter after submission.

## Expected local review package

```text
validation_predictions.csv
december_predictions.csv
scorer_results/candidate_december.png
output/pdf/freight_rate_assessment_report.pdf
scorer_results/phase3_prediction_manifest.json
```

Important limitations are the undocumented quote-time provenance of
`market_index`/`quote_signal`, the earlier October exposure described in
`PHASE2_REPORT.md`, unseen validation cities, median signal fallback for December,
and the fact that the official scorer validates file structure and creates the
December chart but does not provide Spotter's private evaluation result.
