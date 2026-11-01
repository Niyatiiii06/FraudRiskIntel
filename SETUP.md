# Setup

Tested on Windows (PowerShell) with Python 3.12.
Python 3.15 does not work yet: MLflow's dependencies (`pyarrow`, `pyyaml`) have no Windows wheels for it.

## 1. Create the environment
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2. Get the data
1. Download the PaySim dataset from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1 (free account needed).
2. Unzip it. You get `PS_20174392719_1491204439457_log.csv` (~470 MB, ~6.3M rows).
3. Move and rename it to `data/raw/paysim.csv`:
```powershell
Move-Item "$HOME\Downloads\PS_20174392719_1491204439457_log.csv" data\raw\paysim.csv
```

## 3. Load into DuckDB
```powershell
python -m fraud.load_data
```
Creates `data/fraud.duckdb` with one table, `transactions`, and prints a summary
(row count, fraud rate, rows per transaction type).

## 4. Build features
```powershell
python -m fraud.build_features
```
Runs `sql/02_*.sql` to `sql/04_*.sql` in order and creates the `features` table
(one row per TRANSFER / CASH_OUT transaction). Prints feature averages for fraud vs normal rows.

| File | Features |
|---|---|
| `02_transaction_features.sql` | `hour_of_day`, `is_transfer`, `amount`, balance-mismatch features (`bal_*`) |
| `03_receiver_history.sql` | receiver's txn count and amount in the previous 24h, full-history count, average amount, hours since previous txn |
| `04_feature_table.sql` | joins everything, adds `dest_is_new` and `amount_vs_dest_avg`, keeps TRANSFER + CASH_OUT |

`bal_*` features are very strong in PaySim because of how the simulator records fraud;
step 4 will compare models with and without them.

## 5. Build graph features
```powershell
python -m fraud.build_graph_features
```
Creates `model_table` (step 2 features + graph features), the table models train on.
Takes about a minute and needs ~3 GB of free RAM (close other heavy apps if it fails).

| Source | Features |
|---|---|
| `sql/05_receiver_known_fraud.sql` | `dest_known_fraud_count`: confirmed fraud this receiver got before |
| `fraud/graph_features.py` | `orig_component_size`, `dest_component_size`, `already_connected`, `dest_component_known_fraud` |
| `sql/06_model_table.sql` | joins everything into `model_table` |

How leakage is avoided:
- The graph (accounts = nodes, transactions = edges) is grown hour by hour with union-find.
  Each transaction's features come from the graph built from earlier hours only.
- A fraud label counts only `LABEL_DELAY_HOURS` (24, in `fraud/config.py`) after the
  transaction, since banks confirm fraud later.

## 6. Train and compare models (validation only)
```powershell
pip install -r requirements.txt
python -m fraud.train_models
```
- Splits `model_table` by time: the hours holding the first ~70% of rows are train,
  the next ~15% validation, the last ~15% test. An hour is never split across sets.
- Trains logistic regression and LightGBM on 3 feature sets
  (`base`, `base+graph`, `base+graph+balance`, defined in `fraud/modeling.py`).
- Reports PR-AUC (main metric) and ROC-AUC on **validation**, saved to `reports/step4_validation.csv`.
- The test set is not touched here; it is used once in step 5.

## 7. Final evaluation (opens the test set once)
```powershell
python -m fraud.evaluate_final
```
- Final features: `base` (graph added nothing on validation; `bal_*` is a simulator artifact).
- Class weights on/off and the decision threshold are chosen on validation.
  The threshold minimises total cost, with costs set in `fraud/config.py`
  (`COST_MISSED_FRAUD = 50`, `COST_FALSE_ALARM = 1`).
- Then the test set is scored once and compared with "flag nothing" and PaySim's built-in rule.
- Saves `models/fraud_model.joblib` and `reports/step5_test_results.json`.

## 7b. Compare with a rules baseline
```powershell
python -m fraud.compare_baseline
```
- Rules baseline: flag if `amount >= A`, or if the receiver has no history and `amount >= B`.
  A and B are picked on validation with the same cost rule as the model.
- The saved model is not retrained or re-tuned; both are scored on the test set.
- Money view: a missed fraud loses its amount; every alarm costs `REVIEW_FEE` (100, in `fraud/config.py`).
- Saves `reports/baseline_comparison.json`.

## 7c. Precision vs recall for other cost ratios
```powershell
python -m fraud.cost_tradeoff
```
For cost ratios 5:1 to 100:1 (missed fraud : false alarm) the threshold is picked on validation,
then scored on test. The final model and its 50:1 headline result don't change.
Saves `reports/cost_tradeoff.csv`.

## 8. Explanations (SHAP)
```powershell
pip install -r requirements.txt
python -m fraud.explain_report
```
- Uses LightGBM's built-in SHAP values (`pred_contrib=True`), no extra `shap` package.
- Saves `reports/shap_global.png` (which features matter most) and `reports/shap_global.csv`.
- Prints plain-English reasons for three test transactions: a caught fraud,
  the most confident false alarm, and the fraud the model was most sure was normal.

## 9. Experiment tracking (MLflow)
`train_models`, `evaluate_final` and `compare_baseline` log every run automatically
(settings, scores, and the model/result files) to `mlflow.db` and `mlruns/` in the project folder.

To browse the runs:
```powershell
pip install -r requirements.txt
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Then open http://127.0.0.1:5000 and choose the `fraud-risk-intel` experiment. Press Ctrl+C to stop.

## 10. Run tests
```powershell
pytest
```
Tests use small fake CSVs with hand-checked expected values, so they don't need the real dataset.

## Project layout
```
fraud/        Python package (config, loading; later features, models, API)
sql/          SQL files, run in order (01_..., 02_...)
tests/        pytest tests
notebooks/    exploration and learning (e.g. window functions primer)
data/raw/     raw CSV (git-ignored)
```
