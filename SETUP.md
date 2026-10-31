# Setup

Tested on Windows (PowerShell) with Python 3.12.

## 1. Create the environment
```powershell
python -m venv .venv
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

## 5. Run tests
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
