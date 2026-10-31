"""Step 2: build the `features` table by running the feature SQL files in order.

Needs step 1 first (the `transactions` table). Run from the project root:
    python -m fraud.build_features
"""
from pathlib import Path

import duckdb

from fraud.config import DB_PATH, SQL_DIR

FEATURE_SQL_FILES = [
    "02_transaction_features.sql",  # transaction-only features
    "03_receiver_history.sql",      # receiver history (past-only windows)
    "04_feature_table.sql",         # final table: TRANSFER + CASH_OUT
]


def build_features(db_path: Path) -> int:
    """Run each SQL file in order. Returns the number of rows in `features`."""
    with duckdb.connect(str(db_path)) as con:
        for name in FEATURE_SQL_FILES:
            print(f"Running {name} ...")
            con.execute((SQL_DIR / name).read_text())
        row_count = con.execute("SELECT COUNT(*) FROM features").fetchone()[0]
    return row_count


def print_summary(db_path: Path) -> None:
    """Sanity check: feature averages for fraud vs non-fraud rows.

    This is a quick look at the data, not a model result.
    """
    with duckdb.connect(str(db_path), read_only=True) as con:
        overall = con.execute(
            """
            SELECT COUNT(*) AS rows,
                   CAST(SUM(is_fraud) AS BIGINT) AS fraud_rows,
                   ROUND(AVG(is_fraud) * 100, 4) AS fraud_pct
            FROM features
            """
        ).df()

        by_label = con.execute(
            """
            SELECT
                is_fraud,
                ROUND(AVG(amount), 0)                     AS avg_amount,
                ROUND(AVG(bal_orig_emptied), 3)           AS orig_emptied_rate,
                ROUND(AVG(bal_dest_zero_before_after), 3) AS dest_zero_rate,
                ROUND(AVG(dest_txns_prev_24h), 2)         AS avg_dest_txns_24h,
                ROUND(AVG(dest_is_new), 3)                AS dest_new_rate,
                ROUND(MEDIAN(amount_vs_dest_avg), 2)      AS median_amount_vs_dest_avg
            FROM features
            GROUP BY is_fraud
            ORDER BY is_fraud
            """
        ).df()

    print("\n=== Feature table ===")
    print(overall.to_string(index=False))
    print("\n=== Averages by label (0 = normal, 1 = fraud) ===")
    print(by_label.T.to_string(header=False))


if __name__ == "__main__":
    n = build_features(DB_PATH)
    print(f"\nBuilt features table with {n:,} rows")
    print_summary(DB_PATH)
