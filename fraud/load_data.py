"""Step 1: load the PaySim CSV into DuckDB and print a quick sanity summary.

Run from the project root:
    python -m fraud.load_data
"""
from pathlib import Path

import duckdb

from fraud.config import DB_PATH, RAW_CSV, SQL_DIR


def load_transactions(csv_path: Path, db_path: Path) -> int:
    """Create the `transactions` table from the CSV. Returns the row count."""
    csv_path = Path(csv_path)
    db_path = Path(db_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"CSV not found at {csv_path}. Download PaySim from Kaggle and "
            "save it as data/raw/paysim.csv (see SETUP.md)."
        )

    db_path.parent.mkdir(parents=True, exist_ok=True)

    sql_template = (SQL_DIR / "01_load_transactions.sql").read_text()
    # DuckDB is happy with forward slashes on Windows too.
    sql = sql_template.format(csv_path=csv_path.as_posix())

    with duckdb.connect(str(db_path)) as con:
        con.execute(sql)
        row_count = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]

    return row_count


def print_summary(db_path: Path) -> None:
    """Print a few numbers to check the load looks right."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        overall = con.execute(
            """
            SELECT
                COUNT(*)              AS rows,
                CAST(SUM(is_fraud) AS BIGINT) AS fraud_rows,
                ROUND(AVG(is_fraud) * 100, 4) AS fraud_pct,
                MIN(step)             AS first_step,
                MAX(step)             AS last_step
            FROM transactions
            """
        ).df()

        by_type = con.execute(
            """
            SELECT
                txn_type,
                COUNT(*)      AS rows,
                CAST(SUM(is_fraud) AS BIGINT) AS fraud_rows
            FROM transactions
            GROUP BY txn_type
            ORDER BY rows DESC
            """
        ).df()

    print("\n=== Overall ===")
    print(overall.to_string(index=False))
    print("\n=== By transaction type ===")
    print(by_type.to_string(index=False))


if __name__ == "__main__":
    n = load_transactions(RAW_CSV, DB_PATH)
    print(f"Loaded {n:,} rows into {DB_PATH}")
    print_summary(DB_PATH)
