"""Step 3: graph features, then the final `model_table`.

Needs steps 1 and 2 first. Run from the project root:
    python -m fraud.build_graph_features
"""
import time
from pathlib import Path

import duckdb

from fraud.config import DB_PATH, LABEL_DELAY_HOURS, SQL_DIR
from fraud.graph_features import compute_component_features


def build_graph_features(db_path: Path, label_delay_hours: int = LABEL_DELAY_HOURS) -> int:
    """Create receiver_known_fraud, graph_components and model_table.
    Returns the number of rows in model_table."""
    with duckdb.connect(str(db_path)) as con:
        print("Running 05_receiver_known_fraud.sql ...")
        sql = (SQL_DIR / "05_receiver_known_fraud.sql").read_text()
        con.execute(sql.format(label_delay=label_delay_hours))

        print("Growing the transaction graph hour by hour (takes a minute or two) ...")
        start = time.time()
        components = compute_component_features(con, label_delay_hours)
        print(f"  done in {time.time() - start:.0f}s")

        # Save the pandas DataFrame as a DuckDB table.
        con.register("components_df", components)
        con.execute("CREATE OR REPLACE TABLE graph_components AS SELECT * FROM components_df")
        con.unregister("components_df")

        print("Running 06_model_table.sql ...")
        con.execute((SQL_DIR / "06_model_table.sql").read_text())
        row_count = con.execute("SELECT COUNT(*) FROM model_table").fetchone()[0]
    return row_count


def print_summary(db_path: Path) -> None:
    """Sanity check: graph feature averages for fraud vs non-fraud rows (not a model result)."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        by_label = con.execute(
            """
            SELECT
                is_fraud,
                COUNT(*)                                    AS rows,
                ROUND(AVG(orig_component_size), 2)          AS avg_orig_component,
                ROUND(MEDIAN(dest_component_size), 0)       AS median_dest_component,
                ROUND(AVG(already_connected), 4)            AS already_connected_rate,
                -- (x > 0) is TRUE/FALSE; CAST turns it into 1/0 so AVG gives a rate
                ROUND(AVG(CAST(dest_known_fraud_count > 0 AS INTEGER)), 4)      AS dest_known_fraud_rate,
                ROUND(AVG(CAST(dest_component_known_fraud > 0 AS INTEGER)), 4)  AS component_known_fraud_rate
            FROM model_table
            GROUP BY is_fraud
            ORDER BY is_fraud
            """
        ).df()

    print("\n=== Graph features by label (0 = normal, 1 = fraud) ===")
    print(by_label.T.to_string(header=False))


if __name__ == "__main__":
    n = build_graph_features(DB_PATH)
    print(f"\nBuilt model_table with {n:,} rows")
    print_summary(DB_PATH)
