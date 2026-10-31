"""Tests for step 1. They use a tiny fake CSV, so they run without the real dataset (and in CI)."""
import duckdb
import pytest

from fraud.load_data import load_transactions

HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)
ROWS = [
    "1,PAYMENT,9839.64,C1231006815,170136.0,160296.36,M1979787155,0.0,0.0,0,0",
    "1,TRANSFER,181.0,C1305486145,181.0,0.0,C553264065,0.0,0.0,1,0",
    "1,CASH_OUT,181.0,C840083671,181.0,0.0,C38997010,21182.0,0.0,1,0",
    "2,CASH_IN,500.0,C1111111111,100.0,600.0,C2222222222,1000.0,500.0,0,0",
]


@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "sample.csv"
    path.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    return path


def test_loads_all_rows(sample_csv, tmp_path):
    db = tmp_path / "test.duckdb"
    assert load_transactions(sample_csv, db) == len(ROWS)


def test_columns_are_renamed(sample_csv, tmp_path):
    db = tmp_path / "test.duckdb"
    load_transactions(sample_csv, db)
    with duckdb.connect(str(db)) as con:
        cols = [row[0] for row in con.execute("DESCRIBE transactions").fetchall()]
    assert cols == [
        "txn_id", "step", "txn_type", "amount",
        "name_orig", "old_balance_orig", "new_balance_orig",
        "name_dest", "old_balance_dest", "new_balance_dest",
        "is_fraud", "is_flagged_fraud",
    ]


def test_txn_id_is_unique_and_fraud_count_matches(sample_csv, tmp_path):
    db = tmp_path / "test.duckdb"
    load_transactions(sample_csv, db)
    with duckdb.connect(str(db)) as con:
        distinct_ids, fraud = con.execute(
            "SELECT COUNT(DISTINCT txn_id), SUM(is_fraud) FROM transactions"
        ).fetchone()
    assert distinct_ids == len(ROWS)
    assert fraud == 2


def test_reload_replaces_table(sample_csv, tmp_path):
    db = tmp_path / "test.duckdb"
    load_transactions(sample_csv, db)
    assert load_transactions(sample_csv, db) == len(ROWS)  # no duplicates


def test_missing_csv_gives_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="paysim.csv"):
        load_transactions(tmp_path / "nope.csv", tmp_path / "test.duckdb")
