"""Tests for step 2. Every expected value below was worked out by hand.

Receiver D1 gets 4 transactions:
    txn 1  step 1   CASH_IN   500   (history only: not a TRANSFER/CASH_OUT)
    txn 2  step 1   TRANSFER  200   fraud
    txn 3  step 10  CASH_OUT  300
    txn 4  step 30  TRANSFER  3000
Plus txn 5, a PAYMENT to merchant M1, which must not appear in `features`.
"""
import duckdb
import pytest

from fraud.build_features import build_features
from fraud.load_data import load_transactions

HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)
ROWS = [
    "1,CASH_IN,500.0,C1,100.0,600.0,D1,1000.0,500.0,0,0",
    "1,TRANSFER,200.0,C2,200.0,0.0,D1,0.0,0.0,1,0",
    "10,CASH_OUT,300.0,C3,1000.0,700.0,D1,200.0,500.0,0,0",
    "30,TRANSFER,3000.0,C4,3000.0,0.0,D1,500.0,3500.0,0,0",
    "5,PAYMENT,10.0,C5,50.0,40.0,M1,0.0,0.0,0,0",
]


@pytest.fixture
def features(tmp_path):
    """Load the fake CSV, build features, return them as {txn_id: row dict}."""
    csv = tmp_path / "sample.csv"
    csv.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    db = tmp_path / "test.duckdb"
    load_transactions(csv, db)
    build_features(db)
    with duckdb.connect(str(db)) as con:
        df = con.execute("SELECT * FROM features").df()
    return {int(row["txn_id"]): row for _, row in df.iterrows()}


def test_only_transfer_and_cash_out_rows(features):
    assert sorted(features) == [2, 3, 4]


def test_hour_and_type(features):
    assert features[2]["hour_of_day"] == 1
    assert features[4]["is_transfer"] == 1
    assert features[3]["is_transfer"] == 0


def test_balance_features(features):
    fraud = features[2]  # 200 sent, sender 200 -> 0, receiver 0 -> 0
    assert fraud["bal_orig_error"] == 0
    assert fraud["bal_dest_error"] == 200
    assert fraud["bal_orig_emptied"] == 1
    assert fraud["bal_dest_zero_before_after"] == 1

    normal = features[3]  # balances add up exactly
    assert normal["bal_orig_error"] == 0
    assert normal["bal_dest_error"] == 0
    assert normal["bal_orig_emptied"] == 0


def test_same_hour_is_not_counted(features):
    # txn 1 is in the same hour as txn 2, so it must NOT count as history.
    first = features[2]
    assert first["dest_txns_prev_24h"] == 0
    assert first["dest_txns_prev_all"] == 0
    assert first["dest_is_new"] == 1
    assert first["dest_avg_amount_prev"] != first["dest_avg_amount_prev"]  # NaN check


def test_history_includes_all_types(features):
    # At step 10, the last 24h (steps -14..9) holds txn 1 (CASH_IN) and txn 2.
    row = features[3]
    assert row["dest_txns_prev_24h"] == 2
    assert row["dest_amount_prev_24h"] == 700
    assert row["dest_txns_prev_all"] == 2
    assert row["dest_avg_amount_prev"] == 350
    assert row["dest_hours_since_prev"] == 9
    assert row["amount_vs_dest_avg"] == pytest.approx(0.8571)


def test_24h_window_drops_old_transactions(features):
    # At step 30, the last 24h is steps 6..29: only txn 3 (step 10) is inside.
    row = features[4]
    assert row["dest_txns_prev_24h"] == 1
    assert row["dest_amount_prev_24h"] == 300
    assert row["dest_txns_prev_all"] == 3
    assert row["dest_avg_amount_prev"] == pytest.approx(1000 / 3)
    assert row["dest_hours_since_prev"] == 20
    assert row["dest_is_new"] == 0


def test_rebuild_gives_same_row_count(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    db = tmp_path / "test.duckdb"
    load_transactions(csv, db)
    assert build_features(db) == 3
    assert build_features(db) == 3
