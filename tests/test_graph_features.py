"""Tests for step 3. Expected values worked out by hand, with LABEL_DELAY_HOURS = 24.

txn  step  type      sender -> receiver   fraud
 1    1    TRANSFER  C1 -> D1               1    (label known from step 25)
 2    1    CASH_IN   C2 -> D2
 3    2    TRANSFER  C3 -> D1
 4    2    PAYMENT   D2 -> M1
 5   24    TRANSFER  C5 -> D1                    (1 hour before the label is known)
 6   25    CASH_OUT  C4 -> D1                    (label now known)
 7   25    TRANSFER  C2 -> D1
 8   31    TRANSFER  D2 -> D1                    (D2 and D1 got connected at step 25)

Groups before step 25: {C1, D1, C3, C5} and {C2, D2, M1}.
After step 25 everything (8 accounts) is one group.
"""
import duckdb
import pytest

from fraud.build_features import build_features
from fraud.build_features import print_summary as print_feature_summary
from fraud.build_graph_features import build_graph_features
from fraud.build_graph_features import print_summary as print_graph_summary
from fraud.graph_features import UnionFind
from fraud.load_data import load_transactions

HEADER = (
    "step,type,amount,nameOrig,oldbalanceOrg,newbalanceOrig,"
    "nameDest,oldbalanceDest,newbalanceDest,isFraud,isFlaggedFraud"
)
ROWS = [
    "1,TRANSFER,100.0,C1,100.0,0.0,D1,0.0,0.0,1,0",
    "1,CASH_IN,50.0,C2,0.0,50.0,D2,50.0,0.0,0,0",
    "2,TRANSFER,10.0,C3,10.0,0.0,D1,0.0,10.0,0,0",
    "2,PAYMENT,5.0,D2,5.0,0.0,M1,0.0,0.0,0,0",
    "24,TRANSFER,20.0,C5,20.0,0.0,D1,10.0,30.0,0,0",
    "25,CASH_OUT,30.0,C4,30.0,0.0,D1,30.0,60.0,0,0",
    "25,TRANSFER,40.0,C2,40.0,0.0,D1,60.0,100.0,0,0",
    "31,TRANSFER,60.0,D2,60.0,0.0,D1,100.0,160.0,0,0",
]

COLUMNS = [
    "orig_component_size",
    "dest_component_size",
    "already_connected",
    "dest_component_known_fraud",
    "dest_known_fraud_count",
]

# txn_id: (orig_size, dest_size, already_connected, component_fraud, dest_known_fraud)
EXPECTED = {
    1: (1, 1, 0, 0, 0),  # empty graph
    3: (1, 2, 0, 0, 0),  # D1 is in {C1, D1}
    5: (1, 3, 0, 0, 0),  # step 24: the step-1 fraud is NOT known yet
    6: (1, 4, 0, 1, 1),  # step 25: fraud now known; D1 in {C1, D1, C3, C5}
    7: (3, 4, 0, 1, 1),  # C2 in {C2, D2, M1}; not yet connected to D1
    8: (8, 8, 1, 1, 1),  # D2 and D1 already in the same group
}


@pytest.fixture
def model_rows(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    db = tmp_path / "test.duckdb"
    load_transactions(csv, db)
    build_features(db)
    build_graph_features(db, label_delay_hours=24)
    with duckdb.connect(str(db)) as con:
        df = con.execute("SELECT * FROM model_table").df()
    return {int(row["txn_id"]): row for _, row in df.iterrows()}


def test_model_table_keeps_only_transfer_and_cash_out(model_rows):
    assert sorted(model_rows) == [1, 3, 5, 6, 7, 8]


@pytest.mark.parametrize("txn_id", sorted(EXPECTED))
def test_graph_features_match_hand_calculation(model_rows, txn_id):
    row = model_rows[txn_id]
    actual = tuple(int(row[c]) for c in COLUMNS)
    assert actual == EXPECTED[txn_id]


def test_label_is_last_column(model_rows):
    assert list(next(iter(model_rows.values())).index)[-1] == "is_fraud"


def test_union_find_basics():
    uf = UnionFind(5)
    uf.union(0, 1)
    uf.union(3, 4)
    uf.union(1, 4)
    assert uf.find(0) == uf.find(3)
    assert uf.size[uf.find(0)] == 4
    assert uf.size[uf.find(2)] == 1
    uf.union(0, 4)  # already connected: nothing changes
    assert uf.size[uf.find(0)] == 4


def test_summaries_run_without_errors(tmp_path, capsys):
    # The summary SQL is only run by the scripts, so check it here too.
    csv = tmp_path / "sample.csv"
    csv.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    db = tmp_path / "test.duckdb"
    load_transactions(csv, db)
    build_features(db)
    build_graph_features(db, label_delay_hours=24)
    print_feature_summary(db)
    print_graph_summary(db)
    out = capsys.readouterr().out
    assert "Averages by label" in out
    assert "Graph features by label" in out
