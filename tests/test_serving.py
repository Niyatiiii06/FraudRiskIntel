import math

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraud.api import app, get_bundle
from fraud.build_features import build_features as build_sql_features
from fraud.load_data import load_transactions
from fraud.modeling import BASE_FEATURES, LABEL, make_model, time_split
from fraud.serving import build_features, score_transaction
from tests.test_build_features import HEADER, ROWS
from tests.test_modeling import fake_model_table

HISTORY_COLUMNS = [
    "dest_txns_prev_24h", "dest_amount_prev_24h", "dest_txns_prev_all",
    "dest_avg_amount_prev", "dest_hours_since_prev",
]


def as_input(row):
    """Turn a row of the SQL features table into what a caller would send the API."""
    txn = {"step": int(row["step"]), "txn_type": row["txn_type"], "amount": row["amount"]}
    for col in HISTORY_COLUMNS:
        txn[col] = None if pd.isna(row[col]) else row[col]
    return txn


def test_api_features_match_the_sql_pipeline(tmp_path):
    csv = tmp_path / "sample.csv"
    csv.write_text(HEADER + "\n" + "\n".join(ROWS) + "\n")
    db = tmp_path / "test.duckdb"
    load_transactions(csv, db)
    build_sql_features(db)
    with duckdb.connect(str(db)) as con:
        sql_rows = con.execute("SELECT * FROM features").df()

    for _, row in sql_rows.iterrows():
        served = build_features(as_input(row), BASE_FEATURES).iloc[0]
        for col in BASE_FEATURES:
            expected, actual = row[col], served[col]
            if pd.isna(expected):
                assert math.isnan(actual), col
            else:
                assert actual == pytest.approx(expected), col


@pytest.fixture(scope="module")
def bundle():
    df = fake_model_table()
    train, _, _, _, _ = time_split(df)
    model = make_model("lightgbm").fit(train[BASE_FEATURES], train[LABEL])
    return {"model": model, "features": BASE_FEATURES, "threshold": 0.5}


@pytest.fixture
def client(bundle):
    app.dependency_overrides[get_bundle] = lambda: bundle
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_score_transaction_decision_follows_threshold(bundle):
    txn = {"step": 5, "txn_type": "TRANSFER", "amount": 3.0}  # big amount = fraud in fake data
    result = score_transaction(bundle, txn)
    assert result["decision"] == ("review" if result["fraud_probability"] >= 0.5 else "allow")
    assert len(result["top_reasons"]) <= 3


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_endpoint(client):
    resp = client.post("/score", json={"step": 400, "txn_type": "TRANSFER", "amount": 3.0})
    assert resp.status_code == 200
    body = resp.json()
    assert 0 <= body["fraud_probability"] <= 1
    assert body["decision"] in ("review", "allow")
    assert body["threshold"] == 0.5
    assert all(r["impact"] > 0 for r in body["top_reasons"])


def test_score_rejects_bad_input(client):
    # PAYMENT is never fraud in PaySim, so the model isn't trained on it
    assert client.post("/score", json={"step": 1, "txn_type": "PAYMENT", "amount": 10}).status_code == 422
    assert client.post("/score", json={"step": 1, "txn_type": "TRANSFER", "amount": -5}).status_code == 422
