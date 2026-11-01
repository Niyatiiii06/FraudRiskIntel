"""Score one transaction: build its features, predict, decide, explain.

The caller sends the transaction and the receiver's history numbers (the ones
sql/03_receiver_history.sql computes). The other features are worked out here
with the same formulas as the SQL, so training and serving can't drift apart.
"""
import math

import joblib
import pandas as pd

from fraud.explain import top_reasons


def load_bundle(path):
    return joblib.load(path)


def build_features(txn, feature_names):
    avg = txn.get("dest_avg_amount_prev")
    hours = txn.get("dest_hours_since_prev")
    prev_all = txn.get("dest_txns_prev_all", 0)

    row = {
        "hour_of_day": txn["step"] % 24,
        "is_transfer": 1 if txn["txn_type"] == "TRANSFER" else 0,
        "amount": float(txn["amount"]),
        "dest_txns_prev_24h": txn.get("dest_txns_prev_24h", 0),
        "dest_amount_prev_24h": float(txn.get("dest_amount_prev_24h", 0)),
        "dest_txns_prev_all": prev_all,
        "dest_avg_amount_prev": math.nan if avg is None else float(avg),
        "dest_hours_since_prev": math.nan if hours is None else float(hours),
        "dest_is_new": 1 if prev_all == 0 else 0,
        # same as ROUND(amount / NULLIF(avg, 0), 4) in sql/04_feature_table.sql
        "amount_vs_dest_avg": round(txn["amount"] / avg, 4) if avg else math.nan,
    }
    return pd.DataFrame([row])[feature_names]


def score_transaction(bundle, txn, n_reasons=3):
    model, features, threshold = bundle["model"], bundle["features"], bundle["threshold"]
    X = build_features(txn, features)
    probability = float(model.predict_proba(X)[0, 1])
    decision = "review" if probability >= threshold else "allow"
    return {
        "fraud_probability": round(probability, 4),
        "decision": decision,
        "threshold": threshold,
        "top_reasons": top_reasons(model, X, k=n_reasons, direction="up"),
    }
