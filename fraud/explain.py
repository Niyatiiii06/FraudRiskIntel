"""SHAP explanations for the LightGBM model.

LightGBM computes SHAP values itself (pred_contrib=True), so we don't need
the separate shap package. Each value is in log-odds: positive pushes the
prediction towards fraud, negative pushes it away.
"""
import math

import numpy as np
import pandas as pd

DESCRIPTIONS = {
    "hour_of_day": "hour of the day",
    "is_transfer": "transaction is a TRANSFER",
    "amount": "amount",
    "dest_txns_prev_24h": "receiver's transactions in the last 24h",
    "dest_amount_prev_24h": "amount the receiver got in the last 24h",
    "dest_txns_prev_all": "receiver's earlier transactions",
    "dest_avg_amount_prev": "receiver's usual amount",
    "dest_hours_since_prev": "hours since the receiver's last transaction",
    "dest_is_new": "receiver has no history",
    "amount_vs_dest_avg": "amount vs receiver's usual amount",
    "dest_known_fraud_count": "confirmed frauds sent to this receiver before",
    "orig_component_size": "size of the sender's network",
    "dest_component_size": "size of the receiver's network",
    "already_connected": "sender and receiver already linked",
    "dest_component_known_fraud": "confirmed frauds in the receiver's network",
    "bal_orig_error": "sender balance mismatch",
    "bal_dest_error": "receiver balance mismatch",
    "bal_orig_emptied": "sender's account emptied",
    "bal_dest_zero_before_after": "receiver balance stayed at zero",
}


def shap_values(model, X):
    """One column per feature. The extra last column LightGBM returns (the base value) is dropped."""
    contrib = model.predict(X, pred_contrib=True)
    return pd.DataFrame(contrib[:, :-1], columns=X.columns, index=X.index)


def global_importance(model, X):
    """Average size of each feature's SHAP value: how much the model relies on it overall."""
    return shap_values(model, X).abs().mean().sort_values(ascending=False)


def format_value(feature, value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none (no history)"
    if feature in ("is_transfer", "dest_is_new", "already_connected",
                   "bal_orig_emptied", "bal_dest_zero_before_after"):
        return "yes" if value == 1 else "no"
    if feature == "amount_vs_dest_avg":
        return f"{value:.1f}x"
    if float(value).is_integer():
        return f"{int(value):,}"
    return f"{value:,.2f}"


def top_reasons(model, X_row, k=3, direction="up"):
    """The k features that pushed this one transaction most towards fraud
    (direction="up") or away from fraud (direction="down").

    X_row is a one-row DataFrame. Returns a list of dicts, biggest push first.
    """
    values = shap_values(model, X_row).iloc[0]
    if direction == "up":
        chosen = values[values > 0].sort_values(ascending=False).head(k)
    else:
        chosen = values[values < 0].sort_values().head(k)
    reasons = []
    for feature, impact in chosen.items():
        raw = X_row.iloc[0][feature]
        reasons.append({
            "feature": feature,
            "description": DESCRIPTIONS.get(feature, feature),
            "value": format_value(feature, None if pd.isna(raw) else raw),
            "impact": round(float(impact), 3),
        })
    return reasons


def reasons_as_text(reasons):
    if not reasons:
        return "none"
    return "; ".join(f"{r['description']} = {r['value']} ({r['impact']:+})" for r in reasons)
