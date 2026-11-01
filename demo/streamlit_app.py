"""Streamlit demo: score a transaction and see why it was flagged.

Run from the project root:  streamlit run demo/streamlit_app.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # so "fraud" can be imported from demo/

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from fraud.config import MODEL_PATH, REPORTS_DIR  # noqa: E402
from fraud.serving import load_bundle, score_transaction  # noqa: E402

st.set_page_config(page_title="FraudRiskIntel", page_icon="🛡️", layout="centered")


@st.cache_resource
def get_bundle():
    return load_bundle(MODEL_PATH)


st.title("FraudRiskIntel")
st.caption(
    "Fraud probability, a cost-based decision, and the reasons behind each flag. "
    "Trained on the PaySim synthetic mobile-money dataset."
)

tab_score, tab_results = st.tabs(["Score a transaction", "Results"])

with tab_score:
    left, right = st.columns(2)
    txn_type = left.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"])
    amount = right.number_input("Amount", min_value=0.0, value=850_000.0, step=10_000.0)
    hour = left.slider("Hour of day", 0, 23, 16)
    is_new = right.checkbox("Receiver is brand new (no history)", value=True)

    # only the hour matters to the model, so any day works
    txn = {"step": 24 * 10 + hour, "txn_type": txn_type, "amount": amount}

    if not is_new:
        st.markdown("**Receiver's history before this transaction**")
        a, b = st.columns(2)
        txn["dest_txns_prev_all"] = a.number_input("Earlier transactions", min_value=1, value=10)
        txn["dest_avg_amount_prev"] = b.number_input("Usual amount received", min_value=1.0, value=900_000.0)
        txn["dest_txns_prev_24h"] = a.number_input("Transactions in the last 24h", min_value=0, value=1)
        txn["dest_amount_prev_24h"] = b.number_input("Amount received in the last 24h", min_value=0.0, value=500_000.0)
        txn["dest_hours_since_prev"] = a.number_input("Hours since its last transaction", min_value=1, value=5)

    if st.button("Score", type="primary"):
        result = score_transaction(get_bundle(), txn)
        prob, threshold = result["fraud_probability"], result["threshold"]

        st.metric("Fraud probability", f"{prob:.1%}")
        if result["decision"] == "review":
            st.error(f"Decision: REVIEW (probability is at or above the {threshold:.1%} cut-off)")
        else:
            st.success(f"Decision: ALLOW (probability is below the {threshold:.1%} cut-off)")

        if result["top_reasons"]:
            st.markdown("**What pushed this towards fraud (SHAP)**")
            reasons = pd.DataFrame(result["top_reasons"])[["description", "value", "impact"]]
            reasons.columns = ["Clue", "Value", "Push towards fraud"]
            st.dataframe(reasons, hide_index=True, width="stretch")
        else:
            st.info("Nothing in this transaction pushed it towards fraud.")

        st.caption(
            "The cut-off comes from business costs: a missed fraud is treated as 50x as costly "
            "as a false alarm. Tip: move the hour to 6 am and score again."
        )

with tab_results:
    st.markdown(
        "Test set: the last ~15% of the month (408,077 transactions), used once. "
        "The rules baseline was tuned on earlier data."
    )
    comparison_file = REPORTS_DIR / "baseline_comparison.json"
    if comparison_file.exists():
        rows = json.loads(comparison_file.read_text())["results"]
        table = pd.DataFrame(rows)[["approach", "caught", "missed", "precision", "recall", "money_lost"]]
        st.dataframe(table, hide_index=True, width="stretch")
    chart = REPORTS_DIR / "shap_global.png"
    if chart.exists():
        st.image(str(chart), caption="Which clues the model relies on most (SHAP)")
