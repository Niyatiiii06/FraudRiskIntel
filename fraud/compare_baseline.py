"""Compare the saved model with a tuned rules baseline on the test set.

The model is NOT retrained or re-tuned here (it was fixed in step 5).
The rules are tuned on validation, then both are scored on test.

Run: python -m fraud.compare_baseline
"""
import json
import math

import joblib
import pandas as pd

from fraud.config import (
    COST_FALSE_ALARM,
    COST_MISSED_FRAUD,
    DB_PATH,
    MODEL_PATH,
    REPORTS_DIR,
    REVIEW_FEE,
)
from fraud.decision import decision_metrics
from fraud.modeling import LABEL, load_model_table, time_split
from fraud.rules_baseline import money_lost, rule_flags, tune_rules
from fraud.tracking import log_run, start_tracking


def limit_text(value):
    return "off" if math.isinf(value) else f"{value:,.0f}"


def main():
    bundle = joblib.load(MODEL_PATH)
    model, features, threshold = bundle["model"], bundle["features"], bundle["threshold"]

    df = load_model_table(DB_PATH)
    _, val, test, _, _ = time_split(df)

    big, new_amount = tune_rules(val, LABEL, COST_MISSED_FRAUD, COST_FALSE_ALARM)
    print("Rules tuned on validation:")
    print(f"  flag if amount >= {limit_text(big)}")
    print(f"  or receiver is new and amount >= {limit_text(new_amount)}")

    approaches = {
        "flag nothing": [0] * len(test),
        "rules baseline": rule_flags(test, big, new_amount),
        "our model": (model.predict_proba(test[features])[:, 1] >= threshold).astype(int),
    }

    rows = []
    for name, flags in approaches.items():
        counts = decision_metrics(test[LABEL], flags, 1, COST_MISSED_FRAUD, COST_FALSE_ALARM)
        money = money_lost(test[LABEL], test["amount"], flags, REVIEW_FEE)
        rows.append({
            "approach": name,
            "alarms": counts["alarms"],
            "caught": counts["caught"],
            "missed": counts["missed"],
            "precision": counts["precision"],
            "recall": counts["recall"],
            "cost_points": counts["total_cost"],
            **money,
        })
    results = pd.DataFrame(rows)

    pd.set_option("display.width", 200)
    print(f"\nTest set ({len(test):,} rows, review fee {REVIEW_FEE} per alarm):")
    print(results.to_string(index=False))

    rules_lost = results.loc[results["approach"] == "rules baseline", "money_lost"].item()
    model_lost = results.loc[results["approach"] == "our model", "money_lost"].item()
    saved = rules_lost - model_lost
    change = "less" if saved >= 0 else "more"
    print(f"\nMoney saved vs rules baseline: {saved:,.0f} ({abs(saved) / rules_lost:.1%} {change} lost)")

    REPORTS_DIR.mkdir(exist_ok=True)
    out = {
        "rules": {"big_amount": None if math.isinf(big) else big,
                  "new_receiver_amount": None if math.isinf(new_amount) else new_amount},
        "review_fee": REVIEW_FEE,
        "results": results.to_dict(orient="records"),
        "money_saved_vs_rules": saved,
    }
    out_path = REPORTS_DIR / "baseline_comparison.json"
    out_path.write_text(json.dumps(out, indent=2))
    print("Saved reports/baseline_comparison.json")

    start_tracking()
    log_run(
        "rules baseline vs model",
        params={"rule_big_amount": limit_text(big), "rule_new_receiver_amount": limit_text(new_amount),
                "review_fee": REVIEW_FEE, "scored_on": "test"},
        metrics={"rules_money_lost": rules_lost, "model_money_lost": model_lost,
                 "money_saved_vs_rules": saved},
        files=[out_path],
    )
    print("Logged to MLflow")


if __name__ == "__main__":
    main()
