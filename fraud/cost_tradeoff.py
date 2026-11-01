"""What-if table: precision vs recall for different cost ratios.

For each ratio (missed fraud : false alarm) the threshold is picked on
validation, then scored on test. The saved model is not changed, and the
headline result stays at the 50:1 ratio set in config.py.

Run: python -m fraud.cost_tradeoff
"""
import joblib
import pandas as pd

from fraud.config import DB_PATH, MODEL_PATH, REPORTS_DIR
from fraud.decision import decision_metrics, pick_threshold
from fraud.modeling import LABEL, load_model_table, time_split

COST_RATIOS = [5, 10, 20, 50, 100]


def main():
    bundle = joblib.load(MODEL_PATH)
    model, features = bundle["model"], bundle["features"]

    df = load_model_table(DB_PATH)
    _, val, test, _, _ = time_split(df)
    val_scores = model.predict_proba(val[features])[:, 1]
    test_scores = model.predict_proba(test[features])[:, 1]

    rows = []
    for ratio in COST_RATIOS:
        threshold = pick_threshold(val[LABEL], val_scores, ratio, 1)
        m = decision_metrics(test[LABEL], test_scores, threshold, ratio, 1)
        rows.append({
            "cost_ratio": f"{ratio}:1",
            "threshold": threshold,
            "alarms": m["alarms"],
            "caught": m["caught"],
            "false_alarms": m["false_alarms"],
            "precision": m["precision"],
            "recall": m["recall"],
        })
    table = pd.DataFrame(rows)

    print("Thresholds picked on validation, scored on test:")
    print(table.to_string(index=False))

    REPORTS_DIR.mkdir(exist_ok=True)
    table.to_csv(REPORTS_DIR / "cost_tradeoff.csv", index=False)
    print("\nSaved reports/cost_tradeoff.csv")


if __name__ == "__main__":
    main()
