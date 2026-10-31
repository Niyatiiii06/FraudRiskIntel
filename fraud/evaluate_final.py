"""Final evaluation.

Class weights and the threshold are picked on validation, then the test set
is scored once. Run: python -m fraud.evaluate_final
"""
import json

import duckdb
import joblib
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraud.config import COST_FALSE_ALARM, COST_MISSED_FRAUD, DB_PATH, MODEL_PATH, REPORTS_DIR
from fraud.decision import decision_metrics, pick_threshold
from fraud.modeling import FEATURE_SETS, LABEL, load_model_table, make_model, time_split

# graph features didn't help on validation and bal_* is a simulator artifact
FINAL_FEATURE_SET = "base"


def train_lightgbm(train, features, use_class_weight):
    model = make_model("lightgbm")
    if use_class_weight:
        n_fraud = int(train[LABEL].sum())
        model.set_params(scale_pos_weight=(len(train) - n_fraud) / n_fraud)
    model.fit(train[features], train[LABEL])
    return model


def predict(model, df, features):
    return model.predict_proba(df[features])[:, 1]


def main():
    df = load_model_table(DB_PATH)
    train, val, test, _, _ = time_split(df)
    features = FEATURE_SETS[FINAL_FEATURE_SET]

    print("Class weights (validation PR-AUC):")
    models, val_pr = {}, {}
    for use_weight in [False, True]:
        models[use_weight] = train_lightgbm(train, features, use_weight)
        val_pr[use_weight] = average_precision_score(val[LABEL], predict(models[use_weight], val, features))
        print(f"  {'on ' if use_weight else 'off'}  {val_pr[use_weight]:.4f}")
    use_class_weight = val_pr[True] > val_pr[False]
    model = models[use_class_weight]

    threshold = pick_threshold(val[LABEL], predict(model, val, features), COST_MISSED_FRAUD, COST_FALSE_ALARM)
    print(f"\nThreshold (missed fraud = {COST_MISSED_FRAUD}, false alarm = {COST_FALSE_ALARM}): {threshold}")

    test_scores = predict(model, test, features)
    test_rate = test[LABEL].mean()
    test_pr = average_precision_score(test[LABEL], test_scores)
    test_roc = roc_auc_score(test[LABEL], test_scores)
    print(f"\nTest set: {len(test):,} rows, {int(test[LABEL].sum()):,} fraud ({test_rate:.2%})")
    print(f"  PR-AUC  {test_pr:.4f}  (random = {test_rate:.4f})")
    print(f"  ROC-AUC {test_roc:.4f}")

    with duckdb.connect(str(DB_PATH), read_only=True) as con:
        rule = con.execute("SELECT txn_id, is_flagged_fraud FROM transactions").df()
    rule_flags = test[["txn_id"]].merge(rule, on="txn_id", how="left")["is_flagged_fraud"]

    costs = (COST_MISSED_FRAUD, COST_FALSE_ALARM)
    decisions = pd.DataFrame([
        {"approach": "flag nothing", **decision_metrics(test[LABEL], [0] * len(test), 1, *costs)},
        {"approach": "PaySim rule", **decision_metrics(test[LABEL], rule_flags, 1, *costs)},
        {"approach": "our model", **decision_metrics(test[LABEL], test_scores, threshold, *costs)},
    ]).drop(columns="threshold")
    print("\nDecisions on test:")
    print(decisions.to_string(index=False))

    # same settings, only the feature set changes
    print("\nTest PR-AUC by feature set:")
    ablation = []
    for name, cols in FEATURE_SETS.items():
        m = train_lightgbm(train, cols, use_class_weight)
        pr = average_precision_score(test[LABEL], predict(m, test, cols))
        ablation.append({"features": name, "test_pr_auc": round(pr, 4)})
        print(f"  {name:20s} {pr:.4f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump({
        "model": model,
        "features": features,
        "threshold": threshold,
        "feature_set": FINAL_FEATURE_SET,
        "class_weight": use_class_weight,
    }, MODEL_PATH)

    REPORTS_DIR.mkdir(exist_ok=True)
    results = {
        "final_feature_set": FINAL_FEATURE_SET,
        "class_weight": bool(use_class_weight),
        "threshold": threshold,
        "costs": {"missed_fraud": COST_MISSED_FRAUD, "false_alarm": COST_FALSE_ALARM},
        "test_rows": len(test),
        "test_fraud_rate": round(float(test_rate), 5),
        "test_pr_auc": round(float(test_pr), 4),
        "test_roc_auc": round(float(test_roc), 4),
        "decisions": decisions.to_dict(orient="records"),
        "ablation": ablation,
    }
    (REPORTS_DIR / "step5_test_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {MODEL_PATH.name} and step5_test_results.json")


if __name__ == "__main__":
    main()
