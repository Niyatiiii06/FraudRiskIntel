"""Explain the final model: global feature importance + three example transactions.

Needs step 5 first (models/fraud_model.joblib). Run: python -m fraud.explain_report
"""
import joblib
import matplotlib

matplotlib.use("Agg")  # save to file, no window
import matplotlib.pyplot as plt  # noqa: E402

from fraud.config import DB_PATH, MODEL_PATH, REPORTS_DIR  # noqa: E402
from fraud.explain import DESCRIPTIONS, global_importance, reasons_as_text, top_reasons  # noqa: E402
from fraud.modeling import LABEL, load_model_table, time_split  # noqa: E402


def save_importance_chart(importance, path):
    top = importance.head(10)[::-1]
    labels = [DESCRIPTIONS.get(f, f) for f in top.index]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(labels, top.values, color="#4C72B0")
    ax.set_xlabel("average |SHAP value| (log-odds)")
    ax.set_title("What the fraud model relies on most (test set)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_example(title, model, test, features, idx):
    row = test.loc[[idx], features]
    score = model.predict_proba(row)[0, 1]
    print(f"\n{title}")
    print(f"  txn_id {int(test.loc[idx, 'txn_id'])}, fraud probability {score:.3f}")
    print(f"  pushed towards fraud: {reasons_as_text(top_reasons(model, row, direction='up'))}")
    print(f"  pushed away:          {reasons_as_text(top_reasons(model, row, direction='down'))}")


def main():
    bundle = joblib.load(MODEL_PATH)
    model, features, threshold = bundle["model"], bundle["features"], bundle["threshold"]

    df = load_model_table(DB_PATH)
    _, _, test, _, _ = time_split(df)
    test = test.copy()
    test["score"] = model.predict_proba(test[features])[:, 1]

    # global importance on a fixed sample (all 400k rows works too, just slower)
    sample = test.sample(n=min(50_000, len(test)), random_state=0)
    importance = global_importance(model, sample[features])

    REPORTS_DIR.mkdir(exist_ok=True)
    chart_path = REPORTS_DIR / "shap_global.png"
    save_importance_chart(importance, chart_path)
    importance.rename("mean_abs_shap").to_csv(REPORTS_DIR / "shap_global.csv")

    print("Global importance (average |SHAP|):")
    for feature, value in importance.items():
        print(f"  {DESCRIPTIONS.get(feature, feature):45s} {value:.3f}")

    fraud = test[test[LABEL] == 1]
    normal = test[test[LABEL] == 0]
    caught = fraud[fraud["score"] >= threshold]
    missed = fraud[fraud["score"] < threshold]
    false_alarms = normal[normal["score"] >= threshold]

    print(f"\nExamples (threshold {threshold}):")
    if len(caught):
        show_example("1. Caught fraud (highest score)", model, test, features, caught["score"].idxmax())
    if len(false_alarms):
        show_example("2. False alarm (most confident mistake)", model, test, features,
                     false_alarms["score"].idxmax())
    if len(missed):
        show_example("3. Missed fraud (lowest score)", model, test, features, missed["score"].idxmin())

    print(f"\nSaved {chart_path}")


if __name__ == "__main__":
    main()
