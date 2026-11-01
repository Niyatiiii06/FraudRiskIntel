"""Drift check: compare the model's input features in training vs later data.

Run: python -m fraud.drift_report
"""
import joblib

from fraud.config import DB_PATH, MODEL_PATH, REPORTS_DIR
from fraud.drift import drift_table
from fraud.modeling import load_model_table, time_split


def main():
    features = joblib.load(MODEL_PATH)["features"]
    df = load_model_table(DB_PATH)
    train, val, test, _, _ = time_split(df)

    REPORTS_DIR.mkdir(exist_ok=True)
    for name, new in [("validation", val), ("test", test)]:
        table = drift_table(train, new, features)
        print(f"\nPSI: train vs {name}  (< 0.1 stable, 0.1-0.25 moderate, > 0.25 big shift)")
        print(table.to_string(index=False))
        table.to_csv(REPORTS_DIR / f"drift_train_vs_{name}.csv", index=False)

    print("\nSaved reports/drift_train_vs_validation.csv and reports/drift_train_vs_test.csv")


if __name__ == "__main__":
    main()
