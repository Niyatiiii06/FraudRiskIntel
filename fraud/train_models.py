"""Step 4: train 2 models x 3 feature sets, compare them on the VALIDATION set.

The test set is NOT used here. It is opened once, in step 5, after all
choices are made, so the final number stays honest.

Run from the project root:
    python -m fraud.train_models
"""
import time

import pandas as pd

from fraud.config import DB_PATH, PROJECT_ROOT
from fraud.modeling import (
    FEATURE_SETS,
    LABEL,
    MODEL_NAMES,
    evaluate,
    load_model_table,
    make_model,
    time_split,
)
from fraud.tracking import log_run, start_tracking

REPORT_PATH = PROJECT_ROOT / "reports" / "step4_validation.csv"


def describe_splits(train, val, test, train_end, val_end) -> pd.DataFrame:
    rows = []
    for name, part, steps in [
        ("train", train, f"1-{train_end}"),
        ("validation", val, f"{train_end + 1}-{val_end}"),
        ("test (locked)", test, f"{val_end + 1}-{int(test['step'].max())}"),
    ]:
        rows.append(
            {
                "split": name,
                "steps": steps,
                "rows": len(part),
                "fraud_rows": int(part[LABEL].sum()),
                "fraud_pct": round(part[LABEL].mean() * 100, 3),
            }
        )
    return pd.DataFrame(rows)


def main():
    print("Loading model_table ...")
    df = load_model_table(DB_PATH)
    train, val, test, train_end, val_end = time_split(df)

    print("\n=== Time-based split ===")
    print(describe_splits(train, val, test, train_end, val_end).to_string(index=False))

    start_tracking()
    results = []
    for set_name, features in FEATURE_SETS.items():
        for model_name in MODEL_NAMES:
            start = time.time()
            model = make_model(model_name)
            model.fit(train[features], train[LABEL])
            metrics = evaluate(model, val[features], val[LABEL])
            seconds = round(time.time() - start)
            print(f"  {model_name:20s} {set_name:20s} PR-AUC {metrics['pr_auc']:.4f}  ({seconds}s)")
            results.append(
                {"model": model_name, "features": set_name, "n_features": len(features), **metrics}
            )
            log_run(
                f"step4 {model_name} {set_name}",
                params={"step": 4, "model": model_name, "feature_set": set_name,
                        "n_features": len(features), "scored_on": "validation"},
                metrics={"val_pr_auc": metrics["pr_auc"], "val_roc_auc": metrics["roc_auc"]},
            )

    results = pd.DataFrame(results)
    print("\n=== Validation results (PR-AUC is the main metric) ===")
    print(results.to_string(index=False))
    print(f"\nPR-AUC of random guessing = validation fraud rate = {val[LABEL].mean():.4f}")

    REPORT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(REPORT_PATH, index=False)
    print(f"Saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
