"""Debug: why does LightGBM collapse on base+graph+balance (validation PR-AUC ~0.03)?

Run from the project root:
    python notebooks/01_debug_lightgbm_balance.py

Checks:
1. Overfitting: train PR-AUC vs validation PR-AUC.
2. Does validation PR-AUC start good and then fall as more trees are added?
3. Do the balance features look different in train vs validation?
4. Which features the model relies on most.
"""
import sys
from pathlib import Path

# Scripts in notebooks/ can't see the fraud package by default,
# so add the project root (one folder up) to Python's search path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402

from fraud.config import DB_PATH  # noqa: E402
from fraud.modeling import BALANCE_FEATURES, FEATURE_SETS, LABEL, load_model_table, make_model, time_split  # noqa: E402

features = FEATURE_SETS["base+graph+balance"]
df = load_model_table(DB_PATH)
train, val, _, _, _ = time_split(df)  # test stays locked

model = make_model("lightgbm")
model.fit(train[features], train[LABEL])

# 1. Overfitting check
train_scores = model.predict_proba(train[features])[:, 1]
val_scores = model.predict_proba(val[features])[:, 1]
print("\n=== 1. Train vs validation ===")
print(f"train PR-AUC {average_precision_score(train[LABEL], train_scores):.4f}")
print(f"val   PR-AUC {average_precision_score(val[LABEL], val_scores):.4f}")
print(f"distinct validation scores: {pd.Series(val_scores).round(6).nunique():,} of {len(val):,}")

# 2. Validation PR-AUC after the first k trees
print("\n=== 2. Validation scores as trees are added ===")
for k in [5, 10, 25, 50, 100, 200, 300]:
    raw = model.predict(val[features], raw_score=True, num_iteration=k)
    print(
        f"after {k:3d} trees: PR-AUC {average_precision_score(val[LABEL], raw):.4f}"
        f"  ROC-AUC {roc_auc_score(val[LABEL], raw):.4f}"
        f"  raw score range {raw.min():8.1f} to {raw.max():8.1f}"
    )

# 3. Balance features: train vs validation, by label
print("\n=== 3. Balance features: mean by split and label ===")
parts = []
for name, part in [("train", train), ("val", val)]:
    means = part.groupby(LABEL)[BALANCE_FEATURES].mean()
    means.index = [f"{name}_label{i}" for i in means.index]
    parts.append(means)
print(pd.concat(parts).T.round(3).to_string())

# 4. Feature importance (total gain)
print("\n=== 4. Top features by gain ===")
importance = pd.Series(
    model.booster_.feature_importance(importance_type="gain"), index=features
).sort_values(ascending=False)
print((importance / importance.sum()).round(3).head(10).to_string())
