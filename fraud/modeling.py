"""Shared pieces for training: feature groups, the time-based split, models, metrics."""
from pathlib import Path

import duckdb
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# Feature groups. Kept separate so we can measure what each group adds.
# ---------------------------------------------------------------------
BASE_FEATURES = [
    "hour_of_day",
    "is_transfer",
    "amount",
    "dest_txns_prev_24h",
    "dest_amount_prev_24h",
    "dest_txns_prev_all",
    "dest_avg_amount_prev",
    "dest_hours_since_prev",
    "dest_is_new",
    "amount_vs_dest_avg",
]
GRAPH_FEATURES = [
    "dest_known_fraud_count",
    "orig_component_size",
    "dest_component_size",
    "already_connected",
    "dest_component_known_fraud",
]
# PaySim simulator artifact (see README): very strong, not realistic.
BALANCE_FEATURES = [
    "bal_orig_error",
    "bal_dest_error",
    "bal_orig_emptied",
    "bal_dest_zero_before_after",
]

FEATURE_SETS = {
    "base": BASE_FEATURES,
    "base+graph": BASE_FEATURES + GRAPH_FEATURES,
    "base+graph+balance": BASE_FEATURES + GRAPH_FEATURES + BALANCE_FEATURES,
}

LABEL = "is_fraud"


def load_model_table(db_path: Path) -> pd.DataFrame:
    """Read the step 3 model_table into pandas."""
    with duckdb.connect(str(db_path), read_only=True) as con:
        return con.execute("SELECT * FROM model_table").df()


def time_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """Split by time (step), never randomly.

    Train = earliest hours, then validation, then test.
    The cut points are the hours where ~70% and ~85% of the rows have passed.
    One hour is never split across two sets.

    Returns (train, val, test, train_end_step, val_end_step).
    """
    rows_per_step = df.groupby("step").size().sort_index()
    cumulative_share = rows_per_step.cumsum() / rows_per_step.sum()

    # First hour where the running share reaches the target.
    train_end = int(cumulative_share.index[(cumulative_share >= train_frac).argmax()])
    val_end = int(cumulative_share.index[(cumulative_share >= train_frac + val_frac).argmax()])

    train = df[df["step"] <= train_end]
    val = df[(df["step"] > train_end) & (df["step"] <= val_end)]
    test = df[df["step"] > val_end]
    return train, val, test, train_end, val_end


def make_model(name: str):
    """Create an untrained model by name."""
    if name == "logistic_regression":
        # Fill missing values (accounts with no history), scale, then fit.
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        )
    if name == "lightgbm":
        # LightGBM handles missing values itself, no scaling needed.
        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            # L2 penalty on leaf values. Without it (default 0), features that
            # separate fraud perfectly make leaf values explode to millions
            # and the model breaks (see notebooks/01_debug_lightgbm_balance.py).
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}")


MODEL_NAMES = ["logistic_regression", "lightgbm"]


def evaluate(model, X: pd.DataFrame, y: pd.Series) -> dict:
    """PR-AUC is the main metric (works well when fraud is rare).
    ROC-AUC is shown too, mainly to show how flattering it can look."""
    scores = model.predict_proba(X)[:, 1]
    return {
        "pr_auc": round(average_precision_score(y, scores), 4),
        "roc_auc": round(roc_auc_score(y, scores), 4),
    }
