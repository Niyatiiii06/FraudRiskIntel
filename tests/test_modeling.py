"""Tests for step 4: the time split, the feature lists, and that both models train."""
import numpy as np
import pandas as pd
import pytest

from fraud.modeling import (
    FEATURE_SETS,
    LABEL,
    MODEL_NAMES,
    evaluate,
    make_model,
    time_split,
)


def fake_model_table(n_steps=100, rows_per_step=20, seed=0):
    """Random features, with fraud more likely for big amounts."""
    rng = np.random.default_rng(seed)
    n = n_steps * rows_per_step
    all_features = FEATURE_SETS["base+graph+balance"]
    df = pd.DataFrame(rng.normal(size=(n, len(all_features))), columns=all_features)
    df["step"] = np.repeat(np.arange(1, n_steps + 1), rows_per_step)
    df[LABEL] = (df["amount"] + rng.normal(scale=0.5, size=n) > 1.8).astype(int)
    df.loc[::7, "dest_avg_amount_prev"] = np.nan  # some missing values, like real data
    return df


def test_split_is_by_time_and_complete():
    df = fake_model_table()
    train, val, test, train_end, val_end = time_split(df)
    assert train["step"].max() == train_end < val["step"].min()
    assert val["step"].max() == val_end < test["step"].min()
    assert len(train) + len(val) + len(test) == len(df)


def test_split_sizes_are_roughly_70_15_15():
    df = fake_model_table()
    train, val, test, _, _ = time_split(df)
    assert len(train) / len(df) == pytest.approx(0.70, abs=0.02)
    assert len(val) / len(df) == pytest.approx(0.15, abs=0.02)


def test_no_hour_is_split_across_sets():
    df = fake_model_table()
    train, val, test, _, _ = time_split(df)
    assert not set(train["step"]) & set(val["step"])
    assert not set(val["step"]) & set(test["step"])


def test_feature_sets_grow_and_have_no_duplicates():
    sizes = [len(f) for f in FEATURE_SETS.values()]
    assert sizes == sorted(sizes)
    for features in FEATURE_SETS.values():
        assert len(features) == len(set(features))
        assert LABEL not in features and "step" not in features


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_models_train_and_give_valid_metrics(model_name):
    df = fake_model_table()
    train, val, _, _, _ = time_split(df)
    features = FEATURE_SETS["base+graph+balance"]
    model = make_model(model_name)
    model.fit(train[features], train[LABEL])
    metrics = evaluate(model, val[features], val[LABEL])
    assert 0 <= metrics["pr_auc"] <= 1
    assert 0 <= metrics["roc_auc"] <= 1
    # The signal in "amount" is easy, so both models should beat random guessing.
    assert metrics["pr_auc"] > val[LABEL].mean()


def test_lightgbm_has_leaf_regularisation():
    # reg_lambda = 0 let leaf values explode on the balance features (step 4 debug).
    # (The explosion itself only showed up on the real PaySim data, so it is
    # checked with notebooks/01_debug_lightgbm_balance.py, not a unit test.)
    assert make_model("lightgbm").get_params()["reg_lambda"] > 0
