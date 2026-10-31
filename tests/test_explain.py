import numpy as np
import pytest

from fraud.explain import (
    DESCRIPTIONS,
    format_value,
    global_importance,
    reasons_as_text,
    shap_values,
    top_reasons,
)
from fraud.modeling import FEATURE_SETS, LABEL, make_model, time_split
from tests.test_modeling import fake_model_table

FEATURES = FEATURE_SETS["base+graph+balance"]


@pytest.fixture(scope="module")
def trained():
    df = fake_model_table()
    train, val, _, _, _ = time_split(df)
    model = make_model("lightgbm").fit(train[FEATURES], train[LABEL])
    return model, val


def test_shap_values_add_up_to_the_prediction(trained):
    # SHAP property: base value + sum of feature contributions = model's raw score
    model, val = trained
    X = val[FEATURES].head(50)
    full = model.predict(X, pred_contrib=True)
    raw = model.predict(X, raw_score=True)
    assert np.allclose(full.sum(axis=1), raw)
    assert shap_values(model, X).shape == (50, len(FEATURES))


def test_amount_matters_most(trained):
    # in the fake data fraud depends only on amount
    model, val = trained
    assert global_importance(model, val[FEATURES]).index[0] == "amount"


def test_top_reasons_are_sorted_and_signed(trained):
    model, val = trained
    row = val.loc[[val["amount"].idxmax()], FEATURES]
    up = top_reasons(model, row, k=3, direction="up")
    down = top_reasons(model, row, k=3, direction="down")
    assert up[0]["feature"] == "amount"
    assert [r["impact"] for r in up] == sorted([r["impact"] for r in up], reverse=True)
    assert all(r["impact"] > 0 for r in up)
    assert all(r["impact"] < 0 for r in down)


def test_every_feature_has_a_description():
    assert set(FEATURES) <= set(DESCRIPTIONS)


def test_format_value():
    assert format_value("dest_is_new", 1) == "yes"
    assert format_value("amount_vs_dest_avg", 8.04) == "8.0x"
    assert format_value("amount", 1500000.0) == "1,500,000"
    assert format_value("dest_avg_amount_prev", None) == "none (no history)"


def test_reasons_as_text():
    reasons = [{"description": "amount", "value": "10", "impact": 1.5}]
    assert reasons_as_text(reasons) == "amount = 10 (+1.5)"
    assert reasons_as_text([]) == "none"
