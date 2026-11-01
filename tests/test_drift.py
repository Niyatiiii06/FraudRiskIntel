import numpy as np
import pandas as pd
import pytest

from fraud.drift import bucket_labels, drift_table, psi, psi_status

rng = np.random.default_rng(0)


def test_same_distribution_is_stable():
    a = rng.normal(size=20_000)
    b = rng.normal(size=20_000)
    assert psi(a, b) < 0.01


def test_shifted_distribution_is_a_big_shift():
    a = rng.normal(size=20_000)
    b = rng.normal(loc=1.0, size=20_000)
    assert psi(a, b) > 0.25


def test_hand_calculated_flag_feature():
    # reference: 50% ones; new: 80% ones
    ref = [0] * 50 + [1] * 50
    new = [0] * 20 + [1] * 80
    expected = (0.2 - 0.5) * np.log(0.2 / 0.5) + (0.8 - 0.5) * np.log(0.8 / 0.5)
    assert psi(ref, new) == pytest.approx(expected)


def test_missing_values_get_their_own_bucket():
    labels = bucket_labels([0, 1, 1, 0], [0, None, 1])
    assert list(labels) == ["0.0", "missing", "1.0"]
    # more missing values in new data counts as drift
    assert psi([1.0] * 90 + [None] * 10, [1.0] * 50 + [None] * 50) > 0.25


def test_status_cutoffs():
    assert psi_status(0.05) == "stable"
    assert psi_status(0.15) == "moderate"
    assert psi_status(0.4) == "big shift"


def test_drift_table_sorted_biggest_first():
    ref = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(size=5000)})
    new = pd.DataFrame({"a": rng.normal(size=5000), "b": rng.normal(loc=2, size=5000)})
    table = drift_table(ref, new, ["a", "b"])
    assert list(table["feature"]) == ["b", "a"]
    assert table.loc[0, "status"] == "big shift"
