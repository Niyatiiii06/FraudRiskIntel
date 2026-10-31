from fraud.decision import decision_metrics, pick_threshold

# 2 frauds out of 6
Y = [1, 0, 1, 0, 0, 0]
S = [0.9, 0.8, 0.4, 0.3, 0.1, 0.05]


def test_metrics_at_half():
    # flags 0.9 (fraud) and 0.8 (normal), misses the 0.4 fraud
    m = decision_metrics(Y, S, 0.5, cost_missed_fraud=50, cost_false_alarm=1)
    assert (m["alarms"], m["caught"], m["missed"], m["false_alarms"]) == (2, 1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["total_cost"] == 51


def test_flag_nothing():
    m = decision_metrics(Y, [0] * 6, 1, cost_missed_fraud=50, cost_false_alarm=1)
    assert (m["alarms"], m["missed"], m["total_cost"]) == (0, 2, 100)
    assert m["precision"] == 0.0 and m["recall"] == 0.0


def test_costly_misses_lower_the_threshold():
    # catching both frauds costs one false alarm (0.8)
    t = pick_threshold(Y, S, cost_missed_fraud=50, cost_false_alarm=1)
    assert 0.3 < t <= 0.4
    assert decision_metrics(Y, S, t, 50, 1)["total_cost"] == 1


def test_costly_false_alarms_raise_the_threshold():
    t = pick_threshold(Y, S, cost_missed_fraud=1, cost_false_alarm=50)
    assert 0.8 < t <= 0.9
    m = decision_metrics(Y, S, t, 1, 50)
    assert (m["caught"], m["false_alarms"]) == (1, 0)
