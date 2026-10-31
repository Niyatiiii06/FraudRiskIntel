import numpy as np


def decision_metrics(y_true, scores, threshold, cost_missed_fraud, cost_false_alarm):
    y_true = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold

    caught = int(np.sum(flagged & (y_true == 1)))
    missed = int(np.sum(~flagged & (y_true == 1)))
    false_alarms = int(np.sum(flagged & (y_true == 0)))
    alarms = caught + false_alarms

    return {
        "threshold": float(threshold),
        "alarms": alarms,
        "caught": caught,
        "missed": missed,
        "false_alarms": false_alarms,
        "precision": round(caught / alarms, 4) if alarms else 0.0,
        "recall": round(caught / (caught + missed), 4) if caught + missed else 0.0,
        "total_cost": missed * cost_missed_fraud + false_alarms * cost_false_alarm,
    }


def pick_threshold(y_true, scores, cost_missed_fraud, cost_false_alarm):
    # only call this on validation data
    candidates = np.round(np.arange(0.001, 1.0, 0.001), 3)
    costs = [
        decision_metrics(y_true, scores, t, cost_missed_fraud, cost_false_alarm)["total_cost"]
        for t in candidates
    ]
    return float(candidates[int(np.argmin(costs))])
