"""Rules baseline: the kind of simple if-then rules a basic risk product starts with.

Flag a TRANSFER / CASH_OUT if
    amount >= big_amount
    or (receiver has no history and amount >= new_receiver_amount)

Both limits are picked on validation, with the same cost rule as the model,
so the baseline gets a fair chance.
"""
import numpy as np

from fraud.decision import decision_metrics

# candidate limits; inf means "this rule is switched off"
AMOUNT_STEPS = [
    50_000, 100_000, 200_000, 300_000, 500_000, 750_000,
    1_000_000, 1_500_000, 2_000_000, 3_000_000, 5_000_000, 10_000_000,
    float("inf"),
]


def rule_flags(df, big_amount, new_receiver_amount):
    big = df["amount"] >= big_amount
    new_and_large = (df["dest_is_new"] == 1) & (df["amount"] >= new_receiver_amount)
    return (big | new_and_large).astype(int).to_numpy()


def tune_rules(df, label, cost_missed_fraud, cost_false_alarm):
    # only call this on validation data
    best_cost, best_limits = None, None
    for big in AMOUNT_STEPS:
        for new_amount in AMOUNT_STEPS:
            flags = rule_flags(df, big, new_amount)
            cost = decision_metrics(df[label], flags, 1, cost_missed_fraud, cost_false_alarm)["total_cost"]
            if best_cost is None or cost < best_cost:
                best_cost, best_limits = cost, (big, new_amount)
    return best_limits


def money_lost(y_true, amounts, flags, review_fee):
    y_true = np.asarray(y_true)
    amounts = np.asarray(amounts, dtype=float)
    flags = np.asarray(flags).astype(bool)

    fraud_missed = float(amounts[(y_true == 1) & ~flags].sum())
    fraud_caught = float(amounts[(y_true == 1) & flags].sum())
    review_cost = float(flags.sum() * review_fee)
    return {
        "fraud_amount_caught": round(fraud_caught),
        "fraud_amount_missed": round(fraud_missed),
        "review_cost": round(review_cost),
        "money_lost": round(fraud_missed + review_cost),
    }
