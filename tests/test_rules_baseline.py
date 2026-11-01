import pandas as pd

from fraud.rules_baseline import money_lost, rule_flags, tune_rules

# 5 transactions: frauds are the big one and the new-receiver one
DF = pd.DataFrame({
    "amount":      [6_000_000, 400_000, 400_000, 80_000, 20_000],
    "dest_is_new": [0,         1,       0,       1,      0],
    "is_fraud":    [1,         1,       0,       0,      0],
})


def test_rule_flags():
    flags = rule_flags(DF, big_amount=5_000_000, new_receiver_amount=300_000)
    assert list(flags) == [1, 1, 0, 0, 0]


def test_switched_off_rules_flag_nothing():
    flags = rule_flags(DF, big_amount=float("inf"), new_receiver_amount=float("inf"))
    assert flags.sum() == 0


def test_tuning_finds_rules_that_catch_both_frauds_without_false_alarms():
    big, new_amount = tune_rules(DF, "is_fraud", cost_missed_fraud=50, cost_false_alarm=1)
    assert list(rule_flags(DF, big, new_amount)) == [1, 1, 0, 0, 0]


def test_money_lost():
    # flag only the first txn: miss the 400k fraud, pay one review
    m = money_lost(DF["is_fraud"], DF["amount"], [1, 0, 0, 0, 0], review_fee=100)
    assert m == {
        "fraud_amount_caught": 6_000_000,
        "fraud_amount_missed": 400_000,
        "review_cost": 100,
        "money_lost": 400_100,
    }
