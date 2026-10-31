-- Step 2a: features that use ONLY the transaction itself (no history needed).
-- Computed for every transaction; the final table (04) keeps TRANSFER + CASH_OUT.

CREATE OR REPLACE TABLE txn_features AS
SELECT
    txn_id,

    -- Time: 1 step = 1 hour, so step % 24 is the hour of the day (0-23).
    step % 24 AS hour_of_day,

    CASE WHEN txn_type = 'TRANSFER' THEN 1 ELSE 0 END AS is_transfer,
    amount,

    -- Balance-mismatch features.
    -- In a "clean" transaction: old_balance - amount = new_balance (sender)
    --                           old_balance + amount = new_balance (receiver)
    -- The "error" is how far the recorded balances are from that.
    -- PaySim's fraud leaves very strong traces here, which is a simulator
    -- artifact. They are prefixed bal_ so step 4 can train with and without
    -- them (ablation) and show how much they inflate the scores.
    old_balance_orig - amount - new_balance_orig AS bal_orig_error,
    old_balance_dest + amount - new_balance_dest AS bal_dest_error,

    -- Sender's account went from having money to exactly zero.
    CASE WHEN old_balance_orig > 0 AND new_balance_orig = 0
         THEN 1 ELSE 0 END AS bal_orig_emptied,

    -- Receiver shows 0 before AND after, even though money was sent to it.
    CASE WHEN old_balance_dest = 0 AND new_balance_dest = 0
         THEN 1 ELSE 0 END AS bal_dest_zero_before_after

FROM transactions;
