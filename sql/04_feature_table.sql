-- Step 2c: the final model table, one row per TRANSFER / CASH_OUT transaction.
--
-- Only these two types ever contain fraud in PaySim, so they are the
-- population we score. History (03) was still built from ALL types.

CREATE OR REPLACE TABLE features AS
SELECT
    -- IDs and time: used for joining and the time-based split, NOT as model inputs.
    t.txn_id,
    t.step,
    t.txn_type,

    -- Transaction features (02)
    f.hour_of_day,
    f.is_transfer,
    f.amount,
    f.bal_orig_error,
    f.bal_dest_error,
    f.bal_orig_emptied,
    f.bal_dest_zero_before_after,

    -- Receiver history features (03)
    h.dest_txns_prev_24h,
    h.dest_amount_prev_24h,
    h.dest_txns_prev_all,
    h.dest_avg_amount_prev,
    h.dest_hours_since_prev,
    CASE WHEN h.dest_txns_prev_all = 0 THEN 1 ELSE 0 END AS dest_is_new,

    -- How big is this amount compared with what the receiver usually gets?
    -- NULLIF avoids dividing by zero; no history gives NULL.
    ROUND(t.amount / NULLIF(h.dest_avg_amount_prev, 0), 4) AS amount_vs_dest_avg,

    -- Label
    t.is_fraud

FROM transactions AS t
JOIN txn_features AS f ON f.txn_id = t.txn_id
JOIN dest_history AS h ON h.txn_id = t.txn_id
WHERE t.txn_type IN ('TRANSFER', 'CASH_OUT');
