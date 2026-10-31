-- Step 2b: history features for the RECEIVING account (name_dest).
--
-- Why the receiver? 99.85% of senders appear only once in PaySim, so
-- sender history is almost always empty. Receivers repeat (~2.3 txns each).
--
-- Rules that keep these features honest (no leakage):
--   * history uses ALL transaction types (a receiver's CASH_INs are still its history)
--   * only STRICTLY EARLIER hours count: "... AND 1 PRECEDING" on step.
--     Transactions in the same hour are excluded, because inside one step
--     we don't know their real order.
--
-- New syntax: the WINDOW clause at the bottom names a window once,
-- so we can reuse it with "OVER w_24h" instead of repeating it.

CREATE OR REPLACE TABLE dest_history AS
SELECT
    txn_id,

    -- Velocity: activity in the previous 24 hours.
    COUNT(*) OVER w_24h                 AS dest_txns_prev_24h,
    COALESCE(SUM(amount) OVER w_24h, 0) AS dest_amount_prev_24h,

    -- Full history before this hour.
    COUNT(*) OVER w_all                 AS dest_txns_prev_all,
    AVG(amount) OVER w_all              AS dest_avg_amount_prev,   -- NULL if no history
    step - MAX(step) OVER w_all         AS dest_hours_since_prev   -- NULL if no history

FROM transactions
WINDOW
    w_24h AS (PARTITION BY name_dest ORDER BY step
              RANGE BETWEEN 24 PRECEDING AND 1 PRECEDING),
    w_all AS (PARTITION BY name_dest ORDER BY step
              RANGE BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING);
