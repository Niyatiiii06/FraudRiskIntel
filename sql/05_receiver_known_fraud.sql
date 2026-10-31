-- Step 3a: has this receiving account been involved in CONFIRMED fraud before?
-- {label_delay} is filled in by fraud/build_graph_features.py (LABEL_DELAY_HOURS).
--
-- A fraud label only becomes known {label_delay} hours after the transaction.
-- So the window stops at "{label_delay} PRECEDING": a transaction at step 30
-- can only see fraud labels from step 30 - {label_delay} or earlier.
-- Using is_fraud without this delay would leak the answer into the features.

CREATE OR REPLACE TABLE receiver_known_fraud AS
SELECT
    txn_id,
    CAST(COALESCE(
        SUM(is_fraud) OVER (
            PARTITION BY name_dest ORDER BY step
            RANGE BETWEEN UNBOUNDED PRECEDING AND {label_delay} PRECEDING
        ),
        0
    ) AS INTEGER) AS dest_known_fraud_count
FROM transactions;
