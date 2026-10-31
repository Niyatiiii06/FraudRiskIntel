-- Step 3c: the table step 4 trains on.
-- Step 2 features + receiver known fraud (05) + graph component features
-- (graph_components, written by fraud/build_graph_features.py).
-- Still one row per TRANSFER / CASH_OUT transaction.

CREATE OR REPLACE TABLE model_table AS
SELECT
    f.* EXCLUDE (is_fraud),        -- all step 2 columns except the label (DuckDB syntax)
    k.dest_known_fraud_count,
    g.orig_component_size,
    g.dest_component_size,
    g.already_connected,
    g.dest_component_known_fraud,
    f.is_fraud                     -- label last
FROM features AS f
JOIN receiver_known_fraud AS k ON k.txn_id = f.txn_id
JOIN graph_components     AS g ON g.txn_id = f.txn_id;
