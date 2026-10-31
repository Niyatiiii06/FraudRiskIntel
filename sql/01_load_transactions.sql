-- Step 1: load the raw PaySim CSV into one clean DuckDB table.
-- {csv_path} is filled in by fraud/load_data.py.
--
-- What this does:
--   * reads the CSV with fixed column types (no guessing)
--   * renames columns to snake_case (and fixes the "oldbalanceOrg" typo)
--   * adds txn_id, a stable row number we use as the transaction key
--
-- Feature engineering happens later, in separate SQL files.

CREATE OR REPLACE TABLE transactions AS
SELECT
    row_number() OVER ()  AS txn_id,
    step,                                  -- 1 step = 1 simulated hour (743 steps ~ 30 days)
    type                  AS txn_type,     -- CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    amount,
    nameOrig              AS name_orig,    -- sender account
    oldbalanceOrg         AS old_balance_orig,
    newbalanceOrig        AS new_balance_orig,
    nameDest              AS name_dest,    -- receiver account (names starting with M = merchants)
    oldbalanceDest        AS old_balance_dest,
    newbalanceDest        AS new_balance_dest,
    isFraud               AS is_fraud,     -- the label
    isFlaggedFraud        AS is_flagged_fraud  -- PaySim's simple rule (flags TRANSFER > 200,000)
FROM read_csv(
    '{csv_path}',
    header = true,
    columns = {{
        'step': 'INTEGER',
        'type': 'VARCHAR',
        'amount': 'DOUBLE',
        'nameOrig': 'VARCHAR',
        'oldbalanceOrg': 'DOUBLE',
        'newbalanceOrig': 'DOUBLE',
        'nameDest': 'VARCHAR',
        'oldbalanceDest': 'DOUBLE',
        'newbalanceDest': 'DOUBLE',
        'isFraud': 'INTEGER',
        'isFlaggedFraud': 'INTEGER'
    }}
);
