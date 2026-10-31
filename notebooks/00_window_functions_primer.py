"""Window functions primer: 6 fake transactions, one idea at a time.

Run from the project root:
    python notebooks/00_window_functions_primer.py

Change the queries and re-run. It uses an in-memory database, so nothing is saved.
"""
import duckdb

con = duckdb.connect()  # in-memory

con.execute(
    """
    CREATE TABLE toy AS
    SELECT * FROM (VALUES
        (1, 'A', 1,  100.0),
        (2, 'B', 2,  50.0),
        (3, 'A', 5,  120.0),
        (4, 'A', 20, 90.0),
        (5, 'B', 40, 60.0),
        (6, 'A', 30, 5000.0)
    ) AS t(txn_id, account, step, amount)
    """
)


def show(title, sql):
    print(f"\n=== {title} ===")
    print(con.execute(sql).df().to_string(index=False))


show("The toy table (step = hour)", "SELECT * FROM toy ORDER BY account, step")

# Idea 1: GROUP BY collapses rows. A window function keeps them.
show(
    "1a. GROUP BY: one row per account",
    """
    SELECT account, COUNT(*) AS n_txns
    FROM toy
    GROUP BY account
    """,
)

show(
    "1b. Window: every row kept, each gets its account's total",
    """
    SELECT txn_id, account, amount,
           COUNT(*) OVER (PARTITION BY account) AS n_txns_for_account
    FROM toy
    ORDER BY account, step
    """,
)

# Idea 2: add ORDER BY inside OVER and it becomes a running count.
show(
    "2. Running count and running total per account",
    """
    SELECT txn_id, account, step, amount,
           COUNT(*)    OVER (PARTITION BY account ORDER BY step) AS txns_so_far,
           SUM(amount) OVER (PARTITION BY account ORDER BY step) AS total_so_far
    FROM toy
    ORDER BY account, step
    """,
)

# Idea 3: LAG looks at the previous row in the same account.
show(
    "3. LAG: hours since the account's previous transaction",
    """
    SELECT txn_id, account, step,
           LAG(step) OVER (PARTITION BY account ORDER BY step)        AS prev_step,
           step - LAG(step) OVER (PARTITION BY account ORDER BY step) AS hours_since_prev
    FROM toy
    ORDER BY account, step
    """,
)

# Idea 4: a frame picks WHICH rows the window looks at.
# RANGE uses the value of step, so "24 PRECEDING" means "last 24 hours".
show(
    "4. Transactions by this account in the last 24 hours (including this one)",
    """
    SELECT txn_id, account, step,
           COUNT(*) OVER (
               PARTITION BY account ORDER BY step
               RANGE BETWEEN 24 PRECEDING AND CURRENT ROW
           ) AS txns_last_24h
    FROM toy
    ORDER BY account, step
    """,
)

# Idea 5: compare a transaction to the account's PAST only.
# ROWS ... AND 1 PRECEDING stops one row before the current one.
show(
    "5. Amount vs the account's previous average",
    """
    SELECT txn_id, account, step, amount,
           AVG(amount) OVER (
               PARTITION BY account ORDER BY step
               ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
           ) AS avg_before
    FROM toy
    ORDER BY account, step
    """,
)

# Idea 6: a CTE (WITH ... AS) names a step so the next one stays readable.
show(
    "6. CTE: amount divided by the previous average",
    """
    WITH history AS (
        SELECT txn_id, account, amount,
               AVG(amount) OVER (
                   PARTITION BY account ORDER BY step
                   ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
               ) AS avg_before
        FROM toy
    )
    SELECT txn_id, account, amount, avg_before,
           ROUND(amount / avg_before, 2) AS amount_ratio
    FROM history
    ORDER BY account, txn_id
    """,
)