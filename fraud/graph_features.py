"""Graph features from the account-to-account transaction graph, without leakage.

The graph: every account is a node, every transaction adds an edge sender -> receiver.

The catch: a normal "build the whole graph, then compute components" approach
uses FUTURE transactions. So we grow the graph hour by hour instead:

    for each hour (step), in time order:
        1. read features for this hour's transactions from the graph as it is NOW
           (built only from earlier hours)
        2. then add this hour's edges to the graph

To do this fast we use union-find (also called disjoint set union), which keeps
track of connected components as edges are added.
"""
from collections import deque

import pandas as pd


class UnionFind:
    """Tracks which accounts are connected, the size of each group,
    and how many confirmed frauds each group has seen."""

    def __init__(self, n):
        self.parent = list(range(n))  # parent[x] == x means x is a group's root
        self.size = [1] * n           # only correct at roots
        self.fraud = [0] * n          # known fraud count, only correct at roots

    def find(self, x):
        """Return the root of x's group."""
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        # Path compression: point everything on the way directly at the root,
        # so the next find is faster.
        while self.parent[x] != root:
            next_x = self.parent[x]
            self.parent[x] = root
            x = next_x
        return root

    def union(self, a, b):
        """Connect the groups of a and b."""
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        # Attach the smaller group under the bigger one (keeps trees shallow).
        if self.size[root_a] < self.size[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        self.size[root_a] += self.size[root_b]
        self.fraud[root_a] += self.fraud[root_b]


def compute_component_features(con, label_delay_hours):
    """Return a DataFrame with one row per transaction:
    txn_id, orig_component_size, dest_component_size, already_connected,
    dest_component_known_fraud.
    """
    # 1. Give every account a number 0..n-1 (union-find works with list indexes),
    #    and get all transactions in time order.
    data = con.execute(
        """
        WITH accounts AS (
            SELECT name_orig AS name FROM transactions
            UNION
            SELECT name_dest FROM transactions
        ),
        account_ids AS (
            SELECT name, CAST(row_number() OVER () - 1 AS INTEGER) AS id
            FROM accounts
        )
        SELECT t.txn_id, t.step, t.is_fraud, o.id AS orig_id, d.id AS dest_id
        FROM transactions AS t
        JOIN account_ids AS o ON o.name = t.name_orig
        JOIN account_ids AS d ON d.name = t.name_dest
        ORDER BY t.step, t.txn_id
        """
    ).fetchnumpy()

    # Plain Python lists are much faster than numpy arrays inside a for loop.
    txn_ids = data["txn_id"].tolist()
    steps = data["step"].tolist()
    is_fraud = data["is_fraud"].tolist()
    orig = data["orig_id"].tolist()
    dest = data["dest_id"].tolist()
    del data  # free the numpy copies; we only need the lists now
    n_txns = len(txn_ids)
    n_accounts = max(max(orig), max(dest)) + 1 if n_txns else 0

    uf = UnionFind(n_accounts)
    pending_fraud = deque()  # (step when the label becomes known, account id)

    orig_size = [0] * n_txns
    dest_size = [0] * n_txns
    connected = [0] * n_txns
    dest_fraud = [0] * n_txns

    i = 0
    while i < n_txns:
        step = steps[i]
        # Find where this hour's transactions end.
        j = i
        while j < n_txns and steps[j] == step:
            j += 1

        # a) Fraud labels that have become known by now get added to their group.
        while pending_fraud and pending_fraud[0][0] <= step:
            _, account = pending_fraud.popleft()
            uf.fraud[uf.find(account)] += 1

        # b) Read features from the graph built from EARLIER hours only.
        for k in range(i, j):
            root_o = uf.find(orig[k])
            root_d = uf.find(dest[k])
            orig_size[k] = uf.size[root_o]
            dest_size[k] = uf.size[root_d]
            connected[k] = 1 if root_o == root_d else 0
            dest_fraud[k] = uf.fraud[root_d]

        # c) Now add this hour's edges, and schedule fraud labels for later.
        for k in range(i, j):
            uf.union(orig[k], dest[k])
            if is_fraud[k] == 1:
                pending_fraud.append((step + label_delay_hours, dest[k]))

        i = j

    return pd.DataFrame(
        {
            "txn_id": txn_ids,
            "orig_component_size": orig_size,
            "dest_component_size": dest_size,
            "already_connected": connected,
            "dest_component_known_fraud": dest_fraud,
        }
    )
