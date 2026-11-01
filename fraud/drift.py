"""Population Stability Index (PSI): has a feature's distribution shifted?

PSI = sum over buckets of (new% - ref%) * ln(new% / ref%)

Common reading: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 big shift.
"""
import numpy as np
import pandas as pd

SMALL = 1e-4  # stands in for an empty bucket, so ln() never sees 0


def bucket_labels(reference, values, bins=10):
    """Put each value into a bucket defined by the REFERENCE data.

    - Features with few distinct values (like yes/no flags) get one bucket per value.
    - Other features get decile buckets from the reference quantiles.
    - Missing values get their own bucket.
    """
    ref = pd.Series(reference, dtype=float)
    vals = pd.Series(values, dtype=float)
    distinct = np.unique(ref.dropna())

    if len(distinct) <= bins:
        labels = vals.where(vals.isin(distinct), other=np.inf).astype(str)  # unseen values -> "inf"
    else:
        edges = np.unique(np.quantile(ref.dropna(), np.linspace(0, 1, bins + 1)))
        labels = pd.Series(np.digitize(vals, edges[1:-1]), index=vals.index).astype(str)

    return labels.where(vals.notna(), "missing")


def psi(reference, new, bins=10):
    ref_share = bucket_labels(reference, reference, bins).value_counts(normalize=True)
    new_share = bucket_labels(reference, new, bins).value_counts(normalize=True)

    buckets = ref_share.index.union(new_share.index)
    ref_share = ref_share.reindex(buckets, fill_value=0).clip(lower=SMALL)
    new_share = new_share.reindex(buckets, fill_value=0).clip(lower=SMALL)
    return float(((new_share - ref_share) * np.log(new_share / ref_share)).sum())


def psi_status(value):
    if value < 0.1:
        return "stable"
    if value < 0.25:
        return "moderate"
    return "big shift"


def drift_table(reference_df, new_df, features):
    rows = []
    for col in features:
        value = psi(reference_df[col], new_df[col])
        rows.append({"feature": col, "psi": round(value, 4), "status": psi_status(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False, ignore_index=True)
