
"""Label utilities for CIC-IDS2017."""

import pandas as pd


def normalize_labels(labels: pd.Series) -> pd.Series:
    """Remove extra whitespace and normalize attack labels."""
    return labels.astype("string").str.strip().str.upper()


def to_binary_labels(labels: pd.Series) -> pd.Series:
    """Map BENIGN traffic to 0 and all attack traffic to 1."""
    normalized = normalize_labels(labels)

    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("Labels contain missing or empty values.")

    return (normalized != "BENIGN").astype("int8")
