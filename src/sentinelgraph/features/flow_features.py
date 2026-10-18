
"""Prepare CIC-IDS2017 flow features for machine learning."""

import numpy as np
import pandas as pd


IDENTIFIER_COLUMNS = {
    "flow id",
    "source ip",
    "source port",
    "destination ip",
    "destination port",
    "timestamp",
    "label",
}

# These columns describe packet/flow statistics, not endpoint identity.
# Destination Port is excluded from this strict initial feature set.
# We can evaluate it separately later if needed.


def prepare_flow_features(
    df: pd.DataFrame,
    label_column: str = "Label",
) -> tuple[pd.DataFrame, pd.Series]:
    """Return numeric flow features and binary labels.

    This function does not impute missing values or remove rows.
    Those decisions are handled in the cleaning/model pipeline.
    """
    from sentinelgraph.data.labels import to_binary_labels

    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    if label_column not in data.columns:
        raise ValueError(f"Missing label column: {label_column}")

    labels = to_binary_labels(data[label_column])

    excluded = {
        column
        for column in data.columns
        if column.strip().lower() in IDENTIFIER_COLUMNS
    }
    excluded.add(label_column)

    features = data.drop(columns=list(excluded), errors="ignore")

    # Convert malformed numeric values to NaN; do not silently drop rows.
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce")

    features = features.replace([np.inf, -np.inf], np.nan)

    return features, labels
