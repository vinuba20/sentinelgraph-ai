
"""Data-cleaning utilities for CIC-IDS2017."""

import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Remove whitespace around column names."""
    result = df.copy()
    result.columns = result.columns.astype(str).str.strip()
    return result


def replace_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace positive and negative infinity with missing values."""
    result = df.copy()
    numeric_columns = result.select_dtypes(include="number").columns

    result[numeric_columns] = result[numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )

    return result


def remove_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    return df.drop_duplicates().reset_index(drop=True)


def clean_dataframe(
    df: pd.DataFrame,
    remove_duplicates: bool = True,
) -> pd.DataFrame:
    """Apply basic cleaning without silently dropping rows with missing values."""
    result = clean_column_names(df)
    result = replace_infinite_values(result)

    if remove_duplicates:
        result = remove_duplicate_rows(result)

    return result


def cleaning_report(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> dict:
    """Summarize the effect of cleaning."""
    return {
        "rows_before": len(before),
        "rows_after": len(after),
        "duplicates_removed": len(before) - len(after),
        "missing_values_after": int(after.isna().sum().sum()),
        "infinite_values_after": int(
            np.isinf(
                after.select_dtypes(include="number").to_numpy(dtype=float)
            ).sum()
        ),
    }
