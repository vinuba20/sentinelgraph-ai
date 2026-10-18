
"""Verify the cleaned SentinelGraph AI dataset in chunks."""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flows_cleaned.csv"
CHUNK_SIZE = 100_000

REQUIRED_COLUMNS = [
    "Label",
    "label_binary",
    "Source IP",
    "Destination IP",
    "Timestamp",
]


def validate_prepared_dataset() -> None:
    if not DATA_FILE.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {DATA_FILE}")

    total_rows = 0
    binary_counts = Counter()
    label_counts = Counter()
    missing_labels = 0
    invalid_binary_labels = 0
    infinite_values = 0
    missing_graph_values = Counter()
    first_chunk = True

    for chunk in pd.read_csv(
        DATA_FILE,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        if first_chunk:
            missing_columns = [
                col for col in REQUIRED_COLUMNS
                if col not in chunk.columns
            ]

            print(f"File: {DATA_FILE}")
            print(f"Columns found: {len(chunk.columns)}")
            print(f"Required columns: {REQUIRED_COLUMNS}")

            if missing_columns:
                raise ValueError(
                    f"Missing required columns: {missing_columns}"
                )

            print("Required columns: PASS")
            first_chunk = False

        total_rows += len(chunk)

        labels = chunk["Label"].astype("string").str.strip()
        missing_labels += int(
            (labels.isna() | labels.eq("")).sum()
        )
        label_counts.update(labels.dropna().tolist())

        binary = pd.to_numeric(
            chunk["label_binary"], errors="coerce"
        )

        invalid_binary_labels += int(
            (~binary.isin([0, 1])).sum()
        )
        binary_counts.update(binary.dropna().astype(int).tolist())

        for column in ["Source IP", "Destination IP", "Timestamp"]:
            missing_graph_values[column] += int(
                chunk[column].isna().sum()
            )

        numeric = chunk.select_dtypes(include=[np.number])
        infinite_values += int(
            np.isinf(numeric.to_numpy(dtype=float)).sum()
        )

    print("\n" + "=" * 60)
    print("PREPARED DATASET VALIDATION")
    print("=" * 60)
    print(f"Total rows: {total_rows:,}")
    print(f"Binary label counts: {dict(binary_counts)}")
    print(f"Original label categories: {len(label_counts)}")
    print(f"Missing labels: {missing_labels:,}")
    print(f"Invalid binary labels: {invalid_binary_labels:,}")
    print(f"Infinite numeric values: {infinite_values:,}")
    print(f"Missing graph-related values: {dict(missing_graph_values)}")

    expected_rows = 2_830_743
    checks = {
        "Expected row count": total_rows == expected_rows,
        "No missing labels": missing_labels == 0,
        "Only binary labels 0 and 1": invalid_binary_labels == 0,
        "No infinite numeric values": infinite_values == 0,
    }

    print("\nChecks:")
    for name, passed in checks.items():
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")

    if not all(checks.values()):
        raise SystemExit(
            "\nValidation failed. Investigate before proceeding."
        )

    print("\nRESULT: Prepared dataset validation passed.")


if __name__ == "__main__":
    validate_prepared_dataset()
