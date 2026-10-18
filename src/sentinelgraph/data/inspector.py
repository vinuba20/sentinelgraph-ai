
"""Inspect CIC-IDS2017 CSV files in memory-efficient chunks."""

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from sentinelgraph.data.loader import list_csv_files


def inspect_dataset(
    data_dir: Path | None = None,
    chunk_size: int = 100_000,
) -> pd.DataFrame:
    """Count rows, labels, missing values, and infinities per CSV."""
    files = (
        list_csv_files(data_dir)
        if data_dir is not None
        else list_csv_files()
    )

    reports = []

    for file_path in files:
        print(f"Inspecting {file_path.name} ...")

        row_count = 0
        label_counts = Counter()
        missing_values = 0
        infinite_values = 0
        columns = None

        for chunk in pd.read_csv(
            file_path,
            chunksize=chunk_size,
            low_memory=False,
            encoding="utf-8-sig",
        ):
            chunk.columns = chunk.columns.astype(str).str.strip()
            columns = chunk.columns.tolist()
            row_count += len(chunk)

            if "Label" in chunk.columns:
                labels = chunk["Label"].astype("string").str.strip()
                label_counts.update(labels.fillna("<MISSING>").tolist())

            missing_values += int(chunk.isna().sum().sum())

            numeric = chunk.select_dtypes(include="number")
            if not numeric.empty:
                infinite_values += int(
                    np.isinf(numeric.to_numpy(dtype=float)).sum()
                )

        reports.append(
            {
                "file": file_path.name,
                "rows": row_count,
                "columns": len(columns) if columns else 0,
                "missing_values": missing_values,
                "infinite_values": infinite_values,
                "label_counts": dict(label_counts),
            }
        )

    return pd.DataFrame(reports)


if __name__ == "__main__":
    report = inspect_dataset()

    print("\n=== DATASET INSPECTION REPORT ===")
    print(report[["file", "rows", "columns", "missing_values", "infinite_values"]].to_string(index=False))

    print("\n=== LABEL DISTRIBUTION ===")
    for _, row in report.iterrows():
        print(f"\n{row['file']}")
        for label, count in sorted(row["label_counts"].items()):
            print(f"  {label}: {count:,}")

    print("\nInspection complete.")
