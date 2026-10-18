
"""Inspect CIC-IDS2017 CSV files in chunks and save corrected reports."""

from collections import Counter
from pathlib import Path

import pandas as pd

from sentinelgraph.data.config import load_dataset_config
from sentinelgraph.data.loader import list_csv_files


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"
CHUNK_SIZE = 100_000


def inspect_dataset() -> None:
    config = load_dataset_config()
    files = list_csv_files(config["raw_dir"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    file_summaries = []
    global_labels = Counter()
    global_missing = Counter()
    total_rows = 0

    for file_path in files:
        print(f"\nInspecting: {file_path.name}")

        row_count = 0
        label_counts = Counter()
        missing_counts = Counter()
        column_names = []

        encoding = (
            "cp1252"
            if "WebAttacks" in file_path.name
            else "utf-8-sig"
        )

        try:
            chunks = pd.read_csv(
                file_path,
                chunksize=CHUNK_SIZE,
                low_memory=False,
                encoding=encoding,
            )

            for chunk in chunks:
                chunk.columns = (
                    chunk.columns.astype(str).str.strip()
                )

                # Remove only rows where every field is empty.
                chunk = chunk.dropna(how="all")

                if chunk.empty:
                    continue

                column_names = chunk.columns.tolist()
                row_count += len(chunk)

                label_col = config["label_column"]

                if label_col not in chunk.columns:
                    raise ValueError(
                        f"Missing label column: {label_col}"
                    )

                labels = (
                    chunk[label_col]
                    .astype("string")
                    .str.strip()
                )

                label_counts.update(
                    labels.fillna("<MISSING>").tolist()
                )

                missing_counts.update(
                    chunk.isna().sum().to_dict()
                )

            global_labels.update(label_counts)
            global_missing.update(missing_counts)
            total_rows += row_count

            file_summaries.append(
                {
                    "file": file_path.name,
                    "rows": row_count,
                    "columns": len(column_names),
                    "label_categories": len(label_counts),
                    "missing_cells": sum(missing_counts.values()),
                }
            )

            print(f"Non-empty rows: {row_count:,}")
            print(f"Columns: {len(column_names)}")
            print(f"Labels: {dict(label_counts)}")
            print(
                "Missing cells in non-empty rows: "
                f"{sum(missing_counts.values()):,}"
            )

        except Exception as exc:
            print(f"ERROR inspecting {file_path.name}: {exc}")
            raise

    pd.DataFrame(file_summaries).to_csv(
        REPORT_DIR / "dataset_file_summary.csv",
        index=False,
    )

    label_df = pd.DataFrame(
        [
            {"label": label, "count": count}
            for label, count in global_labels.most_common()
        ]
    )

    if not label_df.empty and total_rows:
        label_df["percentage"] = (
            label_df["count"] / total_rows * 100
        ).round(4)

    label_df.to_csv(
        REPORT_DIR / "label_distribution.csv",
        index=False,
    )

    missing_df = pd.DataFrame(
        [
            {"column": column, "missing_count": count}
            for column, count in global_missing.items()
            if count > 0
        ],
        columns=["column", "missing_count"],
    )

    if not missing_df.empty and total_rows:
        missing_df["percentage_of_rows"] = (
            missing_df["missing_count"] / total_rows * 100
        ).round(4)

        missing_df = missing_df.sort_values(
            "missing_count",
            ascending=False,
        )

    missing_df.to_csv(
        REPORT_DIR / "missing_values.csv",
        index=False,
    )

    print("\n" + "=" * 60)
    print("DATASET INSPECTION COMPLETED")
    print("=" * 60)
    print(f"Total CSV files: {len(files)}")
    print(f"Total non-empty rows: {total_rows:,}")
    print(f"Total label categories: {len(global_labels)}")

    print("\nCorrected label distribution:")
    if not label_df.empty:
        print(label_df.to_string(index=False))

    print("\nReports saved in:")
    print(REPORT_DIR)
    print("- dataset_file_summary.csv")
    print("- label_distribution.csv")
    print("- missing_values.csv")


if __name__ == "__main__":
    inspect_dataset()
