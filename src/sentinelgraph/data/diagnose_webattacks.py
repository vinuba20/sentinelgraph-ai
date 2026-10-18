
"""Diagnose missing labels and empty records in the Web Attacks CSV."""

from pathlib import Path

import pandas as pd

from sentinelgraph.data.config import load_dataset_config


CHUNK_SIZE = 100_000
TARGET_NAME = "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv"


def diagnose_webattacks() -> None:
    config = load_dataset_config()
    file_path = Path(config["raw_dir"]) / TARGET_NAME

    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    print(f"File: {file_path}")
    print("\nReading CSV in chunks...")

    total_rows = 0
    missing_label_rows = 0
    fully_empty_rows = 0
    missing_label_with_features = 0
    sample_missing_label_rows = []

    for chunk in pd.read_csv(
        file_path,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="cp1252",
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()
        total_rows += len(chunk)

        labels = chunk[config["label_column"]].astype("string").str.strip()
        missing_labels = labels.isna() | labels.eq("")

        # Count rows where every parsed field is missing.
        empty_mask = chunk.isna().all(axis=1)

        missing_label_rows += int(missing_labels.sum())
        fully_empty_rows += int(empty_mask.sum())
        missing_label_with_features += int(
            (missing_labels & ~empty_mask).sum()
        )

        if len(sample_missing_label_rows) < 5:
            examples = chunk.loc[missing_labels].head(
                5 - len(sample_missing_label_rows)
            )
            sample_missing_label_rows.extend(
                examples.to_dict(orient="records")
            )

    print("\n" + "=" * 60)
    print("WEB ATTACKS CSV DIAGNOSTIC")
    print("=" * 60)
    print(f"Total parsed rows: {total_rows:,}")
    print(f"Rows with missing/blank labels: {missing_label_rows:,}")
    print(f"Rows with every field missing: {fully_empty_rows:,}")
    print(
        "Rows with missing labels but other data present: "
        f"{missing_label_with_features:,}"
    )

    if sample_missing_label_rows:
        print("\nExamples of rows with missing labels:")
        for index, row in enumerate(sample_missing_label_rows, start=1):
            print(f"\nExample {index}:")
            for column, value in list(row.items())[:12]:
                print(f"  {column}: {value!r}")

    print("\nDiagnostic completed. No data was modified.")


if __name__ == "__main__":
    diagnose_webattacks()
