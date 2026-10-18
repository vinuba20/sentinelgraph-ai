
"""Clean and combine CIC-IDS2017 CSV files in chunks."""

from pathlib import Path

import numpy as np
import pandas as pd

from sentinelgraph.data.config import load_dataset_config
from sentinelgraph.data.loader import list_csv_files


PROJECT_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "flows_cleaned.csv"
CHUNK_SIZE = 100_000


def prepare_dataset() -> None:
    config = load_dataset_config()
    files = list_csv_files(config["raw_dir"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Do not accidentally append to an old output.
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()

    total_rows = 0
    benign_rows = 0
    attack_rows = 0
    skipped_empty_rows = 0
    skipped_missing_label_rows = 0
    first_output_chunk = True

    for file_path in files:
        print(f"\nProcessing: {file_path.name}")

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

            file_rows = 0

            for chunk in chunks:
                chunk.columns = (
                    chunk.columns.astype(str).str.strip()
                )

                # Exclude rows where every field is empty.
                empty_mask = chunk.isna().all(axis=1)
                skipped_empty_rows += int(empty_mask.sum())
                chunk = chunk.loc[~empty_mask].copy()

                if chunk.empty:
                    continue

                label_col = config["label_column"]

                if label_col not in chunk.columns:
                    raise ValueError(
                        f"Missing label column in {file_path.name}"
                    )

                # Exclude only rows whose label is missing or blank.
                labels = chunk[label_col].astype("string").str.strip()
                missing_label_mask = (
                    labels.isna() | labels.eq("")
                )

                skipped_missing_label_rows += int(
                    missing_label_mask.sum()
                )

                chunk = chunk.loc[~missing_label_mask].copy()
                labels = labels.loc[~missing_label_mask]

                if chunk.empty:
                    continue

                # Preserve the original label text, with surrounding
                # whitespace removed.
                chunk[label_col] = labels.astype(str)

                # Binary target: benign = 0, attack = 1.
                normalized_labels = (
                    chunk[label_col].str.strip().str.upper()
                )
                chunk["label_binary"] = (
                    normalized_labels.ne("BENIGN").astype("int8")
                )

                # Convert numeric feature columns and replace infinities.
                excluded_columns = {
                    label_col,
                    config["source_column"],
                    config["destination_column"],
                    config["timestamp_column"],
                    "Flow ID",
                }

                for column in chunk.columns:
                    if column in excluded_columns:
                        continue

                    # Leave non-numeric columns unchanged; convert
                    # numeric-looking columns to numeric values.
                    converted = pd.to_numeric(
                        chunk[column],
                        errors="coerce",
                    )

                    original_nonmissing = chunk[column].notna()
                    converted_nonmissing = converted.notna()

                    if (
                        converted_nonmissing.sum() > 0
                        or original_nonmissing.sum() == 0
                    ):
                        chunk[column] = converted

                numeric_columns = chunk.select_dtypes(
                    include=[np.number]
                ).columns

                chunk[numeric_columns] = chunk[
                    numeric_columns
                ].replace([np.inf, -np.inf], np.nan)

                # Count the rows written to the output.
                rows_written = len(chunk)
                file_rows += rows_written
                total_rows += rows_written
                benign_rows += int(
                    (chunk["label_binary"] == 0).sum()
                )
                attack_rows += int(
                    (chunk["label_binary"] == 1).sum()
                )

                chunk.to_csv(
                    OUTPUT_FILE,
                    mode="w" if first_output_chunk else "a",
                    header=first_output_chunk,
                    index=False,
                )
                first_output_chunk = False

            print(f"Rows retained: {file_rows:,}")

        except Exception as exc:
            print(f"ERROR processing {file_path.name}: {exc}")
            raise

    if total_rows == 0:
        raise ValueError("No labelled records were written.")

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Output file: {OUTPUT_FILE}")
    print(f"Rows retained: {total_rows:,}")
    print(f"Benign rows: {benign_rows:,}")
    print(f"Attack rows: {attack_rows:,}")
    print(f"Empty rows skipped: {skipped_empty_rows:,}")
    print(
        "Rows skipped for missing labels: "
        f"{skipped_missing_label_rows:,}"
    )
    print(f"Output size: {OUTPUT_FILE.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    prepare_dataset()
