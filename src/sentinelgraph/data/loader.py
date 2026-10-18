
"""Load and inspect CIC-IDS2017 CSV files."""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def list_csv_files(data_dir: Path = RAW_DATA_DIR) -> list[Path]:
    """Return all CSV files directly inside the raw data directory."""
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    files = sorted(data_dir.glob("*.csv"))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            "Place CIC-IDS2017 CSV files in data/raw/."
        )

    return files


def load_csv(file_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Load a CSV file and normalize column names."""
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    df = pd.read_csv(
        file_path,
        nrows=nrows,
        low_memory=False,
        encoding="utf-8-sig",
    )

    # CIC-IDS2017 column names may contain leading/trailing spaces.
    df.columns = df.columns.astype(str).str.strip()

    return df


def inspect_csv(file_path: str | Path, sample_rows: int = 5) -> dict:
    """Return basic information about a CSV without loading it twice."""
    df = load_csv(file_path, nrows=sample_rows)

    return {
        "file": Path(file_path).name,
        "sample_rows": len(df),
        "columns": df.columns.tolist(),
        "sample_labels": (
            df["Label"].astype(str).unique().tolist()
            if "Label" in df.columns
            else []
        ),
    }


if __name__ == "__main__":
    files = list_csv_files()

    print(f"Dataset directory: {RAW_DATA_DIR}")
    print(f"CSV files found: {len(files)}")

    for file_path in files:
        print(f"\nInspecting: {file_path.name}")

        try:
            info = inspect_csv(file_path)
            print(f"Columns: {len(info['columns'])}")
            print(f"Sample labels: {info['sample_labels']}")
            print(f"First columns: {info['columns'][:8]}")
        except Exception as exc:
            print(f"Could not inspect file: {exc}")
