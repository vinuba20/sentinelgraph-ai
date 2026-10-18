
"""Validate the CIC-IDS2017 dataset files and required columns."""

from sentinelgraph.data.config import load_dataset_config
from sentinelgraph.data.loader import list_csv_files, load_csv


def validate_dataset() -> bool:
    config = load_dataset_config()
    data_dir = config["raw_dir"]

    required_columns = [
        config["label_column"],
        config["source_column"],
        config["destination_column"],
        config["timestamp_column"],
    ]

    files = list_csv_files(data_dir)
    print(f"Dataset: {config['name']}")
    print(f"Dataset folder: {data_dir}")
    print(f"CSV files found: {len(files)}")
    print("-" * 60)

    all_valid = True

    for file_path in files:
        try:
            # Read column headers only; do not load the full dataset.
            df = load_csv(file_path, nrows=0)
            columns = set(df.columns)

            missing = [
                column
                for column in required_columns
                if column not in columns
            ]

            if missing:
                all_valid = False
                print(f"[FAIL] {file_path.name}")
                print(f"       Missing columns: {missing}")
            else:
                print(f"[PASS] {file_path.name}")
                print(f"       Required columns are present.")

        except Exception as exc:
            all_valid = False
            print(f"[ERROR] {file_path.name}: {exc}")

    print("-" * 60)

    if all_valid:
        print("RESULT: All CSV files passed column validation.")
    else:
        print("RESULT: Validation failed. Check the errors above.")

    return all_valid


if __name__ == "__main__":
    success = validate_dataset()
    raise SystemExit(0 if success else 1)
