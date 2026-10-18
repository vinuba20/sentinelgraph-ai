
"""Load and validate the SentinelGraph AI dataset configuration."""

from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = PROJECT_ROOT / "configs" / "dataset.yaml"


def load_dataset_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Read dataset settings from a YAML configuration file."""
    config_path = Path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict) or not isinstance(
        config.get("dataset"), dict
    ):
        raise ValueError(
            "Configuration must contain a 'dataset' section."
        )

    required_keys = [
        "name",
        "raw_dir",
        "label_column",
        "source_column",
        "destination_column",
        "timestamp_column",
        "graph_window",
    ]

    dataset = config["dataset"]
    missing_keys = [
        key for key in required_keys if not dataset.get(key)
    ]

    if missing_keys:
        raise ValueError(
            f"Missing dataset configuration keys: {missing_keys}"
        )

    dataset["raw_dir"] = Path(dataset["raw_dir"])
    return dataset


if __name__ == "__main__":
    settings = load_dataset_config()
    print("Configuration loaded successfully!")
    print(f"Dataset: {settings['name']}")
    print(f"Dataset folder: {settings['raw_dir']}")
    print(f"Folder exists: {settings['raw_dir'].is_dir()}")
    print(f"Label column: {settings['label_column']}")
    print(f"Source column: {settings['source_column']}")
    print(f"Destination column: {settings['destination_column']}")
    print(f"Timestamp column: {settings['timestamp_column']}")
    print(f"Graph window: {settings['graph_window']}")
