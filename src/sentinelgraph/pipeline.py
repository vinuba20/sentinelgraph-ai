"""Reproducible training entry point for the Member 2 modeling pipeline.

Member 1 provides a processed CSV with one row per flow. Required contract: a
normalized ``label`` column (``BENIGN`` or ``ATTACK``) and numeric flow fields.
Graph fields must be joined to that same row before this script is called and use
the ``graph_`` prefix.  Identity/timestamp columns are retained only for outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import ParameterGrid, train_test_split

from sentinelgraph.evaluation import compare_models, evaluate_classifier
from sentinelgraph.models import FlowOnlyRandomForest, GraphEnhancedRandomForest
from sentinelgraph.models.base import RandomForestSettings

IDENTITY_COLUMNS = {"timestamp", "time_window", "source_ip", "destination_ip", "src_ip", "dst_ip", "flow_id"}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def infer_feature_columns(frame: pd.DataFrame, target_column: str) -> tuple[list[str], list[str]]:
    """Infer numeric flow and graph inputs using the documented `graph_` contract."""
    candidates = [
        name for name in frame.columns
        if name != target_column and name.lower() not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(frame[name])
    ]
    graph = [name for name in candidates if name.lower().startswith("graph_")]
    flow = [name for name in candidates if name not in graph]
    if not flow:
        raise ValueError("No numeric flow features found. Specify cleaned numeric CIC flow fields.")
    return flow, graph


def split_data(
    frame: pd.DataFrame,
    target_column: str,
    timestamp_column: str = "timestamp",
    test_size: float = 0.2,
    validation_size: float = 0.2,
    time_split: bool = True,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test partitions without leaking test rows into fitting."""
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        raise ValueError("test_size and validation_size must be positive and sum to less than one.")
    if target_column not in frame:
        raise ValueError(f"Missing target column '{target_column}'.")
    usable = frame.dropna(subset=[target_column]).copy()
    if time_split and timestamp_column in usable:
        timestamps = pd.to_datetime(usable[timestamp_column], errors="coerce", utc=True)
        if timestamps.notna().all():
            ordered = usable.assign(_split_time=timestamps).sort_values("_split_time", kind="stable").drop(columns="_split_time")
            test_start = int(len(ordered) * (1 - test_size))
            validation_start = int(len(ordered) * (1 - test_size - validation_size))
            return ordered.iloc[:validation_start], ordered.iloc[validation_start:test_start], ordered.iloc[test_start:]
    train_val, test = train_test_split(usable, test_size=test_size, stratify=usable[target_column], random_state=random_state)
    validation_relative = validation_size / (1 - test_size)
    train, validation = train_test_split(train_val, test_size=validation_relative, stratify=train_val[target_column], random_state=random_state)
    return train, validation, test


def select_settings(
    feature_columns: list[str],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    target_column: str,
    base_settings: RandomForestSettings,
    candidates: list[dict[str, list[Any]]] | None,
    model_cls: type[FlowOnlyRandomForest] | type[GraphEnhancedRandomForest],
) -> RandomForestSettings:
    """Choose hyperparameters only on the validation partition using macro F1."""
    grids = candidates or [{}]
    combinations = chain.from_iterable(ParameterGrid(grid) for grid in grids)
    best_settings, best_score = base_settings, -np.inf
    for changes in combinations:
        settings = RandomForestSettings(**(asdict(base_settings) | changes))
        model = model_cls(feature_columns, settings).fit(train, train[target_column])
        result = evaluate_classifier(model_name="validation", y_true=validation[target_column], y_pred=model.predict(validation))
        if result.macro_f1 > best_score:
            best_settings, best_score = settings, result.macro_f1
    return best_settings


def prediction_table(model, test: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Return analyst-facing predictions with preserved flow context and confidence."""
    predicted = model.predict(test)
    probabilities = model.predict_proba(test)
    confidence = probabilities.max(axis=1)
    context_columns = [column for column in test.columns if column.lower() in IDENTITY_COLUMNS]
    output = test.loc[:, context_columns].copy()
    output["actual_class"] = test[target_column].values
    output["predicted_class"] = predicted
    output["confidence"] = confidence
    output["is_error"] = output["actual_class"] != output["predicted_class"]
    # Keep the actual numeric inputs visible to an analyst without exposing them
    # to feature selection on a subsequent training run.
    for feature in model.feature_columns:
        output[f"feature__{feature}"] = test[feature].values
    return output


def run_experiment(input_csv: str | Path, output_dir: str | Path, config: dict[str, Any], data_version: str = "unspecified") -> pd.DataFrame:
    """Train baseline and graph-enhanced forests and write reproducible artifacts."""
    frame = pd.read_csv(input_csv, low_memory=False).replace([np.inf, -np.inf], np.nan)
    target = config.get("target_column", "label")
    timestamp = config.get("timestamp_column", "timestamp")
    flow_features, graph_features = infer_feature_columns(frame, target)
    if not graph_features:
        raise ValueError("No graph_* columns found. Join Member 1 graph features before running the combined experiment.")
    train, validation, test = split_data(
        frame, target, timestamp, config.get("test_size", 0.2), config.get("validation_size", 0.2),
        config.get("time_split", True), config.get("random_state", 42),
    )
    parsed_timestamps = pd.to_datetime(frame[timestamp], errors="coerce", utc=True) if timestamp in frame else pd.Series(dtype="datetime64[ns, UTC]")
    used_time_split = bool(config.get("time_split", True) and timestamp in frame and parsed_timestamps.notna().all())
    model_config = config.get("model", {})
    settings = RandomForestSettings(random_state=config.get("random_state", 42), **model_config)
    candidate_grids = config.get("tuning", {}).get("candidates") if config.get("tuning", {}).get("enabled", False) else None
    baseline_settings = select_settings(flow_features, train, validation, target, settings, candidate_grids, FlowOnlyRandomForest)
    combined_features = flow_features + graph_features
    combined_settings = select_settings(combined_features, train, validation, target, settings, candidate_grids, GraphEnhancedRandomForest)
    baseline = FlowOnlyRandomForest(flow_features, baseline_settings).fit(train, train[target])
    combined = GraphEnhancedRandomForest(combined_features, combined_settings).fit(train, train[target])

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    summary, _ = compare_models({"flow_only": baseline, "graph_enhanced": combined}, test, target, destination / "tables")
    baseline.save(destination / "models" / "flow_only.joblib")
    combined.save(destination / "models" / "graph_enhanced.joblib")
    prediction_table(baseline, test, target).assign(model="flow_only").to_csv(destination / "tables" / "flow_only_predictions.csv", index=False)
    prediction_table(combined, test, target).assign(model="graph_enhanced").to_csv(destination / "tables" / "graph_enhanced_predictions.csv", index=False)
    metadata = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(), "input_csv": str(input_csv), "data_version": data_version,
        "split": {"strategy": "time" if used_time_split else "stratified_random", "train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "flow_features": flow_features, "graph_features": graph_features,
        "baseline_settings": asdict(baseline_settings), "combined_settings": asdict(combined_settings),
    }
    (destination / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SentinelGraph flow-only and graph-enhanced Random Forests.")
    parser.add_argument("--input", required=True, help="Processed flow CSV including label and graph_* features.")
    parser.add_argument("--output", default="artifacts/latest", help="Directory for models, tables, and metadata.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-version", default="unspecified")
    args = parser.parse_args()
    summary = run_experiment(args.input, args.output, load_config(args.config), args.data_version)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
