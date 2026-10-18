from pathlib import Path
import json
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "flows_with_graph_features.csv"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "tables"

CHUNK_SIZE = 50_000
RANDOM_STATE = 42

TARGET = "label_binary"
DATE_COLUMN = "Timestamp"

IDENTIFIER_COLUMNS = {
    "Flow ID",
    "Source IP",
    "Destination IP",
    "Timestamp",
    "Label",
    "label_binary",
}

GRAPH_COLUMNS = [
    "graph_source_out_degree",
    "graph_destination_in_degree",
    "graph_edge_flow_count",
]

DUPLICATE_GRAPH_COLUMNS = {
    "graph_source_unique_destinations",
    "graph_destination_unique_sources",
}

TRAIN_DATES = {"2017-07-03", "2017-07-04", "2017-07-05"}
TEST_DATES = {"2017-07-06", "2017-07-07"}


def get_date_series(series):
    return pd.to_datetime(
        series.astype("string").str.strip(),
        format="mixed",
        dayfirst=True,
        errors="coerce",
    ).dt.strftime("%Y-%m-%d")


def numeric_features(columns, include_graph):
    excluded = IDENTIFIER_COLUMNS | DUPLICATE_GRAPH_COLUMNS

    if not include_graph:
        excluded.update(GRAPH_COLUMNS)

    return [
        column for column in columns
        if column not in excluded
    ]


def prepare_features(frame, features, medians):
    values = frame[features].apply(pd.to_numeric, errors="coerce")
    values = values.replace([np.inf, -np.inf], np.nan)
    values = values.fillna(medians)
    return values.to_numpy(dtype=np.float64, copy=False)


def collect_training_medians(features):
    sample_parts = []
    samples_per_chunk = 1_500
    chunk_number = 0
    collected = 0

    print("Estimating medians across training chunks...")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        dates = get_date_series(chunk[DATE_COLUMN])
        train = chunk.loc[dates.isin(TRAIN_DATES)]

        if train.empty:
            continue

        take = train.sample(
            n=min(samples_per_chunk, len(train)),
            random_state=RANDOM_STATE + chunk_number,
        )

        sample_parts.append(
            take[features].apply(pd.to_numeric, errors="coerce")
        )

        collected += len(take)
        chunk_number += 1

    if not sample_parts:
        raise RuntimeError("No training rows found for median estimation.")

    sample = pd.concat(sample_parts, ignore_index=True)
    sample = sample.replace([np.inf, -np.inf], np.nan)
    medians = sample.median().fillna(0.0)

    print(
        f"Median estimates calculated from {collected:,} sampled "
        f"training rows across {chunk_number} chunks."
    )
    return medians

def fit_scaler(features, medians):
    scaler = StandardScaler()
    rows_seen = 0

    print("Fitting feature scaler on training dates...")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        dates = get_date_series(chunk[DATE_COLUMN])
        train = chunk.loc[dates.isin(TRAIN_DATES)]

        if train.empty:
            continue

        x = prepare_features(train, features, medians)
        scaler.partial_fit(x)
        rows_seen += len(train)

    print(f"Scaler fitted on {rows_seen:,} training rows.")
    return scaler


def fit_classifier(features, medians, scaler, model_name):
    classifier = SGDClassifier(
        loss="log_loss",
        class_weight={0: 1.0, 1: 5.26},
        max_iter=1,
        tol=None,
        random_state=RANDOM_STATE,
        learning_rate="optimal",
    )

    rows_seen = 0
    started = time.time()
    classes = np.array([0, 1])

    print(f"Training {model_name}...")

    for epoch in range(3):
        epoch_rows = 0
        rng = np.random.default_rng(RANDOM_STATE + epoch)

        for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
            dates = get_date_series(chunk[DATE_COLUMN])
            train = chunk.loc[dates.isin(TRAIN_DATES)]

            if train.empty:
                continue

            y = pd.to_numeric(
                train[TARGET], errors="coerce"
            ).to_numpy(dtype=np.int32)

            x = prepare_features(train, features, medians)
            x = scaler.transform(x)

            order = rng.permutation(len(y))
            x = x[order]
            y = y[order]

            if epoch == 0 and epoch_rows == 0:
                classifier.partial_fit(x, y, classes=classes)
            else:
                classifier.partial_fit(x, y)

            epoch_rows += len(train)

        rows_seen = epoch_rows
        print(
            f"  Epoch {epoch + 1}/3 complete: "
            f"{epoch_rows:,} training rows processed."
        )

    print(
        f"Training completed in {time.time() - started:.1f} seconds."
    )

    return classifier, rows_seen


def evaluate_model(classifier, scaler, features, medians, model_name):
    y_true_parts = []
    y_score_parts = []
    y_pred_parts = []
    rows_seen = 0

    print(f"Evaluating {model_name} on July 6–7...")

    for chunk in pd.read_csv(DATA_FILE, chunksize=CHUNK_SIZE):
        dates = get_date_series(chunk[DATE_COLUMN])
        test = chunk.loc[dates.isin(TEST_DATES)]

        if test.empty:
            continue

        y = pd.to_numeric(
            test[TARGET], errors="coerce"
        ).to_numpy(dtype=np.int32)

        x = prepare_features(test, features, medians)
        x = scaler.transform(x)

        scores = classifier.predict_proba(x)[:, 1]
        predictions = (scores >= 0.5).astype(np.int32)

        y_true_parts.append(y)
        y_score_parts.append(scores)
        y_pred_parts.append(predictions)
        rows_seen += len(test)

    if rows_seen == 0:
        raise RuntimeError("No test rows found.")

    y_true = np.concatenate(y_true_parts)
    y_scores = np.concatenate(y_score_parts)
    y_pred = np.concatenate(y_pred_parts)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "model": model_name,
        "test_rows": int(rows_seen),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "threshold": 0.5,
    }

    print(f"\n{model_name} results:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    return metrics


def run_experiment(include_graph, columns):
    name = "graph_enhanced" if include_graph else "flow_only"
    features = numeric_features(columns, include_graph)

    print("\n" + "=" * 60)
    print(f"EXPERIMENT: {name}")
    print("Number of features:", len(features))
    if include_graph:
        print("Graph features:", GRAPH_COLUMNS)
    print("=" * 60)

    medians = collect_training_medians(features)
    scaler = fit_scaler(features, medians)
    classifier, training_rows = fit_classifier(
        features, medians, scaler, name
    )

    metrics = evaluate_model(
        classifier, scaler, features, medians, name
    )
    metrics["training_rows_per_epoch"] = int(training_rows)
    metrics["feature_count"] = len(features)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "classifier": classifier,
            "scaler": scaler,
            "features": features,
            "medians": medians,
            "threshold": 0.5,
            "model_name": name,
        },
        MODEL_DIR / f"{name}.joblib",
    )

    return metrics


def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_FILE}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    columns = pd.read_csv(DATA_FILE, nrows=0).columns.tolist()

    required = {TARGET, DATE_COLUMN, *GRAPH_COLUMNS}
    missing = required.difference(columns)
    if missing:
        raise ValueError(f"Required columns are missing: {sorted(missing)}")

    results = []

    for include_graph in (False, True):
        results.append(run_experiment(include_graph, columns))

    report = pd.DataFrame(results)
    output_csv = REPORT_DIR / "model_comparison.csv"
    output_json = REPORT_DIR / "model_comparison.json"

    report.to_csv(output_csv, index=False)
    output_json.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(
        report[
            [
                "model",
                "precision",
                "recall",
                "f1_score",
                "pr_auc",
                "false_positive_rate",
            ]
        ].to_string(index=False)
    )
    print(f"\nCSV report: {output_csv}")
    print(f"JSON report: {output_json}")
    print(f"Saved models in: {MODEL_DIR}")


if __name__ == "__main__":
    main()
