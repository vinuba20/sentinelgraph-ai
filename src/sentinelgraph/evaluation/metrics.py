"""Metrics required for a fair intrusion-detection model comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


@dataclass
class EvaluationResult:
    model_name: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    macro_f1: float
    false_positive_rate: float
    pr_auc: float | None
    confusion_matrix: list[list[int]]
    class_labels: list[str]
    per_class: dict[str, Any]

    def summary(self) -> dict[str, Any]:
        """Flat values suitable for a comparison CSV or dashboard table."""
        values = asdict(self)
        return {key: value for key, value in values.items() if key not in {"confusion_matrix", "per_class", "class_labels"}}


def _positive_scores(classes: np.ndarray, probabilities: np.ndarray, positive_label: object) -> np.ndarray | None:
    if probabilities.ndim != 2 or positive_label not in classes:
        return None
    return probabilities[:, int(np.where(classes == positive_label)[0][0])]


def evaluate_classifier(
    *,
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
    probabilities: np.ndarray | None = None,
    classes: np.ndarray | None = None,
    positive_label: object = "ATTACK",
) -> EvaluationResult:
    """Evaluate binary or multiclass predictions without silently assuming label order."""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    observed_labels = np.unique(np.concatenate([actual, predicted]))

    if len(observed_labels) == 2 and set(observed_labels) == {"BENIGN", "ATTACK"}:
        labels = np.array(["BENIGN", "ATTACK"])
    else:
        labels = observed_labels    is_binary = len(labels) == 2
    positive = positive_label if positive_label in labels else labels[-1]
    average = "binary" if is_binary else "weighted"
    metric_kwargs = {"average": average, "zero_division": 0}
    if is_binary:
        metric_kwargs["pos_label"] = positive

    matrix = confusion_matrix(actual, predicted, labels=labels)
    if is_binary:
        negative_index = 0 if labels[1] == positive else 1
        positive_index = 1 - negative_index
        tn, fp = matrix[negative_index, negative_index], matrix[negative_index, positive_index]
        false_positive_rate = float(fp / (fp + tn)) if fp + tn else 0.0
    else:
        false_positive_rate = float("nan")

    pr_auc: float | None = None
    if is_binary and probabilities is not None and classes is not None:
        scores = _positive_scores(np.asarray(classes), probabilities, positive)
        if scores is not None:
            pr_auc = float(average_precision_score(actual == positive, scores))

    report = classification_report(actual, predicted, labels=labels, output_dict=True, zero_division=0)
    return EvaluationResult(
        model_name=model_name,
        n_samples=len(actual),
        accuracy=float(accuracy_score(actual, predicted)),
        precision=float(precision_score(actual, predicted, **metric_kwargs)),
        recall=float(recall_score(actual, predicted, **metric_kwargs)),
        f1=float(f1_score(actual, predicted, **metric_kwargs)),
        macro_f1=float(f1_score(actual, predicted, average="macro", zero_division=0)),
        false_positive_rate=false_positive_rate,
        pr_auc=pr_auc,
        confusion_matrix=matrix.astype(int).tolist(),
        class_labels=[str(label) for label in labels],
        per_class=report,
    )
