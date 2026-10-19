"""Same-split model comparison and serializable evaluation outputs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import EvaluationResult, evaluate_classifier


def compare_models(models: dict[str, object], test_frame: pd.DataFrame, target_column: str, output_dir: str | Path | None = None) -> tuple[pd.DataFrame, dict[str, EvaluationResult]]:
    """Evaluate every supplied fitted model on precisely the same held-out rows."""
    if target_column not in test_frame:
        raise ValueError(f"Test data has no target column '{target_column}'.")
    y_true = test_frame[target_column]
    results: dict[str, EvaluationResult] = {}
    for name, model in models.items():
        predictions = model.predict(test_frame)
        probabilities = model.predict_proba(test_frame)
        results[name] = evaluate_classifier(
            model_name=name,
            y_true=y_true,
            y_pred=predictions,
            probabilities=probabilities,
            classes=model.classes_,
        )
    summary = pd.DataFrame([result.summary() for result in results.values()]).sort_values("macro_f1", ascending=False)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        summary.to_csv(destination / "model_comparison.csv", index=False)
        for name, result in results.items():
            pd.DataFrame(result.confusion_matrix, index=result.class_labels, columns=result.class_labels).to_csv(destination / f"{name}_confusion_matrix.csv")
            pd.DataFrame(result.per_class).transpose().to_csv(destination / f"{name}_per_class_metrics.csv")
    return summary, results
