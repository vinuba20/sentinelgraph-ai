"""Explainability utilities with no hard dependency on SHAP."""

from __future__ import annotations

import numpy as np
import pandas as pd


def global_feature_importance(model, top_n: int | None = 20) -> pd.DataFrame:
    """Return the fitted forest's global impurity-based feature ranking."""
    ranking = pd.DataFrame({"feature": model.feature_columns, "importance": model.feature_importances_})
    ranking = ranking.sort_values("importance", ascending=False, ignore_index=True)
    return ranking if top_n is None else ranking.head(top_n)


def explain_prediction(model, flow: pd.DataFrame | pd.Series, top_n: int = 10) -> pd.DataFrame:
    """Provide an auditable local approximation of the factors behind one prediction.

    The contribution is the standardized distance from the training median multiplied
    by global feature importance. It is intentionally labelled as an approximation;
    use :func:`shap_values` when the optional SHAP package is installed.
    """
    row = flow.to_frame().T if isinstance(flow, pd.Series) else flow.iloc[[0]]
    values = model._features(row).iloc[0]
    imputer = model.pipeline.named_steps["imputer"]
    medians = pd.Series(imputer.statistics_, index=model.feature_columns)
    contribution = (values - medians).abs().fillna(0) * model.feature_importances_
    output = pd.DataFrame(
        {
            "feature": model.feature_columns,
            "value": values.values,
            "training_median": medians.values,
            "importance": model.feature_importances_,
            "approximate_contribution": contribution.values,
        }
    )
    return output.sort_values("approximate_contribution", ascending=False, ignore_index=True).head(top_n)


def shap_values(model, frame: pd.DataFrame):
    """Calculate TreeSHAP values when SHAP is installed by the user."""
    try:
        import shap
    except ImportError as error:  # pragma: no cover - optional dependency
        raise RuntimeError("Install the 'explainability' extra to use SHAP explanations.") from error
    matrix = model.pipeline.named_steps["imputer"].transform(model._features(frame))
    return shap.TreeExplainer(model.pipeline.named_steps["classifier"]).shap_values(matrix)
