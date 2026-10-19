"""Shared, leakage-safe Random Forest model implementation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class RandomForestSettings:
    n_estimators: int = 300
    max_depth: int | None = None
    min_samples_leaf: int = 1
    max_features: str | float | None = "sqrt"
    class_weight: str | dict | None = "balanced_subsample"
    random_state: int = 42
    n_jobs: int = -1


class RandomForestIntrusionModel:
    """A numeric-feature model that fits imputation only on training rows.

    ``feature_columns`` is deliberately explicit.  This avoids accidentally using a
    target, identifier, or future-derived column as a model input.
    """

    model_kind = "random_forest"

    def __init__(
        self,
        feature_columns: Sequence[str],
        settings: RandomForestSettings | None = None,
    ) -> None:
        if not feature_columns:
            raise ValueError("At least one feature column is required.")
        self.feature_columns = list(feature_columns)
        self.settings = settings or RandomForestSettings()
        self.pipeline = Pipeline(
            [
                # Retain an all-missing input as an explicit zero-filled feature so
                # feature names and importance vectors remain aligned in reports.
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("classifier", RandomForestClassifier(**asdict(self.settings))),
            ]
        )
        self.is_fitted = False

    def _features(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = sorted(set(self.feature_columns).difference(frame.columns))
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")
        data = frame.loc[:, self.feature_columns].copy()
        for column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.replace([np.inf, -np.inf], np.nan)
        return data

    def fit(self, frame: pd.DataFrame, target: Iterable[object]) -> "RandomForestIntrusionModel":
        labels = pd.Series(target, index=frame.index)
        if labels.nunique(dropna=True) < 2:
            raise ValueError("Training data must contain at least two classes.")
        self.pipeline.fit(self._features(frame), labels)
        self.is_fitted = True
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline.predict(self._features(frame))

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline.predict_proba(self._features(frame))

    @property
    def classes_(self) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline.named_steps["classifier"].classes_

    @property
    def feature_importances_(self) -> np.ndarray:
        self._ensure_fitted()
        return self.pipeline.named_steps["classifier"].feature_importances_

    def save(self, path: str | Path) -> None:
        self._ensure_fitted()
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestIntrusionModel":
        model = joblib.load(path)
        if not isinstance(model, RandomForestIntrusionModel):
            raise TypeError("Artifact is not a SentinelGraph Random Forest model.")
        return model

    def _ensure_fitted(self) -> None:
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted. Call fit with training data first.")
