"""Flow-only Random Forest baseline."""

from __future__ import annotations

from .base import RandomForestIntrusionModel, RandomForestSettings


class FlowOnlyRandomForest(RandomForestIntrusionModel):
    """Random Forest trained exclusively on per-flow numeric features."""

    model_kind = "flow_only"

    def __init__(self, flow_feature_columns: list[str], settings: RandomForestSettings | None = None):
        super().__init__(flow_feature_columns, settings)
