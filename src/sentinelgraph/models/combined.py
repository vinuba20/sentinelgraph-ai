"""Graph-enhanced Random Forest model."""

from __future__ import annotations

from .base import RandomForestIntrusionModel, RandomForestSettings


class GraphEnhancedRandomForest(RandomForestIntrusionModel):
    """Random Forest trained on flow features plus precomputed graph features."""

    model_kind = "graph_enhanced"

    def __init__(self, combined_feature_columns: list[str], settings: RandomForestSettings | None = None):
        super().__init__(combined_feature_columns, settings)
