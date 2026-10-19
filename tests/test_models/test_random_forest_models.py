import pandas as pd

from sentinelgraph.models import FlowOnlyRandomForest, GraphEnhancedRandomForest
from sentinelgraph.models.base import RandomForestSettings


def training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "flow_duration": [1, 2, 1, 9, 10, 11, 2, 12],
            "bytes": [20, 21, 19, 900, 850, 1000, 18, 970],
            "graph_unique_destinations": [1, 1, 2, 9, 10, 8, 1, 11],
            "label": ["BENIGN", "BENIGN", "BENIGN", "ATTACK", "ATTACK", "ATTACK", "BENIGN", "ATTACK"],
        }
    )


def test_flow_model_predicts_and_persists(tmp_path):
    frame = training_frame()
    settings = RandomForestSettings(n_estimators=10, random_state=3, n_jobs=1)
    model = FlowOnlyRandomForest(["flow_duration", "bytes"], settings).fit(frame, frame["label"])
    assert set(model.predict(frame)) <= {"BENIGN", "ATTACK"}
    artifact = tmp_path / "model.joblib"
    model.save(artifact)
    assert (FlowOnlyRandomForest.load(artifact).predict(frame) == model.predict(frame)).all()


def test_combined_model_requires_its_graph_feature():
    frame = training_frame()
    model = GraphEnhancedRandomForest(
        ["flow_duration", "bytes", "graph_unique_destinations"], RandomForestSettings(n_estimators=10, n_jobs=1)
    ).fit(frame, frame["label"])
    assert model.predict_proba(frame).shape == (len(frame), 2)
    try:
        model.predict(frame.drop(columns="graph_unique_destinations"))
    except ValueError as error:
        assert "graph_unique_destinations" in str(error)
    else:
        raise AssertionError("Expected required graph feature validation")
