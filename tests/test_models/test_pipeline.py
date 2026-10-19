import pandas as pd

from sentinelgraph.pipeline import prediction_table, split_data
from sentinelgraph.models import FlowOnlyRandomForest
from sentinelgraph.models.base import RandomForestSettings


def test_time_split_keeps_chronological_partitions():
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=20, freq="min").astype(str),
            "duration": list(range(20)),
            "label": ["BENIGN", "ATTACK"] * 10,
        }
    )
    train, validation, test = split_data(frame, "label", validation_size=0.2, test_size=0.2)
    assert len(train) == 12 and len(validation) == 4 and len(test) == 4
    assert train["timestamp"].max() < validation["timestamp"].min() < test["timestamp"].min()


def test_prediction_output_keeps_investigation_context():
    frame = pd.DataFrame(
        {
            "source_ip": ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"],
            "destination_ip": ["8.8.8.8"] * 4,
            "time_window": ["2024-01-01T00:00:00Z"] * 4,
            "duration": [1, 2, 8, 9],
            "label": ["BENIGN", "BENIGN", "ATTACK", "ATTACK"],
        }
    )
    model = FlowOnlyRandomForest(["duration"], RandomForestSettings(n_estimators=10, n_jobs=1)).fit(frame, frame.label)
    output = prediction_table(model, frame, "label")
    assert {"source_ip", "destination_ip", "time_window", "predicted_class", "confidence"} <= set(output)
