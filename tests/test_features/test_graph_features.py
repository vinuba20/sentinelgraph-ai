
import pandas as pd

from sentinelgraph.features.graph_features import add_graph_features


def test_graph_features_count_repeated_flows():
    df = pd.DataFrame({
        "Timestamp": [
            "2026-01-01 10:00:05",
            "2026-01-01 10:00:35",
            "2026-01-01 10:01:10",
        ],
        "Source IP": ["A", "A", "B"],
        "Destination IP": ["B", "B", "C"],
    })

    result = add_graph_features(df)

    assert result["graph_edge_flow_count"].tolist() == [2, 2, 1]


def test_graph_features_preserve_rows_and_order():
    df = pd.DataFrame({
        "Timestamp": [
            "2026-01-01 10:00:05",
            "invalid",
            "2026-01-01 10:01:10",
        ],
        "Source IP": ["A", "X", "B"],
        "Destination IP": ["B", "Y", "C"],
    })

    result = add_graph_features(df)

    assert len(result) == len(df)
    assert result["Source IP"].tolist() == ["A", "X", "B"]
    assert result["graph_edge_flow_count"].tolist() == [1, 0, 1]


def test_graph_features_do_not_require_unique_dataframe_index():
    df = pd.DataFrame({
        "Timestamp": [
            "2026-01-01 10:00:05",
            "2026-01-01 10:00:10",
        ],
        "Source IP": ["A", "A"],
        "Destination IP": ["B", "B"],
    }, index=[5, 5])

    result = add_graph_features(df)

    assert result["graph_edge_flow_count"].tolist() == [2, 2]
