
import pandas as pd
import pytest

from sentinelgraph.graph.builder import (
    build_communication_graph,
    build_time_window_graphs,
)


def test_communication_graph_counts_flows():
    df = pd.DataFrame({
        "Source IP": ["A", "A", "B"],
        "Destination IP": ["B", "B", "C"],
    })

    graph = build_communication_graph(df)

    assert graph.number_of_nodes() == 3
    assert graph.number_of_edges() == 2
    assert graph["A"]["B"]["flow_count"] == 2


def test_time_window_graphs_separate_minutes():
    df = pd.DataFrame({
        "Timestamp": [
            "2026-01-01 10:00:05",
            "2026-01-01 10:00:35",
            "2026-01-01 10:01:10",
        ],
        "Source IP": ["A", "A", "B"],
        "Destination IP": ["B", "B", "C"],
    })

    graphs = build_time_window_graphs(df)

    assert len(graphs) == 2

    flow_counts = [
        sum(
            attributes["flow_count"]
            for _, _, attributes in graph.edges(data=True)
        )
        for graph in graphs.values()
    ]

    assert flow_counts == [2, 1]


def test_graph_requires_endpoint_columns():
    df = pd.DataFrame({"other": [1]})

    with pytest.raises(ValueError):
        build_communication_graph(df)
