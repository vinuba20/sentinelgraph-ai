
"""Build communication graphs from network-flow records."""

import pandas as pd
import networkx as nx


def build_communication_graph(
    df: pd.DataFrame,
    source_col: str = "Source IP",
    destination_col: str = "Destination IP",
) -> nx.DiGraph:
    """Build a directed graph from network-flow records."""
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    required = {source_col, destination_col}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data = data.dropna(subset=[source_col, destination_col])
    graph = nx.DiGraph()

    for source, destination in zip(
        data[source_col].astype(str),
        data[destination_col].astype(str),
    ):
        source = source.strip()
        destination = destination.strip()

        if not source or not destination:
            continue

        if graph.has_edge(source, destination):
            graph[source][destination]["flow_count"] += 1
        else:
            graph.add_edge(source, destination, flow_count=1)

    return graph


def build_time_window_graphs(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    source_col: str = "Source IP",
    destination_col: str = "Destination IP",
    window: str = "1min",
) -> dict[pd.Timestamp, nx.DiGraph]:
    """Build one directed communication graph per time window.

    Rows without valid timestamps or endpoint addresses are excluded.
    Timestamps are interpreted using pandas datetime parsing.
    """
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    required = {timestamp_col, source_col, destination_col}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    data[timestamp_col] = pd.to_datetime(
        data[timestamp_col], errors="coerce"
    )

    data = data.dropna(
        subset=[timestamp_col, source_col, destination_col]
    )

    data[source_col] = data[source_col].astype(str).str.strip()
    data[destination_col] = data[destination_col].astype(str).str.strip()

    data = data[
        data[source_col].ne("") & data[destination_col].ne("")
    ].copy()

    if data.empty:
        return {}

    data["__window"] = data[timestamp_col].dt.floor(window)

    graphs = {}

    for window_start, group in data.groupby("__window", sort=True):
        graphs[window_start] = build_communication_graph(
            group,
            source_col=source_col,
            destination_col=destination_col,
        )

    return graphs


def graph_summary(graph: nx.DiGraph) -> dict:
    """Return basic graph statistics."""
    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "total_communication_flows": sum(
            attributes.get("flow_count", 1)
            for _, _, attributes in graph.edges(data=True)
        ),
        "density": nx.density(graph),
    }
