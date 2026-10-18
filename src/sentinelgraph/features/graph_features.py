
"""Calculate graph-derived features for network flows."""

import pandas as pd

from sentinelgraph.graph.builder import build_time_window_graphs


def add_graph_features(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    source_col: str = "Source IP",
    destination_col: str = "Destination IP",
) -> pd.DataFrame:
    """Add graph features to each flow using its one-minute graph.

    Original row order and index are preserved.
    Invalid timestamps or missing endpoints receive zero graph features.
    Raw IP addresses are used for graph construction, not returned as
    predictive features by this function.
    """
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    required = {timestamp_col, source_col, destination_col}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Unique internal row identifier allows features to be mapped back
    # even when the original DataFrame index contains duplicates.
    data["__row_id"] = range(len(data))

    parsed_timestamps = pd.to_datetime(
        data[timestamp_col], errors="coerce"
    )
    valid = (
        parsed_timestamps.notna()
        & data[source_col].notna()
        & data[destination_col].notna()
    )

    features = pd.DataFrame(
        {
            "graph_source_out_degree": 0,
            "graph_destination_in_degree": 0,
            "graph_source_unique_destinations": 0,
            "graph_destination_unique_sources": 0,
            "graph_edge_flow_count": 0,
        },
        index=range(len(data)),
        dtype="int64",
    )

    if valid.any():
        valid_data = data.loc[valid].copy()
        valid_data[timestamp_col] = parsed_timestamps.loc[valid]
        valid_data["__window"] = (
            valid_data[timestamp_col].dt.floor("1min")
        )

        for _, group in valid_data.groupby("__window", sort=True):
            graph = build_time_window_graphs(
                group,
                timestamp_col=timestamp_col,
                source_col=source_col,
                destination_col=destination_col,
            )

            # A group contains only one minute, so it should produce
            # at most one graph. Still handle an empty result safely.
            if not graph:
                continue

            current_graph = next(iter(graph.values()))

            in_degree = dict(current_graph.in_degree())
            out_degree = dict(current_graph.out_degree())

            unique_destinations = {
                node: len(set(current_graph.successors(node)))
                for node in current_graph.nodes
            }
            unique_sources = {
                node: len(set(current_graph.predecessors(node)))
                for node in current_graph.nodes
            }

            edge_counts = {
                (source, destination): attrs["flow_count"]
                for source, destination, attrs
                in current_graph.edges(data=True)
            }

            for row_id, source, destination in zip(
                group["__row_id"],
                group[source_col].astype(str).str.strip(),
                group[destination_col].astype(str).str.strip(),
            ):
                if source not in current_graph or destination not in current_graph:
                    continue

                features.loc[row_id] = [
                    out_degree.get(source, 0),
                    in_degree.get(destination, 0),
                    unique_destinations.get(source, 0),
                    unique_sources.get(destination, 0),
                    edge_counts.get((source, destination), 0),
                ]

    result = df.copy()
    result.columns = result.columns.astype(str).str.strip()

    feature_names = features.columns.tolist()
    for name in feature_names:
        result[name] = features[name].to_numpy()

    return result
