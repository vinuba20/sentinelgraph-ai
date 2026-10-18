
"""Calculate graph-derived features for network flows."""

import networkx as nx
import pandas as pd


def add_graph_features(
    df: pd.DataFrame,
    timestamp_col: str = "Timestamp",
    source_col: str = "Source IP",
    destination_col: str = "Destination IP",
    window: str = "1min",
) -> pd.DataFrame:
    """Add communication-graph features to each network flow.

    Features are calculated from all valid flows in the same time window.
    Original row order, index, and columns are preserved, with graph
    feature columns appended.

    Invalid timestamps or missing/blank endpoints receive zero features.
    IP addresses are used internally and are not added as model features.
    """
    data = df.copy()
    data.columns = data.columns.astype(str).str.strip()

    required = {timestamp_col, source_col, destination_col}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    feature_names = [
        "graph_source_out_degree",
        "graph_destination_in_degree",
        "graph_source_unique_destinations",
        "graph_destination_unique_sources",
        "graph_edge_flow_count",
    ]

    # Use positional row IDs so duplicate DataFrame indices are safe.
    data["__row_id"] = range(len(data))

    parsed_timestamps = pd.to_datetime(
        data[timestamp_col], errors="coerce"
    )

    source_values = data[source_col].astype("string").str.strip()
    destination_values = data[destination_col].astype("string").str.strip()

    valid = (
        parsed_timestamps.notna()
        & source_values.notna()
        & destination_values.notna()
        & source_values.ne("")
        & destination_values.ne("")
    )

    features = pd.DataFrame(
        0,
        index=range(len(data)),
        columns=feature_names,
        dtype="int64",
    )

    if valid.any():
        valid_data = pd.DataFrame({
            "__row_id": data.loc[valid, "__row_id"].to_numpy(),
            "__timestamp": parsed_timestamps.loc[valid].to_numpy(),
            "__source": source_values.loc[valid].to_numpy(),
            "__destination": destination_values.loc[valid].to_numpy(),
        })

        valid_data["__window"] = valid_data["__timestamp"].dt.floor(window)

        for _, group in valid_data.groupby("__window", sort=True):
            graph = nx.DiGraph()

            # Aggregate repeated source-destination flows.
            edge_counts = group.groupby(
                ["__source", "__destination"], sort=False
            ).size()

            for (source, destination), count in edge_counts.items():
                graph.add_edge(
                    source,
                    destination,
                    flow_count=int(count),
                )

            in_degree = dict(graph.in_degree())
            out_degree = dict(graph.out_degree())

            unique_destinations = dict(graph.out_degree())
            unique_sources = dict(graph.in_degree())

            row_ids = group["__row_id"].to_numpy()

            for position, (_, row) in enumerate(group.iterrows()):
                source = row["__source"]
                destination = row["__destination"]

                features.loc[row_ids[position]] = [
                    out_degree.get(source, 0),
                    in_degree.get(destination, 0),
                    unique_destinations.get(source, 0),
                    unique_sources.get(destination, 0),
                    graph[source][destination]["flow_count"],
                ]

    result = df.copy()
    result.columns = result.columns.astype(str).str.strip()

    for name in feature_names:
        result[name] = features[name].to_numpy()

    return result
