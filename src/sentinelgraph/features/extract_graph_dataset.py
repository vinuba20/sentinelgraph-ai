
"""Extract graph features for the prepared CIC-IDS2017 dataset.

Uses SQLite to group unsorted network flows by one-minute windows.
Preserves original CSV row order and adds five graph-derived features.
"""

from collections import Counter
from pathlib import Path
import sqlite3

import networkx as nx
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "flows_cleaned.csv"
OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed"
    / "flows_with_graph_features.csv"
)
DATABASE_FILE = (
    PROJECT_ROOT / "data" / "interim" / "graph_extraction.sqlite"
)

CHUNK_SIZE = 25_000
EXPECTED_ROWS = 2_830_743

FEATURE_COLUMNS = [
    "graph_source_out_degree",
    "graph_destination_in_degree",
    "graph_source_unique_destinations",
    "graph_destination_unique_sources",
    "graph_edge_flow_count",
]

REQUIRED_COLUMNS = {
    "Timestamp",
    "Source IP",
    "Destination IP",
}


def calculate_window_features(
    rows: list[tuple[int, str | None, str | None]],
) -> list[tuple[int, int, int, int, int, int]]:
    """Calculate graph features for all flows in one time window."""
    edge_counts = Counter(
        (source, destination)
        for _, source, destination in rows
        if source and destination
    )

    graph = nx.DiGraph()

    for (source, destination), count in edge_counts.items():
        graph.add_edge(source, destination, flow_count=count)

    in_degree = dict(graph.in_degree())
    out_degree = dict(graph.out_degree())

    results = []

    for row_id, source, destination in rows:
        if (
            not source
            or not destination
            or not graph.has_edge(source, destination)
        ):
            results.append((row_id, 0, 0, 0, 0, 0))
            continue

        source_out = out_degree[source]
        destination_in = in_degree[destination]
        edge_flow_count = graph[source][destination]["flow_count"]

        results.append(
            (
                row_id,
                source_out,
                destination_in,
                source_out,
                destination_in,
                edge_flow_count,
            )
        )

    return results


def create_database(connection: sqlite3.Connection) -> None:
    """Create SQLite tables for flows and graph features."""
    connection.execute(
        """
        CREATE TABLE flows (
            row_id INTEGER PRIMARY KEY,
            window TEXT,
            source TEXT,
            destination TEXT
        )
        """
    )

    connection.execute(
        """
        CREATE TABLE graph_features (
            row_id INTEGER PRIMARY KEY,
            graph_source_out_degree INTEGER NOT NULL,
            graph_destination_in_degree INTEGER NOT NULL,
            graph_source_unique_destinations INTEGER NOT NULL,
            graph_destination_unique_sources INTEGER NOT NULL,
            graph_edge_flow_count INTEGER NOT NULL
        )
        """
    )


def load_flows(
    connection: sqlite3.Connection,
    input_file: Path = INPUT_FILE,
    expected_rows: int | None = EXPECTED_ROWS,
) -> int:
    """Load flow identifiers and graph fields into SQLite in chunks."""
    total_rows = 0
    invalid_timestamps = 0

    print("Stage 1/3: Loading flows into SQLite...")

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        missing = REQUIRED_COLUMNS - set(chunk.columns)
        if missing:
            raise ValueError(
                f"Input dataset is missing columns: {sorted(missing)}"
            )

        timestamps = pd.to_datetime(
            chunk["Timestamp"].astype("string").str.strip(),
            errors="coerce",
            format="mixed",
            dayfirst=True,
        )

        invalid_timestamps += int(timestamps.isna().sum())

        windows = (
            timestamps.dt.floor("min")
            .dt.strftime("%Y-%m-%dT%H:%M:%S")
            .astype(object)
        )
        windows = windows.where(timestamps.notna(), None)

        sources = chunk["Source IP"].astype("string").str.strip()
        destinations = (
            chunk["Destination IP"].astype("string").str.strip()
        )

        sources = sources.astype(object)
        destinations = destinations.astype(object)

        sources = sources.where(
            pd.notna(sources) & sources.ne(""), None
        )
        destinations = destinations.where(
            pd.notna(destinations) & destinations.ne(""), None
        )

        row_ids = range(total_rows, total_rows + len(chunk))

        records = zip(
            row_ids,
            windows.tolist(),
            sources.tolist(),
            destinations.tolist(),
        )

        connection.executemany(
            """
            INSERT INTO flows (row_id, window, source, destination)
            VALUES (?, ?, ?, ?)
            """,
            records,
        )

        total_rows += len(chunk)
        connection.commit()

        print(f"  Loaded {total_rows:,} rows")

    if invalid_timestamps:
        raise ValueError(
            f"Found {invalid_timestamps:,} invalid timestamps. "
            "Investigate them before graph extraction."
        )

    if expected_rows is not None and total_rows != expected_rows:
        raise ValueError(
            f"Expected {expected_rows:,} rows, found {total_rows:,}"
        )

    connection.execute(
        "CREATE INDEX idx_flows_window ON flows(window)"
    )
    connection.commit()

    return total_rows


def build_all_graph_features(
    connection: sqlite3.Connection,
) -> int:
    """Calculate graph features for each complete minute."""
    cursor = connection.execute(
        """
        SELECT window
        FROM flows
        WHERE window IS NOT NULL
        GROUP BY window
        ORDER BY window
        """
    )
    window_list = [row[0] for row in cursor.fetchall()]

    print(
        f"\nStage 2/3: Processing {len(window_list):,} time windows..."
    )

    processed_rows = 0

    for window_number, window in enumerate(window_list, start=1):
        rows = connection.execute(
            """
            SELECT row_id, source, destination
            FROM flows
            WHERE window = ?
            ORDER BY row_id
            """,
            (window,),
        ).fetchall()

        features = calculate_window_features(rows)

        connection.executemany(
            """
            INSERT INTO graph_features
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            features,
        )

        processed_rows += len(features)

        if window_number % 100 == 0 or window_number == len(window_list):
            connection.commit()
            print(
                f"  Windows: {window_number:,}/{len(window_list):,}; "
                f"flows processed: {processed_rows:,}"
            )

    # Rows without a valid timestamp receive zero-valued graph features.
    connection.execute(
        """
        INSERT INTO graph_features
        SELECT f.row_id, 0, 0, 0, 0, 0
        FROM flows AS f
        LEFT JOIN graph_features AS g ON f.row_id = g.row_id
        WHERE g.row_id IS NULL
        """
    )
    connection.commit()

    feature_count = connection.execute(
        "SELECT COUNT(*) FROM graph_features"
    ).fetchone()[0]

    flow_count = connection.execute(
        "SELECT COUNT(*) FROM flows"
    ).fetchone()[0]

    if feature_count != flow_count:
        raise ValueError(
            f"Feature rows ({feature_count:,}) do not match "
            f"input rows ({flow_count:,})."
        )

    return feature_count


def write_output(
    connection: sqlite3.Connection,
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
) -> int:
    """Append graph features while preserving original row order."""
    print("\nStage 3/3: Writing the feature-augmented CSV...")

    total_written = 0
    first_chunk = True

    for chunk in pd.read_csv(
        input_file,
        chunksize=CHUNK_SIZE,
        low_memory=False,
        encoding="utf-8",
    ):
        chunk.columns = chunk.columns.astype(str).str.strip()

        start_id = total_written
        end_id = start_id + len(chunk)

        feature_rows = connection.execute(
            """
            SELECT
                row_id,
                graph_source_out_degree,
                graph_destination_in_degree,
                graph_source_unique_destinations,
                graph_destination_unique_sources,
                graph_edge_flow_count
            FROM graph_features
            WHERE row_id >= ? AND row_id < ?
            ORDER BY row_id
            """,
            (start_id, end_id),
        ).fetchall()

        if len(feature_rows) != len(chunk):
            raise ValueError(
                f"Feature row mismatch for input rows "
                f"{start_id:,} to {end_id - 1:,}."
            )

        feature_df = pd.DataFrame(
            feature_rows,
            columns=["row_id", *FEATURE_COLUMNS],
        )

        for name in FEATURE_COLUMNS:
            chunk[name] = feature_df[name].to_numpy()

        chunk.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            index=False,
            header=first_chunk,
            encoding="utf-8",
        )

        first_chunk = False
        total_written = end_id
        print(f"  Written {total_written:,} rows")

    return total_written


def extract_graph_dataset(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    database_file: Path = DATABASE_FILE,
    expected_rows: int | None = EXPECTED_ROWS,
) -> None:
    """Run and validate the complete extraction pipeline."""
    input_file = Path(input_file)
    output_file = Path(output_file)
    database_file = Path(database_file)

    if not input_file.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    database_file.parent.mkdir(parents=True, exist_ok=True)

    # Prevent a previous partial run from being mistaken for a new result.
    for path in (database_file, output_file):
        if path.exists():
            path.unlink()

    connection = sqlite3.connect(database_file)

    try:
        create_database(connection)

        total_rows = load_flows(
            connection,
            input_file=input_file,
            expected_rows=expected_rows,
        )

        feature_rows = build_all_graph_features(connection)

        written_rows = write_output(
            connection,
            input_file=input_file,
            output_file=output_file,
        )

        if (
            written_rows != total_rows
            or feature_rows != total_rows
        ):
            raise ValueError(
                "Final row-count validation failed: "
                f"input={total_rows:,}, features={feature_rows:,}, "
                f"output={written_rows:,}"
            )

        print("\n" + "=" * 60)
        print("GRAPH FEATURE EXTRACTION COMPLETED")
        print("=" * 60)
        print(f"Input rows: {total_rows:,}")
        print(f"Feature rows: {feature_rows:,}")
        print(f"Output rows: {written_rows:,}")
        print(f"Output file: {output_file}")
        print(f"Temporary database: {database_file}")
        print("Row counts: PASS")

    finally:
        connection.close()


def main() -> None:
    """Run extraction on the full prepared dataset."""
    extract_graph_dataset()


if __name__ == "__main__":
    main()
