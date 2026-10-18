from pathlib import Path
import pandas as pd

file_path = Path("data/processed/flows_with_graph_features.csv")
chunk_size = 100_000

if not file_path.is_file():
    raise FileNotFoundError(f"Dataset not found: {file_path}")

total_rows = 0
invalid_timestamps = 0
min_time = None
max_time = None
label_counts = {}
window_counts = {}

print("Checking timestamps and labels...")

for chunk in pd.read_csv(
    file_path,
    usecols=["Timestamp", "Label"],
    chunksize=chunk_size,
    low_memory=False,
):
    total_rows += len(chunk)

    timestamps = pd.to_datetime(
        chunk["Timestamp"],
        format="mixed",
        errors="coerce",
    )

    invalid_timestamps += int(timestamps.isna().sum())

    valid_times = timestamps.dropna()
    if not valid_times.empty:
        chunk_min = valid_times.min()
        chunk_max = valid_times.max()

        min_time = chunk_min if min_time is None else min(min_time, chunk_min)
        max_time = chunk_max if max_time is None else max(max_time, chunk_max)

        windows = valid_times.dt.floor("min")
        counts = windows.value_counts()

        for window, count in counts.items():
            window_counts[window] = window_counts.get(window, 0) + int(count)

    labels = chunk["Label"].value_counts(dropna=False)
    for label, count in labels.items():
        label = str(label)
        label_counts[label] = label_counts.get(label, 0) + int(count)

print("\n========== TIMESTAMP DIAGNOSTICS ==========")
print(f"Total rows: {total_rows:,}")
print(f"Invalid timestamps: {invalid_timestamps:,}")
print(f"Earliest timestamp: {min_time}")
print(f"Latest timestamp:   {max_time}")
print(f"Unique one-minute windows: {len(window_counts):,}")

if window_counts:
    print(f"Largest window: {max(window_counts.values()):,} flows")
    print(f"Smallest window: {min(window_counts.values()):,} flows")

print("\n========== LABEL DISTRIBUTION ==========")
for label, count in sorted(label_counts.items(), key=lambda item: -item[1]):
    print(f"{label}: {count:,} ({count / total_rows:.2%})")

print("\n========== CHECK ==========")
if invalid_timestamps == 0:
    print("PASS: Every timestamp was parsed.")
else:
    print("WARNING: Invalid timestamps need investigation.")

if window_counts:
    print("Chronological splitting can be planned using whole time windows.")
