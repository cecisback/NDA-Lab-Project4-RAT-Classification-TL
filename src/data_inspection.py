"""
Dataset inspection — safe for large CSV files.
Reads only a sample of rows for schema inspection on files > 100 MB.
Run after placing downloaded CSV(s) in data/raw/.
"""
import os
import sys
import pandas as pd
from zipfile import ZipFile

sys.path.append("..")
from src.config import path_list
from src.utils import upload_csv_files

SAMPLE_ROWS = 10_000
LARGE_FILE_THRESHOLD_MB = 100

# Semantic grouping of columns based on dataset README
SEMANTIC_GROUPS = {
    "rat_label": ["rat", "rat_name"],
    "country": ["country", "iso_code", "mcc"],
    "operator": ["operator_anon"],
    "timestamp": ["timestamp", "timetamp_ms"],
    "node_device": ["node_name", "modem_name"],
    "session": ["id", "run"],
    "location": ["location"],
    "latency_rtt": ["rtt_ms", "ttl", "icmp_seq"],
    "packet_loss": ["icmp_seq"],
    "throughput": ["average_speed", "size_total", "time_total", "filesize", "timeout", "direction"],
    "current": ["current"],
    "power_energy": ["voltage", "current", "diff"],
}

def classify_column(col: str) -> str:
    for group, members in SEMANTIC_GROUPS.items():
        if col in members:
            return group
    return "other"

def list_raw_files():  
    if not os.path.exists(path_list['DATASET_DIR']):
        raise RuntimeError("No data/raw directory found")
    elif len(os.listdir(path_list['DATASET_DIR'])) == 1:
        print("Downloading the .csv files...")
        upload_csv_files()

    files = [f for f in os.listdir(path_list["DATASET_DIR"]) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {path_list["DATASET_DIR"]}/. Download the dataset first."
        )
    return sorted(files)


def inspect_file(filepath: str):
    if not os.path.exists(path_list['DATASET_DIR']):
        raise RuntimeError("No data/raw directory found")
    
    path = os.path.join(path_list["DATASET_DIR"], filepath)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    is_large = size_mb > LARGE_FILE_THRESHOLD_MB

    print(f"\n{"=" * 70}")
    print(f"File: {filepath}")
    print(f"Size: {size_mb:,.1f} MB")
    print(f"{"=" * 70}")

    if is_large:
        print(f"  [Large file — reading first {SAMPLE_ROWS:,} rows for schema inspection]")
        df = pd.read_csv(path, nrows=SAMPLE_ROWS)
        print(f"  Sampled shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        print(f"  (Full row count not measured to avoid loading {size_mb:.0f} MB into memory)")
    else:
        df = pd.read_csv(path)
        print(f"  Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")

    # Column names and dtypes
    print(f"\n  Columns ({len(df.columns)}):")
    for i, col in enumerate(df.columns):
        group = classify_column(col)
        print(f"[{i:2d}] {col:<20s}  dtype={str(df[col].dtype):<12s}  -> {group}")

    # Missing values in sampled rows
    print(f"\n  Missing values (sampled rows, top 15):")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("    (none)")
    else:
        for col, cnt in missing.head(15).items():
            pct = 100 * cnt / len(df)
            print(f"    {col:<20s}  {cnt:>10,}  ({pct:.1f}%)")

    # First 5 rows
    print(f"\n  First 5 rows:")
    print(df.head(5).to_string(max_colwidth=30))

    # Value counts for key categorical columns if present
    for col in ["rat_name", "rat", "country", "iso_code", "operator_anon", "direction"]:
        if col in df.columns:
            print(f"\n  Value counts — {col}:")
            counts = df[col].value_counts().head(20)
            for val, cnt in counts.items():
                print(f"    {str(val):<30s}  {cnt:>10,}")

    # Numeric summary for measurement columns
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    if len(numeric_cols) > 0:
        print(f"\n  Numeric column summary (sampled rows):")
        print(df[numeric_cols].describe().to_string())

    if is_large:
        # Read last 5 rows too, using a quick scan
        print(f"\n  Last 5 rows (from sampled data):")
        print(df.tail(5).to_string(max_colwidth=30))


def main():
    files = list_raw_files()
    print(f"Found {len(files)} CSV file(s) in {path_list["DATASET_DIR"]}/")
    for f in files:
        inspect_file(f)

    print(f"\n{"=" * 70}")
    print("Inspection complete — no models trained, no assumptions hardcoded.")

if __name__ == "__main__":
    main()
