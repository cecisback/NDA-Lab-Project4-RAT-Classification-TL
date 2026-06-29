"""
Multi-source feature aggregation.
Aggregates measurement statistics per (group, rat) from different CSV files
and joins them into a single feature matrix.
"""
import pandas as pd
import numpy as np
import sys
sys.path.append("..")

from src.config import(
                        GROUP_COLS,
                        TARGET_COL,
                        EXCLUDED_COLS
                    )


AGG_FUNCS = ["mean", "std", "min", "max"]

# File definitions: (filename, measurement type label)
SOURCE_FILES = {
    "ping": "figure5_packet_loss.csv",
    "throughput": "figure6_throughput.csv",
    "power": "figure8_power_upload.csv",
}

def get_measurement_cols(df: pd.DataFrame) -> list:
    """Numeric columns with variance that are not metadata."""
    cols = []
    for c in df.columns:
        if c in EXCLUDED_COLS:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() <= 1:
            continue
        cols.append(c)
    return cols


def aggregate_file(filepath: str, label: str) -> pd.DataFrame:
    """
    Aggregate a single measurement file by (group, rat).
    Computes mean, std, min, max for each numeric measurement column.
    Returns a DataFrame indexed by (group_key, rat) with prefixed columns.
    """
    df = pd.read_csv(filepath)
    meas_cols = get_measurement_cols(df)
    agg_dict = {col: AGG_FUNCS for col in meas_cols}
    grouped = df.groupby(GROUP_COLS + [TARGET_COL], sort=False)
    agg_df = grouped.agg(agg_dict)

    # Flatten MultiIndex columns: rtt_ms_mean, rtt_ms_std, ...
    agg_df.columns = ["_".join(c).strip("_") for c in agg_df.columns]
    
    # Prefix with source label
    agg_df = agg_df.add_prefix(f"{label}_")
    agg_df = agg_df.reset_index()
    return agg_df


def build_multisource_matrix(raw_dir: str, sources: list) -> pd.DataFrame:
    """
    Aggregate each source file and join them on (group, rat).
    `sources` is a list of keys from SOURCE_FILES to include.
    Returns (X, y, group_keys) as separate objects.
    """
    frames = []
    for key in sources:
        path = f"{raw_dir}/{SOURCE_FILES[key]}"
        agg = aggregate_file(path, key)
        frames.append(agg)

    # Inner join all frames on GROUP_COLS + TARGET_COL
    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=GROUP_COLS + [TARGET_COL], how="inner")

    y = merged[TARGET_COL].copy()
    group_key = merged[GROUP_COLS].astype(str).agg("-".join, axis=1)

    # Feature columns: everything except group cols and target
    feature_cols = [
        c for c in merged.columns
        if c not in GROUP_COLS + [TARGET_COL]
    ]
    X = merged[feature_cols].copy()

    # Drop zero-variance columns
    for c in X.columns:
        if X[c].nunique() <= 1:
            X = X.drop(columns=c)

    return X, y, group_key, list(merged.columns), feature_cols
