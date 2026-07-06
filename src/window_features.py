"""
Time-window feature extraction for RAT classification.
Computes rolling-window statistics (mean, std, min, max) for measurement
columns within groups sorted by timestamp.
"""
import pandas as pd
import numpy as np
from tqdm import tqdm

import sys
sys.path.append("..")
from src.config import (GROUP_COLS, SORT_COLS)

"""
To flatten multiindex columns obtained with attributes aggregation in compute_statistics().

For each column and each of their indexes, we retrieve the corresponding values 
(eg. tot_size_throughput and min) and we concatenate them with the _ character as separator.
Then, we generate a new list of columns.
"""
def modified_multilevel_col(processed_dataset: pd.DataFrame) -> pd.DataFrame:
    processed_dataset.columns = ["_".join(col).strip() for col in processed_dataset.columns.values]
    return processed_dataset.reset_index()

"""
To compute statistic measurements on aggregated features.
Given the selected_X dataframe in input, corresponding to the set of rows extracted from a window of size z,
the following function extracts aggregated information from each group of data.
"""
def compute_aggregated_data(selected_X: pd.DataFrame, measurement_cols) -> pd.DataFrame:
    processed_file_content = selected_X.groupby(GROUP_COLS)[measurement_cols].agg(["min","max","std","mean"]).fillna(0.0)
    return modified_multilevel_col(processed_file_content)

def compute_windowed_dataset(dataset: pd.DataFrame, z_value: int, measurement_cols) -> pd.DataFrame:
    windowed_dataset = pd.DataFrame()
    last_value = z_value
    first_value = 0

    with tqdm(total = len(dataset), colour="green") as pbar:
        while first_value + z_value <= len(dataset):
            last_value = first_value + z_value
            selected_window = dataset.iloc[first_value:last_value,:]
            windowed_dataset = pd.concat([windowed_dataset, compute_aggregated_data(selected_window, measurement_cols)])
            first_value = last_value
            last_value += z_value
            pbar.update(z_value)

        if first_value < len(dataset):
            selected_window = dataset.iloc[first_value:,:]
            windowed_dataset = pd.concat([windowed_dataset, compute_aggregated_data(selected_window, measurement_cols)])
            pbar.update(z_value)
    return windowed_dataset

def compute_window_features(
    df: pd.DataFrame,
    measurement_cols: list,
    z: int,
) -> pd.DataFrame:
    """
    Compute rolling-window statistics for each measurement column.

    For each group defined by GROUP_COLS, rows are sorted by timestamp.
    A rolling window of size `z` (expanding for the first z-1 rows)
    computes mean, std, min, max.

    Returns a DataFrame with the same index as df containing only the
    engineered feature columns (target not included).
    """
    features_dataset = df.copy()
    SORTED_COL = [col for col in df.columns if any(value in col for value in SORT_COLS)]
    features_dataset = features_dataset.sort_values(SORTED_COL).reset_index(drop=True)

    feature_frames = []

    with tqdm(total = len(features_dataset), colour="green") as pbar:
        for _, grp in features_dataset.groupby(GROUP_COLS, sort=False):
            grp_features = pd.DataFrame(index=grp.index)

            for col in measurement_cols:
                series = grp[col]
                roll = series.rolling(window=z, min_periods=1)

                grp_features[f"{col}_mean"] = roll.mean().values
                grp_features[f"{col}_std"] = roll.std().values
                grp_features[f"{col}_min"] = roll.min().values
                grp_features[f"{col}_max"] = roll.max().values

            common_info = [col for col in grp.columns if col not in measurement_cols]
            grp_features = pd.concat([grp_features,features_dataset[common_info]], axis=1)

            feature_frames.append(grp_features)
            pbar.update(1)

    features = pd.concat(feature_frames)

    # Reorder to match df index order
    features = features.loc[features_dataset.index]

    # Fill NaN std values (occurs when z=1 or window has 1 row)
    features = features.fillna(0.0)

    return features
