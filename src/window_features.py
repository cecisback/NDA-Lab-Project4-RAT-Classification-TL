import pandas as pd
from tqdm import tqdm

from src.config import (GROUP_COLS, SORT_COLS)

# The following function takes in input an aggregated dataset with a multilevel index.
# Then, it takes all values of the selected column indexes on all of its levels and
# it concatenates those strings with an underscore to obtain the final value of the column index.
def modified_multilevel_col(df):
    df.columns = [
        "_".join(str(part) for part in col if str(part)).strip("_")
        for col in df.columns.values
    ]
    return df.reset_index()

def compute_windowed_dataset(dataset, z_value, measurement_cols):
    if z_value <= 0:
        raise ValueError("z_value must be a positive integer")

    cols = [col for col in measurement_cols if col in dataset.columns]
    sort_cols = [col for col in SORT_COLS if col in dataset.columns]
    group_cols = [col for col in GROUP_COLS if col in dataset.columns]
    ordered = dataset.sort_values(by=sort_cols, axis=1, ascending=True)

    windowed_dataset = []

    with tqdm(total=len(ordered), colour="green") as bar:
        start = 0
        while start < len(ordered):
            window = ordered.iloc[start : start + z_value]
            grouped_window = window.groupby(group_cols)[cols].agg(["min","max","std","mean"])
            aggregated_window = modified_multilevel_col(grouped_window)
            aggregated_window = aggregated_window.fillna(0.0)
            windowed_dataset.append(aggregated_window)
            start = start + z_value
            bar.update(len(window))

    if not windowed_dataset:
        raise RuntimeError("Some errors occurred while computing the windowed dataset")

    return pd.concat(windowed_dataset, ignore_index=True).fillna(0.0)
