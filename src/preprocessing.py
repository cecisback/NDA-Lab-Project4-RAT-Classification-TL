"""
Unified preprocessing pipeline for RAT classification.

Integrates data inspection, column renaming, NaN analysis, and per-group
time-window feature extraction into a single callable pipeline.

v1: single-source (figure5_packet_loss.csv only).
"""
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import math as m
import os

import sys
sys.path.append("..")

from src.data_inspection import list_raw_files
from src.utils import load_raw, store_json_content
from src.config import (path_list, 
                        NODE_INFO_COLS,
                        EXCLUDED_COLS, 
                        SOURCE_LABELS,
                        NUMERICAL_COL
                        )

# ---- Column utilities ----

def _filepath_to_label(filepath: str) -> str:
    fname = os.path.basename(filepath)
    return SOURCE_LABELS.get(fname, "")

def rename_columns(df: pd.DataFrame, source_label: str) -> pd.DataFrame:
    """
    Apply renaming rules from the original notebook:
      "total"   → "tot_"  (drop "_total" suffix)
      "average" → "avg_"  (drop "average_" prefix)
    Then append the source_label suffix to measurement columns.
    Column "id" is left unchanged.
    """
    df = df.copy()
    mapping = {}

    for col in df.columns:
        if col == "id":
            continue
        new_col = col
        if "total" in col:
            new_col = "tot_" + col.replace("_total", "")
        elif "average" in col:
            new_col = "avg_" + col.replace("average_", "")
        if new_col != col:
            mapping[col] = new_col

    df = df.rename(columns=mapping)

    # Append source suffix to measurement columns (not node_info or columns such as operatore_anon
    # dropped to fulfill the project requirements)
    RENAME_SKIP = set(NODE_INFO_COLS) | {"operator_anon"}
    renamed = {}
    for col in df.columns:
        if col in RENAME_SKIP:
            continue
        renamed[col] = col + source_label

    df = df.rename(columns=renamed)
    return df


# ---- NaN analysis ----

def nan_analysis(df: pd.DataFrame) -> tuple:
    """
    Compute per-column NaN statistics.
    Returns (analysis_df, high_nan_columns) where high_nan_columns are those
    whose NaN percentage exceeds the column-average loss rate.
    """
    analysis_df = pd.DataFrame()
    for col in df.columns:
        nan_count = int(len(df[df[col].isna()][col]))
        if nan_count == 0:
            continue
        total = len(df[col])
        pct = round(100.0 * nan_count / total, 2)

        new_data = {
            "name_col": col,
            "tot_row": total,
            "info_lost": nan_count,
            "valid_info": total - nan_count,
            "percentage_loss": pct,
        }

        analysis_df = pd.concat([analysis_df, pd.DataFrame(new_data, index=[0])], ignore_index=True)

    if analysis_df.empty:
        return analysis_df, []

    avg_loss = analysis_df["percentage_loss"].mean()

    print("Average loss computed: {}\n".format(round(avg_loss,2)))

    high_nan = analysis_df[analysis_df["percentage_loss"] > avg_loss]["name_col"].to_numpy()
    return analysis_df, high_nan


def save_nan_analysis(analysis_df: pd.DataFrame, output_dir: str = path_list["ANALYSIS_DIR"]):
    """Persist NaN analysis results to disk."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "nan_analysis.json")

    for _, row in analysis_df.iterrows():
        new_data = {
            "column_name": row["name_col"],
            "tot_row": row["tot_row"],
            "lost_info": row["info_lost"],
            "valid_info": row["valid_info"],
            "percentage_loss": row["percentage_loss"]
        }
        store_json_content(path, new_data)

        print("nan values analysis stored in {}\n".format(path))

# ---- Feature selection ----

def select_measurement_cols(df: pd.DataFrame) -> list:
    """Return numeric measurement columns with variance, excluding metadata."""
    cols = []
    for c in df.columns:
        if any(col in c for col in EXCLUDED_COLS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() <= 1:
            continue
        cols.append(c)
    return cols

def convert_to_numeric(dataset: pd.DataFrame) -> pd.DataFrame:
    to_be_converted = [col for col in dataset.columns if any(numeric_type_col in col for numeric_type_col in NUMERICAL_COL)]

    for col in to_be_converted:
        if col == "rat_name":
            continue
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0.0)
    return dataset

# ---- Extraction of node information ----

def compute_info_node():
    list_file = list_raw_files()
    meta_df= pd.DataFrame()
    
    for file in list_file:
        filepath = os.path.join(path_list["DATASET_DIR"],file)

        df = load_raw(filepath)
        available_meta = [c for c in NODE_INFO_COLS if c in df.columns]
        meta_df = pd.concat([meta_df,df[available_meta].copy()], ignore_index=True)

        # Drop NODE_INFO_COLS and keep just non metadata 
        drop_meta = [c for c in available_meta if c != "id"]
        df = df.drop(columns=drop_meta, errors="ignore")
        
    meta_df = meta_df.drop("operator_anon", axis=1)
    return meta_df

# ---- Generation of the original dataset ----

def compute_dataset():
    list_file = list_raw_files()
    
    non_meta_df = pd.DataFrame()

    meta_df = compute_info_node()
    meta_df = meta_df.drop_duplicates()
    meta_df.to_csv(path_list["NODE_INFO_FILE"], index=False)

    print("Generated metadata with fields: {}\n".format(meta_df.columns))

    for file in list_file:
        df = load_raw(os.path.join(path_list["DATASET_DIR"],file))

        # Set up of measurements columns containing more than one unique value
        MEANINGFUL_COLS = [col for col in df.columns if len(np.unique(df[col])) > 1 or col in NODE_INFO_COLS]
        df = df[MEANINGFUL_COLS]

        # Rename measurement columns
        source_label = _filepath_to_label(file)

        df = rename_columns(df, source_label)
        df.drop(columns=["operator_anon"], inplace=True, errors="ignore")

        if non_meta_df.empty:
            non_meta_df = meta_df.copy()

        common_col = [col for col in non_meta_df.columns if col in meta_df.columns]
 
        non_meta_df = non_meta_df.join(df.set_index(common_col).copy(), how="left", on=common_col)

    return meta_df,non_meta_df

# ---- Main pipeline entry point ----

def build_single_source_pipeline() -> tuple:
    """
    End-to-end single-source preprocessing.

    1. Compute common node information.
       This step is mandatory since, from all the original files located in the raw directory,
       we should retrieve a list of IDs and their corresponding source node - target IP
       information.

    2. Drop operator_anon (assignment requirement)
    3. Rename measurement columns with source suffix
    4. NaN analysis → drop columns with loss information rate above the average
    5. Fill remaining NaN with 0.0
    6. Convert all columns marked as NUMERICAL_COL to numerical type

    Returns (non_meta_df):
      non_meta_df — non metadata DataFrame (id, columns not in NODE_INFO_COLS)
    """
    meta_df, non_meta_df = compute_dataset()
    print("Generated non metadata with fields: {}\n".format(non_meta_df.columns))
    analysis_df, high_nan = nan_analysis(non_meta_df)
    print(analysis_df)

    # Persist NaN analysis
    save_nan_analysis(analysis_df)

    # Drop columns exceeding average NaN rate
    non_meta_df = non_meta_df.drop(columns=high_nan, errors="ignore")
    print(f"Dropped {len(high_nan)} columns with above-average NaN rate")

    # Fill remaining NaN
    non_meta_df = non_meta_df.fillna(0.0)
    non_meta_df = non_meta_df.drop_duplicates()

    return non_meta_df

"""
To encode values of type string within the dataset (such as modem_name, country...).
- It retrieves a list of columns containing data of type string and it selects, among them, only
the ones with some meaningful values;
- For each column, it computes a list of unique values obtained after the deletion of NaN 
and 0.0 values;
- It fits the LabelEncoder() with the found list of unique values;
- It transforms all non nan values within the intended column in their corresponding encoded label;
- It builds a dataframe with a list of mapped label-encoded value;
- 
"""
def encode_values(dataset):
    str_attr_list = [col for col in dataset.columns if not pd.api.types.is_numeric_dtype(dataset[col]) and any(str(value).strip() != str(0.0) or not m.isnan(float(value)) for value in dataset[col])]

    print("Column list with meaningful values of type string\n")
    print(str_attr_list)

    if len(str_attr_list) == 0:
        print("Feature dataset columns do not contain objects of type string\n")
        return dataset
    else:
        print("Encoding string objects within the feature dataset...\n")

    encoder = LabelEncoder()
    step_encoding = 1

    for attr in str_attr_list:
        map_encoding_labels = pd.DataFrame()

        content_column = dataset[attr].copy().to_numpy()
        
        meaninful_str_values = [values for values in content_column if str(values).strip() != str(0.0) or not m.isnan(float(values))]

        unique_str_labels = np.unique(meaninful_str_values)
        encoder = encoder.fit(unique_str_labels)
            
        encoded_values = encoder.transform(unique_str_labels)

        for label,encoded in zip(unique_str_labels, encoded_values):
            new_mapping = {
                str(attr): label,
                "encoded": encoded
            }
            map_encoding_labels = pd.concat([map_encoding_labels, pd.DataFrame(new_mapping, index=[0])], ignore_index=True)
        
        print("Outcome {} step encoding\n".format(step_encoding))
        print(map_encoding_labels)

        step_encoding += 1

        to_modify = dataset[attr].copy()
        dataset[attr] = pd.merge(to_modify, map_encoding_labels, on=attr, how="left")["encoded"]

        print("Column: {}, Number of rows with NaN value: {}\n".format(attr, len(dataset[dataset[attr] == 0.0][attr])))

    return dataset