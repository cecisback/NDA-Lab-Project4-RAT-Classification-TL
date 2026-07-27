"""
Unified preprocessing pipeline for RAT classification.

Integrates data inspection, column renaming, NaN analysis, and per-group
time-window feature extraction into a single source pipeline.
"""

# Data were collected with a *distributed mobile measurement platform* called MARKlab.
# The whole dataset is organized into six csv files, each one with its own set of measurement data indexed by
# a unique identifier for the measurement session.
# 
# At first sight, three important aspects grabbed my attention:
# - all files share common fields, known as metadata, such as the identifier, a subindex within the measurement
#   session, the hardware node name performing the measurement, city and country where the measurement was conducted;
# - unlike the most convential datasets, which are particularly focused on packet loss and latency, it contains data
#   related to power measurements and data upload activity too;
# - there are unmatched indentifiers among pairs of csv files included in the original dataset. In other words, an 
#   identifier can optionally appear in multiple files even if it's not required;
# This introduces an additional complexity during the dataset preprocessing step.

# *Dataset evaluation and preprocessing* aims at generating a consistent dataset with meaningful data designed for 
# feature extraction and data analysis.
# We decided to build a *common source pipeline* that takes in input each raw csv file and concatenates all their fields 
# in a unique dataset, paying attention to:
# - preserve the content of each file and the overall number of measurement sessions;
# - remove unnecessary fields (like the operator_anon explicitly required by the project specifications);

# This means that, in the final dataset, missing information related to a given identifier and its correspondent field
# should be initialized with NaN.

# After executing the *concatenation step*, we faced a major issue related to the *huge size of the dataset*. 
# This would have resulted in:
# - additional time spent windowing the dataset;
# - additional complexity in features extraction;
# - increased training time for both classifiers;
# - reduced accuracy of predictions due to the presence of drift and trends;

# Furthermore, we introduced an *unknown amount of NaN elements* per each field that doesn't provide any useful 
# informational value to the content of the dataset.
# To leverage the impact of the generated dataset on the system performance, we implemented the following optimizations:
# - *analysis of NaN values*, to compute the percentage of NaN values per each field and drop columns whose percentage 
#   is higher than the average among the precomputed ones;
# - *quantification of a given field content variation*, to drop columns with the same value reported for each measurement
#   session;
# - *remotion of remaining NaN values* by filling each of them with a float value (e.g., 0.0);
# - *division of numerical from nonnumerical columns* to encode values of each nonnumerical column;
# The encoding process was correctly implemented through a LabelEncoder fitted with unique values retrieved 
# from each numerical column.
# We decided to manage each set of labels independently to guarantee the same division also among the encoded dataset.

# The *preprocessed dataset* is adopted as a starting point for computing the *windowed dataset*.
# There are no effective requirements on how to implement the feature set except for fields being ordered according to
# measurement session timestamps.
# The key aspects to reason about while defining a way to compute the windowed dataset are:
# - how to determine the window size? What are the most suitable values for retrieving meaningful set of data?
# - how to select grouped attributes?
# - how to deal with nonmetadata fields? 
# - what criteria should be used to select and aggregate nonmetadata columns?
# - which aggregation attributes we should compute for each grouped feature? 
# - how to compute, in the most efficient way, aggregated features?
# - how to optimize the final code implemented to further reduce the complexity of the algorithm executed?

# The choice of the most appropriate *window size* aims to reduce the time spent computing aggregated features.
# Due to the high number of rows within the dataset, we defined a range of window sizes from 1000 to 500000.
# To generalize performance metrics obtained in the subsequent steps of the classification pipeline and to evaluate
# the behaviour of each performance metric under z value variation, we retrained both classifiers multiple times.
# Each time with a new windowed dataset computed with a linearly increased window size.
# All fields within the original dataset are splitted in metadata and nonmetadata columns.

# *Feature set* was extracted from the windowed dataset with the following workflow:
# - we selected all columns containing the substring "timestamp" in their corresponding indices. 
#  This is a *preventive measure* required by the automation of the preprocessing step in the classification pipeline.
#  After the initial *data inspection* conducted on raw files, we don't know a priori how many timestamp references 
#  will be in the column indices of the final dataset.
#  Timestamps selection is performed regardless of the context in which they were collected.
#  In other words, this step of the features extraction doesn't distinguish between timestamps related to current 
#  consumption of the device during measurements, those related to power usage during idle times and those related to 
#  data upload activity.

# - While defining the set of *grouped attributes*, we selected *all metadata* with the exclusion of the *identifier field* 
# (unique for each measurement session).
# We didn't group data according to the *direction flow* (uni or bidirectional).
# Furthermore, we performed *feature-based* rather than *port-based classification*, so that all metadata fields
# are dropped from the windowed dataset.

# - we determined two distinct ways to compute aggregated features: the first one is based on the use of predefined 
# methods from the *pandas library* for *rolling* the content of each non metadata grouped feature, while the second
# one consists in the implementation of a *self-made algorithm* for the same purpose.

# My colleagues, in their experiments, tested and implemented an *alternative* way for extracting the feature set, 
# which consists in the *definition of a combination of attributes and keywords* to select a desired subset of information 
# from the original dataset (if there are indices of columns matching the preinitialized searching keys).
# I think that this process is efficient only in case of data inspection before preprocessing and the initialization 
# of searched attributes, otherwise it may happen that the feature dataset obtained would be empty.

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import math as m
import os

from src.utils import load_raw, store_json_content, list_raw_files
from src.config import (path_main_folders, 
                        path_files,
                        NODE_INFO_COLS,
                        EXCLUDE_COLS, 
                        SOURCE_LABELS,
                        NUMERICAL_COL_KEYWORDS
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

# To store the outcome of the NaN analysis in a .json format file
def save_nan_analysis(analysis_df: pd.DataFrame, output_dir: str = path_main_folders["ANALYSIS_DIR"]):
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

    print("NaN values analysis stored in {}\n".format(path))

# ---- Feature selection ----

def select_measurement_cols(df: pd.DataFrame) -> list:
    """Return numeric measurement columns with variance, excluding metadata."""
    cols = []
    for c in df.columns:
        if any(col in c and col != "id" for col in EXCLUDE_COLS):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() <= 1:
            continue
        cols.append(c)
    return cols

def convert_to_numeric(dataset: pd.DataFrame):
    to_be_converted = [col for col in dataset.columns if any(numeric_type_key in col for numeric_type_key in NUMERICAL_COL_KEYWORDS)]

    for col in to_be_converted:
        if col == "rat_name" or col == "direction":
            dataset[col].fillna(0.0)
            continue
        dataset[col] = pd.to_numeric(dataset[col], errors="coerce").fillna(0.0)
    return dataset

# ---- Extraction of node information ----

def compute_info_node():
    list_file = list_raw_files()
    meta_df= pd.DataFrame()
    
    for file in list_file:
        filepath = os.path.join(path_main_folders["DATASET_DIR"],file)

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
    meta_df.to_csv(path_files["NODE_INFO_FILE"], index=False)

    print("Generated metadata with fields: {}\n".format(meta_df.columns))

    for file in list_file:
        df = load_raw(os.path.join(path_main_folders["DATASET_DIR"],file))

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

# End-to-end single-source preprocessing.

#   1. Compute common node information.
#   This step is mandatory since, from all the original files located in the raw directory,
#   we should retrieve a list of IDs and their corresponding source node - target IP information.
#   2. Drop operator_anon (assignment requirement)
#   3. Rename measurement columns with source suffix
#   4. NaN analysis → drop columns with loss information rate above the average
#   5. Fill remaining NaN with 0.0
#   6. Convert all columns marked as NUMERICAL_COL to numerical type

#   Returns (non_meta_df):
#      non_meta_df — non metadata DataFrame (id, columns not in NODE_INFO_COLS)

def build_single_source_pipeline() -> tuple:
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
    non_meta_df = non_meta_df.drop_duplicates()
    non_meta_df = non_meta_df.fillna(0.0)

    return non_meta_df

# To encode values of type string within the given dataset (such as modem_name, country...).
# - It takes in input a dataset and the preselected non numerical columns containing values to be encoded;
# - It retrieves a list of columns with meaningful data of type string;
# - For each column, it computes a list of unique values obtained after the deletion of NaN and 0.0 values;
# - It fits the LabelEncoder() with the found list of unique values;
# - It transforms all non NaN values within the intended column in their corresponding encoded label;
# - It builds a dataframe with a list of mapped label-encoded value; 

def encode_values(dataset, nonnumerical_col):
    str_attr_list = [col for col in nonnumerical_col if any(str(value).strip() != str(0.0) or not m.isnan(float(value)) for value in dataset[col])]

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
        print("Outcome {} step encoding\n".format(step_encoding))

        map_encoding_labels = pd.DataFrame()
        content_column = dataset[attr].copy().to_numpy()
        
        meaningful_str_values = [str(values) for values in content_column if str(values) != str(0.0) or not m.isnan(float(values))]
            
        unique_str_labels = np.unique(meaningful_str_values)
        for i, value in enumerate(unique_str_labels):
            if str(value) == str(0.0):
                unique_str_labels = np.delete(unique_str_labels, i)
        
        encoder = encoder.fit(unique_str_labels)
        encoded_values = encoder.transform(unique_str_labels)

        for label,encoded in zip(unique_str_labels, encoded_values):
            new_mapping = {
                str(attr): str(label),
                "encoded": encoded
            }
            map_encoding_labels = pd.concat([map_encoding_labels, pd.DataFrame(new_mapping, index=[0])], ignore_index=True)
        
        print(map_encoding_labels)

        step_encoding += 1

        to_modify = dataset[attr].copy()
        outcome = pd.merge(to_modify, map_encoding_labels, on=attr, how="left")
        dataset[attr] = outcome["encoded"].fillna(0.0).astype(int)
        print("Column: {}, Number of rows with NaN value: {}\n".format(attr, len(dataset[dataset[attr] == 0.0][attr])))

    return dataset