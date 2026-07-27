"""
The following section contains a set of useful functions adopted by multiple files of the project
to perform simple operations such as loading and storing precomputed results in a given format.
We widely choose the json format to keep data easily accessible for further comparisons and 
analysis. 
To reduce the size of the project folder, we decided to automatize the initial steps related to the extraction
of the original dataset from the previously downloaded zip file and the generation of the preprocessed dataset
that will be used for computing the windowed dataset.
The last two functions are indicated for the scope mentioned above; more precisely, the first one
reads sequentially all raw files, while the second one extracts them from the zip file.
"""

from zipfile import ZipFile
import pandas as pd
import json
import os

from src.config import path_files, path_main_folders


def load_json_content(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except:
        data = None
    
    return data

def store_json_content(path: str, new_data): 
    file_content = load_json_content(path)
    
    if file_content != None:
        file_content.append(new_data)
    else:
        file_content = [new_data]

    with open(path, "w", encoding="utf-8") as f:
        f.seek(0)
        json.dump(file_content, f, indent=3)


def list_raw_files():  
    if not os.path.exists(path_main_folders['DATASET_DIR']):
        raise RuntimeError("No data/raw directory found")
    
    list_file = os.listdir(path_main_folders['DATASET_DIR'])
    if any(file in path_files["CSV_FILES"] for file in list_file) and len(list_file) == 1:
        print("Downloading the .csv files...")
        upload_csv_files()

    files = [f for f in os.listdir(path_main_folders["DATASET_DIR"]) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {path_main_folders["DATASET_DIR"]}/. Download the dataset first."
        )
    return sorted(files)


def load_raw(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def upload_csv_files() -> pd.DataFrame:
    with ZipFile(path_files["CSV_FILES"],"r") as zDataset:
        zDataset.extractall(path=path_main_folders["DATASET_DIR"])