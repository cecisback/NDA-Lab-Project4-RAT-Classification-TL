import pandas as pd
from zipfile import ZipFile
import json
import os

import sys
sys.path.append("..")
from src.config import path_list

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

def load_raw(filepath: str) -> pd.DataFrame:
    """Read a raw measurement CSV and ensure timestamp is numeric."""
    df = pd.read_csv(filepath)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df

def upload_csv_files() -> pd.DataFrame:
    with ZipFile(path_list["CSV_FILES"],"r") as zDataset:
        zDataset.extractall(path=path_list["DATASET_DIR"])