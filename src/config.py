from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Columns with node information list ----
NODE_INFO_COLS = [
    "id", "run", "node_name", "location", "modem_name",
    "mcc", "country", "iso_code", "rat", "rat_name",
    "operator_anon",
]

# ---- Non-predictive metadata columns — excluded from features ----
EXCLUDED_COLS = [
    "id", "node_name", "modem_name", "location",
    "mcc", "country", "iso_code", "operator_anon",
    "rat", "rat_name", "timestamp", "target_ip",
    "timetamp_ms", "run",
]

# ---- Columns to be converted in numerical type ----
NUMERICAL_COL = [
    "id", "run", "mcc", "iso_code", "rat", "timestamp",
    "tot_size", "avg_speed", "tot_time", "diff",
    "voltage", "current"
]

# ---- Columns with measurement data list ----
MEASUREMENT_COLS = []

# ---- Target column for training models ----
TARGET_COL = "rat"

# ---- Filename → source label mapping ----
SOURCE_LABELS = {
    "figure4_latency.csv":         "_pck",
    "figure5_packet_loss.csv":     "_pck_loss",
    "figure6_throughput.csv":      "_throughput",
    "figure7_current.csv":         "_current",
    "figure8_power_upload.csv":    "_pw_upload",
    "figure8_power_idle.csv":      "_pw_idle",
}

# ---- List of columns required for grouping and sorting the windowed dataset ----
GROUP_COLS = [ "run", "node_name", "location", "modem_name",
                "mcc", "country", "iso_code", "rat"
            ]
SORT_COLS = ["timestamp", "timestamp_ms"]

path_list = {
    "RESULTS_METRICS": os.path.join(PROJECT_ROOT,"results/metrics"),
    "RESULTS_FIGURES": os.path.join(PROJECT_ROOT,"results/figures"),
    "DATASET_DIR": os.path.join(PROJECT_ROOT,"data/raw"),
    "CSV_FILES": os.path.join(PROJECT_ROOT,"data/raw/dataset.zip"),
    "ANALYSIS_DIR": os.path.join(PROJECT_ROOT,"data/analysis"),
    "FEATURES_DATASET_DIR": os.path.join(PROJECT_ROOT,"data/outcome_preprocess"),
    "FEATURES_DATASET_FILEPATH": "feature_dataset.csv",
    "EXPORTED_DIR": os.path.join(PROJECT_ROOT,"results/exported"),
    "EXPORTED_FEATURES_PATH": os.path.join(PROJECT_ROOT,"results/exported/aggregated_features.csv"),
    "NODE_INFO_FILE": os.path.join(PROJECT_ROOT,"data/node_info.csv"),
    "ONNX_NN_MODEL": os.path.join(PROJECT_ROOT,"exported_NN_ONNX.jpg"),
    "ENCODED_DATASET": os.path.join(PROJECT_ROOT,"data/outcome_preprocess/encoded.csv"),
    "NN_FIGURES": os.path.join(PROJECT_ROOT,"results/figures/classification_NN"),
    "Z_WINDOW_EVAL_NN": "outcome_windowed_features_NN.json",
    "Z_WINDOW_EVAL_RF": "outcome_windowed_features_RF.json",
    "RF_FIGURES": os.path.join(PROJECT_ROOT,"results/figures/classification_RF"),
    "EVAL_METRICS": os.path.join(PROJECT_ROOT, "results/metrics/performance_eval_results.json"),
    "METRIC_EVAL_FIG": os.path.join(PROJECT_ROOT,"results/figures/metric_eval/")
}
