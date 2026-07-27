from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- Target column for training models ----
TARGET_COL = "rat"

# ---- List of columns required for grouping and sorting the windowed dataset ----
GROUP_COLS = [ 
            "run", 
            "node_name", 
            "location", 
            "modem_name",
            "mcc", 
            "country", 
            "iso_code", 
            TARGET_COL
        ]

SORT_COLS = ["timestamp", "timestamp_ms"]

# ---- Columns with node information list ----
NODE_INFO_COLS = GROUP_COLS + [
                                "id",
                                "rat_name",
                                "operator_anon",
                            ]

# ---- Non-predictive metadata columns — excluded from features ----
EXCLUDE_COLS = GROUP_COLS + [   
                                "operator_anon",
                                "time",
                                "ip",
                                "direction",
                            ]

# ---- Columns to be converted in numerical type ----
NUMERICAL_COL_KEYWORDS = [
                            "id", 
                            "run", 
                            "mcc", 
                            "iso_code", 
                            "rat", 
                            "time",
                            "size", 
                            "speed",
                            "diff",
                            "voltage"
                        ]

# ---- Columns with measurement data list ----
MEASUREMENT_COLS = []

# ---- Filename → source label mapping ----
SOURCE_LABELS = {
    "figure4_latency.csv":         "_pck",
    "figure5_packet_loss.csv":     "_pck_loss",
    "figure6_throughput.csv":      "_throughput",
    "figure7_current.csv":         "_current",
    "figure8_power_upload.csv":    "_pw_upload",
    "figure8_power_idle.csv":      "_pw_idle",
}

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

path_main_folders = {
    "RESULT_METRICS": os.path.join(PROJECT_ROOT,"results/metrics"),
    "RESULT_FIGURES": os.path.join(PROJECT_ROOT,"results/figures"),
    "DATASET_DIR": os.path.join(PROJECT_ROOT,"data/raw"),
    "ANALYSIS_DIR": os.path.join(PROJECT_ROOT,"data/analysis"),
    "FEATURES_DATASET_DIR": os.path.join(PROJECT_ROOT,"data/outcome_preprocess"),
    "EXPORTED_DIR": os.path.join(PROJECT_ROOT,"results/exported"),
}

path_nested_folders = {
    "METRIC_EVAL_FIG": os.path.join(path_main_folders["RESULT_FIGURES"], "metric_eval"),
    "RESULT_METRICS_RF_EXP": os.path.join(path_main_folders["RESULT_METRICS"],"rf_experiments_metrics"),
    "RESULT_FIGURES_RF_EXP": os.path.join(path_main_folders["RESULT_FIGURES"],"rf_experiments_figures"),
    "RESULT_METRICS_NN_EXP": os.path.join(path_main_folders["RESULT_METRICS"],"nn_experiments_metrics"),
    "RESULT_FIGURES_NN_EXP": os.path.join(path_main_folders["RESULT_FIGURES"],"nn_experiments_metrics"),
    "COMPARISON_METRICS_EXP": os.path.join(path_main_folders["RESULT_METRICS"],"comparison_experiments_metrics"),
    "COMPARISON_FIGURES_EXP": os.path.join(path_main_folders["RESULT_FIGURES"],"comparison_experiments_figures"),
    "NN_FIGURES": os.path.join(path_main_folders["RESULT_FIGURES"],"classification_NN"),
    "RF_FIGURES": os.path.join(path_main_folders["RESULT_FIGURES"],"classification_RF"),
}

path_files = {
    "CSV_FILES": os.path.join(path_main_folders["DATASET_DIR"],"dataset.zip"),
    "FEATURES_DATASET": os.path.join(path_main_folders["FEATURES_DATASET_DIR"],"feature_dataset.csv"),
    "NODE_INFO_FILE": os.path.join(PROJECT_ROOT,"data/node_info.csv"),
    "ONNX_NN_MODEL": os.path.join(PROJECT_ROOT,"exported_NN_ONNX.jpg"),
    "ENCODED_DATASET": os.path.join(path_main_folders["FEATURES_DATASET_DIR"],"encoded.csv"),
    "Z_WINDOW_EVAL_NN": os.path.join(path_main_folders["EXPORTED_DIR"],"outcome_windowed_features_NN.json"),
    "Z_WINDOW_EVAL_RF": os.path.join(path_main_folders["EXPORTED_DIR"],"outcome_windowed_features_RF.json"),
    "EVAL_METRICS": os.path.join(path_main_folders["RESULT_METRICS"], "performance_eval_results.json"),
    "EXP_COMPARISON_CSV": os.path.join(path_nested_folders["COMPARISON_METRICS_EXP"],"model_comparison_summary.csv"),
    "EXP_COMPARISON_JSON": os.path.join(path_nested_folders["COMPARISON_METRICS_EXP"],"model_comparison_summary.json"),
    "EXP_COMPARISON_PNG": os.path.join(path_nested_folders["COMPARISON_FIGURES_EXP"],"model_comparison_accuracy.png"),
}