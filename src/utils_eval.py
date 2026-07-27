"""
Shared evaluation utilities for RAT classification.
Used by RF, NN, and TL pipelines — no dependency on preprocessing.
"""
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
from sklearn.metrics import (
                        confusion_matrix,
                        accuracy_score,
                        precision_score,
                        f1_score,
                        recall_score,
                        )
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

from src.config import path_nested_folders, path_files, path_main_folders
from src.utils import store_json_content

def evaluate_dataset(X, y):
    """Return basic dataset statistics as a dict."""

    n_samples = len(X)
    n_classes = y.nunique()

    if n_samples < 10:
        raise RuntimeError(
            f"Too few samples ({n_samples}) for meaningful evaluation"
        )

    return {
        "Samples": n_samples,
        "Features": X.shape[1],
        "Classes": n_classes,
        "Class distribution": y.value_counts().sort_index().to_string(),
    }

def compute_classification_report(X_test, y_test, model, target, labels):
    y_pred = model.predict(X_test)
    
    classification_analysis = classification_report(y_test, y_pred, target_names=target, labels=labels)

    return classification_analysis

def compute_statistics(outcome_training_per_z, RAT_NAME):
    accuracy_scores = []

    for _, outcome_training in outcome_training_per_z.iterrows():
        accuracy, global_precision, global_recall, global_f1score = performance_eval(outcome_training["features_set"].copy(), 
                                                                                     outcome_training["y_test"].copy(), 
                                                                                     outcome_training["y_pred"].copy(), 
                                                                                     outcome_training["extracted_labels"].copy(),
                                                                                     RAT_NAME, 
                                                                                     outcome_training["z_value"], 
                                                                                     outcome_training["model_name"])
        accuracy_scores.append(accuracy)

        print("------------------------------------\n")
        print("Results for Z: {}\n".format(outcome_training["z_value"]))
        print("Training time[s]: {}\n".format(outcome_training["training_time"]))
        print("Accuracy: {}\n".format(accuracy))
        print("Global precision: {}\n".format(global_precision))
        print("Global recall: {}\n".format(global_recall))
        print("Global f1score: {}\n".format(global_f1score))
        print("------------------------------------\n")
    return accuracy_scores

# To draw both confusion matrix and confusion matrix normalized of notebooks "RF" and "NN"
# otherwise we should rearrange the root_path
def draw_cm(cm, labels, z_value, model_name, normalized):
    cm_png_filename = ""
    title = ""
    root_path = ""
    id_model = "RF" if model_name == "RandomForest" else "NN"

    root_path = path_nested_folders["RF_FIGURES"] if model_name == "RandomForest" else path_nested_folders["NN_FIGURES"] 

    os.makedirs(root_path, exist_ok=True)

    if normalized:
        title = "Normalized confusion matrix"
    else:
        title = "Confusion matrix"

    fig_n, ax_n = plt.subplots()
    im = ax_n.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_n.figure.colorbar(im, ax=ax_n)
    ax_n.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    fmt_n = ".2f"
    thresh_n = cm.max() / 2.0
    for w in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_n.text(
                j,
                w,
                format(cm[w, j], fmt_n),
                ha="center",
                va="center",
                color="white" if cm[w, j] > thresh_n else "black",
            )
    fig_n.tight_layout()

    if normalized:
        cm_png_filename = "classification_{}_conf_matrix_{}_normalized.png".format(id_model,z_value)
    else:
        cm_png_filename = "classification_{}_conf_matrix_{}.png".format(id_model,z_value)

    fig_n.savefig(os.path.join(root_path, cm_png_filename))
    plt.close(fig_n)

def performance_eval(
    X,
    y_true: pd.DataFrame,
    y_pred,
    lab: list,
    l_names: list,
    z_value: int,
    model_name: str
):

    """Full evaluation with metrics dict, saved report, and confusion matrix plots."""
    ds_eval = evaluate_dataset(X, y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=lab, average=None, zero_division=0)
    global_precision = precision_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, labels=lab, average=None, zero_division=0)
    global_recall = recall_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0)
    f1score = f1_score(y_true, y_pred, labels=lab, average=None, zero_division=0)
    global_f1score = f1_score(y_true, y_pred, labels=lab, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=lab, normalize=None)
    cm_norm = confusion_matrix(y_true, y_pred, labels=lab, normalize="true")
    mse_error = mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
    mae_error = mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')

    results = {
        "z_value": z_value,
        "model_name": model_name,
        "dataset_samples": ds_eval["Samples"],
        "dataset_features": int(ds_eval["Features"]),
        "dataset_classes": int(ds_eval["Classes"].iloc[0]),
        "accuracy": accuracy,
        "precision": list(precision),
        "global_precision": global_precision,
        "recall": list(recall),
        "global_recall": global_recall,
        "f1score": list(f1score),
        "global_f1score": global_f1score,
        "mse_error": mse_error,
        "mae_error": mae_error
    }


    # To store performance evaluation metrics
    results_path = path_files["EVAL_METRICS"]
    os.makedirs(path_main_folders["RESULT_METRICS"], exist_ok=True)

    draw_cm(cm, l_names, z_value, model_name, False)
    draw_cm(cm_norm, l_names, z_value, model_name, True)

    store_json_content(results_path, results)

    return accuracy, global_precision, global_recall, global_f1score
