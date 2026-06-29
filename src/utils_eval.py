"""
Shared evaluation utilities for RAT classification.
Used by RF, NN, and TL pipelines — no dependency on preprocessing.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
import sys
sys.path.append("..")
from src.config import path_list

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

def performance_eval(
    X,
    y_true,
    y_pred,
    lab,
    l_names,
    results_dir= path_list["RESULTS_METRICS"],
    figures_dir=path_list["RESULTS_FIGURES"],
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

    results = {
        "dataset_samples": ds_eval["Samples"],
        "dataset_features": ds_eval["Features"],
        "dataset_classes": ds_eval["Classes"],
        "dataset_class_distribution": ds_eval["Class distribution"],
        "accuracy": accuracy,
        "precision": precision,
        "global_precision": global_precision,
        "recall": recall,
        "global_recall": global_recall,
        "f1score": f1score,
        "global_f1score": global_f1score,
    }

    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    results_path = os.path.join(results_dir, "performance_eval_results.txt")
    with open(results_path, "a") as f:
        f.write("---------------------------------\n")
        f.write("OUTCOME PERFORMANCE EVALUATION\n")
        f.write(f"Dataset samples: {results["dataset_samples"]}\n")
        f.write(f"Dataset features: {results["dataset_features"]}\n")
        f.write(f"Dataset classes: {results["dataset_classes"]}\n")
        f.write(f"Dataset class distribution: {results["dataset_class_distribution"]}\n")
        f.write(f"Accuracy: {results["accuracy"]}\n")
        f.write(f"Precision: {results["precision"]}\n")
        f.write(f"Global precision: {results["global_precision"]}\n")
        f.write(f"Recall: {results["recall"]}\n")
        f.write(f"Global recall: {results["global_recall"]}\n")
        f.write(f"F1score: {results["f1score"]}\n")
        f.write(f"Global F1score: {results["global_f1score"]}\n")
        f.write("---------------------------------\n")

    # Confusion matrix
    title = "Confusion matrix"
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=l_names,
        yticklabels=l_names,
        title=title,
        ylabel="True label",
        xlabel="Predicted label",
    )
    fmt = "d"
    thresh = cm.max() / 2.0
    for w in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                w,
                format(cm[w, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[w, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(os.path.join(figures_dir, "classification_NN_conf_matrix.png"))
    plt.close(fig)

    # Normalized confusion matrix
    title_norm = "Normalized confusion matrix"
    fig_n, ax_n = plt.subplots()
    im = ax_n.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues)
    ax_n.figure.colorbar(im, ax=ax_n)
    ax_n.set(
        xticks=np.arange(cm_norm.shape[1]),
        yticks=np.arange(cm_norm.shape[0]),
        xticklabels=l_names,
        yticklabels=l_names,
        title=title_norm,
        ylabel="True label",
        xlabel="Predicted label",
    )
    fmt_n = ".2f"
    thresh_n = cm_norm.max() / 2.0
    for w in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            ax_n.text(
                j,
                w,
                format(cm_norm[w, j], fmt_n),
                ha="center",
                va="center",
                color="white" if cm_norm[w, j] > thresh_n else "black",
            )
    fig_n.tight_layout()
    fig_n.savefig(os.path.join(figures_dir, "classification_NN_conf_matrix_normalized.png"))
    plt.close(fig_n)

    return accuracy, global_precision, global_recall, global_f1score