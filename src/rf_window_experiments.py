"""
Random Forest experiments across different time-window lengths z.
Trains and evaluates a model for each z, saves results and figures.
"""
import sys
sys.path.append("..")

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from src.window_features import compute_window_features
from src.config import TARGET_COL, path_list, SORT_COLS
from src.preprocessing import (
    load_raw, select_measurement_cols
)

RAW_DIR = "data/raw"
RESULTS_METRICS = "results/metrics"
RESULTS_FIGURES = "results/figures"
INPUT_FILE = os.path.join(path_list["DATASET_DIR"], "figure5_packet_loss.csv")

Z_VALUES = [1, 3, 5, 10]
RANDOM_STATE = 42
TEST_SIZE = 0.25


def train_evaluate(X, y, z):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = sorted(y.unique())
    label_names = [str(l) for l in labels]

    return {
        "z": z,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "class_labels": label_names,
        "feature_importances": dict(zip(X.columns, clf.feature_importances_.tolist())),
    }


def main():
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_FIGURES, exist_ok=True)

    df = load_raw(INPUT_FILE)
    print(f"Loaded: {INPUT_FILE}  ({df.shape[0]:,} rows)")

    measurement_cols = select_measurement_cols(df)
    print(f"Measurement columns ({len(measurement_cols)}): {measurement_cols}")

    y = df[TARGET_COL].copy()

    all_results = []

    for z in Z_VALUES:
        print(f"\n{'='*50}")
        print(f"z = {z}")
        print(f"{'='*50}")

        X = compute_window_features(df, measurement_cols, z)
        print(f"Features: {X.shape[1]} columns  (z={z})")

        result = train_evaluate(X, y, z)
        all_results.append(result)

        print(f"  Accuracy:       {result['accuracy']:.4f}")
        print(f"  Precision (w):  {result['precision_weighted']:.4f}")
        print(f"  Recall (w):     {result['recall_weighted']:.4f}")
        print(f"  F1-score (w):   {result['f1_weighted']:.4f}")

    # Save results CSV
    results_df = pd.DataFrame([{
        "z": r["z"],
        "n_features": r["n_features"],
        "accuracy": r["accuracy"],
        "precision_weighted": r["precision_weighted"],
        "recall_weighted": r["recall_weighted"],
        "f1_weighted": r["f1_weighted"],
    } for r in all_results])
    csv_path = os.path.join(path_list["RESULTS_METRICS"], "rf_window_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save detailed results JSON
    json_path = os.path.join(path_list["RESULTS_METRICS"], "rf_window_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Plot accuracy and F1 vs z
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(results_df["z"], results_df["accuracy"], "o-", color="tab:blue", label="Accuracy")
    ax1.plot(results_df["z"], results_df["f1_weighted"], "s--", color="tab:red", label="F1-score (w)")
    ax1.set_xlabel("Window size z")
    ax1.set_ylabel("Score")
    ax1.set_title("Random Forest — Accuracy and F1 vs. Time-Window Size z")
    ax1.set_xticks(Z_VALUES)
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    acc_fig = os.path.join(path_list["RESULTS_FIGURES"], "rf_window_accuracy_vs_z.png")
    fig.savefig(acc_fig, dpi=150)
    plt.close(fig)
    print(f"Saved: {acc_fig}")

    # Best-z confusion matrix
    best = max(all_results, key=lambda r: r["accuracy"])
    best_z = best["z"]
    print(f"\nBest z = {best_z}  (accuracy = {best['accuracy']:.4f})")

    cm = best["confusion_matrix"]
    labels = best["class_labels"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — z={best_z} (Accuracy: {best['accuracy']:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    cm_fig = os.path.join(path_list["RESULTS_FIGURES"], "rf_window_best_confusion_matrix.png")
    fig.savefig(cm_fig, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_fig}")

    # Feature importance for best z
    print(f"\nFeature importance (z={best_z}, top 10):")
    fi = best["feature_importances"]
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat:<30s}  {imp:.4f}")

    print(f"\n--- Summary ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
