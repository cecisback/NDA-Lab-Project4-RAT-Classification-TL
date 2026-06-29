"""
Baseline Random Forest classifier for RAT classification.
Uses a single measurement file with numeric measurement features only.
No time-window feature extraction, no neural network, no transfer learning.
"""
import sys
sys.path.append("..")
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
                            accuracy_score, 
                            precision_score, 
                            recall_score, 
                            f1_score, 
                            confusion_matrix
                            )
from src.config import (path_list, 
                        EXCLUDED_COLS, 
                        TARGET_COL)

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

INPUT_FILE = os.path.join(path_list["DATASET_DIR"], "figure5_packet_loss.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.25

def main():
    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded: {INPUT_FILE}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")

    # Identify numeric measurement columns (exclude all known metadata)
    feature_cols = [
        c for c in df.columns
        if c not in EXCLUDED_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    # Drop zero-variance columns
    feature_cols = [c for c in feature_cols if df[c].nunique() > 1]
    print(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = df[feature_cols].copy()
    y = df[TARGET_COL].copy()

    # Handle potential missing values
    if X.isnull().any().any():
        print(f"Dropping {X.isnull().any(axis=1).sum()} rows with missing values")
        mask = X.notnull().all(axis=1)
        X = X[mask]
        y = y[mask]

    print(f"Target distribution:\n{y.value_counts().to_string()}")
    print(f"Classes: {sorted(y.unique())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    clf = RandomForestClassifier(
        n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    labels = sorted(y.unique())
    label_names = [str(l) for l in labels]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n--- Results ---")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Precision (w):  {prec:.4f}")
    print(f"Recall (w):     {rec:.4f}")
    print(f"F1-score (w):   {f1:.4f}")

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print(f"\nConfusion Matrix:\n{pd.DataFrame(cm, index=label_names, columns=label_names)}")

    os.makedirs(path_list["RESULTS_METRICS"], exist_ok=True)
    os.makedirs(path_list["RESULTS_FIGURES"], exist_ok=True)

    # Save confusion matrix figure
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f"RAT Classification — Random Forest Baseline (Accuracy: {acc:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    fig_path = os.path.join(path_list["RESULTS_FIGURES"], "baseline_rf_confusion_matrix.png")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"\nSaved: {fig_path}")

    # Save metrics as JSON
    metrics = {
        "input_file": INPUT_FILE,
        "n_samples": len(df),
        "feature_columns": feature_cols,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted": round(float(rec), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": cm.tolist(),
        "class_labels": label_names,
    }
    metrics_path = os.path.join(path_list["RESULTS_METRICS"], "baseline_rf_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved: {metrics_path}")

    # Feature importance
    importance = pd.DataFrame({
        "feature": feature_cols, "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)
    print(f"\nFeature Importance:\n{importance.to_string(index=False)}")


if __name__ == "__main__":
    main()
