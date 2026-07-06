"""
Random Forest with multi-source aggregated features and group-aware evaluation.
Merges aggregated statistics from ping, throughput, and power measurement files.
"""
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
                            accuracy_score, 
                            precision_score, 
                            recall_score, 
                            f1_score, 
                            confusion_matrix
                            )
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

import sys
sys.path.append("..")
from src.multisource_features import build_multisource_matrix
from src.utils_eval import evaluate_dataset
from src.config import path_list

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_ESTIMATORS = 200  # more trees for small-sample stability

# Try multiple source combinations
COMBINATIONS = [
    ("ping",),
    ("ping", "throughput"),
    ("ping", "power"),
    ("ping", "throughput", "power"),
]

def train_evaluate(X, y, groups, combo_name, n_features_before):
    gss = GroupShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    g_train, g_test = groups.iloc[train_idx], groups.iloc[test_idx]

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE, n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = sorted(y.unique())

    return {
        "combination": combo_name,
        "n_samples": len(X),
        "n_features_before_cleanup": n_features_before,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_groups": len(set(g_train)),
        "test_groups": len(set(g_test)),
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_test, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(
            y_test, y_pred, labels=labels).tolist(),
        "class_labels": [str(l) for l in labels],
        "feature_columns": list(X.columns),
        "feature_importances": dict(
            zip(X.columns, clf.feature_importances_.tolist())),
    }


def main():
    os.makedirs(path_list["RESULTS_METRICS"], exist_ok=True)
    os.makedirs(path_list["RESULTS_FIGURES"], exist_ok=True)

    all_results = []

    for combo in COMBINATIONS:
        combo_name = "+".join(combo)
        print(f"\n{'='*60}")
        print(f"Combination: {combo_name}")
        print(f"{'='*60}")

        try:
            X, y, groups, all_cols, feature_cols = build_multisource_matrix(
                path_list["DATASET_DIR"], list(combo)
            )
        except Exception as e:
            print(f"  SKIPPED: {e}")
            continue

        try:
            evaluate_dataset(X,y)
        except RuntimeError as e:
            print(f" SKIPPED: {e}")
            continue

        result = train_evaluate(X, y, groups, combo_name, len(feature_cols))
        all_results.append(result)

        print(f"  Train: {result['n_train']}  |  Test: {result['n_test']}")
        print(f"  Accuracy:       {result['accuracy']:.4f}")
        print(f"  Precision (w):  {result['precision_weighted']:.4f}")
        print(f"  Recall (w):     {result['recall_weighted']:.4f}")
        print(f"  F1-score (w):   {result['f1_weighted']:.4f}")

    # --- Save results ---
    results_df = pd.DataFrame([{
        "combination": r["combination"],
        "n_samples": r["n_samples"],
        "n_features": r["n_features"],
        "n_train": r["n_train"],
        "n_test": r["n_test"],
        "accuracy": r["accuracy"],
        "precision_weighted": r["precision_weighted"],
        "recall_weighted": r["recall_weighted"],
        "f1_weighted": r["f1_weighted"],
    } for r in all_results])

    csv_path = os.path.join(path_list["RESULTS_METRICS"], "rf_multisource_group_split_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")
    print(results_df.to_string(index=False))

    json_path = os.path.join(path_list["RESULTS_METRICS"], "rf_multisource_group_split_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Best confusion matrix
    if all_results:
        best = max(all_results, key=lambda r: r["accuracy"])
        cm = best["confusion_matrix"]
        labels = best["class_labels"]
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set_title(
            f"Multi-Source RF — {best['combination']} "
            f"(Acc: {best['accuracy']:.4f}, N={best['n_samples']})"
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        fig.tight_layout()
        cm_path = os.path.join(path_list["RESULTS_FIGURES"], "rf_multisource_group_split_confusion_matrix.png")
        fig.savefig(cm_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {cm_path}")

        # Feature importance
        print(f"\nFeature importance ({best['combination']}, top 10):")
        fi = best["feature_importances"]
        for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"{feat:<40s}  {imp:.4f}")

    # Comparison with single-source group split (from previous experiment)
    prev_path = os.path.join(path_list["RESULTS_METRICS"], "rf_window_group_split_results.json")
    if os.path.exists(prev_path):
        with open(prev_path, "r") as f:
            prev_results = json.load(f)
        # Find best z from previous single-source experiment
        prev_best = max(prev_results, key=lambda r: r["accuracy"])
        print(f"\n--- Comparison with single-source (figure5 only, group split) ---")
        print(f"Best single-source (z={prev_best['z']}): "
              f"acc={prev_best['accuracy']:.4f}")
        for r in all_results:
            print(f"Multi-source ({r['combination']}):"
                  f"acc={r['accuracy']:.4f}  "
                  f"(N={r['n_samples']})")

if __name__ == "__main__":
    main()