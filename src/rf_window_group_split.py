"""
Random Forest window-feature evaluation with group-aware splitting.
Uses GroupShuffleSplit so that rows from the same run never appear
in both train and test, preventing leakage through rolling windows.
"""
import sys
sys.path.append('..')
from src.window_features import compute_window_features
from src.preprocessing import (
                            load_raw, 
                            select_measurement_cols,
                            upload_csv_files
                            )
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
                            accuracy_score, 
                            precision_score, 
                            recall_score, 
                            f1_score, 
                            confusion_matrix
                            )
from src.config import (
                        GROUP_COLS,
                        TARGET_COL,
                        path_list
                        )

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import os

RESULTS_METRICS = "results/metrics"
RESULTS_FIGURES = "results/figures"
INPUT_FILE = os.path.join(path_list["DATASET_DIR"], "figure5_packet_loss.csv")

Z_VALUES = [1, 3, 5, 10]
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 1  # single train/test split via GroupShuffleSplit

def make_group_key(df: pd.DataFrame) -> pd.Series:
    """Create a single string group key from the grouping columns."""
    return df[GROUP_COLS].astype(str).agg("-".join, axis=1)


def train_evaluate_group_split(X, y, groups, z):
    gss = GroupShuffleSplit(
        n_splits=N_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    unique_train = set(groups.iloc[train_idx])
    unique_test = set(groups.iloc[test_idx])
    overlap = unique_train & unique_test
    print(f"  Train groups: {len(unique_train)}  |  Test groups: {len(unique_test)}  |  Overlap: {len(overlap)}")
    assert len(overlap) == 0, f"Group leakage detected! Overlap: {overlap}"

    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    labels = sorted(y.unique())

    return {
        "z": z,
        "n_features": X.shape[1],
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_groups": len(unique_train),
        "test_groups": len(unique_test),
        "group_overlap": len(overlap),
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
        "feature_importances": dict(
            zip(X.columns, clf.feature_importances_.tolist())),
    }

def main():
    os.makedirs(RESULTS_METRICS, exist_ok=True)
    os.makedirs(RESULTS_FIGURES, exist_ok=True)
    
    df = pd.DataFrame()

    try:
        df = load_raw(INPUT_FILE)
    except:
        print('Downloading the input files...')
        upload_csv_files()
        df = load_raw(INPUT_FILE)

    print(f"Loaded: {INPUT_FILE}  ({df.shape[0]:,} rows)")

    measurement_cols = select_measurement_cols(df)
    print(f"Measurement columns: {measurement_cols}")

    y = df[TARGET_COL].copy()
    groups = make_group_key(df)

    all_results = []

    for z in Z_VALUES:
        print(f"\n{'='*50}")
        print(f"z = {z} (GroupShuffleSplit)")
        print(f"{'='*50}")

        X = compute_window_features(df, measurement_cols, z)
        print(f"Features: {X.shape[1]} columns")

        result = train_evaluate_group_split(X, y, groups, z)
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
        "train_groups": r["train_groups"],
        "test_groups": r["test_groups"],
    } for r in all_results])
    csv_path = os.path.join(RESULTS_METRICS, "rf_window_group_split_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    # Save JSON
    json_path = os.path.join(RESULTS_METRICS, "rf_window_group_split_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    # Accuracy / F1 vs z plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(results_df["z"], results_df["accuracy"], "o-", color="tab:blue", label="Accuracy (group split)")
    ax.plot(results_df["z"], results_df["f1_weighted"], "s--", color="tab:red", label="F1-score (group split)")
    ax.set_xlabel("Window size z")
    ax.set_ylabel("Score")
    ax.set_title("Random Forest — Group-Shuffle-Split Evaluation")
    ax.set_xticks(Z_VALUES)
    ax.legend()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig_path = os.path.join(RESULTS_FIGURES, "rf_window_group_split_accuracy_f1.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {fig_path}")

    # Best-z confusion matrix
    best = max(all_results, key=lambda r: r["accuracy"])
    best_z = best["z"]
    print(f"\nBest z = {best_z}  (accuracy = {best['accuracy']:.4f})")

    cm = best["confusion_matrix"]
    labels = best["class_labels"]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(f"Confusion Matrix — Group Split, z={best_z} (Acc: {best['accuracy']:.4f})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    cm_path = os.path.join(RESULTS_FIGURES, "rf_window_group_split_confusion_matrix_best.png")
    fig.savefig(cm_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {cm_path}")

    # --- Comparison: random vs group split ---
    # Load previous random-split results
    random_path = os.path.join(RESULTS_METRICS, "rf_window_results.json")
    random_results = {}
    if os.path.exists(random_path):
        with open(random_path, "r") as f:
            random_results = {r["z"]: r for r in json.load(f)}

    print(f"\n{'='*60}")
    print(f"Comparison: Random Split vs Group Split")
    print(f"{'='*60}")
    print(f"{'z':>3s}  {'Random Acc':>10s}  {'Group Acc':>10s}  {'Diff':>8s}  {'Note'}")
    print(f"{'-'*3}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*20}")
    for r in all_results:
        z = r["z"]
        rand_acc = random_results.get(z, {}).get("accuracy", None)
        grp_acc = r["accuracy"]
        if rand_acc is not None:
            diff = grp_acc - rand_acc
            note = "OPTIMISTIC BIAS" if abs(diff) > 0.03 else "comparable"
            print(f"{z:3d}  {rand_acc:10.4f}  {grp_acc:10.4f}  {diff:+8.4f}  {note}")
        else:
            print(f"{z:3d}  {'N/A':>10s}  {grp_acc:10.4f}")

    # Feature importance for best z (group split)
    print(f"\nFeature importance (group split, z={best_z}, top 10):")
    fi = best["feature_importances"]
    for feat, imp in sorted(fi.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {feat:<30s}  {imp:.4f}")

    print(f"\n--- Group Split Summary ---")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
