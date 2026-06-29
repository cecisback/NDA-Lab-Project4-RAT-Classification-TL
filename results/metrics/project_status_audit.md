# Project Status Audit — RAT Classification with Transfer Learning

**Date**: 2026-06-12 | **Branch**: `main` | **Latest commits**: `27408ea` (audit), `be00c20` (eval format), `b3890a6` (NN added)

---

## 1. Repository Overview

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            6 CSV files (git-ignored, from Zenodo 10.5281/zenodo.15420422)
│   ├── utils/           data_inspection.py
│   └── data_preprocessing.ipynb
├── notebooks/
│   ├── utils/evaluation_perf.py     (new — shared evaluation utility)
│   ├── 02_baseline_random_forest.ipynb
│   ├── 03_window_feature_experiments.ipynb
│   ├── 04_multisource_rf_experiments.ipynb
│   └── 05_NN.ipynb                 (new — 2026-06-07)
├── src/
│   ├── baseline_random_forest.py
│   ├── window_features.py
│   ├── rf_window_experiments.py
│   ├── rf_window_group_split.py
│   ├── multisource_features.py
│   └── rf_multisource_experiments.py
├── results/
│   ├── figures/         8 PNGs (6 RF + 2 NN confusion matrices)
│   └── metrics/         6 JSON/CSV/TXT + data_structure_summary.md + this audit
└── slides/              empty (no presentation materials)
```

**Dataset**: 6 CSV files from MARKlab (TMA 2025). 5 RAT classes (2G, 3G, LTE CAT1, LTE-M, NB-IoT) across 4 countries (DE, HR, IT, NO). 4 anonymized operators.

**Target**: `rat` column (integer 0, 2, 7, 8, 9). `operator_anon` excluded from features per project requirements.

---

## 2. Completed Work

### 2.1 RF Baseline — `02_baseline_random_forest.ipynb` / `src/baseline_random_forest.py`

- **Data**: `figure5_packet_loss.csv` (95,016 rows)
- **Features**: 3 raw numeric columns — `icmp_seq`, `ttl`, `rtt_ms`
- **Split**: `train_test_split` (75/25, stratified, **random split** — NOT group-aware)
- **Model**: RandomForestClassifier (n_estimators=100)
- **Results**:

| Metric | Value |
|--------|-------|
| Accuracy | 0.6953 |
| Precision (weighted) | 0.6890 |
| Recall (weighted) | 0.6953 |
| F1 (weighted) | 0.6914 |

- **Status**: COMPLETE but results are optimistic (see Section 6.1)

### 2.2 Window Feature Experiments — `03_window_feature_experiments.ipynb` / `src/rf_window_experiments.py`

- **Approach**: Rolling-window statistics (mean, std, min, max) over z ∈ {1, 3, 5, 10}, computed within (country, node, modem, run) groups sorted by timestamp
- **Split**: `train_test_split` (75/25, stratified, **random split**)
- **Results (random split)**:

| z | Features | Accuracy | F1 (weighted) |
|---|----------|----------|---------------|
| 1 | 12 | 0.3695 | 0.3663 |
| 3 | 12 | 0.6326 | 0.6300 |
| 5 | 12 | 0.7758 | 0.7748 |
| 10 | 12 | **0.9108** | **0.9107** |

- **Status**: COMPLETE but results are optimistic (see Section 6.1)

### 2.3 Group-Aware Window Evaluation — `src/rf_window_group_split.py`

- **Approach**: Same window features, split by `GroupShuffleSplit` on (country, node, modem, run) groups — no run appears in both train and test
- **Split**: 9 train groups, 3 test groups, 0 overlap confirmed
- **Results (group split)**:

| z | Features | Accuracy | F1 (weighted) |
|---|----------|----------|---------------|
| 1 | 12 | 0.2245 | 0.2190 |
| 3 | 12 | 0.2307 | 0.2281 |
| 5 | 12 | 0.2334 | 0.2307 |
| 10 | 12 | 0.2244 | 0.2091 |

- **Includes side-by-side comparison** of random-split vs group-split accuracy
- **Status**: COMPLETE. Most honest RF evaluation to date.

### 2.4 Multisource RF Experiments — `04_multisource_rf_experiments.ipynb` / `src/rf_multisource_experiments.py`

- **Approach**: Aggregate each measurement file to (group, rat) level (mean/std/min/max), inner-join across sources, evaluate with GroupShuffleSplit
- **Sources**: ping (figure5), throughput (figure6), power (figure8_power_upload)
- **Results (group split)**:

| Combination | N | Features | Accuracy | F1 (weighted) |
|-------------|---|----------|----------|---------------|
| ping only | 30 | 11 | 0.5556 | 0.4593 |
| ping+throughput | 23 | 22 | **0.8750** | 0.8333 |
| ping+power | 23 | 29 | 0.7500 | 0.7500 |
| ping+throughput+power | 18 | 41 | 0.6667 | 0.5444 |

- **Status**: COMPLETE but N too small (see Section 6.2)

### 2.5 Neural Network — `05_NN.ipynb` (added 2026-06-07) + `notebooks/utils/evaluation_perf.py`

- **Approach**: sklearn `MLPClassifier` on a **pre-processed merged dataset** (`data/processed_dataset/features_dataset.csv`)
- **Data pipeline** (completely separate from RF pipeline):
  - Merged throughput (figure6) + power_idle (figure8) columns into 27M-row dataset
  - Groups by `(id, node_name, location, modem_name, mcc, country, iso_code, rat, direction_throughput)` and computes mean/std/min/max
  - Window sizes: z ∈ {1000, 10000, 100000} (applied sequentially across entire dataset, NOT per-group)
  - Drops `id, node_name, location, modem_name, country, mcc, iso_code` before training
  - Label-encodes categorical columns
- **Model**: MLPClassifier(hidden_layer_sizes=(5,5), activation='logistic', solver='adam', max_iter=200)
- **Split**: `train_test_split` (75/25, **random split** — NOT group-aware)
- **Result** (recorded for one z value):

| Metric | Value |
|--------|-------|
| Accuracy | 0.4872 |
| Precision (weighted) | 0.3571 |
| Recall (weighted) | 0.4872 |
| F1 (weighted) | 0.3878 |

- **Per-class precision**: 3G = 0.00, LTE-M = 0.00 — model completely fails on two classes
- **Status**: PRESENT but has **P1 issues** (see Section 6.3)

### 2.6 Data Preprocessing — `data/data_preprocessing.ipynb`

- Updated with NaN value analysis in latest remote commits
- **Status**: COMPLETE

---

## 3. NOT YET IMPLEMENTED

### 3.1 Transfer Learning — MISSING

- No transfer learning code, results, or experimental design exists.
- No cross-country source/target domain splitting logic.
- This is the **central theme of the project** ("RAT Classification with Transfer Learning") and is entirely absent.

### 3.2 Model Comparison (RF vs NN) — MISSING

- RF and NN use **completely different data pipelines** — not comparable:
  - RF: figure5_packet_loss.csv (ping only), 95k rows, per-group window features
  - NN: merged throughput+power_idle, 27M rows, global sequential windows
- No common evaluation framework exists

### 3.3 Presentation Materials — MISSING

- `slides/` directory is empty.

---

## 4. Existing Results Usable for Final Presentation

| Result | File | Usefulness | Caveats |
|--------|------|------------|---------|
| RF baseline (69.5%) | `baseline_rf_metrics.json` | Low | Random split — optimistic bias |
| Window z vs accuracy curve | `rf_window_accuracy_vs_z.png` | Medium | Shows 91% at z=10 with random split |
| Group vs random split comparison | `rf_window_group_split_results.json` | **High** | Demonstrates ~68pp optimistic bias — key methodological insight |
| Multisource ping+throughput (87.5%) | `rf_multisource_group_split_results.json` | Low-Med | N=23; test set = 8 samples |
| NN confusion matrices | `classification_NN_conf_matrix*.png` | Low | 2 classes have 0 precision, random split, different data pipeline |
| Data structure summary | `data_structure_summary.md` | High | Well-documented dataset understanding |

**Bottom line**: No result is currently presentation-ready. Group-split RF is near chance (22-23%). Multisource RF is honest but N=18-30. NN has class-level failures (3G and LTE-M = 0 precision) and uses random split on a different data pipeline.

---

## 5. Where Each Requirement Stands (UPDATED)

| Requirement | Status | Evidence |
|-------------|--------|----------|
| operator_anon excluded from features | Done | Confirmed in all RF source files; NN doesn't include it |
| RF baseline | Done | `02_baseline_random_forest.ipynb` |
| Window z experiments | Done | `03_window_feature_experiments.ipynb` + group split variant |
| Group-aware evaluation | Done | `src/rf_window_group_split.py` |
| Multisource RF | Done | `04_multisource_rf_experiments.ipynb` |
| Neural Network | **Done (P1 issues)** | `05_NN.ipynb` — exists but has critical problems |
| RF vs NN comparison | **Not started** | RF and NN use different data pipelines; no common framework |
| Transfer Learning (cross-country) | **Not started** | No TL code or results |
| Final presentation | **Not started** | `slides/` empty |

---

## 6. Identified Issues

### 6.1 CRITICAL: Random Split Optimistic Bias (RF)

**The single most important finding so far.**

| Evaluation | z=10 Accuracy | Gap |
|------------|---------------|-----|
| Random split | 0.9108 | — |
| Group split | 0.2244 | **-0.6864** |

With group-aware splitting, RF performs at near-chance level (~23% for 5 classes, chance = 20%). The 91% accuracy from random splitting is an artifact of temporal leakage: rows within the same (run, node, modem) group share the same RAT, and rolling windows cause rows from the same group to appear in both train and test.

**Implication**: ALL random-split results (RF baseline 69.5%, RF z=10 at 91%, NN at 48.7%) are invalid for reporting as generalization performance. They represent an upper bound on leakage, not model capability.

### 6.2 CRITICAL: Multisource Sample Size Too Small

Per-group aggregation collapses 95k+ rows into 18-30 rows. Test splits contain 6-9 samples. The 87.5% accuracy for ping+throughput is based on 8 test samples — directionally interesting but not statistically robust.

### 6.3 CRITICAL: Neural Network Has Multiple Design Issues

The NN notebook (`05_NN.ipynb`, added 2026-06-07) has the following problems:

1. **Random split, not group-aware**: Uses `train_test_split` instead of `GroupShuffleSplit`. Same leakage problem as early RF experiments. The 48.7% accuracy is likely inflated.

2. **Incomparable to RF**: RF uses figure5 (ping only), NN uses merged figure6+figure8 (throughput+power_idle). Different data, different feature extraction, different window sizes (z=1000/10000/100000 vs z=1/3/5/10). Cannot draw any RF-vs-NN conclusions.

3. **Suspicious 27M rows**: The processed dataset has 27 million rows but the largest raw file is only 335k rows. This suggests the merge produced mostly-zero rows (throughput columns set to 0 for power_idle-only rows and vice versa). Most power_idle columns are zero-filled. The model is learning from sparse/noisy features.

4. **3G and LTE-M completely failed**: 0.00 precision and 0.00 recall for classes 2 (3G) and 8 (LTE-M) — the model only predicts 2G, LTE CAT1, and NB-IoT. This is worse than random guessing for those classes.

5. **Windows applied globally, not per-group**: `compute_X_windowed` slices the dataframe sequentially (rows 0..z, z..2z, etc.) without respecting run/country/modem boundaries. This mixes data from unrelated measurement sessions in the same window.

6. **Includes country/iso_code/mcc as grouping attributes**: While dropped before training, the feature statistics are computed within groups that include country — meaning the aggregated features are country-specific, undermining cross-country generalization.

7. **Uses sklearn MLPClassifier, not PyTorch**: Despite `torch>=2.0` in requirements.txt. Simpler but less flexible for TL.

### 6.4 MODERATE: Only 12 Groups Available for RF Evaluation

GroupShuffleSplit uses (country, node_name, modem_name, run) as key. Only 12 unique groups in figure5. This prevents meaningful cross-validation (k-fold would have k ≤ 4).

### 6.5 MODERATE: Unused Data Assets

- `figure4_latency.csv` (201 MB) — never loaded for modeling
- `figure8_power_idle.csv` (1.85 GB) — used only in NN's merged dataset (as mostly-zero rows)
- `figure7_current.csv` — schema mismatch (has throughput columns, not current/voltage)

### 6.6 MODERATE: Single Split, No Cross-Validation

All experiments use a single train/test split. No variance estimates. Cannot distinguish signal from noise in small-N scenarios.

### 6.7 MODERATE: Two Disconnected Data Pipelines

```
RF pipeline:  figure5 (ping) → per-group window features (z=1-10) → RF
NN pipeline:  figure6+figure8 (throughput+power) → global window agg (z=1000+) → MLP
```

These cannot be compared. A unified evaluation framework is needed.

---

## 7. Recommendations — What To Do Next

### Immediate Priority (blocking for presentation)

1. **Unify the data pipeline** — Run both RF and NN on the SAME features with the SAME group-aware split. Either:
   - Extend NN to use the existing per-group window features (figure5, z=1-10), OR
   - Port the NN's merged-dataset features to RF for comparison
   - The current two-pipeline approach prevents any meaningful comparison.

2. **Fix NN evaluation** — Replace `train_test_split` with `GroupShuffleSplit` using the same group key as RF. This alone will likely drop accuracy below the reported 48.7%.

3. **Design and implement Transfer Learning** — Define source/target splits by country:
   - Leave-one-country-out: train on 3 countries, test on 1 held-out
   - Rotate held-out country for 4-fold evaluation
   - This directly addresses the "cross-country transfer" project goal

### Secondary Priority

4. **Increase group count** — Use `figure8_power_upload.csv` (334k rows) which likely has more unique groups, or relax grouping (e.g., drop `run` from group key)

5. **Combine window + multisource approaches** — Build per-group window features on EACH source file, then join. This preserves both temporal and cross-source information.

6. **Add cross-validation** — Replace single split with repeated splits or k-fold.

7. **Build presentation slides** — Once NN+TL results exist on a common pipeline.

---

## 8. Summary Table (UPDATED)

| Item | Status | Blocking? | Notes |
|------|--------|-----------|-------|
| Data exploration & understanding | Done | No | |
| Data preprocessing | Done | No | Updated with NaN analysis |
| RF baseline (random split) | Done | No | Results biased |
| Window feature experiments | Done | No | Random split results biased |
| Group-aware evaluation | Done | No | Key methodological insight |
| Multisource RF (group split) | Done | No | N too small (18-30) |
| Neural Network | **Done** | **No, but P1 issues** | Different pipeline, random split, 2 failed classes |
| Unified RF vs NN comparison | **Missing** | **Yes** | Pipelines are incompatible |
| Transfer Learning (cross-country) | **Missing** | **Yes** | Core project requirement |
| Final slides & presentation | **Missing** | **Yes** | |

**Overall assessment**: ~50% of required components exist in some form. The methodological insight about random-split bias is the most valuable finding so far. However: (1) RF and NN use incompatible data pipelines, preventing comparison; (2) Transfer Learning is entirely absent; (3) no result is presentation-ready with group-aware evaluation. The 4 remaining gaps (unified comparison, TL, cross-validation, slides) are all blocking.
