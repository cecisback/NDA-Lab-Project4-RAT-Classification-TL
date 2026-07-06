# Project 4: What Network Am I Using? RAT Classification with Transfer Learning

Classify the Radio Access Technology (RAT) using mobile network measurement data.

## Dataset

The dataset contains crowdsourced mobile network measurements from multiple countries.

- **DOI**: [10.5281/zenodo.15420422](https://doi.org/10.5281/zenodo.15420422)
- Download and place raw CSV files under `data/raw/`.

## Project Structure

```
.
| - .gitignore
| - README.md
| - requirements.txt
| - data/
	| - analysis/
		| - nan_analysis.txt
		| - outcome_windowed_features_NN.json
		| - outcome_windowed_features_RF.json
	| - outcome_preprocess/
		| - encoded.csv
		| - feature_dataset.csv
	| - raw/                                           (git-ignored — folder dataset.zip with .csv file)
		| - dataset.zip 
		| - *.csv
		| - README.md
	| - node_info.csv
| - notebooks/                                               Jupyter notebooks for exploration & visualisation
	| - .gitkeep
	| - 01_RandomForest_RAT_classification.ipynb
	| - 02_RandomForest_baseline.ipynb
	| - 03_window_feature_experiments.ipynb
	| - 04_multisource_rf_experiments.ipynb
	| - 05_NN_RAT_classification.ipynb
	| - 06_outcome_visualization.ipynb
| - results/
	| - exported/
		| - aggregated_features.csv
		| - MLPClassifier.onnx
		| - MLPClassifier.pkl
		| - MLPClassifier.png
	| - figures/                                             Plots and graphics
		| - classification_NN/
			| - classification_NN_conf_matrix_{z_value}_normalized.png
			| - classification_NN_conf_matrix_{z_value}.png
		| - classification_RF
			| - classification_NN_conf_matrix_{z_value}_normalized.png
			| - classification_NN_conf_matrix_{z_value}.png
		| - metric_eval
			| - {metric}_vs_z.png
		| - .gitkeep
		| - baseline_rf_confusion_matrix.png
		| - rf_multisource_group_split_confusion_matrix.png
		| - rf_window_group_split_accuracy_f1.png
		| - rf_window_accuracy_vs_z.png 
		| - rf_window_group_split_confusion_matrix_best.png
		| - rf_window_accuracy_vs_z.png  
		| - rf_window_best_confusion_matrix.png
	| - metrics/                                             Evaluation metrics (CSV / JSON)
		| - .gitkeep
		| - baseline_rf_metrics.json
		| - project_status_audit.md 
		| - rf_performance_eval_results.txt
		| - rf_window_results.csv
		| - data_structure_summary.md
		| - rf_multisource_group_split_results.csv
		| - rf_window_group_split_results.csv
		| - rf_window_results.json
		| - performance_eval_results.json 
		| - rf_multisource_group_split_results.json 
		| - rf_window_group_split_results.json
	| - slides/                                              Final presentation materials
	| - src/                                        Python source modules (preprocessing, features, models, utils)
		| - __init__.py
		| - utils.py
		| - data_inspection.py
		| - rf_multisource_experiments.py   
		| - utils_eval.py
		| - baseline_random_forest.py
		| - multisource_features.py
		| - rf_window_experiments.py
		| - utils_train_models.py
		| - config.py
		| - preprocessing.py
		| - rf_window_group_split.py 
		| - window_features.py
```

## Main Tasks

1. Dataset understanding & exploration
2. Data preprocessing
3. Feature extraction with different time-window lengths *z*
4. Random Forest baseline
5. Neural Network model
6. Model comparison
7. Transfer Learning across countries
8. Final slides and results

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
```
