# Project 4: What Network Am I Using? RAT Classification with Transfer Learning

Classify the Radio Access Technology (RAT) using mobile network measurement data.

## Dataset

The dataset contains crowdsourced mobile network measurements from multiple countries.

- **DOI**: [10.5281/zenodo.15420422](https://doi.org/10.5281/zenodo.15420422)
- Download and place raw CSV files under `data/raw/`.

## Project Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/            (git-ignored — place downloaded CSVs here)
│   └── processed/      (git-ignored)
├── notebooks/           Jupyter notebooks for exploration & visualisation
├── src/                 Python source modules (preprocessing, features, models, utils)
├── results/
│   ├── figures/         Plots and graphics
│   └── metrics/         Evaluation metrics (CSV / JSON)
└── slides/              Final presentation materials
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
