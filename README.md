# Drought-GT: Analysis and Modeling of Drought in Guatemala

This project implements a **robust and reproducible pipeline** to analyze and model drought in Guatemala using monthly **climate** and **soil** information.  
It was originally developed as part of a Statistical Learning course, and has been restructured to showcase good practices in **Data Science engineering**, **feature engineering**, **modeling**, and **interpretability**.

---

## 🔍 Project Overview

- **Objective**: Build a data-driven model to classify drought events in Guatemala.
- **Definition of drought**: Based on the precipitation percentile (20th percentile) over a monthly climatology.
- **Data sources**: Climate reanalysis (temperature, humidity, winds, precipitation, etc.) and soil variables (elevation, land use, vegetation).
- **Target variable**: `sequia` (1 = drought, 0 = non-drought).

---

## ⚙️ Pipeline Structure

The project is organized as a **modular pipeline** controlled by a `Makefile`.  
Each step can be run independently or chained together:

```
make data          # Build processed dataset (merge climate + soil)
make blocks        # Add spatial blocks (grouping by ~50 km)
make features      # Add anomalies, lags, rolling features
make target        # Build drought target based on precipitation percentile
make eda           # Generate exploratory analysis figures/tables
make select_features  # Automatic feature selection (correlation filter, leakage removal)
make calibrate     # Train model with imbalance handling + probability calibration
make fit           # Fit final model and save artifacts
make importance    # Analyze permutation importance of trained model
```

---

## 📂 Repository Structure

```
├── config/
│   └── default.yaml        # Default configuration
├── data/
│   ├── raw/                # Raw input data (climate + soil)
│   ├── processed/          # Processed datasets (CSV)
│   └── external/           # (optional) external sources
├── models/
│   └── artifacts/          # Saved models, metadata, feature columns
├── reports/
│   ├── figures/            # EDA and importance plots
│   └── metrics/            # Evaluation metrics
├── scripts/
│   ├── make_dataset.py
│   ├── add_spatial_blocks.py
│   ├── features_add_basic.py
│   ├── build_target.py
│   ├── run_eda.py
│   ├── select_features.py
│   ├── train_baseline.py
│   ├── train_smote.py
│   ├── train_calibrated.py
│   ├── fit_full_model.py
│   ├── predict_cli.py
│   └── analyze_importance.py
├── .gitignore
├── Makefile
└── README.md
```
---

## 🧰 Features Implemented

- **Feature engineering**
  - Per-location z-score anomalies (`__z`)
  - Lags of climate variables (`__lag1..6`)
  - Past-only rolling means (`__roll_prev3m`, `__roll_prev6m`) → no leakage
- **Target construction**
  - Binary drought label from precipitation percentiles (20th by default)
- **Handling imbalance**
  - SMOTE oversampling integrated into the pipeline
- **Probability calibration**
  - Isotonic regression with inner CV
- **Cross-validation**
  - GroupKFold to avoid spatial leakage (`block_id`)
- **Interpretability**
  - Permutation importance (PR-AUC, ROC-AUC) with bar plots
- **Automation**
  - All steps reproducible through `make`

---

## 📊 Example Results

Calibrated model (logistic regression + SMOTE + isotonic calibration): 

```
=== Calibrated CV Summary (mean ± std) ===
pr_auc: 0.8685 ± 0.0124
roc_auc: 0.9561 ± 0.0050
f1: 0.7723 ± 0.0133
recall: 0.7073 ± 0.0173
precision: 0.8511 ± 0.0210
```

Top features by permutation importance (no leakage of raw precipitation): 
	- YEAR, MONTH (seasonality)
   - RH2M, RH2M__z (relative humidity)
   - T2M_RANGE__z (temperature range anomaly)
   - PRECTOTCORR__lag* and PRECTOTCORR__roll_prev* (past precipitation signals)
   - Secondary: temperature and humidity lags, pressure, specific humidity
   
---

## 🚀 Getting Started

### Requirements
- Python 3.9+
- Dependencies listed in `requirements.txt`:
  - pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, optuna

### Setup
```bash
git clone https://github.com/<your-username>/drought-gt.git
cd drought-gt
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the pipeline

```bash
make data
make blocks BLOCK_KM=50
make features DATE_COL=date FE_COLS="PRECTOTCORR T2M RH2M ..." LAGS="1 2 3 4 5 6" WINDOWS="3 6"
make target PERCENTILE=20
make eda
make select_features
make calibrate DATASET_TRAIN=data/processed/dataset_selected.csv
make fit DATASET_TRAIN=data/processed/dataset_selected.csv
make importance
```

## 📌 Future Work

	•	Explore advanced models (Random Forest, XGBoost, LightGBM, HistGradientBoosting)
	•	Add SHAP values for local interpretability
	•	Extend rolling features to 12 months (long-term drought)
	•	Separate true temporal holdout sets for validation
	•	Integrate additional soil/land-use variables

## ✍️ Author
Luis Tun

Data Scientist
