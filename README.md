# 🌱 Soil Saturated Hydraulic Conductivity (Ksat) Prediction

A reproducible machine learning pipeline to **predict saturated hydraulic conductivity (Ksat)** from processed soil property data.  
Includes **hyperparameter tuning, subset experiments, visualizations, interpretability analysis, and a deployed web application** for interactive predictions.

🔗 **Live Demo:** [Streamlit App](https://ksattest-cqjzbncryj9gavgmnuzkqr.streamlit.app/)

---

## 📌 Overview

**Ksat** is a key soil property for hydrologic modeling, irrigation planning, and soil science research.  
This project builds and evaluates a **Random Forest Regressor** (and optional LightGBM model) to estimate Ksat from soil characteristics.

### Key Features:
- 📊 **Hyperparameter tuning** on the full dataset  
- 🔁 **Subset experiments** on progressively smaller training sizes  
- 📈 **Visualizations** of RMSLE and R² performance across trials  
- 🧠 **Model interpretability** via permutation importance & partial dependence plots  
- 💾 **Model saving** for reproducible predictions  
- 🌐 **Streamlit dashboard** for interactive use

---

## 📁 Project Structure

```
KSAT_QUEST_REGRESSION_RUNOFF_NERDY_POTATOES/
│
├── data/
│ ├── cleaned_data.xlsx # Cleaned raw data (intermediate)
│ ├── data.xlsx # Original raw data
│ └── preprocessed_data.xlsx # Final data used for modeling
│
├── outputs/
│ ├── best_rf_model.joblib # Trained Random Forest model
│ ├── rf_r2_plot.png # R² vs training sample size
│ ├── rf_rmsle_plot.png # RMSLE vs training sample size
│ └── rf_test_predictions.csv # Predictions from best-performing model
│
├── src/
│ ├── data_cleaning.py # Data cleaning and preprocessing functions
│ ├── evaluate_rf_subsets.py # Model evaluation on various subset sizes
│ ├── feature_selection.py # (Optional) Feature selection logic
│ └── train_rf_model.py # Training and hyperparameter tuning
│
├── tests/ # Unit tests for reproducibility
│ ├── test_data.py
│ └── test_model.py
│
├── assets/ # Images for README
│ ├── app.png
│ ├── pred_vs_actual.png
│ ├── perm_importance.png
│ └── pdp_top_feature.png
│
├── .gitignore
├── app.py # Streamlit dashboard
├── main.py # Pipeline orchestration
├── requirements.txt # Python dependencies
├── Makefile # Quick-run commands
├── MODEL_CARD.md # Model documentation
└── README.md # This file

```

---

## 🚀 How to Use

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
make setup

```

### 2. 🔁 Run Full Pipeline (Data → Model)

```bash
python main.py
```

This script will:
- Load and clean raw Excel sheets
- Select relevant features
- Train a Random Forest model
- Evaluates performance on progressively smaller subsets

### 3. 🌐 Launch the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501) to use the interactive predictor.

---

## 📊 Model Evaluation

- **RMSLE and R² metrics** are calculated over 50 trials for each subset size.
- Results are saved to:
  - `rf_test_predictions.csv`
  - Plots: `rf_rmsle_plot.png`, `rf_r2_plot.png`.
  - Model: `best_rf_model`

---

## 🧠 Model Info

- **Model:** Random Forest Regressor
- **Target Variable:** `ksat_cm_hr`
- **Features:** Encoded soil characteristics + lab/field metadata
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 5-fold CV

---

## 🌐 Deployment
The model is deployed via Streamlit, allowing:

- Manual soil property entry via a clean frontend

- Instant Ksat predictions (cm/hr)

- Responsive interface optimized for usability

## 🗺️ Roadmap

-  Add uncertainty estimation via quantile regression forests
-  Experiment with LightGBM/XGBoost + Optuna tuning
-  Deploy Streamlit app to cloud
-  Add domain-shift evaluation across sites

---
