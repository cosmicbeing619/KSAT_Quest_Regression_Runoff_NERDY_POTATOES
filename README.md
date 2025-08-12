# 🌱 Soil Saturated Hydraulic Conductivity (Ksat) Prediction

A reproducible ML pipeline to **predict saturated hydraulic conductivity (Ksat)** from soil property data.  
Includes **hyperparameter tuning, subset experiments, visualizations, interpretability,** and a **Streamlit** web app.

🔗 **Live Demo:** [Streamlit App](https://ksattest-cqjzbncryj9gavgmnuzkqr.streamlit.app/)  
👉 See the full **[Model Card](MODEL_CARD.md)** for data, training, metrics, and limitations.

---

## 📌 Overview
**Ksat** is key for hydrologic modeling, irrigation planning, and soil science.  
This project trains and evaluates a **Random Forest Regressor** (plus optional LightGBM baseline) to estimate Ksat.

**Highlights**
- 📊 Hyperparameter tuning with **RandomizedSearchCV (5-fold, R²)**
- 🔁 Subset experiments across shrinking training sizes
- 📈 Visualizations: R², RMSLE, Predicted vs Actual
- 🧠 Interpretability: permutation importance (optional)
- 🌐 Streamlit app for interactive predictions

---

## 📂 Project Structure

```
KSAT_QUEST_REGRESSION_RUNOFF_NERDY_POTATOES/
├─ data/
│ ├─ data.xlsx # Raw data
│ ├─ cleaned_data.xlsx # Cleaned (generated)
│ └─ preprocessed_data.xlsx # Modeling table (generated)
├─ outputs/
│ ├─ best_rf_model.joblib # Trained RF model
│ ├─ rf_test_predictions.csv # Test predictions
│ ├─ rf_r2_plot.png # R² vs training size
│ └─ rf_rmsle_plot.png # RMSLE vs training size
├─ assets/
│ ├─ pred_vs_actual.png
│ ├─ rf_r2_plot.png
│ └─ rf_rmsle_plot.png
├─ src/
│ ├─ data_cleaning.py
│ ├─ train_rf_model.py
│ ├─ evaluate_rf_subsets.py
│ └─ make_pred_vs_actual_plot.py
├─ tests/
│ ├─ test_data.py
│ └─ test_model.py
├─ app.py
├─ main.py
├─ MODEL_CARD.md
├─ requirements.txt
├─ Makefile
└─ README.md

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

### 4. 📊 Results

```bash
Held-out test (20% split)

R²: 0.936

RMSLE: 0.488 cm/hr

Figures

Predicted vs Actual

R² vs Training Size

RMSLE vs Training Size

Artifacts are saved in outputs/ and copied into assets/ for display.
```
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
