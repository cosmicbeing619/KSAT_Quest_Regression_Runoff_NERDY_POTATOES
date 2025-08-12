# Model Card — KSAT (Soil Saturated Hydraulic Conductivity) Predictor

## Overview
- **Task:** Regression — predict saturated hydraulic conductivity (**Ksat**) from soil properties.
- **Audience:** Soil/hydrology students and practitioners needing quick, indicative Ksat estimates.
- **Intended use:** Educational & exploratory decision support. Not a replacement for lab/field measurements.

## Data
- **Source:** Consolidated Excel workbook (`data/data.xlsx`) with multiple “Ref #” sheets.
- **Preprocessing:**
  - Standardized column names across sheets; kept common fields.
  - Converted heterogeneous **units → cm/hr** (handles `mm/hr`, `in/hr`, `cm/s`, `m/s`, `cm/day`, `um/s`, several `log[...]` forms).
  - Converted **20,035 / 20,501** rows successfully; saved to:
    - `data/cleaned_data.xlsx`
    - `data/preprocessed_data.xlsx`
- **Target:** `ksat_cm_hr` (continuous).

## Features
Mixed numeric & categorical soil descriptors (e.g., texture class, method, field/lab flag, % sand/silt/clay, bulk density). Final columns depend on sheet intersection after cleaning.

## Model & Training
- **Algorithm:** RandomForestRegressor (scikit-learn).
- **Preprocessing:** `ColumnTransformer` — median imputation (numeric) + most-frequent imputation & One-Hot (categorical).
- **Tuning:** `RandomizedSearchCV` (5-fold CV, `scoring="r2"`, `n_iter=40`, `random_state=42`).
- **Artifacts:** `outputs/best_rf_model.joblib`, `outputs/metrics.json`, `outputs/rf_test_predictions.csv`.

## Evaluation (held-out test)
- **R²:** **0.9361**
- **RMSLE:** **0.4879** (cm/hr)
- **Method:** 80/20 train/test split (`random_state=42`).  
- **Visuals:**  
  - Predicted vs Actual → `assets/pred_vs_actual.png`  
  - Learning curves (subset study) → `assets/rf_r2_plot.png`, `assets/rf_rmsle_plot.png`

## Subset Study (Data Efficiency)
- Repeated training across decreasing training fractions (1.0 → 0.1).
- Trend: R² drops and RMSLE rises as data shrinks; stability improves at larger sample sizes.

## Limitations & Risks
- **Domain shift:** Sites/protocols unlike training data may underperform.
- **Unit conversion:** Rows with ambiguous log units were dropped; assumes successful normalization to cm/hr.
- **Not physics-based:** Use with expert judgment; validate in high-stakes decisions.

## Responsible Use
- Treat predictions as indicative. Confirm with field/lab measurements for planning, permitting, or safety-critical use.
- Respect data licenses and cite sources when redistributing.

## Reproducibility
- **Run all:** `python main.py`
- **Environment:** see `requirements.txt`
- **Determinism:** `random_state=42` for splits & search
- **Metrics/params:** `outputs/metrics.json` (includes best params and shapes)

## Versioning
- **Model file:** `outputs/best_rf_model.joblib`
- **Last trained:** _update date on retrain_
- **Changelog:** Document data/feature/hyperparameter changes here.

## Demo
- Streamlit app: https://ksattest-cqjzbncryj9gavgmnuzkqr.streamlit.app/

