# 🌱 Soil Saturated Hydraulic Conductivity (Ksat) Prediction

This project predicts **saturated hydraulic conductivity (Ksat)** from preprocessed soil data using a **Random Forest Regressor**. It includes:

- Hyperparameter tuning on the full dataset
- Repeated experiments on progressively smaller random subsets
- Visualization of RMSLE and R² performance
- Model saving and prediction outputs

---

## 📁 Project Structure

```
KSAT_QUEST_REGRESSION_RUNOFF_NERDY_POTATOES/
│
├── data/
│   ├── cleaned_data.xlsx              # Cleaned raw data (intermediate)
│   ├── data.xlsx                      # Original raw data
│   └── preprocessed_data.xlsx         # Final data used for modeling
│
├── outputs/
│   ├── best_rf_model.joblib           # Trained Random Forest model
│   ├── rf_r2_plot.png                 # R² vs training sample size
│   ├── rf_rmsle_plot.png              # RMSLE vs training sample size
│   └── rf_test_predictions.csv        # Predictions from best-performing model
│
├── src/
│   ├── data_cleaning.py               # Data cleaning and preprocessing functions
│   ├── evaluate_rf_subsets.py         # Model evaluation on various subset sizes
│   ├── feature_selection.py           # (Optional) Feature selection logic
│   └── train_rf_model.py              # Training and hyperparameter tuning
│
├── .gitignore                         # Ignore models, __pycache__, etc.
├── app.py                             # (Optional) Flask/FastAPI app or Streamlit dashboard
├── main.py                            # Entry point (optional orchestration or CLI)
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies

```

---

## 🚀 How to Use

### 1. 📦 Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 🔁 Run Full Pipeline (Data → Model)

```bash
python main.py
```

This script will:
- Load and clean raw Excel sheets
- Select relevant features
- Train a Random Forest model using progressively smaller subsets

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

- **Model:** LightGBM Regressor
- **Target Variable:** `ksat_cm_hr`
- **Features:** Encoded soil characteristics + lab/field metadata
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 5-fold CV

---

## 😋 Deployment & Web App

- To make the model accessible and easy to use, we built a Streamlit-based web application that allows users to:

- Input soil properties manually through a clean frontend

- Run predictions on saturated hydraulic conductivity (Ksat) using our trained LightGBM model

- Instantly view the predicted Ksat (in cm/hr) value with a responsive and intuitive interface

## 📌 Notes

- The `data/` folder and large files are excluded via `.gitignore`
- The Streamlit app dynamically reads feature names from the trained model
- Make sure `ksat_model.joblib` is present before launching the app

---

## 🙋‍♀️ Author

Made by Nerdy Potatoes — feel free to contribute, report issues, or fork this repo!
