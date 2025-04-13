# 🌱 Soil Saturated Hydraulic Conductivity (Ksat) Prediction

This project uses machine learning to predict **saturated hydraulic conductivity (Ksat)** from soil characteristics. It includes data cleaning, feature selection, model training, and an interactive [Streamlit](https://streamlit.io) app for prediction.

---

## 📁 Project Structure

```
ksat-project/
├── app.py                  # Streamlit app
├── main.py                 # Pipeline runner (clean → feature → train)
├── src/
│   ├── data_cleaning.py    # Load & clean soil Excel sheets
│   ├── feature_selection.py # Feature selection + unit conversion
│   └── model_training.py   # LightGBM training + subset experiments
├── data/                   # Raw and cleaned data (in .gitignore)
├── ksat_model.joblib       # Trained LightGBM model
├── requirements.txt        # Project dependencies
├── .gitignore              # Git ignore rules
└── README.md               # You're reading it!
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
- Train a LightGBM model using progressively smaller subsets

### 3. 🌐 Launch the Streamlit App

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501) to use the interactive predictor.

---

## 📊 Model Evaluation

- **RMSLE and R² metrics** are calculated over 50 trials for each subset size.
- Results are saved to:
  - `subset_experiment_metrics.csv`
  - `subset_test_predictions.csv`
  - Plots: `rmsle_vs_sample_size.png`, `r2_vs_sample_size.png`, etc.

---

## 🧠 Model Info

- **Model:** LightGBM Regressor
- **Target Variable:** `ksat_cm_hr`
- **Features:** Encoded soil characteristics + lab/field metadata
- **Hyperparameter Tuning:** `RandomizedSearchCV` with 5-fold CV

---

## 📌 Notes

- The `data/` folder and large files are excluded via `.gitignore`
- The Streamlit app dynamically reads feature names from the trained model
- Make sure `ksat_model.joblib` is present before launching the app

---

## 🙋‍♀️ Author

Made by [Your Name] — feel free to contribute, report issues, or fork this repo!
