# train_rf_model.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load dataset
df = pd.read_excel("preprocessed_data.xlsx")

# Encode categorical columns
categorical_cols = df.select_dtypes(include="object").columns
for col in categorical_cols:
    df[col] = df[col].astype(str)
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(columns="ksat_cm_hr")
y = df["ksat_cm_hr"]

# Hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf = RandomForestRegressor(random_state=42, n_jobs=-1)

print("🔍 Performing hyperparameter tuning...")
search = RandomizedSearchCV(
    rf,
    param_distributions=param_grid,
    n_iter=10,
    scoring='r2',
    cv=5,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
search.fit(X, y)

# Save best model
best_model = search.best_estimator_
joblib.dump(best_model, "best_rf_model.joblib")
print("✅ Saved best model as 'best_rf_model.joblib'")
print("📊 Best hyperparameters:", search.best_params_)
