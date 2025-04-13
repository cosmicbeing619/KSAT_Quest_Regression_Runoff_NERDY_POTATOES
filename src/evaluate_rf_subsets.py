# evaluate_rf_subsets.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
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

# Load best model for hyperparameters
best_model = joblib.load("best_rf_model.joblib")
best_params = best_model.get_params()

# Subset sizes
subset_sizes = list(range(len(df), 2000, -2000)) + [2000]
rmsle_results = []
r2_results = []

# Track best subset performance
best_r2 = -np.inf
final_predictions_df = None

for subset_size in subset_sizes:
    rmsle_scores = []
    r2_scores = []

    for _ in tqdm(range(50), desc=f"Subset size {subset_size}"):
        sample = df.sample(n=subset_size, random_state=np.random.randint(10000))
        X_sample = sample.drop(columns='ksat_cm_hr')
        y_sample = sample['ksat_cm_hr']

        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)
        model = RandomForestRegressor(**best_params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        rmsle_scores.append(rmsle)
        r2_scores.append(r2)

        if r2 > best_r2:
            best_r2 = r2
            final_predictions_df = pd.DataFrame({
                "Actual_ksat_cm_hr": y_test.values,
                "Predicted_ksat_cm_hr": y_pred
            })

    rmsle_results.append(np.mean(rmsle_scores))
    r2_results.append(np.mean(r2_scores))

# Save best predictions
if final_predictions_df is not None:
    final_predictions_df.to_csv("rf_test_predictions.csv", index=False)
    print("📄 Saved best predictions to 'rf_test_predictions.csv'.")

# Plot RMSLE
plt.figure(figsize=(12, 6))
plt.plot(subset_sizes, rmsle_results, marker='o', label='RMSLE')
plt.xlabel('Training Sample Size')
plt.ylabel('RMSLE')
plt.title('RMSLE vs Training Sample Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rf_rmsle_plot.png")
plt.show()

# Plot R²
plt.figure(figsize=(12, 6))
plt.plot(subset_sizes, r2_results, marker='s', color='green', label='R² Score')
plt.xlabel('Training Sample Size')
plt.ylabel('R² Score')
plt.title('R² Score vs Training Sample Size')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rf_r2_plot.png")
plt.show()
