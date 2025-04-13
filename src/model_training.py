# ========================
# Part 1: Train and Evaluate Model with Subset Experiments
# ========================
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_log_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Load and preprocess dataset
df = pd.read_excel(r"./data/preprocessed_data.xlsx")
df.columns = [re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns]
df = df.loc[:, ~df.columns.duplicated()]
df = df.applymap(lambda x: int(x) if isinstance(x, bool) else x)

if "ksat_cm_hr" not in df.columns:
    raise ValueError("Dataset must include 'ksat_cm_hr' column.")

# Define subset sizes
subset_sizes = list(range(len(df), 2000, -2000)) + [2000]

# Hyperparameter grid for LightGBM
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [-1, 10, 20],
    'num_leaves': [31, 64],
    'learning_rate': [0.01, 0.05],
    'min_child_samples': [20, 50]
}

# Store results
rmsle_results = []
r2_results = []
accuracy_results = []
prediction_dfs = []
best_model_overall = None
best_r2_overall = -np.inf

for subset_size in subset_sizes:
    rmsle_scores = []
    r2_scores = []
    acc_scores = []

    for i in tqdm(range(50), desc=f"Subset size {subset_size}"):
        sample = df.sample(n=subset_size, random_state=np.random.randint(10000))
        X_sample = sample.drop(columns='ksat_cm_hr')
        y_sample = sample['ksat_cm_hr']

        X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.2, random_state=42)

        model = LGBMRegressor(random_state=42)
        search = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=5, cv=5, n_jobs=-1, scoring='r2', random_state=42)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        y_pred = best_model.predict(X_test)
        y_pred = np.maximum(y_pred, 0)

        rmsle = np.sqrt(mean_squared_log_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        acc = np.mean(cross_val_score(best_model, X_train, y_train, scoring='r2', cv=5))

        rmsle_scores.append(rmsle)
        r2_scores.append(r2)
        acc_scores.append(acc)

        if r2 > best_r2_overall:
            best_r2_overall = r2
            best_model_overall = best_model

        if i == 0:
            pred_df = pd.DataFrame({
                "Subset_Size": subset_size,
                "Actual_ksat_cm_hr": y_test.values,
                "Predicted_ksat_cm_hr": y_pred
            })
            prediction_dfs.append(pred_df)

    rmsle_results.append(np.mean(rmsle_scores))
    r2_results.append(np.mean(r2_scores))
    accuracy_results.append(np.mean(acc_scores))

# Save the best overall model
if best_model_overall is not None:
    joblib.dump(best_model_overall, "ksat_model.joblib")

# Combine predictions into a single DataFrame
all_predictions_df = pd.concat(prediction_dfs, ignore_index=True)
all_predictions_df.to_csv("subset_test_predictions.csv", index=False)

# Save performance metrics
results_df = pd.DataFrame({
    "Sample_Size": subset_sizes,
    "RMSLE": rmsle_results,
    "R2_Score": r2_results,
    "Accuracy": accuracy_results
})
results_df.to_csv("subset_experiment_metrics.csv", index=False)

# Plot RMSLE
plt.figure(figsize=(10, 6))
plt.plot(subset_sizes, rmsle_results, marker='o', label='RMSLE')
plt.xlabel("Training Sample Size")
plt.ylabel("RMSLE")
plt.title("RMSLE vs Training Sample Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("rmsle_vs_sample_size.png")
plt.show()

# Plot R²
plt.figure(figsize=(10, 6))
plt.plot(subset_sizes, r2_results, marker='s', color='green', label='R² Score')
plt.xlabel("Training Sample Size")
plt.ylabel("R² Score")
plt.title("R² Score vs Training Sample Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("r2_vs_sample_size.png")
plt.show()

# Plot Accuracy
plt.figure(figsize=(10, 6))
plt.plot(subset_sizes, accuracy_results, marker='^', color='orange', label='Cross-validated Accuracy (R²)')
plt.xlabel("Training Sample Size")
plt.ylabel("Accuracy (R² CV)")
plt.title("Accuracy vs Training Sample Size")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("accuracy_vs_sample_size.png")
plt.show()