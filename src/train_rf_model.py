# src/train_rf_model.py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PREPROC_PATH = DATA_DIR / "preprocessed_data.xlsx"
MODEL_PATH = OUT_DIR / "best_rf_model.joblib"
METRICS_JSON = OUT_DIR / "metrics.json"
PRED_CSV = OUT_DIR / "rf_test_predictions.csv"

TARGET = "ksat_cm_hr"

def run():
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"Preprocessed data not found: {PREPROC_PATH}")

    df = pd.read_excel(PREPROC_PATH)

    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found in data columns.")

    # Split target / features
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).copy()

    # Identify types
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    # Impute numeric NaNs with median (avoid dropping to 0 rows)
    if num_cols:
        med = X[num_cols].median()
        X[num_cols] = X[num_cols].fillna(med)

    # Label-encode categoricals (fill NA -> 'NA' then label encode)
    for c in cat_cols:
        X[c] = X[c].astype(str).fillna("NA")
        le = LabelEncoder()
        X[c] = le.fit_transform(X[c])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Randomized search on RF
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        "n_estimators": [200, 400, 600, 800, 1000],
        "max_depth": [None, 8, 12, 16, 24, 32],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 4, 8],
        "max_features": ["sqrt", "log2", 0.5, 0.8],
    }

    search = RandomizedSearchCV(
        rf,
        param_distributions=param_grid,
        n_iter=40,
        cv=5,
        scoring="r2",
        random_state=42,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    # Eval
    y_pred = best.predict(X_test)
    r2 = float(np.corrcoef(y_test, y_pred)[0, 1] ** 2)  # quick R^2 (or use sklearn)
    # safer:
    from sklearn.metrics import r2_score, mean_squared_log_error
    r2 = float(r2_score(y_test, y_pred))
    rmsle = float(np.sqrt(mean_squared_log_error(np.clip(y_test, 0, None), np.clip(y_pred, 0, None))))

    # Save artifacts
    dump(best, MODEL_PATH)
    pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred}).to_csv(PRED_CSV, index=False)
    with open(METRICS_JSON, "w") as f:
        json.dump(
            {
                "r2": r2,
                "rmsle": rmsle,
                "best_params": search.best_params_,
                "train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
                "test_shape": [int(X_test.shape[0]), int(X_test.shape[1])],
                "num_features": num_cols,
                "cat_features": cat_cols,
            },
            f,
            indent=2,
        )

    print(f"✅ Trained RF. R²={r2:.4f}, RMSLE={rmsle:.4f}")
    print(f"💾 Saved: {MODEL_PATH}")
    print(f"📊 Predictions: {PRED_CSV}")
    print(f"🧾 Metrics: {METRICS_JSON}")

if __name__ == "__main__":
    run()
