# src/evaluate_existing_model.py
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_log_error

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PREPROC_PATH = DATA_DIR / "preprocessed_data.xlsx"
MODEL_JOBLIB = OUT_DIR / "best_rf_model.joblib"
MODEL_PKL = OUT_DIR / "best_rf_model.pkl"

PRED_CSV = OUT_DIR / "rf_test_predictions_eval.csv"
PRED_VS_ACT_PNG = OUT_DIR / "pred_vs_actual_eval.png"
METRICS_JSON = OUT_DIR / "metrics_eval.json"

TARGET = "ksat_cm_hr"

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def is_pipeline(model):
    return isinstance(model, Pipeline) or hasattr(model, "named_steps")

def main():
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"Missing {PREPROC_PATH}")

    # Data
    df = pd.read_excel(PREPROC_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found.")
    y = df[TARGET].astype(float)
    X = df.drop(columns=[TARGET]).copy()

    # Model
    model_path = MODEL_JOBLIB if MODEL_JOBLIB.exists() else MODEL_PKL
    if not model_path.exists():
        raise FileNotFoundError(f"No model found at {MODEL_JOBLIB} or {MODEL_PKL}")
    model = load(model_path)

    # Predict
    if is_pipeline(model):
        # If you saved a Pipeline, it includes preprocessing
        y_pred = model.predict(X)
    else:
        # Bare estimator: do quick preprocessing to allow predictions
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in X.columns if c not in num_cols]
        pre = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), num_cols),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]), cat_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        Xp = pre.fit_transform(X)  # quick fit for evaluation only
        y_pred = model.predict(Xp)

    # Metrics
    r2 = r2_score(y, y_pred)
    _rmsle = rmsle(y.values, y_pred)

    # Artifacts
    pd.DataFrame({"y_true": y.values, "y_pred": y_pred}).to_csv(PRED_CSV, index=False)
    with open(METRICS_JSON, "w") as f:
        json.dump({"r2": float(r2), "rmsle": float(_rmsle)}, f, indent=2)

    plt.figure()
    plt.scatter(y, y_pred, alpha=0.6)
    lims = [float(min(y.min(), y_pred.min())), float(max(y.max(), y_pred.max()))]
    plt.plot(lims, lims)
    plt.xlabel("Actual Ksat (cm/hr)")
    plt.ylabel("Predicted Ksat (cm/hr)")
    plt.title(f"Predicted vs Actual (Eval)  R²={r2:.3f}, RMSLE={_rmsle:.3f}")
    plt.savefig(PRED_VS_ACT_PNG, bbox_inches="tight")
    plt.close()

    print(f"✅ Eval done. R²={r2:.4f}, RMSLE={_rmsle:.4f}")
    print(f"📊 {PRED_CSV}")
    print(f"🖼️ {PRED_VS_ACT_PNG}")
    print(f"📄 {METRICS_JSON}")

if __name__ == "__main__":
    main()
