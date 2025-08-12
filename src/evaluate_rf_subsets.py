# src/evaluate_rf_subsets.py
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_log_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

PREPROC_PATH = DATA_DIR / "preprocessed_data.xlsx"
R2_PNG = OUT_DIR / "rf_r2_plot.png"
RMSLE_PNG = OUT_DIR / "rf_rmsle_plot.png"

TARGET = "ksat_cm_hr"

def rmsle(y_true, y_pred):
    y_pred = np.clip(y_pred, 0, None)
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

def build_pipeline(num_cols, cat_cols):
    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )
    model = RandomForestRegressor(
        n_estimators=400, max_depth=None, min_samples_split=2,
        min_samples_leaf=1, max_features="sqrt", n_jobs=-1, random_state=42
    )
    return Pipeline([("prep", pre), ("model", model)])

def run(fracs=(1.0, 0.75, 0.5, 0.25, 0.1), trials=30, base_seed=42):
    if not PREPROC_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessed file: {PREPROC_PATH}")

    df = pd.read_excel(PREPROC_PATH)
    if TARGET not in df.columns:
        raise ValueError(f"Target '{TARGET}' not found.")

    y_full = df[TARGET].astype(float)
    X_full = df.drop(columns=[TARGET]).copy()

    num_cols = X_full.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X_full.columns if c not in num_cols]

    r2_means, rmsle_means, sizes = [], [], []

    rng = np.random.RandomState(base_seed)

    for frac in fracs:
        r2s, rmsles = [], []
        for _ in range(trials):
            # sample subset for this trial
            X_s = X_full.sample(frac=frac, random_state=rng.randint(0, 10_000))
            y_s = y_full.loc[X_s.index]

            X_tr, X_te, y_tr, y_te = train_test_split(
                X_s, y_s, test_size=0.2, random_state=42
            )

            pipe = build_pipeline(num_cols, cat_cols)
            pipe.fit(X_tr, y_tr)
            y_hat = pipe.predict(X_te)

            r2s.append(r2_score(y_te, y_hat))
            rmsles.append(rmsle(y_te.values, y_hat))

        r2_means.append(float(np.mean(r2s)))
        rmsle_means.append(float(np.mean(rmsles)))
        sizes.append(int(len(X_full) * frac))

    # R² plot
    plt.figure()
    plt.plot(sizes, r2_means, marker="o")
    plt.xlabel("Training sample size")
    plt.ylabel("R²")
    plt.title("Random Forest: R² vs Training Size (mean over trials)")
    plt.savefig(R2_PNG, bbox_inches="tight")
    plt.close()

    # RMSLE plot
    plt.figure()
    plt.plot(sizes, rmsle_means, marker="o")
    plt.xlabel("Training sample size")
    plt.ylabel("RMSLE")
    plt.title("Random Forest: RMSLE vs Training Size (mean over trials)")
    plt.savefig(RMSLE_PNG, bbox_inches="tight")
    plt.close()

    print(f"🖼️ Saved: {R2_PNG}")
    print(f"🖼️ Saved: {RMSLE_PNG}")

if __name__ == "__main__":
    run()
