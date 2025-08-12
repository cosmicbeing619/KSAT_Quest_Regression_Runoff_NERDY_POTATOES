# src/make_pred_vs_actual_plot.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "rf_test_predictions.csv")
y = df["y_true"]
yhat = df["y_pred"]

plt.figure()
plt.scatter(y, yhat, alpha=0.6)
lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
plt.plot(lims, lims)
plt.xlabel("Actual Ksat (cm/hr)")
plt.ylabel("Predicted Ksat (cm/hr)")
plt.title("Predicted vs Actual")
plt.savefig(OUT / "pred_vs_actual.png", bbox_inches="tight")
plt.close()
print("Saved:", OUT / "pred_vs_actual.png")
