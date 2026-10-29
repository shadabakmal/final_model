"""Calibrated Avrami transfer — the practical model for report Section 5.3."""
import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

def avrami(t, k, n):
    return 1.0 - np.exp(-np.power(np.maximum(k*t, 1e-12), n))

def fit_paper(sub, k_cal=3):
    sub = sub.dropna(subset=["time_min", "reduction_degree_pct"])
    if len(sub) < k_cal + 2:
        return None
    order = np.argsort(sub["time_min"].values)
    step = max(1, len(sub) // k_cal)
    cal = sub.iloc[order[::step][:k_cal]]
    test = sub.drop(cal.index)
    if len(test) < 2:
        return None
    try:
        popt, _ = curve_fit(
            avrami,
            cal["time_min"].values.astype(float),
            cal["reduction_degree_pct"].values / 100.0,
            p0=[0.02, 1.0],
            bounds=([1e-4, 0.3], [2.0, 5.0]),
            maxfev=20000,
        )
    except RuntimeError:
        return None
    pred = np.clip(avrami(test["time_min"].values.astype(float), *popt) * 100, 0, 100)
    return (r2_score(test["reduction_degree_pct"], pred),
            mean_absolute_error(test["reduction_degree_pct"], pred),
            len(test))

df = pd.read_csv("combined_reduction_degree_dataset.csv").dropna(subset=["reduction_degree_pct"])
all_r2, all_mae = [], []
print(f"{'PAPER':16s} {'n_test':>6s} {'R2':>8s} {'MAE':>8s}")
print("-" * 46)
for paper, sub in df.groupby("meta_source"):
    r = fit_paper(sub, k_cal=3)
    if r is None:
        continue
    r2, mae, n = r
    all_r2.append(r2)
    all_mae.append(mae)
    print(f"{paper:16s} {n:>6d} {r2:>8.3f} {mae:>8.2f}")
print("-" * 46)
if all_r2:
    print(f"\nCalibrated R²  = {np.mean(all_r2):+.3f} ± {np.std(all_r2):.3f}")
    print(f"Calibrated MAE = {np.mean(all_mae):.2f} %")
    json.dump({
        "n_papers": len(all_r2),
        "n_calibration_points": 3,
        "mean_R2": round(float(np.mean(all_r2)), 4),
        "std_R2":  round(float(np.std(all_r2)), 4),
        "mean_MAE": round(float(np.mean(all_mae)), 4),
    }, open("metrics_v9_calibrated.json", "w"), indent=2)
    print("Saved metrics_v9_calibrated.json")