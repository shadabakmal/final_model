"""Calibrated Avrami — one fit per unique kinetic condition."""
import numpy as np, pandas as pd, json
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score, mean_absolute_error

def avrami(t, k, n):
    return 1.0 - np.exp(-np.power(np.maximum(k * t, 1e-12), n))

def fit_avrami(t, rd):
    if len(t) < 4 or np.ptp(rd) < 5:
        return None
    try:
        popt, _ = curve_fit(
            avrami, t, rd / 100.0,
            p0=[0.02, 1.0], bounds=([1e-4, 0.3], [2.0, 5.0]), maxfev=20000,
        )
    except (RuntimeError, ValueError):
        return None
    pred = np.clip(avrami(t, *popt) * 100, 0, 100)
    return popt[0], popt[1], r2_score(rd, pred)

def loocv_predict(t, rd):
    n = len(t)
    preds = np.full(n, np.nan)
    for i in range(n):
        mask = np.ones(n, dtype=bool); mask[i] = False
        if np.ptp(rd[mask]) < 3: continue
        try:
            popt, _ = curve_fit(
                avrami, t[mask], rd[mask] / 100.0,
                p0=[0.02, 1.0], bounds=([1e-4, 0.3], [2.0, 5.0]), maxfev=20000,
            )
        except (RuntimeError, ValueError):
            continue
        preds[i] = np.clip(avrami(t[i:i+1], *popt)[0] * 100, 0, 100)
    return preds

# --- Load ---
df = pd.read_csv("combined_reduction_degree_dataset.csv").dropna(subset=["reduction_degree_pct"])
print(f"Loaded {len(df)} rows, {df['meta_source'].nunique()} papers\n")

# Group by ALL discriminating columns. Fill NaN with sentinel so NaN groups together.
GROUP_COLS = ["meta_source", "temperature_c", "particle_size_mm", "basicity",
              "h2_frac", "co_frac", "ch4_frac"]

def key_val(x):
    return "NA" if pd.isna(x) else round(float(x), 6)

df["_grp"] = df[GROUP_COLS].apply(
    lambda r: "|".join([str(r["meta_source"])] +
                       [str(key_val(r[c])) for c in GROUP_COLS[1:]]),
    axis=1,
)

rows = []
for grp, sub in df.groupby("_grp"):
    sub = sub.dropna(subset=["time_min"])
    if len(sub) < 5: continue
    t = sub["time_min"].values.astype(float)
    rd = sub["reduction_degree_pct"].values.astype(float)
    if np.ptp(rd) < 5: continue
    full = fit_avrami(t, rd)
    if full is None: continue
    k, n, fit_r2 = full
    preds = loocv_predict(t, rd)
    valid = ~np.isnan(preds)
    if valid.sum() < 3: continue
    loo_mae = mean_absolute_error(rd[valid], preds[valid])
    rows.append({
        "paper": sub["meta_source"].iloc[0],
        "T": int(sub["temperature_c"].iloc[0]),
        "n": len(t),
        "fit_R2": fit_r2,
        "loo_MAE": loo_mae,
        "RD_range": float(np.ptp(rd)),
    })

res = pd.DataFrame(rows).sort_values(["paper", "T"])
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda x: f"{x:7.3f}"))

# --- Aggregate ---
print("\n" + "=" * 70)
print(f"Groups analysed: {len(res)}  |  Papers: {res['paper'].nunique()}")
print("=" * 70)

print(f"\nAvrami FIT quality per isothermal curve:")
print(f"  Median fit R²        : {res['fit_R2'].median():.3f}")
print(f"  Mean   fit R²        : {res['fit_R2'].mean():.3f}")
print(f"  Fraction with R²>0.9 : {(res['fit_R2'] > 0.9).mean()*100:.1f} %")

print(f"\nLeave-one-out prediction MAE:")
print(f"  Median LOOCV MAE     : {res['loo_MAE'].median():.2f} %")
print(f"  Weighted mean        : {np.average(res['loo_MAE'], weights=res['n']):.2f} %")

print("\n" + "=" * 70)
print("REPORT NUMBERS (Section 5.3)")
print("=" * 70)
print(f"  Calibrated Avrami over {len(res)} isothermal series (from {res['paper'].nunique()} papers):")
print(f"    Median fit R²           = {res['fit_R2'].median():.3f}")
print(f"    Fraction with fit R²>0.9 = {(res['fit_R2']>0.9).mean()*100:.0f} %")
print(f"    Median LOOCV MAE        = {res['loo_MAE'].median():.2f} %")

json.dump({
    "n_groups": int(len(res)),
    "n_papers": int(res['paper'].nunique()),
    "median_fit_R2":   round(float(res['fit_R2'].median()), 4),
    "fraction_R2_gt_09": round(float((res['fit_R2']>0.9).mean()), 4),
    "median_loo_MAE":  round(float(res['loo_MAE'].median()), 4),
    "per_group": res.to_dict(orient="records"),
}, open("metrics_v9_calibrated.json", "w"), indent=2)
print("\nSaved metrics_v9_calibrated.json")