"""make_figures.py — Three key plots for the report."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("combined_reduction_degree_dataset.csv").dropna(subset=["reduction_degree_pct"])
CAT = ["reductant", "heating_mode", "bed_type"]
NUM = ["temperature_c", "time_min", "basicity", "porosity_pct", "gangue_pct",
       "cpr", "h2_frac", "co_frac", "ch4_frac"]

prep = ColumnTransformer([
    ("cat", Pipeline([("f", SimpleImputer(strategy="constant", fill_value="NA")),
                      ("o", OneHotEncoder(handle_unknown="ignore"))]), CAT),
    ("num", Pipeline([("i", SimpleImputer(strategy="median", add_indicator=True)),
                      ("s", StandardScaler())]), NUM),
])
pipe = Pipeline([("prep", prep), ("m", HistGradientBoostingRegressor(
    max_iter=200, max_depth=5, min_samples_leaf=5,
    l2_regularization=1.0, learning_rate=0.05, random_state=42))])

X, y = df[CAT + NUM], df["reduction_degree_pct"]

# Figure 1 — Interpolation parity
preds = cross_val_predict(pipe, X, y, cv=KFold(5, shuffle=True, random_state=42))
r2 = r2_score(y, preds); mae = mean_absolute_error(y, preds)

fig, ax = plt.subplots(figsize=(6, 6))
for red, c in zip(["C", "H2", "CO", "H2+CO", "H2+CH4"],
                  ["tab:brown","tab:blue","tab:green","tab:purple","tab:red"]):
    m = df["reductant"] == red
    if m.sum() == 0: continue
    ax.scatter(y[m], preds[m], s=15, alpha=0.55, label=red, color=c, edgecolor="none")
ax.plot([0,100],[0,100],"k--",lw=1)
ax.set_xlabel("Measured RD (%)"); ax.set_ylabel("Predicted RD (%)")
ax.set_title(f"Interpolation (5-fold CV): R² = {r2:.3f}, MAE = {mae:.2f} %")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig1_interpolation.png", dpi=150); plt.close()
print(f"Figure 1 saved (R² = {r2:.3f})")

# Figure 2 — Per-paper spread (motivation for 5.2)
fig, ax = plt.subplots(figsize=(9, 6))
sub = df[(df["reductant"].isin(["H2","CO","H2+CO"])) &
         (df["temperature_c"].between(850, 950))]
for paper, s in sub.groupby("meta_source"):
    if len(s) < 4: continue
    ax.plot(s["time_min"], s["reduction_degree_pct"], "o-", alpha=0.7,
            markersize=4, label=paper)
ax.set_xlabel("Time (min)"); ax.set_ylabel("Reduction Degree (%)")
ax.set_title("Same nominal range (850–950 °C) — huge inter-paper spread")
ax.legend(fontsize=7, ncol=3); ax.grid(alpha=0.3); plt.tight_layout()
plt.savefig("fig2_per_paper.png", dpi=150); plt.close()
print("Figure 2 saved")

# Figure 3 — Calibrated Avrami example (pick a clean paper)
sub = df[(df["meta_source"]=="Singh2024") & (df["temperature_c"]==900) &
         (df["basicity"]==0.0) & (df["particle_size_mm"]==0.175)].sort_values("time_min")
if len(sub) >= 4:
    t = sub["time_min"].values.astype(float)
    rd = sub["reduction_degree_pct"].values.astype(float)
    from scipy.optimize import curve_fit
    def avr(t,k,n): return 1-np.exp(-np.power(np.maximum(k*t,1e-12),n))
    popt,_ = curve_fit(avr, t, rd/100.0, p0=[0.02,1.0],
                       bounds=([1e-4,0.3],[2.0,5.0]), maxfev=20000)
    tfit = np.linspace(t.min(), t.max(), 100)
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(t, rd, s=40, color="tab:blue", label="Data points", zorder=3)
    ax.plot(tfit, avr(tfit,*popt)*100, "r-", lw=1.5,
            label=f"Avrami fit (k={popt[0]:.4f}, n={popt[1]:.2f})")
    ax.set_xlabel("Time (min)"); ax.set_ylabel("Reduction Degree (%)")
    ax.set_title("Calibrated Avrami example — Singh2024, 900 °C, r=0.175 mm")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig("fig3_avrami.png", dpi=150); plt.close()
    print("Figure 3 saved")

print("\nAll figures written.")