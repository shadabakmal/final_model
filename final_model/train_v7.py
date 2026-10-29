"""
Iron ore reduction - v7.1: two-stage kinetic modelling (NaN-safe)
"""
import json, warnings, joblib
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.linear_model import Ridge, HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_REPEATS = 30
R2_FIT_MIN = 0.5            # was 0.8 in v7

# ---------- Avrami ----------
def avrami(t, k, n):
    return 1.0 - np.exp(-np.power(np.maximum(k * t, 1e-12), n))

# ---------- Stage 1 ----------
def fit_kinetics(df):
    rows = []
    keys = ["meta_source", "temperature_c", "h2_frac", "co_frac",
            "basicity", "gangue_pct", "porosity_pct", "reductant"]
    for k_, grp in df.groupby(keys, dropna=False):
        if len(grp) < 3:
            continue
        t = grp["time_min"].values.astype(float)
        X = grp["reduction_degree_pct"].values.astype(float) / 100.0
        if t.max() <= 0 or np.ptp(X) < 0.03:
            continue
        try:
            popt, _ = curve_fit(avrami, t, X, p0=[0.02, 1.0],
                                bounds=([1e-4, 0.3], [2.0, 5.0]),
                                maxfev=20000)
            k_fit, n_fit = popt
            pred = avrami(t, k_fit, n_fit)
            ss_res = np.sum((X - pred) ** 2)
            ss_tot = np.sum((X - X.mean()) ** 2)
            r2_fit = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            rows.append(dict(zip(keys, k_), k=k_fit, n=n_fit,
                             r2_fit=r2_fit, n_points=len(grp)))
        except RuntimeError:
            continue
    return pd.DataFrame(rows)

# ---------- Stage 2 ----------
STAGE2_FEATURES = ["inv_T", "total_red", "h2_share",
                   "basicity", "gangue_pct", "porosity_pct"]

def stage2_models():
    return {
        "Ridge": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler()),
                           ("m",   Ridge(alpha=1.0))]),
        "Huber": Pipeline([("imp", SimpleImputer(strategy="median")),
                           ("sc",  StandardScaler()),
                           ("m",   HuberRegressor(alpha=1e-3, max_iter=500))]),
        "RandomForest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                  ("m",   RandomForestRegressor(
                                      n_estimators=300, max_depth=5,
                                      min_samples_leaf=2,
                                      random_state=42, n_jobs=-1))]),
    }

def grouped_cv_stage2(X, y, groups, model, n_repeats=N_REPEATS):
    if groups.nunique() < 4:
        return np.nan, np.nan
    r2s = []
    for i in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                random_state=100 + i)
        tr, te = next(gss.split(X, y, groups))
        if len(np.unique(groups.iloc[tr])) < 3:
            continue
        m = model
        m.fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        r2s.append(r2_score(y.iloc[te], p))
    if not r2s:
        return np.nan, np.nan
    return float(np.mean(r2s)), float(np.std(r2s))

# ---------- End-to-end ----------
def build_stage2_features(df):
    d = df.copy()
    d["inv_T"]     = 1000.0 / (d["temperature_c"] + 273.15)
    d["total_red"] = d["h2_frac"].fillna(0) + d["co_frac"].fillna(0)
    d["h2_share"]  = d["h2_frac"].fillna(0) / (d["total_red"] + 1e-6)
    return d[STAGE2_FEATURES]

def predict_rows(df_raw, model_k, model_n, feat_medians):
    Xr = build_stage2_features(df_raw).fillna(feat_medians)
    logk = model_k.predict(Xr)
    n    = np.clip(model_n.predict(Xr), 0.4, 4.0)
    k    = np.exp(logk)
    X    = avrami(df_raw["time_min"].values.astype(float), k, n) * 100.0
    return np.clip(X, 0, 100)

# ---------- Main ----------
def main():
    raw = pd.read_csv("combined_reduction_degree_dataset.csv")
    print(f"Loaded {len(raw)} rows, {raw['meta_source'].nunique()} papers\n")

    # Stage 1
    kin = fit_kinetics(raw)
    print(f"Stage 1: fitted kinetics for {len(kin)} groups")
    print(f"  median Avrami fit R2 = {kin['r2_fit'].median():.3f}")
    print(f"  papers represented    = {kin['meta_source'].nunique()}")
    kin.to_csv("stage1_kinetics.csv", index=False)

    kin = kin.dropna(subset=["k", "n"])
    kin = kin[kin["r2_fit"] > R2_FIT_MIN]
    print(f"  kept after R2>{R2_FIT_MIN}: {len(kin)} groups, "
          f"{kin['meta_source'].nunique()} papers")
    print("  survivors by paper:")
    print(kin.groupby("meta_source").size().to_string())
    print()

    kin = kin.copy()
    kin["log_k"]     = np.log(kin["k"].clip(lower=1e-6))
    kin["inv_T"]     = 1000.0 / (kin["temperature_c"] + 273.15)
    kin["total_red"] = kin["h2_frac"].fillna(0) + kin["co_frac"].fillna(0)
    kin["h2_share"]  = kin["h2_frac"].fillna(0) / (kin["total_red"] + 1e-6)

    X2 = kin[STAGE2_FEATURES]
    g2 = kin["meta_source"]

    print(f"Stage 2: {len(kin)} kinetic points from {g2.nunique()} papers\n")

    print(f"{'MODEL (log k)':18s} {'GroupedR2':>10s} {'std':>8s}")
    best_k, best_score = None, -1e9
    for name, mdl in stage2_models().items():
        m, s = grouped_cv_stage2(X2, kin["log_k"], g2, mdl)
        print(f"{name:18s} {m:10.3f} {s:8.3f}")
        if m > best_score:
            best_score, best_k = m, name
    print(f"  --> best for k: {best_k}  (R2 = {best_score:.3f})\n")

    print(f"{'MODEL (n)':18s} {'GroupedR2':>10s} {'std':>8s}")
    best_n, best_ns = None, -1e9
    for name, mdl in stage2_models().items():
        m, s = grouped_cv_stage2(X2, kin["n"], g2, mdl)
        print(f"{name:18s} {m:10.3f} {s:8.3f}")
        if m > best_ns:
            best_ns, best_n = m, name
    print(f"  --> best for n: {best_n}  (R2 = {best_ns:.3f})\n")

    # End-to-end grouped check
    mk = stage2_models()[best_k]
    mn = stage2_models()[best_n]
    medians = X2.median(numeric_only=True)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
    tr, te = next(gss.split(raw, raw["reduction_degree_pct"],
                            raw["meta_source"]))
    tr_papers = raw.iloc[tr]["meta_source"].unique()
    kin_tr = kin[kin["meta_source"].isin(tr_papers)]
    if len(kin_tr) > 5 and kin_tr["meta_source"].nunique() >= 3:
        mk.fit(kin_tr[STAGE2_FEATURES].fillna(medians), kin_tr["log_k"])
        mn.fit(kin_tr[STAGE2_FEATURES].fillna(medians), kin_tr["n"])
        yhat = predict_rows(raw.iloc[te], mk, mn, medians)
        ytrue = raw.iloc[te]["reduction_degree_pct"].values
        print(f"End-to-end grouped (1 hold-out): "
              f"R2 = {r2_score(ytrue, yhat):.3f}, "
              f"MAE = {mean_absolute_error(ytrue, yhat):.2f}")

    # Final fit on all kinetic data & save
    mk.fit(X2.fillna(medians), kin["log_k"])
    mn.fit(X2.fillna(medians), kin["n"])
    joblib.dump({"model_k": mk, "model_n": mn,
                 "features": STAGE2_FEATURES,
                 "medians": medians.to_dict()},
                "model_v7_kinetic.pkl")

    meta = {
        "stage1_groups":    int(len(kin)),
        "stage1_papers":    int(kin["meta_source"].nunique()),
        "stage1_median_R2": float(kin["r2_fit"].median()),
        "best_model_for_k": best_k,
        "grouped_R2_log_k": round(best_score, 4),
        "best_model_for_n": best_n,
        "grouped_R2_n":     round(best_ns, 4),
        "features":         STAGE2_FEATURES,
    }
    with open("metrics_v7.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved model_v7_kinetic.pkl, metrics_v7.json, stage1_kinetics.csv")

if __name__ == "__main__":
    main()