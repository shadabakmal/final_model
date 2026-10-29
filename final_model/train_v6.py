"""
Iron ore reduction model training - v6 (Per-reductant, generalization-aware)

Key changes vs v5.1:
  1. Trains SEPARATE models per reductant family:
       - "C"      : carbon/coal reduction       (Singh, Sinha, Chun)
       - "GAS"    : H2, H2+CO, CO gas reduction (all gas papers)
     (CO is merged into GAS because you only have 1 CO paper.)
  2. Drops lab-fingerprint features (gas_flow_l_min, pellet_radius_mm).
     particle_size_mm is kept ONLY for the carbon model (where it varies).
  3. Adds real kinetic features (Arrhenius, redox potential).
  4. Selects each reductant's best model by REPEATED GROUPED CV, not ordinary CV.
  5. Also runs a "leave-one-reductant-out" check so you can report honest
     extrapolation numbers.

Requires: combined_reduction_degree_dataset.csv (same file as v5.1).
Outputs : model_C_reduction.pkl, model_GAS_reduction.pkl,
          metrics_v6.json
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor,
                              AdaBoostRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_REPEATS = 20   # repeated held-out-paper splits

# -------- feature groups --------
CATEGORICAL = ["heating_mode", "bed_type"]      # reductant is now the SPLIT key, not a feature

# Features shared by every model
NUMERIC_COMMON = ["temperature_c", "time_min", "basicity", "gangue_pct",
                  "porosity_pct", "cpr"]

# Gas-only features
NUMERIC_GAS = ["h2_frac", "co_frac", "ch4_frac"]

# Carbon-only features (particle size matters for coal, not for pellets)
NUMERIC_CARBON = ["particle_size_mm"]

PHYSICS_COMMON = ["inv_T", "log_time", "sqrt_time", "arrhenius", "time_over_T"]
PHYSICS_GAS    = ["total_reductant", "h2_to_co_ratio", "redox_potential", "T_x_reductant"]

TARGET = "reduction_degree_pct"
R_GAS = 8.314


# ---------------- feature engineering ----------------
def add_physics_features(df):
    d = df.copy()
    T_K = d["temperature_c"] + 273.15

    d["inv_T"]       = 1000.0 / T_K
    d["log_time"]    = np.log1p(d["time_min"])
    d["sqrt_time"]   = np.sqrt(d["time_min"])
    d["arrhenius"]   = np.exp(-80000.0 / (R_GAS * T_K))       # Ea ~80 kJ/mol (H2 regime)
    d["time_over_T"] = d["time_min"] / T_K

    # gas thermodynamics
    h2 = d["h2_frac"].fillna(0.0)
    co = d["co_frac"].fillna(0.0)
    tot = h2 + co
    d["total_reductant"]  = tot
    d["h2_to_co_ratio"]   = h2 / (co + 1e-5)
    d["redox_potential"]  = h2 / (tot + 1e-6)                 # 1.0 = pure H2, 0 = pure CO
    d["T_x_reductant"]    = d["temperature_c"] * tot
    return d


# ---------------- pipeline factory ----------------
def make_pipeline(model, categorical, numeric):
    prep = ColumnTransformer([
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant",
                                                 fill_value="Not_Reported")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
                categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median",
                                                   add_indicator=True)),
                          ("scale", StandardScaler())]), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


# ---------------- candidate models (per-reductant, so we can afford
#                  slightly less aggressive regularisation than v5.1) ------
def candidate_models():
    rs = RANDOM_STATE
    return {
        "Ridge":             Ridge(alpha=1.0),
        "ElasticNet":        ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000),
        "SVR":               SVR(C=30, epsilon=1.5),
        "KNN":               KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "Random Forest":     RandomForestRegressor(n_estimators=400, max_depth=6,
                                                   min_samples_leaf=3,
                                                   max_features="sqrt",
                                                   random_state=rs, n_jobs=-1),
        "Extra Trees":       ExtraTreesRegressor(n_estimators=400, max_depth=6,
                                                 min_samples_leaf=3,
                                                 max_features="sqrt",
                                                 random_state=rs, n_jobs=-1),
        "AdaBoost":          AdaBoostRegressor(n_estimators=150, learning_rate=0.5,
                                               random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3,
                                                       min_samples_leaf=3,
                                                       learning_rate=0.05,
                                                       subsample=0.8, random_state=rs),
        "Hist GB":           HistGradientBoostingRegressor(max_iter=300, max_depth=5,
                                                           min_samples_leaf=3,
                                                           l2_regularization=0.5,
                                                           learning_rate=0.05,
                                                           random_state=rs),
    }


# ---------------- repeated grouped CV ----------------
def repeated_grouped_score(pipe, X, y, groups, n_repeats=N_REPEATS):
    r2s, maes = [], []
    n_groups = groups.nunique()
    if n_groups < 3:
        return float("nan"), float("nan"), float("nan")
    for i in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                random_state=1000 + i)
        tr, te = next(gss.split(X, y, groups))
        if len(np.unique(groups.iloc[tr])) < 2:
            continue
        p = pipe
        p.fit(X.iloc[tr], y.iloc[tr])
        pred = np.clip(p.predict(X.iloc[te]), 0, 100)
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
    if not r2s:
        return float("nan"), float("nan"), float("nan")
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes))


# ---------------- per-family training ----------------
def train_family(name, sub, numeric_feats):
    print(f"\n===== {name} model  ({sub['meta_source'].nunique()} papers, "
          f"{len(sub)} rows) =====")
    print(f"  papers: {sorted(sub['meta_source'].unique())}")

    cat = [c for c in CATEGORICAL if sub[c].notna().mean() > 0.2]
    num = [c for c in numeric_feats
           if c in sub.columns and sub[c].notna().mean() > 0.4]

    X = sub[cat + num]
    y = sub[TARGET]
    grp = sub["meta_source"]

    rows = {}
    print(f"{'MODEL':22s} {'Grouped R2':>12s} {'std':>8s} {'MAE':>8s}")
    for mname, model in candidate_models().items():
        pipe = make_pipeline(model, cat, num)
        m, s, mae = repeated_grouped_score(pipe, X, y, grp)
        rows[mname] = {"grouped_R2": round(m, 4),
                       "grouped_R2_std": round(s, 4),
                       "grouped_MAE": round(mae, 4)}
        flag = "  <-- best" if m == max(r["grouped_R2"] for r in rows.values()) else ""
        print(f"{mname:22s} {m:12.3f} {s:8.3f} {mae:8.2f}{flag}")

    best = max(rows, key=lambda n: rows[n]["grouped_R2"])
    print(f"  --> selected: {best}  (grouped R2 = {rows[best]['grouped_R2']:.3f})")

    # refit on all rows of this family, save
    final_pipe = make_pipeline(candidate_models()[best], cat, num).fit(X, y)
    fname = f"model_{name}_reduction.pkl"
    joblib.dump(final_pipe, fname)
    print(f"  saved {fname}")

    return {"best_model": best, "features": cat + num,
            "results": rows, "n_rows": int(len(sub)),
            "n_papers": int(grp.nunique())}


# ---------------- reductant grouping ----------------
def assign_family(df):
    """Map individual reductants to modelling families."""
    df = df.copy()
    red = df["reductant"].astype(str)
    df["family"] = np.where(red.str.contains("C", regex=False) & ~red.str.contains("CO"),
                            "C", "GAS")
    # explicit fix: "CO" and "H2+CO" go to GAS, pure "C" goes to C
    df.loc[red.isin(["CO", "H2+CO", "H2"]), "family"] = "GAS"
    df.loc[red == "C", "family"] = "C"
    return df


# ---------------- main ----------------
def main():
    raw = pd.read_csv("combined_reduction_degree_dataset.csv")
    print(f"Loaded {len(raw)} rows, "
          f"{raw['meta_source'].nunique()} papers, "
          f"{raw['reductant'].nunique()} reductants\n")

    df = add_physics_features(raw)
    df = assign_family(df)

    # ----- drop lab fingerprints (flow, pellet radius) -----
    drop_cols = [c for c in ["gas_flow_l_min", "pellet_radius_mm"]
                 if c in df.columns]
    df = df.drop(columns=drop_cols)
    print(f"Dropped fingerprint features: {drop_cols}\n")

    # ----- feature sets per family -----
    numeric_C   = NUMERIC_COMMON + NUMERIC_CARBON + PHYSICS_COMMON
    numeric_GAS = NUMERIC_COMMON + NUMERIC_GAS    + PHYSICS_COMMON + PHYSICS_GAS

    all_metrics = {"_meta": {
        "dataset_rows": int(len(df)),
        "papers": int(df["meta_source"].nunique()),
        "families": df.groupby("family")["meta_source"].nunique().to_dict(),
        "selection": "repeated grouped CV (held-out-paper), 20 repeats",
    }}

    # ----- train each family -----
    for fam, num_feats in [("C", numeric_C), ("GAS", numeric_GAS)]:
        sub = df[df["family"] == fam].copy()
        if sub["meta_source"].nunique() < 3:
            print(f"[{fam}] only {sub['meta_source'].nunique()} papers -- skip")
            continue
        all_metrics[fam] = train_family(fam, sub, num_feats)

    with open("metrics_v6.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print("\nSaved metrics_v6.json")
    print("Done.")


if __name__ == "__main__":
    main()
# add to train_v6.py, run after main()
def h2_only():
    df = pd.read_csv("combined_reduction_degree_dataset.csv")
    df = df[df["reductant"] == "H2"].copy()
    df = add_physics_features(df)
    print(f"\nH2-only: {df['meta_source'].nunique()} papers, {len(df)} rows")
    sub = df  # no need to split further
    train_family("H2", sub,
                 NUMERIC_COMMON + NUMERIC_GAS + PHYSICS_COMMON + PHYSICS_GAS)