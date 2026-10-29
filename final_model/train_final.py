"""
Iron ore reduction model training - v5.4 (Ore Type & Family Split)
Updates:
  1. Added 'ore_type' to categorical features.
  2. Split training into GAS (H2, CO, mixtures) and CARBON (coal) models.
  3. Kept the best interpolation / best grouped dual-selection logic.
"""
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, KFold, cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
N_GROUPED_REPEATS = 10

# --- NEW: Added 'ore_type' to categorical features ---
CATEGORICAL = ["reductant", "heating_mode", "bed_type", "ore_type"]

NUMERIC_BASE = ["temperature_c", "time_min", "basicity", "porosity_pct", "gangue_pct",
                "cpr", "h2_frac", "co_frac", "ch4_frac"]
PHYSICS = ["inv_T", "log_time", "sqrt_time", "total_reductant", "h2_to_co_ratio", "T_x_reductant"]
TARGET = "reduction_degree_pct"


def add_physics_features(df):
    d = df.copy()
    d["inv_T"] = 1000.0 / (d["temperature_c"] + 273.15)
    d["log_time"] = np.log1p(d["time_min"])
    d["sqrt_time"] = np.sqrt(d["time_min"])
    d["total_reductant"] = d["h2_frac"].fillna(0) + d["co_frac"].fillna(0)
    d["h2_to_co_ratio"] = d["h2_frac"].fillna(0) / (d["co_frac"].fillna(1e-5) + 1e-5)
    d["T_x_reductant"] = d["temperature_c"] * d["total_reductant"]
    return d


def make_pipeline(model, categorical=CATEGORICAL, numeric=None):
    if numeric is None:
        numeric = NUMERIC_BASE + PHYSICS
    prep = ColumnTransformer([
        ("cat", Pipeline([("fill", SimpleImputer(strategy="constant", fill_value="Not_Reported")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), numeric),
    ])
    return Pipeline([("prep", prep), ("model", model)])


def candidate_models():
    rs = RANDOM_STATE
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=10.0),
        "Lasso": Lasso(alpha=0.5, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=10000),
        "SVR": SVR(C=10, epsilon=2.0),
        "KNN": KNeighborsRegressor(n_neighbors=8, weights="uniform"),
        "Decision Tree": DecisionTreeRegressor(max_depth=4, min_samples_leaf=5, random_state=rs),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                               max_features="sqrt", random_state=rs, n_jobs=-1),
        "Extra Trees": ExtraTreesRegressor(n_estimators=300, max_depth=4, min_samples_leaf=5,
                                           max_features="sqrt", random_state=rs, n_jobs=-1),
        "AdaBoost": AdaBoostRegressor(n_estimators=100, learning_rate=0.5, random_state=rs),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, min_samples_leaf=5,
                                                       learning_rate=0.05, subsample=0.8, random_state=rs),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(max_iter=200, max_depth=5, min_samples_leaf=5,
                                                                l2_regularization=1.0, learning_rate=0.05,
                                                                random_state=rs),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                         subsample=0.7, colsample_bytree=0.7,
                                         min_child_weight=5, reg_alpha=1.0, reg_lambda=5.0,
                                         random_state=rs, verbosity=0, n_jobs=-1)
    except ImportError:
        pass
    try:
        from lightgbm import LGBMRegressor
        models["LightGBM"] = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=8,
                                           min_child_samples=10, reg_alpha=0.5, reg_lambda=2.0,
                                           random_state=rs, verbose=-1)
    except ImportError:
        pass
    return models


def repeated_grouped_score(pipe, X, y, groups, n_repeats=N_GROUPED_REPEATS):
    r2s, maes = [], []
    for i in range(n_repeats):
        gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1000 + i)
        tr, te = next(gss.split(X, y, groups))
        if len(np.unique(groups.iloc[tr])) < 2:
            continue
        p = pipe
        p.fit(X.iloc[tr], y.iloc[tr])
        pred = np.clip(p.predict(X.iloc[te]), 0, 100)
        r2s.append(r2_score(y.iloc[te], pred))
        maes.append(mean_absolute_error(y.iloc[te], pred))
    if not r2s:
        return 0.0, 0.0, 0.0
    return float(np.mean(r2s)), float(np.std(r2s)), float(np.mean(maes))


def train_family(df, family_name, is_main_app_model=False):
    """Trains models on a specific family (GAS or CARBON)."""
    if len(df) == 0:
        print(f"Skipping {family_name} - no data.")
        return None, {}

    df = add_physics_features(df)
    NUMERIC = NUMERIC_BASE + PHYSICS
    X, y, groups = df[CATEGORICAL + NUMERIC], df[TARGET], df["meta_source"]

    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    
    print(f"\n{'='*80}")
    print(f"TRAINING {family_name} MODEL ({len(df)} rows, {groups.nunique()} papers)")
    print(f"{'='*80}")
    print(f"{'MODEL':24s} {'Ord_CV_R2':>10s} {'Ord_MAE':>8s} {'Grp_R2':>10s} {'Grp_MAE':>8s}")
    
    for name, model in candidate_models().items():
        pipe = make_pipeline(model, numeric=NUMERIC)
        ord_r2  = cross_val_score(pipe, X, y, cv=kf, scoring="r2").mean()
        ord_mae = -cross_val_score(pipe, X, y, cv=kf, scoring="neg_mean_absolute_error").mean()
        
        # Only run grouped CV if there are at least 4 papers in this family
        if groups.nunique() >= 4:
            grp_mean, grp_std, grp_mae = repeated_grouped_score(
                make_pipeline(model, numeric=NUMERIC), X, y, groups)
        else:
            grp_mean, grp_std, grp_mae = np.nan, np.nan, np.nan
            
        results[name] = {
            "Ordinary_CV_R2":     round(float(ord_r2), 4),
            "Ordinary_CV_MAE":    round(float(ord_mae), 4),
            "Grouped_CV_R2_mean": round(float(grp_mean), 4) if not np.isnan(grp_mean) else None,
            "Grouped_CV_R2_std":  round(float(grp_std), 4) if not np.isnan(grp_std) else None,
            "Grouped_CV_MAE":     round(float(grp_mae), 4) if not np.isnan(grp_mae) else None,
            "model": model,
        }
        print(f"{name:24s} {ord_r2:10.3f} {ord_mae:8.2f} {grp_mean if not np.isnan(grp_mean) else -99:10.3f} {grp_mae if not np.isnan(grp_mae) else -99:8.2f}")

    best_interp  = max(results, key=lambda n: results[n]["Ordinary_CV_R2"])
    best_grouped = max(results, key=lambda n: results[n]["Grouped_CV_R2_mean"]) if groups.nunique() >= 4 else best_interp

    print(f"\n>> Best INTERPOLATION for {family_name}: {best_interp} (R2 = {results[best_interp]['Ordinary_CV_R2']:.3f})")
    if groups.nunique() >= 4:
        print(f">> Best GROUPED for {family_name}: {best_grouped} (R2 = {results[best_grouped]['Grouped_CV_R2_mean']:.3f})")

    final_pipe = make_pipeline(results[best_interp]["model"], numeric=NUMERIC).fit(X, y)
    
    # Save the model
    if is_main_app_model:
        model_filename = "model_reduction_degree.pkl"
    else:
        model_filename = f"model_{family_name.lower()}_reduction.pkl"
        
    joblib.dump(final_pipe, model_filename)
    print(f"Saved {model_filename}")

    metrics = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in results.items()}
    meta = {
        "best_interpolation_model": best_interp,
        "best_grouped_model":       best_grouped,
        "n_rows": len(df), "n_papers": int(groups.nunique()),
        "models": metrics,
    }
    return best_interp, meta


def train_nh3_model():
    try:
        d = pd.read_csv("data_NH3.csv").dropna(subset=["metallization_pct"])
    except FileNotFoundError:
        print("\nNH3: file not found. Skipping.")
        return None
        
    if len(d) < 6:
        print("\nNH3: n<6, skipping.")
        return None
    cat = ["heating_mode", "bed_type", "feed_form"]
    num = ["temperature_c", "time_min", "porosity_pct"]
    X, y = d[cat + num], d["metallization_pct"]
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    pipe = make_pipeline(ExtraTreesRegressor(200, random_state=RANDOM_STATE),
                         categorical=cat, numeric=num)
    loo = np.clip(cross_val_predict(pipe, X, y, cv=LeaveOneOut()), 0, 100)
    r2 = r2_score(y, loo)
    mae = mean_absolute_error(y, loo)
    print(f"\nNH3 model: n={len(d)}, leave-one-out R2 = {r2:.3f}, MAE = {mae:.1f} (EXPLORATORY ONLY)")
    pipe.fit(X, y)
    joblib.dump(pipe, "model_NH3_metallization.pkl")
    with open("metrics_NH3.json", "w") as f:
        json.dump({"n_rows": len(d), "leave_one_out_R2": round(float(r2), 4),
                  "leave_one_out_MAE": round(float(mae), 4),
                  "warning": "n=9. Not statistically meaningful - exploratory only."}, f, indent=2)


if __name__ == "__main__":
    # Load data
    try:
        df = pd.read_csv("DATA/combined_reduction_degree_dataset_CLEAN.csv", on_bad_lines="skip")
    except FileNotFoundError:
        df = pd.read_csv("combined_reduction_degree_dataset.csv", on_bad_lines="skip")
        
    df = df.dropna(subset=[TARGET])
    
    # Auto-inject 'ore_type' if missing so the script doesn't crash
    if "ore_type" not in df.columns:
        print("WARNING: 'ore_type' column not found in CSV. Defaulting to 'Not_Reported'.")
        print("Please update your CSV to include actual ore types for better results!")
        df["ore_type"] = "Not_Reported"
        
    print(f"Loaded {len(df)} rows, {df['meta_source'].nunique()} papers.")
    
    # Split into Gas and Carbon families
    df_gas = df[df["reductant"].isin(["H2", "CO", "H2+CO", "H2+CH4"])].copy()
    df_carbon = df[df["reductant"] == "C"].copy()
    
    # Train Gas Model (Saved as model_reduction_degree.pkl for the Streamlit App)
    best_gas, meta_gas = train_family(df_gas, "GAS", is_main_app_model=True)
    
    # Train Carbon Model (Saved as model_carbon_reduction.pkl)
    best_carbon, meta_carbon = train_family(df_carbon, "CARBON", is_main_app_model=False)
    
    # Save combined metrics
    final_meta = meta_gas.copy()
    final_meta["CARBON_FAMILY"] = meta_carbon
    with open("metrics_reduction_degree.json", "w") as f:
        json.dump(final_meta, f, indent=2)
        
    train_nh3_model()
    print("\nAll done! Streamlit app will use the GAS model automatically.")