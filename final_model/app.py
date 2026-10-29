"""
app.py — Iron Ore Reduction Predictor (v3)
Updates:
  1. Added 'ore_type' categorical input.
  2. Dynamically switches between GAS and CARBON models based on reductant.
"""
import streamlit as st
import joblib
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit

st.set_page_config(
    page_title="Iron Ore Reduction Predictor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
#  Load models & metrics
# ------------------------------------------------------------------ #
@st.cache_resource
def load_models():
    gas_model = None
    carbon_model = None
    if Path("model_reduction_degree.pkl").exists():
        gas_model = joblib.load("model_reduction_degree.pkl")
    if Path("model_carbon_reduction.pkl").exists():
        carbon_model = joblib.load("model_carbon_reduction.pkl")
    return gas_model, carbon_model

@st.cache_data
def load_metrics():
    p = Path("metrics_reduction_degree.json")
    return json.loads(p.read_text()) if p.exists() else {}

GAS_MODEL, CARBON_MODEL = load_models()
METRICS = load_metrics()

# ------------------------------------------------------------------ #
#  Feature engineering — MUST match train_final.py v5.4
# ------------------------------------------------------------------ #
CATEGORICAL = ["reductant", "heating_mode", "bed_type", "ore_type"]
NUMERIC_BASE = ["temperature_c", "time_min", "basicity", "porosity_pct",
                "gangue_pct", "cpr", "h2_frac", "co_frac", "ch4_frac"]
PHYSICS = ["inv_T", "log_time", "sqrt_time", "total_reductant",
           "h2_to_co_ratio", "T_x_reductant"]

def add_physics_features(df):
    d = df.copy()
    d["inv_T"] = 1000.0 / (d["temperature_c"] + 273.15)
    d["log_time"] = np.log1p(d["time_min"])
    d["sqrt_time"] = np.sqrt(d["time_min"])
    d["total_reductant"] = d["h2_frac"].fillna(0) + d["co_frac"].fillna(0)
    d["h2_to_co_ratio"] = d["h2_frac"].fillna(0) / (d["co_frac"].fillna(1e-5) + 1e-5)
    d["T_x_reductant"] = d["temperature_c"] * d["total_reductant"]
    return d

def build_input_row(inputs: dict) -> pd.DataFrame:
    df = pd.DataFrame([inputs])
    df = add_physics_features(df)
    return df[CATEGORICAL + NUMERIC_BASE + PHYSICS]

# ------------------------------------------------------------------ #
#  Metallization & Morphology
# ------------------------------------------------------------------ #
def rd_to_metallization(rd_pct, reductant="H2"):
    if reductant == "C":
        met = max(0.0, (rd_pct - 33.3) / 66.7 * 100 * 0.85)
    else:
        met = max(0.0, (rd_pct - 33.3) / 66.7 * 100)
    return float(np.clip(met, 0.0, 100.0))

def predict_morphology(T, h2_frac, co_frac, time_min, reductant):
    total_reducing = h2_frac + co_frac
    h2_share = h2_frac / total_reducing if total_reducing > 0 else 0.0

    if reductant == "C":
        if T < 950:
            return ("Fine micropores + dense iron layer", "Low-temperature carbon reduction produces fine-grained iron with micropores. Slow kinetics; iron shell forms around unreacted core.", "subsurface")
        return ("Porous sponge iron", "High-temperature carbon reduction yields porous sponge iron with good reducibility. Whisker formation suppressed by high temperature.", "sponge")

    if T < 700:
        return ("Fine micropores, dense iron layer", "Below 700 °C, reduction produces dense iron with fine micropores. Slow kinetics.", "dense")

    if h2_share < 0.4 and T >= 800:
        return ("Iron whiskers + dense shell", "CO-rich atmosphere above 800 °C promotes iron whisker growth. The dense iron shell impedes gas diffusion.", "whiskers")

    if h2_share > 0.7 and T >= 750:
        if T >= 900:
            return ("Coarse porous iron, high porosity", "H₂-rich reduction above 900 °C produces coarse porous iron. Sintering begins; pores are interconnected and large.", "coarse_porous")
        return ("Porous sponge iron, uniform", "H₂-rich reduction at 750-900 °C produces uniform porous sponge iron. Fine cracks observed.", "sponge")

    if 0.4 <= h2_share <= 0.7:
        return ("Mixed morphology: nodular iron", "H₂+CO mixtures at 800-900 °C produce nodular iron with low whisker content.", "mixed")

    return ("Porous sponge iron (generic)", "Typical reduction morphology.", "sponge")

# (Schematic drawing function remains exactly the same)
def draw_morphology_schematic(morph_key):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Ellipse, Rectangle

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    np.random.seed(42)

    if morph_key == "whiskers":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Circle((5, 3), 2.5, color="#b0b0b0", ec="k", lw=1))
        for angle in np.linspace(-np.pi*0.9, -np.pi*0.1, 14):
            length = 2.5 + np.random.rand() * 1.5
            x0, y0 = 5 + 2.5*np.cos(angle), 3 + 2.5*np.sin(angle)
            x1 = x0 + length*np.cos(angle); y1 = y0 + length*np.sin(angle)
            ax.plot([x0, x1], [y0, y1], color="#404040", lw=1.2)
        for _ in range(30):
            x, y = np.random.rand()*8 + 1, np.random.rand()*6 + 3
            ax.add_patch(Circle((x, y), 0.1, color="#1a1a1a"))
        ax.text(5, 9.5, "Iron whiskers + dense shell\n(CO-rich, T ≥ 800 °C)", ha="center", fontsize=10, weight="bold")
    elif morph_key == "sponge":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Rectangle((1.5, 1.5), 7, 7, color="#a0a0a0", ec="k", lw=1))
        for _ in range(60):
            x, y = np.random.rand()*6 + 2, np.random.rand()*6 + 2
            r = 0.08 + np.random.rand()*0.25
            ax.add_patch(Circle((x, y), r, color="#f5f5f5", ec="k", lw=0.4))
        ax.text(5, 9.5, "Porous sponge iron\n(H₂-rich, 750-900 °C)", ha="center", fontsize=10, weight="bold")
    elif morph_key == "coarse_porous":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Rectangle((1.5, 1.5), 7, 7, color="#a0a0a0", ec="k", lw=1))
        for _ in range(20):
            x, y = np.random.rand()*6 + 2, np.random.rand()*6 + 2
            r = 0.3 + np.random.rand()*0.6
            ax.add_patch(Circle((x, y), r, color="#f5f5f5", ec="k", lw=0.6))
        ax.text(5, 9.5, "Coarse porous iron\n(H₂-rich, T ≥ 900 °C)", ha="center", fontsize=10, weight="bold")
    elif morph_key == "dense":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Rectangle((1.5, 1.5), 7, 7, color="#c0c0c0", ec="k", lw=1))
        for _ in range(200):
            x, y = np.random.rand()*6.5 + 1.75, np.random.rand()*6.5 + 1.75
            ax.add_patch(Circle((x, y), 0.03, color="#808080"))
        ax.text(5, 9.5, "Fine micropores + dense iron layer\n(T < 700 °C)", ha="center", fontsize=10, weight="bold")
    elif morph_key == "mixed":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Rectangle((1.5, 1.5), 7, 7, color="#a5a5a5", ec="k", lw=1))
        for _ in range(30):
            x, y = np.random.rand()*6 + 2, np.random.rand()*6 + 2
            r = 0.1 + np.random.rand()*0.2
            ax.add_patch(Circle((x, y), r, color="#f5f5f5", ec="k", lw=0.4))
        for _ in range(8):
            x, y = np.random.rand()*5.5 + 2, np.random.rand()*5.5 + 2
            ax.add_patch(Ellipse((x, y), 0.7, 0.25, color="#606060", ec="k", lw=0.5))
        ax.text(5, 9.5, "Mixed morphology: nodular iron\n(H₂+CO mixture, 800-900 °C)", ha="center", fontsize=10, weight="bold")
    elif morph_key == "subsurface":
        ax.set_facecolor("#f5f5f5")
        ax.add_patch(Circle((5, 5), 4, color="#b8b8b8", ec="k", lw=1))
        ax.add_patch(Circle((5, 5), 2.5, color="#d8d8d8", ec="k", lw=0.8))
        ax.add_patch(Circle((5, 5), 1.2, color="#e8e8e8", ec="k", lw=0.6))
        ax.text(5, 9.8, "Core-shell structure\n(carbon reduction, T < 950 °C)", ha="center", fontsize=10, weight="bold")
        ax.text(5, 5, "Unreacted\ncore", ha="center", va="center", fontsize=8)
        ax.text(5, 7.5, "Fe shell", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    return fig

def avrami(t, k, n):
    return 1.0 - np.exp(-np.power(np.maximum(k * t, 1e-12), n))

def fit_avrami(t_array, rd_array):
    try:
        popt, _ = curve_fit(
            avrami, t_array, rd_array / 100.0,
            p0=[0.02, 1.0], bounds=([1e-4, 0.3], [2.0, 5.0]), maxfev=20000,
        )
        return popt
    except Exception:
        return None

# ------------------------------------------------------------------ #
#  Sidebar presets
# ------------------------------------------------------------------ #
st.sidebar.title("⚙️ Quick Presets")

PRESETS = {
    "Custom": None,
    "H₂ — hematite pellet, 900 °C": {
        "reductant": "H2", "heating_mode": "Resistance", "bed_type": "Static", "ore_type": "Hematite",
        "temperature_c": 900.0, "time_min": 30.0,
        "basicity": 0.3, "porosity_pct": 30.0, "gangue_pct": 5.0, "cpr": 0.0,
        "h2_frac": 1.0, "co_frac": 0.0, "ch4_frac": 0.0,
    },
    "H₂+CO — Midrex-like mix, 850 °C": {
        "reductant": "H2+CO", "heating_mode": "Resistance", "bed_type": "Static", "ore_type": "Hematite",
        "temperature_c": 850.0, "time_min": 30.0,
        "basicity": 0.3, "porosity_pct": 30.0, "gangue_pct": 5.0, "cpr": 0.0,
        "h2_frac": 0.55, "co_frac": 0.21, "ch4_frac": 0.0,
    },
    "CO — magnetite pellet, 900 °C": {
        "reductant": "CO", "heating_mode": "Resistance", "bed_type": "Static", "ore_type": "Magnetite",
        "temperature_c": 900.0, "time_min": 30.0,
        "basicity": 0.1, "porosity_pct": 30.0, "gangue_pct": 4.6, "cpr": 0.0,
        "h2_frac": 0.0, "co_frac": 1.0, "ch4_frac": 0.0,
    },
    "Coal — carbon reduction, 1000 °C": {
        "reductant": "C", "heating_mode": "Resistance", "bed_type": "Static", "ore_type": "Hematite",
        "temperature_c": 1000.0, "time_min": 60.0,
        "basicity": 1.0, "porosity_pct": 30.0, "gangue_pct": 7.14, "cpr": 1.0,
        "h2_frac": 0.0, "co_frac": 0.0, "ch4_frac": 0.0,
    },
}

preset_name = st.sidebar.selectbox("Preset", list(PRESETS.keys()))
preset = PRESETS[preset_name] or {}

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ Interpolation model. Predictions outside training grid carry higher uncertainty.")

# ------------------------------------------------------------------ #
#  MAIN
# ------------------------------------------------------------------ #
st.title("🔬 Iron Ore Reduction Predictor")
st.caption("Reduction degree · Metallization · Morphology")

if GAS_MODEL is None:
    st.error("**Model file `model_reduction_degree.pkl` not found.** Run `python train_final.py` first.")
    st.stop()

tab_ml, tab_avrami, tab_about = st.tabs(["🤖 Prediction", "📈 Calibrated Avrami", "ℹ️ About"])

with tab_ml:
    st.markdown("### Input conditions")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Process conditions**")
        temperature_c = st.number_input("Temperature (°C)", 400.0, 1400.0, float(preset.get("temperature_c", 900.0)), 10.0)
        time_min = st.number_input("Time (min)", 0.0, 400.0, float(preset.get("time_min", 30.0)), 1.0)
        heating_mode = st.selectbox("Heating mode", ["Resistance", "Microwave", "Not_Reported"], index=["Resistance","Microwave","Not_Reported"].index(preset.get("heating_mode","Resistance")))
        bed_type = st.selectbox("Bed type", ["Static","Fluidized","Not_Reported"], index=["Static","Fluidized","Not_Reported"].index(preset.get("bed_type","Static")))

    with col2:
        st.markdown("**Reductant / gas**")
        reductant = st.selectbox("Reductant", ["H2","CO","H2+CO","H2+CH4","C"], index=["H2","CO","H2+CO","H2+CH4","C"].index(preset.get("reductant","H2")))
        h2_frac  = st.slider("H₂ fraction",  0.0, 1.0, float(preset.get("h2_frac",  1.0)), 0.05)
        co_frac  = st.slider("CO fraction",  0.0, 1.0, float(preset.get("co_frac",  0.0)), 0.05)
        ch4_frac = st.slider("CH₄ fraction", 0.0, 1.0, float(preset.get("ch4_frac", 0.0)), 0.05)

    with col3:
        st.markdown("**Ore properties**")
        # --- NEW: Ore Type Dropdown ---
        ore_type = st.selectbox("Ore Type", ["Hematite", "Magnetite", "Titanomagnetite", "Not_Reported"], 
                                index=["Hematite", "Magnetite", "Titanomagnetite", "Not_Reported"].index(preset.get("ore_type", "Hematite")))
        basicity = st.number_input("Basicity (CaO/SiO₂)", 0.0, 3.0, float(preset.get("basicity", 0.3)), 0.05)
        porosity_pct = st.number_input("Porosity (%)", 0.0, 70.0, float(preset.get("porosity_pct", 30.0)), 0.5)
        gangue_pct   = st.number_input("Gangue (%)", 0.0, 30.0, float(preset.get("gangue_pct", 5.0)), 0.1)
        cpr          = st.number_input("CPR", 0.0, 5.0, float(preset.get("cpr", 0.0)), 0.1)

    if st.button("🔮 Predict", type="primary", use_container_width=True):
        inputs = {
            "reductant": reductant, "heating_mode": heating_mode, "bed_type": bed_type, "ore_type": ore_type,
            "temperature_c": temperature_c, "time_min": time_min,
            "basicity": basicity, "porosity_pct": porosity_pct, "gangue_pct": gangue_pct, "cpr": cpr,
            "h2_frac": h2_frac, "co_frac": co_frac, "ch4_frac": ch4_frac,
        }
        X = build_input_row(inputs)

        # --- NEW: Select the right model ---
        if reductant == "C" and CARBON_MODEL is not None:
            active_model = CARBON_MODEL
            model_name = "Carbon Model"
        else:
            active_model = GAS_MODEL
            model_name = "Gas Model"

        rd_pred = float(np.clip(active_model.predict(X)[0], 0, 100))

        # Get MAE based on active model
        if reductant == "C" and "CARBON_FAMILY" in METRICS:
            best = METRICS["CARBON_FAMILY"]["best_interpolation_model"]
            interp_mae = float(METRICS["CARBON_FAMILY"]["models"][best]["Ordinary_CV_MAE"])
        else:
            best = METRICS.get("best_interpolation_model")
            interp_mae = float(METRICS.get("models", {}).get(best, {}).get("Ordinary_CV_MAE", 5.0))

        met_pred = rd_to_metallization(rd_pred, reductant)
        morph_class, morph_desc, morph_key = predict_morphology(temperature_c, h2_frac, co_frac, time_min, reductant)

        st.markdown("---")
        st.markdown("## Prediction Results")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Reduction degree", f"{rd_pred:.1f} %", help=f"±{interp_mae:.1f} % (5-fold CV MAE) | {model_name}")
            st.progress(min(rd_pred/100, 1.0))
        with c2:
            st.metric("Metallization (Fe⁰)", f"{met_pred:.1f} %", help="Estimated from RD via stepwise hematite reduction")
            st.progress(min(met_pred/100, 1.0))
        with c3:
            st.metric("Confidence range", f"[{max(0,rd_pred-interp_mae):.0f} – {min(100,rd_pred+interp_mae):.0f}] %")
            st.caption(f"Predicting with: **{model_name}**")

        st.markdown("---")
        st.markdown("### Predicted morphology")
        cm1, cm2 = st.columns([1, 2])

        with cm1:
            fig = draw_morphology_schematic(morph_key)
            st.pyplot(fig)

        with cm2:
            st.markdown(f"#### {morph_class}")
            st.markdown(morph_desc)
            st.markdown("**Regime conditions:**")
            total_red = h2_frac + co_frac
            h2_share = h2_frac / total_red if total_red > 0 else 0
            st.markdown(
                f"- Temperature: **{temperature_c:.0f} °C**\n"
                f"- H₂ share: **{h2_share*100:.0f} %**\n"
                f"- Reductant: **{reductant}**\n"
                f"- Time: **{time_min:.0f} min**"
            )
            st.caption("Morphology is rule-based, derived from literature reduction-regime maps.")

# =================================================================== #
#  TAB 2 — Calibrated Avrami (unchanged)
# =================================================================== #
with tab_avrami:
    # ... (Keep the exact same code as before) ...
    st.markdown("### Calibrated kinetic prediction")
    st.caption("Enter 3 calibration points from your own experiment. The Avrami model is fitted and used to predict the full curve.")
    cal_cols = st.columns(3)
    cal_t, cal_rd = [], []
    for i, c in enumerate(cal_cols):
        with c:
            st.markdown(f"**Point {i+1}**")
            t_i = st.number_input(f"Time (min) #{i+1}", 0.5, 400.0, [15.0, 30.0, 60.0][i], key=f"t{i}")
            rd_i = st.slider(f"RD (%) #{i+1}", 0.0, 100.0, [40.0, 70.0, 90.0][i], key=f"rd{i}")
            cal_t.append(t_i); cal_rd.append(rd_i)

    t_pred_min = st.number_input("Predict curve from (min)", 0.5, 400.0, 5.0)
    t_pred_max = st.number_input("Predict curve to (min)", 1.0, 400.0, 120.0)

    if st.button("🧪 Fit & Predict Curve", type="primary", use_container_width=True):
        cal_t_arr = np.array(sorted(cal_t))
        cal_rd_arr = np.array([rd for _, rd in sorted(zip(cal_t, cal_rd))])
        popt = fit_avrami(cal_t_arr, cal_rd_arr)
        if popt is None:
            st.error("Avrami fit failed — check that calibration points span a range.")
        else:
            k, n = popt
            t_fit = np.linspace(t_pred_min, t_pred_max, 200)
            rd_fit = np.clip(avrami(t_fit, k, n) * 100, 0, 100)
            met_fit = np.array([rd_to_metallization(r, "H2") for r in rd_fit])

            st.success(f"**Avrami parameters:** k = {k:.5f} min⁻¹, n = {n:.3f}")

            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(t_fit, rd_fit, "r-", lw=2, label="Reduction degree")
            ax.plot(t_fit, met_fit, "b--", lw=2, label="Metallization (estimated)")
            ax.scatter(cal_t_arr, cal_rd_arr, s=90, color="tab:red", zorder=3, label="Calibration points")
            ax.set_xlabel("Time (min)"); ax.set_ylabel("Percent (%)"); ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend()
            st.pyplot(fig)

            df_pred = pd.DataFrame({"time_min": np.round(t_fit[::20], 1), "RD_pct": np.round(rd_fit[::20], 2), "Met_pct": np.round(met_fit[::20], 2)})
            st.dataframe(df_pred, use_container_width=True)
            st.download_button("⬇️ Download as CSV", df_pred.to_csv(index=False), file_name="avrami_prediction.csv")

# =================================================================== #
#  TAB 3 — About
# =================================================================== #
with tab_about:
    st.markdown("### About this model")
    st.markdown(
        """
        **Dataset**: 554 experimental records from 25 papers.
        
        **Architecture**:
        - **GAS Model**: Trained on H₂, CO, and H₂+CO (229 rows, 21 papers).
        - **CARBON Model**: Trained on Coal/Carbon reduction (325 rows, 4 papers).
        - Both models use `ore_type` (Hematite, Magnetite, Titanomagnetite) to account for mineralogy.

        **Three outputs**:
        1. **Reduction degree** — ML interpolation (Gas Model R² = 0.809).
        2. **Metallization** — derived stepwise from RD.
        3. **Morphology** — rule-based physics classification.
        """
    )