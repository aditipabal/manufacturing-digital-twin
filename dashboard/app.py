from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Statistical Manufacturing Digital Twin | Semiconductor Yield Optimization",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# PATHS & ARTIFACT LOADING
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODELS_DIR / "random_forest_model.joblib")
    imputer = joblib.load(MODELS_DIR / "median_imputer.joblib")
    sensors = list(joblib.load(MODELS_DIR / "final_sensor_list.joblib"))
    return model, imputer, sensors

@st.cache_data
def load_results_data():
    ranges = pd.read_csv(RESULTS_DIR / "training_operating_ranges.csv")
    profile = pd.read_csv(RESULTS_DIR / "oof_low_risk_profile.csv", index_col=0)
    robustness = pd.read_csv(RESULTS_DIR / "robustness_simulations.csv")
    business = pd.read_csv(RESULTS_DIR / "business_impact_scenarios.csv")
    return ranges, profile, robustness, business

try:
    rf_model, imputer, final_sensors = load_artifacts()
    ranges_df, candidate_profile, robustness_sims, business_scenarios = load_results_data()
except Exception as exc:
    st.error(f"Error loading project artifacts: {exc}")
    st.stop()

# -----------------------------------------------------------------------------
# SCHEMA VALIDATION
# -----------------------------------------------------------------------------
required_range_cols = {"sensor", "P05", "P50", "P95"}
missing_range_cols = required_range_cols - set(ranges_df.columns)
if missing_range_cols:
    st.error(
        "training_operating_ranges.csv is missing required columns: "
        + ", ".join(sorted(missing_range_cols))
    )
    st.stop()

if "OOF_low_risk_median" not in candidate_profile.columns:
    st.error(
        "oof_low_risk_profile.csv must contain the column "
        "'OOF_low_risk_median'."
    )
    st.stop()

# Make sensor identifiers strings and validate every model sensor has a range/profile.
ranges_df["sensor"] = ranges_df["sensor"].astype(str)
candidate_series = candidate_profile["OOF_low_risk_median"].copy()
candidate_series.index = candidate_series.index.astype(str)

missing_ranges = [s for s in final_sensors if s not in set(ranges_df["sensor"])]
missing_profile = [s for s in final_sensors if s not in candidate_series.index]

if missing_ranges or missing_profile:
    if missing_ranges:
        st.error(f"Missing operating ranges for: {missing_ranges}")
    if missing_profile:
        st.error(f"Missing candidate profile values for: {missing_profile}")
    st.stop()

# -----------------------------------------------------------------------------
# CONSTANTS / HELPERS
# -----------------------------------------------------------------------------
BASELINE_RISK = 0.1348691767708998
CANDIDATE_DISTANCE_PERCENTILE = 16.52035

def get_range(sensor):
    row = ranges_df.loc[ranges_df["sensor"] == sensor].iloc[0]
    return float(row["P05"]), float(row["P50"]), float(row["P95"])

def safe_step(low, high):
    span = float(high) - float(low)
    return max(abs(span) / 100.0, 1e-6)

def predict_risk(values_by_sensor):
    """Predict using exact training feature order and the saved median imputer."""
    frame = pd.DataFrame(
        [[float(values_by_sensor[s]) for s in final_sensors]],
        columns=final_sensors,
    )
    transformed = imputer.transform(frame)
    return float(rf_model.predict_proba(transformed)[0, 1])

def candidate_dict():
    return {s: float(candidate_series.loc[s]) for s in final_sensors}

candidate_values = candidate_dict()
candidate_risk = predict_risk(candidate_values)
candidate_change = ((candidate_risk - BASELINE_RISK) / BASELINE_RISK) * 100

# -----------------------------------------------------------------------------
# STYLE
# -----------------------------------------------------------------------------
st.markdown(
    """
<style>
.stApp {background:#090A0F;color:#E5E7EB;}
.block-container {max-width:1400px;padding-top:1.5rem;padding-bottom:4rem;}
header, footer, #MainMenu {visibility:hidden;}
.hero {background:linear-gradient(180deg,#12141C,#0D0E15);border:1px solid #272A38;
       border-radius:6px;padding:2.3rem 2.5rem;margin-bottom:1.4rem;}
.eyebrow {color:#DC2626;font-size:.82rem;font-weight:800;letter-spacing:.15em;}
.hero h1 {color:white;font-size:2.35rem;margin:.45rem 0 .7rem 0;}
.hero p {color:#A7AFBD;max-width:900px;font-size:1.02rem;line-height:1.6;margin:0;}
.section {display:flex;align-items:center;border-bottom:1px solid #1F2937;
          padding-bottom:.75rem;margin-top:2.6rem;margin-bottom:1.35rem;}
.num {background:#DC2626;color:white;font-weight:800;font-size:.78rem;padding:.25rem .58rem;
      border-radius:2px;margin-right:.75rem;}
.title {font-size:1.25rem;font-weight:800;color:#F3F4F6;}
.card {background:#12141C;border:1px solid #272A38;border-radius:5px;padding:1.15rem;text-align:center;}
.card-accent {background:#12141C;border:1px solid #7F1D1D;border-radius:5px;padding:1.15rem;text-align:center;}
.value {font-size:2rem;font-weight:800;color:#F9FAFB;}
.value-good {font-size:2rem;font-weight:800;color:#10B981;}
.value-warn {font-size:2rem;font-weight:800;color:#EF4444;}
.label {font-size:.76rem;font-weight:700;color:#9CA3AF;letter-spacing:.07em;text-transform:uppercase;}
.callout {background:#12141C;border:1px solid #272A38;border-left:3px solid #DC2626;
          border-radius:4px;padding:1.25rem 1.4rem;color:#D1D5DB;line-height:1.6;}
.disclaimer {background:#1A1215;border:1px solid #7F1D1D;border-radius:4px;
             padding:.9rem 1.1rem;color:#FCA5A5;font-size:.86rem;line-height:1.5;margin-top:1rem;}
.goodnote {background:#0D1815;border:1px solid #14532D;border-radius:4px;
           padding:.85rem 1rem;color:#86EFAC;font-size:.88rem;line-height:1.5;}
.flow {display:flex;gap:.5rem;flex-wrap:wrap;}
.flowitem {flex:1;min-width:135px;background:#12141C;border:1px solid #272A38;
           border-top:2px solid #DC2626;padding:.85rem;text-align:center;}
.flowitem b {color:#F9FAFB;font-size:.84rem;}
.flowitem small {display:block;color:#6B7280;margin-top:.25rem;}
div[data-testid="stMetric"] {background:#12141C;border:1px solid #272A38;border-radius:5px;padding:1rem;}
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# HERO
# -----------------------------------------------------------------------------
st.markdown(
    """
<div class="hero">
<div class="eyebrow">STATISTICAL MANUFACTURING DIGITAL TWIN</div>
<h1>Semiconductor Yield &amp; Process Risk Optimization</h1>
<p>Use historical manufacturing data to identify lower-risk process regions,
simulate process changes, test stability under variation, and translate potential
defect reduction into business scenarios.</p>
</div>
""",
    unsafe_allow_html=True,
)

cols = st.columns(4)
hero_metrics = [
    ("1,567", "Production Runs"),
    ("590", "Process Measurements"),
    ("104", "Historical Failures"),
    ("6.64%", "Historical Failure Rate"),
]
for col, (value, label) in zip(cols, hero_metrics):
    with col:
        st.markdown(
            f'<div class="card"><div class="value">{value}</div>'
            f'<div class="label">{label}</div></div>',
            unsafe_allow_html=True,
        )

# -----------------------------------------------------------------------------
# 01 PROBLEM
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">01</span><span class="title">THE MANUFACTURING PROBLEM</span></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="callout">
<b style="color:white">Hundreds of process signals can hide the conditions associated with failure.</b><br>
This project analyzes 590 anonymized semiconductor process measurements to identify
statistical patterns associated with failure and explore historically plausible,
lower-risk candidate operating regions. The goal is decision support for controlled
engineering testing—not autonomous process control.
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 02 KEY FINDING
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">02</span><span class="title">KEY FINDING: CANDIDATE LOWER-RISK REGION</span></div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(f'<div class="card"><div class="label">Historical Average Model Risk</div><div class="value-warn">{BASELINE_RISK:.3f}</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="card"><div class="label">Candidate Model Risk</div><div class="value-good">{candidate_risk:.3f}</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="card-accent"><div class="label">Model Risk-Score Change</div><div class="value-good">{candidate_change:.1f}%</div><small style="color:#86EFAC">Historically plausible · {CANDIDATE_DISTANCE_PERCENTILE:.1f}% distance percentile</small></div>', unsafe_allow_html=True)

st.markdown(
    """
<div class="disclaimer"><b>Interpretation:</b> the reduction above is a change in the
Random Forest's uncalibrated model risk score. It is not evidence of the same percentage
reduction in physical defects. The candidate is a process state for controlled testing,
not a causal production recommendation.</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 03 DIGITAL TWIN
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">03</span><span class="title">INTERACTIVE DIGITAL TWIN SIMULATION</span></div>', unsafe_allow_html=True)
st.write("Move one anonymized process measurement within its historically observed P05–P95 range and see how the model risk score responds.")

left, right = st.columns([1.15, 1.4])
with left:
    selected_sensor = st.selectbox(
        "Select process sensor to tune",
        final_sensors,
        help="The sensor names are anonymized in the SECOM dataset.",
    )
    p05_val, med_val, p95_val = get_range(selected_sensor)
    cand_val = float(candidate_values[selected_sensor])
    # Clamp candidate to slider range defensively.
    default_val = min(max(cand_val, p05_val), p95_val)

    st.caption(
        f"Historical low (P05): {p05_val:.4f}  •  "
        f"Typical (P50): {med_val:.4f}  •  "
        f"Historical high (P95): {p95_val:.4f}"
    )
    slider_val = st.slider(
        f"Adjust {selected_sensor}",
        min_value=p05_val,
        max_value=p95_val,
        value=default_val,
        step=safe_step(p05_val, p95_val),
        format="%.5f",
        help="P05–P95 are empirical historical ranges, not engineering safety limits.",
    )

# Start from the OOF-derived candidate profile, then change only selected sensor.
simulation_values = candidate_values.copy()
simulation_values[selected_sensor] = slider_val
raw_risk = predict_risk(simulation_values)
risk_delta_pct = ((raw_risk - BASELINE_RISK) / BASELINE_RISK) * 100

with right:
    r1, r2 = st.columns(2)
    with r1:
        st.metric("Simulated model risk", f"{raw_risk:.3f}")
    with r2:
        st.metric("Change vs historical average", f"{risk_delta_pct:+.1f}%")

    if raw_risk <= 0.035:
        st.success("Lower model-risk region relative to the historical average.")
    elif raw_risk <= 0.080:
        st.warning("Intermediate model-risk region. Treat as a predictive signal, not a process guarantee.")
    else:
        st.error("Elevated model-risk region relative to the candidate profile.")

st.caption("The Digital Twin performs predictive what-if analysis. It does not establish that changing a sensor causes the modeled change in real-world defect rate.")

with st.expander("Advanced process controls — all 10 sensors"):
    st.write("Adjust all selected measurements simultaneously. Values remain constrained to empirical P05–P95 ranges.")
    adv_values = {}
    adv_cols = st.columns(2)
    for idx, sensor in enumerate(final_sensors):
        low, median, high = get_range(sensor)
        default = min(max(float(candidate_values[sensor]), low), high)
        with adv_cols[idx % 2]:
            adv_values[sensor] = st.slider(
                sensor,
                min_value=low,
                max_value=high,
                value=default,
                step=safe_step(low, high),
                format="%.5f",
                key=f"advanced_{sensor}",
            )

    adv_risk = predict_risk(adv_values)
    adv_delta = ((adv_risk - BASELINE_RISK) / BASELINE_RISK) * 100
    a1, a2 = st.columns(2)
    with a1:
        st.metric("Combined model risk", f"{adv_risk:.3f}")
    with a2:
        st.metric("Change vs historical average", f"{adv_delta:+.1f}%")
    st.caption("Multi-sensor combinations can be statistically unusual even when each individual value is inside its historical range. Engineering validation is still required.")

# -----------------------------------------------------------------------------
# 04 ROBUSTNESS
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">04</span><span class="title">DOES THE LOWER-RISK REGION REMAIN STABLE?</span></div>', unsafe_allow_html=True)

risk_col = None
for candidate in ["predicted_risk", "risk", "risk_score", "simulated_risk"]:
    if candidate in robustness_sims.columns:
        risk_col = candidate
        break

if risk_col is None:
    st.warning("Robustness simulation file loaded, but no recognized risk-score column was found.")
else:
    risk_series = pd.to_numeric(robustness_sims[risk_col], errors="coerce").dropna()
    rob_mean = float(risk_series.mean())
    rob_p90 = float(risk_series.quantile(0.90))
    rc1, rc2, rc3, rc4 = st.columns(4)
    for col, value, label in [
        (rc1, f"{len(risk_series):,}", "Nearby Simulations"),
        (rc2, f"{rob_mean:.3f}", "Average Neighborhood Risk"),
        (rc3, f"{rob_p90:.3f}", "90th Percentile Risk"),
        (rc4, f"{BASELINE_RISK:.3f}", "Historical Average Risk"),
    ]:
        with col:
            st.markdown(f'<div class="card"><div class="value">{value}</div><div class="label">{label}</div></div>', unsafe_allow_html=True)

    fig, ax = plt.subplots(figsize=(10, 3.7))
    ax.hist(risk_series, bins=35, alpha=0.8)
    ax.axvline(rob_mean, linestyle="--", linewidth=2, label=f"Neighborhood mean ({rob_mean:.3f})")
    ax.axvline(rob_p90, linestyle=":", linewidth=2, label=f"P90 ({rob_p90:.3f})")
    ax.axvline(BASELINE_RISK, linewidth=2, label=f"Historical average ({BASELINE_RISK:.3f})")
    ax.set_title("Modeled risk across nearby process perturbations")
    ax.set_xlabel("Model risk score")
    ax.set_ylabel("Simulation count")
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown(
        '<div class="goodnote"><b>Takeaway:</b> under the project’s perturbation scheme, '
        'the candidate neighborhood generally remained lower-risk than the historical average. '
        'This is model robustness, not proof of physical process causality.</div>',
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# 05 BUSINESS IMPACT
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">05</span><span class="title">WHAT COULD FEWER DEFECTS BE WORTH?</span></div>', unsafe_allow_html=True)
st.write("Illustrative scenario economics: 100,000 units/year, $500 per failed unit, and a historical failure rate of approximately 6.64%.")

# Known project scenario values; used for the executive cards.
biz_cards = [
    ("10% realized defect reduction", "$332K / year", "≈ 664 failures avoided"),
    ("20% realized defect reduction", "$664K / year", "≈ 1,327 failures avoided"),
    ("30% realized defect reduction", "$996K / year", "≈ 1,991 failures avoided"),
]
bcols = st.columns(3)
for col, (label, value, detail) in zip(bcols, biz_cards):
    with col:
        st.markdown(
            f'<div class="card"><div class="label">{label}</div>'
            f'<div class="value-good">{value}</div>'
            f'<small style="color:#9CA3AF">{detail}</small></div>',
            unsafe_allow_html=True,
        )

# Robustly identify business chart columns if present.
pct_col = next((c for c in ["reduction_pct", "defect_reduction_pct", "realized_defect_reduction_pct"] if c in business_scenarios.columns), None)
sav_col = next((c for c in ["annual_savings_usd", "savings_usd", "annual_savings"] if c in business_scenarios.columns), None)

if pct_col and sav_col:
    plot_df = business_scenarios[[pct_col, sav_col]].copy()
    plot_df[pct_col] = pd.to_numeric(plot_df[pct_col], errors="coerce")
    plot_df[sav_col] = pd.to_numeric(plot_df[sav_col], errors="coerce")
    plot_df = plot_df.dropna()

    fig2, ax2 = plt.subplots(figsize=(10, 3.6))
    ax2.bar([f"{x:g}%" for x in plot_df[pct_col]], plot_df[sav_col] / 1000.0)
    ax2.set_title("Illustrative annual savings by realized defect reduction")
    ax2.set_xlabel("Realized physical defect reduction")
    ax2.set_ylabel("Annual savings ($K)")
    fig2.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)
else:
    st.info("Business scenario cards are available; chart skipped because the CSV column names were not recognized.")

st.caption("These are illustrative scenarios—not realized savings. The model's risk-score reduction is not converted directly into defect reduction or dollars.")

# -----------------------------------------------------------------------------
# 06 MODEL VALIDATION
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">06</span><span class="title">MODEL PERFORMANCE & VALIDATION</span></div>', unsafe_allow_html=True)
v1, v2, v3 = st.columns(3)
with v1:
    st.metric("Held-out ROC-AUC", "0.785")
with v2:
    st.metric("Held-out PR-AUC", "0.352")
with v3:
    st.metric("Failure prevalence", "0.067")

st.markdown(
    """
<div class="callout"><b style="color:white">Why emphasize precision-recall?</b><br>
Only about 6.6% of observations are failures, so accuracy alone can look strong even
when a model misses most failures. PR-AUC is therefore a useful complement to ROC-AUC
for this imbalanced problem.</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 07 PIPELINE
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">07</span><span class="title">HOW THE SYSTEM WORKS</span></div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="flow">
<div class="flowitem"><b>01 · Manufacturing Data</b><small>1,567 runs</small></div>
<div class="flowitem"><b>02 · Data Quality</b><small>Missing / constant checks</small></div>
<div class="flowitem"><b>03 · Statistical Screening</b><small>Welch + Cohen's d + FDR</small></div>
<div class="flowitem"><b>04 · 10-Sensor Representation</b><small>Redundancy reduction</small></div>
<div class="flowitem"><b>05 · Failure-Risk Model</b><small>Class-weighted RF</small></div>
<div class="flowitem"><b>06 · Digital Twin</b><small>What-if simulation</small></div>
<div class="flowitem"><b>07 · Plausibility + Robustness</b><small>Distance + perturbations</small></div>
<div class="flowitem"><b>08 · Business Scenarios</b><small>Decision support</small></div>
</div>
""",
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# 08 METHODOLOGY & LIMITATIONS
# -----------------------------------------------------------------------------
st.markdown('<div class="section"><span class="num">08</span><span class="title">TECHNICAL METHODOLOGY & SCIENTIFIC LIMITATIONS</span></div>', unsafe_allow_html=True)

with st.expander("Technical methodology"):
    st.markdown(
        """
- Data-quality analysis for missingness, constants, and duplicate/redundant variables.
- Training-partition feature screening using Welch's t-test, Cohen's d, and Benjamini-Hochberg FDR.
- Correlation-based redundancy reduction and exclusion of very high-missingness features.
- Final 10-sensor representation with training-fitted median imputation.
- Class-weighted Random Forest for failure-risk modeling.
- Stratified held-out evaluation plus training-set out-of-fold predictions.
- Cost-sensitive threshold analysis using illustrative false-negative/false-positive economics.
- Digital Twin interventions constrained to empirical P05–P95 training ranges.
- Standardized multivariate historical-distance analysis for candidate plausibility.
- 1,000 nearby perturbation simulations for local robustness.
- Business scenarios kept separate from model-risk-score changes.
"""
    )

with st.expander("Scientific limitations & governance"):
    st.markdown(
        """
1. **Anonymized telemetry:** SECOM sensor identifiers have no supplied physical meaning or units.
2. **Observational data:** simulated interventions are predictive what-if evaluations, not causal effects.
3. **Uncalibrated scores:** Random Forest outputs are treated as model risk scores, not true failure probabilities.
4. **Empirical bounds:** P05–P95 ranges are historical ranges, not engineering safety limits.
5. **OOF limitation:** feature selection was performed on the overall training partition before OOF CV rather than independently inside every fold.
6. **Test-set limitation:** the held-out test set was inspected during iterative development, so it is not a pristine final production-validation set.
7. **Illustrative economics:** financial assumptions are scenarios rather than factory-supplied economics.
8. **Deployment requirement:** engineering review and controlled experiments would be required before any process change.
"""
    )

st.divider()
st.caption(
    "Portfolio prototype · Statistical manufacturing Digital Twin · "
    "Python · Statistical analysis · Machine learning · Process simulation · Streamlit"
)
