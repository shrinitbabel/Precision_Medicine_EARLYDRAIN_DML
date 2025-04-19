import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from modules.preprocess import load_and_clean_data, prepare_variables
from modules.dml import load_model, get_feature_importances
from modules.cluster import generate_cate_matrix, plot_cate_tradeoff
from modules.dml import predict_individual_ite  # assume this function exists in dml.py

OUTCOME_LABELS = {
    "mrs_binary": "Modified Rankin Score (Good outcome)",
    "gos_binary": "GOS-E ≥ 5 (Functionally Independent)",
    "vs_clin": "Clinical Vasospasm",
    "infection_dch": "Infection at Discharge",
    "infarct_dch": "Cerebral Infarction",
    "shunt_180": "Shunt Dependency at 6mo"
}

# -- Predict for all outcomes
def predict_all_outcomes(patient_df, df):
    results = {}
    for outcome in OUTCOME_LABELS:
        _, _, _, features = prepare_variables(df, outcome)
        model_path = f"models/cf_model_{outcome}.joblib"
        if os.path.exists(model_path):
            model = load_model(model_path)
            pred = predict_individual_ite(model, patient_df, features)
            results[outcome] = pred
    return results

# Streamlit setup
st.set_page_config(page_title="🧠 EarlyDrain CDSS", layout="wide")
st.title("🧠 EarlyDrain Clinical Decision Support")
st.markdown("Estimate individual-level treatment effects for **prophylactic LD** using causal ML.")

# Load DataFrame and prepare features
df = load_and_clean_data("ed.csv", ed_daily_path="ed_daily.csv")

with st.sidebar:
    st.header("🔎 Select Configuration")

    st.markdown("### Patient Characteristics")
    age = st.number_input("Age", value=60)
    sex = st.radio("Sex", ["Male", "Female"])
    height = st.number_input("Height (cm)", value=170)
    weight = st.number_input("Weight (kg)", value=70)
    bmi = round(weight / ((height / 100) ** 2), 2)
    st.markdown(f"**BMI:** {bmi}")

    st.markdown("### Vital Signs")
    sbp = st.number_input("Systolic BP (mmHg)", value=120)
    dbp = st.number_input("Diastolic BP (mmHg)", value=80)
    map_val = round((2/3)*dbp + (1/3)*sbp, 1)
    st.markdown(f"**Mean Arterial Pressure (MAP):** {map_val}")
    hb = st.number_input("Hemoglobin (g/dL)", value=13.5)
    balance = st.number_input("Fluid Balance (mL)", value=0)
    icp_7am = st.number_input("ICP at 7am (mmHg)", value=10)
    icp_max = st.number_input("Max ICP in 24h (mmHg)", value=20)
    csf = st.number_input("CSF Drainage in 24h (mL)", value=200)

    st.markdown("### Admission Findings")
    paresis = st.radio("Paresis on Admission?", ["Yes", "No"])
    aphasia = st.radio("Aphasia on Admission?", ["Yes", "No"])
    sedation = st.radio("Sedation on Admission?", ["Yes", "No"])
    wfns = st.slider("WFNS Grade", 1, 5, value=1, help="1: GCS 15, 2: GCS 13-14, 3: GCS 13-14 + motor deficit, 4: GCS 7-12, 5: GCS 3-6")

    st.markdown("### CT Scan Findings")
    ct_modfisher = st.slider("Modified Fisher Scale", 1, 4, value=1, help="1: Thin SAH no IVH, 2: Thin SAH + IVH, 3: Thick SAH, 4: Thick SAH + IVH")
    ct_ich = st.radio("ICH Present?", ["Yes", "No"])
    ct_ivh = st.radio("IVH Present?", ["Yes", "No"])
    hh = st.slider("Hunt & Hess Grade", 1, 5, value=1, help="1: Asymptomatic or mild headache, 5: Deep coma")

    st.markdown("### Aneurysm")
    circ = st.radio("Aneurysm Circulation", ["Anterior", "Posterior"])
    size = st.number_input("Aneurysm Size (mm)", value=5.0)
    num = st.slider("Number of Aneurysms", 1, 4, value=1)

    st.markdown("### Medications")
    nimodipine = st.radio("Nimodipine Given?", ["Yes", "No"])
    statin = st.radio("Statin Given?", ["Yes", "No"])
    mg = st.radio("Magnesium Given?", ["Yes", "No"])

    # Map inputs to model-ready format
    patient = pd.DataFrame([{ # keys match feature_names list
        'age': age, 'sex': 1 if sex == "Male" else 0,
        'height': height, 'weight': weight, 'bmi': bmi,
        'rr_map_mean': map_val, 'rr_syst_mean': sbp, 'rr_dia_mean': dbp,
        'hb_mean': hb, 'balance_mean': balance,
        'icp_7am_mean': icp_7am, 'icp_high_mean': icp_max,
        'csf_mean': csf, 'wfns': wfns, 'ct_modfisher': ct_modfisher,
        'ct_ich': 1 if ct_ich == "Yes" else 0,
        'ct_ivh': 1 if ct_ivh == "Yes" else 0,
        'hh': hh,
        'aneurysm_trt': 1,  # Assume default
        'aneurysm_circulation': 1 if circ == "Anterior" else 0,
        'aneurysm_size': size, 'aneurysm_no': num,
        'nimodipine': 1 if nimodipine == "Yes" else 0,
        'statin': 1 if statin == "Yes" else 0,
        'mg': 1 if mg == "Yes" else 0,
        'sedation_adm': 1 if sedation == "Yes" else 0,
        'paresis_adm': 1 if paresis == "Yes" else 0,
        'aphasia_adm': 1 if aphasia == "Yes" else 0,
        'mrs_adm': 0  # placeholder
    }])

# -----------
# RIGHT PANEL
# -----------

col1, col2 = st.columns([2, 3])

with col1:
    st.subheader("📈 Individual Treatment Effects for All Outcomes")
    all_ites = predict_all_outcomes(patient, df)
    for k, v in all_ites.items():
        st.markdown(f"**{OUTCOME_LABELS[k]}:** `{v:.4f}`")

with col2:
    st.subheader("🧠 Cluster Visualization")
    # placeholder: future integration with plotly 3D cluster + overlay
    st.markdown("_(Cluster assignment and position will appear here)_")

st.subheader("💧 CSF vs ICP Effect Curve (Vasospasm)")
# placeholder for future: overlay patient CSF & ICP on colored ITE map
