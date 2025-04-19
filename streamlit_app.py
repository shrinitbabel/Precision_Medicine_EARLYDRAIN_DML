# app.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

from modules.preprocess import load_and_clean_data, prepare_variables
from modules.dml import load_model, get_feature_importances
from modules.cluster import generate_cate_matrix, plot_cate_tradeoff

OUTCOME_LABELS = {
    "mrs_binary": "Modified Rankin Score (Good outcome)",
    "gos_binary": "GOS-E ≥ 5 (Functionally Independent)",
    "vs_clin": "Clinical Vasospasm",
    "infection_dch": "Infection at Discharge",
    "infarct_dch": "Cerebral Infarction",
    "shunt_180": "Shunt Dependency at 6mo"
}

# --- Live input feature form
def patient_input_form():
    st.sidebar.subheader("👤 Patient Characteristics")
    
    col1, col2 = st.sidebar.columns(2)
    age = col1.number_input("Age", min_value=0, max_value=120, value=60)
    sex = col2.selectbox("Sex", options=["Male", "Female"])

    col3, col4 = st.sidebar.columns(2)
    height = col3.number_input("Height (cm)", value=170.0)
    weight = col4.number_input("Weight (kg)", value=70.0)
    bmi = weight / ((height / 100) ** 2)
    st.sidebar.metric("Calculated BMI", f"{bmi:.2f}")

    st.sidebar.subheader("🩺 Vital Signs")
    col5, col6 = st.sidebar.columns(2)
    sbp = col5.number_input("Systolic BP (mmHg)", value=120)
    dbp = col6.number_input("Diastolic BP (mmHg)", value=80)
    map_val = (2 * dbp + sbp) / 3
    st.sidebar.metric("Mean Arterial Pressure", f"{map_val:.1f} mmHg")

    hb = st.sidebar.number_input("Hemoglobin (g/dL)", value=13.0)
    balance = st.sidebar.number_input("Fluid Balance (mL)", value=500)
    icp_7am = st.sidebar.number_input("ICP at 7am (mmHg)", value=15)
    icp_high = st.sidebar.number_input("Max ICP in 24h (mmHg)", value=20)
    csf = st.sidebar.number_input("CSF Drainage (mL/24h)", value=150.0)

    st.sidebar.subheader("🧠 Neurologic Status")
    paresis = st.sidebar.selectbox("Paresis on Admission?", ["No", "Yes"])
    aphasia = st.sidebar.selectbox("Aphasia on Admission?", ["No", "Yes"])
    sedation = st.sidebar.selectbox("Sedation on Admission?", ["No", "Yes"])

    wfns = st.sidebar.slider(
        "WFNS Grade",
        1, 5, 1,
        help="1: GCS 15\n2: GCS 13-14\n3: GCS 13-14 with motor deficit\n4: GCS 7-12\n5: GCS 3-6"
    )

    st.sidebar.subheader("🩻 CT Imaging")
    fisher = st.sidebar.slider(
        "Modified Fisher Scale",
        1, 4, 2,
        help="1: Thin SAH, no IVH\n2: Thin SAH + IVH\n3: Thick SAH (>1 mm), no IVH\n4: Thick SAH + IVH"
    )
    ct_ich = st.sidebar.selectbox("ICH Present?", ["No", "Yes"])
    ct_ivh = st.sidebar.selectbox("IVH Present?", ["No", "Yes"])
    hh = st.sidebar.slider("Hunt and Hess Grade", 1, 5, 2, help="1: Asymptomatic\n5: Deep coma, decerebrate")

    st.sidebar.subheader("🧬 Aneurysm")
    circulation = st.sidebar.selectbox("Aneurysm Circulation", ["Anterior", "Posterior"])
    aneurysm_size = st.sidebar.number_input("Aneurysm Size (mm)", value=5.0)
    aneurysm_no = st.sidebar.slider("Number of Aneurysms", 1, 4, 1)

    st.sidebar.subheader("💊 Medications")
    nimodipine = st.sidebar.selectbox("Nimodipine Given?", ["Yes", "No"])
    statin = st.sidebar.selectbox("Statin Given?", ["Yes", "No"])
    mg = st.sidebar.selectbox("Magnesium Given?", ["Yes", "No"])

    # Convert inputs to feature dict
    patient_data = {
        "age": age,
        "sex": 1 if sex == "Male" else 0,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "rr_syst_mean": sbp,
        "rr_dia_mean": dbp,
        "rr_map_mean": map_val,
        "hb_mean": hb,
        "balance_mean": balance,
        "icp_7am_mean": icp_7am,
        "icp_high_mean": icp_high,
        "csf_mean": csf,
        "paresis_adm": 1 if paresis == "Yes" else 0,
        "aphasia_adm": 1 if aphasia == "Yes" else 0,
        "sedation_adm": 1 if sedation == "Yes" else 0,
        "wfns": wfns,
        "ct_modfisher": fisher,
        "ct_ich": 1 if ct_ich == "Yes" else 0,
        "ct_ivh": 1 if ct_ivh == "Yes" else 0,
        "hh": hh,
        "aneurysm_circulation": 1 if circulation == "Anterior" else 0,
        "aneurysm_size": aneurysm_size,
        "aneurysm_no": aneurysm_no,
        "nimodipine": 1 if nimodipine == "Yes" else 0,
        "statin": 1 if statin == "Yes" else 0,
        "mg": 1 if mg == "Yes" else 0,
        "aneurysm_trt": 1,
        "mrs_adm": 0  # assumed good baseline
    }

    return pd.DataFrame([patient_data])



def predict_individual_ite(model, patient_df, feature_names):
    imputed_X = patient_df[feature_names].values
    return model.effect(imputed_X)[0]


# ------------
# Streamlit UI
# ------------

st.set_page_config(page_title="🧠 EarlyDrain CDSS", layout="wide")
st.title("🧠 EarlyDrain Clinical Decision Support")
st.markdown("Estimate individual-level treatment effects for **prophylactic LD** using causal ML.")

with st.sidebar:
    st.header("🔎 Select Configuration")
    selected_outcome = st.selectbox("🎯 Choose outcome to analyze:", list(OUTCOME_LABELS.keys()),
                                    format_func=lambda x: OUTCOME_LABELS[x])

# ----------
# Backend
# ----------

df = load_and_clean_data("ed.csv", ed_daily_path="ed_daily.csv")
X, Y, T, feature_names = prepare_variables(df, selected_outcome)

model_path = f"models/cf_model_{selected_outcome}.joblib"
if not os.path.exists(model_path):
    st.error(f"❌ No model found for {selected_outcome}")
    st.stop()

cf_model = load_model(model_path)

# Live patient input
input_df = patient_input_form(feature_names)

# Prediction
ite = predict_individual_ite(cf_model, input_df, feature_names)

st.subheader("📈 Individual Treatment Effect (ITE)")
st.success(f"Estimated effect of prophylactic LD on **{OUTCOME_LABELS[selected_outcome]}**: `{ite:.4f}`")

# Feature Importance
feat_imp = get_feature_importances(cf_model, feature_names, top_n=10, verbose=False)
st.subheader("📊 Feature Importances")
st.bar_chart(feat_imp)

# Optional: Clustering/tradeoff plot
with st.expander("📍 Show Visual Placement in Clustering / Tradeoffs"):
    cate_file_dict = {
        "mrs_binary": "cate_results/cate_results_mrs_binary.csv",
        "infarct_dch": "cate_results/cate_results_infarct_dch.csv",
        "vs_clin": "cate_results/cate_results_vs_clin.csv",
        "infection_dch": "cate_results/cate_results_infection_dch.csv",
        "gos_binary": "cate_results/cate_results_gos_binary.csv",
        "shunt_180": "cate_results/cate_results_shunt_180.csv",
    }

    if selected_outcome in cate_file_dict:
        matrix = generate_cate_matrix(X, cate_file_dict)
        st.markdown("📌 Your patient is overlaid below (not interactive yet).")
        plot_cate_tradeoff(pd.DataFrame(matrix, columns=sorted(cate_file_dict.keys())),
                           "mrs_binary", selected_outcome)
