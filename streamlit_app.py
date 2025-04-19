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
def patient_input_form(feature_names):
    st.subheader("🧬 Enter Patient Information")
    inputs = {}

    binary_fields = {
        "sex", "nimodipine", "statin", "mg", "ct_ivh", "ct_ich",
        "aneurysm_trt", "aneurysm_circulation", "sedation_adm",
        "paresis_adm", "aphasia_adm"
    }

    daily_fields = {
        'rr_map_mean': "Mean Arterial Pressure (MAP, mmHg)",
        'rr_syst_mean': "Systolic BP (mmHg)",
        'rr_dia_mean': "Diastolic BP (mmHg)",
        'hb_mean': "Hemoglobin (g/dL)",
        'balance_mean': "Fluid Balance (mL)",
        'icp_7am_mean': "ICP at 7am (mmHg)",
        'icp_high_mean': "Max ICP (mmHg)",
        'csf_mean': "CSF Drainage Volume (mL)"
    }

    for feat in feature_names:
        if feat in binary_fields:
            inputs[feat] = st.selectbox(f"{feat} (binary)", options=[0, 1], index=1)
        elif feat in daily_fields:
            inputs[feat] = st.number_input(daily_fields[feat], value=0.0, step=1.0)
        else:
            inputs[feat] = st.number_input(feat, value=0.0, step=0.1)

    return pd.DataFrame([inputs])


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
