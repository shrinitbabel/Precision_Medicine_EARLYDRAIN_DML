# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

from modules.preprocess import load_and_clean_data, prepare_variables
from modules.dml import load_model
from modules.cluster import generate_cate_matrix, plot_cate_tradeoff
from modules.dml import get_feature_importances

OUTCOME_LABELS = {
    "mrs_binary": "Modified Rankin Score (Good outcome)",
    "gos_binary": "GOS-E ≥ 5 (Functionally Independent)",
    "vs_clin": "Clinical Vasospasm",
    "infection_dch": "Infection at Discharge",
    "infarct_dch": "Cerebral Infarction",
    "shunt_180": "Shunt Dependency at 6mo"
}

def predict_individual_ite(model, patient_df, feature_names):
    # Manual predict using causal forest
    imputed_X = patient_df[feature_names].values
    return model.effect(imputed_X)[0]

def load_patient_template():
    return pd.read_csv("patient_template.csv") if os.path.exists("patient_template.csv") else pd.DataFrame()

# ------------
# Streamlit UI
# ------------

st.set_page_config(page_title="🧠 EarlyDrain CDSS", layout="wide")
st.title("🧠 EarlyDrain Clinical Decision Support")
st.markdown("Estimate individual-level treatment effects for **prophylactic LD** using causal ML.")

with st.sidebar:
    st.header("🔎 Select Configuration")
    selected_outcome = st.selectbox("🎯 Choose outcome to analyze:", list(OUTCOME_LABELS.keys()), format_func=lambda x: OUTCOME_LABELS[x])

    st.markdown("🧬 Upload or edit patient info:")
    upload = st.file_uploader("Upload CSV (1 row = 1 patient)", type="csv")
    template = load_patient_template()

    if upload:
        input_df = pd.read_csv(upload)
    elif not template.empty:
        input_df = template.head(1)
    else:
        st.warning("No template or file found.")
        st.stop()

    st.dataframe(input_df)

# ----------
# Backend
# ----------

df = load_and_clean_data("ed.csv", ed_daily_path="ed_daily.csv")
X, Y, T, feature_names = prepare_variables(df, selected_outcome)

# Load model
model_path = f"models/cf_model_{selected_outcome}.joblib"
if not os.path.exists(model_path):
    st.error(f"❌ No model found for {selected_outcome}")
    st.stop()

cf_model = load_model(model_path)
ite = predict_individual_ite(cf_model, input_df, feature_names)

st.subheader("📈 Individual Treatment Effect (ITE)")
st.success(f"Estimated effect of prophylactic LD on **{OUTCOME_LABELS[selected_outcome]}**: `{ite:.4f}`")

# Feature importance plot
feat_imp = get_feature_importances(cf_model, feature_names, top_n=10, verbose=False)
st.subheader("📊 Feature Importances")
st.bar_chart(feat_imp)

# Visualizations (optional)
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
        idx = 0  # currently single row
        st.markdown("📌 Your patient is overlaid below (marked as red dot).")
        plot_cate_tradeoff(pd.DataFrame(matrix, columns=sorted(cate_file_dict.keys())), "mrs_binary", selected_outcome)

