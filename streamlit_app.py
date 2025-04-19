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
    "vs_clin": "Clinical Vasospasm",
    "infection_dch": "Infection at Discharge",
    "infarct_dch": "Cerebral Infarction",
    "mrs_binary": "Modified Rankin Score (Good outcome)",
    "gos_binary": "GOS-E ≥ 5 (Functionally Independent)",
    "shunt_180": "Shunt Dependency at 6mo"
}


# --- Live input feature form
# patient_input_form()

def patient_input_form():
    def display_labeled_value(label, value, unit=""):
        st.markdown(f"<span style='font-size: 0.85rem; font-weight: 500'>{label}:</span> "
                    f"<span style='font-size: 0.85rem'>{value:.2f} {unit}</span>", unsafe_allow_html=True)

    with st.sidebar:
        with st.container():
            st.markdown("### 👤 Patient Characteristics")
            col1, col2 = st.columns(2)
            age = col1.number_input("Age", min_value=0, max_value=120, value=60)
            sex = col2.selectbox("Sex", options=["Male", "Female"])

            col3, col4 = st.columns(2)
            height = col3.number_input("Height (cm)", value=170.0)
            weight = col4.number_input("Weight (kg)", value=70.0)
            bmi = weight / ((height / 100) ** 2)
            display_labeled_value("Calculated BMI", bmi)

        st.divider()

        with st.container():
            st.markdown("### 🩺 Vitals & Labs")
            col5, col6 = st.columns(2)
            sbp = col5.number_input("Systolic BP (mmHg)", value=120)
            dbp = col6.number_input("Diastolic BP (mmHg)", value=80)
            map_val = (2 * dbp + sbp) / 3
            display_labeled_value("Mean Arterial Pressure", map_val, "mmHg")

            hb = st.number_input("Hemoglobin (g/dL)", value=13.0)
            balance = st.number_input("Fluid Balance (mL)", value=500)
            icp_7am = st.number_input("ICP at 7am (mmHg)", value=15)
            icp_high = st.number_input("Max ICP in 24h (mmHg)", value=20)
            csf = st.number_input("CSF Drainage (mL/24h)", value=150.0)

        st.divider()

        with st.container():
            st.markdown("### 🧠 Neurologic Status")
            paresis = st.selectbox("Paresis on Admission?", ["No", "Yes"])
            aphasia = st.selectbox("Aphasia on Admission?", ["No", "Yes"])
            sedation = st.selectbox("Sedation on Admission?", ["No", "Yes"])

            wfns = st.slider(
                "WFNS Grade", 1, 5, 1,
                help="1: GCS 15\n2: GCS 13-14\n3: GCS 13-14 with motor deficit\n4: GCS 7-12\n5: GCS 3-6"
            )

        st.divider()

        with st.container():
            st.markdown("### 🩻 CT Imaging")
            fisher = st.slider(
                "Modified Fisher Scale", 1, 4, 2,
                help="1: Thin SAH, no IVH\n2: Thin SAH + IVH\n3: Thick SAH (>1 mm), no IVH\n4: Thick SAH + IVH"
            )
            ct_ich = st.selectbox("ICH Present?", ["No", "Yes"])
            ct_ivh = st.selectbox("IVH Present?", ["No", "Yes"])
            hh = st.slider("Hunt and Hess Grade", 1, 5, 2,
                           help="1: Asymptomatic\n5: Deep coma, decerebrate")

        st.divider()

        with st.container():
            st.markdown("### 🧬 Aneurysm")
            circulation = st.selectbox("Aneurysm Circulation", ["Anterior", "Posterior"])
            aneurysm_size = st.number_input("Aneurysm Size (mm)", value=5.0)
            aneurysm_no = st.slider("Number of Aneurysms", 1, 4, 1)

        st.divider()

        with st.container():
            st.markdown("### 💊 Medications")
            nimodipine = st.selectbox("Nimodipine Given?", ["Yes", "No"])
            statin = st.selectbox("Statin Given?", ["Yes", "No"])
            mg = st.selectbox("Magnesium Given?", ["Yes", "No"])

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
        "mrs_adm": 0
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
input_df = patient_input_form()

# Prediction
ite = predict_individual_ite(cf_model, input_df, feature_names)

def render_ite_result(ite, outcome_key):
    abs_percent = f"{abs(ite) * 100:.1f}%"

    # Determine outcome-specific direction/interpretation
    if outcome_key in ["vs_clin", "infection_dch", "infarct_dch", "shunt_180"]:
        # These are risks: lower is better
        color = "green" if ite < 0 else "red"
        direction = "reduction" if ite < 0 else "increase"
    elif outcome_key in ["mrs_binary", "gos_binary"]:
        # These are benefits: higher is better
        color = "green" if ite > 0 else "red"
        direction = "increase" if ite > 0 else "reduction"
    else:
        # Fallback
        color = "gray"
        direction = "change"

    outcome_text = {
        "mrs_binary": "Treatment effect of prophylactic LD on achieving **Modified Rankin Score ≤ 2** at 6 months",
        "gos_binary": "Treatment effect of prophylactic LD on **functional independence (GOS-E ≥ 5)** at 6 months",
        "vs_clin": "Treatment effect of prophylactic LD on **risk of clinical vasospasm**",
        "infection_dch": "Treatment effect of prophylactic LD on **risk of infection at discharge**",
        "infarct_dch": "Treatment effect of prophylactic LD on **risk of cerebral infarction**",
        "shunt_180": "Treatment effect of prophylactic LD on **risk of shunt dependency at 6 months**"
    }

    st.subheader("📈 Individual Treatment Effect (ITE)")
    st.markdown(f"**{outcome_text[outcome_key]}**:")
    st.markdown(
        f"<span style='font-size:1.4rem; color:{color}; font-weight:700'>{abs_percent} {direction}</span>",
        unsafe_allow_html=True
    )

# -------------------------------
# Cluster Assignment (No Plot)
# -------------------------------

st.subheader("🧩 Cluster Membership")

try:
    from joblib import load
    from sklearn.preprocessing import Normalizer

    # Load saved models
    reducer = load("clusters/umap_model.joblib")
    kmeans = load("clusters/kmeans_model.joblib")

    # Patient CATE vector
    patient_cates = []
    for key in sorted(OUTCOME_LABELS.keys()):
        model = load_model(f"models/cf_model_{key}.joblib")
        effect = model.effect(input_df[feature_names].values)[0]
        if key in ['infarct_dch', 'shunt_180', 'infection_dch', 'vs_clin']:
            effect *= -1
        patient_cates.append(effect)

    patient_cates = np.array(patient_cates).reshape(1, -1)
    patient_cates_norm = Normalizer().fit_transform(patient_cates)
    patient_embed = reducer.transform(patient_cates_norm)

    cluster_label = int(kmeans.predict(patient_embed)[0])

    # Optional: Human-friendly labels for clusters
    cluster_themes = {
        0: "High Benefit, Low Risk",
        1: "Low Benefit, High Risk",
        2: "Neutral Effect",
        3: "Benefit in Function, Risk in Vasospasm"
    }

    label = cluster_themes.get(cluster_label, "Unknown Pattern")
    st.markdown(f"🧬 This patient belongs to **Cluster {cluster_label}** → _{label}_")

except Exception as e:
    st.error(f"⚠️ Could not determine cluster membership: {e}")


render_ite_result(ite, selected_outcome)
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
