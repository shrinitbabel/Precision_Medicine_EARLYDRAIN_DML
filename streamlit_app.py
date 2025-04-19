import streamlit as st
import pandas as pd
import numpy as np

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
