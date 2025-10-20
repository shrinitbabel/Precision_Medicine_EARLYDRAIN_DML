import os
import json
from typing import Optional, Literal, Dict, Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse

# --- Your modules ---
from modules.preprocess import load_and_clean_data, prepare_variables
from modules.dml import load_model, get_feature_importances

# ----------------------------
# Config & labels
# ----------------------------
OUTCOME_LABELS = {
    "vs_clin": "Clinical Vasospasm",
    "infection_dch": "Infection at Discharge",
    "infarct_dch": "Cerebral Infarction",
    "mrs_binary": "Modified Rankin Score (Good outcome)",
    "gos_binary": "GOS-E ≥ 5 (Functionally Independent)",
    "shunt_180": "Shunt Dependency at 6mo",
}

VALID_OUTCOMES = tuple(OUTCOME_LABELS.keys())

# ----------------------------
# App
# ----------------------------
app = FastAPI(title="EarlyDrain CDSS (Render)",
              version="1.0.0",
              description="Causal ML individual treatment effect (ITE) service for prophylactic LD.")

# ----------------------------
# Warm-up: load data once and cache feature sets per outcome
# ----------------------------
_df_main_path = os.getenv("ED_MAIN_CSV", "ed.csv")
_df_daily_path = os.getenv("ED_DAILY_CSV", "ed_daily.csv")

try:
    _df = load_and_clean_data(_df_main_path, ed_daily_path=_df_daily_path)
except Exception as e:
    # We don't crash the service; some endpoints (like /health) still work.
    _df = None
    print(f"[WARN] Could not load datasets at startup: {e}")

# Map outcome -> feature_names (all outcomes share the same list today, but we compute via API)
_feature_names_cache: Dict[str, list] = {}

def get_feature_names_for_outcome(outcome: str) -> list:
    if outcome in _feature_names_cache:
        return _feature_names_cache[outcome]
    if _df is None:
        raise RuntimeError("Dataframe not loaded; cannot derive feature names.")
    _, _, _, feature_names = prepare_variables(_df, outcome)
    _feature_names_cache[outcome] = feature_names
    return feature_names

# ----------------------------
# Utility: build a single-row DF like your Streamlit form
# ----------------------------
def build_patient_df(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Accepts raw request JSON keys mirroring the Streamlit sidebar and
    returns a single-row DataFrame with all engineered fields present.
    """
    # Required base fields with defaults matching your Streamlit UI
    age = float(payload.get("age", 60))
    sex_str = payload.get("sex", "Male")  # "Male" / "Female"
    height = float(payload.get("height", 170.0))
    weight = float(payload.get("weight", 70.0))
    sbp = float(payload.get("rr_syst_mean", payload.get("sbp", 120)))
    dbp = float(payload.get("rr_dia_mean", payload.get("dbp", 80)))
    hb = float(payload.get("hb_mean", payload.get("hb", 13.0)))
    balance = float(payload.get("balance_mean", payload.get("balance", 500)))
    icp_7am = float(payload.get("icp_7am_mean", payload.get("icp_7am", 15)))
    icp_high = float(payload.get("icp_high_mean", payload.get("icp_high", 20)))
    csf = float(payload.get("csf_mean", payload.get("csf", 150.0)))

    paresis = payload.get("paresis_adm", payload.get("paresis", "No"))
    aphasia = payload.get("aphasia_adm", payload.get("aphasia", "No"))
    sedation = payload.get("sedation_adm", payload.get("sedation", "No"))

    wfns = int(payload.get("wfns", 1))
    fisher = int(payload.get("ct_modfisher", payload.get("fisher", 2)))
    ct_ich = payload.get("ct_ich", payload.get("ich", "No"))
    ct_ivh = payload.get("ct_ivh", payload.get("ivh", "No"))
    hh = int(payload.get("hh", 2))

    circulation = payload.get("aneurysm_circulation", payload.get("circulation", "Anterior"))  # "Anterior"/"Posterior"
    aneurysm_size = float(payload.get("aneurysm_size", 5.0))
    aneurysm_no = int(payload.get("aneurysm_no", 1))

    nimodipine = payload.get("nimodipine", "Yes")
    statin = payload.get("statin", "No")
    mg = payload.get("mg", "No")

    # Derived:
    bmi = weight / ((height / 100.0) ** 2) if height > 0 else 0.0
    rr_map = (2 * dbp + sbp) / 3.0

    # Binary mapping consistent with your preprocess.py
    def yn(val):  # accepts "Yes"/"No" or 1/0
        if isinstance(val, (int, float)):
            return 1 if val else 0
        return 1 if str(val).strip().lower() in ("yes", "y", "1", "true") else 0

    row = {
        "age": age,
        "sex": 1 if str(sex_str).lower().startswith("m") else 0,
        "height": height,
        "weight": weight,
        "bmi": bmi,
        "rr_syst_mean": sbp,
        "rr_dia_mean": dbp,
        "rr_map_mean": rr_map,
        "hb_mean": hb,
        "balance_mean": balance,
        "icp_7am_mean": icp_7am,
        "icp_high_mean": icp_high,
        "csf_mean": csf,
        "paresis_adm": yn(paresis),
        "aphasia_adm": yn(aphasia),
        "sedation_adm": yn(sedation),
        "wfns": wfns,
        "ct_modfisher": fisher,
        "ct_ich": yn(ct_ich),
        "ct_ivh": yn(ct_ivh),
        "hh": hh,
        "aneurysm_circulation": 1 if str(circulation).lower().startswith("a") else 0,
        "aneurysm_size": aneurysm_size,
        "aneurysm_no": aneurysm_no,
        "nimodipine": yn(nimodipine),
        "statin": yn(statin),
        "mg": yn(mg),

        # constants used in your Streamlit code
        "aneurysm_trt": 1,
        "mrs_adm": 0,
    }

    return pd.DataFrame([row])


def load_cf_model(outcome: str):
    path = os.path.join("models", f"cf_model_{outcome}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found: {path}")
    return load_model(path)

# ----------------------------
# Simple pages & health
# ----------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    opts = "".join([f"<li><code>{k}</code> — {v}</li>" for k, v in OUTCOME_LABELS.items()])
    return f"""
    <html>
      <head><title>EarlyDrain CDSS</title></head>
      <body style="font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; padding:24px;">
        <h1>🧠 EarlyDrain CDSS (Render)</h1>
        <p>This service computes <b>individual treatment effects</b> (ITE) for prophylactic lumbar drain (LD) using your trained Causal Forest models.</p>
        <h3>Endpoints</h3>
        <ul>
          <li><code>GET /health</code></li>
          <li><code>POST /predict?outcome=&lt;key&gt;</code> — JSON body with patient fields (see README).</li>
          <li><code>GET /feature-importances?outcome=&lt;key&gt;&top_n=10</code></li>
          <li><code>GET /umap</code> — returns saved Plotly JSON if available.</li>
        </ul>
        <h3>Outcomes</h3>
        <ul>{opts}</ul>
        <p>Example curl:</p>
        <pre>
curl -X POST "$HOST/predict?outcome=vs_clin" -H "Content-Type: application/json" -d '{{
  "age": 60, "sex": "Male", "height": 170, "weight": 70,
  "sbp": 120, "dbp": 80, "hb": 13, "balance": 500,
  "icp_7am": 15, "icp_high": 20, "csf": 150,
  "paresis": "No", "aphasia": "No", "sedation": "No",
  "wfns": 1, "fisher": 2, "ich": "No", "ivh": "No", "hh": 2,
  "circulation": "Anterior", "aneurysm_size": 5, "aneurysm_no": 1,
  "nimodipine": "Yes", "statin": "No", "mg": "No"
}}'
        </pre>
      </body>
    </html>
    """

@app.get("/health")
def health():
    return {"status": "ok", "models_dir": os.path.isdir("models")}

# ----------------------------
# Core: prediction
# ----------------------------
def interpret_ite(ite: float, outcome_key: str) -> Dict[str, str]:
    abs_percent = f"{abs(ite) * 100:.1f}%"
    if outcome_key in ["vs_clin", "infection_dch", "infarct_dch", "shunt_180"]:
        color = "green" if ite < 0 else "red"
        direction = "reduction" if ite < 0 else "increase"
    elif outcome_key in ["mrs_binary", "gos_binary"]:
        color = "green" if ite > 0 else "red"
        direction = "increase" if ite > 0 else "reduction"
    else:
        color = "gray"
        direction = "change"
    return {"abs_percent": abs_percent, "direction": direction, "color_hint": color}

@app.post("/predict")
def predict(
    outcome: Literal[VALID_OUTCOMES] = Query(..., description="Outcome key, see / for options."),
    payload: Dict[str, Any] = None
):
    try:
        feature_names = get_feature_names_for_outcome(outcome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not derive feature_names: {e}")

    model_path = os.path.join("models", f"cf_model_{outcome}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for '{outcome}' at {model_path}")

    try:
        cf_model = load_cf_model(outcome)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

    payload = payload or {}
    patient_df = build_patient_df(payload)

    try:
        X = patient_df[feature_names].values
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing required feature for model: {e}")

    try:
        ite_val = float(cf_model.effect(X)[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction error: {e}")

    interp = interpret_ite(ite_val, outcome)

    return {
        "outcome": outcome,
        "outcome_label": OUTCOME_LABELS[outcome],
        "ite": ite_val,
        "ite_percent": interp["abs_percent"],
        "direction": interp["direction"],
        "feature_order_used": feature_names,
    }

# ----------------------------
# Feature importances
# ----------------------------
@app.get("/feature-importances")
def feature_importances(
    outcome: Literal[VALID_OUTCOMES],
    top_n: Optional[int] = Query(10, ge=1, le=50)
):
    model_path = os.path.join("models", f"cf_model_{outcome}.joblib")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found for '{outcome}'")

    cf_model = load_cf_model(outcome)
    feature_names = get_feature_names_for_outcome(outcome)
    try:
        s = get_feature_importances(cf_model, feature_names, top_n=top_n, verbose=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not compute importances: {e}")

    # Return as sorted dict
    return {"outcome": outcome, "top_n": top_n, "importances": s.to_dict()}

# ----------------------------
# UMAP figure passthrough
# ----------------------------
@app.get("/umap")
def get_umap():
    path = os.path.join("clusters", "umap_plot.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="UMAP JSON not found at clusters/umap_plot.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return JSONResponse(content=data)

# ----------------------------
# Uvicorn dev entrypoint (Render will use Start Command)
# ----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
