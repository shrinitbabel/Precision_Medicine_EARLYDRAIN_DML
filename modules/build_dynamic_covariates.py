import os
import pandas as pd

def extract_daily_features(ed_daily):
    """
    Converts ed_daily (long-format) to per-patient covariate summaries.
    Applies mean aggregation per patient for key physiological metrics.
    """

    agg_cols = [
        "rr_syst", "rr_dia", "rr_map", "hb", "balance",
        "icp_7am", "icp_high", "csf"
    ]

    # Group by patient ID and take per-patient mean across ICU days
    patient_summary = ed_daily.groupby("no")[agg_cols].mean().reset_index()

    # Rename columns to reflect aggregation
    patient_summary.columns = ["no"] + [f"{col}_mean" for col in agg_cols]

    return patient_summary
