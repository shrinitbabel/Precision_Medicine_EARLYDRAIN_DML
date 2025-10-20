import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from .build_dynamic_covariates import extract_daily_features

def load_and_clean_data(main_path, ed_daily_path=None):
    df = pd.read_csv(main_path)

    # Binary encoding
    binary_map = {'yes': 1, 'no': 0, 'male': 1, 'female': 0,
                  'coiling': 1, 'clipping': 0, 'anterior': 1, 'posterior': 0,
                  'LD': 1, 'NoLD': 0}

    binary_cols = ['random', 'ct_ivh', 'ct_ich', 'sex', 'aneurysm_trt',
                   'aneurysm_circulation', 'shunt_180', 'sedation_adm',
                   'paresis_adm', 'aphasia_adm', 'nimodipine', 'statin', 'mg', 'infarct_trt']
    df['mrs_adm'] = 1 - df['mrs_adm']

    df['bmi'] = (df['weight'] / ((df['height'] / 100) ** 2)).fillna(df['weight'].mean())

    for col in binary_cols:
        if col in df:
            df[col] = df[col].map(binary_map)

    # Outcomes
    df['mrs_binary'] = df['mrs_180'].apply(lambda x: 1 if x <= 2 else 0)
    df['shunt_180'] = df['infarct_dch'].map({'yes': 1, 'no': 0})
    df['infarct_dch'] = df['infarct_dch'].map({'yes': 1, 'no': 0})
    df['infection_dch'] = df['infection_dch'].map({'yes': 1, 'no': 0})
    df['vs_clin'] = df['vs_clin'].map({'yes': 1, 'no': 0})
    df['gos_binary'] = df['gos8_180'].apply(lambda x: 1 if x >= 5 else 0)

    # ⛏️ Merge ed_daily features
    if ed_daily_path:
        ed_daily = pd.read_csv(ed_daily_path)
        dynamic_feats = extract_daily_features(ed_daily)

        # Drop old dynamic cols if already in df
        for col in dynamic_feats.columns:
            if col != "no" and col in df.columns:
                df.drop(columns=col, inplace=True)

        df = df.merge(dynamic_feats, on='no', how='left')

    return df



def prepare_variables(df, outcome_col):
    feature_names = [
    'age', 'hh', 'ct_ivh', 'ct_ich', 'sex', 'aneurysm_trt',
    'aneurysm_circulation', 'mrs_adm', 'aneurysm_size', 'aneurysm_no',
    'wfns', 'sedation_adm', 'paresis_adm', 'aphasia_adm',
    'nimodipine', 'statin', 'mg', 'ct_modfisher', 'bmi',
    'rr_map_mean', 'rr_syst_mean', 'rr_dia_mean', 'hb_mean',
    'balance_mean', 'icp_7am_mean', 'icp_high_mean', 'csf_mean']


    Y = df[outcome_col].values
    T = df['random'].values
    X = IterativeImputer(random_state=42, max_iter=10).fit_transform(df[feature_names].values)

    return X, Y, T, feature_names