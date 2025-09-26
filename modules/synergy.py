import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
from .dml import train_causal_forest
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def prepare_variables_exclude(df, outcome_col, exclude_cols=None):
    if exclude_cols is None:
        exclude_cols = []
    
    bmi = df['weight'] / ((df['height'] / 100) ** 2)
    df['bmi'] = bmi
    df['bmi'] = df['bmi'].fillna(df['bmi'].mean())  # Fill NaN with mean

    base_features = ['age', 'hh', 'ct_ivh', 'ct_ich', 'sex', 'aneurysm_trt',
                     'aneurysm_circulation', 'mrs_adm', 'aneurysm_size', 'aneurysm_no',
                     'wfns', 'sedation_adm', 'paresis_adm', 'aphasia_adm',
                     'nimodipine', 'statin', 'mg', 'ct_modfisher', 'bmi']
    

    # Remove excluded
    feature_names = [feat for feat in base_features if feat not in exclude_cols]

    Y = df[outcome_col].values
    T = df['as_treated_flag'].values
    X = IterativeImputer(random_state=42, max_iter=10).fit_transform(df[feature_names].values)

    return X, Y, T, feature_names


def compute_synergy(df, X, Y, T1_name, T2_name, outcome, feature_names):
    """
    Estimate synergy between two binary treatments on a given outcome.

    Parameters:
        df: pandas DataFrame with original data
        X: covariates
        Y: outcome array
        T1_name: string of treatment 1 column name
        T2_name: string of treatment 2 column name
        outcome: name of outcome
        feature_names: list of X columns

    Returns:
        synergy array, summary dict
    """
    T1 = df[T1_name].values
    T2 = df[T2_name].values
    T12 = T1 * T2

    # Fit CausalForest for each treatment variable
    cf1 = train_causal_forest(X, Y, T1)
    cf2 = train_causal_forest(X, Y, T2)
    cf12 = train_causal_forest(X, Y, T12)

    # Predict CATEs
    cate1 = cf1.effect(X)
    cate2 = cf2.effect(X)
    cate12 = cf12.effect(X)

    expected = cate1 + cate2
    synergy = cate12 - expected

    # Summary
    synergy_mean = np.mean(synergy)
    synergy_std = np.std(synergy)
    tstat, pval = ttest_1samp(synergy, 0)

    summary = {
        "T1": T1_name,
        "T2": T2_name,
        "mean_synergy": synergy_mean,
        "std_synergy": synergy_std,
        "pval": pval
    }

    return synergy, summary


def plot_synergy(synergy, title="Synergy Distribution", bins=40):
    plt.figure(figsize=(8, 5), dpi=300)
    sns.histplot(synergy, kde=True, color="darkorange", bins=bins)
    plt.axhline(0, linestyle="--", color="black")
    plt.title(title)
    plt.xlabel("Observed Synergy (CATE combo - CATEs additively)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
