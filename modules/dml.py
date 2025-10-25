import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from econml.dml import CausalForestDML
from scipy.stats import norm
from econml.policy import PolicyTree, PolicyForest
import joblib


def train_causal_forest(X, Y, T, model_y=None, model_t=None):
    if model_y is None:
        model_y = RandomForestRegressor(n_estimators=100, random_state=0)
    if model_t is None:
        model_t = RandomForestClassifier(n_estimators=100, random_state=0)

    cf = CausalForestDML(
        model_t=model_t,
        model_y=model_y,
        discrete_treatment=True,
        n_estimators=1000,
        min_samples_leaf=10,
        max_depth=10,
        random_state=42
    )
    cf.fit(Y, T, X=X)
    return cf


def evaluate_ate(cf, X_test):
    ate_point = cf.ate(X_test)
    ate_interval = cf.ate_interval(X_test)
    ate_se = (ate_interval[1] - ate_interval[0]) / (2 * norm.ppf(0.975))
    z_score = ate_point / ate_se
    p_value = 2 * (1 - norm.cdf(abs(z_score)))

    return {
        'ate': ate_point,
        'ci': ate_interval,
        'se': ate_se,
        'pval': p_value
    }


def plot_cates(cates, bins=50):
    plt.figure(figsize=(10, 6))
    sns.histplot(cates, bins=bins, kde=True, color='royalblue', stat='count')
    plt.title("Distribution of Estimated CATEs (LD effect)")
    plt.xlabel("Estimated Treatment Effect")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def export_cate_results(X, cates, Y, T, feature_names, outcome_name="outcome", path_prefix="cate_results"):
    filename = f"cate_results\{path_prefix}_{outcome_name}.csv"
    df_out = pd.DataFrame(X, columns=feature_names)
    df_out['treatment_effect'] = cates
    df_out[outcome_name] = Y
    df_out['treatment'] = T
    df_out.to_csv(filename, index=False)
    print(f"Exported all {len(cates)} patients' CATEs to {filename}")


def learn_policy(cf_model, X, max_depth=3):
    cate_preds = cf_model.effect(X)

    # Corrected order: X (features), y (effects)
    tree = PolicyTree(max_depth=max_depth)
    tree.fit(X, cate_preds)  # <--- this was backwards before!

    return tree

def learn_policy_forest(cf_model, X, max_depth=5, n_estimators=100):
    cate_preds = cf_model.effect(X)

    forest = PolicyForest(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    forest.fit(X, cate_preds.reshape(-1, 1))  # X first, then cate_preds

    return forest

def save_model(model, path="cf_model.joblib"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Model saved to {path}")

def load_model(path="cf_model.joblib"):
    return joblib.load(path)

def get_feature_importances(cf_model, feature_names, top_n=None, verbose=True):
    importances = cf_model.feature_importances_
    feat_imp = pd.Series(importances, index=feature_names).sort_values(ascending=False)

    if top_n:
        feat_imp = feat_imp.head(top_n)

    if verbose:
        print("Feature Importances (CausalForestDML):")
        for name, val in feat_imp.items():
            print(f"{name}: {val:.4f}")

    return feat_imp
