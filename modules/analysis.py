import os
from scipy.stats import ttest_ind, f_oneway, kruskal
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import doubleml as dml
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

# Binary Features
def analyze_binary_subgroup_cates(cate_df, df, cates, binary_features, reverse_maps):
    """
    Perform CATE difference tests and violin plots for binary subgroup features.

    Parameters:
        cate_df (pd.DataFrame): DataFrame containing feature data and treatment_effect column.
        df (pd.DataFrame): Original dataframe to restore label mappings.
        cates (np.ndarray): Array of CATE estimates.
        binary_features (list): List of binary column names to test.
        reverse_maps (dict): Dictionary mapping 0/1 back to labels.
    """
    # Inject CATEs
    cate_df['treatment_effect'] = cates

    # Restore human-readable labels
    for col in binary_features:
        label_col = col + '_label'
        if col in df and col in reverse_maps:
            cate_df[label_col] = df[col].map(reverse_maps[col])

    # Test and plot per subgroup
    for col in binary_features:
        label_col = col + '_label'
        group1, group2 = cate_df[label_col].unique()

        g1_vals = cate_df[cate_df[label_col] == group1]['treatment_effect']
        g2_vals = cate_df[cate_df[label_col] == group2]['treatment_effect']

        diff = g1_vals.mean() - g2_vals.mean()
        t_stat, p_val = ttest_ind(g1_vals, g2_vals, equal_var=False)

        print(f"\n🔬 Subgroup CATEs: {col}")
        print(f"Group 1 = {group1} → Mean CATE: {g1_vals.mean():.4f}")
        print(f"Group 2 = {group2} → Mean CATE: {g2_vals.mean():.4f}")
        print(f"ΔCATE ({group1} - {group2}) = {diff:.4f}")
        print(f"🧪 P-value: {p_val:.4f}")

        # Violin Plot
        plt.figure(figsize=(6, 4))
        sns.violinplot(x=label_col, y='treatment_effect', data=cate_df, palette='Set2', inner='box')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"CATE by {col}")
        plt.xlabel(col)
        plt.ylabel("Estimated Treatment Effect (LD)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Continuous Features
def plot_cate_vs_feature(cate_df, feature, effect_col='treatment_effect', title_prefix='Individual CATEs vs'):
    """
    Plot a scatterplot of CATEs against a continuous feature.

    Parameters:
        cate_df (pd.DataFrame): DataFrame with feature and treatment_effect columns.
        feature (str): Name of the continuous feature to plot against CATEs.
        effect_col (str): Name of the column containing treatment effects.
        title_prefix (str): Prefix for the plot title.
    """
    if feature not in cate_df.columns:
        raise ValueError(f"'{feature}' not found in DataFrame columns.")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(x=feature, y=effect_col, data=cate_df, alpha=0.5, color='darkblue', s=30)
    plt.title(f"{title_prefix} {feature}")
    plt.xlabel(feature)
    plt.ylabel("Estimated Treatment Effect (LD)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Ordinal Features
def analyze_ordinal_features(df, features, effect_col='treatment_effect'):
    """
    Visualize and statistically test CATE variation across ordinal feature levels.

    Parameters:
        df (pd.DataFrame): DataFrame containing treatment effects and ordinal features.
        features (list of str): List of ordinal feature column names.
        effect_col (str): Column name of treatment effect estimates (default: 'treatment_effect').
    """
    for col in features:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' not in DataFrame; skipping.")
            continue

        plt.figure(figsize=(7, 4))
        sns.pointplot(x=col, y=effect_col, data=df, errorbar='se', color='royalblue')
        plt.axhline(0, color='black', linestyle='--')
        plt.title(f"CATE vs {col}")
        plt.ylabel("Estimated Treatment Effect (LD)")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Group and test
        groups = [df[df[col] == val][effect_col] for val in sorted(df[col].dropna().unique())]
        if all(len(g) >= 2 for g in groups):
            fstat, pval_anova = f_oneway(*groups)
            hstat, pval_kw = kruskal(*groups)
            print(f"📊 {col} – ANOVA p = {pval_anova:.4f}, Kruskal-Wallis p = {pval_kw:.4f}")
        else:
            print(f"⚠️ Not enough samples per group in '{col}' for valid statistical tests.")

# Sensitivity Analysis
def run_doubleml_analysis(X, Y, T, feature_names, n_folds=5, score='partialling out', cf_y=0.03, cf_d=0.03, rho=1.0):
    """
    Run DoubleML partially linear regression with sensitivity analysis.

    Parameters:
        X (ndarray): Covariates
        Y (array): Outcome variable
        T (array): Treatment variable
        feature_names (list): Column names for X
        n_folds (int): Number of cross-fitting folds
        score (str): Score function for DoubleML
        cf_y (float): Confounding strength in outcome for sensitivity
        cf_d (float): Confounding strength in treatment for sensitivity
        rho (float): Correlation for adversarial confounding (1.0 = worst-case)

    Returns:
        dml_model: Trained DoubleMLPLR model
    """
    # Assemble dataset
    df_dml = pd.DataFrame(X, columns=feature_names)
    df_dml['y'] = Y
    df_dml['d'] = T

    dml_data = dml.DoubleMLData(df_dml, y_col='y', d_cols='d')

    learner_y = RandomForestRegressor(n_estimators=100, random_state=0)
    learner_t = RandomForestClassifier(n_estimators=100, random_state=0)

    dml_model = dml.DoubleMLPLR(
        dml_data,
        learner_y,
        learner_t,
        n_folds=n_folds,
        score=score
    )

    dml_model.fit()
    print("🎯 Estimated ATE via DoubleML:", dml_model.coef)
    print("\n📏 95% Confidence Interval:\n", dml_model.confint())

    # Sensitivity
    dml_model.sensitivity_analysis(cf_y=cf_y, cf_d=cf_d, rho=rho)
    print("\n🧪 Sensitivity Summary:\n", dml_model.sensitivity_summary)

    # Plot
    dml_model.sensitivity_plot().show()
    
    return dml_model
