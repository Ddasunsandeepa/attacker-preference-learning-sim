"""
============================================================
DevSecOps Adaptive Security Simulation
ML Analysis Pipeline — V1.0
============================================================
Research Title:
    Adaptive Attacker-Defender Simulation in DevSecOps
    Environments with ML-Based Security Prediction

This script:
    1. Loads and cleans the BehaviorSpace CSV dataset
    2. Performs Exploratory Data Analysis (EDA)
    3. Trains Random Forest and XGBoost regression models
    4. Evaluates and compares models (MAE, RMSE, R²)
    5. Generates SHAP explainability plots (XAI layer)
    6. Saves all plots + trained models to /outputs/

Research Questions answered:
    RQ1 — How does attacker adaptability affect success?
           → See adaptation_effect.png, shap plots
    RQ2 — How does defender capability affect resilience?
           → See correlation table, feature importance
    RQ3 — What combinations lead to highest/lowest compromise?
           → See scenario_heatmap.png
    RQ4 — Can ML predict attack outcomes from sim data?
           → See model_comparison.png, metrics table

Usage:
    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn
    python devsecops_ml_pipeline.py

Output files (all saved to ./outputs/):
    eda_distributions.png
    eda_correlation_heatmap.png
    adaptation_effect.png
    model_comparison.png
    shap_summary_beeswarm.png
    shap_importance_bar.png
    shap_dependence_skill.png
    scenario_heatmap.png
    rf_model.pkl
    xgb_model.pkl
    metrics_summary.csv
============================================================
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import shap

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION — change these paths if needed
# ============================================================

CSV_PATH    = "DevSecOps11_Simulation_V2_final DevSecOps_Datasetfinal_V1-table111.csv"
OUTPUT_DIR  = "outputs"
CSV_SKIPROWS = 6        # BehaviorSpace header rows to skip
RANDOM_SEED  = 42
TEST_SIZE    = 0.2
SHAP_SAMPLE  = 2000     # rows to use for SHAP (full 18k is slow)

# Features used as model inputs (X)
FEATURES = [
    'attacker-skill',
    'attacker-aggressiveness',
    'attacker-stealth',
    'defender-detection-rate',
    'defender-patch-speed',
    'alpha',
    'beta'
]

# Human-readable labels for plots
FEATURE_LABELS = [
    'Attacker Skill',
    'Attacker Aggressiveness',
    'Attacker Stealth',
    'Detection Rate',
    'Patch Speed',
    'Alpha (learning+)',
    'Beta (forgetting-)'
]

# Primary prediction target
PRIMARY_TARGET = 'attack-success-rate'

# All targets to evaluate
ALL_TARGETS = [
    'attack-success-rate',
    'system-resilience-score',
    'total-compromises-ever',
    'compromise-rate'
]


# ============================================================
# STEP 0 — SETUP
# ============================================================

def setup_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")


def save_fig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


# ============================================================
# STEP 1 — LOAD AND CLEAN DATA
# ============================================================

def load_data(path):
    print("=" * 60)
    print("STEP 1 — Loading dataset")
    print("=" * 60)

    df = pd.read_csv(path, skiprows=CSV_SKIPROWS)

    print(f"  Rows loaded     : {len(df):,}")
    print(f"  Columns         : {len(df.columns)}")
    print(f"  Null values     : {df.isnull().sum().sum()}")
    print(f"  Columns present :")
    for c in df.columns:
        print(f"    {c}")

    # Drop metadata columns not used in modelling
    drop_cols = ['[run number]', '[step]']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    print(f"\n  Working shape after dropping metadata: {df.shape}")
    return df


# ============================================================
# STEP 2 — EXPLORATORY DATA ANALYSIS
# ============================================================

def run_eda(df):
    print("\n" + "=" * 60)
    print("STEP 2 — Exploratory Data Analysis")
    print("=" * 60)

    # 2a: Summary statistics
    print("\n--- Feature summary ---")
    print(df[FEATURES].describe().round(3).to_string())

    print("\n--- Target summary ---")
    print(df[ALL_TARGETS].describe().round(3).to_string())

    # 2b: Correlation table
    print("\n--- Correlation with attack-success-rate ---")
    corr = df[FEATURES + ALL_TARGETS].corr()[PRIMARY_TARGET].drop(PRIMARY_TARGET)
    print(corr.sort_values(ascending=False).round(3).to_string())

    # 2c: Adaptation effect table
    print("\n--- Adaptation effect by alpha ---")
    grp = df.groupby('alpha').agg(
        attack_success=(PRIMARY_TARGET, 'mean'),
        compromises=('total-compromises-ever', 'mean'),
        resilience=('system-resilience-score', 'mean'),
        top_pref=('top-preference', 'mean')
    ).round(3)
    print(grp.to_string())

    # --- Plot: EDA distributions ---
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Feature Distributions — DevSecOps Simulation Dataset',
                 fontsize=14, fontweight='bold')
    axes = axes.flatten()
    for i, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        axes[i].hist(df[feat], bins=20, color='#4A90D9', edgecolor='white',
                     linewidth=0.5, alpha=0.85)
        axes[i].set_title(label, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Count')
    axes[7].hist(df[PRIMARY_TARGET], bins=30, color='#E8632A',
                 edgecolor='white', linewidth=0.5, alpha=0.85)
    axes[7].set_title('Attack Success Rate (target)', fontweight='bold', fontsize=10)
    axes[7].set_xlabel('Value')
    plt.tight_layout()
    save_fig('eda_distributions.png')

    # --- Plot: Correlation heatmap ---
    fig, ax = plt.subplots(figsize=(10, 8))
    corr_matrix = df[FEATURES + [PRIMARY_TARGET, 'system-resilience-score']].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=FEATURE_LABELS + ['Attack Success', 'Resilience'],
                yticklabels=FEATURE_LABELS + ['Attack Success', 'Resilience'],
                ax=ax)
    ax.set_title('Feature Correlation Matrix', fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    save_fig('eda_correlation_heatmap.png')

    # --- Plot: Adaptation effect (RQ1 evidence) ---
    alpha_grp = df.groupby('alpha').agg(
        top_pref=('top-preference', 'mean'),
        compromises=('total-compromises-ever', 'mean')
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('RQ1 Evidence — Adaptation Effect (Alpha vs Outcomes)',
                 fontsize=13, fontweight='bold')

    axes[0].plot(alpha_grp['alpha'], alpha_grp['top_pref'],
                 'o-', color='#E8632A', linewidth=2.5, markersize=10)
    axes[0].set_xlabel('Alpha (attacker learning rate)', fontweight='bold')
    axes[0].set_ylabel('Average top preference score', fontweight='bold')
    axes[0].set_title('Higher Alpha → Stronger Target Fixation\n'
                      '(Adaptive attacker learns preferred targets)', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, alpha=0.3)
    for _, row in alpha_grp.iterrows():
        axes[0].annotate(f"{row['top_pref']:.3f}",
                         (row['alpha'], row['top_pref']),
                         textcoords='offset points', xytext=(0, 12),
                         ha='center', fontsize=11, fontweight='bold')

    axes[1].plot(alpha_grp['alpha'], alpha_grp['compromises'],
                 's-', color='#C0392B', linewidth=2.5, markersize=10)
    axes[1].set_xlabel('Alpha (attacker learning rate)', fontweight='bold')
    axes[1].set_ylabel('Average total compromises per run', fontweight='bold')
    axes[1].set_title('Alpha vs Compromise Count\n'
                      '(Stronger learning → marginal attack improvement)', fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    for _, row in alpha_grp.iterrows():
        axes[1].annotate(f"{row['compromises']:.2f}",
                         (row['alpha'], row['compromises']),
                         textcoords='offset points', xytext=(0, 10),
                         ha='center', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_fig('adaptation_effect.png')

    # --- Plot: Scenario heatmap (RQ3) ---
    pivot = df.groupby(['attacker-skill', 'defender-detection-rate'])[PRIMARY_TARGET].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=0.65)
    ax.set_title('RQ3 — Attack Success Rate by Skill × Detection Rate\n'
                 '(Mean across all runs)', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Defender Detection Rate', fontweight='bold')
    ax.set_ylabel('Attacker Skill', fontweight='bold')
    plt.tight_layout()
    save_fig('scenario_heatmap.png')


# ============================================================
# STEP 3 — TRAIN MODELS
# ============================================================

def train_models(df):
    print("\n" + "=" * 60)
    print("STEP 3 — Model Training")
    print("=" * 60)

    X = df[FEATURES].values
    y = df[PRIMARY_TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )
    print(f"  Train size: {len(X_train):,}  |  Test size: {len(X_test):,}")

    # --- Random Forest ---
    print("\n  Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    rf_metrics = {
        'MAE':  round(mean_absolute_error(y_test, y_pred_rf), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_rf)), 4),
        'R2':   round(r2_score(y_test, y_pred_rf), 4)
    }
    print(f"    MAE={rf_metrics['MAE']}  RMSE={rf_metrics['RMSE']}  R²={rf_metrics['R2']}")

    # --- XGBoost ---
    print("\n  Training XGBoost...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        verbosity=0
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)

    xgb_metrics = {
        'MAE':  round(mean_absolute_error(y_test, y_pred_xgb), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_xgb)), 4),
        'R2':   round(r2_score(y_test, y_pred_xgb), 4)
    }
    print(f"    MAE={xgb_metrics['MAE']}  RMSE={xgb_metrics['RMSE']}  R²={xgb_metrics['R2']}")

    # --- Cross-validation ---
    print("\n  Running 5-fold cross-validation on XGBoost...")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for tr_idx, val_idx in kf.split(X):
        m_cv = xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6,
                                  subsample=0.8, colsample_bytree=0.8,
                                  random_state=RANDOM_SEED, verbosity=0)
        m_cv.fit(X[tr_idx], y[tr_idx])
        cv_scores.append(r2_score(y[val_idx], m_cv.predict(X[val_idx])))
    cv_scores = np.array(cv_scores)
    print(f"    CV R² scores: {[round(s,3) for s in cv_scores]}")
    print(f"    CV mean R²:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # --- Plot: Model comparison ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle('ML Model Comparison — Predicting Attack Success Rate',
                 fontsize=13, fontweight='bold')

    models    = ['Random Forest', 'XGBoost']
    r2_vals   = [rf_metrics['R2'],  xgb_metrics['R2']]
    rmse_vals = [rf_metrics['RMSE'], xgb_metrics['RMSE']]
    mae_vals  = [rf_metrics['MAE'],  xgb_metrics['MAE']]
    colors    = ['#4A90D9', '#E8632A']

    for ax, vals, title, ylim, better in zip(
        axes,
        [r2_vals, rmse_vals, mae_vals],
        ['R² Score', 'RMSE', 'MAE'],
        [1.0, 0.12, 0.08],
        ['higher', 'lower', 'lower']
    ):
        bars = ax.bar(models, vals, color=colors, width=0.5,
                      edgecolor='white', linewidth=1.5)
        ax.set_title(f'{title}\n({better} is better)', fontweight='bold')
        ax.set_ylim(0, ylim)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    v + ylim * 0.02, f'{v:.4f}',
                    ha='center', fontsize=12, fontweight='bold')

    plt.tight_layout()
    save_fig('model_comparison.png')

    # --- Save metrics CSV ---
    metrics_df = pd.DataFrame({
        'Model':   ['RandomForest', 'XGBoost'],
        'MAE':     [rf_metrics['MAE'],  xgb_metrics['MAE']],
        'RMSE':    [rf_metrics['RMSE'], xgb_metrics['RMSE']],
        'R2':      [rf_metrics['R2'],   xgb_metrics['R2']],
        'Target':  [PRIMARY_TARGET, PRIMARY_TARGET]
    })
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics_summary.csv')
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n  Metrics saved: metrics_summary.csv")

    # --- Save models ---
    with open(os.path.join(OUTPUT_DIR, 'rf_model.pkl'), 'wb') as f:
        pickle.dump(rf, f)
    with open(os.path.join(OUTPUT_DIR, 'xgb_model.pkl'), 'wb') as f:
        pickle.dump(xgb_model, f)
    print("  Models saved:  rf_model.pkl, xgb_model.pkl")

    return rf, xgb_model, X, X_test, y_test


# ============================================================
# STEP 4 — XAI WITH SHAP
# ============================================================

def run_shap(xgb_model, X, df):
    print("\n" + "=" * 60)
    print("STEP 4 — XAI: SHAP Explainability")
    print("=" * 60)

    # Sample for speed
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), SHAP_SAMPLE, replace=False)
    X_sample = pd.DataFrame(X[idx], columns=FEATURES)

    print(f"  Computing SHAP values on {SHAP_SAMPLE} sample rows...")
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)

    # --- Plot 1: Beeswarm summary ---
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=FEATURE_LABELS,
                      show=False)
    plt.title('SHAP — Feature Impact on Attack Success Rate\n'
              '(Red = high feature value, Blue = low feature value)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    save_fig('shap_summary_beeswarm.png')

    # --- Plot 2: Bar importance ---
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_sample,
                      feature_names=FEATURE_LABELS,
                      plot_type='bar', show=False)
    plt.title('SHAP — Mean Absolute Feature Importance',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    save_fig('shap_importance_bar.png')

    # --- Plot 3: Dependence plot for top feature (attacker-skill = index 0) ---
    plt.figure(figsize=(9, 5))
    shap.dependence_plot(
        0, shap_values, X_sample.values,
        feature_names=FEATURE_LABELS,
        interaction_index=3,
        show=False, alpha=0.4
    )
    plt.title('SHAP Dependence — Attacker Skill\n'
              '(Colour = Defender Detection Rate)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    save_fig('shap_dependence_skill.png')

    # --- Print top SHAP values ---
    mean_shap = np.abs(shap_values).mean(axis=0)
    print("\n  SHAP mean |value| per feature:")
    for label, val in sorted(zip(FEATURE_LABELS, mean_shap), key=lambda x: -x[1]):
        bar = '█' * int(val * 200)
        print(f"    {label:<25} {val:.4f}  {bar}")


# ============================================================
# STEP 5 — FEATURE IMPORTANCE COMPARISON PLOT
# ============================================================

def plot_feature_importance(rf, xgb_model):
    print("\n" + "=" * 60)
    print("STEP 5 — Feature Importance Comparison")
    print("=" * 60)

    rf_imp  = rf.feature_importances_
    xgb_imp = xgb_model.feature_importances_

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Feature Importance — RF vs XGBoost', fontsize=13, fontweight='bold')

    for ax, imp, title, color in zip(
        axes,
        [rf_imp, xgb_imp],
        ['Random Forest', 'XGBoost'],
        ['#4A90D9', '#E8632A']
    ):
        idx_sorted = np.argsort(imp)
        ax.barh([FEATURE_LABELS[i] for i in idx_sorted],
                [imp[i] for i in idx_sorted],
                color=color, edgecolor='white', linewidth=0.8)
        ax.set_title(title, fontweight='bold', fontsize=12)
        ax.set_xlabel('Feature Importance Score')
        ax.set_xlim(0, max(imp) * 1.15)
        for i, v in enumerate(sorted(imp)):
            ax.text(v + max(imp) * 0.01, i, f'{v:.3f}',
                    va='center', fontsize=9)

    plt.tight_layout()
    save_fig('feature_importance_comparison.png')
    print("  Saved: feature_importance_comparison.png")


# ============================================================
# MAIN
# ============================================================

# ============================================================
# STEP 6 — RESIDUAL PLOT (Predicted vs Actual)
# ============================================================

def plot_residuals(rf, xgb_model, X_test, y_test):
    print("\n" + "=" * 60)
    print("STEP 6 — Residual Analysis (Predicted vs Actual)")
    print("=" * 60)

    y_pred_rf  = rf.predict(X_test)
    y_pred_xgb = xgb_model.predict(X_test)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Predicted vs Actual — Attack Success Rate\n(Model Residual Analysis)',
                 fontsize=14, fontweight='bold')

    for ax, y_pred, name, color in zip(
        axes,
        [y_pred_rf, y_pred_xgb],
        ['Random Forest', 'XGBoost'],
        ['#4A90D9', '#E8632A']
    ):
        residuals = y_test - y_pred
        r2 = r2_score(y_test, y_pred)
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]

        ax.scatter(y_pred, y_test, alpha=0.15, s=8, color=color, label='Data points')
        ax.plot(lims, lims, 'k--', linewidth=1.5, label='Perfect prediction', zorder=5)
        ax.set_xlabel('Predicted Attack Success Rate', fontweight='bold', fontsize=11)
        ax.set_ylabel('Actual Attack Success Rate', fontweight='bold', fontsize=11)
        ax.set_title(f'{name}  —  R² = {r2:.4f}', fontweight='bold', fontsize=12)
        ax.legend(fontsize=9)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.text(0.05, 0.92,
                f'Mean residual: {residuals.mean():.4f}\nStd residual:  {residuals.std():.4f}',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8))

    plt.tight_layout()
    save_fig('residual_plot.png')

    for y_pred, name in [(y_pred_rf,'RF'), (y_pred_xgb,'XGBoost')]:
        r = y_test - y_pred
        print(f"  {name:<12} mean residual={r.mean():.4f}  std={r.std():.4f}  "
              f"max_over={r.max():.4f}  max_under={r.min():.4f}")


# ============================================================
# STEP 7 — SHAP WATERFALL (3 scenario explanations)
# ============================================================

def plot_shap_waterfall(xgb_model, X):
    print("\n" + "=" * 60)
    print("STEP 7 — SHAP Waterfall (Scenario Explanations)")
    print("=" * 60)

    X_df = pd.DataFrame(X, columns=FEATURES)
    idx  = np.random.RandomState(RANDOM_SEED).choice(len(X), SHAP_SAMPLE, replace=False)

    explainer  = shap.TreeExplainer(xgb_model)
    base_val   = float(np.atleast_1d(explainer.expected_value)[0])

    # Three representative security scenarios
    scenarios = {
        'High Risk\n(Skill=0.9, Detection=0.2)': [0.9, 0.8, 0.3, 0.2, 0.2, 0.2, 0.1],
        'Balanced\n(Skill=0.5, Detection=0.5)':  [0.5, 0.7, 0.4, 0.5, 0.5, 0.2, 0.1],
        'Low Risk\n(Skill=0.1, Detection=0.8)':  [0.1, 0.4, 0.1, 0.8, 0.8, 0.2, 0.1],
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('SHAP Waterfall — How Each Feature Drives the Risk Prediction\n'
                 '(Three representative DevSecOps security scenarios)',
                 fontsize=13, fontweight='bold')

    for ax, (title, inputs) in zip(axes, scenarios.items()):
        x_single = np.array(inputs).reshape(1, -1)
        sv_single = explainer.shap_values(
            pd.DataFrame(x_single, columns=FEATURES)
        )[0]
        pred = xgb_model.predict(x_single)[0]

        contributions = sorted(
            zip(FEATURE_LABELS, sv_single, inputs),
            key=lambda x: abs(x[1]), reverse=True
        )
        labels = [f'{lbl}\n(val={val:.2f})' for lbl, _, val in contributions]
        values = [sv for _, sv, _ in contributions]
        colors = ['#E8632A' if v > 0 else '#4A90D9' for v in values]

        bars = ax.barh(labels, values, color=colors, edgecolor='white', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)

        for bar, v in zip(bars, values):
            ax.text(v + (0.001 if v >= 0 else -0.001),
                    bar.get_y() + bar.get_height() / 2,
                    f'{v:+.4f}', va='center',
                    ha='left' if v >= 0 else 'right',
                    fontsize=8.5, fontweight='bold')

        ax.set_title(f'{title}\nPredicted: {pred:.3f}  |  Base: {base_val:.3f}',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('SHAP contribution to attack success rate', fontsize=9)
        ax.text(0.98, 0.02,
                '■ Orange = increases risk\n■ Blue   = decreases risk',
                transform=ax.transAxes, fontsize=7.5, ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

        print(f"  {title.split(chr(10))[0]:<30} predicted={pred:.3f}")

    plt.tight_layout()
    save_fig('shap_waterfall_scenarios.png')


def main():
    print("\n" + "=" * 60)
    print("DevSecOps ML Pipeline — Starting")
    print("=" * 60 + "\n")

    setup_output_dir()

    # Check CSV exists
    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found at '{CSV_PATH}'")
        print("Update CSV_PATH at the top of this script.")
        sys.exit(1)

    df = load_data(CSV_PATH)

    run_eda(df)

    rf, xgb_model, X, X_test, y_test = train_models(df)

    run_shap(xgb_model, X, df)

    plot_feature_importance(rf, xgb_model)

    plot_residuals(rf, xgb_model, X_test, y_test)

    plot_shap_waterfall(xgb_model, X)

    print("\n" + "=" * 60)
    print("Pipeline complete — ALL 14 outputs generated.")
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 60)
    print("""
Outputs generated:
  eda_distributions.png          — Feature histograms
  eda_correlation_heatmap.png    — Correlation matrix
  adaptation_effect.png          — RQ1 evidence (alpha effect)
  scenario_heatmap.png           — RQ3 skill × detection matrix
  model_comparison.png           — RF vs XGBoost (MAE/RMSE/R²)
  shap_summary_beeswarm.png      — SHAP beeswarm (XAI)
  shap_importance_bar.png        — SHAP bar chart (XAI)
  shap_dependence_skill.png      — SHAP dependence plot (XAI)
  feature_importance_comparison.png — RF vs XGBoost importance
  residual_plot.png              — Predicted vs Actual (model validation)
  shap_waterfall_scenarios.png   — Per-scenario risk explanation (XAI)
  rf_model.pkl                   — Saved Random Forest model
  xgb_model.pkl                  — Saved XGBoost model
  metrics_summary.csv            — Model metrics table

Next step: Web dashboard using these saved models.
    """)


if __name__ == '__main__':
    main()