"""
============================================================
DevSecOps Adaptive Security Simulation
ML Analysis Pipeline — V2.0  (Publication Grade + Novel Algorithm)
============================================================
Research Title:
    Adaptive Attacker-Defender Simulation in DevSecOps
    Environments with Explainable ML-Based Security Prediction

V2 ADDS over V1:
    1. ARIS-OPT — Optimized Adaptive Risk Intelligence Score
       Replaces hardcoded weights (0.4/0.3/0.3) with weights
       mathematically derived via constrained Bayesian optimization
       using scipy.optimize.minimize (SLSQP method).
       Objective: maximize Spearman correlation with total-compromises-ever
       (the most direct security outcome in the dataset).
       Constraint: w1 + w2 + w3 = 1, all wi in [0.05, 0.95]
       This makes ARIS weights data-driven and reviewer-defensible.

    2. Statistical significance: Wilcoxon → Friedman + Nemenyi post-hoc
       Friedman test is the correct non-parametric test for comparing
       multiple ML models across multiple CV folds simultaneously.
       Nemenyi post-hoc test shows which specific pairs differ.
       Replaces the suspicious identical W=15 results from 5-fold Wilcoxon.

    3. 10-fold CV instead of 5-fold
       More folds = more stable estimates, better statistical power.

    4. Ablation study
       Removes one feature at a time, shows ΔR² impact.
       Confirms SHAP findings scientifically.
       Attacker Skill removal should produce largest ΔR².

    5. Weight sensitivity analysis for ARIS-OPT
       Shows how ARIS changes when optimized weights vary ±0.1
       Proves the index is robust, not brittle.

Usage:
    pip install pandas numpy scikit-learn xgboost lightgbm
               catboost shap matplotlib seaborn scipy scikit-posthocs
    python devsecops_ml_pipeline_v2.py [--csv PATH] [--output-dir DIR] [--skiprows N]

Outputs (./outputs/) — all V2 outputs PLUS:
    aris_weight_optimization.png   — optimization convergence + weight bar chart
    aris_weight_sensitivity.png    — sensitivity of ARIS to weight perturbations
    friedman_nemenyi.png           — statistical significance heatmap
    ablation_study.png             — per-feature ablation ΔR²
    full_results_summary.csv       — all results including ARIS-OPT
============================================================
"""

import os
import sys
import pickle
import argparse
import warnings
import time
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.optimize import minimize

from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                               GradientBoostingRegressor)
from sklearn.model_selection import (train_test_split, KFold,
                                      GridSearchCV, cross_val_score)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import shap

try:
    import scikit_posthocs as sp
    HAS_POSTHOCS = True
except ImportError:
    HAS_POSTHOCS = False
    print("  NOTE: scikit-posthocs not installed. "
          "Run: pip install scikit-posthocs  for Nemenyi post-hoc test.")

# Only silence the noisy, known-benign warnings rather than everything —
# a blanket filterwarnings('ignore') can hide real convergence/data issues.
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ============================================================
# CONFIGURATION
# ============================================================

CSV_PATH     = "DevSecOps11_Simulation_V2_final DevSecOps_Datasetfinal_V1-table111.csv"
OUTPUT_DIR   = "outputs"
CSV_SKIPROWS = 6
RANDOM_SEED  = 42
TEST_SIZE    = 0.2
SHAP_SAMPLE  = 2000
N_CV_FOLDS   = 10   # Increased from 5 to 10 for more stable estimates + better stat power

FEATURES = [
    'attacker-skill', 'attacker-aggressiveness', 'attacker-stealth',
    'defender-detection-rate', 'defender-patch-speed', 'alpha', 'beta'
]
FEATURE_LABELS = [
    'Attacker Skill', 'Attacker Aggressiveness', 'Attacker Stealth',
    'Detection Rate', 'Patch Speed', 'Alpha (learning+)', 'Beta (forgetting-)'
]

PRIMARY_TARGET = 'attack-success-rate'
ALL_TARGETS    = ['attack-success-rate', 'system-resilience-score',
                  'total-compromises-ever', 'compromise-rate']

# Columns the rest of the pipeline assumes exist — checked right after load
# so a schema mismatch fails fast with a clear message instead of a random
# KeyError deep inside EDA/ARIS/plotting.
REQUIRED_COLUMNS = FEATURES + ALL_TARGETS + ['top-preference']


def parse_args():
    p = argparse.ArgumentParser(description="DevSecOps ML Pipeline V2 — Publication Grade")
    p.add_argument('--csv', default=CSV_PATH, help="Path to the BehaviorSpace CSV export")
    p.add_argument('--output-dir', default=OUTPUT_DIR, help="Directory to write plots/models/results to")
    p.add_argument('--skiprows', type=int, default=CSV_SKIPROWS, help="Header rows to skip in the CSV")
    return p.parse_args()

# ARIS_W1/W2/W3 are now COMPUTED by optimize_aris_weights() — NOT hardcoded.
# These are placeholders overwritten at runtime.
ARIS_W1 = None
ARIS_W2 = None
ARIS_W3 = None

# Colour palette (consistent across all plots)
MODEL_COLORS = {
    'Linear Regression':    '#888888',
    'Decision Tree':        '#F59E0B',
    'Extra Trees':          '#06B6D4',
    'Random Forest':        '#4A90D9',
    'Gradient Boosting':    '#10B981',
    'XGBoost':              '#E8632A',
    'LightGBM':             '#7C3AED',
    'CatBoost':             '#EF4444',
}


# ============================================================
# UTILITIES
# ============================================================

def setup():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {os.path.abspath(OUTPUT_DIR)}\n")


def savefig(name):
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {name}")


def section(title):
    print("\n" + "=" * 64)
    print(f"  {title}")
    print("=" * 64)


def cv_score(model, X, y, kf):
    """K-fold CV returning mean R², std, and the per-fold score list."""
    scores = []
    for tr, val in kf.split(X):
        m = clone(model)
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[val], m.predict(X[val])))
    return np.mean(scores), np.std(scores), scores


# ============================================================
# STEP 1 — LOAD DATA
# ============================================================

def load_data():
    section("STEP 1 — Loading dataset")
    df = pd.read_csv(CSV_PATH, skiprows=CSV_SKIPROWS)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"\nERROR: CSV is missing required column(s): {missing}")
        print("Check --skiprows / the BehaviorSpace export format.")
        sys.exit(1)
    drop = ['[run number]', '[step]']
    df = df.drop(columns=[c for c in drop if c in df.columns])
    print(f"  Rows: {len(df):,}  |  Columns: {len(df.columns)}")
    print(f"  Nulls: {df.isnull().sum().sum()}")
    print(f"  Target mean: {df[PRIMARY_TARGET].mean():.4f}  "
          f"std: {df[PRIMARY_TARGET].std():.4f}  "
          f"range: [{df[PRIMARY_TARGET].min():.3f}, {df[PRIMARY_TARGET].max():.3f}]")
    return df


# ============================================================
# STEP 2 — ARIS-OPT (NOVEL ALGORITHM)
# ============================================================

def optimize_aris_weights(df):
    """
    ARIS-OPT: Optimized Adaptive Risk Intelligence Score.

    ALGORITHM DESCRIPTION (for paper Section 3):
    ─────────────────────────────────────────────
    Instead of assigning weights by researcher judgment, ARIS-OPT
    uses constrained numerical optimization to find weights w1, w2, w3
    that maximize the Spearman rank correlation between the composite
    index and total-compromises-ever (the most direct security outcome).

    Formally:
        maximize   ρ(ARIS(w), total_compromises_ever)
        subject to w1 + w2 + w3 = 1
                   0.05 ≤ wi ≤ 0.95  for i = 1, 2, 3

    Where:
        ARIS(w) = w1 × attack_success_rate
                + w2 × compromise_rate
                + w3 × (1 − system_resilience_score)

    Optimization method: SLSQP (Sequential Least Squares Programming)
    from scipy.optimize.minimize — a gradient-based method suitable
    for smooth constrained optimization.

    This makes the weights mathematically derived from the data,
    not chosen by the researcher — which is the key novelty claim.
    """
    section("STEP 2 — ARIS-OPT: Optimized Weight Discovery (Novel Algorithm)")

    asr   = df['attack-success-rate'].values
    cr    = df['compromise-rate'].values
    rl    = (1 - df['system-resilience-score'].values)
    tc    = df['total-compromises-ever'].values

    # Stack components into matrix for clean vectorized computation
    components = np.stack([asr, cr, rl], axis=1)   # shape (N, 3)

    # ── Objective: negative Spearman correlation (minimize = maximize ρ) ─────
    convergence_log = []   # track optimization progress for plot

    def objective(w):
        aris_raw = components @ w          # weighted sum, shape (N,)
        rho, _   = stats.spearmanr(aris_raw, tc)
        convergence_log.append(-rho)       # log for convergence plot
        return -rho                        # negative because we minimize

    # ── Constraints and bounds ────────────────────────────────────────────────
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}]
    bounds      = [(0.05, 0.95)] * 3
    w0          = np.array([1/3, 1/3, 1/3])   # uniform starting point

    print("  Optimization objective: maximize Spearman ρ(ARIS, total_compromises_ever)")
    print("  Method: SLSQP  |  Constraint: w1+w2+w3=1  |  Bounds: [0.05, 0.95]")
    print(f"  Initial weights (uniform): w1={w0[0]:.3f}  w2={w0[1]:.3f}  w3={w0[2]:.3f}")

    result = minimize(objective, w0,
                      method='SLSQP',
                      bounds=bounds,
                      constraints=constraints,
                      options={'ftol': 1e-9, 'maxiter': 1000})

    w_opt = result.x
    w1_opt, w2_opt, w3_opt = w_opt

    # Compute initial ρ for comparison
    rho_init, _ = stats.spearmanr(components @ w0, tc)
    rho_opt,  _ = stats.spearmanr(components @ w_opt, tc)

    print(f"\n  Optimized weights:")
    print(f"    w1 (attack_success_rate) = {w1_opt:.4f}")
    print(f"    w2 (compromise_rate)     = {w2_opt:.4f}")
    print(f"    w3 (1-resilience_score)  = {w3_opt:.4f}")
    print(f"    Sum                      = {w_opt.sum():.6f}  (must = 1.0)")
    print(f"\n  Spearman ρ improvement:")
    print(f"    Uniform weights (1/3,1/3,1/3): ρ = {rho_init:.4f}")
    print(f"    Optimized weights:              ρ = {rho_opt:.4f}")
    print(f"    Improvement:                    Δρ = +{rho_opt - rho_init:.4f}")
    print(f"\n  Optimizer converged: {result.success}  |  Iterations: {result.nit}")

    # Compare with V2 hardcoded weights
    w_v2 = np.array([0.4, 0.3, 0.3])
    rho_v2, _ = stats.spearmanr(components @ w_v2, tc)
    print(f"\n  Comparison with V2 hardcoded (0.4, 0.3, 0.3): ρ = {rho_v2:.4f}")
    print(f"  ARIS-OPT improvement over hardcoded:           Δρ = +{rho_opt - rho_v2:.4f}")

    # ── Plot 1: Weight comparison bar chart + convergence ────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle('ARIS-OPT — Mathematically Optimized Composite Risk Index\n'
                 '(Weights derived via SLSQP constrained optimization)',
                 fontsize=13, fontweight='bold')

    # Weight comparison: uniform vs V2 hardcoded vs ARIS-OPT
    weight_labels = ['Attack\nSuccess', 'Compromise\nRate', '1 − Resilience']
    x = np.arange(3)
    width = 0.25
    axes[0].bar(x - width, w0,     width, label='Uniform (1/3)', color='#888888', alpha=0.8)
    axes[0].bar(x,         w_v2,   width, label='V2 Hardcoded', color='#4A90D9', alpha=0.8)
    axes[0].bar(x + width, w_opt,  width, label='ARIS-OPT', color='#E8632A', alpha=0.9)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(weight_labels, fontsize=10)
    axes[0].set_ylabel('Weight Value')
    axes[0].set_title('Weight Comparison\n(Three weight schemes)', fontweight='bold')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=9)
    axes[0].axhline(1/3, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    for i, (u, v2, opt) in enumerate(zip(w0, w_v2, w_opt)):
        axes[0].text(i - width, u + 0.02, f'{u:.3f}', ha='center', fontsize=8)
        axes[0].text(i,         v2 + 0.02, f'{v2:.3f}', ha='center', fontsize=8)
        axes[0].text(i + width, opt + 0.02, f'{opt:.3f}', ha='center', fontsize=8, fontweight='bold')

    # Spearman ρ comparison
    schemes = ['Uniform\n(1/3,1/3,1/3)', 'V2 Hardcoded\n(0.4,0.3,0.3)', 'ARIS-OPT\n(optimized)']
    rho_vals = [rho_init, rho_v2, rho_opt]
    colors_bar = ['#888888', '#4A90D9', '#E8632A']
    bars = axes[1].bar(schemes, rho_vals, color=colors_bar, width=0.5, edgecolor='white')
    axes[1].set_title('Spearman ρ with Total Compromises\n(Higher = better ARIS quality)',
                      fontweight='bold')
    axes[1].set_ylabel('Spearman ρ')
    axes[1].set_ylim(0.8, 1.0)
    for bar, v in zip(bars, rho_vals):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     v + 0.002, f'ρ={v:.4f}',
                     ha='center', fontsize=10, fontweight='bold')

    # Convergence curve
    axes[2].plot(range(len(convergence_log)), convergence_log,
                 color='#E8632A', linewidth=1.5)
    axes[2].set_xlabel('Optimization Iteration')
    axes[2].set_ylabel('Objective (−Spearman ρ)')
    axes[2].set_title('SLSQP Convergence Curve\n(Decreasing = improving)',
                      fontweight='bold')
    axes[2].axhline(convergence_log[-1], color='#10B981', linestyle='--',
                    linewidth=1.5, label=f'Final: {-convergence_log[-1]:.4f}')
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('aris_weight_optimization.png')

    # ── Plot 2: Weight sensitivity analysis ──────────────────────────────────
    # Perturb each weight ±0.05, ±0.10 and re-normalize, show effect on ρ
    perturbations = np.arange(-0.15, 0.20, 0.05)
    rho_sensitivity = {name: [] for name in ['w1 (Attack)', 'w2 (Compromise)', 'w3 (Resilience)']}

    for delta in perturbations:
        for i, key in enumerate(rho_sensitivity):
            w_perturbed = w_opt.copy()
            w_perturbed[i] = np.clip(w_opt[i] + delta, 0.01, 0.99)
            w_perturbed = w_perturbed / w_perturbed.sum()   # re-normalize
            rho_p, _ = stats.spearmanr(components @ w_perturbed, tc)
            rho_sensitivity[key].append(rho_p)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_sens = ['#E8632A', '#4A90D9', '#10B981']
    for (key, vals), color in zip(rho_sensitivity.items(), colors_sens):
        ax.plot(perturbations, vals, 'o-', color=color, linewidth=2,
                markersize=6, label=key)
    ax.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5,
               label='Optimal weights (Δ=0)')
    ax.axhline(rho_opt, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.set_xlabel('Weight Perturbation (Δ applied to each weight individually)',
                  fontweight='bold')
    ax.set_ylabel('Spearman ρ (ARIS vs Total Compromises)', fontweight='bold')
    ax.set_title('ARIS-OPT Weight Sensitivity Analysis\n'
                 '(Stable ρ across perturbations = robust index)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig('aris_weight_sensitivity.png')

    return w1_opt, w2_opt, w3_opt, rho_opt


def compute_aris(df, w1, w2, w3):
    """
    Compute ARIS using optimized weights.
    Called after optimize_aris_weights() returns the data-driven weights.
    """
    aris_raw = (w1 * df['attack-success-rate']
              + w2 * df['compromise-rate']
              + w3 * (1 - df['system-resilience-score']))

    # Normalise to [0,1]
    aris_min, aris_max = aris_raw.min(), aris_raw.max()
    aris = (aris_raw - aris_min) / (aris_max - aris_min)
    df['ARIS'] = aris

    print(f"\n  ARIS-OPT formula: {w1:.4f}×ASR + {w2:.4f}×CR + {w3:.4f}×(1-RS)")
    print(f"  ARIS stats: mean={aris.mean():.4f}  std={aris.std():.4f}  "
          f"range=[{aris.min():.3f}, {aris.max():.3f}]")

    # Validate correlations
    tc = df['total-compromises-ever']
    print("\n  Final component correlations with total-compromises-ever:")
    for col in ['attack-success-rate', 'compromise-rate', 'system-resilience-score', 'ARIS']:
        r, p = stats.pearsonr(df[col], tc)
        print(f"    {col:<35} r={r:+.3f}  p={'<0.001' if p < 0.001 else f'{p:.3f}'}")

    # --- Plot: ARIS distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('ARIS-OPT — Adaptive Risk Intelligence Score (Optimized Weights)\n'
                 f'Formula: {w1:.3f}×Attack Success + {w2:.3f}×Compromise Rate '
                 f'+ {w3:.3f}×(1−Resilience)',
                 fontsize=11, fontweight='bold')

    axes[0].hist(aris, bins=40, color='#E8632A', edgecolor='white',
                 linewidth=0.5, alpha=0.85)
    axes[0].set_title('ARIS-OPT Distribution', fontweight='bold')
    axes[0].set_xlabel('ARIS Score (0=safe, 1=critical risk)')
    axes[0].set_ylabel('Count')
    axes[0].axvline(aris.mean(), color='#7C3AED', linewidth=2,
                    linestyle='--', label=f'Mean={aris.mean():.3f}')
    axes[0].legend()

    axes[1].scatter(df['attack-success-rate'], aris,
                    alpha=0.08, s=4, color='#4A90D9')
    axes[1].set_title('ARIS-OPT vs Attack Success Rate', fontweight='bold')
    axes[1].set_xlabel('Attack Success Rate')
    axes[1].set_ylabel('ARIS Score')
    r_val, _ = stats.pearsonr(df['attack-success-rate'], aris)
    axes[1].text(0.05, 0.92, f'r = {r_val:.3f}', transform=axes[1].transAxes,
                 fontsize=11, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    savefig('aris_distribution.png')

    return df


# ============================================================
# STEP 3 — EDA
# ============================================================

def run_eda(df):
    section("STEP 3 — Exploratory Data Analysis")

    print("\n--- Feature summary ---")
    print(df[FEATURES].describe().round(3).to_string())
    print("\n--- Target summary ---")
    print(df[ALL_TARGETS + ['ARIS']].describe().round(3).to_string())

    corr = df[FEATURES + [PRIMARY_TARGET, 'ARIS']].corr()
    print(f"\n--- Top correlations with {PRIMARY_TARGET} ---")
    c = corr[PRIMARY_TARGET].drop(PRIMARY_TARGET).sort_values(ascending=False)
    print(c.round(3).to_string())

    # EDA distributions
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle('Feature Distributions — DevSecOps Simulation Dataset\n(18,225 runs)',
                 fontsize=13, fontweight='bold')
    axes = axes.flatten()
    for i, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        axes[i].hist(df[feat], bins=20, color='#4A90D9',
                     edgecolor='white', linewidth=0.5, alpha=0.85)
        axes[i].set_title(label, fontweight='bold', fontsize=10)
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Count')
    axes[7].hist(df[PRIMARY_TARGET], bins=30, color='#E8632A',
                 edgecolor='white', linewidth=0.5, alpha=0.85)
    axes[7].set_title('Attack Success Rate (primary target)',
                       fontweight='bold', fontsize=10)
    axes[7].set_xlabel('Value')
    plt.tight_layout()
    savefig('eda_distributions.png')

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(11, 9))
    cols_hm = FEATURES + [PRIMARY_TARGET, 'system-resilience-score', 'ARIS']
    labels_hm = FEATURE_LABELS + ['Attack Success', 'Resilience', 'ARIS']
    cm = df[cols_hm].corr()
    mask = np.triu(np.ones_like(cm, dtype=bool))
    sns.heatmap(cm, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-1, vmax=1, linewidths=0.5,
                xticklabels=labels_hm, yticklabels=labels_hm, ax=ax)
    ax.set_title('Feature Correlation Matrix (including ARIS)',
                  fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
    plt.tight_layout()
    savefig('eda_correlation_heatmap.png')

    # Adaptation effect
    alpha_grp = df.groupby('alpha').agg(
        top_pref=('top-preference', 'mean'),
        compromises=('total-compromises-ever', 'mean'),
        aris=('ARIS', 'mean')
    ).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('RQ1 Evidence — Adaptation Effect (Alpha vs Outcomes)',
                  fontsize=13, fontweight='bold')
    for ax, col, label, color in zip(
        axes,
        ['top_pref', 'compromises', 'aris'],
        ['Avg Top Preference Score', 'Avg Total Compromises', 'Avg ARIS Score'],
        ['#E8632A', '#C0392B', '#7C3AED']
    ):
        ax.plot(alpha_grp['alpha'], alpha_grp[col],
                'o-', color=color, linewidth=2.5, markersize=10)
        ax.set_xlabel('Alpha (attacker learning rate)', fontweight='bold')
        ax.set_ylabel(label, fontweight='bold')
        ax.grid(True, alpha=0.3)
        for _, row in alpha_grp.iterrows():
            ax.annotate(f"{row[col]:.3f}", (row['alpha'], row[col]),
                        textcoords='offset points', xytext=(0, 10),
                        ha='center', fontsize=10, fontweight='bold')
    axes[0].set_title('Target Fixation (Adaptation Working)', fontweight='bold')
    axes[1].set_title('Compromise Count (Stable)', fontweight='bold')
    axes[2].set_title('ARIS Score (Novel Index)', fontweight='bold')
    plt.tight_layout()
    savefig('adaptation_effect.png')

    # Scenario heatmap
    pivot = df.groupby(['attacker-skill', 'defender-detection-rate'])\
               [PRIMARY_TARGET].mean().unstack()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='RdYlGn_r',
                linewidths=0.5, ax=ax, vmin=0, vmax=0.65)
    ax.set_title('RQ3 — Attack Success Rate: Attacker Skill × Detection Rate\n'
                  '(Mean across all 18,225 runs)', fontsize=12,
                  fontweight='bold', pad=15)
    ax.set_xlabel('Defender Detection Rate', fontweight='bold')
    ax.set_ylabel('Attacker Skill', fontweight='bold')
    plt.tight_layout()
    savefig('scenario_heatmap.png')


# ============================================================
# STEP 4 — TRAIN 8 MODELS WITH HYPERPARAMETER TUNING
# ============================================================

def define_models():
    """Define all 8 models. Tuned models use GridSearchCV."""
    return {
        'Linear Regression': LinearRegression(),
        'Decision Tree':     DecisionTreeRegressor(max_depth=8, random_state=RANDOM_SEED),
        'Extra Trees':       ExtraTreesRegressor(n_estimators=200, random_state=RANDOM_SEED,
                                                  n_jobs=-1),
        'Random Forest':     RandomForestRegressor(n_estimators=200, min_samples_split=5,
                                                    random_state=RANDOM_SEED, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                        max_depth=5, random_state=RANDOM_SEED),
        'XGBoost':           xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                                max_depth=6, subsample=0.8,
                                                colsample_bytree=0.8,
                                                random_state=RANDOM_SEED, verbosity=0),
        'LightGBM':          lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                                                 num_leaves=63, random_state=RANDOM_SEED,
                                                 verbosity=-1, n_jobs=-1),
        'CatBoost':          CatBoostRegressor(iterations=300, learning_rate=0.05,
                                                depth=6, random_seed=RANDOM_SEED,
                                                verbose=0),
    }


def hyperparameter_tuning(X_train, y_train):
    """
    GridSearchCV tuning for the 4 tree-ensemble models.
    Returns best params found.
    """
    section("STEP 4a — Hyperparameter Tuning (GridSearchCV)")
    kf_tune = KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)

    grids = {
        'Random Forest': (
            RandomForestRegressor(random_state=RANDOM_SEED, n_jobs=-1),
            {'n_estimators': [100, 200], 'max_depth': [None, 10], 'min_samples_split': [2, 5]}
        ),
        'XGBoost': (
            xgb.XGBRegressor(random_state=RANDOM_SEED, verbosity=0),
            {'n_estimators': [200, 300], 'learning_rate': [0.05, 0.1], 'max_depth': [5, 6]}
        ),
        'LightGBM': (
            lgb.LGBMRegressor(random_state=RANDOM_SEED, verbosity=-1, n_jobs=-1),
            {'n_estimators': [200, 300], 'learning_rate': [0.05, 0.1], 'num_leaves': [31, 63]}
        ),
        'CatBoost': (
            CatBoostRegressor(random_seed=RANDOM_SEED, verbose=0),
            {'iterations': [200, 300], 'learning_rate': [0.05, 0.1], 'depth': [5, 6]}
        ),
    }

    best_params = {}
    for name, (model, param_grid) in grids.items():
        print(f"  Tuning {name}...", end=' ', flush=True)
        t0 = time.time()
        gs = GridSearchCV(model, param_grid, cv=kf_tune,
                          scoring='r2', n_jobs=-1, refit=True)
        gs.fit(X_train, y_train)
        elapsed = time.time() - t0
        best_params[name] = gs.best_params_
        print(f"best R²={gs.best_score_:.4f}  params={gs.best_params_}  ({elapsed:.1f}s)")

    return best_params


def apply_best_params(models, best_params):
    """Update model definitions with tuned hyperparameters."""
    for name, params in best_params.items():
        if name in models:
            models[name].set_params(**params)
            print(f"  Applied tuned params to {name}: {params}")
    return models


def train_all_models(df):
    section("STEP 4b — Training 8 ML Models")

    X = df[FEATURES].values
    y = df[PRIMARY_TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")

    # Hyperparameter tuning
    best_params = hyperparameter_tuning(X_train, y_train)

    # Define and update models
    models = define_models()
    models = apply_best_params(models, best_params)

    # Train, evaluate, CV
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    results = {}
    trained  = {}
    cv_score_lists = {}  # for Wilcoxon test

    print(f"\n  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8} "
          f"{'CV R²':>10} {'CV±':>8}  Time")
    print("  " + "-" * 78)

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2   = round(r2_score(y_test, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        mae  = round(mean_absolute_error(y_test, y_pred), 4)

        cv_mean, cv_std, cv_list = cv_score(model, X, y, kf)
        cv_score_lists[name] = cv_list

        elapsed = time.time() - t0
        results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae,
                         'CV_R2': round(cv_mean, 4), 'CV_Std': round(cv_std, 4)}
        trained[name] = model
        print(f"  {name:<22} {r2:>8} {rmse:>8} {mae:>8} "
              f"{cv_mean:>10.4f} {cv_std:>8.4f}  {elapsed:.1f}s")

    return results, trained, cv_score_lists, X_train, X_test, y_train, y_test, X, y


# ============================================================
# STEP 5 — STATISTICAL SIGNIFICANCE TESTS
# ============================================================

def statistical_tests(cv_score_lists, results):
    """
    Statistical significance across all 8 models.

    Step 1: Friedman test — the correct omnibus test for "is there any
    difference among N related models across the same CV folds", run BEFORE
    any pairwise comparison. Running pairwise tests without this first step
    (as a plain repeated-Wilcoxon approach does) inflates the false-positive
    rate through the multiple-comparisons problem.

    Step 2: if — and only if — the Friedman test is significant, run a
    post-hoc pairwise comparison that corrects for multiple comparisons:
      - Nemenyi test (via scikit-posthocs) if installed — the standard choice
        after a significant Friedman test.
      - Otherwise, a Bonferroni-corrected pairwise Wilcoxon fallback, which is
        clearly weaker/more conservative but at least doesn't overstate
        significance the way N uncorrected Wilcoxon tests would.
    """
    section("STEP 5 — Statistical Significance (Friedman + Nemenyi Post-Hoc)")

    names = list(cv_score_lists.keys())
    n_models = len(names)
    # scikit-posthocs expects rows=blocks (folds), columns=groups (models)
    score_df = pd.DataFrame({n: cv_score_lists[n] for n in names})

    best_name = max(results, key=lambda n: results[n]['CV_R2'])
    print(f"  Best model: {best_name} (CV R²={results[best_name]['CV_R2']:.4f})")

    stat, p_friedman = stats.friedmanchisquare(*[cv_score_lists[n] for n in names])
    sig_word = 'significant' if p_friedman < 0.05 else 'not significant'
    print(f"\n  Friedman test across {n_models} models ({N_CV_FOLDS}-fold CV):")
    print(f"    chi²={stat:.4f}  p={p_friedman:.6f}  ({sig_word} at α=0.05)")

    pval_matrix = pd.DataFrame(np.ones((n_models, n_models)), index=names, columns=names)
    sig_results = {}

    if p_friedman >= 0.05:
        method_used = 'none (Friedman n.s.)'
        print("\n  Friedman test not significant — the 8 models are not reliably")
        print("  distinguishable on this data. Skipping pairwise post-hoc tests to")
        print("  avoid reporting false-positive 'significant' differences.")
        for name in names:
            if name != best_name:
                sig_results[name] = {'p': p_friedman, 'significant': False}
    else:
        if HAS_POSTHOCS:
            method_used = 'Nemenyi'
            alpha_used = 0.05
            pval_matrix = sp.posthoc_nemenyi_friedman(score_df.values)
            pval_matrix.index = names
            pval_matrix.columns = names
            print("\n  Nemenyi post-hoc test (all-pairs, multiple-comparison safe):")
        else:
            method_used = 'Wilcoxon + Bonferroni (fallback)'
            pairs = list(combinations(names, 2))
            alpha_used = 0.05 / len(pairs)
            print("\n  scikit-posthocs not installed — falling back to pairwise Wilcoxon")
            print(f"  with Bonferroni correction (install scikit-posthocs for the")
            print(f"  standard Nemenyi post-hoc test). Corrected α = {alpha_used:.5f} "
                  f"across {len(pairs)} pairs.")
            for a, b in pairs:
                _, p_w = stats.wilcoxon(cv_score_lists[a], cv_score_lists[b])
                pval_matrix.loc[a, b] = p_w
                pval_matrix.loc[b, a] = p_w

        print(f"\n  {best_name} vs all other models:")
        print(f"  {'Model':<22} {'p-value':>12} {'Significant?':>14}")
        print("  " + "-" * 50)
        for name in names:
            if name == best_name:
                continue
            p = float(pval_matrix.loc[best_name, name])
            sig = p < alpha_used
            sig_results[name] = {'p': p, 'significant': sig}
            print(f"  {name:<22} {p:>12.4f} {'Yes' if sig else 'No':>14}")

    # --- Plot: pairwise significance heatmap ---
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(pval_matrix.astype(float), annot=True, fmt='.3f', cmap='RdYlGn',
                vmin=0, vmax=0.1, linewidths=0.5, ax=ax,
                cbar_kws={'label': 'p-value (green = more significant)'})
    method_label = 'none — Friedman n.s.' if p_friedman >= 0.05 else (
        'Nemenyi' if HAS_POSTHOCS else 'Wilcoxon+Bonferroni')
    ax.set_title(f'Post-Hoc Pairwise Significance — {method_label}\n'
                 f'Friedman χ²={stat:.2f}, p={p_friedman:.4g} '
                 f'({sig_word} @ α=0.05)',
                 fontsize=12, fontweight='bold', pad=15)
    plt.xticks(rotation=35, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    savefig('friedman_nemenyi.png')

    return best_name, sig_results


# ============================================================
# STEP 5b — ABLATION STUDY
# ============================================================

def run_ablation_study(df, best_model_name, trained_models):
    """
    Per-feature ablation: drop one feature at a time, retrain the best model
    (using its already-tuned hyperparameters) on the remaining 6 features, and
    measure the drop in test R². This is an independent, model-training-based
    check on what SHAP already suggests from attribution alone — if Attacker
    Skill dominates SHAP, it should also cause the largest ΔR² when removed.
    """
    section("STEP 5b — Ablation Study (Per-Feature Contribution)")

    X_full = df[FEATURES].values
    y = df[PRIMARY_TARGET].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    base_model = clone(trained_models[best_model_name])
    base_model.fit(X_train, y_train)
    baseline_r2 = r2_score(y_test, base_model.predict(X_test))
    print(f"  Baseline ({best_model_name}, all {len(FEATURES)} features): R²={baseline_r2:.4f}")

    deltas = {}
    print(f"\n  {'Feature removed':<28} {'R² w/o feature':>16} {'ΔR²':>10}")
    print("  " + "-" * 58)
    for i, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        cols = [j for j in range(len(FEATURES)) if j != i]
        m = clone(trained_models[best_model_name])
        m.fit(X_train[:, cols], y_train)
        r2_wo = r2_score(y_test, m.predict(X_test[:, cols]))
        delta = baseline_r2 - r2_wo
        deltas[label] = delta
        print(f"  {label:<28} {r2_wo:>16.4f} {delta:>+10.4f}")

    order = sorted(deltas, key=deltas.get)
    fig, ax = plt.subplots(figsize=(9, 6))
    vals = [deltas[l] for l in order]
    colors = ['#EF4444' if v > 0 else '#94A3B8' for v in vals]
    bars = ax.barh(order, vals, color=colors, edgecolor='white', linewidth=0.8)
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_xlabel('ΔR² when feature is removed  (higher = more important)', fontweight='bold')
    ax.set_title(f'Ablation Study — {best_model_name} on {PRIMARY_TARGET}\n'
                 f'Baseline R² (all 7 features) = {baseline_r2:.4f}',
                 fontsize=12, fontweight='bold')
    for bar, v in zip(bars, vals):
        ax.text(v + (0.002 if v >= 0 else -0.002), bar.get_y() + bar.get_height()/2,
                f'{v:+.4f}', va='center', ha='left' if v >= 0 else 'right', fontsize=9)
    plt.tight_layout()
    savefig('ablation_study.png')

    return deltas, baseline_r2


# ============================================================
# STEP 6 — MODEL COMPARISON PLOTS
# ============================================================

def plot_model_comparison(results, cv_score_lists):
    section("STEP 6 — Model Comparison Plots")
    names   = list(results.keys())
    colors  = [MODEL_COLORS.get(n, '#888') for n in names]

    r2s   = [results[n]['R2']   for n in names]
    rmses = [results[n]['RMSE'] for n in names]
    maes  = [results[n]['MAE']  for n in names]

    # 8-model comparison: R², RMSE, MAE
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('8-Model Comparison — Attack Success Rate Prediction\n'
                 'DevSecOps Simulation Dataset (18,225 runs, 80/20 split)',
                 fontsize=13, fontweight='bold')

    for ax, vals, title, ylim, better in zip(
        axes,
        [r2s, rmses, maes],
        ['R² Score', 'RMSE', 'MAE'],
        [1.0, 0.12, 0.08],
        ['higher', 'lower', 'lower']
    ):
        bars = ax.bar(names, vals, color=colors, width=0.6,
                      edgecolor='white', linewidth=1.5)
        ax.set_title(f'{title} ({better} is better)', fontweight='bold')
        ax.set_ylim(0, ylim)
        ax.tick_params(axis='x', rotation=35)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    v + ylim*0.015, f'{v:.4f}',
                    ha='center', fontsize=8, fontweight='bold')
        # Highlight best
        best_idx = (vals.index(max(vals)) if better == 'higher'
                    else vals.index(min(vals)))
        bars[best_idx].set_edgecolor('#2E4057')
        bars[best_idx].set_linewidth(3)

    plt.tight_layout()
    savefig('model_comparison_8models.png')

    # CV comparison with error bars
    cv_means = [results[n]['CV_R2']  for n in names]
    cv_stds  = [results[n]['CV_Std'] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(names, cv_means, color=colors, width=0.6,
                  edgecolor='white', linewidth=1.5,
                  yerr=cv_stds, capsize=5,
                  error_kw={'elinewidth': 1.8, 'ecolor': '#334155'})
    ax.set_title(f'{N_CV_FOLDS}-Fold Cross-Validation R² — All 8 Models\n'
                  '(Error bars = ± standard deviation across folds)',
                  fontsize=13, fontweight='bold')
    ax.set_ylabel('CV R² Score', fontweight='bold')
    ax.set_ylim(0.5, 0.85)
    ax.tick_params(axis='x', rotation=35)
    ax.grid(True, alpha=0.2, axis='y')
    for bar, v, s in zip(bars, cv_means, cv_stds):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + s + 0.008, f'{v:.4f}±{s:.4f}',
                ha='center', fontsize=8, fontweight='bold')
    plt.tight_layout()
    savefig('cv_comparison_8models.png')

    # Full metrics table
    metrics_df = pd.DataFrame([
        {'Model': n,
         'R²':      results[n]['R2'],
         'RMSE':    results[n]['RMSE'],
         'MAE':     results[n]['MAE'],
         'CV R²':   results[n]['CV_R2'],
         'CV Std':  results[n]['CV_Std']}
        for n in names
    ])
    path = os.path.join(OUTPUT_DIR, 'publication_metrics_table_v2.csv')
    metrics_df.to_csv(path, index=False)
    print(f"\n  Publication metrics table:")
    print(metrics_df.to_string(index=False))


# ============================================================
# STEP 7 — ARIS MODEL TRAINING
# ============================================================

def train_aris_models(df, best_model_name, trained_models):
    section("STEP 7 — ARIS Model Training (Novel Contribution)")

    X = df[FEATURES].values
    y = df['ARIS'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)

    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)

    # Train all 8 on ARIS
    aris_results = {}
    aris_trained = {}
    aris_models = define_models()

    print(f"  {'Model':<22} {'R²':>8} {'RMSE':>8} {'MAE':>8} {'CV R²':>10}")
    print("  " + "-" * 60)

    for name, model in aris_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2   = round(r2_score(y_test, y_pred), 4)
        rmse = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
        mae  = round(mean_absolute_error(y_test, y_pred), 4)
        cv_m, cv_s, _ = cv_score(model, X, y, kf)
        aris_results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae,
                               'CV_R2': round(cv_m, 4), 'CV_Std': round(cv_s, 4)}
        aris_trained[name] = model
        print(f"  {name:<22} {r2:>8} {rmse:>8} {mae:>8} {cv_m:>10.4f}")

    # ARIS comparison plot
    names   = list(aris_results.keys())
    colors  = [MODEL_COLORS.get(n, '#888') for n in names]
    r2s     = [aris_results[n]['R2'] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('ARIS Prediction — 8 Model Comparison\n'
                 '(Adaptive Risk Intelligence Score as prediction target)',
                 fontsize=13, fontweight='bold')

    bars = axes[0].bar(names, r2s, color=colors, width=0.6,
                       edgecolor='white', linewidth=1.5)
    axes[0].set_title('R² Score on ARIS Target', fontweight='bold')
    axes[0].set_ylim(0, 1.0)
    axes[0].tick_params(axis='x', rotation=35)
    for bar, v in zip(bars, r2s):
        axes[0].text(bar.get_x() + bar.get_width()/2,
                     v + 0.01, f'{v:.4f}', ha='center',
                     fontsize=9, fontweight='bold')

    # ARIS vs attack-success-rate performance comparison
    attack_r2s = [aris_results[n]['R2'] for n in names]  # now actually ARIS
    axes[1].scatter(range(len(names)), attack_r2s, color=colors,
                    s=120, zorder=5, edgecolors='white', linewidths=1.5)
    axes[1].set_xticks(range(len(names)))
    axes[1].set_xticklabels(names, rotation=35, ha='right')
    axes[1].set_title('ARIS R² Scores per Model', fontweight='bold')
    axes[1].set_ylabel('R² Score')
    axes[1].set_ylim(0.5, 1.0)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    savefig('aris_model_comparison.png')

    # Save best ARIS model (XGBoost)
    best_aris = aris_trained['XGBoost']
    path = os.path.join(OUTPUT_DIR, 'aris_xgb_model.pkl')
    with open(path, 'wb') as f:
        pickle.dump(best_aris, f)
    print(f"\n  Saved: aris_xgb_model.pkl")

    # SHAP on ARIS XGBoost
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), SHAP_SAMPLE, replace=False)
    X_s = pd.DataFrame(X[idx], columns=FEATURES)
    exp  = shap.TreeExplainer(best_aris)
    sv   = exp.shap_values(X_s)

    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_s, feature_names=FEATURE_LABELS, show=False)
    plt.title('SHAP — Feature Impact on ARIS Score\n'
              '(Adaptive Risk Intelligence Score)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig('aris_shap_summary.png')

    return aris_results, aris_trained


# ============================================================
# STEP 8 — SHAP ON PRIMARY TARGET (XGBoost)
# ============================================================

def run_shap(trained_models, X, df):
    section("STEP 8 — SHAP Explainability (XGBoost on attack-success-rate)")

    xgb_model = trained_models['XGBoost']
    idx = np.random.RandomState(RANDOM_SEED).choice(len(X), SHAP_SAMPLE, replace=False)
    X_s = pd.DataFrame(X[idx], columns=FEATURES)

    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_s)
    base_val    = float(np.atleast_1d(explainer.expected_value)[0])

    print(f"  SHAP base value: {base_val:.4f}  (dataset mean: {df[PRIMARY_TARGET].mean():.4f})")

    # Beeswarm
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_s, feature_names=FEATURE_LABELS, show=False)
    plt.title('SHAP — Feature Impact on Attack Success Rate\n'
              '(Red=high feature value, Blue=low)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig('shap_summary_beeswarm.png')

    # Bar
    plt.figure(figsize=(9, 5))
    shap.summary_plot(shap_values, X_s, feature_names=FEATURE_LABELS,
                      plot_type='bar', show=False)
    plt.title('SHAP — Mean Absolute Feature Importance',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig('shap_importance_bar.png')

    # Dependence
    plt.figure(figsize=(9, 5))
    shap.dependence_plot(0, shap_values, X_s.values,
                         feature_names=FEATURE_LABELS,
                         interaction_index=3, show=False, alpha=0.4)
    plt.title('SHAP Dependence — Attacker Skill (colour=Detection Rate)',
              fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    savefig('shap_dependence_skill.png')

    # Waterfall — 3 scenarios
    scenarios = {
        'High Risk\n(Skill=0.9, Det=0.2)': [0.9, 0.8, 0.3, 0.2, 0.2, 0.2, 0.1],
        'Balanced\n(Skill=0.5, Det=0.5)':  [0.5, 0.7, 0.4, 0.5, 0.5, 0.2, 0.1],
        'Low Risk\n(Skill=0.1, Det=0.8)':  [0.1, 0.4, 0.1, 0.8, 0.8, 0.2, 0.1],
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle('SHAP Waterfall — Per-Scenario Risk Explanation\n'
                 '(How each feature drives the prediction from baseline)',
                 fontsize=13, fontweight='bold')
    for ax, (title, inputs) in zip(axes, scenarios.items()):
        x1 = np.array(inputs).reshape(1, -1)
        sv1 = explainer.shap_values(pd.DataFrame(x1, columns=FEATURES))[0]
        pred = float(xgb_model.predict(x1)[0])
        contribs = sorted(zip(FEATURE_LABELS, sv1, inputs),
                          key=lambda x: abs(x[1]), reverse=True)
        labels = [f'{l}\n(v={v:.2f})' for l, _, v in contribs]
        values = [s for _, s, _ in contribs]
        colors = ['#E8632A' if v > 0 else '#4A90D9' for v in values]
        bars = ax.barh(labels, values, color=colors,
                       edgecolor='white', linewidth=0.8)
        ax.axvline(0, color='black', linewidth=0.8)
        for bar, v in zip(bars, values):
            ax.text(v + (0.001 if v >= 0 else -0.001),
                    bar.get_y() + bar.get_height()/2,
                    f'{v:+.4f}', va='center',
                    ha='left' if v >= 0 else 'right', fontsize=8.5)
        ax.set_title(f'{title}\nPredicted: {pred:.3f} | Base: {base_val:.3f}',
                     fontweight='bold', fontsize=10)
        ax.set_xlabel('SHAP contribution', fontsize=9)
    plt.tight_layout()
    savefig('shap_waterfall_scenarios.png')

    mean_shap = np.abs(shap_values).mean(axis=0)
    print("\n  SHAP mean |value| ranking:")
    for label, val in sorted(zip(FEATURE_LABELS, mean_shap), key=lambda x: -x[1]):
        bar = '█' * int(val * 200)
        print(f"    {label:<25} {val:.4f}  {bar}")


# ============================================================
# STEP 9 — RESIDUALS + SENSITIVITY
# ============================================================

def plot_residuals_and_sensitivity(trained_models, X_test, y_test, df):
    section("STEP 9 — Residual Analysis & Sensitivity")

    # Residuals for top 3 models
    top3 = ['XGBoost', 'LightGBM', 'Random Forest']
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Predicted vs Actual — Top 3 Models\n(Attack Success Rate)',
                  fontsize=13, fontweight='bold')
    for ax, name in zip(axes, top3):
        model = trained_models[name]
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        residuals = y_test - y_pred
        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.scatter(y_pred, y_test, alpha=0.12, s=6,
                   color=MODEL_COLORS.get(name, '#888'))
        ax.plot(lims, lims, 'k--', linewidth=1.5)
        ax.set_xlabel('Predicted', fontweight='bold')
        ax.set_ylabel('Actual', fontweight='bold')
        ax.set_title(f'{name}\nR²={r2:.4f}  Mean residual={residuals.mean():.4f}',
                     fontweight='bold')
        ax.set_xlim(lims); ax.set_ylim(lims)
    plt.tight_layout()
    savefig('residual_plot_best_models.png')

    # Sensitivity — XGBoost
    xgb_model = trained_models['XGBoost']
    medians = [np.median(df[f].unique()) for f in FEATURES]
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle('Sensitivity Analysis — XGBoost Model\n'
                  'Effect of each parameter (others held at median)',
                  fontsize=13, fontweight='bold')
    axes = axes.flatten()
    for i, (feat, label) in enumerate(zip(FEATURES, FEATURE_LABELS)):
        sweep = np.sort(df[feat].unique())
        preds = []
        for v in sweep:
            row = medians.copy(); row[i] = v
            preds.append(float(xgb_model.predict(np.array([row]))[0]))
        axes[i].plot(sweep, preds, 'o-', color='#E8632A',
                     linewidth=2.2, markersize=8)
        axes[i].set_title(label, fontweight='bold', fontsize=11)
        axes[i].set_xlabel('Value'); axes[i].set_ylabel('Predicted attack success')
        axes[i].grid(True, alpha=0.25)
        for x, y_ in zip(sweep, preds):
            axes[i].annotate(f'{y_:.3f}', (x, y_),
                              textcoords='offset points', xytext=(0, 7),
                              ha='center', fontsize=8)
    axes[7].axis('off')
    plt.tight_layout()
    savefig('sensitivity_analysis.png')


# ============================================================
# STEP 10 — SAVE MODELS + SUMMARY
# ============================================================

def save_models_and_summary(trained_models, results, aris_results):
    section("STEP 10 — Saving Models & Summary")

    # Save key models
    for name, fname in [('Random Forest', 'rf_model.pkl'),
                         ('XGBoost',       'xgb_model.pkl'),
                         ('LightGBM',      'lgbm_model.pkl')]:
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, 'wb') as f:
            pickle.dump(trained_models[name], f)
        print(f"  Saved: {fname}")

    # Full results summary
    rows = []
    for name in results:
        row = {'Model': name, 'Target': 'attack-success-rate',
               **results[name]}
        rows.append(row)
    for name in aris_results:
        row = {'Model': name, 'Target': 'ARIS',
               'R2': aris_results[name]['R2'],
               'RMSE': aris_results[name]['RMSE'],
               'MAE': aris_results[name]['MAE'],
               'CV_R2': aris_results[name]['CV_R2'],
               'CV_Std': aris_results[name]['CV_Std']}
        rows.append(row)
    summary_df = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, 'full_results_summary.csv')
    summary_df.to_csv(path, index=False)
    print(f"  Saved: full_results_summary.csv")

    print("\n  === FINAL SUMMARY ===")
    print(f"  {'Model':<22} {'R² (ASR)':>10} {'CV R²':>10} {'R² (ARIS)':>12}")
    print("  " + "-" * 58)
    for name in results:
        ar = aris_results.get(name, {}).get('R2', '-')
        print(f"  {name:<22} {results[name]['R2']:>10} "
              f"{results[name]['CV_R2']:>10} {ar:>12}")


# ============================================================
# MAIN
# ============================================================

def main():
    global CSV_PATH, OUTPUT_DIR, CSV_SKIPROWS
    args = parse_args()
    CSV_PATH, OUTPUT_DIR, CSV_SKIPROWS = args.csv, args.output_dir, args.skiprows

    print("\n" + "=" * 64)
    print("  DevSecOps ML Pipeline V2.0 — Publication Grade")
    print("=" * 64 + "\n")

    if not os.path.exists(CSV_PATH):
        print(f"ERROR: CSV not found: '{CSV_PATH}'")
        print("Update CSV_PATH at the top of the script.")
        sys.exit(1)

    setup()

    # Steps
    df = load_data()
    w1, w2, w3, rho = optimize_aris_weights(df)
    df = compute_aris(df, w1, w2, w3)
    run_eda(df)

    results, trained, cv_lists, X_train, X_test, y_train, y_test, X, y = \
        train_all_models(df)

    best_name, sig_results = statistical_tests(cv_lists, results)
    ablation_deltas, ablation_baseline_r2 = run_ablation_study(df, best_name, trained)
    plot_model_comparison(results, cv_lists)

    aris_results, aris_trained = train_aris_models(df, best_name, trained)
    run_shap(trained, X, df)
    plot_residuals_and_sensitivity(trained, X_test, y_test, df)
    save_models_and_summary(trained, results, aris_results)

    print("\n" + "=" * 64)
    print("  Pipeline V2.0 complete — ALL outputs generated.")
    print(f"  Output folder: {os.path.abspath(OUTPUT_DIR)}/")
    print("=" * 64)
    print("""
Outputs generated (./outputs/):
  eda_distributions.png            — Feature histograms
  eda_correlation_heatmap.png      — Correlation matrix incl. ARIS
  adaptation_effect.png            — RQ1 evidence (alpha effect)
  scenario_heatmap.png             — RQ3 skill x detection matrix
  aris_weight_optimization.png     — ARIS-OPT convergence + weight bars
  aris_weight_sensitivity.png      — ARIS-OPT weight robustness sweep
  aris_distribution.png            — ARIS score distribution
  model_comparison_8models.png     — 8-model R²/RMSE/MAE chart
  cv_comparison_8models.png        — CV with error bars (8 models)
  publication_metrics_table_v2.csv — Full metrics table
  friedman_nemenyi.png             — Friedman + Nemenyi/Wilcoxon post-hoc
  ablation_study.png               — Per-feature ΔR² (real, not hardcoded)
  aris_model_comparison.png        — 8 models on ARIS target
  aris_shap_summary.png            — SHAP on ARIS
  aris_xgb_model.pkl               — ARIS prediction model
  shap_summary_beeswarm.png        — SHAP beeswarm (XAI)
  shap_importance_bar.png          — SHAP bar chart (XAI)
  shap_dependence_skill.png        — SHAP dependence plot (XAI)
  shap_waterfall_scenarios.png     — Per-scenario risk explanation
  residual_plot_best_models.png    — Residuals for top 3 models
  sensitivity_analysis.png         — Per-feature sensitivity sweep
  rf_model.pkl / xgb_model.pkl / lgbm_model.pkl
  full_results_summary.csv         — All results in one table
    """)


if __name__ == '__main__':
    main()