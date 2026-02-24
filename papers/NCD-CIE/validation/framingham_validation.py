#!/usr/bin/env python3
"""NCD-CIE Validation on Framingham Heart Study dataset.

Maps patient risk factors to NCD-CIE knowledge graph edge weights,
computes composite risk scores, and evaluates against actual 10-year CHD outcomes.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_PATH = '/home/clawdbot/clawd/datasets/framingham/framingham.csv'

# NCD-CIE Knowledge Graph Edge Weights (from the paper)
KG_EDGES = {
    'smoking': 0.45,
    'hypertension': 0.52,
    'high_cholesterol': 0.38,
    'diabetes': 0.42,
    'obesity': 0.35,
    'age_risk': 0.30,       # age contribution
    'male_sex': 0.15,       # sex-based risk differential
    'glucose_risk': 0.20,   # elevated glucose
}

# Intervention effects (ATE from paper)
INTERVENTIONS = {
    'statin': -0.06,        # -6% absolute risk reduction
    'bp_medication': -0.045, # -4.5% risk reduction
    'smoking_cessation': -0.04,
}

GAMMA = 0.7  # blending factor from paper


def load_data():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} patients, {df['TenYearCHD'].sum()} CHD positive ({df['TenYearCHD'].mean():.1%})")
    return df


def map_risk_factors(df):
    """Map Framingham variables to NCD-CIE KG binary risk nodes."""
    rf = pd.DataFrame(index=df.index)
    rf['smoking'] = df['currentSmoker'].fillna(0).astype(float)
    rf['hypertension'] = ((df['sysBP'] >= 140) | (df['diaBP'] >= 90) | (df['prevalentHyp'] == 1)).astype(float)
    rf['high_cholesterol'] = (df['totChol'].fillna(df['totChol'].median()) >= 240).astype(float)
    rf['diabetes'] = df['diabetes'].fillna(0).astype(float)
    rf['obesity'] = (df['BMI'].fillna(df['BMI'].median()) >= 30).astype(float)
    rf['age_risk'] = ((df['age'] - 32) / (70 - 32)).clip(0, 1)  # normalized age
    rf['male_sex'] = df['male'].astype(float)
    rf['glucose_risk'] = (df['glucose'].fillna(df['glucose'].median()) >= 126).astype(float)
    return rf


def compute_ncd_cie_score(risk_factors):
    """Compute NCD-CIE composite risk score: sum(edge_weight * risk_factor)."""
    score = np.zeros(len(risk_factors))
    for factor, weight in KG_EDGES.items():
        score += weight * risk_factors[factor].values
    return score


def compute_blended_score(kg_score, lr_prob, gamma=GAMMA):
    """Blend KG score with statistical model: gamma * KG + (1-gamma) * stat."""
    # Normalize KG score to [0,1] range
    kg_norm = (kg_score - kg_score.min()) / (kg_score.max() - kg_score.min() + 1e-8)
    return gamma * kg_norm + (1 - gamma) * lr_prob


def evaluate(y_true, y_score, label):
    """Compute evaluation metrics."""
    auc = roc_auc_score(y_true, y_score)
    brier = brier_score_loss(y_true, np.clip(y_score, 0, 1))
    print(f"  {label}: AUC={auc:.4f}, Brier={brier:.4f}")
    return {'label': label, 'auc': auc, 'brier': brier}


def plot_roc(y_true, scores_dict, outpath):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(8, 6))
    for label, y_score in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        plt.plot(fpr, tpr, label=f'{label} (AUC={auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('NCD-CIE Validation: ROC Curves on Framingham Data')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  ROC plot saved: {outpath}")


def plot_calibration(y_true, scores_dict, outpath):
    """Plot calibration curves."""
    plt.figure(figsize=(8, 6))
    for label, y_score in scores_dict.items():
        y_score_clipped = np.clip(y_score, 0, 1)
        prob_true, prob_pred = calibration_curve(y_true, y_score_clipped, n_bins=10, strategy='uniform')
        plt.plot(prob_pred, prob_true, 's-', label=label)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Observed Frequency')
    plt.title('Calibration Curves on Framingham Data')
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Calibration plot saved: {outpath}")


def simulate_interventions(df, risk_factors):
    """Simulate what-if interventions on high-risk patients."""
    kg_score = compute_ncd_cie_score(risk_factors)
    kg_norm = (kg_score - kg_score.min()) / (kg_score.max() - kg_score.min() + 1e-8)

    # Identify high-risk patients (top quartile)
    high_risk_mask = kg_norm >= np.percentile(kg_norm, 75)
    high_risk_df = df[high_risk_mask].copy()
    high_risk_scores = kg_norm[high_risk_mask]

    results = {}
    for intervention, ate in INTERVENTIONS.items():
        # Who would benefit
        if intervention == 'statin':
            eligible = high_risk_df['totChol'].fillna(0) >= 200
        elif intervention == 'bp_medication':
            eligible = (high_risk_df['sysBP'] >= 140) | (high_risk_df['diaBP'] >= 90)
        elif intervention == 'smoking_cessation':
            eligible = high_risk_df['currentSmoker'] == 1
        else:
            eligible = pd.Series(True, index=high_risk_df.index)

        n_eligible = eligible.sum()
        if n_eligible > 0:
            original_risk = high_risk_scores[eligible].mean()
            adjusted_risk = max(0, original_risk + ate)
            actual_chd = high_risk_df.loc[eligible, 'TenYearCHD'].mean()
            results[intervention] = {
                'n_eligible': int(n_eligible),
                'original_risk': float(original_risk),
                'predicted_post_intervention': float(adjusted_risk),
                'predicted_reduction': float(ate),
                'actual_chd_rate': float(actual_chd),
            }
            print(f"  {intervention}: {n_eligible} eligible, risk {original_risk:.3f}→{adjusted_risk:.3f}, actual CHD={actual_chd:.3f}")

    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("NCD-CIE Validation: Framingham Heart Study")
    print("=" * 60)

    # Load and prepare
    df = load_data()
    risk_factors = map_risk_factors(df)
    y_true = df['TenYearCHD'].values

    # Drop rows with missing outcome (shouldn't be any)
    valid = ~np.isnan(y_true)
    y_true = y_true[valid].astype(int)
    risk_factors = risk_factors[valid]
    df = df[valid]

    # 1. KG-only score
    print("\n--- Model Evaluation ---")
    kg_score = compute_ncd_cie_score(risk_factors)
    kg_metrics = evaluate(y_true, kg_score, "NCD-CIE KG-only")

    # 2. Baseline logistic regression
    features = risk_factors.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_scaled, y_true)
    lr_prob = lr.predict_proba(X_scaled)[:, 1]
    lr_metrics = evaluate(y_true, lr_prob, "Logistic Regression")

    # 3. Blended (KG + statistical) — the NCD-CIE approach
    blended = compute_blended_score(kg_score, lr_prob, gamma=GAMMA)
    blended_metrics = evaluate(y_true, blended, f"NCD-CIE Blended (γ={GAMMA})")

    # 4. Try different gamma values
    print("\n--- Gamma Sensitivity ---")
    gamma_results = []
    for g in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0]:
        b = compute_blended_score(kg_score, lr_prob, gamma=g)
        auc = roc_auc_score(y_true, b)
        gamma_results.append({'gamma': g, 'auc': auc})
        print(f"  γ={g}: AUC={auc:.4f}")

    # 5. Reference AUCs from paper
    print("\n--- Reference Comparison ---")
    print(f"  SCORE2 (paper):         AUC=0.704")
    print(f"  D'Agostino (paper):     AUC=0.721")
    print(f"  NCD-CIE KG-only:        AUC={kg_metrics['auc']:.4f}")
    print(f"  NCD-CIE Blended:        AUC={blended_metrics['auc']:.4f}")
    print(f"  Logistic Regression:    AUC={lr_metrics['auc']:.4f}")

    # Plots
    print("\n--- Generating Plots ---")
    scores = {
        'NCD-CIE KG-only': kg_score,
        f'NCD-CIE Blended (γ={GAMMA})': blended,
        'Logistic Regression': lr_prob,
    }
    plot_roc(y_true, scores, os.path.join(RESULTS_DIR, 'framingham_roc.png'))
    plot_calibration(y_true, scores, os.path.join(RESULTS_DIR, 'framingham_calibration.png'))

    # 6. Intervention simulation
    print("\n--- What-If Intervention Simulation ---")
    intervention_results = simulate_interventions(df, risk_factors)

    # Save results
    all_results = {
        'dataset': 'Framingham Heart Study',
        'n_patients': len(df),
        'n_chd_positive': int(y_true.sum()),
        'prevalence': float(y_true.mean()),
        'metrics': {
            'kg_only': kg_metrics,
            'logistic_regression': lr_metrics,
            'blended': blended_metrics,
        },
        'gamma_sensitivity': gamma_results,
        'interventions': intervention_results,
        'reference': {
            'SCORE2_AUC': 0.704,
            'DAgostino_AUC': 0.721,
        }
    }
    with open(os.path.join(RESULTS_DIR, 'framingham_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/framingham_results.json")

    return all_results


if __name__ == '__main__':
    main()
