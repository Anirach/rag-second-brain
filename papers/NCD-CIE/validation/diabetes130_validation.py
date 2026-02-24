#!/usr/bin/env python3
"""NCD-CIE What-If Validation on Diabetes 130 US Hospitals dataset.

Identifies multi-visit patients with medication changes between visits,
compares NCD-CIE predicted intervention effects against actual readmission outcomes.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
DATA_PATH = '/home/clawdbot/clawd/datasets/diabetes-130/diabetic_data.csv'

# Key diabetes medications to track
KEY_MEDS = ['metformin', 'insulin', 'glipizide', 'glyburide', 'pioglitazone',
            'rosiglitazone', 'glimepiride', 'acarbose', 'repaglinide', 'nateglinide']

# NCD-CIE predicted intervention effects (ATE from knowledge graph)
KG_INTERVENTION_ATE = {
    'metformin': -0.08,       # strong evidence: reduces readmission
    'insulin': -0.05,         # moderate: better glucose control
    'glipizide': -0.04,
    'glyburide': -0.03,
    'pioglitazone': -0.035,
    'rosiglitazone': -0.03,
    'glimepiride': -0.035,
    'acarbose': -0.02,
    'repaglinide': -0.025,
    'nateglinide': -0.02,
}

# Additional KG risk factors
KG_RISK_WEIGHTS = {
    'HbA1c_elevated': 0.40,
    'num_medications_high': 0.15,
    'num_diagnoses_high': 0.25,
    'emergency_history': 0.30,
    'inpatient_history': 0.35,
    'long_stay': 0.10,
}


def load_data():
    df = pd.read_csv(DATA_PATH, na_values='?')
    print(f"Loaded {len(df)} encounters, {df['patient_nbr'].nunique()} unique patients")
    return df


def encode_readmission(df):
    """Encode readmission as binary (readmitted <30 days = 1)."""
    df = df.copy()
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    df['readmitted_any'] = (df['readmitted'] != 'NO').astype(int)
    return df


def get_multi_visit_patients(df, min_visits=2):
    """Get patients with multiple encounters, sorted by encounter."""
    visit_counts = df['patient_nbr'].value_counts()
    multi = visit_counts[visit_counts >= min_visits].index
    multi_df = df[df['patient_nbr'].isin(multi)].sort_values(['patient_nbr', 'encounter_id'])
    print(f"Multi-visit patients (>={min_visits}): {len(multi)} patients, {len(multi_df)} encounters")
    return multi_df


def detect_med_changes(multi_df):
    """Detect medication changes between consecutive visits for the same patient."""
    changes = []
    for pid, group in multi_df.groupby('patient_nbr'):
        if len(group) < 2:
            continue
        rows = group.sort_values('encounter_id').reset_index(drop=True)
        for i in range(len(rows) - 1):
            visit1 = rows.iloc[i]
            visit2 = rows.iloc[i + 1]
            for med in KEY_MEDS:
                if med not in rows.columns:
                    continue
                v1_status = visit1[med]
                v2_status = visit2[med]
                # Detect changes: No→Steady/Up/Down means started, Steady/Up/Down→No means stopped
                if pd.isna(v1_status) or pd.isna(v2_status):
                    continue
                started = (v1_status == 'No') and (v2_status in ['Steady', 'Up', 'Down'])
                stopped = (v1_status in ['Steady', 'Up', 'Down']) and (v2_status == 'No')
                dose_up = (v1_status in ['No', 'Steady', 'Down']) and (v2_status == 'Up')
                dose_down = (v1_status in ['No', 'Steady', 'Up']) and (v2_status == 'Down')
                if started or stopped or dose_up or dose_down:
                    change_type = 'started' if started else 'stopped' if stopped else 'dose_up' if dose_up else 'dose_down'
                    changes.append({
                        'patient_nbr': pid,
                        'medication': med,
                        'change_type': change_type,
                        'encounter_before': visit1['encounter_id'],
                        'encounter_after': visit2['encounter_id'],
                        'readmitted_any_before': visit1.get('readmitted_any', np.nan),
                        'readmitted_any_after': visit2.get('readmitted_any', np.nan),
                        'readmitted_30_before': visit1.get('readmitted_30', np.nan),
                        'readmitted_30_after': visit2.get('readmitted_30', np.nan),
                    })

    changes_df = pd.DataFrame(changes)
    print(f"Detected {len(changes_df)} medication changes across {changes_df['patient_nbr'].nunique() if len(changes_df) > 0 else 0} patients")
    return changes_df


def validate_interventions(changes_df):
    """Compare KG-predicted vs observed effects of medication changes."""
    results = {}
    for med in KEY_MEDS:
        med_changes = changes_df[changes_df['medication'] == med]
        started = med_changes[med_changes['change_type'] == 'started']
        if len(started) < 20:
            continue

        # Observed: readmission rate before vs after starting medication
        rate_before = started['readmitted_any_before'].mean()
        rate_after = started['readmitted_any_after'].mean()
        observed_effect = rate_after - rate_before

        # KG predicted effect
        predicted_effect = KG_INTERVENTION_ATE.get(med, 0)

        # Statistical test
        if len(started) > 1:
            t_stat, p_val = stats.ttest_rel(
                started['readmitted_any_before'].values,
                started['readmitted_any_after'].values
            ) if len(started) >= 2 else (0, 1)
        else:
            t_stat, p_val = 0, 1

        results[med] = {
            'n_started': int(len(started)),
            'rate_before': float(rate_before),
            'rate_after': float(rate_after),
            'observed_effect': float(observed_effect),
            'predicted_effect': float(predicted_effect),
            'prediction_error': float(abs(observed_effect - predicted_effect)),
            'direction_match': bool((observed_effect < 0) == (predicted_effect < 0)) if observed_effect != 0 else None,
            'p_value': float(p_val),
        }
        print(f"  {med}: n={len(started)}, observed={observed_effect:+.4f}, predicted={predicted_effect:+.4f}, "
              f"direction={'✓' if results[med]['direction_match'] else '✗'}, p={p_val:.4f}")

    return results


def compute_kg_risk_scores(df):
    """Compute NCD-CIE risk scores for diabetes patients."""
    scores = pd.Series(0.0, index=df.index)

    # HbA1c elevated
    scores += KG_RISK_WEIGHTS['HbA1c_elevated'] * (df['A1Cresult'].isin(['>7', '>8'])).astype(float)
    # High medication count
    scores += KG_RISK_WEIGHTS['num_medications_high'] * (df['num_medications'] > df['num_medications'].median()).astype(float)
    # High diagnosis count
    scores += KG_RISK_WEIGHTS['num_diagnoses_high'] * (df['number_diagnoses'] > df['number_diagnoses'].median()).astype(float)
    # Emergency history
    scores += KG_RISK_WEIGHTS['emergency_history'] * (df['number_emergency'] > 0).astype(float)
    # Inpatient history
    scores += KG_RISK_WEIGHTS['inpatient_history'] * (df['number_inpatient'] > 0).astype(float)
    # Long hospital stay
    scores += KG_RISK_WEIGHTS['long_stay'] * (df['time_in_hospital'] > df['time_in_hospital'].median()).astype(float)

    return scores


def plot_intervention_comparison(results, outpath):
    """Plot predicted vs observed intervention effects."""
    if not results:
        return
    meds = list(results.keys())
    predicted = [results[m]['predicted_effect'] for m in meds]
    observed = [results[m]['observed_effect'] for m in meds]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Bar comparison
    x = np.arange(len(meds))
    width = 0.35
    ax1.bar(x - width/2, predicted, width, label='KG Predicted', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, observed, width, label='Observed', color='coral', alpha=0.8)
    ax1.set_xlabel('Medication')
    ax1.set_ylabel('Effect on Readmission Rate')
    ax1.set_title('NCD-CIE Predicted vs Observed Intervention Effects')
    ax1.set_xticks(x)
    ax1.set_xticklabels(meds, rotation=45, ha='right')
    ax1.legend()
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)

    # Scatter: predicted vs observed
    ax2.scatter(predicted, observed, s=80, c='steelblue', alpha=0.7)
    for i, med in enumerate(meds):
        ax2.annotate(med, (predicted[i], observed[i]), fontsize=8, ha='left')
    lims = [min(min(predicted), min(observed)) - 0.02, max(max(predicted), max(observed)) + 0.02]
    ax2.plot(lims, lims, 'k--', alpha=0.5, label='Perfect prediction')
    ax2.set_xlabel('KG Predicted Effect')
    ax2.set_ylabel('Observed Effect')
    ax2.set_title('Prediction vs Observation Correlation')
    ax2.legend()

    # Correlation
    if len(predicted) >= 3:
        r, p = stats.pearsonr(predicted, observed)
        ax2.text(0.05, 0.95, f'r={r:.3f}, p={p:.3f}', transform=ax2.transAxes, va='top')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"  Intervention plot saved: {outpath}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print("=" * 60)
    print("NCD-CIE What-If Validation: Diabetes 130 US Hospitals")
    print("=" * 60)

    # Load and prepare
    df = load_data()
    df = encode_readmission(df)
    print(f"Readmission rates: <30d={df['readmitted_30'].mean():.3f}, any={df['readmitted_any'].mean():.3f}")

    # KG risk score evaluation
    print("\n--- KG Risk Score vs Readmission ---")
    kg_scores = compute_kg_risk_scores(df)
    from sklearn.metrics import roc_auc_score
    auc_any = roc_auc_score(df['readmitted_any'], kg_scores)
    auc_30 = roc_auc_score(df['readmitted_30'], kg_scores)
    print(f"  KG Risk Score AUC (any readmission): {auc_any:.4f}")
    print(f"  KG Risk Score AUC (<30d readmission): {auc_30:.4f}")

    # Multi-visit analysis
    print("\n--- Multi-Visit Medication Change Analysis ---")
    multi_df = get_multi_visit_patients(df, min_visits=2)
    changes_df = detect_med_changes(multi_df)

    if len(changes_df) > 0:
        print(f"\nMedication changes by type:")
        print(changes_df['change_type'].value_counts().to_string())
        print(f"\nMedication changes by drug:")
        print(changes_df['medication'].value_counts().head(10).to_string())

        # Validate interventions
        print("\n--- Intervention Validation ---")
        intervention_results = validate_interventions(changes_df)

        # Correlation summary
        if len(intervention_results) >= 3:
            pred = [intervention_results[m]['predicted_effect'] for m in intervention_results]
            obs = [intervention_results[m]['observed_effect'] for m in intervention_results]
            r, p = stats.pearsonr(pred, obs)
            print(f"\n  Overall correlation: r={r:.3f}, p={p:.3f}")
            direction_matches = sum(1 for m in intervention_results if intervention_results[m].get('direction_match'))
            total = sum(1 for m in intervention_results if intervention_results[m].get('direction_match') is not None)
            print(f"  Direction accuracy: {direction_matches}/{total} ({direction_matches/max(total,1):.0%})")
        else:
            r, p = None, None
            intervention_results = {}

        # Plot
        print("\n--- Generating Plots ---")
        plot_intervention_comparison(intervention_results, os.path.join(RESULTS_DIR, 'diabetes130_interventions.png'))
    else:
        intervention_results = {}
        r, p = None, None

    # Save results
    all_results = {
        'dataset': 'Diabetes 130 US Hospitals',
        'n_encounters': len(df),
        'n_patients': int(df['patient_nbr'].nunique()),
        'readmission_rates': {
            'any': float(df['readmitted_any'].mean()),
            '30_day': float(df['readmitted_30'].mean()),
        },
        'kg_risk_auc': {
            'any_readmission': float(auc_any),
            '30d_readmission': float(auc_30),
        },
        'n_multi_visit_patients': int(multi_df['patient_nbr'].nunique()) if len(multi_df) > 0 else 0,
        'n_medication_changes': len(changes_df),
        'intervention_results': intervention_results,
        'overall_correlation': {'r': float(r) if r else None, 'p': float(p) if p else None},
    }
    with open(os.path.join(RESULTS_DIR, 'diabetes130_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {RESULTS_DIR}/diabetes130_results.json")

    return all_results


if __name__ == '__main__':
    main()
