#!/usr/bin/env python3
"""Run all NCD-CIE validation pipelines and generate summary report."""

import json
import os
import sys

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')

def main():
    print("=" * 70)
    print("NCD-CIE DUAL DATASET VALIDATION")
    print("=" * 70)

    # Run Framingham
    print("\n\n" + "=" * 70)
    from framingham_validation import main as run_framingham
    fram_results = run_framingham()

    # Run Diabetes 130
    print("\n\n" + "=" * 70)
    from diabetes130_validation import main as run_diabetes
    diab_results = run_diabetes()

    # Generate summary report
    print("\n\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)

    report = []
    report.append("# NCD-CIE Validation Summary Report\n")
    report.append("## 1. Framingham Heart Study — Risk Prediction Validation\n")
    report.append(f"- **Dataset:** {fram_results['n_patients']} patients, {fram_results['n_chd_positive']} CHD+ ({fram_results['prevalence']:.1%})")
    report.append(f"- **NCD-CIE KG-only AUC:** {fram_results['metrics']['kg_only']['auc']:.4f}")
    report.append(f"- **NCD-CIE Blended AUC:** {fram_results['metrics']['blended']['auc']:.4f}")
    report.append(f"- **Logistic Regression AUC:** {fram_results['metrics']['logistic_regression']['auc']:.4f}")
    report.append(f"- **SCORE2 reference AUC:** 0.704")
    report.append(f"- **D'Agostino reference AUC:** 0.721\n")

    report.append("### Intervention Simulations")
    for med, res in fram_results.get('interventions', {}).items():
        report.append(f"- **{med}:** {res['n_eligible']} eligible, risk {res['original_risk']:.3f}→{res['predicted_post_intervention']:.3f} (actual CHD={res['actual_chd_rate']:.3f})")

    report.append(f"\n## 2. Diabetes 130 US Hospitals — What-If Intervention Validation\n")
    report.append(f"- **Dataset:** {diab_results['n_encounters']} encounters, {diab_results['n_patients']} patients")
    report.append(f"- **Multi-visit patients:** {diab_results['n_multi_visit_patients']}")
    report.append(f"- **Medication changes detected:** {diab_results['n_medication_changes']}")
    report.append(f"- **KG Risk AUC (any readmission):** {diab_results['kg_risk_auc']['any_readmission']:.4f}")
    report.append(f"- **KG Risk AUC (30d readmission):** {diab_results['kg_risk_auc']['30d_readmission']:.4f}")

    if diab_results.get('intervention_results'):
        report.append("\n### Medication Intervention Results")
        report.append("| Medication | N | Observed Δ | Predicted Δ | Direction |")
        report.append("|-----------|---|-----------|------------|-----------|")
        for med, res in diab_results['intervention_results'].items():
            direction = '✓' if res.get('direction_match') else '✗'
            report.append(f"| {med} | {res['n_started']} | {res['observed_effect']:+.4f} | {res['predicted_effect']:+.4f} | {direction} |")

        corr = diab_results.get('overall_correlation', {})
        if corr.get('r') is not None:
            report.append(f"\n- **Overall correlation:** r={corr['r']:.3f}, p={corr['p']:.3f}")

    report.append("\n## 3. Generated Artifacts\n")
    report.append("- `results/framingham_roc.png` — ROC curves")
    report.append("- `results/framingham_calibration.png` — Calibration curves")
    report.append("- `results/diabetes130_interventions.png` — Intervention comparison")
    report.append("- `results/framingham_results.json` — Full Framingham metrics")
    report.append("- `results/diabetes130_results.json` — Full Diabetes 130 metrics")

    report_text = '\n'.join(report)
    report_path = os.path.join(RESULTS_DIR, 'validation_summary.md')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(report_text)
    print(f"\n\nReport saved to {report_path}")


if __name__ == '__main__':
    main()
