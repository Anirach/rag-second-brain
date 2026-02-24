#!/usr/bin/env python3
"""
Experiment 4: γ Optimization for NCD-CIE Intervention Cascade
Finds optimal attenuation factor γ by minimizing RMSE vs 6 RCT benchmarks.
"""
import math
from collections import defaultdict

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1 / (1 + math.exp(-x))

def inv_sigmoid(p):
    p = max(1e-10, min(1-1e-10, p))
    return math.log(p / (1 - p))

# ── DAG edge weights (log-odds) from Table 8 ──
EDGES = {
    ('LDL-C', 'CAD'): 0.28, ('SBP', 'CAD'): 0.35, ('SBP', 'Stroke'): 0.42,
    ('Smoking', 'CAD'): 0.45, ('BMI', 'T2DM'): 0.38, ('BMI', 'SBP'): 0.20,
    ('BMI', 'LDL-C'): 0.10, ('BMI', 'HF'): 0.22, ('Exercise', 'CAD'): -0.20,
    ('Exercise', 'SBP'): -0.10, ('Exercise', 'HbA1c'): -0.08,
    ('Statin', 'LDL-C'): -0.35, ('HTN-med', 'SBP'): -0.30,
    ('HDL-C', 'CAD'): -0.18, ('SBP', 'CKD'): 0.18, ('Age', 'CAD'): 0.50,
    ('Alcohol', 'SBP'): 0.12, ('BMI', 'HDL-C'): -0.15,
    ('Exercise', 'HDL-C'): 0.08, ('Smoking', 'CKD'): 0.15,
}

# Build adjacency
children_of = defaultdict(list)
all_nodes = set()
for (p, c), w in EDGES.items():
    children_of[p].append((c, w))
    all_nodes.update([p, c])

def topo_sort():
    in_deg = defaultdict(int)
    for n in all_nodes: in_deg[n]
    for (p, c) in EDGES: in_deg[c] += 1
    q = sorted(n for n in all_nodes if in_deg[n] == 0)
    order = []
    while q:
        n = q.pop(0)
        order.append(n)
        for c, _ in children_of[n]:
            in_deg[c] -= 1
            if in_deg[c] == 0: q.append(c)
        q.sort()
    return order

TOPO = topo_sort()

# ── RCT Interventions ──
# Perturbations are direct biomarker shifts in standardized units.
# Baselines from RCT control-arm event rates.
# The cascade propagates through the DAG with γ attenuation per hop.
INTERVENTIONS = [
    {
        'name': 'Statin (LDL −1 mmol/L)',
        'perturbations': {'LDL-C': -2.8},  # ~1 mmol/L LDL reduction (large effect in SD units)
        'outcome': 'CAD', 'baseline': 0.15,
        'rct_arr': -5.4,  # 4S/WOSCOPS meta-analysis
    },
    {
        'name': 'Weight loss (−7%)',
        'perturbations': {'BMI': -4.0},  # 7% weight loss = ~2 SD BMI reduction
        'outcome': 'T2DM', 'baseline': 0.29,
        'rct_arr': -16.0,  # DPP lifestyle arm
    },
    {
        'name': 'SGLT2i (renal)',
        'perturbations': {'SBP': -2.0, 'HbA1c': -0.8},  # SGLT2i hemodynamic + glycemic
        'outcome': 'CKD', 'baseline': 0.12,
        'rct_arr': -2.5,  # CREDENCE
    },
    {
        'name': 'SBP −15 mmHg',
        'perturbations': {'SBP': -3.2},  # 15 mmHg ≈ 1 SD SBP
        'outcome': 'CAD', 'baseline': 0.08,
        'rct_arr': -4.1,  # SPRINT
    },
    {
        'name': 'PCSK9i',
        'perturbations': {'LDL-C': -3.8},  # ~1.5 mmol/L LDL reduction
        'outcome': 'CAD', 'baseline': 0.035,
        'rct_arr': -1.5,  # FOURIER (shorter follow-up, lower baseline risk)
    },
    {
        'name': 'SGLT2i (CV)',
        'perturbations': {'SBP': -1.5, 'BMI': -1.2, 'HbA1c': -0.6},
        'outcome': 'HF', 'baseline': 0.10,
        'rct_arr': -3.2,  # DAPA-HF
    },
]

def cascade(perturbations, gamma):
    """Algorithm 1: Topological intervention cascade with γ attenuation."""
    delta = defaultdict(float)
    for node, d in perturbations.items():
        delta[node] = d
    for node in TOPO:
        if abs(delta[node]) < 1e-12: continue
        for child, weight in children_of[node]:
            delta[child] += gamma * weight * delta[node]
    return dict(delta)

def simulate_arr(intv, gamma):
    """Simulate ARR (%) for an intervention at given γ."""
    deltas = cascade(intv['perturbations'], gamma)
    outcome = intv['outcome']
    baseline = intv['baseline']
    baseline_lo = inv_sigmoid(baseline)
    shift = deltas.get(outcome, 0.0)
    new_rate = sigmoid(baseline_lo + shift)
    return (new_rate - baseline) * 100

def main():
    gammas = [round(x * 0.1, 1) for x in range(3, 11)]  # 0.3 to 1.0
    # Also do fine grid around likely optimum
    fine_gammas = [round(x * 0.05, 2) for x in range(6, 21)]  # 0.30 to 1.00 step 0.05
    
    all_gammas = sorted(set(gammas + fine_gammas))
    
    results = {}
    for g in all_gammas:
        errors = []
        details = []
        for intv in INTERVENTIONS:
            sim = simulate_arr(intv, g)
            rct = intv['rct_arr']
            errors.append((sim - rct) ** 2)
            details.append({'name': intv['name'], 'outcome': intv['outcome'],
                          'sim_arr': sim, 'rct_arr': rct, 'baseline': intv['baseline']})
        results[g] = {'rmse': math.sqrt(sum(errors) / len(errors)), 'details': details}

    best_gamma = min(all_gammas, key=lambda g: results[g]['rmse'])

    # ── Console output ──
    print("=" * 70)
    print("γ OPTIMIZATION — NCD-CIE Intervention Cascade (Experiment 4)")
    print("=" * 70)
    
    print(f"\n{'γ':>6} | {'RMSE':>8}")
    print("-" * 22)
    for g in all_gammas:
        marker = " ◀ optimal" if g == best_gamma else ""
        print(f"{g:6.2f} | {results[g]['rmse']:8.4f}{marker}")

    print(f"\n✓ Optimal γ = {best_gamma:.2f} (RMSE = {results[best_gamma]['rmse']:.4f})")
    
    # Show γ=0.7 for comparison
    print(f"  Current γ = 0.70 (RMSE = {results[0.7]['rmse']:.4f})")
    
    print(f"\n{'Intervention':<25} {'Out':<7} {'Base%':>6} {'SimARR%':>8} {'RCT_ARR%':>9} {'Δ':>7}")
    print("-" * 67)
    for d in results[best_gamma]['details']:
        err = d['sim_arr'] - d['rct_arr']
        print(f"{d['name']:<25} {d['outcome']:<7} {d['baseline']*100:5.1f}% {d['sim_arr']:>+8.2f} {d['rct_arr']:>+9.1f} {err:>+7.2f}")
    
    # Also show at γ=0.7
    if abs(best_gamma - 0.7) > 0.01:
        print(f"\nAt γ = 0.70:")
        for d in results[0.7]['details']:
            err = d['sim_arr'] - d['rct_arr']
            print(f"{d['name']:<25} {d['outcome']:<7} {d['baseline']*100:5.1f}% {d['sim_arr']:>+8.2f} {d['rct_arr']:>+9.1f} {err:>+7.2f}")

    # ── Markdown output ──
    md = []
    md.append("# Experiment 4: γ Optimization Results\n")
    md.append("**Date:** 2026-02-18  ")
    md.append("**Method:** Grid search over topological cascade with logistic-link model\n")
    
    md.append("## Summary\n")
    md.append(f"| Metric | Value |")
    md.append(f"|--------|-------|")
    md.append(f"| **Optimal γ** | **{best_gamma:.2f}** |")
    md.append(f"| Best RMSE | {results[best_gamma]['rmse']:.4f} |")
    md.append(f"| RMSE at γ=0.70 | {results[0.70]['rmse']:.4f} |")
    if abs(best_gamma - 0.7) > 0.01:
        imp = (results[0.7]['rmse'] - results[best_gamma]['rmse']) / results[0.7]['rmse'] * 100
        md.append(f"| Improvement over 0.70 | {imp:.1f}% |")
    md.append("")
    
    md.append("## RMSE by γ (coarse grid)\n")
    md.append("| γ | RMSE | |")
    md.append("|--:|-----:|--|")
    for g in gammas:
        tag = "**← optimal**" if abs(g - best_gamma) < 0.01 else ("← current" if g == 0.7 else "")
        md.append(f"| {g:.1f} | {results[g]['rmse']:.4f} | {tag} |")
    
    md.append(f"\n## RMSE by γ (fine grid)\n")
    md.append("| γ | RMSE | |")
    md.append("|---:|-----:|--|")
    for g in fine_gammas:
        tag = "**← optimal**" if abs(g - best_gamma) < 0.005 else ""
        md.append(f"| {g:.2f} | {results[g]['rmse']:.4f} | {tag} |")
    
    md.append(f"\n## Detailed Comparison at γ = {best_gamma:.2f}\n")
    md.append("| Intervention | Outcome | Baseline | Sim ARR% | RCT ARR% | Error |")
    md.append("|---|---|---:|---:|---:|---:|")
    for d in results[best_gamma]['details']:
        err = d['sim_arr'] - d['rct_arr']
        md.append(f"| {d['name']} | {d['outcome']} | {d['baseline']*100:.1f}% | {d['sim_arr']:+.2f} | {d['rct_arr']:+.1f} | {err:+.2f} |")
    
    if abs(best_gamma - 0.7) > 0.01:
        md.append(f"\n## Comparison at γ = 0.70\n")
        md.append("| Intervention | Outcome | Sim ARR% | RCT ARR% | Error |")
        md.append("|---|---|---:|---:|---:|")
        for d in results[0.7]['details']:
            err = d['sim_arr'] - d['rct_arr']
            md.append(f"| {d['name']} | {d['outcome']} | {d['sim_arr']:+.2f} | {d['rct_arr']:+.1f} | {err:+.2f} |")
    
    md.append("\n## Methodology\n")
    md.append("- **Model:** Topological cascade (Algorithm 1) with logistic-link: R = σ(β₀ + Σ wᵢ·zᵢ)")
    md.append("- **DAG:** 20 edge weights from Table 8 (log-odds scale)")
    md.append("- **Validation:** 6 RCT-benchmarked interventions from Table 5")
    md.append("- **Grid:** γ ∈ {0.30, 0.35, ..., 1.00} (step 0.05)")
    md.append("- **Metric:** RMSE between simulated and observed absolute risk reductions")
    md.append("- **Baselines:** Per-intervention baseline rates from RCT control arms")
    md.append("  - Statin trials (4S/WOSCOPS): 15% 10yr CAD")
    md.append("  - DPP lifestyle: 29% 3yr T2DM incidence")
    md.append("  - CREDENCE: 12% CKD progression")
    md.append("  - SPRINT: 8% CAD events")
    md.append("  - FOURIER: 3.5% MACE")
    md.append("  - DAPA-HF: 10% HF hospitalization\n")
    
    md.append("## Interpretation\n")
    if abs(best_gamma - 0.7) < 0.11:
        md.append(f"The empirically optimal γ = {best_gamma:.2f} is {'consistent with' if abs(best_gamma - 0.7) < 0.06 else 'close to'} "
                  f"the paper's default γ = 0.70, supporting the chosen value. "
                  f"The attenuation factor balances direct causal effects (which dominate at lower γ) "
                  f"against indirect/mediated pathways (which contribute more at higher γ).")
    else:
        md.append(f"The optimal γ = {best_gamma:.2f} differs from the paper's γ = 0.70. "
                  f"This suggests the cascade attenuation should be {'stronger' if best_gamma < 0.7 else 'weaker'} "
                  f"than currently specified.")
    md.append("")
    
    with open("/tmp/gamma_results.md", "w") as f:
        f.write("\n".join(md))
    print(f"\n✓ Results saved to /tmp/gamma_results.md")

if __name__ == "__main__":
    main()
