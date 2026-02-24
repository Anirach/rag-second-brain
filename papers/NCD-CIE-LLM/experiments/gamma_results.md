# Experiment 4: γ Optimization Results

**Date:** 2026-02-18  
**Method:** Grid search over topological cascade with logistic-link model

## Summary

| Metric | Value |
|--------|-------|
| **Optimal γ** | **0.65** |
| Best RMSE | 0.7363 |
| RMSE at γ=0.70 | 0.7515 |
| Improvement over 0.70 | 2.0% |

## RMSE by γ (coarse grid)

| γ | RMSE | |
|--:|-----:|--|
| 0.3 | 3.5729 |  |
| 0.4 | 2.5366 |  |
| 0.5 | 1.6261 |  |
| 0.6 | 0.9196 |  |
| 0.7 | 0.7515 | ← current |
| 0.8 | 1.1851 |  |
| 0.9 | 1.7439 |  |
| 1.0 | 2.2841 |  |

## RMSE by γ (fine grid)

| γ | RMSE | |
|---:|-----:|--|
| 0.30 | 3.5729 |  |
| 0.35 | 3.0406 |  |
| 0.40 | 2.5366 |  |
| 0.45 | 2.0634 |  |
| 0.50 | 1.6261 |  |
| 0.55 | 1.2359 |  |
| 0.60 | 0.9196 |  |
| 0.65 | 0.7363 | **← optimal** |
| 0.70 | 0.7515 |  |
| 0.75 | 0.9308 |  |
| 0.80 | 1.1851 |  |
| 0.85 | 1.4631 |  |
| 0.90 | 1.7439 |  |
| 0.95 | 2.0188 |  |
| 1.00 | 2.2841 |  |

## Detailed Comparison at γ = 0.65

| Intervention | Outcome | Baseline | Sim ARR% | RCT ARR% | Error |
|---|---|---:|---:|---:|---:|
| Statin (LDL −1 mmol/L) | CAD | 15.0% | -5.41 | -5.4 | -0.01 |
| Weight loss (−7%) | T2DM | 29.0% | -15.80 | -16.0 | +0.20 |
| SGLT2i (renal) | CKD | 12.0% | -2.26 | -2.5 | +0.24 |
| SBP −15 mmHg | CAD | 8.0% | -3.97 | -4.1 | +0.13 |
| PCSK9i | CAD | 3.5% | -1.72 | -1.5 | -0.22 |
| SGLT2i (CV) | HF | 10.0% | -1.44 | -3.2 | +1.76 |

## Comparison at γ = 0.70

| Intervention | Outcome | Sim ARR% | RCT ARR% | Error |
|---|---|---:|---:|---:|
| Statin (LDL −1 mmol/L) | CAD | -5.75 | -5.4 | -0.35 |
| Weight loss (−7%) | T2DM | -16.65 | -16.0 | -0.65 |
| SGLT2i (renal) | CKD | -2.42 | -2.5 | +0.08 |
| SBP −15 mmHg | CAD | -4.18 | -4.1 | -0.08 |
| PCSK9i | CAD | -1.81 | -1.5 | -0.31 |
| SGLT2i (CV) | HF | -1.54 | -3.2 | +1.66 |

## Methodology

- **Model:** Topological cascade (Algorithm 1) with logistic-link: R = σ(β₀ + Σ wᵢ·zᵢ)
- **DAG:** 20 edge weights from Table 8 (log-odds scale)
- **Validation:** 6 RCT-benchmarked interventions from Table 5
- **Grid:** γ ∈ {0.30, 0.35, ..., 1.00} (step 0.05)
- **Metric:** RMSE between simulated and observed absolute risk reductions
- **Baselines:** Per-intervention baseline rates from RCT control arms
  - Statin trials (4S/WOSCOPS): 15% 10yr CAD
  - DPP lifestyle: 29% 3yr T2DM incidence
  - CREDENCE: 12% CKD progression
  - SPRINT: 8% CAD events
  - FOURIER: 3.5% MACE
  - DAPA-HF: 10% HF hospitalization

## Interpretation

The empirically optimal γ = 0.65 is consistent with the paper's default γ = 0.70, supporting the chosen value. The RMSE curve shows a clear minimum in the 0.60–0.70 range, with γ = 0.70 only 2.0% worse than optimal. This validates the paper's choice as empirically grounded rather than ad hoc.

The largest residual is SGLT2i (CV) → HF (+1.76 pp), suggesting SGLT2i has HF-protective mechanisms beyond the hemodynamic/metabolic pathways captured in the current DAG (likely direct cardiac effects). Excluding this intervention, the remaining 5 show excellent agreement (max error < 0.4 pp).
