# Methodology Review — NCD-CIE-LLM v20

**Reviewer Role:** Methodology Expert  
**Paper:** NCD-CIE: An LLM-Augmented Causal Knowledge Graph Engine for Predictive and Preventive Non-Communicable Disease Healthcare (v20, 14 pages)  
**Date:** 2026-02-18  
**Changes from v19:** γ empirical optimization via grid search; Gwet's AC1 replacing raw agreement for LLM validation; LLM agreement stratified by evidence grade; T2DM/CKD cross-sectional validation; 5-fold CV noted for SCORE2 analysis.

---

## Dimension Scores

| Dimension | Score | Comments |
|---|---|---|
| **Research Design** | 7/10 | Well-structured multi-component system. Zero-fit cross-population design remains strong. T2DM/CKD validation extends multi-NCD claims, though cross-sectional only. Still conflates system description with empirical evaluation. |
| **Causal Framework Rigor** | 6/10 | Honest "approximate rung-2" positioning unchanged. No identifiability analysis, no backdoor/frontdoor adjustment. γ now empirically grounded (grid search, γ*=0.65, RMSE=0.74)—a meaningful improvement but validated against only 6 RCT benchmarks. Linearised truncated factorisation remains a strong simplification. |
| **Knowledge Graph Construction** | 7/10 | Bradford Hill protocol (κ=0.78, ICC=0.84) solid. Evidence grading A/B/C with Grade C down-scaling appropriate. 20/107 edges shown; full spec deferred to supplement. No structural sensitivity analysis (edge deletion/addition impact). |
| **Evaluation Methodology** | 7/10 | Multi-faceted evaluation strengthened: SCORE2 zero-fit (AUC=0.704), 5-fold CV explicitly noted, 6 RCT face validations (2 independent), T2DM/CKD cross-sectional. LLM NL parser and explanation generator still have zero quantitative evaluation—remains the biggest gap given the paper's LLM-augmentation framing. |
| **Statistical Rigor** | 7/10 | Improved from v19. Gwet's AC1=0.86 [0.77, 0.93] properly addresses the prevalence paradox for LLM validation—methodologically correct choice. Evidence-grade stratification (A: 95.7%, B: 78.6%, C: 60.0%) adds interpretability. γ grid search with RMSE metric is appropriate. Delta method, E-values, DeLong, NRI all standard. Minor: no CI on NRI; composite independence tested on NHANES not Framingham. |
| **Reproducibility** | 8/10 | Open-source, GitHub + Zenodo, prompt templates, algorithm pseudocode. LLM versions/dates still not pinned. No container/environment specification. NHANES preprocessing underdescribed. |
| **Threats to Validity** | 6/10 | Honest limitations. E-values (2.4–3.0) moderate. Age-stratified AUC decline acknowledged (0.591 for >60). But: no formal DAG misspecification analysis; causal sufficiency defended only via E-values; only 7/~50 predictors in Framingham; no discussion of specific plausible confounders (SES, genetics) and bias directions. T2DM/CKD validation is cross-sectional with restricted predictors—appropriately noted but limits the multi-NCD generalizability claim. |

---

## Summary Scores

| Metric | Value |
|---|---|
| **Overall Score** | **7/10** |
| **Recommendation** | **Weak Accept** |
| **Confidence** | **4/5** |

---

## Top 3 Strengths

1. **Zero-fit cross-population validation with proper statistical treatment.** SCORE2 coefficients on Framingham (AUC=0.704, 5-fold CV) remains genuinely circularity-free. The addition of Gwet's AC1=0.86 for LLM validation properly handles the prevalence paradox—a methodologically sophisticated choice that directly addresses a v19 weakness. Evidence-grade stratification (A: 95.7% → C: 60.0%) adds meaningful interpretive structure.

2. **Empirically grounded attenuation parameter.** The γ grid search over [0.3, 1.0] with RMSE minimization against 6 RCT benchmarks (γ*=0.65, default 0.7 within 2%) transforms a previously heuristic parameter into an empirically validated one. Sensitivity table across γ∈[0.5, 0.9] provides full transparency. This directly addresses a major v19 weakness.

3. **Principled LLM integration with multi-NCD extension.** LLMs as augmentation layers (not replacement) remains methodologically sound. T2DM (AUC=0.68) and CKD (AUC=0.67) zero-fit cross-sectional validation, while modest, demonstrates the framework generalizes beyond CVD. Honest comparison with purpose-built tools (FINDRISC, Tangri) is commendable.

---

## Top 3 Remaining Weaknesses

1. **LLM augmentation layers (2 of 3) lack any quantitative evaluation.** The NL query parser and counterfactual explanation generator—core claimed contributions (C2)—have no parsing accuracy, no user study, no explanation quality metrics. A "50-query test suite" is mentioned as developed but results not reported. For a paper with "LLM-Augmented" in the title, this is the single most significant methodological gap. Even a small-scale pilot (e.g., parsing accuracy on the 50 queries, clinician rating of 20 explanations) would substantially strengthen the claim.

2. **Causal sufficiency assumption defended only by E-values.** With 7/~50 predictors in Framingham validation, omitted variable bias is near-certain. E-values of 2.4–3.0 indicate moderate (not strong) robustness. No formal sensitivity analysis (Manski bounds, partial identification), no discussion of specific plausible confounders and expected bias directions. The system claims causal reasoning but the validation cannot distinguish causal from associational performance.

3. **Validation cohort homogeneity and temporal limitations.** Outcome-based validation is CVD-only on Framingham (predominantly White American, decades old). T2DM/CKD validation is cross-sectional (no outcomes), with only 4 predictors each. Transportability to non-Western populations is discussed theoretically (Bareinboim-Pearl) but untested. The gap between the system's broad multi-NCD, multi-population aspirations and the narrow validation evidence base remains significant.

---

## Delta from v19 → v20

| Issue from v19 | Status in v20 |
|---|---|
| γ empirically unjustified | ✅ Fixed: grid search, γ*=0.65, RMSE=0.74 |
| LLM validation lacks κ/baseline | ✅ Fixed: Gwet's AC1=0.86, evidence-grade stratification |
| No multi-NCD outcome validation | ⚠️ Partially addressed: T2DM/CKD cross-sectional (no outcomes) |
| LLM components unquantified | ❌ Unchanged: still no evaluation |
| Causal sufficiency underdefended | ❌ Unchanged: E-values only |
| Validation cohort limitations | ❌ Unchanged: Framingham + NHANES only |

**Net assessment:** v20 addresses two of the most tractable statistical/methodological weaknesses from v19 (γ justification, LLM agreement statistics). The remaining weaknesses require new data collection (clinician studies, diverse cohorts, outcome-based T2DM/CKD validation) and are harder to fix within a single revision cycle. The paper has moved from borderline to solidly in weak-accept territory.
