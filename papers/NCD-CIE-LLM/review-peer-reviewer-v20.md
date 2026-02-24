# Peer Review: NCD-CIE v20

**Paper:** NCD-CIE: An LLM-Augmented Causal Knowledge Graph Engine for Predictive and Preventive Non-Communicable Disease Healthcare

**Version:** v20 (Feb 2026)

---

## Scores (1–10)

| Dimension | Score | Comments |
|---|---|---|
| **Novelty & Contribution** | 7 | Genuine integration of expert-curated causal KG + LLM augmentation layers is novel. The "LLM as augmentation, not replacement" framing is well-motivated. Each individual component is incremental, but the integration is the contribution. |
| **Methodology & Soundness** | 6 | Logistic-link scoring over a DAG with topological propagation is principled but approximate. Honest about rung-2 approximation status. Key concern: linearised cascade with γ attenuation lacks theoretical justification beyond empirical grid search. |
| **Experiments & Results** | 7 | Strong v20 improvements: γ optimisation, Gwet's AC1, T2DM/CKD validation. Zero-fit SCORE2 validation (AUC 0.704) is the right primary metric. RCT face validity across 6 trials is compelling. T2DM (0.68) and CKD (0.67) AUCs modest but fairly contextualised. |
| **Related Work** | 7 | Comprehensive coverage. Table 1 comparison effective. Could better discuss CausalBERT-type hybrid approaches and clinical NLP systems. |
| **Writing Quality** | 8 | Well-structured, clear, appropriately technical. Honest about limitations. Dense for 14 pages but well-organised. |
| **Reproducibility** | 7 | Open-source code + GitHub + Zenodo. Only 20/107 edges shown; preprocessing not detailed; LLM validation protocol lacks full prompt text. |

---

## Overall Assessment

| | |
|---|---|
| **Overall Score** | **6.5 / 10** |
| **Recommendation** | **Weak Accept** |
| **Confidence** | **4 / 5** (familiar with causal inference, clinical ML, and LLM evaluation) |

---

## Top 3 Strengths

1. **Principled LLM integration philosophy.** The "LLM augments but never replaces the formal engine" design is exactly right. The three-layer architecture (validation, NL interface, explanation) is clean with clear justification for each.

2. **Honest, multi-faceted evaluation.** SCORE2 zero-fit as primary result avoids circularity. Reporting both SCORE2 (0.704) and D'Agostino (0.721) with DeLong comparison, plus 6 RCT benchmarks, γ sensitivity, ablation, and E-values provides a thorough evidence package.

3. **Clinical utility design.** Multi-NCD composite risk, what-if simulation with uncertainty propagation, and the NL interface address real clinical workflow gaps. Lifestyle intervention mapping is practical and well-sourced.

---

## Top 3 Remaining Weaknesses

1. **No quantitative evaluation of LLM layers (NL parser + explainer).** A 50-query test suite is mentioned but zero results reported. For a paper centred on LLM augmentation, having no accuracy/faithfulness metrics for 2 of 3 LLM layers is a significant gap. Even small-scale precision/recall of query parsing or faithfulness scoring of explanations would help.

2. **Limited external generalisability.** Outcome validation on Framingham only (predominantly White American, 1990s). T2DM/CKD validation is cross-sectional (no outcomes). Transportability discussion cites Bareinboim-Pearl but provides no empirical recalibration evidence. AUC decline to 0.591 for >60 is concerning for the highest-risk demographic.

3. **Theoretical gap in cascade propagation.** γ attenuation justified only empirically (grid search over 6 RCTs). No formal analysis of when linearised truncated factorisation with multiplicative attenuation approximates true do-calculus. Under what conditions does Algorithm 1 yield consistent estimates?

---

## Improvements from Prior Versions

v20 shows clear progress:
- **γ optimisation:** Grid search over [0.3, 1.0] with RMSE; default γ=0.7 within 2% of optimum (γ*=0.65)
- **Gwet's AC1 = 0.86:** Addresses κ prevalence paradox (Feinstein & Cicchetti citation)
- **T2DM/CKD validation:** Cross-sectional NHANES results with honest comparison to purpose-built tools
- **Inter-model LLM agreement:** Claude 3.5 Sonnet as second validator (85.0%; inter-model 91.6%)
- **Stratified LLM agreement by evidence grade:** A=95.7%, B=78.6%, C=60.0%
- **Sensitivity table expanded** to all 6 interventions across γ range

---

## Minor Issues

- Table 2: "Consistency (3 runs)" — clarify if inconsistent runs changed direction or just confidence
- Composite risk independence assumption: when does the Spearman 0.97 approximation break down?
- Algorithm 1 line 4: `depth` function undefined formally
- Consider using Appendix space for LLM evaluation results instead of the 20-edge table
