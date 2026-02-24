# NCD-CIE v5 — Methodology Expert Review

## VERDICT: WEAK ACCEPT

---

## Methodology Assessment

### 1. Research Design
The multi-source validation strategy (synthetic → NHANES → RCT face-validity) is well-structured and follows a logical progression. The ablation study (shuffled weights control) is a nice touch that confirms the causal structure matters. However, the overall design validates *concordance with existing models* rather than *causal correctness*—a fundamental tension the paper partially acknowledges but doesn't resolve.

### 2. Statistical Methods
- Logistic-link scoring is appropriate and well-justified for the event rates described.
- Inverse-variance meta-analysis for edge weight fusion is standard and correct.
- First-order Taylor expansion for uncertainty propagation is reasonable but may underestimate variance for high-risk patients (where the sigmoid is non-linear).
- C-statistics, calibration slopes, Brier scores, NRI, and DCA are all appropriate metrics.
- Pearson r for comparing two risk scores is acceptable but should be supplemented with Bland-Altman analysis.

### 3. Reproducibility
Moderate. The paper specifies the graph structure (51 nodes, 107 edges, 8 domains), the algorithm (Algorithm 1), hyperparameters (γ=0.7, d_max=3), and data source (NHANES 2017–2020). However, the actual edge weights and their sources are not provided in the paper—this is the single biggest reproducibility gap. The Bradford Hill assessment process is described but the individual edge assessments are not available. The claim of open-source release partially mitigates this.

### 4. Baselines
- Framingham is the primary baseline, which is appropriate for CVD.
- No comparison against QRISK3 or SCORE2 despite mentioning them in related work.
- No comparison against any ML-based model despite citing Alaa et al.
- No comparison against DoWhy or CausalNex for the causal inference components.
- The baseline set is **too narrow** for a paper making broad claims.

### 5. Data
- NHANES is an appropriate external validation source (large, representative, publicly available).
- Selection criteria (age 30–75, complete labs, no prior CVD) are reasonable but the "complete lab panel" requirement likely introduces selection bias toward healthier/more compliant patients.
- Synthetic data generated from the model's own structural equations is circular for testing discrimination—it can only test internal consistency, not validity.

---

## Specific Methodology Issues

1. **[CRITICAL] Circular validation with synthetic data.** Generating synthetic data from the model's own structural equations and then showing the model fits that data well (C-statistics 0.76–0.82) proves nothing about the model's validity. This is testing whether the model can recover its own parameters. The C-statistics being <1.0 is merely due to added Gaussian noise. This section should be reframed as "internal consistency check" rather than "validation."

2. **[CRITICAL] No outcome validation.** NHANES is cross-sectional. The paper compares NCD-CIE *risk scores* against *Framingham risk scores*, not against actual outcomes. High correlation with Framingham (r=0.91) shows the models agree, not that either is correct. The C-statistics reported for NHANES appear to use Framingham scores as pseudo-outcomes, which is methodologically problematic. This needs clarification.

3. **[MAJOR] Rung-3 claim is overstated.** The system uses a linear approximation on log-odds with topological propagation. This is closer to a linear structural equation model with do-calculus interventions (rung 2) than true counterfactual reasoning (rung 3). Rung 3 requires reasoning about individual-level counterfactuals with abduction-action-prediction steps. The paper mentions ETT but the algorithm doesn't perform abduction—it simply replaces values and propagates. The distinction matters and should be more carefully argued.

4. **[MAJOR] Hyperparameter tuning on validation data.** The sensitivity analysis (Table 6) shows γ=0.7 was selected because it best matches RCT evidence. If the same RCT comparisons are then used for face-validity assessment, this is circular. The paper should clarify whether hyperparameters were tuned and validated on separate data.

5. **[MAJOR] PC algorithm comparison is weak.** 64.5% edge agreement with SHD=112 is presented positively but is actually mediocre. With 51 nodes, the maximum possible edges are ~1,275 (directed). SHD of 112 relative to 107 expert edges is high. The PC algorithm's faithfulness assumption is known to fail in finite samples with correlated biomarkers. This comparison needs more nuanced interpretation.

6. **[MAJOR] Missing confidence intervals on key metrics.** C-statistics, calibration slopes, and Pearson r are reported without confidence intervals (except for NRI). Bootstrap CIs should be provided for all discrimination and calibration metrics.

7. **[MINOR] Inter-rater reliability reporting.** κ=0.78 and ICC=0.84 are reported for edge curation, but the number of raters (2+1 adjudicator) and the total number of candidate edges assessed are not stated. How many edges were rejected?

8. **[MINOR] Composite score independence assumption.** Eq. 5 assumes conditional independence of CVD, T2DM, and CKD given risk factors. This is acknowledged as a limitation but no sensitivity analysis is provided for this assumption.

9. **[MINOR] Attenuation factor lacks theoretical justification.** γ=0.7 is selected empirically. A principled approach (e.g., derived from the SCM framework) would strengthen the claim of causal grounding.

---

## Validation Concerns

- **Synthetic data experiments:** Not convincing as validation—they demonstrate internal consistency only. Acceptable as a sanity check if reframed.

- **NHANES validation:** Partially rigorous. The correlation with Framingham is informative for concordance but not for accuracy. The paper needs to clarify what the C-statistics are computed against (actual outcomes or Framingham labels). If the latter, this is a serious methodological flaw.

- **RCT face-validity:** Appropriate as a qualitative check. The comparison is reasonable—simulating trial interventions and checking alignment with observed effect sizes. The DPP discrepancy is honestly reported. However, only 4 RCTs is thin; expanding to 8–10 would strengthen this considerably.

- **PC algorithm / SHD:** SHD is a standard metric for graph comparison but has known limitations (treats all edge errors equally regardless of clinical importance). A weighted version prioritizing high-impact edges would be more informative. The comparison is useful but the interpretation is too generous.

- **Subgroup fairness:** The groups are standard and meaningful (race/ethnicity, age, sex). The max C-statistic gap of 0.05 is acceptable. However, calibration by subgroup is not reported—a model can discriminate well but be miscalibrated for specific groups.

---

## Statistical Red Flags

1. **Potential circularity:** Hyperparameter selection (γ, d_max) using RCT comparisons, then reporting those same comparisons as validation. Not p-hacking per se, but methodologically concerning.

2. **Selective reporting risk:** Only 4 RCTs are compared. Were others tried and excluded? The paper should state whether these were pre-specified.

3. **No multiple comparison correction:** Subgroup analyses across 10 strata without correction. Given these are descriptive rather than hypothesis-testing, this is minor.

4. **No leakage concerns** detected—NHANES data is used appropriately as external validation, and the model was not trained on it.

---

## Soundness Score: 3/5

The core framework is well-motivated and technically competent, but critical validation gaps (no outcome data, circular synthetic validation, overstated rung-3 claims) prevent a higher score.

## Reproducibility Score: 3/5

Algorithm, hyperparameters, and data sources are specified, but the 107 edge weights and their evidence sources—the core intellectual contribution—are not provided in the paper. Open-source release is promised but not verified.

---

## Recommendations

### Must fix before publication:
1. **Clarify NHANES C-statistics:** State explicitly what outcome/label they are computed against. If using Framingham as pseudo-outcome, reframe as "concordance" not "discrimination."
2. **Reframe synthetic validation** as internal consistency check, not validation.
3. **Separate hyperparameter tuning from RCT validation** or acknowledge the circularity explicitly.
4. **Add confidence intervals** to all reported metrics (bootstrap or analytical).
5. **Tone down rung-3 claims** or provide a rigorous argument for why the linear propagation constitutes true counterfactual reasoning vs. interventional reasoning.

### Strongly recommended:
6. Add at least one more baseline (QRISK3 or SCORE2) for the NHANES comparison.
7. Provide subgroup calibration (not just discrimination).
8. Include a supplementary table of all 107 edges with weights, sources, and evidence grades.
9. Add more RCT comparisons (target 8–10).
10. Report Bland-Altman analysis alongside Pearson r for model comparison.

### Nice to have:
11. Prospective validation plan with timeline and target cohort.
12. Sensitivity analysis for the conditional independence assumption in the composite score.
