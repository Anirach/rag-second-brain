# Peer Review: NCD-CIE v5

## VERDICT: WEAK ACCEPT

## Summary
The paper presents NCD-CIE, a causal knowledge graph with 107 expert-curated edges across 8 clinical domains, combined with a logistic-link risk engine and topological counterfactual simulator for NCD risk prediction. The system is validated on synthetic data, NHANES, and compared against four landmark RCTs, claiming to operate at rung 3 of Pearl's ladder of causation.

## Strengths
1. **Well-motivated problem.** The gap between associational risk calculators and causal/counterfactual reasoning is real and clinically important. The "what-if" framing is compelling.
2. **Comprehensive validation strategy.** The paper covers synthetic validation, NHANES external validation, RCT face validity, PC algorithm structural validation, subgroup fairness, and decision curve analysis—an unusually thorough validation suite for a systems paper.
3. **Formal rigor.** The causal KG is formally defined (Eq. 1), the uncertainty quantification is principled (Eqs. 3-4), and the topological cascade algorithm is clearly specified. The Bradford Hill curation protocol with inter-rater reliability adds methodological credibility.
4. **Transportability discussion.** The Bareinboim-Pearl transportability analysis for Thai/Asian populations is a thoughtful addition that grounds cross-population adaptation theoretically.
5. **Honest limitations.** Section 9 is refreshingly candid about the linear-on-log-odds assumption, static graph, and cross-sectional validation limitations.

## Weaknesses
1. **Overclaimed rung-3 positioning.** The paper claims rung-3 (counterfactual) reasoning, but the actual mechanism is a linear propagation through a DAG with attenuation—this is closer to a structured sensitivity analysis or interventional query (rung 2) than true counterfactual reasoning in Pearl's sense. True rung-3 requires reasoning about specific individuals with abduction of exogenous variables, which the linear-on-log-odds model does not perform. The ETT formula is cited but not actually computed via the three-step counterfactual procedure (abduction, action, prediction). This is the paper's most significant conceptual weakness.
2. **Circular validation concern.** The synthetic cohort is generated *from the model's own structural equations* (Section 7.1), so good C-statistics there are expected and uninformative. The NHANES validation compares against Framingham (correlation), not against actual 10-year outcomes—this shows agreement with another associational model, not ground-truth accuracy. There is no true prospective or even retrospective outcome validation.
3. **RCT face validity is weak evidence.** Showing that simulated effects "fall within 95% CIs" of RCTs is a very loose criterion—RCT confidence intervals are often wide. The DPP discrepancy (−11.3% vs. −16.0%) is substantial and the explanation ("pathways beyond weight loss") is hand-waving that actually undermines the claim of comprehensive causal modeling.
4. **PC algorithm comparison is inconclusive.** 64.5% edge agreement with SHD=112 is mediocre. The paper spins this positively but doesn't discuss whether the 38 expert-only edges might be wrong (not just "below detection threshold") or whether the 74 PC-only edges reveal missing structure.
5. **No real patient data validation.** The entire validation rests on synthetic data (circular), cross-sectional survey data (no outcomes), and qualitative RCT comparison. For a clinical decision support tool, this is insufficient. At minimum, a retrospective cohort with actual CVD/T2DM/CKD events is needed.
6. **Scalability and sensitivity to expert curation.** With 107 hand-curated edges, the system's accuracy is entirely dependent on expert judgment. The κ=0.78 is decent but not exceptional. What happens when experts disagree on edge weights? The ICC=0.84 suggests non-trivial disagreement.

## Questions for Authors
1. Can you clarify exactly how your counterfactual computation implements the three-step procedure (abduction, action, prediction) from Pearl's SCM framework? As described, Algorithm 1 appears to perform an interventional (do-calculus) computation rather than a true counterfactual. If so, the paper should be repositioned at rung 2.
2. Why was no retrospective cohort with actual longitudinal outcomes used for validation? NHANES has linked mortality data (NHANES-LMF) that could provide at least CVD mortality outcomes. Was this considered?
3. How sensitive are the results to the choice of evidence sources for edge weights? If you use different meta-analyses for the same edge, how much do risk predictions change?
4. The composite score (Eq. 5) assumes conditional independence—have you quantified the error this introduces for the diabetic nephropathy pathway you mention?

## Novelty Score: 3
The integration of causal KG + logistic scoring + counterfactual simulation is a reasonable systems contribution, but each component uses well-established techniques. The novelty is in the assembly, not the parts.

## Soundness Score: 3
The formal framework is sound, but the rung-3 claim is overclaimed, the validation lacks ground-truth outcomes, and the synthetic validation is circular. The methodology is competent but doesn't fully support the claims.

## Clarity Score: 4
Well-written, clearly structured, good use of tables and formal notation. The paper is easy to follow and the contributions are clearly stated. Minor issues with some overloaded tables.

## Significance Score: 3
Addresses an important clinical need, but without prospective or outcome-based validation, the clinical impact remains theoretical. The transportability discussion adds potential significance for underserved populations.

## Detailed Comments

### Section 1 (Introduction)
- Strong motivation. The Framingham/QRISK3 limitation is well-articulated.
- Six contributions (C1-C6) is aggressive for a 14-page paper; consider whether C5 and C6 are truly independent contributions or supporting analyses.

### Section 2 (Related Work)
- Fair treatment of DoWhy and CausalNex. However, missing discussion of recent causal ML work (e.g., CATE estimation, causal forests) that also targets heterogeneous treatment effects.
- No mention of existing causal approaches in clinical risk prediction (e.g., targeted learning/TMLE).

### Section 3 (Architecture)
- Table 1: The comparison is somewhat self-serving. Claiming BNs are "Slow" for real-time scoring and SCMs are "Slow" needs qualification—modern BN inference can be very fast for sparse networks of this size.
- Figure 1 is a placeholder box—this must be replaced with an actual diagram.

### Section 4 (Causal KG)
- Bradford Hill criteria application is well-described. The three-tier evidence grading is sensible.
- The 0.5 down-scaling for Grade C edges is arbitrary. Why not a Bayesian prior that reflects the actual uncertainty?
- Cycle handling via Tarjan's + fixed-point iteration is reasonable but the claim of "negligible impact" (Section 7.4 ablation) should be more precisely quantified.

### Section 5 (Risk Engine)
- The logistic-link justification (event rates <30%) is valid but should cite Green & Swets or similar for the Cox-logistic equivalence.
- Eq. 4 (Taylor expansion for variance) is a first-order approximation that may underestimate uncertainty for extreme profiles. Was this assessed?

### Section 6 (What-If Simulator)
- Algorithm 1: Line 4 has a subtle issue—the attenuation uses depth from $v_j$ to $v_p$ (parent), but shouldn't it use depth from $v_j$ to $v_k$ (current node)? This seems inconsistent.
- Default γ=0.7 and d_max=3 are chosen via sensitivity analysis, but the sensitivity analysis (Table 6) only shows results for the statin intervention. Is the optimal γ the same across all intervention types?

### Section 7 (Validation)
- Table 3 (synthetic): As noted, this is circular validation.
- Table 4 (NHANES): Pearson r=0.91 vs. Framingham is agreement, not accuracy. The paper should be more careful in distinguishing these.
- Table 5 (RCT): The column headers are confusing—both say "RCT" and "NCD-CIE" but the last column header is just "RCT" without explanation of what the number represents (observed relative risk reduction?).
- Table 7 (PC algorithm): SHD of 112 is quite high for a 51-node graph. This deserves more critical discussion.
- Table 8 (subgroup): C-stat for Asian subgroup (0.78) is lowest—ironic given the Thai transportability goal. Sample size for the Asian NHANES subgroup should be reported.
- Table 9 (DCA): Net benefit differences are small (0.2-0.5 × 10⁻²). Are these statistically significant?

### Section 8 (Implementation)
- The clinical use case is illustrative and effective.
- Thai transportability: The recalibration values (BMI→T2DM from 0.68 to 0.85) are stated without justification. How were these derived?

### Section 9 (Discussion)
- Limitation (5) about cross-sectional NHANES is the most critical and should be elevated in prominence.
- The future work list is extensive but lacks prioritization.

### Section 10 (Conclusion)
- Appropriately summarizes contributions without overclaiming (except the rung-3 language).

## Minor Issues
1. **Figure 1** is a placeholder text box—needs an actual architecture diagram before publication.
2. **Table 5** column headers: the third and fourth columns are labeled "NCD-CIE" and "RCT" but the numbers aren't clearly defined (absolute risk reduction? relative?). Add units/description.
3. **Reference [19]** (Naci 2018/2019): The citation year in the bibliography (2019) doesn't match the cite key (naci2018comparative).
4. **Equation numbering**: Eq. 5 (composite score) isn't referenced in the text before its appearance—add a forward reference.
5. **Section 7.4** has two labels (`\label{sec:sensitivity}\label{sec:ablation}`)—split into proper subsections.
6. **"pp" abbreviation** (percentage points) in Tables 4 and the use case—define on first use.
7. **Line spacing** around Algorithm 1 could be tightened to save space.
8. The paper uses both "standardised" and could use "standardized"—pick one spelling convention consistently (LNCS typically uses American English).
9. **Reference completeness**: CausalNex [9] lacks a venue/DOI. Several arXiv references should be updated to published versions if available.
