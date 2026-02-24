# PEER REVIEW REPORT

## Journal: AI in Medicine - Knowledge Graphs and Multi-Omics Interpretation

**Manuscript Title:** A Hybrid Causal Neuro-Symbolic Framework for Integrated Non-Communicable Disease Risk Assessment and Prevention: The NCD Causal Insight Engine with Target Trial Emulation Validation Roadmap

**Authors:** Anirach Mingkhwan and Kongkiat Kespechara

**Reviewer:** Editorial Board, Knowledge Graph and Multi-Omics Section

**Date:** January 26, 2026

**Recommendation:** ⚠️ **MAJOR REVISION REQUIRED**

---

## EXECUTIVE SUMMARY

This manuscript presents the NCD Causal Insight Engine (NCD-CIE), a platform combining knowledge graphs with statistical methods for non-communicable disease risk assessment. The conceptual framework is promising, and the authors demonstrate transparency about limitations. However, several critical issues require substantial revision before publication consideration.

**Major Concerns:**
1. The "neuro-symbolic" terminology is misleading—no neural network components exist
2. Single-patient (N=1) validation is insufficient to support the claims made
3. Multi-omics integration is absent despite the platform's potential
4. Risk algorithms lack external validation and statistical rigor
5. Causal claims exceed what the evidence supports

---

## 1. CONCEPT REVIEW

### 1.1 Strengths

- **Novel integration approach:** Combining symbolic AI (knowledge graphs, DAGs) with statistical pattern recognition addresses the interpretability-performance tradeoff appropriately framed in the literature (Rudin, 2019; Ghassemi et al., 2021).

- **Evidence grading system:** The three-tier classification (RCT+MR, MR-only, mechanistic) provides transparency about causal certainty levels.

- **Target trial framework:** Adopting Hernán et al.'s (2025) framework for validation planning demonstrates methodological sophistication.

### 1.2 Critical Issues Requiring Revision

#### Issue 1.2.1: Misleading "Neuro-Symbolic" Terminology

**Problem:** The manuscript claims to employ "hybrid neuro-symbolic AI architecture" (Section 2.1), but examination reveals no neural network components. The "sub-symbolic" layer consists only of:
- Linear regression for trend detection
- Coefficient of variation calculations
- Pearson correlation
- Rule-based clustering

**These are classical statistical methods, not neural approaches.**

The term "neuro-symbolic AI" in the literature specifically refers to integration of neural networks with symbolic reasoning (Garcez & Lamb, 2023; Kautz, 2022). Using this terminology without neural components is misleading.

**Recommendation:** Either:
(a) Replace "neuro-symbolic" with "hybrid symbolic-statistical" throughout the manuscript, OR
(b) Implement actual neural components (e.g., graph neural networks for relationship prediction, transformer-based biomarker embedding)

**Reference:** Lamb, L.C., et al. (2020). Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective. *IJCAI*, 4877-4884.

#### Issue 1.2.2: Absence of Multi-Omics Integration

**Problem:** For a platform targeting "precision prevention," the complete absence of multi-omics data integration (genomics, transcriptomics, proteomics, metabolomics) is a significant limitation not adequately addressed.

The knowledge graph contains only clinical biomarkers. Modern precision medicine requires integration of:
- Polygenic risk scores (PRS) for genetic predisposition
- Transcriptomic signatures for pathway activation status
- Proteomic markers for early disease detection
- Metabolomic profiles for metabolic phenotyping

**Recommendation:** 
- Add a dedicated section discussing multi-omics integration as a planned enhancement
- Consider incorporating existing multi-omics knowledge graphs (e.g., Hetionet, PrimeKG) as foundational resources
- Discuss how the platform architecture could accommodate omics data layers

**References:**
- Chandak, P., et al. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10, 67.
- Himmelstein, D.S., et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726.

#### Issue 1.2.3: Static Knowledge Base Limitations

**Problem:** The knowledge graph contains "70+ biomarkers and 100+ causal edges" derived from literature synthesis. However:
- No automated update mechanism is described
- Evidence evolves—new MR studies may contradict existing relationships
- The manual curation process lacks systematic review methodology (PRISMA)

**Recommendation:**
- Describe the update protocol for the knowledge base
- Consider semi-automated literature mining approaches (PubMed embeddings, knowledge extraction)
- Provide the date of last literature search and planned update frequency

---

## 2. METHODOLOGY REVIEW

### 2.1 Strengths

- **DAG-based causal modeling:** Using directed acyclic graphs for causal relationship representation is methodologically sound (Pearl, 2009; Textor et al., 2016).

- **Explicit mathematical specification:** The formal risk score algorithm (Section 2.5) provides reproducibility.

- **Acknowledgment of acyclic limitation:** The "time-sliced" interpretation of DAGs (Section 2.2.1) shows awareness of physiological feedback loop challenges.

### 2.2 Critical Issues Requiring Revision

#### Issue 2.2.1: Arbitrary Algorithm Parameters

**Problem:** The risk score algorithm contains unexplained constants:
- β₀ = 10 (base risk)
- β₁ = 0.3 (deviation coefficient)
- Protective factor coefficients: 0.5, 0.2

**No justification is provided for these specific values.** How were they derived? Are they empirically calibrated or heuristically chosen?

**Recommendation:**
- Explain the derivation of all algorithm parameters
- Report sensitivity analyses varying these parameters (±20%, ±50%)
- If empirically derived, describe the training/calibration dataset
- If heuristically chosen, acknowledge this as a limitation requiring future calibration

#### Issue 2.2.2: Confidence Interval Generation

**Problem:** Table 5 reports 95% confidence intervals for risk scores (e.g., CVD 15.2%, 95% CI: 12.5%-17.9%), but the methodology for CI calculation is not described.

Given that the risk scores are calculated deterministically from biomarker values, how are uncertainty bounds generated? The manuscript mentions confidence intervals are "heuristic rather than probabilistically calibrated" (Section 4.5, Limitation Fourth), but this admission appears only in limitations, not in the methods.

**Recommendation:**
- Describe the CI calculation method in Section 2.5
- If using bootstrap, specify the number of iterations and resampling strategy
- If using propagation of biomarker measurement uncertainty, specify the assumed error distributions
- Consider removing CIs if they cannot be methodologically justified

#### Issue 2.2.3: Causal Effect Size Derivation

**Problem:** The what-if simulator uses effect sizes from meta-analyses (Section 2.7), but:
- Table 2 reports correlation coefficients (r values), not effect sizes
- The translation from "r = 0.75" to actionable intervention effects is unexplained
- Correlation strength ≠ intervention effect magnitude

**Recommendation:**
- Clarify the distinction between observational correlations and intervention effect sizes
- For intervention simulation, use only effect sizes from RCTs/interventional studies
- Provide supplementary tables listing all effect sizes with source citations

#### Issue 2.2.4: DAG Structural Assumptions

**Problem:** The DAG structure is presented as derived from evidence synthesis, but:
- No sensitivity analysis for structural misspecification
- Alternative DAG structures (with different adjustment sets) are not considered
- No discussion of potential unmeasured confounders in the graph

**Recommendation:**
- Present alternative DAG structures considered and rejected (with rationale)
- Use tools like DAGitty to identify minimal sufficient adjustment sets
- Discuss key unmeasured confounders that could invalidate causal estimates (e.g., genetic factors, socioeconomic status, diet quality)

**Reference:** Textor, J., et al. (2016). Robust causal inference using directed acyclic graphs: the R package 'dagitty'. *International Journal of Epidemiology*, 45(6), 1887-1894.

---

## 3. SINGLE PATIENT DATA USAGE REVIEW

### 3.1 Critical Issues

#### Issue 3.1.1: N=1 Insufficient for Validation Claims

**Problem:** The manuscript makes validation claims based on a single patient:

> "The close alignment between predicted and observed biomarker changes supports the validity of the causal model structure" (Abstract)

> "mean absolute difference: 3.2%" (Results)

**This is fundamentally inappropriate.** A single case cannot:
- Establish predictive accuracy
- Demonstrate generalizability
- Rule out confounding
- Support claims about "validity"

The patient could be an outlier. The alignment could be coincidental. The intervention effects could be confounded by unmeasured factors (medication changes, dietary changes, seasonal effects, regression to the mean).

**Recommendation:**
- Remove all language suggesting the case study "validates" or "supports validity" of the model
- Explicitly label this as a "feasibility demonstration" or "illustrative example"
- Add statistical context: report the probability of observing 3.2% MAD by chance under various null hypotheses
- Discuss regression to the mean as an alternative explanation

#### Issue 3.1.2: Selection Bias Concerns

**Problem:** The patient selection criteria are not specified:
- Why was this particular patient chosen?
- Were other patients assessed but not reported?
- What is the publication bias risk?

If multiple patients were screened and only the best-performing case is presented, this constitutes selective reporting.

**Recommendation:**
- Describe the patient selection process explicitly
- Report all patients assessed during platform development (if any)
- If this is truly the only patient assessed, explain why

#### Issue 3.1.3: Intervention Attribution

**Problem:** The patient received "lifestyle intervention recommendations generated by the platform" (Section 2.8), but:
- The specific intervention is not described
- Compliance is not measured
- Concomitant medications are not reported
- Other lifestyle changes are not controlled

The observed improvements cannot be causally attributed to the platform-guided intervention versus natural variation, Hawthorne effect, or concurrent treatments.

**Recommendation:**
- Describe the intervention protocol in detail (diet, exercise, specific targets)
- Report medication changes during follow-up
- Measure and report intervention compliance
- Acknowledge that causal attribution is impossible with N=1

#### Issue 3.1.4: Follow-up Timing

**Problem:** 5-month follow-up is insufficient for validating 5-year and 10-year risk predictions.

The "validation" compares predicted biomarker changes to observed changes, not predicted disease risk to actual disease occurrence. These are different constructs.

**Recommendation:**
- Clarify that biomarker change prediction ≠ disease risk prediction validation
- Acknowledge that true risk prediction validation requires long-term outcome follow-up
- Remove "validation" language; use "preliminary assessment" or "proof-of-concept"

---

## 4. TECHNICAL METHODS REVIEW

### 4.1 Strengths

- **Comprehensive biomarker coverage:** 70+ biomarkers across 8 domains provides broad health assessment.
- **Pattern detection variety:** Five detection types (trend, volatility, correlation, cluster, threshold) cover common clinical scenarios.

### 4.2 Critical Issues Requiring Revision

#### Issue 4.2.1: "AI-Powered" Overclaim

**Problem:** The manuscript repeatedly uses "AI-powered" and "AI Insights Engine" terminology (Sections 2.6, 3.4), but the actual methods are:
- Linear regression (trend detection)
- Standard deviation calculation (volatility)
- Pearson correlation (correlation detection)
- Rule-based IF-THEN logic (cluster analysis)
- Threshold comparison (critical values)

**None of these constitute artificial intelligence by contemporary definitions.** These are basic statistical methods available in Excel.

**Recommendation:**
- Replace "AI-powered" with "automated analytics" or "computational"
- Reserve "AI" terminology for methods involving machine learning, deep learning, or knowledge reasoning beyond rule application
- Alternatively, implement actual ML methods (random forests for risk, GNNs for graph reasoning)

#### Issue 4.2.2: Metabolic Syndrome Detection

**Problem:** The pattern detection reports "Metabolic Syndrome Cluster pattern with 5/5 criteria met, classified as severity 'alert' with confidence 0.85" (Section 3.4).

However:
- Metabolic syndrome diagnosis is deterministic (ATP-III criteria)—it does not have "confidence"
- If 5/5 criteria are met, confidence should be 1.0 (certain)
- The 0.85 figure is unexplained

**Recommendation:**
- Explain the confidence calculation methodology
- If confidence relates to measurement uncertainty, specify the error model
- If heuristic, acknowledge and justify

#### Issue 4.2.3: Risk Score Bounds and Calibration

**Problem:** Risk scores are "bounded to [0, 50]" (Section 2.5.3), but:
- CVD 10-year risk in Table 5 shows 22.4% and 15.2%—these are percentage risks, not bounded scores
- The relationship between the bounded [0,50] score and percentage risk is unclear
- No calibration against actual event rates is provided

**Recommendation:**
- Clarify the transformation from bounded score to percentage risk
- Provide calibration plots comparing predicted vs. observed event rates (requires larger dataset)
- Reference established risk scores (Framingham, SCORE2) for calibration comparison

#### Issue 4.2.4: Missing Technical Details

**Problem:** Several technical aspects lack specification:
- Neo4j query complexity and performance
- Computational requirements
- Response time for risk calculations
- Data privacy and security measures

**Recommendation:**
- Add a Technical Implementation section with system requirements
- Report computational performance metrics
- Describe data security and privacy compliance (HIPAA/GDPR)

---

## 5. CONCLUSION REVIEW

### 5.1 Current Conclusion Assessment

The conclusion states:

> "Our proof-of-concept case study demonstrates the platform's ability to identify upstream intervention targets and predict cascade effects that closely matched observed outcomes (mean absolute difference: 3.2%)."

**This overreaches the evidence.** A single case with 3.2% MAD cannot "demonstrate" prediction ability—it illustrates feasibility at best.

### 5.2 Recommended Revision

Revise conclusions to accurately reflect evidence strength:

**Current (overclaimed):**
> "The close alignment between predicted and observed biomarker changes supports the validity of the causal model structure."

**Suggested revision:**
> "The observed alignment between predicted and observed biomarker changes in this single case is consistent with the causal model structure, though validation in larger cohorts is required before any conclusions about model validity can be drawn."

**Current (overclaimed):**
> "The primary contribution at this stage is the conceptual framework, technical architecture, and validation methodology demonstrating how hybrid neuro-symbolic AI can be integrated..."

**Suggested revision:**
> "The primary contribution is the conceptual framework and technical architecture illustrating how hybrid symbolic-statistical approaches can be integrated into digital health platforms, along with a proposed validation methodology for future evaluation."

### 5.3 Missing Elements in Conclusion

The conclusion should address:
- Explicit acknowledgment that N=1 precludes any validity claims
- Timeline for planned validation studies
- Regulatory considerations for clinical deployment
- Comparison with existing validated tools (Framingham, SCORE2)—how does NCD-CIE add value?

---

## 6. ADDITIONAL RECOMMENDATIONS

### 6.1 Statistical Reporting

- Report all statistical tests with effect sizes and p-values
- Use appropriate methods for small samples (exact tests, bootstrapping)
- Pre-register the validation study protocol (OSF, PROSPERO)

### 6.2 Reproducibility

- Deposit code in a public repository (GitHub, Zenodo)
- Provide the complete causal edge list as supplementary material
- Share anonymized patient data or synthetic data for replication

### 6.3 TRIPOD+AI Compliance

The manuscript references TRIPOD+AI (Collins et al., 2024) but does not follow the checklist. Complete the TRIPOD+AI checklist and include as supplementary material.

### 6.4 Knowledge Graph Enhancement

Consider integration with established biomedical knowledge graphs:
- **PrimeKG:** 4 million relationships covering diseases, drugs, genes
- **Hetionet:** 2.25 million edges integrating 29 data sources
- **SPOKE:** Scalable Precision Medicine Open Knowledge Engine

**Reference:** Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10, 67.

---

## 7. SUMMARY OF REQUIRED REVISIONS

### Major Revisions Required:

| # | Issue | Section | Priority |
|---|-------|---------|----------|
| 1 | Replace "neuro-symbolic" with accurate terminology | Throughout | Critical |
| 2 | Remove validation claims based on N=1 | Abstract, Results, Discussion, Conclusion | Critical |
| 3 | Justify algorithm parameters (β₀, β₁, weights) | Methods 2.5 | High |
| 4 | Explain confidence interval methodology | Methods 2.5, Results | High |
| 5 | Describe patient selection and intervention details | Methods 2.8 | High |
| 6 | Replace "AI-powered" with accurate terminology | Methods 2.6 | Moderate |
| 7 | Add multi-omics integration discussion | Discussion | Moderate |
| 8 | Provide DAG sensitivity analysis | Methods 2.2 | Moderate |
| 9 | Address metabolic syndrome confidence calculation | Results 3.4 | Minor |
| 10 | Add technical implementation details | Methods | Minor |

### Minor Revisions Required:

- Complete TRIPOD+AI checklist (Supplementary)
- Deposit code in public repository
- Add system requirements and performance metrics
- Pre-register validation protocol

---

## 8. REFERENCES FOR THIS REVIEW

1. Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. *Scientific Data*, 10, 67.

2. Collins, G.S., et al. (2024). TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods. *BMJ*, 385, e078378.

3. Garcez, A., & Lamb, L.C. (2023). Neurosymbolic AI: The 3rd Wave. *Artificial Intelligence Review*, 56, 12387-12406.

4. Ghassemi, M., et al. (2021). The false hope of current approaches to explainable artificial intelligence in health care. *Lancet Digital Health*, 3(11), e745-e750.

5. Hernán, M.A., Dahabreh, I.J., Dickerman, B.A., & Swanson, S.A. (2025). The Target Trial Framework for Causal Inference From Observational Data. *Annals of Internal Medicine*, 178, 402-407.

6. Himmelstein, D.S., et al. (2017). Systematic integration of biomedical knowledge prioritizes drugs for repurposing. *eLife*, 6, e26726.

7. Kautz, H. (2022). The third AI summer. *AI Magazine*, 43, 105-125.

8. Lamb, L.C., et al. (2020). Graph Neural Networks Meet Neural-Symbolic Computing: A Survey and Perspective. *IJCAI*, 4877-4884.

9. Pearl, J. (2009). Causality: Models, Reasoning, and Inference. 2nd ed. Cambridge University Press.

10. Rudin, C. (2019). Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. *Nature Machine Intelligence*, 1, 206-215.

11. Textor, J., et al. (2016). Robust causal inference using directed acyclic graphs: the R package 'dagitty'. *International Journal of Epidemiology*, 45(6), 1887-1894.

12. van der Bles, A.M., et al. (2019). Communicating uncertainty about facts, numbers and science. *Royal Society Open Science*, 6(5), 181870.

13. Weld, D.S., & Bansal, G. (2019). The challenge of crafting intelligible intelligence. *Communications of the ACM*, 62(6), 70-79.

---

**Reviewer Signature:** Editorial Board, Knowledge Graph and Multi-Omics Section

**Date:** January 26, 2026

**Recommendation:** Major Revision Required

---

*This review was conducted following ICMJE and COPE guidelines for peer review. The reviewer declares no conflicts of interest.*
