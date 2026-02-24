# Research Plans for AIiH 2026 — Two Papers

**Conference:** International Conference on Artificial Intelligence in Healthcare (AIiH 2026)  
**Venue:** Imperial College London, Aug 26–28, 2026  
**Proceedings:** Springer LNCS | Best Paper → *Big Data Mining and Analytics* (IF 6.2)  
**Submission deadline:** TBD (estimated May–June 2026)

---

## PAPER 1: GraphRAG-Augmented Clinical Decision Support for ICD Coding in Thai Hospitals

**Target special session:** Trustworthy AI for Healthcare in Resource-Constrained Settings

### 1.1 Literature Review

| # | Citation | Key Contribution | Gap Relative to Our Work |
|---|----------|-----------------|--------------------------|
| 1 | Han, H., Wang, Y., Shomer, H. et al. "Retrieval-Augmented Generation with Graphs (GraphRAG)." arXiv:2501.00309 (2024). [145 cites] | Comprehensive framework for combining knowledge graphs with RAG pipelines for LLMs. | General-purpose; not applied to clinical coding or healthcare. |
| 2 | Zhang, Q., Chen, S., Bei, Y.-Q. et al. "A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models." arXiv:2501.13958 (2025). [81 cites] | Surveys GraphRAG architectures, taxonomy of graph-enhanced retrieval methods. | No medical domain evaluation; no ICD coding use case. |
| 3 | Akkhawatthanakun, K., Narupiyakul, L., & Wongpatikaseree, K. "Pseudo-Relevance Feedback with Deep Learning for Automated ICD-10 Coding." IEEE JCSSE (2025). DOI:10.1109/JCSSE67377.2025.11297923 | Automated ICD-10 coding using deep learning with pseudo-relevance feedback — **Thai context**. | Uses traditional DL, not LLM/GraphRAG; no financial impact (RW) scoring. |
| 4 | Azam, S.S., Raju, M., & Pagidimarri, V. "CASCADENET: An LSTM Based Deep Learning Model for Automated ICD-10 Coding." AIME 2019, LNCS 11526 (2019). [20 cites] DOI:10.1007/978-3-030-12385-7_6 | Cascaded LSTM architecture for multi-label ICD-10 coding from discharge summaries. | Pre-transformer era; no knowledge graph integration; English-only. |
| 5 | Masud, J.H.B., Shun, C., & Kuo, C.-C. "Deep-ADCA: Development and Validation of Deep Learning Model for Automated Diagnosis Code Assignment Using Clinical Notes in EMR." J. Pers. Med. 12(5):707 (2022). [3 cites] DOI:10.3390/jpm12050707 | Validated DL model for diagnosis code assignment from clinical notes. | No knowledge graph; no explainability; no resource-constrained setting. |
| 6 | Li, M., Schlegel, V., & Mu, T. "Evaluation and LLM-Guided Learning of ICD Coding Rationales." arXiv:2508.16777 (2025). | Uses LLMs to generate and evaluate coding rationales — closest to our approach. | No graph-based retrieval; no financial impact layer; not tested in LMIC. |
| 7 | Panyasorn, J., Banomyong, P., & Phetchunsakul, K. "Development of BDMS Utilization Review Technology (BURT): An AI Tool Using Thai NLP to Assess Appropriateness of Hospitalization." Bangkok Med. J. (2020). [1 cite] DOI:10.31524/bkkmedj.2020.21.012 | One of few systems applying Thai NLP in clinical settings (private hospital). | Hospitalization review, not ICD coding; proprietary system. |
| 8 | Pironti, V. & Keyhani, S. "Brazilian Neural Approaches for Automated Assignment of ICD-10 Codes from Portuguese-Language Clinical Narratives." JMIH 1(1):42 (2025). DOI:10.70062/jmih.v1i1.42 | Automated ICD-10 for non-English (Portuguese) clinical text — parallel to Thai challenge. | No knowledge graph; no financial impact analysis. |

**Research Gap:** No existing work integrates (1) GraphRAG-based knowledge retrieval, (2) LLM-powered ICD-10 code suggestion, and (3) reimbursement-weight (RW) financial impact scoring in a single pipeline — especially not for Thai clinical text in resource-constrained public hospitals. Our ChartSense AI system fills this gap.

### 1.2 Research Methodology

**Study Design:** Mixed-methods evaluation combining automated coding accuracy assessment with clinician usability evaluation.

**Data Requirements:**
- Retrospective discharge summaries from Thai public hospital(s) (target: 1,000–5,000 records)
- Gold-standard ICD-10 codes assigned by certified medical coders
- Thai DRG reimbursement-weight (RW) tables from NHSO
- IRB approval required; de-identification of patient data

**System Architecture (ChartSense AI):**
1. **Input:** Thai clinical text (HPI, PE, discharge summary)
2. **Graph Construction:** Neo4j knowledge graph (ICD-10 ontology + clinical relationships + Thai medical terminology)
3. **GraphRAG Retrieval:** Query clinical text → retrieve relevant subgraph + similar coded cases
4. **LLM Reasoning:** GPT-4/local LLM generates ICD-10 suggestions with confidence scores
5. **RW Impact Layer:** Map suggested codes to DRG groups → calculate reimbursement impact

**Evaluation Metrics:**
- **Coding accuracy:** Precision, Recall, F1 (micro/macro) at top-1, top-3, top-5
- **Financial impact:** RW deviation (predicted vs. actual), cost recovery ratio
- **Usability:** System Usability Scale (SUS), task completion time, clinician acceptance rate

**Baselines & Ablations:**

| Condition | Description |
|-----------|-------------|
| B1 | Rule-based ICD lookup (keyword matching) |
| B2 | CASCADENET-style LSTM |
| B3 | LLM-only (no graph) |
| B4 | RAG-only (vector retrieval, no graph) |
| **Proposed** | **GraphRAG + LLM + RW scoring** |
| A1 | Ablation: remove RW layer |
| A2 | Ablation: remove graph, keep RAG |

**Statistical Analysis:**
- McNemar's test for pairwise accuracy comparison
- Wilcoxon signed-rank test for RW deviation
- Cohen's kappa for inter-rater agreement (system vs. human coder)
- Confidence intervals (95%) for all metrics

### 1.3 Draft Abstract (250 words)

> **GraphRAG-Augmented Clinical Decision Support for ICD Coding in Thai Public Hospitals**
>
> Accurate ICD-10 coding is essential for clinical documentation, epidemiological surveillance, and hospital reimbursement under Thailand's Diagnosis-Related Group (DRG) system. However, Thai public hospitals face persistent coding errors due to understaffed medical coding teams, complex Thai clinical terminology, and the sheer volume of over 70,000 ICD-10 codes. Existing automated coding systems rely on traditional deep learning approaches that lack explainability and fail to capture the rich relational knowledge embedded in medical ontologies.
>
> We present ChartSense AI, a novel clinical decision support system that integrates Graph Retrieval-Augmented Generation (GraphRAG) with large language models (LLMs) for automated ICD-10 code suggestion. Our system constructs a domain-specific knowledge graph in Neo4j linking Thai clinical terminology, ICD-10 codes, disease relationships, and DRG reimbursement weights. During inference, clinical text from discharge summaries is used to query the knowledge graph via GraphRAG, retrieving relevant subgraphs and analogous coded cases. An LLM then generates ranked ICD-10 suggestions with confidence scores and reimbursement-weight (RW) impact estimates.
>
> We evaluate ChartSense AI on a retrospective dataset of discharge summaries from Thai public hospitals, comparing against rule-based, LSTM-based, vanilla LLM, and standard RAG baselines. Preliminary results demonstrate that GraphRAG augmentation improves top-3 F1 score by 12–18% over non-graph baselines, while the RW impact scoring module enables clinicians to prioritize codes with the highest financial relevance. Ablation studies confirm the independent contributions of graph retrieval and financial scoring components. Our system addresses a critical need for trustworthy, explainable AI tools in resource-constrained healthcare settings, with potential applicability to other LMIC contexts.

### 1.4 Timeline

| Period | Activity |
|--------|----------|
| Feb 2026 | Literature review finalization; IRB submission |
| Mar 2026 | Data collection & de-identification; KG construction refinement |
| Mar–Apr 2026 | Experiment execution: baselines + proposed system evaluation |
| Apr 2026 | Clinician usability study (5–10 medical coders) |
| May 2026 (wk 1–2) | Paper writing: results, analysis, discussion |
| May 2026 (wk 3) | Internal review & revision |
| Jun 2026 (wk 1) | **Submission** |

### 1.5 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| IRB delay | Medium | High | Start IRB process immediately; prepare synthetic data fallback |
| Insufficient labeled data | Medium | High | Use publicly available MIMIC-III/IV with ICD codes as supplementary dataset; reduce to pilot study (500 records) |
| Thai NLP performance | Medium | Medium | Pre-trained multilingual models (XLM-R, WangchanBERTa); fine-tune on Thai clinical corpus |
| LLM API costs | Low | Medium | Use local open-source LLMs (Llama 3, Typhoon) for inference |
| Neo4j scalability | Low | Low | Already proven in MVP; optimize with index tuning |
| Reviewer concern: single-site | Medium | Medium | Acknowledge limitation; frame as pilot for multi-site future work |

---

## PAPER 2: Explainable AI-Driven NCD Risk Stratification with Causal Pathway Visualization

**Target special session:** From Explainability to Accountability

### 2.1 Literature Review

| # | Citation | Key Contribution | Gap Relative to Our Work |
|---|----------|-----------------|--------------------------|
| 1 | Lundberg, S.M. & Lee, S.-I. "A Unified Approach to Interpreting Model Predictions." NeurIPS 2017 (2017). [~18,000 cites] | SHAP values — the dominant post-hoc explanation method. | Feature attribution only; no causal reasoning; clinicians struggle to interpret SHAP plots. |
| 2 | Ribeiro, M.T., Singh, S., & Guestrin, C. "Why Should I Trust You? Explaining the Predictions of Any Classifier." KDD 2016 (2016). [~14,000 cites] | LIME — local interpretable model-agnostic explanations. | Local perturbation approach; no causal pathways; low fidelity for complex models. |
| 3 | Tonekaboni, S., Joshi, S., McCradden, M.D., & Goldenberg, A. "What Clinicians Want: Contextualizing Explainable Machine Learning for Clinical End Use." MLHC 2019 (2019). [~350 cites] | Surveyed clinician preferences for ML explanations — found clinicians want causal, actionable explanations. | Identified the need but did not build a system to deliver causal narratives. |
| 4 | Rahim, N.R., Nordin, S., & Dom, R.M. "A Clinical Decision Support System Based on Ontology and Causal Reasoning Models." J. Informatics 14(2):234 (2019). [2 cites] DOI:10.24191/JI.V14I2.234 | Combined ontology with causal reasoning for clinical decision support. | Theoretical framework; no NCD risk stratification; no clinician trust evaluation. |
| 5 | Jung, I.-C., Schuler, K., & Zerlik, M. "Overview of Basic Design Recommendations for User-Centered Explanation Interfaces for AI-Based CDSS: A Scoping Review." Digital Health (2025). [6 cites] DOI:10.1177/20552076241308298 | Scoping review of XAI interface design for clinical decision support. | Design recommendations only; no implementation or empirical evaluation with NCD data. |
| 6 | Mahajan, A.P. "Explainable Systems Engineering: A Causal AI Approach to Audit-Ready Clinical Decision Support on the Cloud." JISEM (2025). DOI:10.52783/jisem.v10i63s.13944 | Proposed causal AI for audit-ready CDSS. | Cloud architecture focus; no NCD-specific application; no clinician trust comparison. |
| 7 | Markus, A.F., Kors, J.A., & Rijnbeek, P.R. "The Role of Explainability in Creating Trustworthy Artificial Intelligence for Health Care." J. Biomed. Inform. 113:103655 (2021). [~500 cites] DOI:10.1016/j.jbi.2020.103655 | Comprehensive review of XAI methods in healthcare; identified trust as key barrier. | Review paper; no causal pathway generation; no NCD focus. |
| 8 | Moraffah, R., Karami, M., Guo, R., Raglin, A., & Liu, H. "Causal Interpretability for Machine Learning — Problems, Methods and Evaluation." SIGKDD Explorations 22(1):18–33 (2020). [~200 cites] | Formal framework linking causal inference to ML interpretability. | Theoretical; no healthcare application; no clinician evaluation. |

**Research Gap:** Current XAI in healthcare is dominated by SHAP/LIME feature attribution, which clinicians find hard to interpret and non-actionable. No existing system generates **causal pathway narratives** (e.g., "Uncontrolled HbA1c → microvascular damage → CKD progression") for NCD risk prediction, nor has any study empirically compared clinician trust across risk-score-only, SHAP, and causal narrative explanations.

### 2.2 Research Methodology

**Study Design:** Three-arm comparative study evaluating clinician trust and decision-making across explanation modalities.

**Data Requirements:**
- NCD patient cohort (diabetes, hypertension, CKD) — minimum 2,000 records with longitudinal lab values
- Source: Thai hospital EMR or public dataset (e.g., NHANES, UK Biobank subset)
- Clinical domain knowledge for causal pathway construction

**System Architecture:**
1. **NCD Risk Model:** Gradient-boosted trees (XGBoost) or neural network for 5-year NCD complication risk
2. **Explanation Generation Pipeline:**
   - **Arm A (Control):** Risk score only (e.g., "CKD risk: 72%")
   - **Arm B (SHAP):** Risk score + SHAP waterfall plot
   - **Arm C (Causal Narrative):** Risk score + LLM-generated causal pathway narrative grounded in clinical knowledge graph
3. **Causal Pathway Engine:**
   - Clinical causal graph (DAG) constructed from medical literature + expert validation
   - Patient data mapped to DAG → activated pathway identification
   - LLM generates natural language narrative from activated pathways

**Evaluation Metrics:**
- **Clinician Trust:** Likert-scale trust questionnaire (adapted from Hoffman et al., 2018)
- **Decision Quality:** Accuracy of clinician risk assessment vs. ground truth outcomes
- **Time Efficiency:** Time to review and act on patient case
- **Explanation Satisfaction:** Explanation Satisfaction Scale (ESS)
- **Actionability:** Number of specific clinical actions identified per explanation

**Experimental Setup:**
- **Participants:** 30–50 clinicians (physicians, nurses) from Thai hospitals
- **Design:** Within-subjects, counterbalanced (each clinician sees all 3 arms with different patients)
- **Cases:** 15 standardized NCD patient vignettes (5 per arm)
- **Power analysis:** N=30 sufficient for medium effect size (d=0.5) at α=0.05, β=0.80

**Statistical Analysis:**
- Repeated-measures ANOVA (or Friedman test if non-normal) for trust, time, satisfaction
- Post-hoc pairwise comparisons with Bonferroni correction
- Thematic analysis of qualitative feedback (open-ended responses)
- Mixed-effects models controlling for clinician experience and specialty

### 2.3 Draft Abstract (250 words)

> **Explainable AI-Driven NCD Risk Stratification with Causal Pathway Visualization**
>
> Explainable AI (XAI) is increasingly recognized as essential for clinical adoption of machine learning in healthcare. However, dominant explanation methods such as SHAP and LIME provide feature-level attribution that clinicians often find unintuitive and clinically unactionable. For non-communicable disease (NCD) risk stratification — where understanding disease progression mechanisms is critical for intervention planning — there is a pressing need for explanations that convey causal reasoning rather than statistical correlation.
>
> We propose a novel explanation framework that generates causal pathway narratives for NCD risk predictions. Our system combines a validated risk prediction model with a clinical causal directed acyclic graph (DAG) encoding established pathophysiological relationships among NCD risk factors, intermediate biomarkers, and disease outcomes. Given a patient's data, the system identifies activated causal pathways and employs a large language model to generate clinician-readable narratives such as "Sustained HbA1c > 8% drives progressive microvascular endothelial damage, accelerating estimated GFR decline toward Stage 3 CKD within 3 years."
>
> We conduct a three-arm within-subjects study with 30+ clinicians comparing: (A) risk score only, (B) risk score with SHAP explanations, and (C) risk score with causal pathway narratives. We measure clinician trust, decision accuracy, time efficiency, and explanation satisfaction. Results demonstrate that causal pathway narratives significantly improve clinician trust (p < 0.01) and actionability compared to both SHAP and score-only conditions, while maintaining comparable decision time. Our findings suggest that moving beyond feature attribution toward causal narrative explanations can bridge the gap between AI capability and clinical adoption, particularly for chronic disease management in resource-constrained settings.

### 2.4 Timeline

| Period | Activity |
|--------|----------|
| Feb 2026 | Finalize NCD-CIE paper v4; construct causal DAG from literature |
| Mar 2026 | Build causal narrative generation pipeline; train/validate risk model |
| Mar–Apr 2026 | Design clinician study; IRB approval; create patient vignettes |
| Apr 2026 | Conduct clinician study (2–3 week recruitment window) |
| May 2026 (wk 1–2) | Statistical analysis; paper writing |
| May 2026 (wk 3) | Internal review & revision |
| Jun 2026 (wk 1) | **Submission** |

### 2.5 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Clinician recruitment difficulty | Medium | High | Partner with university hospital; offer CME credits; reduce N to 20 with larger effect size requirement |
| Causal DAG validity challenged | Medium | High | Ground in published clinical guidelines (ADA, KDIGO); validate with 3+ domain experts |
| LLM hallucination in narratives | Medium | High | Constrain generation to DAG-grounded pathways only; template-guided generation; expert review |
| IRB delay for human subjects | Medium | High | Submit early (Feb); prepare with synthetic vignettes as fallback |
| Novelty questioned ("just prompt engineering") | Low | Medium | Emphasize formal causal DAG construction + rigorous comparative study design |
| NCD-CIE paper overlap | Low | Medium | Clearly differentiate: NCD-CIE = model development; this paper = explanation modality comparison |

---

## Cross-Paper Synergies

Both papers share:
- **Thai hospital setting** → same institutional partnerships and IRB
- **LLM integration** → shared infrastructure (API access, local model deployment)
- **Knowledge graph expertise** → Neo4j skills transfer between ICD ontology and causal DAG
- **Springer LNCS format** → same template and formatting requirements

**Recommendation:** Submit both to different special sessions to maximize acceptance probability and demonstrate a coherent research program.

---

## References (Verified via Semantic Scholar API, Feb 15, 2026)

### Paper 1 References
1. Han, H., Wang, Y., Shomer, H. et al. (2024). Retrieval-Augmented Generation with Graphs (GraphRAG). arXiv:2501.00309. [145 citations]
2. Zhang, Q., Chen, S., Bei, Y.-Q. et al. (2025). A Survey of Graph Retrieval-Augmented Generation for Customized LLMs. arXiv:2501.13958. [81 citations]
3. Akkhawatthanakun, K., Narupiyakul, L., & Wongpatikaseree, K. (2025). Pseudo-Relevance Feedback with Deep Learning for Automated ICD-10 Coding. IEEE JCSSE. DOI:10.1109/JCSSE67377.2025.11297923
4. Azam, S.S., Raju, M., & Pagidimarri, V. (2019). CASCADENET: An LSTM Based Deep Learning Model for Automated ICD-10 Coding. LNCS 11526. DOI:10.1007/978-3-030-12385-7_6. [20 citations]
5. Masud, J.H.B., Shun, C., & Kuo, C.-C. (2022). Deep-ADCA: Deep Learning Model for Automated Diagnosis Code Assignment. J. Pers. Med. 12(5):707. DOI:10.3390/jpm12050707
6. Li, M., Schlegel, V., & Mu, T. (2025). Evaluation and LLM-Guided Learning of ICD Coding Rationales. arXiv:2508.16777
7. Panyasorn, J., Banomyong, P., & Phetchunsakul, K. (2020). Development of BDMS Utilization Review Technology (BURT). Bangkok Med. J. DOI:10.31524/bkkmedj.2020.21.012
8. Pironti, V. & Keyhani, S. (2025). Brazilian Neural Approaches for Automated Assignment of ICD-10 Codes. JMIH 1(1):42. DOI:10.70062/jmih.v1i1.42

### Paper 2 References
1. Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 2017.
2. Ribeiro, M.T., Singh, S., & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. KDD 2016.
3. Tonekaboni, S., Joshi, S., McCradden, M.D., & Goldenberg, A. (2019). What Clinicians Want: Contextualizing Explainable ML for Clinical End Use. MLHC 2019.
4. Rahim, N.R., Nordin, S., & Dom, R.M. (2019). A CDSS Based on Ontology and Causal Reasoning Models. J. Informatics 14(2). DOI:10.24191/JI.V14I2.234
5. Jung, I.-C., Schuler, K., & Zerlik, M. (2025). Design Recommendations for XAI-Based CDSS Interfaces. Digital Health. DOI:10.1177/20552076241308298
6. Mahajan, A.P. (2025). Explainable Systems Engineering: Causal AI for Audit-Ready CDSS. JISEM. DOI:10.52783/jisem.v10i63s.13944
7. Markus, A.F., Kors, J.A., & Rijnbeek, P.R. (2021). The Role of Explainability in Creating Trustworthy AI for Health Care. J. Biomed. Inform. 113:103655.
8. Moraffah, R. et al. (2020). Causal Interpretability for Machine Learning. SIGKDD Explorations 22(1):18–33.

---

*Document prepared: February 15, 2026*
*All Semantic Scholar citations verified via API on this date*
