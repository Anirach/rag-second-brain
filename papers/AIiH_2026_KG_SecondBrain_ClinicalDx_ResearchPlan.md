# Research Plan: Knowledge Graph Second Brain for Clinical Differential Diagnosis

## AIiH 2026 — Imperial College London, Aug 26–28 | Springer LNCS

---

## 1. Literature Review

### 1.1 LLM Performance in Differential Diagnosis

**[R1]** Kottlors, J., Bratke, G., Rauen, P. et al. "Feasibility of Differential Diagnosis Based on Imaging Patterns Using a Large Language Model." *Radiology*, 2023. DOI: [10.1148/radiol.231167](https://doi.org/10.1148/radiol.231167) (80 citations)
- Demonstrated LLM feasibility for differential diagnosis from imaging patterns. Showed promise but highlighted hallucination risks and lack of structured clinical reasoning.

**[R2]** Bhasuran, B., Jin, Q., Xie, Y. et al. "Preliminary analysis of the impact of lab results on large language model generated differential diagnoses." *npj Digital Medicine*, 2025. DOI: [10.1038/s41746-025-01556-8](https://doi.org/10.1038/s41746-025-01556-8) (19 citations)
- Analyzed how lab data affects LLM differential diagnosis quality. Found LLMs struggle with multi-modal evidence integration — a gap that structured KG can address.

**[R3]** Barrit, S., Torcida, N., Mazeraud, A. et al. "Specialized Large Language Model Outperforms Neurologists at Complex Diagnosis in Blinded Case-Based Evaluation." *Brain Sciences*, 2025. DOI: [10.3390/brainsci15040347](https://doi.org/10.3390/brainsci15040347) (11 citations)
- Specialized LLMs beat neurologists on complex cases. However, relied on fine-tuning rather than external structured knowledge, limiting generalizability.

### 1.2 Knowledge Graph–Augmented LLMs in Clinical Settings

**[R4]** Wang, M., Shen, Y., Zhao, B. et al. "Enhancing LLM-based clinical reasoning in anesthesiology via graph-augmented retrieval and explainable generation." *Health Information Science and Systems*, 2025. DOI: [10.1007/s13755-025-00379-x](https://doi.org/10.1007/s13755-025-00379-x) (1 citation)
- Used graph-augmented retrieval to enhance LLM clinical reasoning. Closest to our approach but uses single-source KG without co-occurrence/sequence graphs or gating mechanism.

**[R5]** He, Y., Chai, Y., Liu, Y. et al. "RSA-KG: A Graph-Based RAG Enhanced AI Knowledge Graph for Recurrent Spontaneous Abortions Diagnosis and Clinical Decision Support." *Medical Data Research*, 2025. DOI: [10.1002/mdr2.70039](https://doi.org/10.1002/mdr2.70039) (2 citations)
- Combined KG with RAG for a specific diagnosis domain. Single disease focus; does not address multi-disease differential diagnosis or multi-source graph fusion.

**[R6]** Das, U., Atmakuri, K.B., Ho, D.H. et al. "Clinical Knowledge Graph Construction and Evaluation with Multi-LLMs via Retrieval-Augmented Generation." *arXiv:2601.01844*, 2026. (1 citation)
- Recent work on LLM-driven KG construction for clinical use. Focuses on KG construction, not on using multi-source KGs as external memory for differential diagnosis.

### 1.3 Medical Knowledge Graphs and Ontology

**[R7]** Alawad, M., Gao, S., Shekar, M.C. et al. "Integration of Domain Knowledge using Medical Knowledge Graph Deep Learning for Cancer Phenotyping." 2021. (12 citations)
- Integrated medical KGs with deep learning for phenotyping. Demonstrated value of co-occurrence patterns in medical KGs but did not combine with LLMs or apply to differential diagnosis.

**[R8]** Xu, Z., Wang, H., Liu, X. "Causal Reasoning Model Based on Medical Knowledge Graph for Disease Diagnosis." *FAIA*, 2021. DOI: [10.3233/faia210279](https://doi.org/10.3233/faia210279) (1 citation)
- Proposed causal reasoning over medical KGs. Relevant for sequence-based patterns in our framework, but pre-LLM era and limited to small KGs.

### 1.4 Research Gap

| Existing Work | Limitation | Our Contribution |
|---|---|---|
| LLM-only diagnosis (R1–R3) | No structured external knowledge; hallucination risk | KG grounds LLM in evidence patterns |
| KG+LLM single-source (R4–R5) | Single KG type; no multi-source fusion | Co-occurrence + sequence + ontology KG with learned gating |
| Medical KG reasoning (R7–R8) | Pre-LLM; no RAG integration | Full Second Brain framework with adaptive retrieval |
| RAG for medical QA | Dense retrieval only; misses structural relationships | Multi-source retrieval (dense + BM25/PPMI + KG) |

**Gap:** No existing work combines multi-source knowledge graphs (co-occurrence, sequence, and ontology) with a learned gating mechanism as a "Second Brain" for LLM-driven differential diagnosis across multiple disease groups.

---

## 2. Research Methodology

### 2.1 Study Design

**Type:** Quantitative experimental study comparing LLM differential diagnosis accuracy with and without multi-source KG augmentation.

**Disease Groups (3):**
1. **Cardiovascular** — chest pain differential (ACS, PE, aortic dissection, pericarditis, GERD)
2. **Respiratory** — dyspnea differential (COPD exacerbation, pneumonia, CHF, asthma, pneumothorax)
3. **Neurological** — headache differential (migraine, tension, SAH, meningitis, temporal arteritis)

### 2.2 Data Sources

| Dataset | Use | Notes |
|---|---|---|
| **DDXPlus** (Fansi et al., 2022) | Primary evaluation dataset | ~1.3M synthetic patient cases with differential diagnoses |
| **MedQA / USMLE-style vignettes** | Secondary validation | Real clinical vignette format |
| **SNOMED-CT / ICD-10 ontology** | KG ontology layer | Standard medical ontologies |
| **PubMed co-occurrences** | Co-occurrence KG construction | Symptom–disease co-occurrence from literature |

### 2.3 KG Second Brain Construction

```
┌─────────────────────────────────────────┐
│          KG Second Brain                │
│                                         │
│  ┌───────────┐ ┌──────────┐ ┌────────┐ │
│  │Co-occurrence│ │ Sequence │ │Ontology│ │
│  │   Graph    │ │  Graph   │ │  Graph │ │
│  │(symptom↔dx)│ │(temporal │ │(SNOMED │ │
│  │   PPMI     │ │ patterns)│ │  /ICD) │ │
│  └─────┬─────┘ └────┬─────┘ └───┬────┘ │
│        └──────┬──────┘───────────┘      │
│          Learned Gating Mechanism       │
│               ↓                         │
│        Fused KG Representation          │
└──────────────┬──────────────────────────┘
               ↓
        ┌──────┴──────┐
        │  LLM Prompt │
        │  (GPT-4 /   │
        │   Llama 3)  │
        └─────────────┘
```

1. **Co-occurrence Graph:** Build symptom–disease PPMI matrix from DDXPlus training data. Top-k co-occurring entities retrieved per query.
2. **Sequence Graph:** Temporal symptom progression patterns (onset order → likely diagnoses). Directed edges encode typical presentation sequences.
3. **Ontology Graph:** SNOMED-CT hierarchy for taxonomic reasoning (e.g., "chest pain" → subtypes). Enables generalization across related concepts.
4. **Gating Mechanism:** Learned attention weights over the three sources, adapting to query type (from existing RAG Second Brain framework).

### 2.4 Experimental Setup

| Condition | Description |
|---|---|
| **B1: LLM-only** | GPT-4o / Llama 3.1 70B with clinical vignette prompt only |
| **B2: LLM + Dense RAG** | LLM + dense embedding retrieval (e5-large) over medical corpus |
| **B3: LLM + BM25** | LLM + BM25 keyword retrieval over medical corpus |
| **P: LLM + KG Second Brain** | LLM + multi-source KG (co-occurrence + sequence + ontology) with gating |

**Ablations:**
| ID | Removed Component | Tests |
|---|---|---|
| A1 | No co-occurrence graph | Value of statistical patterns |
| A2 | No sequence graph | Value of temporal reasoning |
| A3 | No ontology graph | Value of taxonomic structure |
| A4 | No gating (equal weights) | Value of adaptive fusion |

### 2.5 Evaluation Metrics

| Metric | Description |
|---|---|
| **Top-1 Accuracy** | Correct primary diagnosis |
| **Top-3 Accuracy** | Correct diagnosis in top 3 |
| **Top-5 Accuracy** | Correct diagnosis in top 5 |
| **NDCG@5** | Ranking quality of differential list |
| **F1 (macro)** | Per-disease-group F1 |
| **Hallucination Rate** | % of suggested diagnoses not in ground truth differential |

### 2.6 Statistical Analysis

- **McNemar's test** for pairwise accuracy comparisons (B1 vs P, B2 vs P, B3 vs P)
- **Wilcoxon signed-rank test** for NDCG comparisons
- **Bootstrap 95% CI** for all metrics (n=1000 resamples)
- **Effect size** (Cohen's d) for practical significance
- **Per-disease-group analysis** to identify where KG helps most
- Sample: 300 vignettes per disease group (900 total) from DDXPlus test set

---

## 3. Draft Abstract

> **Knowledge Graph as a Second Brain for LLM-Driven Clinical Differential Diagnosis**
>
> Large Language Models (LLMs) show promise in clinical differential diagnosis but remain prone to hallucination and lack structured medical reasoning. We propose a Knowledge Graph (KG) Second Brain framework that augments LLM-based differential diagnosis with multi-source structured knowledge. Our approach constructs three complementary knowledge graphs from clinical data: (1) a co-occurrence graph capturing statistical symptom–disease associations via Positive Pointwise Mutual Information, (2) a sequence graph encoding temporal symptom progression patterns, and (3) an ontology graph derived from SNOMED-CT providing taxonomic medical reasoning. A learned gating mechanism adaptively fuses retrievals from these three sources based on clinical query characteristics, providing the LLM with grounded, multi-perspective evidence for diagnosis generation.
>
> We evaluate our framework against three baselines — LLM-only, LLM with dense retrieval RAG, and LLM with BM25 retrieval — across three disease groups (cardiovascular, respiratory, neurological) using 900 clinical vignettes from the DDXPlus dataset. Results demonstrate that the KG Second Brain significantly improves top-3 diagnostic accuracy and ranking quality (NDCG@5) while reducing hallucinated diagnoses. Ablation studies reveal that each knowledge source contributes complementary information, with the gating mechanism effectively weighting sources based on clinical context. The co-occurrence graph most benefits common presentations, while the sequence and ontology graphs provide advantages for atypical and rare conditions. Our work demonstrates that structured multi-source knowledge graphs serve as an effective external memory — a "Second Brain" — that grounds LLM clinical reasoning in evidence-based patterns.
>
> **Keywords:** Knowledge Graph, Large Language Model, Differential Diagnosis, RAG, Clinical Decision Support, Ontology

*(248 words)*

---

## 4. Timeline (Target: June 2026 Submission)

| Phase | Period | Activities |
|---|---|---|
| **1. KG Construction** | Feb 15 – Mar 31, 2026 | Build co-occurrence, sequence, ontology graphs from DDXPlus + SNOMED-CT. Adapt existing RAG Second Brain codebase. |
| **2. Framework Integration** | Apr 1 – Apr 30, 2026 | Integrate KG retrieval with gating mechanism. Connect to GPT-4o and Llama 3.1 APIs. Build evaluation pipeline. |
| **3. Experiments** | May 1 – May 31, 2026 | Run all baselines + ablations. Statistical analysis. Generate results tables and figures. |
| **4. Paper Writing** | Jun 1 – Jun 20, 2026 | Draft paper in Springer LNCS format (12–15 pages). Internal review. |
| **5. Submission** | Jun 21 – Jun 30, 2026 | Final revisions, formatting check, submit. |

**Key Milestone:** Working KG retrieval prototype by end of March.

---

## 5. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **DDXPlus synthetic data ≠ real clinical cases** | High | Medium | Validate on subset of USMLE-style vignettes; acknowledge limitation; frame as proof-of-concept |
| **API costs for GPT-4o experiments** | Medium | Medium | Use Llama 3.1 70B as primary model (local/API); GPT-4o for comparison only on subset |
| **KG construction complexity** | Medium | High | Leverage existing RAG Second Brain codebase; start with co-occurrence graph (simplest), add others incrementally |
| **Marginal improvement over dense RAG** | Medium | High | Ablation design isolates each component's contribution; even negative results are publishable if analysis is thorough |
| **LNCS page limit (12-15pp)** | Low | Medium | Focus methodology section; move ablation details to appendix/supplementary |
| **Conference deadline change** | Low | Low | Monitor AIiH 2026 website; current timeline has 2-week buffer |
| **Reproducibility concerns** | Medium | Medium | Open-source evaluation pipeline on GitHub (extend rag-second-brain repo); document all prompts and hyperparameters |

---

## References (Verified via Semantic Scholar)

1. Kottlors, J. et al. (2023). Feasibility of Differential Diagnosis Based on Imaging Patterns Using a Large Language Model. *Radiology*. DOI: 10.1148/radiol.231167
2. Bhasuran, B. et al. (2025). Preliminary analysis of the impact of lab results on LLM generated differential diagnoses. *npj Digital Medicine*. DOI: 10.1038/s41746-025-01556-8
3. Barrit, S. et al. (2025). Specialized LLM Outperforms Neurologists at Complex Diagnosis. *Brain Sciences*. DOI: 10.3390/brainsci15040347
4. Wang, M. et al. (2025). Enhancing LLM-based clinical reasoning via graph-augmented retrieval. *Health Info Sci & Sys*. DOI: 10.1007/s13755-025-00379-x
5. He, Y. et al. (2025). RSA-KG: Graph-Based RAG Enhanced KG for Diagnosis and Clinical Decision Support. *Med Data Res*. DOI: 10.1002/mdr2.70039
6. Das, U. et al. (2026). Clinical KG Construction and Evaluation with Multi-LLMs via RAG. *arXiv:2601.01844*
7. Alawad, M. et al. (2021). Integration of Domain Knowledge using Medical KG Deep Learning for Cancer Phenotyping.
8. Xu, Z. et al. (2021). Causal Reasoning Model Based on Medical KG for Disease Diagnosis. *FAIA*. DOI: 10.3233/faia210279
