# Review Feedback for v11 → v12

## Review Summary (v11)

**Verdict:** Rejection with encouragement to resubmit

**Reviewer Assessment:** "This manuscript articulates a coherent and practically relevant architectural blueprint for multi-source RAG in personal knowledge management, with clear design rationales, thoughtful complexity analyses, and an actionable evaluation plan."

**Main Blocker:** "The absence of quantitative evaluation, ablations, or comparisons to strong contemporary baselines precludes assessing the validity and magnitude of the claimed benefits."

---

## Strengths Acknowledged

1. ✅ Integrates three distinct retrieval paradigms (dense, co-occurrence, KG+ontology)
2. ✅ Thoughtful coverage of ontology-enhanced KG traversal
3. ✅ Clear evaluation plan with datasets, metrics, baselines
4. ✅ Well-structured discussion of design alternatives
5. ✅ Targets relevant PKM application domain

---

## Weaknesses Addressed in v12

| Issue | Status | How Addressed |
|-------|--------|---------------|
| No quantitative results | ✅ FIXED | Added HotpotQA experiments (R@10: 0.762) |
| No ablation study | ✅ FIXED | Added ablation table showing each source's contribution |
| Missing related work (CatRAG, GFM-RAG, etc.) | ✅ IN V11 | Already addressed in v11 |
| Score calibration concerns | ✅ IN V11 | Discussed in Design Decisions |
| Co-occurrence scaling | ✅ IN V11 | Sparse PPMI alternatives proposed |

---

## Reviewer Questions (from v11 review)

1. **How are scores calibrated across sources?**
   - Addressed in Design Decisions: z-normalization, learned projections

2. **PPR normalization with weighted edges?**
   - Addressed: Stochastic transition matrix normalization discussed

3. **PPMI smoothing for small corpora?**
   - Addressed: SPPMI, context distribution smoothing proposed

4. **Incremental OWL materialization?**
   - Addressed: Truth maintenance, change sets discussed

5. **Any quantitative sanity checks?**
   - ✅ NOW ADDRESSED: Full HotpotQA experiments in v12

6. **Gating robustness with sparse supervision?**
   - Addressed: Min-per-source inclusion, uncertainty fallbacks

7. **Query-adaptive traversal (CatRAG)?**
   - Addressed: Discussed as future enhancement

8. **PKM-specific evaluation tasks?**
   - Addressed: User-centric task definitions proposed

---

## Key Experimental Results (v12)

### Retrieval Performance (HotpotQA, n=1000)

| Method | R@5 | R@10 | R@20 | Latency |
|--------|-----|------|------|---------|
| Dense | 0.586 | 0.703 | 0.797 | 42.4ms |
| BM25 | 0.503 | 0.625 | 0.742 | 10.4ms |
| Entity | 0.340 | 0.447 | 0.561 | 8.7ms |
| **RRF Fusion** | **0.625** | **0.762** | **0.850** | 67.8ms |

### Ablation Study

| Configuration | R@10 | Δ from Full |
|---------------|------|-------------|
| Full (all sources) | 0.762 | - |
| - Dense | 0.656 | -13.9% |
| - BM25 | 0.727 | -4.6% |
| - Entity | 0.742 | -2.7% |

### Key Findings

1. **RRF fusion +8.5%** over best single source (Dense)
2. **Dense most critical** — 13.9% drop when removed
3. **Each source contributes unique signal** — validates multi-source hypothesis

---

## GitHub Repository

Experiments code: https://github.com/Anirach/rag-second-brain/tree/main/experiments
