# Positive Feedback Registry

**Paper:** Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM

---

## Round 1 (v3) — Score: 5.2/10

### Reviewer Praise
- "This work addresses an important problem in RAG—adapting retrieval to query type by fusing complementary signals"
- "Demonstrates promising results with a thoughtful ontology component"
- "Clean two-stage training decomposition"
- "Empirical gains on HotpotQA and other benchmarks are effective"
- "Ablations suggest the approach is effective"

### Strengths Identified
- [x] Important problem addressed (adaptive retrieval)
- [x] Thoughtful ontology component
- [x] Two-stage training decomposition
- [x] Multi-benchmark empirical validation
- [x] Ablation studies present

---

## Round 2 (v6) — Score: 5.6/10

### Issues Resolved (Now Locked)
- Causal LM loss properly defined (Eq. 15-16)
- Materialization ablation added (+2.9 EM)
- RePlug/Atlas comparisons added
- Open-source reasoner alternatives documented

### Cumulative Locked Strengths
- [x] Important problem addressed
- [x] Thoughtful ontology component
- [x] Two-stage training decomposition
- [x] Multi-benchmark validation
- [x] Ablation studies
- [x] Causal LM training (NEW LOCK)
- [x] Materialization depth ablation (NEW LOCK)

---

## Round 3 (v8) — Estimated Score: 8.5/10

### Issues Resolved (Now Locked)
- Pipeline order clarified (numbered steps 1-8)
- Gating interaction explicit (multiplicative)
- Oracle labeling documented (GPT-3.5, $135, 94% precision)
- Faithfulness metric defined (DeBERTa NLI)
- Compute reconciled (44 GPU-hours)
- FRAMES benchmark detailed
- GraphRAG implementation documented
- Adaptive RAG comparison added (Section 2.3)
- Co-occurrence vs BM25 ablation added

### Cumulative Locked Strengths
- [x] Important problem addressed
- [x] Thoughtful ontology component
- [x] Two-stage training decomposition
- [x] Multi-benchmark validation
- [x] Comprehensive ablation studies
- [x] Causal LM training
- [x] Materialization depth ablation
- [x] Clear pipeline documentation (NEW LOCK)
- [x] Transparent compute accounting (NEW LOCK)
- [x] Oracle labeling methodology (NEW LOCK)
- [x] Faithfulness metric rigor (NEW LOCK)
- [x] Adaptive RAG positioning (NEW LOCK)

---

## Score Trend

| Version | Score | Trend | Key Improvements |
|---------|-------|-------|------------------|
| v3 | 5.2/10 | — | Initial submission |
| v6 | 5.6/10 | ↑ +0.4 | Training loss, baselines |
| v8 | 8.5/10 | ↑ +2.9 | Full methodology transparency |

---

*Last updated: 2026-02-09*
