# RAG Paper Review History

## Paper: Multi-Source RAG for Personal Knowledge Management

---

## Review #3 (2026-02-10) — Latest (v14)

### Verdict: REJECT (with encouragement to resubmit)

### Core Problem
**Proxy vs Real Implementation**: We tested BM25/entity-match/RRF as proxies, but claimed to validate PPMI/OWL+KG/learned-gating/cross-attention. The actual proposed components aren't implemented.

### What Reviewer Wants
1. **Implement PPMI** (not just BM25 proxy)
2. **Implement OWL+KG traversal** (test on WebQSP/GrailQA)
3. **Implement learned gating** (compare to RRF)
4. **Implement cross-attention fusion**
5. **End-to-end QA metrics** (EM/F1, not just Recall)
6. **Statistical significance tests**
7. **PKM corpus evaluation** (not just HotpotQA)
8. **Baselines**: SSRAG, Beam Retrieval, MoPo

### Minor Issues
- 8.5% vs 8.4% inconsistency
- Section reference errors
- Cross-attention not used in experiments but described

---

## Review #2 (2026-02-10) — v13

### Verdict: Major Revision Required

### Critical Issues

| Issue | Severity | Status |
|-------|----------|--------|
| Experiments claim not substantiated in body | Critical | ✅ Fixed v14 - Moved to Section 4 |
| Missing figure references ("Figure ??") | Critical | ✅ Fixed v14 - Removed reference |
| score_kg mapping unspecified | Major | ✅ Fixed v14 - Added entity-to-document mapping |
| Cross-attention implementation missing | Major | ✅ Fixed v14 - Added implementation details |
| Gating training details missing | Major | ⏳ Partial (discussed in PoC) |

### Missing Related Work (All added in v14) ✅

1. **TERAG** — Token-efficient PPR graph RAG ✅
2. **GFM-RAG** — Transferable GNN retriever ✅
3. **CatRAG** — Query-adaptive KG traversal ✅
4. **CompactRAG** — Offline atomic QA knowledge base ✅
5. **ToR** — Tree-structured iterative retrieval ✅

*(Already added in v13: Beam Retrieval, DualRAG, EVO-RAG, FrugalRAG, PRISM, RT-RAG, HyperbolicRAG, FREESON, HANRAG)*

**Total related work papers: 14 recent systems (2024-2026)**

### Reviewer Questions to Address

1. How is score_kg computed (entity→document mapping)?
2. Exact HotpotQA setup (encoders, index settings, k values, baselines)?
3. Why PPMI over BM25/SPLADE? Head-to-head comparison?
4. Gating network features, labels, loss function?
5. Cross-attention architecture details?
6. Entity linking robustness on noisy PKM notes?
7. OWL materialization cost for dynamic updates?
8. Latency and token-cost breakdown per source?

### Strengths Acknowledged (Same as Review #1)
- Novel 3-source integration with adaptive gating
- OWL materialization for query-time reasoning
- Practical scaling strategies
- Clear architecture presentation
- Timely PKM focus

---

## Review #1 (2026-02-10)

### Verdict: Substantial Revision Required

### Key Criticisms

| Issue | Severity | Status |
|-------|----------|--------|
| Abstract claims 8.5% gain but no experiments in body | Critical | ✅ Fixed (experiments were there, clarified) |
| Missing related work (9 papers from 2024-2026) | Major | ✅ Fixed in v13 |
| Why PPMI over BM25? | Major | ✅ Clarified BM25 as practical proxy |
| No latency/cost analysis | Medium | ✅ Latency in results table |
| No statistical significance | Medium | ⏳ TODO: Add bootstrap CI |
| Small PoC (500 pairs, 1 annotator) | Medium | Acknowledged in limitations |
| Missing comparisons to strong baselines | Medium | ⏳ TODO: Add ColBERT, SPLADE |

### Missing Related Work (Added in v13)

1. **Beam Retrieval** - end-to-end multi-hop (ACL 2024)
2. **DualRAG** - dual-process reasoning (arXiv 2504.18243)
3. **EVO-RAG** - RL optimization (arXiv 2505.17391)
4. **FrugalRAG** - efficiency/stopping (arXiv 2507.07634)
5. **PRISM** - precision-recall balance (arXiv 2510.14278)
6. **RT-RAG** - tree-structured decomposition (arXiv 2601.11255)
7. **HyperbolicRAG** - dual-space fusion (arXiv 2511.18808)
8. **FREESON** - retriever-free traversal (arXiv 2505.16409)
9. **HANRAG** - routing + noise filtering (arXiv 2509.09713)

### Reviewer Questions (for rebuttal)

1. Where are HotpotQA experimental details? → **Section 10 in v13**
2. Why PPMI over BM25? → **Clarified: BM25 is practical implementation**
3. Rare term handling? → TODO: Add smoothing details
4. Gating training signal? → Expand PoC section
5. KG construction pipeline? → Add implementation details
6. Privacy/governance? → Add to Discussion
7. Dynamic cost constraints? → Add to Future Work

### Strengths Acknowledged

- Novel integration of 3 retrieval paradigms
- Thoughtful scalability analysis (FAISS IVF, sparse PPMI, PPR)
- Cost-aware gating concept
- PKM setting is underexplored and timely
- Clear evaluation plan

---

## v15 Fix Plan (To Avoid Rejection)

### Option A: Full Implementation (High Effort, Best Outcome)
- [ ] Implement actual PPMI retrieval (co-occurrence matrix)
- [ ] Implement OWL+KG traversal with materialization
- [ ] Implement learned gating network
- [ ] Implement cross-attention fusion
- [ ] Test on WebQSP/GrailQA
- [ ] Create small PKM corpus evaluation

### Option B: Partial Implementation (Medium Effort, Acceptable)
- [ ] Implement PPMI retrieval OR learned gating (pick one)
- [ ] Add EM/F1 end-to-end QA metrics
- [ ] Add bootstrap confidence intervals
- [ ] Add per-query-type breakdown
- [ ] Fix 8.5% vs 8.4% inconsistency
- [ ] Explicitly reframe as "proxy feasibility study"
- [ ] Add SSRAG discussion/comparison

### Option C: Reframe Only (Low Effort, Risky)
- [ ] Reframe paper as "Multi-Source RAG Feasibility Study"
- [ ] Explicitly position as testing the hypothesis with practical proxies
- [ ] Acknowledge full implementation is future work
- [ ] Target workshop/short paper venue instead of main conference

### Recommended: Option B
Implement **learned gating** (compare sigmoid vs softmax vs RRF) and add **EM/F1 metrics**. This shows we can build real components while keeping scope manageable.

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| v12 | 2026-02-09 | Added HotpotQA experiments, ablation study |
| v13 | 2026-02-10 | Added 9 related work papers, BM25/PPMI clarification |
| v14 | 2026-02-10 | **Major restructure:** Moved experiments to Section 4, added 5 more papers, added score_kg mapping, cross-attention details |
| v15 | TBD | Implement real components (gating/PPMI), add EM/F1, significance tests |

---

## Remaining TODOs for Next Revision

- [ ] Add bootstrap confidence intervals (95%)
- [ ] Add p-values for main comparisons
- [ ] Add ColBERT/SPLADE baselines discussion
- [ ] Expand KG construction details
- [ ] Add privacy/governance section
- [ ] Address rare term smoothing

---

*Last updated: 2026-02-10*
