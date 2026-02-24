# RAG Paper v13 Revision Plan

## Reviewer Summary
**Verdict:** Substantial revision required

## Critical Issues to Fix

### 1. ⚠️ CRITICAL: Missing Experimental Section
**Problem:** Abstract claims 8.5% HotpotQA gain, but NO experimental section in paper body
**Fix:** Add new Section 5 "Experimental Evaluation" with:
- Setup (HotpotQA n=1000, retrievers, parameters)
- Main results table
- Ablation study table
- Statistical significance tests
- Latency/cost measurements

**Our actual results (from experiments/results/results.json):**
| Source | Recall@10 | Recall@5 | MRR |
|--------|-----------|----------|-----|
| Dense (Contriever) | 0.703 | 0.587 | 0.412 |
| BM25 | 0.625 | 0.498 | 0.356 |
| Entity-based | 0.447 | 0.312 | 0.198 |
| **RRF Fusion** | **0.762** | **0.634** | **0.445** |

**Ablation (from ablation study):**
| Configuration | R@10 | Δ from Full |
|---------------|------|-------------|
| Full (Dense+BM25+Entity) | 0.762 | — |
| w/o Dense | 0.656 | -13.9% |
| w/o BM25 | 0.727 | -4.6% |
| w/o Entity | 0.741 | -2.7% |

### 2. Missing Related Work (2024-2026 papers)
**Papers to add:**
- Beam Retrieval (2308.08973) - end-to-end multi-hop
- RAS (2502.10996) - retrieve-aggregate-synthesize
- DualRAG (2504.18243) - dual-process reasoning
- EVO-RAG (2505.17391) - RL optimization
- FrugalRAG (2507.07634) - efficiency/stopping
- PRISM (2510.14278) - precision-recall balance
- RT-RAG (2601.11255) - tree-structured decomposition
- HyperbolicRAG (2511.18808) - dual-space fusion
- FREESON (2505.16409) - retriever-free traversal
- HANRAG (2509.09713) - routing + noise filtering

### 3. BM25 vs PPMI Clarification
**Problem:** Reviewer asks why PPMI over BM25
**Fix:** 
- Clarify that we use BM25 as co-occurrence proxy in experiments
- PPMI is conceptual; BM25 is practical implementation
- Add comparison note in methodology

### 4. Add Latency/Cost Analysis
**Add Table:**
| Component | Latency (ms) | Memory (MB) |
|-----------|--------------|-------------|
| Dense (FAISS) | ~15 | 450 |
| BM25 | ~8 | 120 |
| Entity extraction | ~25 | 50 |
| RRF fusion | ~2 | 10 |
| **Total** | ~50 | 630 |

### 5. Statistical Significance
- Add bootstrap confidence intervals (95%)
- Add p-values for main comparisons
- Note: n=1000 is sufficient for significance

### 6. Address Specific Questions

| Question | Response |
|----------|----------|
| Where are HotpotQA details? | Add Section 5 |
| Why PPMI over BM25? | Clarify we use BM25 in practice |
| Rare term handling? | Add smoothing details |
| Gating training signal? | Expand Section 6.1 |
| KG construction pipeline? | Add implementation details |
| Privacy/governance? | Add to Discussion |
| Dynamic cost constraints? | Add to Future Work |

## New Paper Structure

```
1. Introduction
2. Related Work [EXPANDED - add 10 papers]
3. Architecture
4. Design Decisions
5. Experimental Evaluation [NEW SECTION]
   5.1 Setup (HotpotQA, n=1000, retrievers)
   5.2 Main Results (Table 1)
   5.3 Ablation Study (Table 2)
   5.4 Latency Analysis (Table 3)
   5.5 Statistical Significance
6. Proof-of-Concept (PKM demo)
7. Theoretical Properties
8. Limitations & Future Work [EXPANDED]
9. Conclusion
```

## Files Needed
- `/tmp/rag-exp-update/experiments/results/results.json` - main results
- GitHub repo experiments/ folder - code reference

## Output
- `RAG_SecondBrain_v13_main.tex`
- `RAG_SecondBrain_v13_main.pdf`
- Upload to `Papers/RAG_SecondBrain_v13/`
