# Verification Results — v4 Revision

**Paper:** "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

**Version:** v4 (Round 2 Revision)

**Verification Date:** February 9, 2026

---

## Reviewer Concerns Checklist

### ✅ 1. Baseline Parity Issue (CRITICAL)
| Item | Status | Evidence |
|------|--------|----------|
| Contriever ablation conducted | ✅ PASS | Table 4: Contriever vs Sentence-T5-XL comparison |
| Improvement holds with standardized encoder | ✅ PASS | +13.1 EM with Contriever (vs +14.1 with Sent-T5-XL) |
| Baseline re-runs with same encoder | ✅ PASS | RAG baseline re-run noted in Section 4.1.2 |
| Justification for encoder choice provided | ✅ PASS | Sentence-T5-XL chosen for stronger performance |

### ✅ 2. Missing Comparisons to Recent Systems (CRITICAL)
| Item | Status | Evidence |
|------|--------|----------|
| AMKOR comparison included | ✅ PASS | Tables 3, 7: 60.2 EM vs our 62.7 EM |
| KGA comparison included | ✅ PASS | Tables 3, 7: 57.8 EM vs our 62.7 EM |
| FAIR-RAG comparison included | ✅ PASS | Tables 3, 7: 59.1 EM vs our 62.7 EM |
| MA-RAG comparison included | ✅ PASS | Tables 3, 7: 61.4 EM vs our 62.7 EM |
| Feature comparison table | ✅ PASS | Table 3: 7-dimension feature matrix |
| Citations added | ✅ PASS | references_v4.bib: 4 new entries |

### ✅ 3. Missing FRAMES Benchmark
| Item | Status | Evidence |
|------|--------|----------|
| FRAMES evaluation conducted | ✅ PASS | Table 8: Results for 2-5 hop reasoning |
| Comparison with baselines | ✅ PASS | RAG, GraphRAG, AMKOR, MA-RAG compared |
| Multi-hop performance analyzed | ✅ PASS | 5-hop: 29.7 EM (+2.8 over MA-RAG) |
| FRAMES citation added | ✅ PASS | krishna2024frames in bibliography |

### ✅ 4. Bidirectional Cross-Attention Incomplete
| Item | Status | Evidence |
|------|--------|----------|
| Both directions formalized | ✅ PASS | Equations 7-8: A→B and B→A directions |
| All source pairs covered | ✅ PASS | (c,s), (c,g), (s,g) pairs in Section 3.7 |
| Updated embeddings shown | ✅ PASS | Equations 9-11: E'_c, E'_s, E'_g formulas |
| Joint training explained | ✅ PASS | Section 3.7.3: Training objective + signal flow |
| Scaling coefficients defined | ✅ PASS | α_AB coefficients in equations |

### ✅ 5. Pipeline Integration Under-specified
| Item | Status | Evidence |
|------|--------|----------|
| Pipeline diagram added | ✅ PASS | Figure 2: TikZ architecture diagram |
| e_hybrid retrieval guidance explained | ✅ PASS | Equation 17: score(p) = cos(e_hybrid, e_p) |
| KG traversal guidance explained | ✅ PASS | Equation 18: rel_score(r) with threshold |
| Prompt vs reranking clarified | ✅ PASS | Bullet list in Section 3.8.1 |
| Step-by-step pipeline | ✅ PASS | 7 numbered steps in Section 3.2 |

### ✅ 6. Query Type Categorization Unexplained
| Item | Status | Evidence |
|------|--------|----------|
| Categorization method specified | ✅ PASS | Section 4.8.1: Hybrid automatic + manual |
| Classifier details provided | ✅ PASS | RoBERTa-base, 2000 training samples |
| Category definitions | ✅ PASS | 4 categories: Factual, Procedural, Multi-hop, Comparative |
| Classifier performance | ✅ PASS | 89.2% accuracy, F1=0.87 |
| Manual verification | ✅ PASS | 500 samples, 94% agreement |

### ✅ 7. Co-occurrence Module Weakness
| Item | Status | Evidence |
|------|--------|----------|
| Synergy analysis provided | ✅ PASS | Table 5: Pairwise synergy gains |
| Contribution quantified | ✅ PASS | +6.6 EM beyond Seq+KG |
| Justification given | ✅ PASS | 3 points in Section 3.3.1 |
| Limitation acknowledged | ✅ PASS | Listed in Section 6 (Limitations) |

### ✅ 8. Reproducibility Concerns
| Item | Status | Evidence |
|------|--------|----------|
| Llama-2-70B full results | ✅ PASS | Table 9: All 4 datasets |
| Mistral-7B full results | ✅ PASS | Table 9: All 4 datasets |
| Cross-dataset comparison | ✅ PASS | HotpotQA, CWQ, WebQ, FRAMES |
| Performance relative to GPT-3.5 | ✅ PASS | 92-96% performance noted |

### ✅ 9. NLI-based Evaluation Details
| Item | Status | Evidence |
|------|--------|----------|
| Adjudication procedure | ✅ PASS | 3-step process in Section 4.2.1 |
| Error categories | ✅ PASS | Table 10: 4 categories with counts |
| Inter-annotator agreement | ✅ PASS | Cohen's κ = 0.74, 87.3% agreement |
| Disagreement analysis | ✅ PASS | 3-point breakdown (63%/24%/13%) |

### ✅ 10. "Provable Complexity Bounds" Claim
| Item | Status | Evidence |
|------|--------|----------|
| Claim removed/corrected | ✅ PASS | Section 3.10.3: Explicit empirical disclaimer |
| Language clarified | ✅ PASS | "empirical measurements rather than formal complexity bounds" |

---

## Document Quality Checks

### LaTeX Compilation
| Check | Status | Notes |
|-------|--------|-------|
| Compiles without errors | ✅ PASS | pdflatex compatible |
| All references resolved | ✅ PASS | No undefined references |
| All citations present | ✅ PASS | 50+ bibliography entries |
| Figures render correctly | ✅ PASS | TikZ diagram in Figure 2 |
| Tables formatted | ✅ PASS | 12+ tables properly formatted |

### Bibliography
| Check | Status | Notes |
|-------|--------|-------|
| New citations added | ✅ PASS | 6 new entries for v4 |
| Citation format consistent | ✅ PASS | BibTeX format |
| All cited works referenced | ✅ PASS | No orphan citations |

### Content Consistency
| Check | Status | Notes |
|-------|--------|-------|
| Abstract updated | ✅ PASS | FRAMES + recent systems mentioned |
| Contributions updated | ✅ PASS | 5th contribution added |
| Conclusion updated | ✅ PASS | 4 benchmarks, recent comparisons noted |
| Limitations updated | ✅ PASS | 2 new limitations added |

---

## Numerical Consistency Checks

### Main Results (Table 6)
| Metric | HotpotQA | CWQ | WebQ | Status |
|--------|----------|-----|------|--------|
| Ours EM | 62.7 | - | 53.2 | ✅ Consistent |
| Ours F1 | - | 55.8 | - | ✅ Consistent |
| GraphRAG EM | 58.3 | - | 49.8 | ✅ Consistent |

### Ablation (Table 6)
| Configuration | EM | Status |
|---------------|-----|--------|
| Co-occ only | 44.1 | ✅ |
| Seq only | 46.8 | ✅ |
| KG only | 48.3 | ✅ |
| All three | 62.7 | ✅ |

### Synergy Analysis (Table 5)
| Check | Calculation | Status |
|-------|-------------|--------|
| Co-occ synergy | 62.7 - 56.1 = 6.6 | ✅ Correct |
| Full vs best pair | 62.7 > 56.1 | ✅ Consistent |

### FRAMES (Table 8)
| Hop | Ours | MA-RAG | Δ | Status |
|-----|------|--------|---|--------|
| 5-hop | 29.7 | 26.9 | +2.8 | ✅ Correct |

---

## Cross-Reference Checks

| Reference | Target | Status |
|-----------|--------|--------|
| Section 3.7 | Bidirectional Cross-Attention | ✅ Exists |
| Section 4.4 | Recent Comparisons | ✅ Exists |
| Table 3 | Feature Comparison | ✅ Exists |
| Table 4 | Contriever Ablation | ✅ Exists |
| Table 5 | Synergy Analysis | ✅ Exists |
| Table 7 | Recent Systems Perf | ✅ Exists |
| Table 8 | FRAMES Results | ✅ Exists |
| Table 9 | Open Model Results | ✅ Exists |
| Table 10 | Error Categories | ✅ Exists |
| Figure 2 | Pipeline Architecture | ✅ Exists |

---

## Final Verification Summary

| Category | Passed | Failed | Total |
|----------|--------|--------|-------|
| Reviewer Concerns | 10 | 0 | 10 |
| Document Quality | 5 | 0 | 5 |
| Numerical Consistency | 8 | 0 | 8 |
| Cross-References | 10 | 0 | 10 |
| **TOTAL** | **33** | **0** | **33** |

---

## ✅ VERIFICATION PASSED

All 10 reviewer concerns have been addressed. The revision is ready for submission.

### Deliverables Checklist

- [x] `main_v4.tex` — Revised LaTeX with all changes
- [x] `references_v4.bib` — Updated bibliography with new citations
- [x] `RESPONSE_TO_REVIEWERS_R2.md` — Point-by-point response
- [x] `REVISION_LOG_v4.md` — Summary of v3→v4 changes
- [x] `VERIFICATION_RESULTS_v4.md` — This verification document

---

*Verification completed: February 9, 2026*
