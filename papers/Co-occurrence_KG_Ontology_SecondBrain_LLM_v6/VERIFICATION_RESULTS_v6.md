# Verification Results — Version 6.0

## Paper: "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

### Verification Date: February 2026
### Version: 6.0

---

## Verification Checklist

### Critical Technical Fixes

| # | Requirement | Status | Location | Notes |
|---|-------------|--------|----------|-------|
| 1 | Eq. 15-16 replaced with causal LM loss | ✅ PASS | Section 3.6.4, Eq. 15-16 | Standard next-token prediction formulation |
| 2 | Gradient flow clarified (fixed candidates) | ✅ PASS | Section 3.6.4, first paragraph | Explicit statement about offline retrieval |
| 3 | ALL latency numbers consistent (1.8s) | ✅ PASS | Tables 8, 9; Sections 4.6, 5 | Audited all occurrences |
| 4 | Generator parity explicit | ✅ PASS | Section 4.1.2, Table 5 caption | GPT-3.5-turbo-0613 standardized |
| 5 | Notation consistent (d=768, 847 classes/312 relations) | ✅ PASS | Throughout | d=768 (encoder), d_c=300 (embeddings) |

### High Priority Additions

| # | Requirement | Status | Location | Notes |
|---|-------------|--------|----------|-------|
| 6 | No-materialization ablation added | ✅ PASS | Section 4.8, Table 11 | +2.9 EM, +5.1% faithfulness |
| 7 | Cross-task protocol clarified | ✅ PASS | Section 4.1.3, Table 10 | Zero-shot evaluation explicit |
| 8 | RDFox alternatives listed | ✅ PASS | Section 3.4.4, Appendix A.4 | Apache Jena, RDFLib+OWL-RL |
| 9 | Missing comparisons added | ✅ PASS | Section 2.2, Tables 3, 5, 8 | RePlug, Atlas included |
| 10 | KG construction details added | ✅ PASS | Section 3.4.1 | Sources, alignment, deduplication |

### Document Format

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| 11 | No line numbers | ✅ PASS | \lineno package removed, \linenumbers commented out |
| 12 | All tables compile | ✅ PASS | LaTeX syntax verified |
| 13 | All equations numbered | ✅ PASS | Eq. 1-20 properly labeled |
| 14 | Bibliography complete | ✅ PASS | references_v6.bib includes all citations |

---

## Detailed Verification

### 1. Causal LM Loss (Equations 15-16)

**Verified Text (Section 3.6.4):**
```latex
\item \textbf{Causal LM reader head}: For decoder-only LLMs, we use standard 
\textbf{causal language modeling loss} (next-token prediction on answer tokens), 
not span-level classification:
\begin{equation}
P(a|q, C) = \prod_{t=1}^{|a|} P(a_t | a_{<t}, q, C)
\label{eq:causal_lm}
\end{equation}
```

**Status:** ✅ Correctly implements causal LM formulation for decoder-only models.

---

### 2. Gradient Flow Clarification

**Verified Text (Section 3.6.4):**
```latex
\textbf{Critical clarification on retrieval}: Retrieval (FAISS top-k) is 
\textbf{non-differentiable and performed offline}. Candidate sets are 
\textbf{fixed per batch}---retrieved before training begins. Training 
optimizes reranking scores, cross-attention parameters, and gating 
weights over these fixed candidate pools. Top-k selection happens 
\textbf{before} the differentiable portion of the pipeline.
```

**Status:** ✅ Explicit clarification addresses discrete top-k concern.

---

### 3. Latency Consistency Audit

**All Latency Mentions:**

| Location | Value | Consistent |
|----------|-------|------------|
| Abstract | 25% faster (than 2.4s) | ✅ |
| Table 1 (GraphRAG comparison) | 1.8s | ✅ |
| Table 8 (Latency comparison) | 1.8s | ✅ |
| Table 9 (Cost breakdown) | 1,800 ms | ✅ |
| Section 4.6 text | 1.8s total | ✅ |
| Section 5 (Comparison with Iterative) | 1.4-2× faster | ✅ |
| Conclusion | 25% lower latency (1.8s) | ✅ |

**Breakdown Verified:**
- Retrieval: 400ms
- Reranking + Gating: 300ms
- LLM Generation: 1,100ms
- Total: 1,800ms ✅

**Status:** ✅ All latency references consistent.

---

### 4. Generator Parity Statement

**Verified Text (Section 4.1.2):**
```latex
\textbf{Critical for fair comparison}: All methods (DPR, ColBERT, GraphRAG, 
AMKOR, KGA, FAIR-RAG, MA-RAG, PRISM, RePlug, Atlas, and ours) use 
\textbf{GPT-3.5-turbo-0613} as the generator with \textbf{identical prompts} 
and \textbf{4,096-token context windows}. Retrieval budgets are standardized 
at \textbf{top-5 passages/evidence pieces} for all methods.
```

**Table 5 Caption:**
```latex
All methods use identical GPT-3.5-turbo generator with standardized prompts 
and top-5 retrieval budget.
```

**Status:** ✅ Generator parity explicitly stated.

---

### 5. Notation Consistency

**Dimension Definitions:**
- d = 768 (Sentence-T5-XL output) ✅
- d_c = 300 (co-occurrence/KG embeddings) ✅
- d_k = 64 (attention key dimension) ✅

**Ontology Statistics (Section 3.4.3):**
```latex
\begin{itemize}
    \item \textbf{847 classes} in a 5-level hierarchy
    \item \textbf{312 object properties}
    \item \textbf{89 datatype properties}
\end{itemize}
```

**Gating Matrix (Section 3.7):**
```latex
\mathbf{W}_g \in \mathbb{R}^{3 \times d}$ (with $d=768$)
```

**Status:** ✅ All notation consistent.

---

### 6. Materialization Ablation

**Verified Table 11 (Section 4.8):**
```latex
\begin{table}[h]
\centering
\caption{Impact of ontology materialization on HotpotQA}
\label{tab:materialization}
\small
\begin{tabular}{@{}lcc@{}}
\toprule
\textbf{Setting} & \textbf{EM} & \textbf{Faithfulness (\%)} \\
\midrule
Full (with materialization) & \textbf{62.7} & \textbf{87.2} \\
KG only (no materialization) & 59.8 & 82.1 \\
No KG (Cooc + Seq only) & 56.2 & 78.4 \\
\bottomrule
\end{tabular}
\end{table}
```

**Delta Verification:**
- Materialization contribution: 62.7 - 59.8 = 2.9 EM ✅
- Faithfulness gain: 87.2 - 82.1 = 5.1% ✅

**Status:** ✅ Ablation correctly shows materialization contribution.

---

### 7. Cross-Task Protocol

**Verified Text (Section 4.1.3):**
```latex
Our gating and cross-attention modules are trained \textbf{only on HotpotQA} 
(90,069 training examples). Evaluation on ComplexWebQuestions (CWQ), 
WebQuestions (WebQ), and FRAMES is performed in a \textbf{zero-shot} 
setting---no task-specific fine-tuning.
```

**Table 10 Verified:**
- Zero-shot vs fine-tuned comparison present ✅
- Marginal gains (+1-2 EM) noted ✅

**Status:** ✅ Training/evaluation protocol clarified.

---

### 8. RDFox Alternatives

**Verified Text (Section 3.4.4):**
```latex
\textbf{Reproducibility: Open-Source Alternatives.} RDFox is commercial 
software. For reproducibility, we provide scripts for two open-source 
alternatives:

\begin{itemize}
    \item \textbf{Apache Jena + Jena Rules}: Open source, approximately 
    2$\times$ slower than RDFox.
    \item \textbf{RDFLib + OWL-RL}: Python-based, approximately 3$\times$ 
    slower.
\end{itemize}

Both alternatives are included in our code repository with usage instructions.
```

**Appendix A.4 Verified:**
```latex
\subsection{Open-Source Reasoner Alternatives}
\begin{verbatim}
# Apache Jena (recommended open alternative)
python materialize.py --reasoner jena

# RDFLib + OWL-RL (pure Python)
python materialize.py --reasoner owlrl
\end{verbatim}
```

**Limitations Section:**
- RDFox licensing mentioned ✅

**Status:** ✅ Open alternatives documented.

---

### 9. RePlug and Atlas Comparisons

**Related Work (Section 2.2):**
```latex
\textbf{RePlug} \cite{shi2024replug} treats the retriever as a plug-in 
module for frozen black-box LLMs...

\textbf{Atlas} \cite{izacard2023atlas} jointly trains retriever and 
reader components end-to-end...
```

**Table 3 (Feature Comparison):**
- RePlug row present ✅
- Atlas row present ✅

**Table 5 (Main Results):**
- RePlug: 56.4 EM, 80.1% Faith ✅
- Atlas: 59.8 EM, 83.2% Faith ✅

**Table 8 (Latency):**
- RePlug: 1.6s ✅
- Atlas: 2.0s ✅

**Bibliography (references_v6.bib):**
- shi2024replug ✅
- izacard2023atlas ✅
- nogueira2019passage ✅

**Status:** ✅ All new comparisons included.

---

### 10. KG Construction Details

**Verified Text (Section 3.4.1):**
```latex
\textbf{Data Sources:}
\begin{itemize}
    \item \textbf{Wikidata}: SPARQL dump (January 2024), filtered to 
    relations with $>$1,000 instances.
    \item \textbf{ConceptNet}: English subset, filtered to high-confidence 
    assertions (weight $>$ 2.0).
\end{itemize}

\textbf{Alignment and Deduplication:}
\begin{itemize}
    \item \textbf{Entity linking}: Alignment via QID matching
    \item \textbf{Deduplication}: Transitive closure on \texttt{owl:sameAs}
    \item \textbf{Conflict resolution}: Wikidata priority
\end{itemize}
```

**Status:** ✅ Complete KG construction pipeline documented.

---

## Summary

| Category | Checks | Passed | Failed |
|----------|--------|--------|--------|
| Critical Fixes | 5 | 5 | 0 |
| High Priority Additions | 5 | 5 | 0 |
| Document Format | 4 | 4 | 0 |
| **Total** | **14** | **14** | **0** |

---

## Final Verification

✅ **ALL CHECKS PASSED**

The v6.0 revision addresses all 9 reviewer concerns comprehensively:

1. ✅ Causal LM loss for decoder-only LLMs
2. ✅ Non-differentiable retrieval explicitly stated
3. ✅ 1.8s latency consistent throughout
4. ✅ Generator parity for all baselines
5. ✅ Notation d=768, 847 classes/312 relations
6. ✅ Materialization ablation (+2.9 EM, +5.1% faith)
7. ✅ Zero-shot cross-task evaluation
8. ✅ Apache Jena and RDFLib alternatives
9. ✅ RePlug, Atlas, cross-encoder comparisons
10. ✅ Wikidata/ConceptNet construction details

**Document is ready for submission.**
