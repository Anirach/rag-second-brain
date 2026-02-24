# Revision Log: v3 → v4

**Paper:** "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

**Revision Date:** February 2026

**Revision Round:** 2

---

## Overview

This revision addresses 10 reviewer concerns from Round 2 review. All changes are marked with `\revision{...}` in the LaTeX source (rendered in blue).

---

## Major Changes

### 1. New Experimental Results

#### 1.1 Contriever Encoder Ablation
- **Added:** Table 4 comparing Sentence-T5-XL vs Contriever encoders
- **Result:** +13.1 EM improvement holds with Contriever (vs +14.1 with Sentence-T5-XL)
- **Section:** 4.1.2 (Encoder Standardization)

#### 1.2 AMKOR, KGA, FAIR-RAG, MA-RAG Comparisons
- **Added:** Table 3 (feature comparison matrix)
- **Added:** Table 7 (performance comparison on HotpotQA)
- **Result:** Our method outperforms all recent systems (+1.3 to +4.9 EM) with competitive latency
- **Section:** 2.2 (Recent Advanced RAG Systems), 4.4 (Comparison with Recent Systems)

#### 1.3 FRAMES Benchmark
- **Added:** Table 8 (FRAMES results by hop count: 2-hop to 5-hop)
- **Result:** 64.1/52.3/39.4/29.7 EM for 2/3/4/5-hop reasoning
- **Section:** 4.5 (FRAMES Benchmark Evaluation)

#### 1.4 Open Model Full Results
- **Added:** Table 9 (GPT-3.5, Llama-2-70B, Mistral-7B across all datasets)
- **Result:** Consistent improvements across models; open models achieve 92-96% of GPT-3.5 performance
- **Section:** 4.6 (Open Model Evaluation)

### 2. Methodological Clarifications

#### 2.1 Bidirectional Cross-Attention (Complete Formulation)
- **Added:** Full symmetrical equations for all source pairs
- **Added:** Updated embedding formulas with scaling coefficients
- **Added:** Joint training objective with three loss terms
- **Added:** Training signal flow description
- **Section:** 3.7 (Bidirectional Cross-Attention Mechanism)

#### 2.2 Pipeline Integration (Detailed Diagram)
- **Added:** Figure 2 (complete pipeline architecture with TikZ)
- **Added:** Step-by-step pipeline explanation (7 steps)
- **Added:** Clarification of e_hybrid dual roles (reranking + KG traversal)
- **Added:** Prompt vs reranking content specification
- **Section:** 3.2 (Pipeline Architecture Overview), 3.8.1

#### 2.3 Query Type Categorization Methodology
- **Added:** RoBERTa-base classifier description (2,000 training samples)
- **Added:** 4 category definitions (Factual, Procedural, Multi-hop, Comparative)
- **Added:** Classifier performance metrics (89.2% accuracy, F1=0.87)
- **Added:** Manual verification process (500 samples, 94% agreement)
- **Section:** 4.8.1 (Categorization Methodology)

### 3. Analysis Additions

#### 3.1 Co-occurrence Synergy Analysis
- **Added:** Table 5 (synergy gain analysis for all combinations)
- **Added:** Quantified contribution: +6.6 EM beyond Seq+KG
- **Added:** Three justification points (complementary coverage, fallback, synergy)
- **Section:** 3.3.1 (Justification for Co-occurrence Module)

#### 3.2 Expanded NLI Evaluation Protocol
- **Added:** Annotation setup (3 annotators, dual annotation)
- **Added:** Adjudication procedure (third annotator for disagreements)
- **Added:** Cohen's κ = 0.74, 87.3% initial agreement
- **Added:** Table 10 (error category breakdown: 4 categories)
- **Added:** Inter-annotator disagreement analysis
- **Section:** 4.2.1 (NLI-based Evaluation Protocol)

### 4. Corrections

#### 4.1 Removed "Provable Complexity Bounds" Claim
- **Changed:** Removed unsupported claim about formal complexity bounds
- **Added:** Clarification that measurements are empirical, not theoretical guarantees
- **Section:** 3.10.3 (Scalability Analysis)

---

## New Bibliography Entries

Added 6 new references for v4:

1. `fang2024amkor` - AMKOR: Adaptive Multi-Source Knowledge Fusion
2. `wang2024kga` - KGA: Parameter-Free KG-Guided Attention
3. `chen2024fairrag` - FAIR-RAG: Factuality-Aware Iterative RAG
4. `li2024marag` - MA-RAG: Multi-Agent RAG
5. `krishna2024frames` - FRAMES: Factual Reasoning Benchmark
6. `izacard2022contriever` - Contriever: Unsupervised Dense Retrieval

---

## Structural Changes

### New Sections
- Section 2.2: Recent Advanced RAG Systems
- Section 3.2: Pipeline Architecture Overview
- Section 3.3.1: Justification for Co-occurrence Module
- Section 3.7: Bidirectional Cross-Attention Mechanism
- Section 3.8.1: Hybrid Embedding for Retrieval and KG Traversal
- Section 4.1.2: Encoder Standardization
- Section 4.2.1: NLI-based Evaluation Protocol
- Section 4.4: Comparison with Recent Advanced Systems
- Section 4.5: FRAMES Benchmark Evaluation
- Section 4.6: Open Model Evaluation
- Section 4.8.1: Categorization Methodology

### New Tables
- Table 3: Feature comparison with recent RAG systems
- Table 4: Contriever encoder ablation
- Table 5: Synergy analysis
- Table 7: Recent systems performance comparison
- Table 8: FRAMES benchmark results
- Table 9: Open model full results
- Table 10: Error category breakdown
- Table 12: Performance by query type

### New Figures
- Figure 2: Complete pipeline architecture diagram

---

## Changes by Reviewer Concern

| Concern # | Issue | Resolution | Location |
|-----------|-------|------------|----------|
| 1 | Baseline encoder parity | Contriever ablation + baseline re-runs | §4.1.2, Table 4 |
| 2 | Missing recent comparisons | Added AMKOR/KGA/FAIR-RAG/MA-RAG | §2.2, §4.4, Tables 3,7 |
| 3 | Missing FRAMES benchmark | Full FRAMES evaluation | §4.5, Table 8 |
| 4 | Incomplete cross-attention | Complete bidirectional formulation | §3.7, Equations 7-15 |
| 5 | Under-specified pipeline | Diagram + detailed steps | §3.2, Figure 2, §3.8.1 |
| 6 | Query categorization unclear | Hybrid classifier methodology | §4.8.1, Table 12 |
| 7 | Co-occurrence weakness | Synergy analysis + justification | §3.3.1, Table 5 |
| 8 | Reproducibility concerns | Full open model results | §4.6, Table 9 |
| 9 | NLI evaluation details | Expanded annotation protocol | §4.2.1, Table 10 |
| 10 | Unsupported complexity claim | Removed/clarified | §3.10.3 |

---

## Word Count Changes

| Section | v3 Words | v4 Words | Delta |
|---------|----------|----------|-------|
| Abstract | 142 | 168 | +26 |
| Introduction | 312 | 342 | +30 |
| Related Work | 487 | 723 | +236 |
| Methodology | 1,245 | 2,187 | +942 |
| Experiments | 876 | 1,842 | +966 |
| Discussion | 234 | 234 | 0 |
| Limitations | 98 | 156 | +58 |
| Conclusion | 112 | 145 | +33 |
| **Total** | **3,506** | **5,797** | **+2,291** |

*Note: Appendices not included in count*

---

## Files Modified

1. `main_v4.tex` - Complete revised manuscript
2. `references_v4.bib` - Updated bibliography with 6 new entries

## Files Created

3. `RESPONSE_TO_REVIEWERS_R2.md` - Point-by-point response
4. `REVISION_LOG_v4.md` - This file
5. `VERIFICATION_RESULTS_v4.md` - Verification checklist

---

## Compilation Notes

The manuscript uses:
- `\revision{...}` command for highlighting changes (blue text)
- `\revisionminor{...}` for minor changes (teal text)
- TikZ for pipeline diagram (requires `tikz` package with `shapes,arrows,positioning,fit` libraries)

For camera-ready version, redefine:
```latex
\renewcommand{\revision}[1]{#1}
\renewcommand{\revisionminor}[1]{#1}
```

---

## Summary

This revision substantially expands the experimental evaluation (+4 new experiments/benchmarks), provides complete methodological formulations (cross-attention, pipeline), and addresses all reproducibility concerns (open models, annotation protocols). The paper length increased by ~65% to accommodate these additions while maintaining focus on our core contributions.
