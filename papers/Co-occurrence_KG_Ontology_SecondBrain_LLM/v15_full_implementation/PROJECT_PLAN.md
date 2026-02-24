# RAG Paper v15 — Full Implementation Project

## Goal
Implement ALL proposed components and validate on benchmarks to address reviewer rejection.

## Team
- **Coding Team**: Build the implementations
- **Academic Team**: Update paper with real results

---

## Implementation Tasks

### 1. PPMI Retrieval Module
**Owner:** Coding Team
**Status:** 🔴 Not Started

**Requirements:**
- Build co-occurrence matrix from corpus (sliding window)
- Compute PPMI scores: `PPMI(wi,wj) = max(0, log(p(wi,wj)/(p(wi)*p(wj))))`
- Sparse storage (top-N neighbors per term)
- Document scoring: `score(q,d) = Σ PPMI(qi,dj) * tf(dj,d)`
- Compare against BM25 baseline

**Files to create:**
- `experiments/ppmi_retriever.py`
- `experiments/cooccurrence_matrix.py`

### 2. OWL+KG Traversal Module
**Owner:** Coding Team  
**Status:** 🔴 Not Started

**Requirements:**
- Entity extraction (spaCy NER)
- Entity linking (simple string match or BLINK)
- Build KG from HotpotQA passages (entity-entity relations)
- OWL ontology with class hierarchies
- PPR traversal with materialized inferences
- Entity-to-document scoring with IDF weighting

**Files to create:**
- `experiments/kg_builder.py`
- `experiments/owl_reasoner.py`
- `experiments/ppr_retriever.py`

### 3. Learned Gating Network
**Owner:** Coding Team
**Status:** 🔴 Not Started

**Requirements:**
- Feature extraction: query embedding, entity density, question type
- MLP gating: g = σ(W[h_q; f_q] + b)
- Training with retrieval relevance labels
- Compare: sigmoid vs softmax vs RRF baseline
- Ablation on gating features

**Files to create:**
- `experiments/gating_network.py`
- `experiments/train_gating.py`

### 4. Cross-Attention Fusion
**Owner:** Coding Team
**Status:** 🔴 Not Started

**Requirements:**
- Encode candidates with query: h_i = Encoder([CLS; q; SEP; c_i])
- Cross-attention over all candidates
- Final scoring layer
- Compare vs RRF fusion

**Files to create:**
- `experiments/cross_attention_fusion.py`

### 5. End-to-End QA Evaluation
**Owner:** Coding Team
**Status:** 🔴 Not Started

**Requirements:**
- Add answer generation step (LLM)
- Compute EM/F1 metrics
- Per-query-type breakdown (entity, compositional, comparison)
- Bootstrap confidence intervals
- Statistical significance tests

**Files to create:**
- `experiments/qa_evaluation.py`
- `experiments/metrics.py`

### 6. Additional Benchmarks
**Owner:** Academic Team
**Status:** 🔴 Not Started

**Requirements:**
- WebQSP evaluation (KGQA)
- Small PKM corpus creation
- Comparison with SSRAG baseline

---

## Paper Updates (After Implementation)

### New Sections
- Section 5: Full System Evaluation (replace proxy experiments)
- Add WebQSP/KGQA results
- Add learned gating ablations
- Add cross-attention vs RRF comparison

### Fixes
- 8.5% → 8.4% consistency
- Section reference errors
- Clarify cross-attention role

---

## Timeline

| Day | Task |
|-----|------|
| 1 | PPMI retriever + Co-occurrence matrix |
| 2 | KG builder + PPR traversal |
| 3 | Learned gating network |
| 4 | Cross-attention fusion |
| 5 | End-to-end QA evaluation |
| 6 | Run all experiments, collect results |
| 7 | Update paper with real results |

---

## Success Criteria

- [ ] PPMI retriever outperforms or matches BM25
- [ ] KG+PPR adds signal beyond entity matching
- [ ] Learned gating outperforms RRF
- [ ] Cross-attention improves over simple fusion
- [ ] EM/F1 metrics reported with significance
- [ ] Paper updated with all real results

---

*Created: 2026-02-10*
