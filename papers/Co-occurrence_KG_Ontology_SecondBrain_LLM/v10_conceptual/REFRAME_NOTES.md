# REFRAME NOTES: v9 → v10 Conceptual Proposal

## Overview

This document describes the major changes from v9 (empirical paper with fabricated results) to v10 (honest conceptual proposal paper).

---

## What Was REMOVED

### 1. Fabricated Experimental Results

| Removed Claim | Location |
|--------------|----------|
| "62.7 EM on HotpotQA" | Abstract, Main Results |
| "55.8 F1 on ComplexWebQuestions" | Abstract |
| "+4.4 points over GraphRAG" | Abstract, Conclusion |
| "25% faster inference" | Abstract |
| All Table entries with specific numbers | Tables 1-12 |

### 2. Unverified Cost Claims

- ❌ "$460 for oracle labeling"
- ❌ "36 GPU-hours total training"
- ❌ "~$700 total reproducibility cost"

### 3. Unsubstantiated Performance Comparisons

- ❌ All baseline comparison numbers
- ❌ Ablation study specific percentages
- ❌ FRAMES benchmark results by hop count
- ❌ Latency measurements

---

## What Was KEPT (with modifications)

### 1. Core Framework Architecture

The three-source fusion concept remains valid:
- Co-occurrence statistics (captures associative patterns)
- Sequential patterns (Sentence-T5 embeddings)
- Knowledge Graph (with ontology reasoning)

**Modified to**: Presented as proposed architecture, not implemented system.

### 2. Two-Stage Training Design

The concept of separating:
- Stage 1: Contrastive gating training (InfoNCE)
- Stage 2: Cross-attention fine-tuning

**Modified to**: Includes mathematical proof that InfoNCE is differentiable and convergent.

### 3. Gating Mechanism

The two-level gating (per-candidate + source-level) is mathematically sound.

**Added**: Formal proof that gating outputs are well-defined and bounded.

### 4. Ontology Reasoning

OWL 2 RL with materialization is a real technique with known benefits.

**Modified to**: Theoretical analysis of what benefits are expected.

---

## What Was ADDED

### 1. Mathematical Proofs (proofs.tex)

1. **Theorem 1**: Gating mechanism well-definedness
   - Proof that $g_{cand}(i) \in [0,1]$ for all candidates
   - Proof that $g_{src}(s)$ forms valid probability distribution

2. **Theorem 2**: InfoNCE convergence properties
   - Connection to mutual information maximization
   - Convergence under standard assumptions

3. **Theorem 3**: Complexity bounds for retrieval pipeline
   - Time complexity: O(n·log(k) + k²·d)
   - Space complexity: O(n·d + k²)

### 2. Algorithm Pseudocode (algorithms.tex)

- Full pipeline algorithm with line-by-line explanation
- Training algorithms for both stages
- Clear input/output specifications
- Complexity annotations

### 3. Proof-of-Concept Code (poc_code.py)

Small-scale Python implementation that demonstrates:
- Co-occurrence scoring works on toy data
- Gating mechanism implementation
- Source fusion with learned weights
- Can run on laptop with 100 samples

### 4. Theoretical Performance Analysis

- Why three sources > one source (information-theoretic argument)
- Why learned gating > uniform weights (adaptation argument)
- Why ontology reasoning helps (deductive inference argument)

### 5. Proposed Evaluation Protocol

Instead of fake results, we now have:
- "We propose the following evaluation protocol..."
- Datasets to use (HotpotQA, CWQ, WebQ, FRAMES)
- Baselines to compare against
- Ablation study design
- Expected results based on component analysis

---

## Structural Changes

### Old Structure (v9)
```
1. Abstract (claims results)
2. Introduction
3. Related Work
4. Methodology
5. Experiments (fake results)
6. Discussion
7. Conclusion (claims achievements)
```

### New Structure (v10)
```
1. Abstract ("We propose...")
2. Introduction (contributions are theoretical + algorithmic)
3. Related Work
4. Proposed Framework
   4.1 Overview and Architecture
   4.2 Mathematical Foundations
   4.3 Three-Source Retrieval (with proofs)
   4.4 Gating Mechanism (with convergence proof)
   4.5 Training Protocol
5. Algorithms and Complexity Analysis
6. Proof-of-Concept Validation (small-scale, runnable)
7. Theoretical Performance Analysis
8. Proposed Evaluation Protocol
9. Discussion and Limitations (honest)
10. Conclusion and Future Work
11. References
```

---

## Language Changes

### Abstract
- ~~"We achieve 62.7 EM"~~ → "We propose a framework"
- ~~"outperforming GraphRAG by 4.4 points"~~ → "theoretically enables improved retrieval"

### Contributions
- ~~"State-of-the-art results"~~ → "Theoretical complexity analysis"
- ~~"62.7 EM on HotpotQA"~~ → "Proof-of-concept validation"

### Experiments → Proposed Evaluation
- All "achieves" → "expects to achieve"
- All specific numbers → ranges or qualitative predictions

---

## Quality Assurance

Every claim in v10 must satisfy one of:
1. **Mathematically proven** in proofs.tex
2. **Demonstrated with runnable code** in poc_code.py
3. **Clearly marked as proposed/expected**
4. **Cited from existing literature**

---

## Target Venues

This paper is now suitable for:
- Workshop papers (e.g., KG4LLM, GraphLLM workshops)
- Position/vision papers
- ArXiv preprint (for community feedback)
- Short papers at main conferences

Future extension: Add full-scale experiments when compute becomes available.
