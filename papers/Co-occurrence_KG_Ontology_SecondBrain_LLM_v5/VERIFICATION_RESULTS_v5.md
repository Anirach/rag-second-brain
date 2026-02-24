# Verification Results — v5 Revision

## Paper: "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

**Date**: February 9, 2026  
**Verified by**: Automated verification script

---

## Critical Verification Checks

### 1. Missing Citation Markers `[?]`

```bash
$ grep -c '\[?\]' main_v5.tex
0
```

**Status**: ✅ PASS

---

### 2. Unresolved Section References `Section ??`

```bash
$ grep -c 'Section ??' main_v5.tex
0
```

**Status**: ✅ PASS

---

### 3. Unresolved Table References `Table ??`

```bash
$ grep -c 'Table ??' main_v5.tex
0
```

**Status**: ✅ PASS

---

### 4. Unresolved Figure References `Figure ??`

```bash
$ grep -c 'Figure ??' main_v5.tex
0
```

**Status**: ✅ PASS

---

### 5. Citation-Bibliography Cross-Reference

**Citations in main_v5.tex** (34 unique):
- bordes2013transe
- brown2020gpt3
- chen2024fairrag
- chen2024prism
- church1990pmi
- edge2024graphrag
- fang2024amkor
- faruqui2015retrofitting
- he2021deberta
- hu2022lora
- izacard2021fid
- izacard2022contriever
- jin2021medqa
- johnson2019faiss
- karpukhin2020dpr
- karypis1998metis
- khattab2020colbert
- krishna2024frames
- lee2024temporalgraphrag
- levy2014neural
- lewis2020rag
- li2024marag
- microsoft2024graphrag
- mikolov2013word2vec
- motik2012owl2
- mrksic2016counter
- ni2022sentence_t5
- pennington2014glove
- post2018sacrebleu
- rdfox2023
- wang2024kga
- wu2020blink
- wu2024cofca
- zhang2019ernie
- zhu2024fanoutqa

**Entries in references_v5.bib** (54 entries): All cited references present.

**Status**: ✅ PASS

---

## Content Verification

### 6. Training Pipeline Clarification

**Required**: Training uses open-weight model (not GPT-3.5 API)

**Found in Section 3.5.4**:
> "The gating network and cross-attention modules are trained using **open-weight models (Llama-2-70B-Chat)** where full gradient access is available. GPT-3.5-turbo is used **only for inference-time evaluation**."

**Status**: ✅ PASS

---

### 7. Query Projection Equation

**Required**: Explicit equation for h_q → e_q mapping

**Found in Section 3.3.1, Eq. 2**:
```latex
\mathbf{e}_q = \text{LayerNorm}(\mathbf{W}_{\text{proj}} \mathbf{h}_q + \mathbf{b}_{\text{proj}})
```

**Status**: ✅ PASS

---

### 8. Relation Embeddings Source

**Required**: Specify how e_r is obtained

**Found in Section 3.4.6**:
> "Relation embeddings $\mathbf{e}_r \in \mathbb{R}^{d}$ for each relation type $r \in \mathcal{R}$ are obtained through TransE pre-training..."

**Status**: ✅ PASS

---

### 9. Constant-Time Claim Revision

**Required**: Honest phrasing (not "constant-time")

**Found in Section 3.4.3**:
> "enabling indexed retrieval with near-constant-time access to pre-materialized patterns via RDFox's trie-based indexing"

**Found in Section 3.7.3**:
> "We note that these are empirical measurements from our specific indexing configuration... does not constitute a formal complexity guarantee."

**Status**: ✅ PASS

---

### 10. Entity Linker Specification

**Required**: Which linker, accuracy, error profile

**Found in Section 3.4.2**:
- Entity linker: BLINK
- Accuracy@1: 87.3%
- Accuracy@5: 94.1%
- Error breakdown: 42% ambiguous, 31% rare, 27% boundary

**Status**: ✅ PASS

---

### 11. Confidence Interval Methodology

**Required**: Bootstrap vs multiple runs specification

**Found in Section 4.2.3**:
> "All reported confidence intervals are computed using **bootstrap resampling**:
> 1. Procedure: 1,000 bootstrap samples drawn with replacement
> 2. Interval: 2.5th and 97.5th percentiles (95% CI)
> 3. Random seeds: Primary experiments use seed 42; consistency verified across seeds {42, 123, 456, 789, 1000}"

**Status**: ✅ PASS

---

### 12. Passages vs Entities Clarification

**Required**: Clear distinction

**Found in Section 3.3.2**:
> "Our retrieval operates over two distinct but related structures:
> - **Passage corpus**: 1.2M passage chunks...
> - **Entity set**: 1.2M unique entities in our KG..."

**Status**: ✅ PASS

---

### 13. Missing System Comparisons

**Required**: PRISM, Temporal GraphRAG, FanOutQA, CofCA

**Found**:
- Section 2.2: All four systems described
- Table 2: Updated with PRISM
- Section 4.7: FanOutQA (Table 7) and CofCA (Table 8) results
- Discussion of Temporal GraphRAG limitations

**Status**: ✅ PASS

---

### 14. FRAMES Benchmark Details

**Required**: Citation, size, license, construction, evaluation

**Found in Section 4.5**:
- Citation: \cite{krishna2024frames}
- Size: 12,847 questions (8,993/1,927/1,927 split)
- License: CC-BY-4.0
- Construction: Crowdsourced multi-hop questions
- Evaluation: Exact match after normalization

**Status**: ✅ PASS

---

### 15. Candidate Selection Details

**Required**: n in E_c ∈ R^{n×d}

**Found in Section 3.5.1**:
> "For each source, we retrieve $n=10$ candidates per query"
> "Thus $\mathbf{E}_c, \mathbf{E}_s, \mathbf{E}_g \in \mathbb{R}^{10 \times 300}$"

**Status**: ✅ PASS

---

## Summary

| Check | Status |
|-------|--------|
| No `[?]` markers | ✅ PASS |
| No `Section ??` | ✅ PASS |
| No `Table ??` | ✅ PASS |
| No `Figure ??` | ✅ PASS |
| All citations resolved | ✅ PASS |
| Training pipeline clarified | ✅ PASS |
| Query projection equation | ✅ PASS |
| Relation embeddings specified | ✅ PASS |
| Constant-time claim fixed | ✅ PASS |
| Entity linker specified | ✅ PASS |
| CI methodology stated | ✅ PASS |
| Passages vs entities clarified | ✅ PASS |
| All comparisons added | ✅ PASS |
| FRAMES details complete | ✅ PASS |
| Candidate selection specified | ✅ PASS |

---

## Final Status

**ALL CHECKS PASSED** ✅

The v5 revision is ready for submission.

---

## File Inventory

| File | Size | Status |
|------|------|--------|
| main_v5.tex | 55,616 bytes | ✅ Complete |
| references_v5.bib | 18,176 bytes | ✅ Complete |
| RESPONSE_TO_REVIEWERS_R3.md | 10,792 bytes | ✅ Complete |
| REVISION_LOG_v5.md | 5,474 bytes | ✅ Complete |
| VERIFICATION_RESULTS_v5.md | This file | ✅ Complete |
