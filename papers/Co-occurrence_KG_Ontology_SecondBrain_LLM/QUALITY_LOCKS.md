# Quality Locks — DO NOT REGRESS

**Paper:** Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM

---

## How This Works

Once an improvement is verified and praised (or resolves a reviewer concern), it becomes a **LOCK**.
Future revisions **MUST preserve** locked items. Any change that would remove or weaken a lock requires explicit justification.

---

## Current Locks

### ✅ VERIFIED — Core Architecture (Immutable)

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-001 | v3 | Two-stage training pipeline (InfoNCE + Cross-attention) | R1 praise | ✅ VALID |
| LOCK-002 | v3 | Three-source fusion (co-occurrence, KG, ontology) | Core contribution | ✅ VALID |
| LOCK-003 | v3 | Ontology materialization for reasoning | Core contribution | ✅ VALID |

### ⚠️ NEEDS CLARIFICATION — Methodology

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-004 | v6 | Causal LM loss (Eq. 15-16) | R2 request resolved | ✅ VALID |
| LOCK-005 | v6 | Training: Llama-2, Inference: GPT-3.5 | R2 clarification | ⚠️ NEEDS CLARIFICATION (how Stage 2 cross-attn used at inference) |
| LOCK-006 | v8 | Pipeline order: Steps 1-8, cross-attention on ALL 30 before gating | R3 request resolved | ⚠️ NEEDS CLARIFICATION (conflated with generation loss) |
| LOCK-007 | v8 | Multiplicative gating: score(i,s) = g_cand(i) × g_src(s) | R3 request resolved | ⚠️ NEEDS CLARIFICATION (InfoNCE → gating head gradient flow) |
| LOCK-008 | v8 | Oracle labeling: GPT-3.5 T=0, 2.7M calls, $135, 94%/87% P/R | R3 request resolved | ❌ **INVALIDATED** (cost wrong by 1000x, needs closed-book control) |

### ✅ VERIFIED — Metrics & Evaluation

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-009 | v6 | Faithfulness = DeBERTa NLI entailment | R3 request resolved | ✅ VALID |
| LOCK-010 | v6 | Entity linking: BLINK with confidence thresholds | R2 clarification | ✅ VALID |
| LOCK-011 | v8 | FRAMES: 12,847 questions, 2-5 hops, CC-BY-4.0 | R3 request resolved | ⚠️ NEEDS DETAIL (split, EM/F1 definitions, deviations) |

### ❌ INVALIDATED — Compute & Reproducibility

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-012 | v8 | Total compute: 44 GPU-hours (8 + 36) | R3 request resolved | ❌ **INVALIDATED** (tokens/sec unrealistic) |
| LOCK-013 | v8 | Stage 1: FP16, batch 128, 85k tokens/sec | R3 detail | ❌ **INVALIDATED** (needs wall-clock logs) |
| LOCK-014 | v8 | Stage 2: 4-bit QLoRA, batch 32, 42k tokens/sec | R3 detail | ❌ **INVALIDATED** (needs wall-clock logs) |

### ⚠️ NEEDS EXPANSION — Baselines & Comparisons

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-015 | v6 | RePlug comparison: 56.4 EM | R2 request resolved | ✅ VALID |
| LOCK-016 | v6 | Atlas comparison: 59.8 EM | R2 request resolved | ✅ VALID |
| LOCK-017 | v8 | GraphRAG: Microsoft official, local mode, Leiden | R3 request resolved | ⚠️ NEEDS NATIVE-CONFIG (global mode missing) |
| LOCK-018 | v8 | Adaptive RAG section (EA-GraphRAG, EfficientRAG, RT-RAG, etc.) | R3 request resolved | ⚠️ NEEDS EXPANSION (RT-RAG, CoopRAG, GenGround, SentGraph, HGRAG) |

### ✅ VERIFIED — Ablations

| Lock ID | Version | Description | Source | Status |
|---------|---------|-------------|--------|--------|
| LOCK-019 | v3 | Gating mechanism ablation | R1 praise | ✅ VALID |
| LOCK-020 | v3 | Cross-attention fusion ablation | R1 praise | ✅ VALID |
| LOCK-021 | v3 | KG source ablation | R1 praise | ✅ VALID |
| LOCK-022 | v6 | Materialization depth ablation (+2.9 EM) | R2 request resolved | ✅ VALID |
| LOCK-023 | v8 | Co-occurrence vs BM25 ablation (+1.3 EM) | R3 request resolved | ✅ VALID |

---

## Pre-Revision Checklist (v9+)

Before creating any future version, verify:

- [ ] LOCK-001 through LOCK-023 are preserved
- [ ] No locked content removed or weakened
- [ ] New improvements added as new locks
- [ ] Score trend remains positive

---

## Lock Violation Protocol

If a revision would violate a lock:

1. **STOP** — Do not proceed with that change
2. **Document** the conflict in REVISION_LOG
3. **Seek approval** from paper owner (Anirach)
4. **If approved**, add explicit rationale for unlock
5. **Update** QUALITY_LOCKS.md with unlock record

---

## Unlock History

*No unlocks recorded.*

---

*Last updated: 2026-02-09*
