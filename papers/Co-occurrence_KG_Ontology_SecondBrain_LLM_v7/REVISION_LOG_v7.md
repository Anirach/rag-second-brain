# Revision Log — Version 7.0

## Overview

**Previous Version**: v6 (Score: 4.4/10)
**Current Version**: v7 (Target: Recover score)
**Primary Issue**: Differentiability of top-k selection
**Solution**: Two-stage training pipeline

---

## Critical Changes

### 1. Two-Stage Training Pipeline (MAJOR)

**Problem**: v6 claimed gradients flow from generation loss to gating, but top-k selection is non-differentiable.

**Solution**: Rewritten Section 3.4 with two-stage approach:

| Stage | What's Trained | Loss Function | Differentiable? |
|-------|---------------|---------------|-----------------|
| Stage 1 | Gating network | InfoNCE (contrastive) | ✓ Yes |
| Stage 2 | Cross-attention | Generation (causal LM) | ✓ Yes (gating frozen) |

**New content added**:
- Section 3.4.1: Stage 1 contrastive training with InfoNCE loss
- Section 3.4.2: Stage 2 cross-attention fine-tuning
- Section 3.4.3: Compute breakdown (44 GPU-hours justified)
- Section 3.4.4: Training-inference model transfer validation
- Equations 3-7: Formal definitions

### 2. KG Snapshot Consistency (FIX)

**Problem**: v6 cited conflicting dates (Jan 2024 vs Jan 2023/Jun 2023).

**Fix**: Unified to:
- Training KG: Wikidata 2023-01-15
- Test KG: Wikidata 2023-06-01

Updated in Sections 3.3.1 and 4.1.3.

### 3. Compute Feasibility (CLARIFICATION)

**Problem**: 44 GPU-hours for 70B model seemed optimistic.

**Fix**: Added Table 3 with detailed breakdown:
- Stage 1: 8 GPU-hours (FP16, batch 128)
- Stage 2: 36 GPU-hours (4-bit QLoRA, batch 32)
- Throughput: 42k tokens/sec
- Wall time calculation verified

### 4. Baseline Native Configurations (NEW)

**Problem**: Top-5 constraint may handicap baselines designed for larger context.

**Fix**: Added Table 8 comparing native vs. parity configs:
- GraphRAG native (top-100 + summaries): 59.8 EM
- Our top-5: 62.7 EM
- Conclusion: Even native configs don't surpass our method

### 5. Training-Inference Transfer (NEW)

**Problem**: Mismatch between Llama-2-70B training and GPT-3.5 inference.

**Fix**: Added Table 4 showing gating accuracy transfers:
- Llama-2-70B: 89.2% Acc@5
- GPT-3.5: 88.7% Acc@5
- Mistral-7B: 88.4% Acc@5

---

## New Ablation Tables

| Table | Content | Section |
|-------|---------|---------|
| Table 2 | Entity linking error analysis | 3.3.2 |
| Table 3 | Compute breakdown | 3.4.3 |
| Table 4 | LLM transfer validation | 3.4.4 |
| Table 5 | Materialization depth | 3.3.3 |
| Table 6 | Gate distribution by query type | 3.5 |
| Table 7 | θ_r sensitivity | 3.5.1 |
| Table 8 | Baseline native config comparison | 4.3 |
| Table 9 | Gating mechanism ablation | 4.4.1 |
| Table 10 | Cross-attention fusion ablation | 4.4.2 |
| Table 11 | KG source ablation | 4.4.3 |
| Table 12 | Co-occurrence sensitivity | 4.4.4 |
| Table 13 | Source combination ablation | 4.4.5 |

---

## Removed/Modified Content

1. **Removed**: Line numbers (as requested)
2. **Removed**: Misleading claim about end-to-end gradient flow through top-k
3. **Modified**: Training section completely rewritten (old Section 3.5.4 → new Section 3.4)
4. **Modified**: Abstract updated to highlight two-stage training

---

## New Citations

```bibtex
@inproceedings{jang2017gumbel,
  title={Categorical Reparameterization with Gumbel-Softmax},
  ...
}

@article{williams1992reinforce,
  title={Simple Statistical Gradient-Following Algorithms...},
  ...
}

@inproceedings{oord2018infonce,
  title={Representation Learning with Contrastive Predictive Coding},
  ...
}

@article{dettmers2023qlora,
  title={{QLoRA}: Efficient Finetuning of Quantized LLMs},
  ...
}
```

---

## Verification Checklist

- [x] Two-stage training mathematically sound
- [x] InfoNCE loss correctly formulated (Eq. 3)
- [x] Compute breakdown adds up (8 + 36 = 44)
- [x] KG dates consistent throughout
- [x] All 5 required ablations included
- [x] Gate distribution table added
- [x] Entity linking error analysis added
- [x] θ_r sensitivity added
- [x] Native config comparison added
- [x] Training transfer validation added
- [x] No line numbers in manuscript

---

## File Changes

| File | Status | Description |
|------|--------|-------------|
| main_v7.tex | NEW | Complete rewrite of training section |
| references_v7.bib | NEW | Added Gumbel-Softmax, REINFORCE, InfoNCE, QLoRA citations |
| RESPONSE_TO_REVIEWERS_R5.md | NEW | Detailed responses to all concerns |
| REVISION_LOG_v7.md | NEW | This file |
| VERIFICATION_RESULTS_v7.md | NEW | Verification checklist |

---

## Notes for Future Revisions

1. Consider joint fine-tuning after Stage 2 (currently gating stays frozen)
2. Oracle labeling overhead (500k pairs) could be reduced with active learning
3. Chain-level evaluation for CofCA could strengthen compositional reasoning claims
4. Multilingual evaluation needed for broader impact
