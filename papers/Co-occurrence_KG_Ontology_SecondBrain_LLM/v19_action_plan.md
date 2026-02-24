# v19 Action Plan — From v18 Reviews

**Goal:** Address all remaining issues to reach Accept/Minor Revision across all 3 reviewers  
**Current status:** Peer=Minor Revision, Methodology=Major Revision, Writing=Accept w/ Minor

---

## Phase 1: Text Fixes (no experiments needed) — ~3 hours

These can be done immediately by editing the LaTeX:

| # | Issue | Fix | Effort |
|---|-------|-----|--------|
| T1 | Error analysis contradiction: "100% KG recall" but "only 28.1% ranking failures" | Clarify: KG *retrieved* true disease in all 57 error cases, but 16/57 had it ranked wrong (ranking failure). Other 41 had true disease NOT in Top-1 but present in candidate pool — these are *scoring* failures where wrong disease scored higher. Rewrite to distinguish retrieval recall vs ranking precision | 30 min |
| T2 | Explain low Macro F1 (0.321 at 94.3% accuracy) | Add paragraph: Macro F1 is computed by treating only the Top-1 prediction as positive for each class. With 49 classes and n=1000, many rare diseases have <10 test cases, making per-class F1 unstable. XGBoost also shows 0.336 despite 99.4% Top-1, confirming this is a dataset artifact not a system flaw. Consider also reporting weighted F1 | 30 min |
| T3 | Fix "26-30%" range in conclusion | Change to "26–33%" to include S2D's 32.5% baseline | 5 min |
| T4 | Acknowledge prompt-constraint confound on hallucination | Add paragraph in Discussion: "We note that our constrained prompt ('consider ONLY these candidates') contributes to low hallucination rates. The retrieval architecture's contribution is providing the *correct* candidates to constrain against, rather than preventing hallucination per se. Without quality retrieval, constraining to wrong candidates would harm accuracy while technically showing zero hallucination." | 20 min |
| T5 | Discuss NDCG decrease (−0.001) when adding KG | Add note in ablation analysis: "The marginal NDCG decrease (−0.001) when adding KG is within statistical noise and reflects a ranking redistribution: KG improves Top-1/3/5 by promoting correct diagnoses but occasionally reorders near-optimal rankings." | 15 min |
| T6 | Fix "zero-shot" claim | Change "zero-shot generalization" to "low-resource generalization" or "knowledge-driven generalization" since PPMI/KG use training data. Reserve "zero-shot" only for the dense retrieval component over disease profiles | 15 min |
| T7 | Explain KG higher hallucination than Dense RAG with GPT-4o | Add note: "GPT-4o shows slightly higher hallucination (1.4%) with Multi-Source KG than Dense RAG alone (1.1%). This occurs because KG-retrieved candidates include diseases connected by data-driven edges that may fall outside the LLM's constrained generation scope when using the more capable model. The effect is small (0.3pp) and absent with GPT-4o-mini." | 15 min |
| T8 | Add S2D confidence intervals | Compute bootstrap CIs for S2D n=320 results and add to Table 4. Can calculate from the existing per-case results | 30 min |
| T9 | Add Bonferroni correction note | Add footnote: "With Bonferroni correction for 4 comparisons (α_adj = 0.0125), all results remain significant as all p-values ≤ 0.0013" | 5 min |
| T10 | Add bootstrap method details | Add to Experimental Setup: "95% CIs computed via 1000 percentile bootstrap resamples with stratified sampling, seed 42" | 10 min |
| T11 | Replace Figure 1 with proper TikZ diagram | Create TikZ architecture diagram showing: Patient Symptoms → [Dense / BM25+PPMI / KG] → RRF → LLM → Diagnoses. Use proper boxes, arrows, colors | 1 hour |
| T12 | Trim extended mind discussion | Keep 1 concise paragraph instead of 2. Remove Kahneman dual-process (weakest link). Keep Clark & Chalmers + Hutchins | 15 min |

---

## Phase 2: New Experiments — ~4-6 hours

These require running code. The repo has the experiment infrastructure; we need to extend it.

| # | Issue | Experiment | Approach | Effort |
|---|-------|-----------|----------|--------|
| E1 | **KG edge type ablation** (CRITICAL for Info Fusion) | Run Multi-Source KG with each edge type removed | Modify KG scoring (Eq. 4) to set α, β, or γ to 0 individually. Run 4 configs: Full KG, KG−Ontological (α=0), KG−CoOccurrence (β=0), KG−PPMI (γ=0) on n=1000 DDXPlus | 2 hours |
| E2 | **Fusion weight sensitivity** (CRITICAL for Info Fusion) | Grid search over weights | Test α,β,γ ∈ {0, 0.25, 0.5, 0.75, 1.0} normalized, and λ ∈ {0.0, 0.25, 0.5, 0.75, 1.0}. Report heatmap or table of Top-1 accuracy | 2 hours |
| E3 | **S2D significance tests** | McNemar's test on S2D | Run pairwise McNemar on S2D per-case predictions (n=320). Already have per-case data | 30 min |
| E4 | **RAG baseline comparison** | Add naive single-source RAG numbers | Run Dense-only RAG as the "standard RAG" baseline. Also add numbers from published papers (MedRAG, Med-PaLM on DDXPlus if available) via literature search. If no DDXPlus numbers exist, cite this gap explicitly | 1-2 hours |

---

## Phase 3: Paper Integration — ~2 hours

| # | Task | Details |
|---|------|---------|
| I1 | New table: KG Edge Ablation | Table showing Full KG vs KG−O vs KG−C vs KG−P with Top-1, Top-5, NDCG@5 |
| I2 | New table/figure: Weight Sensitivity | Heatmap or table of Top-1 across weight configs |
| I3 | New subsection: Fusion Analysis | "These results demonstrate that [X] edges contribute most, while [Y] edges provide complementary signal..." |
| I4 | Update abstract & contributions | Add "fusion weight analysis" as a contribution |
| I5 | Update limitations | Acknowledge remaining gaps honestly |

---

## Work Order

```
Phase 1 (Text fixes)     ─── Sub-agent: technical-writer ──→ v19 draft
                                                               │
Phase 2 (Experiments)     ─── Sub-agent: code-builder ────→ results
                                                               │
Phase 3 (Integration)     ─── Sub-agent: technical-writer ──→ v19 final
                                                               │
Phase 4 (Re-review)       ─── 3 reviewers in parallel ────→ verdict
```

**Parallel execution:**
- Phase 1 and Phase 2 can run simultaneously
- Phase 3 depends on both completing
- Phase 4 is the final quality gate

**Estimated total time:** 6-8 hours of agent work (parallelized to ~4 hours wall time)

---

## Expected Outcome

After v19, all reviewer concerns should be addressed:
- Peer reviewer: Minor → **Accept** (all issues resolved)
- Methodology reviewer: Major → **Minor** (fusion analysis added, remaining items are polish)
- Writing reviewer: Accept w/ Minor → **Accept** (Figure 1 fixed, text polished)

---

## What We're NOT Doing (and why)

| Skipped | Reason |
|---------|--------|
| Real clinical dataset (MIMIC) | Requires IRB/access, out of scope for this revision |
| Open-source LLMs (Llama-3, Meditron) | Acknowledged in limitations, nice-to-have for camera-ready |
| Full test set (134K) | n=1000 with CIs is statistically adequate; 134K would be redundant |
| Learned fusion weights | Acknowledged as future work; manual grid search suffices |
| Per-class confusion matrix | Error analysis + top misdiagnosed diseases covers this |
