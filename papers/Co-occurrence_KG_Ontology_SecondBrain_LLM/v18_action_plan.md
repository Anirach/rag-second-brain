# v18 Action Plan — RAG Second Brain Paper

**Based on:** Peer review + methodology review + writing review (3 agents, 2026-02-16)  
**Verdict:** Major Revision for Information Fusion  
**Target:** v18 submission-ready

---

## 🔴 CRITICAL (Must fix — paper rejected without these)

### 1. Fix hallucination claim inconsistency
- **Problem:** Abstract/conclusion say "zero hallucination" and "completely eliminates hallucination," but Table 5 shows GPT-4o Multi-Source KG has 0.3% hallucination
- **Fix:** Change claims to "reduces hallucination to ≤0.3%" or "near-zero." Reserve "zero" only for GPT-4o-mini results where it's actually true
- **Sections:** Abstract, §1 contributions bullet 3, §2.3 related work, §5.5 hallucination analysis, §6 conclusion
- **Effort:** 1 hour (text edits only)

### 2. Add statistical significance testing
- **Problem:** n=200/320 with no CIs, no significance tests. A single misclassification = 0.5pp change
- **Fix options (pick one or both):**
  - **(a) Scale up:** Run on 5K+ cases from 134K DDXPlus test set (preferred — eliminates sample size objection entirely)
  - **(b) Bootstrap CIs:** Run 1000 bootstrap resamples on existing n=200, report 95% CIs for all metrics
  - **(c) Multiple samples:** Run 5-10 stratified random samples of n=200, report mean ± std
- **Recommendation:** Do (a) if experiment code is available. Otherwise (b) + (c)
- **Effort:** 2-4 hours if code exists; 1-2 days if re-running experiments

### 3. Add competitive baselines
- **Problem:** Only baseline is naked LLM — a straw man. No comparison to existing medical RAG/diagnosis systems
- **Fix — add at minimum:**
  - kNN classifier on symptom vectors (simple but strong baseline)
  - Single-source KG-RAG (e.g., standard KG-RAG without multi-source fusion)
  - If possible: MedRAG or published medical RAG numbers on DDXPlus
- **Effort:** 1-2 days (requires running experiments)

### 4. Replace Figure 1 placeholder
- **Problem:** Figure 1 is a text box, not a proper diagram
- **Fix:** Create a proper architecture diagram showing the 3-layer pipeline, RRF fusion, and LLM prompting
- **Tool:** TikZ, draw.io, or Mermaid → PDF
- **Effort:** 2-3 hours

---

## 🟡 MAJOR (Strongly recommended — significantly strengthens paper)

### 5. Ablate KG edge types
- **Problem:** Paper's core novelty (3 edge types: ontological, co-occurrence, PPMI) is never decomposed
- **Fix:** Run ablation removing each edge type individually from the full system:
  - Full KG vs. KG−ontological vs. KG−co-occurrence vs. KG−PPMI
  - Shows which edges matter most
- **Effort:** 4-8 hours (modify KG code + re-run)

### 6. Discuss low Macro F1
- **Problem:** Macro F1 is 0.163–0.331 even for best system. Buried, not discussed
- **Fix:**
  - Add per-disease analysis (confusion matrix or per-disease accuracy table)
  - Discuss: low Macro F1 likely due to rare diseases with few test cases
  - Show which diseases are misclassified and whether errors are concentrated in similar disease pairs
- **Effort:** 3-4 hours

### 7. Broaden hallucination definition
- **Problem:** Current definition = vocabulary check (disease name ∈ known set). Misses clinically dangerous hallucinations: valid disease name but wrong for symptoms
- **Fix:**
  - Rename current metric to "vocabulary hallucination" or "out-of-vocabulary rate"
  - Add "clinical relevance" analysis: are wrong predictions at least in the right disease family?
  - Discuss both forms in §5.5
- **Effort:** 2-3 hours

### 8. Rework "Second Brain" framing
- **Problem:** Citing Tiago Forte's self-help book is weak for Information Fusion. Peer reviewer called it "marketing"
- **Fix options:**
  - **(a) Drop it:** Remove Second Brain framing, use "multi-modal retrieval fusion" or "multi-source retrieval framework"
  - **(b) Ground it:** Connect to cognitive science (dual-process theory, external cognition, distributed cognition literature) instead of a productivity book
- **Recommendation:** Option (b) — the metaphor is actually good, just needs better grounding
- **Effort:** 2-3 hours

---

## 🟢 MODERATE (Good to have — polish for acceptance)

### 9. Add weight sensitivity analysis
- **Problem:** λ=0.5, α=β=γ=1/3, k_rrf=60 are all arbitrary with no sensitivity analysis
- **Fix:** Grid search or sensitivity table showing performance across different weight configurations
- **Effort:** 3-4 hours

### 10. Show prompt template
- **Problem:** Prompt template not included — critical for reproducibility
- **Fix:** Add full prompt in appendix or supplementary material
- **Effort:** 30 minutes

### 11. Add cost/latency analysis
- **Problem:** Three retrieval calls + LLM inference per case — no cost/time reported
- **Fix:** Add table with wall-clock time and API cost per case for each configuration
- **Effort:** 1-2 hours

### 12. Standardize notation
- **Problem:** Mix of n=200 and n{=}200 formats
- **Fix:** Use n{=}200 consistently throughout (prevents line breaks)
- **Effort:** 15 minutes

### 13. Add more figures
- **Problem:** Only 1 figure (and it's a placeholder)
- **Fix suggestions:**
  - KG visualization showing the 3 edge types on a disease-symptom subgraph
  - Performance comparison bar chart
  - Hallucination rate reduction visualization
- **Effort:** 2-3 hours

### 14. Test with open-source LLMs
- **Problem:** Only GPT-4o family tested — can't claim model-agnostic with only OpenAI
- **Fix:** Add Llama-3 or Meditron results (even just one row in the table)
- **Effort:** 4-8 hours

### 15. Title consistency
- **Problem:** Earlier versions used "Co-occurrence, Sequence and Knowledge Graph..." but current title is different
- **Fix:** Ensure all references, filenames, and metadata use the current title consistently
- **Effort:** 15 minutes

---

## 📋 Suggested Work Order

| Phase | Items | Effort | Impact |
|-------|-------|--------|--------|
| **Phase 1: Text fixes** | #1, #8, #10, #12, #15 | 1 day | Removes easy objections |
| **Phase 2: Experiments** | #2, #3, #5, #6 | 3-5 days | Addresses all critical gaps |
| **Phase 3: Analysis** | #7, #9, #11 | 1-2 days | Strengthens methodology |
| **Phase 4: Figures** | #4, #13 | 1 day | Professional presentation |
| **Phase 5: Stretch** | #14 | 1 day | Nice to have |

**Total estimated effort: 7-10 days**

---

## ✅ What's Already Strong (keep as-is)
- Internal numerical consistency across all tables ✓
- Progressive ablation design ✓
- Model-equalizer finding (genuinely interesting) ✓
- Clean writing quality overall ✓
- Honest limitations section ✓
- Reproducibility details (model versions, temps, seeds) ✓
