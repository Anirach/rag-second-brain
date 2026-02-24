# NCD-CIE v5 — Technical Writer / Presentation Review

**Reviewer Role:** Technical Writer / Presentation Expert
**Paper:** NCD-CIE v5-reviewed.tex
**Date:** 2026-02-17

---

### VERDICT: WEAK ACCEPT

The paper is well-structured and clearly written for a systems paper. It needs a real architecture figure, minor language polish, and a few formatting fixes before camera-ready.

---

### Writing Quality Assessment

1. **Abstract:** Strong. States problem (associational risk calculators can't answer what-if questions), method (causal KG + logistic-link + topological counterfactual propagation), results (C-stats 0.76–0.82, r=0.91, RCT alignment), and contributions. Slightly long (~200 words) — LNCS recommends ≤150. The final sentence packs too many items (PC algorithm, fairness, DCA, transportability) and loses impact.

2. **Introduction:** Effective. The clinician question ("What would this patient's risk be if…") is an excellent motivator. Contributions C1–C6 are clearly enumerated. Minor issue: six contributions is a lot — C5 and C6 feel like validation details rather than standalone contributions.

3. **Flow:** Good overall. The progression Introduction → Related Work → Architecture → KG Details → Risk Engine → What-If → Validation → Implementation → Ethics → Discussion → Conclusion is logical. One hiccup: Section 3 (Architecture) introduces the formal KG definition, but Section 4 (Causal Knowledge Graph) continues with KG details — the boundary between these sections feels blurred. Consider merging 3.1 into Section 4 or renaming sections for clarity.

4. **Clarity:** Generally good. The ladder-of-causation framing is well-used throughout. The formal definition (Eq. 1) is clean. The topological cascade algorithm is clearly presented. Two weak spots:
   - Section 5.2 (uncertainty quantification) jumps quickly from meta-analysis to Taylor expansion without motivating why first-order approximation is sufficient.
   - The transportability section (8.1) is hand-wavy — "preliminary analysis suggests" without showing the analysis.

5. **Conciseness:** Mostly concise. Some redundancy:
   - The phrase "rung 3 of Pearl's ladder of causation" appears 5 times (abstract, intro, §3.1, discussion, conclusion). Twice is enough.
   - The contribution list in the intro partly repeats the abstract.
   - The conclusion largely restates the abstract with numbers.

---

### Structure Issues

- **Section ordering:** Logical, follows standard systems-paper convention.
- **Section balance:** Validation (Section 7) is the longest section with 7 subsections and 6 tables — appropriately so for the paper's main empirical claim. Sections 8 (Implementation) and 9 (Ethics) are thin (~0.5 page combined). Ethics could be folded into Discussion.
- **Related work positioning:** Good. Three clear paragraphs covering risk prediction, causal frameworks, and health KGs, each ending with a gap that NCD-CIE fills. The three-point differentiation at the end of §2 is effective.

---

### Tables & Figures

- **Table formatting:** All tables use booktabs correctly (toprule/midrule/bottomrule). Headers are clear. Units are generally stated (mmol/L, mmHg, pp). Table 5 (sensitivity + ablation) combines two conceptually different analyses in one table via a vertical bar — this works but is slightly confusing. Consider splitting or adding a clearer visual separator.
- **Figure 1:** **The placeholder is NOT acceptable for submission.** The text box with arrows is inadequate. The real figure should be a proper layered architecture diagram showing: (a) data sources (NHANES, EHR, labs), (b) ingestion/normalization layer, (c) the causal KG with sample nodes and edges, (d) risk engine and what-if simulator modules, (e) output layer (API, dashboard). Use TikZ or a vector graphics tool.
- **References in text:** All tables and the figure are referenced. Algorithm 1 is referenced. Equations are referenced where needed. ✓

---

### LNCS Compliance

- **Page count:** The comment says "trimmed to 14 pages." With references, this is likely 14 pages = 12 + 2 including references. Should verify compiled output — 23 references + 10 tables/figures might push it over.
- **Double-blind:** ✓ Author listed as "Anonymous Submission," institution as "Anonymous Institution." No self-citations visible. However, the mention of "open-source Python library" in the conclusion implies identifiability — reviewers could search for it. Add "upon acceptance" caveat.
- **Citation format:** ✓ Numbered citations throughout (e.g., [1], [2]).
- **Orphan headings:** No obvious orphans, but hard to verify without compilation.
- **\paragraph{} vs \subsection{}:** Used appropriately. \paragraph{} for short labeled blocks (Lifestyle Intervention Mapping, Use Case, Strengths, Limitations, Future Work). \subsection{} for substantial content blocks. ✓

---

### Language Issues

Specific sentences needing revision (approximate line locations by context):

1. **Abstract, last sentence:** "We further present data-driven graph validation via the PC algorithm, subgroup fairness analysis, and decision curve analysis demonstrating positive net clinical benefit across thresholds of 5–30%." → Too long, garden-path. Split: "We further validate the graph structure against the PC algorithm and assess subgroup fairness. Decision curve analysis confirms positive net clinical benefit across 5–30% thresholds."

2. **§3.1, paragraph after Table 1:** "Unlike Pearl's full SCM, G operates with a lightweight linear-on-log-odds approximation sufficient for clinical risk scoring." → The word "sufficient" is a strong claim. Soften: "…approximation that is adequate for clinical risk scoring within the validated operating range."

3. **§4.2:** "Each candidate edge undergoes a structured assessment against Bradford Hill's nine criteria for causal inference: strength, consistency, specificity, temporality, biological gradient, plausibility, coherence, experiment, and analogy." → Listing all nine here is unnecessary detail for the reader. Cite Hill and summarize: "…against Bradford Hill's nine criteria for causal inference [7]."

4. **§7.3:** "The DPP comparison shows the largest discrepancy (−11.3% vs. −16.0%), attributable to the DPP trial's intensive lifestyle intervention affecting pathways beyond weight loss alone." → Passive/vague. Better: "…because the DPP's intensive lifestyle intervention affects pathways (e.g., insulin sensitivity, dietary composition) not fully captured by weight loss alone."

5. **§7.5, PC algorithm section:** "The 64.5% edge agreement indicates that the majority of expert-curated edges are data-corroborated." → 64.5% is barely a majority. Rephrase: "…indicates that a substantial proportion of expert-curated edges are data-corroborated."

6. **§8, Use Case:** "A 55-year-old male with LDL-C 4.2 mmol/L, SBP 148 mmHg, HbA1c 6.8%, BMI 31, eGFR 68 receives:" → Missing units for eGFR (mL/min/1.73m²) and BMI (kg/m²).

7. **§9 (Ethics):** The six safeguards are listed as a run-on semicolon-separated sentence. Convert to an enumerated list for readability.

---

### Clarity Score: 4/5

Well-written with clear structure. Loses a point for the blurred §3/§4 boundary and the thin transportability discussion.

### Presentation Score: 3/5

Loses points primarily for: (1) placeholder figure — unacceptable; (2) 10 tables in 14 pages feels table-heavy with some consolidation possible; (3) minor redundancy in key phrases.

---

### Top 5 Writing Fixes (priority order)

1. **Replace Figure 1 placeholder** with a proper vector architecture diagram (TikZ or equivalent). This is the single biggest presentation gap.

2. **Reduce "rung 3" repetition** from 5 occurrences to 2 (introduction + conclusion). In other locations, say "counterfactual reasoning" or "counterfactual-level queries."

3. **Clarify §3/§4 boundary:** Either merge §3.1 (Formal Definition) into §4 (Causal Knowledge Graph) or rename §3 to "Overview and Formal Foundations" to better distinguish architecture overview from KG details.

4. **Add units to the Use Case** (eGFR mL/min/1.73m², BMI kg/m²) and convert the Ethics safeguards list from semicolons to enumeration.

5. **Tighten the abstract** to ≤150 words by cutting the PC algorithm/fairness/DCA sentence and letting the contributions speak through the validation results.
