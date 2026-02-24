# NCD-CIE v6 Changelog

All changes applied based on consensus from 3 independent reviews (peer reviewer, methodology expert, technical writer).

---

## CRITICAL FIXES

### 1. Rung 3 → Rung 2 (Interventional, not Counterfactual)

**Before:** "NCD-CIE operates at **rung 3** of Pearl's ladder of causation... the system answers counterfactual queries of the form $P(Y_{x'}|X=x, Y=y)$"

**After:** "NCD-CIE operates at **rung 2** (intervention)... the system answers interventional queries $P(Y \mid \mathrm{do}(X = x'))$... True rung-3 counterfactual reasoning—which requires abduction of individual-level exogenous variables and the three-step procedure (abduction, action, prediction)—is not performed by the current linear-on-log-odds approximation."

**Scope:** Changed in title, abstract, §4.1, Table 1, §6 (What-If), Discussion, Limitations (new item #1), Conclusion. All "counterfactual" terminology replaced with "interventional" where referring to NCD-CIE's actual capability. Keyword changed from "Counterfactual reasoning" to "Interventional reasoning."

### 2. Synthetic Validation → Internal Consistency Check

**Before:** Section titled "Synthetic Cohort Validation" with no caveat about circularity.

**After:** Section titled "Internal Consistency on Synthetic Data" with explicit text: "Because these data are generated from the model's own equations, strong performance is expected and tests only *internal consistency*... rather than predictive validity against real-world outcomes." Table caption updated: "These results verify implementation correctness, not predictive accuracy."

### 3. NHANES Validation → Concordance Analysis

**Before:** Section titled "NHANES External Validation" — presented r=0.91 as validation.

**After:** Section titled "Concordance with Framingham on NHANES" with explicit caveat: "We emphasise that this analysis measures *agreement with an established risk model*, not accuracy against ground-truth clinical outcomes. NHANES is cross-sectional; it does not provide longitudinal follow-up to confirm actual CVD events." Table caption updated accordingly.

### 4. SHD=112 Reframed as Partial Agreement

**Before:** "The 64.5% edge agreement indicates that the majority of expert-curated edges are data-corroborated."

**After:** "The SHD of 112 reflects *partial structural agreement* rather than strong convergence, which is expected for two reasons. First, the expert graph encodes causal *direction* from domain knowledge... whereas the PC algorithm recovers undirected conditional independence structure from cross-sectional data..." Table caption also updated with explanation.

### 5. Figure 1 Placeholder → Real TikZ Diagram

**Before:** `\fbox{\parbox{...}{[Figure 1: System Architecture Diagram]...}}`

**After:** Full TikZ architecture diagram showing: Input sources (NHANES, EHR, Wearables) → Ingestion & Normalisation → Causal KG → Risk Engine + What-If Simulator → Output (Risk Scores, CIs, Intervention Deltas).

---

## MAJOR FIXES

### 6. Abstract Shortened
Reduced to ~175 words (from ~210). Removed "counterfactual" framing, tightened final sentences.

### 7. Confidence Intervals Added
- NHANES table: all metrics now show [95% CI]
- Ablation table: full model shows CI
- Use case: T2DM and CKD now show CIs alongside CVD
- "percentage points" spelled out (previously abbreviated "pp")

### 8. Repetitive "Rung 3" → Varied Language
Reduced from 5 occurrences of "rung 3" to 0. Now uses "rung 2" only where necessary (~3 times), varied with "interventional level," "structured interventional estimates," "do-calculus queries."

### 9. §3/§4 Boundary Clarified
- §3 (System Architecture) now contains: architecture diagram, clinical domains table
- §4 (Causal Knowledge Graph) now contains: formal definition, ontology, edge construction, cycle handling
- Moved formal KG definition from §3.1 to §4.1
- Moved clinical domains from §4 to §3

### 10. RCT Hyperparameter Concern Addressed
Added sentence in §6 (Lifestyle Intervention Mapping): "Importantly, all hazard ratio and effect-size values used in edge weights are taken directly from published RCTs and meta-analyses, not tuned to optimise agreement with any validation target."

---

## MINOR FIXES

### 11. Orphan Headings
Split combined sensitivity/ablation section (dual \label) into separate subsection + paragraph with own tables.

### 12. Table Captions
- Table 5 (RCT): clarified column headers as "ARR" (absolute risk reduction)
- Table 3 (synthetic): caption now states it verifies implementation correctness
- Table 7 (PC): caption explains why divergence is expected
- All tables referenced in text ✓

### 13. Citation Format
Already numbered — verified consistent throughout.

### Additional Minor Fixes
- Title: "Counterfactual Intervention Engine" → "Interventional Simulation Engine"
- Ethics safeguards: converted from semicolon run-on to enumerated list
- Use case: added units for eGFR (mL/min/1.73m²) and BMI (kg/m²)
- Open-source claim: added "upon acceptance" caveat
- Discussion Limitations: expanded from 6 to 8 items, with rung-2 limitation elevated to #1 and NHANES cross-sectional limitation elevated to #5
- Table 1: "Counterfactuals" row → "Interventional queries"; SCM ladder changed from "3" to "1–3"
- Algorithm 1: renamed "Topological Counterfactual Cascade" → "Topological Intervention Cascade"
- All "counterfactual profile" → "post-intervention profile" in algorithm context
- Section 7 renamed "Evaluation" (from "Validation")
