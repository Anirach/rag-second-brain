# Academic Paper Writing Prompt — Production Template

> Refined from 23+ revision cycles on MGNA/RAG Second Brain paper (AIiH 2026). Incorporates lessons from score regressions, reviewer feedback, and quality-lock workflows.

---

## INSTRUCTIONS FOR THE AGENT

You are writing a rigorous academic research paper. Follow every structural and stylistic rule below **exactly**. Violations will cause reviewer rejection.

---

## PAPER STRUCTURE

### Title
- **Maximum 12 words.** No abbreviations. No acronyms unless universally known (e.g., "AI").
- Must convey the core contribution, not just the topic.

### Abstract
- **Maximum 12 lines** (approximately 200 words).
- Structure (one sentence each, in this order):
  1. **Area**: What research area this work belongs to.
  2. **Goal**: What this paper aims to achieve.
  3. **Concept**: The main idea or approach introduced.
  4. **Results**: The key achieved results with concrete numbers.
- End with **up to 8 keywords** describing the work area (not the specific method).

### Section 1: Introduction
- **Opening (3–4 sentences)**: Strong motivation grounded in literature. General statements about the problem domain with citations. Do NOT cite your own work here.
- **Deep dive**: Narrow progressively into the specific sub-area where your contribution lies. Each claim must be supported by a citation.
- **Research gap**: Make unmistakably clear what is missing in current approaches. Be specific — name what existing methods fail to do, not vague "limitations exist" language.
- **Closing (1 paragraph)**: Very brief roadmap of the paper's structure ("Section 2 reviews… Section 3 presents… Section 4 evaluates… Section 5 concludes."). Maximum 4–5 sentences.

### Section 2: Related Work
- **Opening**: Short classification/taxonomy of related contributions (e.g., "Approaches can be grouped into three categories: …"). Use a small table or enumeration if helpful.
- **Body**: Discuss major contributions closely related to your approach. For **each** work:
  - One sentence: what they do.
  - One sentence: what benefit they provide.
  - One sentence: **criticism** — flaws, side effects, limitations, or negative aspects. Never skip this. Every cited work must have an honest critique.
- **Closing (1 paragraph)**: Synthesize the research gap more precisely than in Section 1. Show explicitly how no existing work addresses the specific combination of problems your approach solves.

### Section 3: Methodology (Main Section)
- **3.1 Overview**: Present the main concept continuing the paper's narrative thread. Provide an overview of major steps. Include a block diagram, architecture figure, or classification scheme that **explicitly labels differences** from existing approaches. Every figure must have a caption and be referenced in text.
- **3.2+ Detailed Approach**: Present the methodical approach as formally as possible.

**Mathematical notation rules (NON-NEGOTIABLE):**
- Every symbol must be **defined before first use**, including its type, domain, and initial value if applicable.
- Each symbol may only represent **one thing** throughout the entire paper.
- Only formalize what is **actually used later** (in algorithms, proofs, or experiments). Remove decorative math.
- Every formula must be **motivated or derived** from preceding text, or cite the source. Never drop a formula without context.
- Keep definitions **short and precise** — one sentence after the formula.

**Structural rules:**
- Use algorithmic notation (Algorithm environment) for procedural steps.
- Every theorem, lemma, or assumption must be either **proved**, **motivated by heuristics with explicit justification**, or **removed**. No unsubstantiated claims.
- Structure subsections logically (e.g., 3.1 Overview → 3.2 Formal Framework → 3.3 Algorithm → 3.4 Complexity Analysis).

### Section 4: Experimental Evaluation
- **4.1 Hypotheses and Goals**: State explicitly what must be shown to prove the advantage of the new approach. Formulate as testable hypotheses (H1, H2, …).
- **4.2 Experimental Setup**: Describe where initial data and/or benchmarks come from and **why those were chosen** (not just "we used X"). Include hardware, software versions, hyperparameters.
- **4.3 Results**: Present results with **statistical evidence** — at minimum: mean, standard deviation, and number of runs. Use tables with proper formatting. Include significance tests where appropriate.
- **4.4 Discussion**: Critically discuss results. **Explicitly address problems, limitations, and failure cases** of your approach. Do not oversell. Acknowledge where baselines perform comparably or better.

**CRITICAL — Number Verification (from v8 regression incident):**
- Every number must be **independently verifiable**. Show the calculation or cite the source.
- Cost estimates: verify unit costs × volume. Off-by-1000x errors destroy credibility.
- Performance claims: verify against cited baselines using the same metrics and conditions.
- If a number cannot be verified, present it as an estimate with error bounds.

### Section 5: Conclusion
- **Exactly ~20 lines.** No more.
- Present **concrete outcomes with exact numbers** (e.g., "achieved 94.2% F1, a 12.3% improvement over…").
- State applicability: where and how this can be used.
- State limitations: what it cannot do or where it fails.
- **No argumentation, no future work speculation, no hedging.** Short, declarative sentences only.

### Bibliography
- **20–25 citations** for a standard article.
- Prioritize **survey papers** for Sections 1–2.
- Every citation must have **complete data**: all authors, full title, venue/journal name, volume, pages, year, DOI/ISBN where available.
- **Forbidden sources**: Wikipedia, blog posts, online-only sources without peer review, inaccessible literature.
- Prefer: journal articles, top-tier conference proceedings (NeurIPS, ICML, AAAI, ACL, etc.), well-known textbooks.
- **Every citation must be real and verifiable.** Never hallucinate a citation. If unsure, search for it first.

---

## GLOBAL STYLE RULES

### Writing Flow
1. **Never start or end** a section/subsection with a formula, bullet list, enumeration, figure, or table. Always begin and end with prose.
2. **Transitions**: Every section must flow logically from the previous one. The last paragraph of Section N should naturally lead into Section N+1.
3. **No redundancy**: Never state the same fact twice. If mentioned in the abstract, don't repeat verbatim in the introduction.
4. **Top-down storytelling**: Present the big picture first, then drill down. Never introduce details before the reader knows why they matter.
5. **Abbreviations**: Define on first use in the body text (not in the title, not in the abstract unless also defined there). Use the full form first: "Knowledge Graph (KG)".

### Length
- **10–12 pages** in Springer LNCS format, OR
- **6–8 pages** in double-column IEEE format.
- Whichever format is specified by the target venue.

### Figures and Tables
- Every figure/table must be referenced in the text **before it appears**.
- Captions must be self-contained — a reader should understand the figure from caption alone.
- Use consistent styling across all figures.

---

## QUALITY GATES (Before Delivery)

The paper must pass ALL of the following before being considered complete:

- [ ] **Citation verification**: Every citation exists and data is complete (search to confirm).
- [ ] **Math consistency**: Every symbol defined before use, used only once, no orphan definitions.
- [ ] **Number verification**: Every quantitative claim is calculable or cited. No unverifiable numbers.
- [ ] **Abstract–body consistency**: Claims in abstract match results in Section 4/5 exactly.
- [ ] **No section starts/ends with formula, list, figure, or table.**
- [ ] **Every related work has explicit criticism.**
- [ ] **Every theorem/lemma is proved or removed.**
- [ ] **Statistical evidence present**: mean, std dev, number of runs for all experiments.
- [ ] **Research gap clearly stated** in both Section 1 and Section 2.
- [ ] **Page count within target range.**

---

## ANTI-PATTERNS TO AVOID (Learned from Experience)

1. ❌ Claiming cost savings without showing the multiplication (e.g., "$135 for 2.7M API calls" — verify unit cost × count).
2. ❌ Citing training throughput without specifying hardware and batch size.
3. ❌ Using vague language like "significant improvement" without exact percentages.
4. ❌ Conflating different architectural components (be precise about what module does what).
5. ❌ Adding formulas that look impressive but aren't used in algorithms or experiments.
6. ❌ Presenting oracle/ground-truth results as model results.
7. ❌ Dropping a reviewer-praised element in a revision (quality regression).
8. ❌ Using "we propose" or "we present" more than twice in the entire paper.
9. ❌ Starting related work descriptions with "Author et al. (2024) proposed…" — describe the method, then cite.
10. ❌ Leaving figure/table references as "Figure ??" or "[?]".

---

*Template version: 1.0 — March 2026*
*Derived from: RAG Second Brain paper (v1–v23.1), 3× ACCEPT at AIiH 2026*
