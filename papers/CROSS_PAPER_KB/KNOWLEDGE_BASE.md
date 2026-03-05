# 📚 Cross-Paper Knowledge Transfer Database

> Lessons, patterns, and insights accumulated across all papers.  
> Every paper review cycle feeds back into this KB.  
> New papers should consult this BEFORE starting.

*Last updated: 2026-03-04*

---

## 🎯 Purpose

When starting Paper B, automatically load lessons from Paper A, C, D...  
This prevents repeating the same mistakes and surfaces proven strategies.

---

## 📋 Lesson Categories

### 1. Reviewer Prediction Patterns
*What reviewers consistently ask about*

| Pattern | Papers Seen In | Frequency | Preemptive Action |
|---------|---------------|-----------|-------------------|
| "How is this different from X?" | RAG-KB, NCD-CIE | Always | Add explicit differentiation table |
| "Why not compare with [recent baseline]?" | RAG-KB | Common | Search arXiv for baselines < 6 months old |
| "Overclaiming in abstract" | RAG-KB v8 | Common | Use "suggests/indicates" not "proves/shows" |
| "Missing limitations" | NCD-CIE | Common | Always include limitations subsection |
| "Reproducibility concerns" | RAG-KB | Common | Include compute budget, seeds, code link |
| "Page count" | NCD-CIE | LNCS-specific | Monitor early, cut proactively |

### 2. Writing Patterns That Work

| Pattern | Evidence | Impact |
|---------|----------|--------|
| Start intro with clinical vignette/scenario | RAG-KB v23 praised by all 3 reviewers | High engagement |
| Contribution list in intro (numbered) | Standard across accepted papers | Clarity |
| Explicit "Our approach differs from X by Y" sentences | RAG-KB differentiation | Preempts novelty questions |
| Compute transparency table | RAG-KB v10+ | Addresses reproducibility |
| Two-stage training decomposition | RAG-KB — reviewer locked this as strength | Methodological clarity |

### 3. Writing Patterns That Fail

| Anti-Pattern | Evidence | Impact |
|-------------|----------|--------|
| Wrong numbers that look precise | RAG-KB v8: "$135 for 2.7M calls" (actual: $135K) | Score dropped 5.6→4.4 |
| Placeholder Figure 1 | NCD-CIE early versions | "Lazy/incomplete" impression |
| Circular validation claims | NCD-CIE: synthetic data "validates" model that generated it | Methodological red flag |
| Overclaiming causal from observational | NCD-CIE: "proves" from correlation | Instant credibility loss |
| Vague "state-of-the-art" without numbers | Multiple | Reviewer demands specifics |

### 4. Venue-Specific Lessons

#### LNCS / Springer
- Page limit is HARD. Plan for 16 pages, overflow to 18 max.
- Reviewers care about format compliance
- 3 reviewers typical, all need at least WEAK ACCEPT
- Review cycle: ~4-6 weeks

#### AIiH Conference
- Healthcare focus — clinical relevance matters
- Interdisciplinary reviewers — don't assume deep ML knowledge
- Practical applicability valued over theoretical novelty

### 5. Tool & Process Lessons

| Lesson | Source | Action |
|--------|--------|--------|
| Sub-agents can't access /home/clawdbot/clawd/papers/ | RAG-KB workflow | Copy files to agent workspace before spawning |
| DOCX style=None causes crashes | Report generation | ALWAYS specify style |
| Numbers must be verified before any review | RAG-KB v8 disaster | Number verification protocol mandatory |
| Lock positive feedback to prevent regression | RAG-KB v8→v9 | QUALITY_LOCKS.md system |
| Red-team BEFORE submission, not after | Post-hoc realization | Pipeline step 11 |

---

## 📊 Paper History & Outcomes

| Paper | Venue | Versions | Internal Score | Outcome | Key Lesson |
|-------|-------|----------|---------------|---------|------------|
| RAG Second Brain | AIiH 2026 | 23 versions | 3× ACCEPT | ACCEPTED ✅ | Persistence + quality locks work |
| NCD-CIE | AIiH 2026 | 20 versions | 3× WEAK ACCEPT | Submitted, pending | Reframe validation honestly |

---

## 🔄 How to Use This KB

### Starting a New Paper
```
1. Read CROSS_PAPER_KB/KNOWLEDGE_BASE.md
2. Check "Reviewer Prediction Patterns" — preemptively address known concerns
3. Check "Writing Patterns That Work" — incorporate proven strategies
4. Check "Writing Patterns That Fail" — avoid known anti-patterns
5. Check venue-specific lessons if targeting a known venue
6. After review, add new lessons back to this KB
```

### After Each Review Cycle
```
1. Extract new reviewer patterns → add to Section 1
2. Identify what worked → add to Section 2
3. Identify what failed → add to Section 3
4. Update venue-specific lessons → Section 4
5. Update paper history → Section 5
6. Run rejection autopsy if rejected → add to AUTOPSY_LOG.md
```

---

## 🧠 Meta-Insights

- **Quality accumulates**: Each paper makes the pipeline better
- **Reviewers are predictable**: 80% of concerns are repeats from prior papers
- **Precise wrong > vague right**: Never fabricate specific numbers. Round numbers with honest ranges are safer.
- **The first version is never the last**: Plan for 5+ revision cycles
- **Lock your wins**: When reviewers praise something, NEVER change it
