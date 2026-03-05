# 🔍 Rejection Autopsy Log

> Structured post-mortems for rejected papers.  
> Every rejection teaches something. Capture it here.  
> Findings feed back into all reviewer agents as permanent lessons.

*Last updated: 2026-03-04*

---

## Template

When a paper is rejected, fill out this template:

```markdown
## Autopsy: [Paper Title] — [Venue] [Year]

**Date:** YYYY-MM-DD
**Decision:** REJECT / DESK REJECT
**Venue:** [Conference/Journal]
**Version submitted:** v[N]
**Internal prediction:** [What our reviewers predicted]
**Actual outcome:** [What happened]

### Reviewer Comments Summary
| Reviewer | Decision | Main Concern |
|----------|----------|-------------|
| R1 | [Accept/WA/WR/Reject] | [One-line summary] |
| R2 | [Accept/WA/WR/Reject] | [One-line summary] |
| R3 | [Accept/WA/WR/Reject] | [One-line summary] |

### Root Cause Analysis (5 Whys)
1. **Why was it rejected?** [Direct reason]
2. **Why did that happen?** [Contributing factor]
3. **Why wasn't it caught?** [Pipeline gap]
4. **Why does the pipeline have that gap?** [Systemic issue]
5. **What's the fix?** [Concrete action]

### What Our Internal Review Missed
| Actual Reviewer Concern | Did We Catch It? | Why Not? |
|------------------------|-----------------|----------|
| [Concern 1] | ❌ No | [Reason] |
| [Concern 2] | ✅ Yes | — |
| [Concern 3] | ❌ No | [Reason] |

### Calibration Impact
- **Peer-reviewer accuracy:** [Did they predict correctly?]
- **Methodology-expert accuracy:** [Did they catch the methodology issues?]
- **Red-team accuracy:** [Did they find the actual fatal flaw?]
- **Calibration adjustment:** [What weights/prompts need updating?]

### Lessons Learned
1. [Lesson → added to KNOWLEDGE_BASE.md Section X]
2. [Lesson → added to reviewer agent prompt]
3. [Lesson → new red-team attack vector]

### Action Items
- [ ] Update KNOWLEDGE_BASE.md with new patterns
- [ ] Update venue profile if venue-specific lesson
- [ ] Adjust reviewer agent prompts
- [ ] Add new red-team attack vector if applicable
- [ ] Decide: Revise & resubmit vs. Pivot venue vs. Abandon
- [ ] If resubmit: Create revision plan

### Resubmission Plan (if applicable)
**Target venue:** [Same or different]
**Key changes needed:**
1. [Change 1]
2. [Change 2]
**Estimated timeline:** [X weeks]
```

---

## Completed Autopsies

*No rejections yet. (RAG-KB accepted, NCD-CIE pending.)*

---

## Aggregate Patterns

*Updated as autopsies accumulate:*

### Most Common Rejection Reasons (Across All Papers)
| Reason | Count | Prevention |
|--------|-------|------------|
| (pending data) | — | — |

### Pipeline Gaps Identified
| Gap | Found Via | Status |
|-----|----------|--------|
| (pending data) | — | — |

### Reviewer Agent Accuracy
| Agent | Correct Predictions | Misses | Accuracy |
|-------|-------------------|--------|----------|
| (pending data) | — | — | — |
