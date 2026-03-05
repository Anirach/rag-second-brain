# Reviewer Score Calibration Database

> Track correlation between internal review scores and actual conference outcomes.
> Updated after each venue decision.

## Completed Papers

| Paper | Version | Internal Score | Venue | Actual Outcome | Delta |
|-------|---------|---------------|-------|----------------|-------|
| RAG Second Brain | v23.1 | 3× ACCEPT | AIiH 2026 | ACCEPT (all 3 reviewers) | 0 ✅ |
| NCD-CIE | v20 | 3× WEAK ACCEPT | AIiH 2026 | Pending | — |

## Calibration Metrics

### Score-to-Outcome Mapping

| Internal Score | Expected Outcome | Actual Outcomes | Accuracy |
|---------------|-------------------|-----------------|----------|
| 3× ACCEPT | Accept | 1/1 (RAG) | 100% |
| 2× ACCEPT + 1× WA | Accept | — | — |
| 3× WEAK ACCEPT | Borderline | Pending (NCD-CIE) | — |
| Mixed (any REJECT) | Revise/Reject | — | — |

### Per-Reviewer Accuracy

| Reviewer Agent | Predictions | Correct | Accuracy | Bias |
|---------------|-------------|---------|----------|------|
| peer-reviewer | 1 | 1 | 100% | neutral (insufficient data) |
| methodology-expert | 1 | 1 | 100% | neutral (insufficient data) |
| technical-writer | 1 | 1 | 100% | neutral (insufficient data) |
| red-team-reviewer | 0 | — | — | new agent, no data yet |

### Lessons Learned

1. **RAG Second Brain (v23.1):** Internal 3× ACCEPT matched actual outcome perfectly. Quality locks and iterative refinement (23 versions) produced a strong paper.
2. **NCD-CIE (v20):** Internal 3× WEAK ACCEPT correctly identified borderline quality. Key issues: rung-3 overclaim, circular validation, no ground-truth outcomes.

### Calibration Notes

- Minimum 5 data points needed before statistical bias analysis
- Minimum 10 data points for formal calibration adjustments
- Current data: 1 confirmed, 1 pending — too early for calibration

---

*Last updated: 2026-03-04*
