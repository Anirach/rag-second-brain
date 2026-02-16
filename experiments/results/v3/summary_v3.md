# Experiment V3 — Publication Quality Results
**Date:** 2026-02-16T00:16:31.593400
**Seed:** 42
**DDXPlus n:** 1000
**S2D:** 5-fold CV

## DDXPlus — GPT-4o-mini (n=1000)

| Method | Top-1 (95% CI) | Top-3 | Top-5 | NDCG@5 | F1 | Halluc. |
|--------|----------------|-------|-------|--------|-----|---------|
| LLM-Only | 0.299 [0.270, 0.327] | 0.412 | 0.458 | 0.389 | 0.153 | 0.620 |
| Dense-Only | 0.846 [0.823, 0.868] | 0.929 | 0.943 | 0.918 | 0.316 | 0.000 |
| BM25+PPMI | 0.915 [0.898, 0.932] | 0.956 | 0.956 | 0.966 | 0.321 | 0.011 |
| Multi-Source | 0.943 [0.928, 0.957] | 0.958 | 0.959 | 0.965 | 0.321 | 0.000 |

### Statistical Significance (McNemar's Test, Top-1)

| Comparison | p-value | Significant? |
|-----------|---------|-------------|
| LLM-Only vs Dense-Only | 0.0000 | ✓ |
| Dense-Only vs BM25+PPMI | 0.0000 | ✓ |
| BM25+PPMI vs Multi-Source | 0.0013 | ✓ |
| LLM-Only vs Multi-Source | 0.0000 | ✓ |

## DDXPlus — GPT-4o (Ablation)

| Method | Top-1 (95% CI) | Top-3 | Top-5 | NDCG@5 | F1 | Halluc. |
|--------|----------------|-------|-------|--------|-----|---------|
| LLM-Only | 0.261 [0.234, 0.290] | 0.388 | 0.456 | 0.369 | 0.155 | 0.671 |
| Dense-Only | 0.812 [0.787, 0.837] | 0.924 | 0.948 | 0.914 | 0.319 | 0.011 |
| Multi-Source | 0.932 [0.917, 0.947] | 0.959 | 0.960 | 0.974 | 0.323 | 0.014 |

## DDXPlus — Traditional ML Baselines

| Method | Top-1 (95% CI) | Top-3 | Top-5 | F1 |
|--------|----------------|-------|-------|-----|
| XGBoost | 0.994 [0.989, 0.998] | 1.000 | 1.000 | 0.336 |
| kNN | 0.991 [0.984, 0.996] | 1.000 | 1.000 | 0.336 |
| RandomForest | 0.929 [0.914, 0.944] | 0.951 | 0.953 | 0.320 |

## Error Analysis (DDXPlus Multi-Source)
Total errors: 57

- True disease found by Dense: 35/57 (61%)
- True disease found by BM25: 39/57 (68%)
- True disease found by KG: 57/57 (100%)
- In candidates but ranked wrong: 16/57

### Most Misdiagnosed Diseases
| Disease | Errors | % of total errors |
|---------|--------|-------------------|
| Pneumonia | 21 | 36.8% |
| SLE | 20 | 35.1% |
| Acute rhinosinusitis | 7 | 12.3% |
| HIV (initial infection) | 3 | 5.3% |
| Anaphylaxis | 2 | 3.5% |
| Possible NSTEMI / STEMI | 1 | 1.8% |
| Pericarditis | 1 | 1.8% |
| Chronic rhinosinusitis | 1 | 1.8% |
| Pancreatic neoplasm | 1 | 1.8% |
