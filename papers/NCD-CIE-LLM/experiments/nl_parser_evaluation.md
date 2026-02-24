# NCD-CIE Natural Language Parser Evaluation Framework

## Overview

This document describes the evaluation methodology for the LLM-powered NL→do-calculus parser in NCD-CIE. The parser converts clinician free-text queries into structured `do(Variable=value) → query(Endpoint)` operations validated against the causal graph schema (51 nodes, 107 edges, 8 clinical domains).

## Test Suite Summary

| Category | Count | Description |
|---|---|---|
| Simple single-intervention | 10 | Single variable, clear intent (e.g., "start a statin") |
| Compound multi-intervention | 10 | 2–3 simultaneous interventions (e.g., "quit smoking AND exercise") |
| Quantitative | 10 | Explicit numeric targets/deltas (e.g., "LDL drops by 1.5 mmol/L") |
| Ambiguous/edge cases | 10 | Vague or underspecified queries requiring clarification |
| Out-of-scope | 10 | Non-interventional or unrelated queries |
| **Total** | **50** | |

## Scoring Rubric

| Score | Label | Criteria |
|---|---|---|
| 1.0 | Full match | Correct variable(s) + correct value(s) + correct target endpoint |
| 0.5 | Partial match | Correct variable(s) but wrong/imprecise values, or missing target |
| 0.0 | Wrong parse | Incorrect intervention variables identified |
| 1.0 | Correct rejection | Properly returns `clarification_needed` or `out_of_scope` |
| 0.0 | Missed rejection | Hallucinates an intervention for ambiguous/out-of-scope query |

## Evaluation Metrics

For each category, compute:
- **Accuracy**: fraction of test cases scoring ≥ 0.5
- **Exact match rate**: fraction scoring 1.0
- **Mean score**: average score across category

Aggregate metrics:
- **Overall accuracy** (weighted by category)
- **Rejection precision**: TP / (TP + FP) for out-of-scope detection
- **Rejection recall**: TP / (TP + FN) for out-of-scope detection

## Expected Accuracy Ranges

Based on the query complexity and current LLM capabilities (GPT-4-class models):

| Category | Expected Exact Match | Expected Accuracy (≥0.5) | Rationale |
|---|---|---|---|
| Simple single-intervention | 85–95% | 95–100% | Clear intent, single variable mapping |
| Compound multi-intervention | 70–85% | 85–95% | Conjunction parsing is reliable; value mapping harder with 3+ vars |
| Quantitative | 75–90% | 90–95% | Numeric extraction is strong; unit conversion (min→MET-h, kg→BMI) may introduce errors |
| Ambiguous/edge cases | 70–85% | 80–90% | Should trigger clarification; risk of over-interpretation |
| Out-of-scope | 85–95% | 90–98% | Clear non-medical queries easy; borderline medical queries (e.g., cancer, age change) harder |
| **Overall (weighted)** | **77–90%** | **88–96%** | |

## Key Evaluation Dimensions

### 1. Variable Identification
Does the parser correctly map natural language to graph node names?
- Synonyms: "blood pressure" → SBP, "cholesterol" → LDL_C, "sugar" → HbA1c
- Medications: "statin" → Statin, "BP meds" → HTN_med

### 2. Value Extraction
Does the parser correctly extract intervention magnitudes?
- Absolute targets: "SBP to 130" → SBP=130
- Relative deltas: "LDL drops by 1.5" → LDL_C=-1.5
- Unit conversion: "150 min/wk exercise" → Exercise=+7.5 MET-h/wk
- Weight-to-BMI: "lose 10 kg" → BMI≈-3.5 (requires height context)
- Binary: "quit smoking" → Smoking=0, "start statin" → Statin=1

### 3. Target Inference
Does the parser correctly identify the query endpoint?
- Default: CVD_Risk (most common clinical concern)
- Context-dependent: "diabetes risk" → T2DM_Risk, "kidney function" → CKD_Risk
- Medication hints: SGLT2i → CKD_Risk, metformin → T2DM_Risk

### 4. Appropriate Rejection
Does the parser correctly identify unparseable queries?
- Out-of-scope: non-medical, non-interventional
- Non-modifiable variables: age, sex
- Ambiguous: vague terms without specific variable mapping

## Execution Protocol

1. Load patient context (representative Framingham cohort profile)
2. For each test case, send query to parser with patient context
3. Record structured output: `{do_operations, target, type}`
4. Score against expected output using rubric
5. Compute per-category and aggregate metrics
6. Report with 95% Wilson confidence intervals

## File References

- Test suite: `nl_parser_test_suite.json` (50 queries with expected outputs)
- Execution script: TBD (requires LLM API access)
- Results: TBD (after execution)
