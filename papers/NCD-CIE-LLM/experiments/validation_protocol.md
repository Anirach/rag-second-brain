# Validation Protocol: T2DM and CKD Outcome Validation

## Overview

This protocol extends the NCD-CIE paper's outcome-based validation (currently CVD-only on Framingham) to T2DM and CKD endpoints using NHANES 2017–2020 data already referenced in the paper.

## Design

**Study type:** Cross-sectional validation (not prospective—acknowledged limitation)  
**Data source:** NHANES 2017–2020 (n ≈ 8,291 adults ≥ 20 years, as used in Section 5 of the paper)  
**Model:** Logistic-link scoring from Section 3.3: R_d = σ(β₀_d + Σ w_i · z_i)  
**Approach:** Zero-fit validation — KG edge weights used directly as logistic coefficients (no re-fitting on outcome data)

---

## T2DM Validation

### Endpoint
- **Primary:** HbA1c ≥ 6.5% OR self-reported diabetes diagnosis (DIQ010 = 1)
- **Expected prevalence:** ~13% (NHANES 2017–2020 adults)

### Predictors (from KG Table 2)
| Predictor | Target | Weight (w_i) | Grade | Source |
|-----------|--------|--------------|-------|--------|
| BMI | T2DM-onset | 0.38 | A | Colditz 1995 |
| Exercise | HbA1c | −0.08 | A | Umpierre 2011 |
| Age | T2DM | 0.25 | A | CDC prevalence data |
| Sex (male) | T2DM | 0.10 | B | Kautzky-Willer 2016 |

### NHANES Variables
- BMI: BMXBMI (continuous, z-scored)
- Exercise: PAQ605/PAQ650 (binary: meets guidelines or not)
- Age: RIDAGEYR (continuous, z-scored)
- Sex: RIAGENDR (binary)
- HbA1c: LBXGH (continuous)
- Diabetes self-report: DIQ010

### Intercept Calibration
β₀ set to match population prevalence via: β₀ = logit(prevalence) − Σ w_i · E[z_i]  
With standardized predictors (mean 0), β₀ ≈ logit(0.13) ≈ −1.90

---

## CKD Validation

### Endpoint
- **Primary:** eGFR < 60 mL/min/1.73m² (CKD Stage 3+)
- **eGFR formula:** CKD-EPI 2021 (race-free)
- **Expected prevalence:** ~15% (NHANES 2017–2020 adults)

### Predictors (from KG Table 2)
| Predictor | Target | Weight (w_i) | Grade | Source |
|-----------|--------|--------------|-------|--------|
| SBP | CKD-prog | 0.18 | A | AASK 2002 |
| Smoking | CKD-prog | 0.15 | B | Orth 2004 |
| Age | CKD-prog | 0.30 | A | Coresh 2007 |
| Diabetes | CKD | 0.25 | A | USRDS/Afkarian 2013 |

### NHANES Variables
- SBP: BPXOSY3 (continuous, z-scored)
- Smoking: SMQ020 (binary: ever/never)
- Age: RIDAGEYR (continuous, z-scored)
- Diabetes: DIQ010 or HbA1c ≥ 6.5% (binary)
- Serum creatinine: LBXSCR → compute eGFR via CKD-EPI 2021

### Intercept Calibration
β₀ ≈ logit(0.15) ≈ −1.73

---

## Evaluation Metrics

For both endpoints:

1. **Discrimination:** AUC-ROC with 95% CI (DeLong method or bootstrap, 2000 replicates)
2. **Calibration:** Calibration slope and intercept (logistic recalibration); Hosmer-Lemeshow by decile
3. **Overall accuracy:** Brier score
4. **Comparison benchmarks:**
   - T2DM: Finnish Diabetes Risk Score (AUC 0.72–0.87), ADA Risk Test
   - CKD: Tangri 4-variable model (AUC 0.91 for progression, ~0.75–0.80 for prevalent CKD screening)

## Expected Results (Literature-Anchored)

| Metric | T2DM (expected) | CKD (expected) | Rationale |
|--------|-----------------|----------------|-----------|
| AUC-ROC | 0.70–0.76 | 0.72–0.78 | Zero-fit with few predictors; comparable to simple screening scores |
| Calibration slope | 0.85–1.10 | 0.80–1.05 | Intercept-calibrated; slopes near 1 expected |
| Brier score | 0.10–0.12 | 0.10–0.13 | Low-prevalence helps Brier |

## Limitations

1. **Cross-sectional design:** Cannot assess prospective prediction; validates association, not prediction
2. **Zero-fit weights:** KG weights are meta-analytic effect sizes, not optimized regression coefficients
3. **Limited predictors:** 4 predictors per endpoint vs. full clinical models (10+)
4. **NHANES sampling:** Complex survey design should use survey weights (svydesign)
5. **Prevalent vs. incident:** Detecting prevalent disease, not predicting future onset

## Proposed Paper Integration

Add to Section 5 (Evaluation) as Section 5.6: "T2DM and CKD Cross-Sectional Validation"  
Add results row to Table 5 (currently CVD-only validation metrics)
