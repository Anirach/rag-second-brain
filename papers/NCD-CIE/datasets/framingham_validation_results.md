# Framingham Validation Results for NCD-CIE Risk Engine

## Dataset
- **Source:** Framingham Heart Study (public dataset)
- **Sample size:** 4172 patients (after exclusions)
- **10-year CHD events:** 625 (15.0%)
- **Exclusions:** Dropped 68 rows with missing required fields

## Model Specification
Published hazard ratios (NOT fit to this data):

| Risk Factor | HR (Source) | beta per SD |
|---|---|---|
| SBP | 1.18/20mmHg (D'Agostino 2008) | 0.1821 |
| Total Cholesterol | 1.11/mmol/L (PSC 2007) | 0.1187 |
| BMI | 1.05/kg/m2 (GBM 2016) | 0.2000 |
| Age | 1.05/year (Framingham) | 0.4196 |
| Smoking | 1.65 binary (Thun 2013) | 0.5008 |
| Diabetes | 1.64 binary (ERFC 2010) | 0.4947 |
| Male sex | 1.50 binary (Framingham) | 0.4055 |

Attenuation alpha=0.85, Intercept beta_0=-1.7361

## Discrimination
| Metric | Value |
|---|---|
| **AUC-ROC** | **0.7228** [0.7027-0.7425] |
| Brier Score | 0.1214 [0.1160-0.1270] |
| Optimal Threshold | 0.2300 |
| Sensitivity | 0.6864 |
| Specificity | 0.6501 |

## Calibration
| Metric | Value |
|---|---|
| Calibration slope | 1.4767 (ideal=1.0) |
| Calibration intercept | 0.0404 (ideal=0.0) |
| Hosmer-Lemeshow chi2 | 146.64 (p=0.0000) |

### Calibration by Decile
| Decile | N | Mean Predicted | Mean Observed | Events |
|---|---|---|---|---|
| 1 | 417 | 0.0866 | 0.0288 | 12 |
| 2 | 417 | 0.1202 | 0.0528 | 22 |
| 3 | 417 | 0.1445 | 0.0528 | 22 |
| 4 | 417 | 0.1685 | 0.0791 | 33 |
| 5 | 417 | 0.1923 | 0.1367 | 57 |
| 6 | 417 | 0.2169 | 0.1199 | 50 |
| 7 | 417 | 0.2440 | 0.1942 | 81 |
| 8 | 417 | 0.2774 | 0.2014 | 84 |
| 9 | 417 | 0.3214 | 0.2638 | 110 |
| 10 | 419 | 0.4130 | 0.3675 | 154 |

## Subgroup Analysis
| Subgroup | N | Events | Rate | AUC-ROC [95% CI] | Brier |
|---|---|---|---|---|---|
| Male | 1808 | 339 | 0.188 | 0.7096 [0.6778-0.7399] | 0.1442 |
| Female | 2364 | 286 | 0.121 | 0.7190 [0.6843-0.7469] | 0.1040 |
| Age <50 | 2187 | 181 | 0.083 | 0.6873 [0.6458-0.7317] | 0.0791 |
| Age 50-60 | 1425 | 285 | 0.200 | 0.6542 [0.6215-0.6884] | 0.1549 |
| Age >60 | 560 | 159 | 0.284 | 0.5906 [0.5310-0.6402] | 0.2013 |
| Diabetes | 106 | 37 | 0.349 | 0.6212 [0.5155-0.7373] | 0.2176 |
| No Diabetes | 4066 | 588 | 0.145 | 0.7187 [0.6967-0.7388] | 0.1189 |

## Limitations
1. **Missing predictors:** LDL-C, HDL-C, triglycerides, HbA1c, physical activity unavailable. Full NCD-CIE uses ~50 risk factors; this validates only 7.
2. **Era mismatch:** Framingham from 1960s-1980s cohort; treatment patterns differ from modern populations.
3. **Population:** Predominantly White American; may not generalize.
4. **Intercept calibrated** to observed base rate (standard external validation practice).
5. **Outcome:** TenYearCHD (MI, fatal CHD, angina) vs broader CVD endpoint in NCD-CIE.
6. **No competing risks** modeled.

## Interpretation
The AUC of 0.723 indicates good discrimination, comparable to the original Framingham Risk Score (C-statistic ~0.75). The calibration slope of 1.48 suggests poor calibration. This is a *partial* validation using only 7 of ~50 risk factors with purely literature-derived HRs (not fit to data).
