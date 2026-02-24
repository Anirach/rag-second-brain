# NCD-CIE Validation Summary Report

## 1. Framingham Heart Study — Risk Prediction Validation

- **Dataset:** 4240 patients, 644 CHD+ (15.2%)
- **NCD-CIE KG-only AUC:** 0.6723
- **NCD-CIE Blended AUC:** 0.6878
- **Logistic Regression AUC:** 0.7240
- **SCORE2 reference AUC:** 0.704
- **D'Agostino reference AUC:** 0.721

### Intervention Simulations
- **statin:** 969 eligible, risk 0.572→0.512 (actual CHD=0.264)
- **bp_medication:** 868 eligible, risk 0.577→0.532 (actual CHD=0.273)
- **smoking_cessation:** 747 eligible, risk 0.581→0.541 (actual CHD=0.257)

## 2. Diabetes 130 US Hospitals — What-If Intervention Validation

- **Dataset:** 101766 encounters, 71518 patients
- **Multi-visit patients:** 16773
- **Medication changes detected:** 25968
- **KG Risk AUC (any readmission):** 0.6141
- **KG Risk AUC (30d readmission):** 0.5962

### Medication Intervention Results
| Medication | N | Observed Δ | Predicted Δ | Direction |
|-----------|---|-----------|------------|-----------|
| metformin | 1819 | -0.4255 | -0.0800 | ✓ |
| insulin | 3922 | -0.4052 | -0.0500 | ✓ |
| glipizide | 1121 | -0.3747 | -0.0400 | ✓ |
| glyburide | 863 | -0.4299 | -0.0300 | ✓ |
| pioglitazone | 727 | -0.4099 | -0.0350 | ✓ |
| rosiglitazone | 565 | -0.3664 | -0.0300 | ✓ |
| glimepiride | 518 | -0.3649 | -0.0350 | ✓ |
| acarbose | 42 | -0.1905 | -0.0200 | ✓ |
| repaglinide | 286 | -0.3986 | -0.0250 | ✓ |
| nateglinide | 92 | -0.2935 | -0.0200 | ✓ |

- **Overall correlation:** r=0.544, p=0.104

## 3. Generated Artifacts

- `results/framingham_roc.png` — ROC curves
- `results/framingham_calibration.png` — Calibration curves
- `results/diabetes130_interventions.png` — Intervention comparison
- `results/framingham_results.json` — Full Framingham metrics
- `results/diabetes130_results.json` — Full Diabetes 130 metrics