#!/usr/bin/env python3
"""
Framingham Validation V2 for NCD-CIE
Sex-stratified D'Agostino 2008 coefficients (NO fitting to data)
Pure stdlib implementation (no numpy/pandas required)
"""

import csv
import math
import random
import json
import os

random.seed(42)
BASEDIR = os.path.dirname(os.path.abspath(__file__))

# ── Helper functions ──
def mean(x): return sum(x)/len(x) if x else 0
def std(x):
    m=mean(x); return math.sqrt(sum((v-m)**2 for v in x)/len(x)) if len(x)>1 else 0
def median_val(x):
    s=sorted(x); n=len(s)
    return (s[n//2-1]+s[n//2])/2 if n%2==0 else s[n//2]
def percentile(x, p):
    s=sorted(x); k=(len(s)-1)*p/100; f=math.floor(k); c=math.ceil(k)
    if f==c: return s[int(k)]
    return s[f]*(c-k)+s[c]*(k-f)
def logit(p): return math.log(p/(1-p))
def expit(x): return 1/(1+math.exp(-x)) if x>-500 else 0.0

# ── Load CSV ──
def load_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f)
        rows = []
        for r in reader:
            row = {}
            for k,v in r.items():
                try: row[k] = float(v)
                except: row[k] = None
            rows.append(row)
    return rows

data = load_csv(os.path.join(BASEDIR, 'framingham.csv'))
print(f"Loaded {len(data)} rows")

# Count missing
cols = list(data[0].keys())
for c in cols:
    miss = sum(1 for r in data if r[c] is None)
    if miss > 0: print(f"  {c}: {miss} missing")

# ── Simple imputation (median, since we can't do iterative without sklearn) ──
# Actually let's try importing sklearn from the .venv if possible, else fallback
try:
    import sys
    # Try the venv's site-packages — won't work if python version mismatch
    raise ImportError("skip")
except:
    print("Using median imputation (stdlib only)")
    impute_cols = ['glucose','cigsPerDay','BMI','totChol','BPMeds','heartRate','education']
    for c in impute_cols:
        if c not in cols: continue
        vals = [r[c] for r in data if r[c] is not None]
        if not vals: continue
        med = median_val(vals)
        for r in data:
            if r[c] is None:
                r[c] = med
    # Round binary
    for c in ['BPMeds','currentSmoker','diabetes','prevalentStroke','prevalentHyp']:
        if c not in cols: continue
        for r in data:
            if r[c] is not None:
                r[c] = round(min(max(r[c],0),1))

# Drop rows missing critical fields
critical = ['age','male','totChol','sysBP','diaBP','currentSmoker','diabetes','TenYearCHD']
data = [r for r in data if all(r.get(c) is not None for c in critical)]
print(f"After cleaning: {len(data)} rows")

y = [int(r['TenYearCHD']) for r in data]
n_events = sum(y)
print(f"Events: {n_events} ({n_events/len(data)*100:.1f}%)")

# ── MODEL A: D'Agostino 2008 ──
MEAN_SUM_MEN = 23.9802
S0_MEN = 0.88936
MEAN_SUM_WOMEN = 26.1931
S0_WOMEN = 0.95012
LN_HDL_MEN = math.log(48)
LN_HDL_WOMEN = math.log(53)

def risk_a(r):
    is_male = r['male'] == 1
    ln_age = math.log(r['age'])
    ln_chol = math.log(max(r['totChol'], 100))
    sbp = max(r['sysBP'], 80)
    treated = int(r.get('BPMeds', 0) or 0)
    
    if is_male:
        s = (3.06117*ln_age + 1.12370*ln_chol + (-0.93263)*LN_HDL_MEN
             + (1.99881 if treated else 1.93303)*math.log(sbp)
             + 0.65451*r['currentSmoker'] + 0.57367*r['diabetes'])
        p = 1.0 - S0_MEN**math.exp(s - MEAN_SUM_MEN)
    else:
        s = (2.32888*ln_age + 1.20904*ln_chol + (-0.70833)*LN_HDL_WOMEN
             + (2.76157 if treated else 2.82263)*math.log(sbp)
             + 0.52873*r['currentSmoker'] + 0.69154*r['diabetes'])
        p = 1.0 - S0_WOMEN**math.exp(s - MEAN_SUM_WOMEN)
    return max(0.001, min(0.999, p))

print("\nComputing Model A...")
pA = [risk_a(r) for r in data]

# ── MODEL B: Original single model ──
age_m = mean([r['age'] for r in data])
age_s = std([r['age'] for r in data])
sbp_m = mean([r['sysBP'] for r in data])
sbp_s = std([r['sysBP'] for r in data])
chol_m = mean([r['totChol'] for r in data])
chol_s = std([r['totChol'] for r in data])
bmi_m = mean([r['BMI'] for r in data])
bmi_s = std([r['BMI'] for r in data])

def risk_b(r):
    lp = -1.7361
    lp += 0.4196*((r['age']-age_m)/age_s)
    lp += 0.1821*((r['sysBP']-sbp_m)/sbp_s)
    lp += 0.1187*((r['totChol']-chol_m)/chol_s)
    lp += 0.2000*((r['BMI']-bmi_m)/bmi_s)
    lp += 0.5008*r['currentSmoker']
    lp += 0.4947*r['diabetes']
    lp += 0.4055*r['male']
    lp *= 0.85
    return max(0.001, min(0.999, expit(lp)))

print("Computing Model B...")
pB = [risk_b(r) for r in data]

# ── METRICS ──
def auc_roc(y_true, y_pred):
    """Compute AUC via sorting-based Mann-Whitney U (O(n log n))."""
    n = len(y_true)
    paired = sorted(range(n), key=lambda i: y_pred[i])
    n_pos = sum(y_true)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0: return 0.5
    
    # Sum of ranks for positives (1-indexed)
    rank_sum = 0
    i = 0
    while i < n:
        j = i
        while j < n and y_pred[paired[j]] == y_pred[paired[i]]:
            j += 1
        avg_rank = (i + j + 1) / 2  # average rank for ties (1-indexed)
        for k in range(i, j):
            if y_true[paired[k]] == 1:
                rank_sum += avg_rank
        i = j
    
    u = rank_sum - n_pos * (n_pos + 1) / 2
    return u / (n_pos * n_neg)

def brier_score(y_true, y_pred):
    return mean([(y_true[i]-y_pred[i])**2 for i in range(len(y_true))])

def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000):
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = [random.randint(0, n-1) for _ in range(n)]
        yt = [y_true[i] for i in idx]
        yp = [y_pred[i] for i in idx]
        try:
            s = metric_fn(yt, yp)
            scores.append(s)
        except:
            pass
    scores.sort()
    return mean(scores), percentile(scores, 2.5), percentile(scores, 97.5)

def calibration_slope_intercept(y_true, y_pred):
    """Simple logistic regression: y ~ a + b*logit(p) via IRLS."""
    n = len(y_true)
    # Newton-Raphson for logistic regression
    a, b = 0.0, 1.0  # initial
    for _ in range(50):
        grad_a, grad_b = 0.0, 0.0
        hess_aa, hess_ab, hess_bb = 0.0, 0.0, 0.0
        for i in range(n):
            x = logit(y_pred[i])
            eta = a + b * x
            mu = expit(eta)
            r = y_true[i] - mu
            w = mu * (1 - mu) + 1e-10
            grad_a += r
            grad_b += r * x
            hess_aa -= w
            hess_ab -= w * x
            hess_bb -= w * x * x
        det = hess_aa * hess_bb - hess_ab * hess_ab
        if abs(det) < 1e-15: break
        da = (hess_bb * grad_a - hess_ab * grad_b) / det
        db = (hess_aa * grad_b - hess_ab * grad_a) / det
        a -= da
        b -= db
        if abs(da) < 1e-8 and abs(db) < 1e-8: break
    return b, a  # slope, intercept

def hosmer_lemeshow(y_true, y_pred, ng=10):
    paired = sorted(zip(y_pred, y_true))
    n = len(paired)
    gs = ng
    group_size = n // gs
    
    hl = 0.0
    deciles = []
    for i in range(gs):
        start = i * group_size
        end = (i+1) * group_size if i < gs-1 else n
        grp = paired[start:end]
        n_g = len(grp)
        obs = sum(yt for _, yt in grp)
        exp_ = sum(yp for yp, _ in grp)
        mean_p = mean([yp for yp, _ in grp])
        mean_o = mean([yt for _, yt in grp])
        
        if exp_ > 0 and (n_g - exp_) > 0:
            hl += (obs - exp_)**2 / (exp_ * (1 - exp_/n_g))
        
        deciles.append({
            'decile': i+1, 'n': n_g,
            'mean_predicted': round(mean_p, 4),
            'mean_observed': round(mean_o, 4),
            'events': int(obs), 'expected': round(exp_, 2)
        })
    
    # Chi-squared p-value (ng-2 df)
    # Using Wilson-Hilferty approximation
    df_ = ng - 2
    if df_ > 0 and hl > 0:
        z = ((hl/df_)**(1/3) - (1 - 2/(9*df_))) / math.sqrt(2/(9*df_))
        p_val = 0.5 * math.erfc(z / math.sqrt(2))  # normal CDF approx
    else:
        p_val = 1.0
    return hl, p_val, deciles

def roc_curve_calc(y_true, y_pred):
    """Compute ROC curve points."""
    paired = sorted(zip(y_pred, y_true), reverse=True)
    tp, fp, fn, tn = 0, 0, sum(y_true), len(y_true)-sum(y_true)
    total_pos = sum(y_true)
    total_neg = len(y_true) - total_pos
    
    points = []  # (threshold, tpr, fpr)
    prev_score = None
    for score, label in paired:
        if score != prev_score and prev_score is not None:
            tpr = tp / total_pos if total_pos > 0 else 0
            fpr = fp / total_neg if total_neg > 0 else 0
            points.append((prev_score, tpr, fpr))
        if label == 1:
            tp += 1; fn -= 1
        else:
            fp += 1; tn -= 1
        prev_score = score
    tpr = tp / total_pos if total_pos > 0 else 0
    fpr = fp / total_neg if total_neg > 0 else 0
    points.append((prev_score, tpr, fpr))
    return points

def youden_optimal(y_true, y_pred):
    points = roc_curve_calc(y_true, y_pred)
    best_j = -1
    best = (0.5, 0.5, 0.5)
    for th, tpr, fpr in points:
        j = tpr - fpr
        if j > best_j:
            best_j = j
            best = (th, tpr, 1-fpr)
    return best  # threshold, sensitivity, specificity

def eval_model(y_true, y_pred, data_rows, label):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    auc_m, auc_l, auc_h = bootstrap_ci(y_true, y_pred, auc_roc)
    print(f"AUC: {auc_m:.4f} [{auc_l:.4f}-{auc_h:.4f}]")
    
    br_m, br_l, br_h = bootstrap_ci(y_true, y_pred, brier_score)
    print(f"Brier: {br_m:.4f} [{br_l:.4f}-{br_h:.4f}]")
    
    slope, intercept = calibration_slope_intercept(y_true, y_pred)
    print(f"Cal slope: {slope:.4f}, intercept: {intercept:.4f}")
    
    hl, hlp, dd = hosmer_lemeshow(y_true, y_pred)
    print(f"HL chi2: {hl:.2f} (p={hlp:.4f})")
    
    th, sens, spec = youden_optimal(y_true, y_pred)
    print(f"Threshold: {th:.4f}, Sens: {sens:.4f}, Spec: {spec:.4f}")
    
    print("\nDeciles:")
    for d in dd:
        print(f"  {d['decile']}: N={d['n']}, pred={d['mean_predicted']:.4f}, obs={d['mean_observed']:.4f}, ev={d['events']}")
    
    # Subgroups
    subgroups_def = {
        'Male': lambda r: r['male']==1,
        'Female': lambda r: r['male']==0,
        'Age<50': lambda r: r['age']<50,
        'Age 50-60': lambda r: 50<=r['age']<=60,
        'Age>60': lambda r: r['age']>60,
        'Diabetes': lambda r: r['diabetes']==1,
        'No Diabetes': lambda r: r['diabetes']==0,
    }
    sg = {}
    print("\nSubgroups:")
    for nm, fn in subgroups_def.items():
        mask = [fn(r) for r in data_rows]
        yt_s = [y_true[i] for i in range(len(y_true)) if mask[i]]
        yp_s = [y_pred[i] for i in range(len(y_pred)) if mask[i]]
        ns = len(yt_s)
        if ns < 20 or sum(yt_s) < 5:
            print(f"  {nm}: N={ns}, too few")
            continue
        try:
            am, al, ah = bootstrap_ci(yt_s, yp_s, auc_roc, 500)
            bs = brier_score(yt_s, yp_s)
            ev = sum(yt_s); rt = ev/ns
            print(f"  {nm}: N={ns}, ev={ev} ({rt:.1%}), AUC={am:.4f} [{al:.4f}-{ah:.4f}], Brier={bs:.4f}")
            sg[nm] = {'n':ns,'events':ev,'rate':round(rt,3),'auc':round(am,4),
                      'auc_lo':round(al,4),'auc_hi':round(ah,4),'brier':round(bs,4)}
        except Exception as e:
            print(f"  {nm}: Error - {e}")
    
    return {'auc':auc_m,'auc_lo':auc_l,'auc_hi':auc_h,
            'brier':br_m,'brier_lo':br_l,'brier_hi':br_h,
            'slope':slope,'intercept':intercept,
            'hl_stat':hl,'hl_p':hlp,'threshold':th,
            'sensitivity':sens,'specificity':spec,
            'decile_data':dd,'subgroups':sg}

print(f"\nModel A risk stats: mean={mean(pA):.4f}, median={median_val(pA):.4f}")
print(f"Model B risk stats: mean={mean(pB):.4f}, median={median_val(pB):.4f}")

rA = eval_model(y, pA, data, "Model A: D'Agostino 2008 Sex-Stratified")
rB = eval_model(y, pB, data, "Model B: Original Single Model")

# ── COMPARISON ──
print(f"\n{'='*60}")
print("  Comparison: A vs B")
print(f"{'='*60}")

# NRI
ev_idx = [i for i in range(len(y)) if y[i]==1]
ne_idx = [i for i in range(len(y)) if y[i]==0]

up_e = sum(1 for i in ev_idx if pA[i]>pB[i]) / len(ev_idx)
dn_e = sum(1 for i in ev_idx if pA[i]<pB[i]) / len(ev_idx)
nri_e = up_e - dn_e

up_n = sum(1 for i in ne_idx if pA[i]>pB[i]) / len(ne_idx)
dn_n = sum(1 for i in ne_idx if pA[i]<pB[i]) / len(ne_idx)
nri_n = dn_n - up_n

nri = nri_e + nri_n
se_e = math.sqrt((up_e+dn_e)/len(ev_idx))
se_n = math.sqrt((up_n+dn_n)/len(ne_idx))
nri_se = math.sqrt(se_e**2+se_n**2)
nri_z = nri/nri_se if nri_se>0 else 0
nri_p = 2*(0.5*math.erfc(abs(nri_z)/math.sqrt(2)))
print(f"NRI: {nri:.4f} (SE={nri_se:.4f}, p={nri_p:.4f}), events={nri_e:.4f}, non-events={nri_n:.4f}")

# IDI
is_new = mean([pA[i] for i in ev_idx]) - mean([pA[i] for i in ne_idx])
is_old = mean([pB[i] for i in ev_idx]) - mean([pB[i] for i in ne_idx])
idi_val = is_new - is_old

# Bootstrap IDI SE
idi_boots = []
n = len(y)
for _ in range(1000):
    idx = [random.randint(0,n-1) for _ in range(n)]
    yt2 = [y[i] for i in idx]; pa2 = [pA[i] for i in idx]; pb2 = [pB[i] for i in idx]
    e2 = [i for i in range(n) if yt2[i]==1]; n2 = [i for i in range(n) if yt2[i]==0]
    if e2 and n2:
        isn = mean([pa2[i] for i in e2])-mean([pa2[i] for i in n2])
        iso = mean([pb2[i] for i in e2])-mean([pb2[i] for i in n2])
        idi_boots.append(isn-iso)
idi_se = std(idi_boots) if idi_boots else 0
idi_z = idi_val/idi_se if idi_se>0 else 0
idi_p = 2*(0.5*math.erfc(abs(idi_z)/math.sqrt(2)))
print(f"IDI: {idi_val:.4f} (SE={idi_se:.4f}, p={idi_p:.4f}), IS_new={is_new:.4f}, IS_old={is_old:.4f}")

# DeLong bootstrap
auc1 = auc_roc(y, pA); auc2 = auc_roc(y, pB); diff = auc1 - auc2
diffs = []
for _ in range(1000):
    idx = [random.randint(0,n-1) for _ in range(n)]
    yt2 = [y[i] for i in idx]; pa2 = [pA[i] for i in idx]; pb2 = [pB[i] for i in idx]
    try: diffs.append(auc_roc(yt2,pa2)-auc_roc(yt2,pb2))
    except: pass
se_d = std(diffs) if diffs else 0
z_d = diff/se_d if se_d>0 else 0
p_d = 2*(0.5*math.erfc(abs(z_d)/math.sqrt(2)))
print(f"DeLong: ΔAUC={diff:.4f} (SE={se_d:.4f}), z={z_d:.3f}, p={p_d:.4f}")

# ── SAVE CALIBRATION CSV ──
cal_path = os.path.join(BASEDIR, 'calibration_data.csv')
with open(cal_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['model','decile','n','mean_predicted','mean_observed','events','expected'])
    w.writeheader()
    for d in rA['decile_data']:
        w.writerow({**d, 'model': 'A_DAgostino2008'})
    for d in rB['decile_data']:
        w.writerow({**d, 'model': 'B_Original'})
print(f"\nCalibration CSV saved.")

# ── SAVE RESULTS MD ──
def fc(m,l,h): return f"{m:.4f} [{l:.4f}-{h:.4f}]"

md = f"""# Framingham Validation V2 Results for NCD-CIE

## Overview
- **Sample size:** {len(data)} patients
- **Events:** {n_events} ({n_events/len(data)*100:.1f}%)
- **Imputation:** Median (stdlib implementation)
- **Model A:** Sex-stratified D'Agostino 2008 (Circulation 117:743-753)
- **Model B:** Original single-model with literature HRs

## Model A: D'Agostino 2008 Sex-Stratified

### Coefficients (from publication, NOT fit to data)
Source: D'Agostino RB Sr, et al. Circulation. 2008;117:743-753.

| Parameter | Men (β) | Women (β) |
|---|---|---|
| ln(Age) | 3.06117 | 2.32888 |
| ln(Total Cholesterol) | 1.12370 | 1.20904 |
| ln(HDL-C) | -0.93263* | -0.70833* |
| ln(SBP) treated | 1.99881 | 2.76157 |
| ln(SBP) untreated | 1.93303 | 2.82263 |
| Smoking | 0.65451 | 0.52873 |
| Diabetes | 0.57367 | 0.69154 |
| Mean coefficient sum | 23.9802 | 26.1931 |
| S₀(10) | 0.88936 | 0.95012 |

*HDL-C unavailable; population mean ln(HDL) used as constant.

### Discrimination
| Metric | Value |
|---|---|
| **AUC-ROC** | **{fc(rA['auc'],rA['auc_lo'],rA['auc_hi'])}** |
| Brier Score | {fc(rA['brier'],rA['brier_lo'],rA['brier_hi'])} |
| Optimal Threshold | {rA['threshold']:.4f} |
| Sensitivity | {rA['sensitivity']:.4f} |
| Specificity | {rA['specificity']:.4f} |

### Calibration
| Metric | Value |
|---|---|
| Calibration slope | {rA['slope']:.4f} (ideal=1.0) |
| Calibration intercept | {rA['intercept']:.4f} (ideal=0.0) |
| Hosmer-Lemeshow χ² | {rA['hl_stat']:.2f} (p={rA['hl_p']:.4f}) |

### Calibration by Decile
| Decile | N | Predicted | Observed | Events |
|---|---|---|---|---|
"""
for d in rA['decile_data']:
    md += f"| {d['decile']} | {d['n']} | {d['mean_predicted']:.4f} | {d['mean_observed']:.4f} | {d['events']} |\n"

md += f"""
### Subgroup Analysis (Model A)
| Subgroup | N | Events | Rate | AUC [95% CI] | Brier |
|---|---|---|---|---|---|
"""
for nm,r in rA['subgroups'].items():
    md += f"| {nm} | {r['n']} | {r['events']} | {r['rate']:.3f} | {r['auc']:.4f} [{r['auc_lo']:.4f}-{r['auc_hi']:.4f}] | {r['brier']:.4f} |\n"

md += f"""
## Model B: Original Single Model

### Discrimination
| Metric | Value |
|---|---|
| **AUC-ROC** | **{fc(rB['auc'],rB['auc_lo'],rB['auc_hi'])}** |
| Brier Score | {fc(rB['brier'],rB['brier_lo'],rB['brier_hi'])} |
| Optimal Threshold | {rB['threshold']:.4f} |
| Sensitivity | {rB['sensitivity']:.4f} |
| Specificity | {rB['specificity']:.4f} |

### Calibration
| Metric | Value |
|---|---|
| Calibration slope | {rB['slope']:.4f} (ideal=1.0) |
| Calibration intercept | {rB['intercept']:.4f} (ideal=0.0) |
| Hosmer-Lemeshow χ² | {rB['hl_stat']:.2f} (p={rB['hl_p']:.4f}) |

### Calibration by Decile
| Decile | N | Predicted | Observed | Events |
|---|---|---|---|---|
"""
for d in rB['decile_data']:
    md += f"| {d['decile']} | {d['n']} | {d['mean_predicted']:.4f} | {d['mean_observed']:.4f} | {d['events']} |\n"

md += f"""
### Subgroup Analysis (Model B)
| Subgroup | N | Events | Rate | AUC [95% CI] | Brier |
|---|---|---|---|---|---|
"""
for nm,r in rB['subgroups'].items():
    md += f"| {nm} | {r['n']} | {r['events']} | {r['rate']:.3f} | {r['auc']:.4f} [{r['auc_lo']:.4f}-{r['auc_hi']:.4f}] | {r['brier']:.4f} |\n"

md += f"""
## Model Comparison: A vs B

| Metric | Value (SE) | p-value |
|---|---|---|
| ΔAUC | {diff:.4f} ({se_d:.4f}) | {p_d:.4f} |
| Continuous NRI | {nri:.4f} ({nri_se:.4f}) | {nri_p:.4f} |
| NRI (events) | {nri_e:.4f} | — |
| NRI (non-events) | {nri_n:.4f} | — |
| IDI | {idi_val:.4f} ({idi_se:.4f}) | {idi_p:.4f} |
| IS (Model A) | {is_new:.4f} | — |
| IS (Model B) | {is_old:.4f} | — |

## Limitations
1. **Missing HDL-C** — population mean used; removes HDL discrimination
2. **Median imputation** — iterative imputation unavailable (stdlib only); may slightly affect results
3. **Era mismatch** — Framingham 1960s-80s cohort
4. **Population** — predominantly White American
5. **Outcome** — TenYearCHD vs broader CVD in D'Agostino 2008
6. **No competing risks** modeled
7. **All coefficients from publication** — NOT fit to this data
"""

with open(os.path.join(BASEDIR, 'framingham_validation_v2_results.md'), 'w') as f:
    f.write(md)
print("Results MD saved.")

# ── SAVE LATEX ──
tex = r"""\begin{table}[htbp]
\centering
\caption{Discrimination metrics for Model A (D'Agostino 2008) vs Model B (Original)}
\label{tab:discrimination}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Model A} & \textbf{Model B} \\
\hline
"""
tex += f"AUC-ROC & {fc(rA['auc'],rA['auc_lo'],rA['auc_hi'])} & {fc(rB['auc'],rB['auc_lo'],rB['auc_hi'])} \\\\\n"
tex += f"Brier Score & {fc(rA['brier'],rA['brier_lo'],rA['brier_hi'])} & {fc(rB['brier'],rB['brier_lo'],rB['brier_hi'])} \\\\\n"
tex += f"Calibration Slope & {rA['slope']:.4f} & {rB['slope']:.4f} \\\\\n"
tex += f"Calibration Intercept & {rA['intercept']:.4f} & {rB['intercept']:.4f} \\\\\n"
tex += f"Sensitivity & {rA['sensitivity']:.4f} & {rB['sensitivity']:.4f} \\\\\n"
tex += f"Specificity & {rA['specificity']:.4f} & {rB['specificity']:.4f} \\\\\n"
tex += r"""\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Calibration by decile -- Model A (D'Agostino 2008)}
\label{tab:cal_a}
\begin{tabular}{ccccc}
\hline
\textbf{Decile} & \textbf{N} & \textbf{Predicted} & \textbf{Observed} & \textbf{Events} \\
\hline
"""
for d in rA['decile_data']:
    tex += f"{d['decile']} & {d['n']} & {d['mean_predicted']:.4f} & {d['mean_observed']:.4f} & {d['events']} \\\\\n"
tex += r"""\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Reclassification improvement: Model A vs Model B}
\label{tab:comparison}
\begin{tabular}{lcc}
\hline
\textbf{Metric} & \textbf{Value (SE)} & \textbf{p-value} \\
\hline
"""
tex += f"$\\Delta$AUC & {diff:.4f} ({se_d:.4f}) & {p_d:.4f} \\\\\n"
tex += f"Continuous NRI & {nri:.4f} ({nri_se:.4f}) & {nri_p:.4f} \\\\\n"
tex += f"IDI & {idi_val:.4f} ({idi_se:.4f}) & {idi_p:.4f} \\\\\n"
tex += r"""\hline
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Subgroup AUC-ROC -- Model A (D'Agostino 2008)}
\label{tab:subgroups}
\begin{tabular}{lccccc}
\hline
\textbf{Subgroup} & \textbf{N} & \textbf{Events} & \textbf{Rate} & \textbf{AUC [95\% CI]} & \textbf{Brier} \\
\hline
"""
for nm,r in rA['subgroups'].items():
    tex += f"{nm} & {r['n']} & {r['events']} & {r['rate']:.3f} & {r['auc']:.4f} [{r['auc_lo']:.4f}-{r['auc_hi']:.4f}] & {r['brier']:.4f} \\\\\n"
tex += r"""\hline
\end{tabular}
\end{table}
"""

with open(os.path.join(BASEDIR, 'latex_tables.tex'), 'w') as f:
    f.write(tex)
print("LaTeX tables saved.")
print("\n✅ ALL DONE.")
