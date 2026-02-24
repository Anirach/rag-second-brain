#!/usr/bin/env python3
"""Cohen's kappa analysis for NCD-CIE LLM validation — Experiment 5."""
import random, math
random.seed(42)

N = 107
GPT4_AGREE = 94
GPT4_DISAGREE = 13

# ========================================
# A) Agreement coefficients
# ========================================

p_o = GPT4_AGREE / N  # 0.8785

# --- Cohen's κ (standard) ---
# Expert: all 107 = "agree". GPT-4: 94 agree, 13 disagree.
# p_e = P(expert=agree)*P(gpt=agree) + P(expert=disagree)*P(gpt=disagree)
#      = 1.0 * (94/107) + 0.0 * (13/107) = 94/107 = p_o
# → κ = 0 (degenerate: one rater has zero marginal variance)
p_e_cohen = 1.0 * (94/N) + 0.0 * (13/N)
kappa_cohen = (p_o - p_e_cohen) / (1 - p_e_cohen) if p_e_cohen < 1 else 0

# --- Brennan-Prediger κ (chance = 1/q, q=categories) ---
# Binary (agree/disagree): q=2, p_e=0.5
kappa_bp2 = (p_o - 0.5) / (1 - 0.5)
# 4-category: q=4, p_e=0.25
kappa_bp4 = (p_o - 0.25) / (1 - 0.25)

# --- Gwet's AC1 (binary) ---
pi_hat = (1.0 + 94/N) / 2  # avg of both raters' P(agree)
p_e_gwet = 2 * pi_hat * (1 - pi_hat)
ac1 = (p_o - p_e_gwet) / (1 - p_e_gwet)

# --- Scott's π ---
p_agree_pooled = (N + GPT4_AGREE) / (2*N)
p_disagree_pooled = 1 - p_agree_pooled
p_e_scott = p_agree_pooled**2 + p_disagree_pooled**2
pi_scott = (p_o - p_e_scott) / (1 - p_e_scott)

# --- Multi-category Brennan-Prediger ---
kappa_bp4_obs = (94/N - 0.25) / (1 - 0.25)

# ========================================
# B) Baselines
# ========================================
baseline_majority = 1.0
baseline_random = 0.5

# ========================================
# C) Stratification by evidence grade
# ========================================
GRADE_A, GRADE_B, GRADE_C = 69, 28, 10
DA, DB, DC = 3, 6, 4
rate_a = (GRADE_A-DA)/GRADE_A
rate_b = (GRADE_B-DB)/GRADE_B
rate_c = (GRADE_C-DC)/GRADE_C

# ========================================
# D) Bootstrap 95% CI
# ========================================
data = [1]*GPT4_AGREE + [0]*GPT4_DISAGREE
n_boot = 10000
boot_agree = []
boot_bp2 = []
boot_ac1 = []
for _ in range(n_boot):
    s = random.choices(data, k=N)
    m = sum(s)/N
    boot_agree.append(m)
    boot_bp2.append((m - 0.5)/0.5)
    pi_h = (1.0 + m)/2
    pe_g = 2*pi_h*(1-pi_h)
    boot_ac1.append((m - pe_g)/(1 - pe_g) if pe_g < 1 else 1.0)

boot_agree.sort(); boot_bp2.sort(); boot_ac1.sort()
idx_lo, idx_hi = int(0.025*n_boot), int(0.975*n_boot)

# ========================================
# Output
# ========================================
print(f"=== Experiment 5: Cohen's κ Analysis ===\n")
print(f"Observed agreement: {p_o:.3f}")
print(f"Cohen's κ: {kappa_cohen:.3f} (degenerate)")
print(f"Gwet's AC1: {ac1:.3f}")
print(f"Brennan-Prediger κ (binary): {kappa_bp2:.3f}")
print(f"Brennan-Prediger κ (4-cat): {kappa_bp4_obs:.3f}")
print(f"Scott's π: {pi_scott:.3f}")
print(f"\nBootstrap 95% CI:")
print(f"  Agreement: [{boot_agree[idx_lo]*100:.1f}%, {boot_agree[idx_hi]*100:.1f}%]")
print(f"  AC1: [{boot_ac1[idx_lo]:.3f}, {boot_ac1[idx_hi]:.3f}]")
print(f"  BP κ: [{boot_bp2[idx_lo]:.3f}, {boot_bp2[idx_hi]:.3f}]")
print(f"\nStratification:")
print(f"  Grade A: {rate_a*100:.1f}%  Grade B: {rate_b*100:.1f}%  Grade C: {rate_c*100:.1f}%")
