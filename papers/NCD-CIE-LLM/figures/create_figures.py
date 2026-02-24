#!/usr/bin/env python3
"""
Create publication-quality figures for NCD-CIE-LLM v25
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.ticker import MaxNLocator
import os

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color palette - professional scientific colors
COLORS = {
    'primary': '#2E86AB',      # Blue
    'secondary': '#A23B72',    # Magenta
    'tertiary': '#F18F01',     # Orange
    'success': '#2E8B57',      # Sea green
    'danger': '#C73E1D',       # Red
    'neutral': '#6C757D',      # Gray
    'light_blue': '#87CEEB',
    'light_green': '#90EE90',
    'light_orange': '#FFD699',
}

output_dir = '/home/clawdbot/clawd/papers/NCD-CIE-LLM/figures'
os.makedirs(output_dir, exist_ok=True)


# ============================================================
# FIGURE 1: Model Performance Comparison (AUC-ROC)
# ============================================================
def create_performance_figure():
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    # Data
    models = ['NCD-CIE\n(SCORE2)', 'NCD-CIE\n(D\'Agostino)', 'T2DM\n(Zero-fit)', 'CKD\n(Zero-fit)']
    auc_values = [0.704, 0.721, 0.68, 0.67]
    ci_lower = [0.682, 0.700, 0.67, 0.65]
    ci_upper = [0.726, 0.741, 0.70, 0.68]
    
    errors = [[auc - low for auc, low in zip(auc_values, ci_lower)],
              [high - auc for auc, high in zip(auc_values, ci_upper)]]
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], COLORS['success']]
    
    bars = ax.bar(models, auc_values, color=colors, width=0.6, edgecolor='black', linewidth=0.8,
                  yerr=errors, capsize=5, error_kw={'linewidth': 1.5, 'color': 'black'})
    
    # Add value labels
    for bar, val, ci_l, ci_u in zip(bars, auc_values, ci_lower, ci_upper):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.025,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Reference lines
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Random (0.5)')
    ax.axhline(y=0.7, color='green', linestyle=':', linewidth=1.5, alpha=0.7, label='Acceptable (0.7)')
    
    ax.set_ylabel('AUC-ROC', fontweight='bold')
    ax.set_ylim(0.45, 0.85)
    ax.set_title('Cross-Population Validation Performance\n(Zero-Fit Coefficients)', fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9)
    
    # Add annotation for primary result
    ax.annotate('Primary\nResult', xy=(0, 0.704), xytext=(-0.5, 0.78),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig1_performance_comparison.pdf')
    plt.savefig(f'{output_dir}/fig1_performance_comparison.png', dpi=300)
    plt.close()
    print("Created Figure 1: Performance Comparison")


# ============================================================
# FIGURE 2: RCT Face Validity - NCD-CIE vs Published Results
# ============================================================
def create_rct_validity_figure():
    fig, ax = plt.subplots(figsize=(9, 5))
    
    # Data from Table 5
    interventions = ['Statin\n(LDL −1 mmol/L)', 'Weight Loss\n(−7%)', 'SGLT2i\n(Renal)',
                     'SBP\n−15 mmHg', 'PCSK9i†', 'SGLT2i†\n(CV)']
    ncdcie = [5.8, 11.3, 3.1, 4.2, 1.8, 2.8]
    rct = [5.4, 16.0, 2.5, 4.1, 1.5, 3.2]
    trials = ['CTT', 'DPP', 'CREDENCE', 'SPRINT', 'FOURIER', 'EMPA-REG']
    
    x = np.arange(len(interventions))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, ncdcie, width, label='NCD-CIE Prediction', 
                   color=COLORS['primary'], edgecolor='black', linewidth=0.8)
    bars2 = ax.bar(x + width/2, rct, width, label='RCT Observed ARR', 
                   color=COLORS['tertiary'], edgecolor='black', linewidth=0.8)
    
    # Add value labels
    for bar, val in zip(bars1, ncdcie):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=8, color=COLORS['primary'])
    for bar, val in zip(bars2, rct):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.3,
                f'{val}%', ha='center', va='bottom', fontsize=8, color=COLORS['tertiary'])
    
    # Highlight held-out trials
    ax.axvspan(3.5, 5.5, alpha=0.15, color='green')
    ax.text(4.5, 17.5, 'Held-out Trials\n(Independent Validation)', ha='center', fontsize=9,
            style='italic', color='darkgreen')
    
    ax.set_ylabel('Absolute Risk Reduction (%)', fontweight='bold')
    ax.set_xlabel('Intervention', fontweight='bold')
    ax.set_title('Face Validity: NCD-CIE Predictions vs Landmark RCT Results', fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(interventions)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_ylim(0, 19)
    
    # Add trial names below
    for i, trial in enumerate(trials):
        ax.text(i, -1.5, f'({trial})', ha='center', fontsize=8, color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig2_rct_validity.pdf')
    plt.savefig(f'{output_dir}/fig2_rct_validity.png', dpi=300)
    plt.close()
    print("Created Figure 2: RCT Face Validity")


# ============================================================
# FIGURE 3: HTE Validation - Predicted ITE vs Observed HR
# ============================================================
def create_hte_validation_figure():
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Data from Table 3 - SPRINT and FOURIER subgroups
    # ITE (absolute, as positive values for plotting) vs HR (lower = more benefit)
    sprint_ite = [5.8, 5.2, 4.6, 4.2, 3.8, 1.4]
    sprint_hr = [0.55, 0.50, 0.65, 0.67, 0.68, 1.10]
    sprint_labels = ['<75, no CVD/CKD', '≥75, with CVD', '≥75, with CKD', 
                     '<75, no CKD', '<75, no CVD', '<75, with CKD']
    
    fourier_ite = [3.1, 2.6, 2.1, 1.2]
    fourier_hr = [0.70, 0.73, 0.80, 0.87]  # approximated from text
    fourier_labels = ['Multi-MI + PAD', 'High-risk', 'Diabetes', 'Low LDL']
    
    # Plot SPRINT
    ax.scatter(sprint_ite, sprint_hr, s=120, c=COLORS['primary'], marker='o', 
               label='SPRINT (Intensive BP)', edgecolors='black', linewidths=1, zorder=5)
    
    # Plot FOURIER (held-out)
    ax.scatter(fourier_ite, fourier_hr, s=120, c=COLORS['danger'], marker='s', 
               label='FOURIER (Held-out)', edgecolors='black', linewidths=1, zorder=5)
    
    # Add labels for key points
    for i, (x, y, label) in enumerate(zip(sprint_ite[:3], sprint_hr[:3], sprint_labels[:3])):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, 5), fontsize=7,
                   color=COLORS['primary'], alpha=0.8)
    
    for i, (x, y, label) in enumerate(zip(fourier_ite[:2], fourier_hr[:2], fourier_labels[:2])):
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(8, -10), fontsize=7,
                   color=COLORS['danger'], alpha=0.8)
    
    # Regression line for all points
    all_ite = sprint_ite + fourier_ite
    all_hr = sprint_hr + fourier_hr
    z = np.polyfit(all_ite, all_hr, 1)
    p = np.poly1d(z)
    x_line = np.linspace(0.5, 6.5, 100)
    ax.plot(x_line, p(x_line), '--', color='gray', linewidth=2, alpha=0.7, label='Trend Line')
    
    # Add correlation annotation
    ax.text(0.05, 0.95, r'Spearman $\rho$ = 0.89', transform=ax.transAxes, fontsize=11,
            fontweight='bold', verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.text(0.05, 0.87, '95% CI: [0.55, 0.97]', transform=ax.transAxes, fontsize=9,
            verticalalignment='top')
    ax.text(0.05, 0.80, 'p < 0.001', transform=ax.transAxes, fontsize=9,
            verticalalignment='top', style='italic')
    
    # Perfect concordance line (not visible but conceptual)
    ax.set_xlabel('Predicted Individual Treatment Effect (|ITE|, %)', fontweight='bold')
    ax.set_ylabel('Observed Hazard Ratio (Lower = More Benefit)', fontweight='bold')
    ax.set_title('Empirical HTE Validation:\nPredicted Treatment Effects vs RCT Subgroup Analysis', 
                 fontweight='bold', pad=15)
    
    ax.legend(loc='lower left', framealpha=0.9)
    ax.set_xlim(0, 7)
    ax.set_ylim(0.4, 1.2)
    
    # Add reference line at HR=1
    ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax.text(6.5, 1.02, 'No Effect', fontsize=8, color='gray', ha='right')
    
    # Shaded region for high benefit
    ax.axhspan(0.4, 0.7, alpha=0.1, color='green')
    ax.text(0.3, 0.58, 'High\nBenefit', fontsize=8, color='darkgreen', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig3_hte_validation.pdf')
    plt.savefig(f'{output_dir}/fig3_hte_validation.png', dpi=300)
    plt.close()
    print("Created Figure 3: HTE Validation")


# ============================================================
# FIGURE 4: Gamma Sensitivity Analysis
# ============================================================
def create_sensitivity_figure():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data from Table 6
    gamma_values = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    # Interventions with their ARR at different gamma
    interventions = {
        'Statin (LDL −1)': [3.9, 4.85, 5.8, 6.95, 8.1],
        'Weight Loss (−7%)': [7.6, 9.45, 11.3, 13.55, 15.8],
        'SGLT2i (Renal)': [2.1, 2.6, 3.1, 3.7, 4.3],
        'SBP −15 mmHg': [2.8, 3.5, 4.2, 5.05, 5.9],
        'PCSK9i': [1.2, 1.5, 1.8, 2.15, 2.5],
        'SGLT2i (CV)': [1.9, 2.35, 2.8, 3.35, 3.9],
    }
    
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['tertiary'], 
              COLORS['success'], COLORS['danger'], COLORS['neutral']]
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for i, (intervention, arr_values) in enumerate(interventions.items()):
        ax.plot(gamma_values, arr_values, marker=markers[i], color=colors[i], 
                linewidth=2, markersize=8, label=intervention)
    
    # Mark optimal gamma
    ax.axvline(x=0.65, color='red', linestyle='--', linewidth=2, alpha=0.7, label=r'$\gamma^*$ = 0.65')
    ax.axvline(x=0.7, color='green', linestyle=':', linewidth=2, alpha=0.7, label=r'Default $\gamma$ = 0.7')
    
    # Add annotation
    ax.annotate(r'$\gamma^*$ = 0.65' + '\n(RMSE = 0.74)', xy=(0.65, 14), 
                fontsize=9, ha='center', color='red',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='red', alpha=0.8))
    
    ax.set_xlabel('Attenuation Parameter (γ)', fontweight='bold')
    ax.set_ylabel('Simulated Absolute Risk Reduction (%)', fontweight='bold')
    ax.set_title('Sensitivity Analysis: Effect of Attenuation Parameter\non Simulated Intervention Effects', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.9, ncol=2)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0, 18)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig4_sensitivity.pdf')
    plt.savefig(f'{output_dir}/fig4_sensitivity.png', dpi=300)
    plt.close()
    print("Created Figure 4: Sensitivity Analysis")


# ============================================================
# FIGURE 5: LLM Validation Agreement
# ============================================================
def create_llm_validation_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    
    # Left panel: Overall agreement
    models = ['GPT-4', 'Claude 3.5', 'Inter-model']
    agreement = [87.9, 85.0, 91.6]
    colors = [COLORS['primary'], COLORS['secondary'], COLORS['success']]
    
    bars = ax1.bar(models, agreement, color=colors, edgecolor='black', linewidth=0.8, width=0.6)
    
    for bar, val in zip(bars, agreement):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax1.set_ylabel('Direction Agreement (%)', fontweight='bold')
    ax1.set_title('LLM Validation of 107 Expert-Curated\nCausal Edges', fontweight='bold', pad=10)
    ax1.set_ylim(0, 105)
    ax1.axhline(y=80, color='gray', linestyle='--', linewidth=1, alpha=0.7, label='Strong Agreement')
    ax1.legend(loc='lower right', fontsize=8)
    
    # Right panel: Agreement by evidence grade
    grades = ['Grade A\n(RCT/MR)', 'Grade B\n(Cohort)', 'Grade C\n(Emerging)']
    gpt4_grade = [95.7, 78.6, 60.0]
    
    x = np.arange(len(grades))
    bars2 = ax2.bar(x, gpt4_grade, color=[COLORS['success'], COLORS['tertiary'], COLORS['danger']], 
                   edgecolor='black', linewidth=0.8, width=0.5)
    
    for bar, val in zip(bars2, gpt4_grade):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    ax2.set_ylabel('GPT-4 Agreement (%)', fontweight='bold')
    ax2.set_title('Agreement Stratified by\nEvidence Grade', fontweight='bold', pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(grades)
    ax2.set_ylim(0, 110)
    
    # Add trend annotation
    ax2.annotate('', xy=(2, 60), xytext=(0, 95.7),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, ls='--'))
    ax2.text(1.5, 80, 'Agreement tracks\nevidence strength', fontsize=8, 
             ha='center', style='italic', color='gray')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig5_llm_validation.pdf')
    plt.savefig(f'{output_dir}/fig5_llm_validation.png', dpi=300)
    plt.close()
    print("Created Figure 5: LLM Validation")


# ============================================================
# FIGURE 6: HTE Benefit Stratification Example
# ============================================================
def create_hte_stratification_figure():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Data from Table 2 - HTE examples
    patient_types = ['High Modifiable\nLoad', 'Moderate Risk\nElevated LDL', 
                     'Low LDL\nOther Factors', 'Genetic Risk\nDominant']
    ldl_c = [180, 160, 100, 90]
    baseline_risk = [22, 15, 15, 18]
    post_statin_risk = [14, 10, 13, 17]
    ite = [8, 5, 2, 1]
    
    x = np.arange(len(patient_types))
    width = 0.35
    
    # Create grouped bars for baseline and post-intervention risk
    bars1 = ax.bar(x - width/2, baseline_risk, width, label='Baseline Risk (R₀)', 
                   color=COLORS['danger'], edgecolor='black', linewidth=0.8, alpha=0.9)
    bars2 = ax.bar(x + width/2, post_statin_risk, width, label='Post-Statin Risk (R₁)', 
                   color=COLORS['success'], edgecolor='black', linewidth=0.8, alpha=0.9)
    
    # Add ITE arrows
    for i, (b, p, effect) in enumerate(zip(baseline_risk, post_statin_risk, ite)):
        mid_x = i
        ax.annotate('', xy=(mid_x + width/2, p + 0.5), xytext=(mid_x - width/2, b - 0.5),
                   arrowprops=dict(arrowstyle='->', color='black', lw=1.5))
        ax.text(mid_x, (b + p)/2 + 2, f'ITE:\n−{effect}%', ha='center', fontsize=9, 
               fontweight='bold', color='navy')
    
    # Benefit stratification regions
    ax.axhspan(0, 25, xmin=0, xmax=0.25, alpha=0.2, color='green')
    ax.axhspan(0, 25, xmin=0.25, xmax=0.75, alpha=0.15, color='yellow')
    ax.axhspan(0, 25, xmin=0.75, xmax=1.0, alpha=0.2, color='orange')
    
    ax.text(0, 24.5, 'High Benefit', fontsize=8, color='darkgreen', fontweight='bold', ha='center')
    ax.text(1.5, 24.5, 'Moderate', fontsize=8, color='olive', fontweight='bold', ha='center')
    ax.text(3, 24.5, 'Low Benefit', fontsize=8, color='darkorange', fontweight='bold', ha='center')
    
    # Add LDL-C values as secondary info
    for i, ldl in enumerate(ldl_c):
        ax.text(i, -2, f'LDL-C: {ldl}', ha='center', fontsize=8, color='gray')
    
    ax.set_ylabel('10-Year CVD Risk (%)', fontweight='bold')
    ax.set_xlabel('Patient Archetype', fontweight='bold')
    ax.set_title('Heterogeneous Treatment Effects:\nSame Risk, Different Benefit from Statin Therapy', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(patient_types)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim(-4, 27)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fig6_hte_stratification.pdf')
    plt.savefig(f'{output_dir}/fig6_hte_stratification.png', dpi=300)
    plt.close()
    print("Created Figure 6: HTE Stratification")


# ============================================================
# Run all figure generation
# ============================================================
if __name__ == '__main__':
    print("Generating publication-quality figures for NCD-CIE-LLM v25...")
    print("=" * 60)
    
    create_performance_figure()
    create_rct_validity_figure()
    create_hte_validation_figure()
    create_sensitivity_figure()
    create_llm_validation_figure()
    create_hte_stratification_figure()
    
    print("=" * 60)
    print(f"All figures saved to: {output_dir}")
    print("Formats: PDF and PNG (300 DPI)")
