"""
Visualization Module for Research Paper Figures
================================================

Generates publication-quality figures for the paper including:
- Performance comparison bar charts
- Multi-hop accuracy line plots
- Ablation study visualizations
- Architecture diagrams (conceptual)

Author: Research Team
License: MIT
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os

# Set publication-quality defaults
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 13,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})


def create_output_dir(path: str = "figures") -> str:
    """Create output directory for figures."""
    os.makedirs(path, exist_ok=True)
    return path


def plot_main_results_comparison(save_path: Optional[str] = None):
    """
    Create bar chart comparing main results across methods and benchmarks.
    """
    # Data
    methods = ['Vanilla LLM', 'RAG', 'KG-RAG', 'MemoryBank', 'Ours (Full)']
    benchmarks = ['NQ EM', 'NQ F1', 'TriviaQA', 'HotpotQA', 'FEVER', 'TruthfulQA']
    
    # Results data (aligned with paper)
    data = {
        'Vanilla LLM': [29.3, 38.7, 52.1, 31.2, 71.4, 38.2],
        'RAG': [41.2, 52.3, 65.8, 42.5, 79.8, 51.6],
        'KG-RAG': [43.7, 54.1, 67.2, 47.3, 82.1, 55.3],
        'MemoryBank': [42.8, 53.6, 66.4, 45.1, 80.9, 53.8],
        'Ours (Full)': [49.4, 60.2, 73.5, 55.2, 86.9, 63.1],
    }
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(benchmarks))
    width = 0.15
    multiplier = 0
    
    colors = ['#8da0cb', '#66c2a5', '#fc8d62', '#e78ac3', '#a6d854']
    
    for (method, values), color in zip(data.items(), colors):
        offset = width * multiplier
        bars = ax.bar(x + offset, values, width, label=method, color=color, edgecolor='white')
        multiplier += 1
    
    ax.set_xlabel('Benchmark')
    ax.set_ylabel('Score (%)')
    ax.set_title('Performance Comparison Across Benchmarks')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(benchmarks)
    ax.legend(loc='upper left', bbox_to_anchor=(0, 1), ncol=3)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on best method
    for i, v in enumerate(data['Ours (Full)']):
        ax.annotate(f'{v:.1f}', xy=(i + width * 4, v + 1), ha='center', va='bottom', 
                    fontsize=8, fontweight='bold', color='#a6d854')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_multihop_accuracy(save_path: Optional[str] = None):
    """
    Create line plot showing accuracy vs number of reasoning hops.
    """
    hops = [1, 2, 3, 4]
    
    # Data aligned with paper
    data = {
        'RAG': [72.3, 55.1, 38.7, 28.9],
        'KG-RAG': [74.8, 61.2, 49.3, 41.2],
        'Ours (Full)': [78.2, 68.7, 58.4, 51.6],
    }
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    markers = ['s', '^', 'o']
    colors = ['#4575b4', '#d73027', '#1a9850']
    
    for (method, values), marker, color in zip(data.items(), markers, colors):
        ax.plot(hops, values, marker=marker, label=method, color=color, 
                linewidth=2, markersize=8)
        
        # Add value annotations
        for x, y in zip(hops, values):
            ax.annotate(f'{y:.1f}', xy=(x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    ax.set_xlabel('Number of Reasoning Hops')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Multi-hop Reasoning Accuracy by Complexity')
    ax.set_xticks(hops)
    ax.set_xlim(0.5, 4.5)
    ax.set_ylim(20, 85)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add shaded region to highlight improvement
    ax.fill_between(hops, data['RAG'], data['Ours (Full)'], alpha=0.2, color='green',
                    label='Improvement')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_ablation_study(save_path: Optional[str] = None):
    """
    Create bar chart showing ablation study results.
    """
    configurations = [
        'Cooc only', 'Seq only', 'KG only',
        'Cooc+Seq', 'Cooc+KG', 'Seq+KG', 'Full'
    ]
    
    # Average scores across benchmarks (simplified)
    nq_f1 = [49.8, 53.2, 54.8, 55.6, 57.1, 57.8, 60.2]
    hotpotqa = [39.7, 44.8, 48.1, 47.9, 50.6, 51.8, 55.2]
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    x = np.arange(len(configurations))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, nq_f1, width, label='NQ F1', color='#66c2a5')
    bars2 = ax.bar(x + width/2, hotpotqa, width, label='HotpotQA F1', color='#fc8d62')
    
    ax.set_xlabel('Configuration')
    ax.set_ylabel('F1 Score (%)')
    ax.set_title('Ablation Study: Component Contributions')
    ax.set_xticks(x)
    ax.set_xticklabels(configurations, rotation=15, ha='right')
    ax.legend()
    ax.set_ylim(0, 70)
    ax.grid(axis='y', alpha=0.3)
    
    # Highlight full system
    ax.axhline(y=nq_f1[-1], color='#66c2a5', linestyle='--', alpha=0.5)
    ax.axhline(y=hotpotqa[-1], color='#fc8d62', linestyle='--', alpha=0.5)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords='offset points', ha='center', va='bottom',
                    fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_factual_consistency(save_path: Optional[str] = None):
    """
    Create dual bar chart for factual consistency and hallucination rates.
    """
    methods = ['Vanilla LLM', 'RAG', 'KG-RAG', 'MemoryBank', 'Ours']
    consistency = [62.3, 74.5, 78.2, 76.1, 80.6]
    hallucination = [24.7, 16.2, 13.8, 15.1, 9.5]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    
    # Consistency plot
    colors1 = ['#d9d9d9', '#bdbdbd', '#969696', '#636363', '#1a9850']
    bars1 = ax1.barh(methods, consistency, color=colors1, edgecolor='white')
    ax1.set_xlabel('Factual Consistency (%)')
    ax1.set_title('Factual Consistency')
    ax1.set_xlim(50, 90)
    ax1.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars1, consistency):
        ax1.text(val + 0.5, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                 va='center', fontsize=9)
    
    # Hallucination plot
    colors2 = ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#1a9850']
    bars2 = ax2.barh(methods, hallucination, color=colors2, edgecolor='white')
    ax2.set_xlabel('Hallucination Rate (%)')
    ax2.set_title('Hallucination Rate (Lower is Better)')
    ax2.set_xlim(0, 30)
    ax2.grid(axis='x', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars2, hallucination):
        ax2.text(val + 0.3, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                 va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_gating_weights_heatmap(save_path: Optional[str] = None):
    """
    Create heatmap showing gating weights for different query types.
    """
    query_types = ['Factual', 'Semantic\nSimilarity', 'Temporal', 'Multi-hop\nReasoning']
    components = ['Co-occurrence', 'Sequence', 'Knowledge Graph']
    
    # Data aligned with paper
    weights = np.array([
        [0.18, 0.22, 0.60],  # Factual
        [0.45, 0.28, 0.27],  # Semantic similarity
        [0.21, 0.52, 0.27],  # Temporal
        [0.15, 0.18, 0.67],  # Multi-hop
    ])
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    im = ax.imshow(weights, cmap='YlOrRd', aspect='auto', vmin=0, vmax=0.7)
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Gating Weight', rotation=270, labelpad=15)
    
    # Set ticks
    ax.set_xticks(np.arange(len(components)))
    ax.set_yticks(np.arange(len(query_types)))
    ax.set_xticklabels(components)
    ax.set_yticklabels(query_types)
    
    # Add text annotations
    for i in range(len(query_types)):
        for j in range(len(components)):
            text = ax.text(j, i, f'{weights[i, j]:.2f}',
                          ha='center', va='center', color='black', fontsize=11)
    
    ax.set_title('Adaptive Gating Weights by Query Type')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_efficiency_comparison(save_path: Optional[str] = None):
    """
    Create scatter plot of latency vs accuracy trade-off.
    """
    methods = ['Vanilla\nLLM', 'RAG', 'KG-RAG', 'Ours']
    latency = [127, 183, 241, 198]
    accuracy = [38.7, 52.3, 54.1, 60.2]  # NQ F1 as representative
    memory = [2.1, 4.7, 6.2, 5.8]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Bubble sizes based on memory
    sizes = [m * 50 for m in memory]
    
    colors = ['#8da0cb', '#66c2a5', '#fc8d62', '#a6d854']
    
    scatter = ax.scatter(latency, accuracy, s=sizes, c=colors, alpha=0.7, edgecolors='black')
    
    # Add method labels
    for i, (x, y, method) in enumerate(zip(latency, accuracy, methods)):
        ax.annotate(method, xy=(x, y), xytext=(10, 5), textcoords='offset points',
                    fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Latency (ms)')
    ax.set_ylabel('NQ F1 Score (%)')
    ax.set_title('Efficiency-Accuracy Trade-off\n(bubble size = memory usage)')
    ax.grid(True, alpha=0.3)
    
    # Add legend for bubble sizes
    legend_elements = [
        plt.scatter([], [], s=2*50, c='gray', alpha=0.5, label='2 GB'),
        plt.scatter([], [], s=4*50, c='gray', alpha=0.5, label='4 GB'),
        plt.scatter([], [], s=6*50, c='gray', alpha=0.5, label='6 GB'),
    ]
    ax.legend(handles=legend_elements, title='Memory', loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_window_size_analysis(save_path: Optional[str] = None):
    """
    Plot effect of co-occurrence window size on performance.
    """
    window_sizes = [2, 5, 10, 20]
    nq_f1 = [57.3, 60.2, 59.1, 57.8]
    semantic_sim = [0.72, 0.78, 0.81, 0.76]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # NQ F1 plot
    ax1.plot(window_sizes, nq_f1, 'o-', color='#1f77b4', linewidth=2, markersize=8)
    ax1.set_xlabel('Window Size')
    ax1.set_ylabel('NQ F1 Score (%)')
    ax1.set_title('Effect of Window Size on QA Performance')
    ax1.set_xticks(window_sizes)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=max(nq_f1), color='green', linestyle='--', alpha=0.5)
    
    # Highlight optimal
    opt_idx = nq_f1.index(max(nq_f1))
    ax1.scatter([window_sizes[opt_idx]], [max(nq_f1)], color='red', s=100, zorder=5,
                marker='*', label='Optimal')
    ax1.legend()
    
    # Semantic similarity plot
    ax2.plot(window_sizes, semantic_sim, 's-', color='#ff7f0e', linewidth=2, markersize=8)
    ax2.set_xlabel('Window Size')
    ax2.set_ylabel('Semantic Similarity Score')
    ax2.set_title('Effect of Window Size on Semantic Capture')
    ax2.set_xticks(window_sizes)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=max(semantic_sim), color='green', linestyle='--', alpha=0.5)
    
    # Highlight optimal
    opt_idx = semantic_sim.index(max(semantic_sim))
    ax2.scatter([window_sizes[opt_idx]], [max(semantic_sim)], color='red', s=100, zorder=5,
                marker='*', label='Optimal')
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def plot_architecture_diagram(save_path: Optional[str] = None):
    """
    Create conceptual architecture diagram.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis('off')
    
    # Component boxes
    box_style = dict(boxstyle="round,pad=0.3", facecolor='lightblue', edgecolor='navy')
    
    # Input
    ax.text(1, 7, 'Query Input', fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='orange'))
    
    # Three components
    components = [
        (2, 5, 'Co-occurrence\nAnalyzer\n(PMI Matrix)', 'lightgreen'),
        (6, 5, 'Sequence\nIndexer\n(Transformer)', 'lightcoral'),
        (10, 5, 'Knowledge\nGraph\n(+ Ontology)', 'lightblue'),
    ]
    
    for x, y, text, color in components:
        ax.add_patch(mpatches.FancyBboxPatch((x-1.3, y-0.8), 2.6, 1.6,
                                             boxstyle="round,pad=0.1",
                                             facecolor=color, edgecolor='black'))
        ax.text(x, y, text, fontsize=9, ha='center', va='center')
    
    # Gating mechanism
    ax.add_patch(mpatches.FancyBboxPatch((4.7, 2.2), 2.6, 1.2,
                                         boxstyle="round,pad=0.1",
                                         facecolor='plum', edgecolor='purple'))
    ax.text(6, 2.8, 'Attention-based\nGating', fontsize=10, ha='center', va='center')
    
    # Output
    ax.text(6, 0.8, 'Hybrid Memory Response', fontsize=11, ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', edgecolor='orange'))
    
    # Arrows
    arrow_style = dict(arrowstyle='->', color='gray', lw=1.5)
    
    # From input to components
    ax.annotate('', xy=(2, 5.8), xytext=(1, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 5.8), xytext=(1, 6.5), arrowprops=arrow_style)
    ax.annotate('', xy=(10, 5.8), xytext=(1, 6.5), arrowprops=arrow_style)
    
    # From components to gating
    ax.annotate('', xy=(4.7, 2.8), xytext=(2, 4.2), arrowprops=arrow_style)
    ax.annotate('', xy=(6, 3.4), xytext=(6, 4.2), arrowprops=arrow_style)
    ax.annotate('', xy=(7.3, 2.8), xytext=(10, 4.2), arrowprops=arrow_style)
    
    # From gating to output
    ax.annotate('', xy=(6, 1.4), xytext=(6, 2.2), arrowprops=arrow_style)
    
    # Labels
    ax.text(3, 3.5, 'α', fontsize=12, color='green', fontweight='bold')
    ax.text(5.5, 3.5, 'β', fontsize=12, color='red', fontweight='bold')
    ax.text(8, 3.5, 'γ', fontsize=12, color='blue', fontweight='bold')
    
    # Title
    ax.text(6, 7.5, 'Hybrid Memory Architecture ("Second Brain")',
            fontsize=14, ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    
    return fig


def generate_all_figures(output_dir: str = "figures"):
    """Generate all figures for the paper."""
    create_output_dir(output_dir)
    
    print("Generating figures...")
    print("=" * 50)
    
    figures = [
        ("main_results.png", plot_main_results_comparison),
        ("multihop_accuracy.png", plot_multihop_accuracy),
        ("ablation_study.png", plot_ablation_study),
        ("factual_consistency.png", plot_factual_consistency),
        ("gating_heatmap.png", plot_gating_weights_heatmap),
        ("efficiency_tradeoff.png", plot_efficiency_comparison),
        ("window_size.png", plot_window_size_analysis),
        ("architecture.png", plot_architecture_diagram),
    ]
    
    for filename, plot_func in figures:
        filepath = os.path.join(output_dir, filename)
        try:
            plot_func(save_path=filepath)
            plt.close()
        except Exception as e:
            print(f"Error generating {filename}: {e}")
    
    print("=" * 50)
    print(f"All figures saved to {output_dir}/")


if __name__ == "__main__":
    generate_all_figures()
