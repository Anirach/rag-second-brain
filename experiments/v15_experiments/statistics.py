#!/usr/bin/env python3
"""
Statistical Significance Testing Module

Implements:
- Bootstrap 95% confidence intervals (1000 resamples)
- Paired t-test for retrieval comparisons
- Wilcoxon signed-rank test (non-parametric alternative)
- Effect size (Cohen's d)

Author: RAG Second Brain Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import json
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def bootstrap_ci(
    data: np.ndarray,
    statistic: callable = np.mean,
    n_resamples: int = 1000,
    ci_level: float = 0.95,
    random_state: int = RANDOM_SEED
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval.
    
    Args:
        data: 1D array of observations
        statistic: Function to compute (default: mean)
        n_resamples: Number of bootstrap samples
        ci_level: Confidence level (default: 0.95)
        random_state: Random seed
    
    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    
    n = len(data)
    point_estimate = statistic(data)
    
    bootstrap_stats = []
    for _ in range(n_resamples):
        resample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic(resample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    alpha = 1 - ci_level
    ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    
    return point_estimate, ci_lower, ci_upper


def paired_ttest(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """
    Perform paired t-test.
    
    H0: mean(scores1) = mean(scores2)
    """
    assert len(scores1) == len(scores2), "Score arrays must have same length"
    
    t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
    
    # Cohen's d for paired samples
    diff = scores1 - scores2
    cohens_d = np.mean(diff) / np.std(diff, ddof=1)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'mean_diff': float(np.mean(diff)),
        'std_diff': float(np.std(diff, ddof=1))
    }


def wilcoxon_test(
    scores1: np.ndarray,
    scores2: np.ndarray,
    alternative: str = 'two-sided'
) -> Dict[str, float]:
    """Perform Wilcoxon signed-rank test (non-parametric)."""
    assert len(scores1) == len(scores2), "Score arrays must have same length"
    
    try:
        stat, p_value = stats.wilcoxon(scores1, scores2, alternative=alternative)
    except ValueError as e:
        return {'statistic': 0.0, 'p_value': 1.0, 'note': str(e)}
    
    return {'statistic': float(stat), 'p_value': float(p_value)}


def compute_all_statistics(
    method_scores: Dict[str, np.ndarray],
    comparisons: List[Tuple[str, str]] = None
) -> Dict:
    """Compute comprehensive statistics for all methods."""
    results = {
        'confidence_intervals': {},
        'pairwise_comparisons': {}
    }
    
    # Bootstrap CIs for each method
    for method, scores in method_scores.items():
        point, ci_low, ci_high = bootstrap_ci(np.array(scores))
        results['confidence_intervals'][method] = {
            'mean': float(point),
            'ci_95_lower': float(ci_low),
            'ci_95_upper': float(ci_high),
            'ci_width': float(ci_high - ci_low)
        }
    
    # Pairwise comparisons
    if comparisons is None:
        methods = list(method_scores.keys())
        baseline = methods[0]
        comparisons = [(baseline, m) for m in methods[1:]]
    
    for method1, method2 in comparisons:
        if method1 not in method_scores or method2 not in method_scores:
            continue
        
        scores1 = np.array(method_scores[method1])
        scores2 = np.array(method_scores[method2])
        
        key = f"{method1}_vs_{method2}"
        results['pairwise_comparisons'][key] = {
            'paired_ttest': paired_ttest(scores2, scores1, alternative='greater'),
            'wilcoxon': wilcoxon_test(scores2, scores1, alternative='greater'),
            'mean_improvement': float(np.mean(scores2) - np.mean(scores1)),
            'percent_improvement': float((np.mean(scores2) - np.mean(scores1)) / np.mean(scores1) * 100)
        }
    
    return results


if __name__ == "__main__":
    print("Statistical Testing Module")
    print("Run run_v15_experiments.py for full evaluation")
