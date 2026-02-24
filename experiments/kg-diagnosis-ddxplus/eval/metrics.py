"""Evaluation metrics for differential diagnosis."""
import logging
import math
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def top_k_accuracy(predicted: List[str], ground_truth: str, k: int) -> float:
    """Check if ground truth appears in top-k predictions."""
    gt_lower = ground_truth.lower().strip()
    for p in predicted[:k]:
        if p.lower().strip() == gt_lower or gt_lower in p.lower() or p.lower() in gt_lower:
            return 1.0
    return 0.0


def ndcg_at_k(predicted: List[str], ground_truth_diff: List[Tuple[str, float]], k: int = 5) -> float:
    """Compute NDCG@k given predicted ranking and ground truth differential."""
    gt_dict = {}
    for disease, prob in ground_truth_diff:
        gt_dict[disease.lower().strip()] = prob

    # DCG
    dcg = 0.0
    for i, pred in enumerate(predicted[:k]):
        pred_lower = pred.lower().strip()
        rel = 0.0
        for gt_name, gt_prob in gt_dict.items():
            if pred_lower == gt_name or gt_name in pred_lower or pred_lower in gt_name:
                rel = gt_prob
                break
        dcg += rel / math.log2(i + 2)

    # Ideal DCG
    ideal_rels = sorted(gt_dict.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))

    return dcg / max(idcg, 1e-8)


def f1_score(predicted: List[str], ground_truth_diff: List[Tuple[str, float]]) -> float:
    """Compute F1 between predicted set and ground truth differential set."""
    gt_set = {d.lower().strip() for d, _ in ground_truth_diff}
    pred_set = {p.lower().strip() for p in predicted}

    if not gt_set or not pred_set:
        return 0.0

    # Fuzzy matching
    tp = 0
    for p in pred_set:
        for g in gt_set:
            if p == g or g in p or p in g:
                tp += 1
                break

    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gt_set) if gt_set else 0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def hallucination_rate(predicted: List[str], valid_pathologies: Set[str]) -> float:
    """Fraction of predictions not in the valid pathology list."""
    if not predicted:
        return 0.0

    valid_lower = {p.lower().strip() for p in valid_pathologies}
    hallucinated = 0
    for p in predicted:
        p_lower = p.lower().strip()
        match = any(p_lower == v or v in p_lower or p_lower in v for v in valid_lower)
        if not match:
            hallucinated += 1

    return hallucinated / len(predicted)


def compute_all_metrics(
    predicted: List[str],
    vignette: Dict,
    valid_pathologies: Set[str],
) -> Dict[str, float]:
    """Compute all metrics for a single prediction."""
    gt = vignette.get("pathology", "")
    gt_diff = vignette.get("differential", [])

    return {
        "top1_acc": top_k_accuracy(predicted, gt, 1),
        "top3_acc": top_k_accuracy(predicted, gt, 3),
        "top5_acc": top_k_accuracy(predicted, gt, 5),
        "ndcg5": ndcg_at_k(predicted, gt_diff, 5),
        "f1": f1_score(predicted, gt_diff),
        "hallucination_rate": hallucination_rate(predicted, valid_pathologies),
    }


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across all vignettes."""
    if not all_metrics:
        return {}

    keys = all_metrics[0].keys()
    agg = {}
    for key in keys:
        values = [m[key] for m in all_metrics]
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
    return agg


def bootstrap_ci(values: List[float], n_bootstrap: int = 1000, ci: float = 0.95, seed: int = 42) -> Tuple[float, float]:
    """Compute bootstrap confidence interval."""
    rng = np.random.RandomState(seed)
    n = len(values)
    if n == 0:
        return (0.0, 0.0)

    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1 - ci) / 2
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1 - alpha) * 100))
    return (lower, upper)


def mcnemar_test(correct_a: List[bool], correct_b: List[bool]) -> float:
    """McNemar's test p-value for comparing two conditions."""
    from scipy.stats import chi2

    n = len(correct_a)
    # b correct, a wrong
    b_not_a = sum(1 for i in range(n) if correct_b[i] and not correct_a[i])
    # a correct, b wrong
    a_not_b = sum(1 for i in range(n) if correct_a[i] and not correct_b[i])

    if b_not_a + a_not_b == 0:
        return 1.0

    chi2_stat = (abs(b_not_a - a_not_b) - 1) ** 2 / (b_not_a + a_not_b)
    p_value = 1 - chi2.cdf(chi2_stat, df=1)
    return float(p_value)


def format_results_table(results: Dict[str, Dict], latex: bool = False) -> str:
    """Format results as a table."""
    metrics = ["top1_acc_mean", "top3_acc_mean", "top5_acc_mean", "ndcg5_mean", "f1_mean", "hallucination_rate_mean"]
    headers = ["Condition", "Top-1", "Top-3", "Top-5", "NDCG@5", "F1", "Halluc."]

    if latex:
        lines = [
            "\\begin{table}[htbp]",
            "\\centering",
            "\\caption{Differential Diagnosis Performance on DDXPlus}",
            "\\label{tab:results}",
            "\\begin{tabular}{l" + "c" * len(metrics) + "}",
            "\\toprule",
            " & ".join(headers) + " \\\\",
            "\\midrule",
        ]
        for cond, agg in results.items():
            vals = [f"{agg.get(m, 0):.3f}" for m in metrics]
            lines.append(f"{cond} & " + " & ".join(vals) + " \\\\")
        lines.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}"])
        return "\n".join(lines)
    else:
        lines = ["\t".join(headers)]
        for cond, agg in results.items():
            vals = [f"{agg.get(m, 0):.3f}" for m in metrics]
            lines.append(f"{cond}\t" + "\t".join(vals))
        return "\n".join(lines)
