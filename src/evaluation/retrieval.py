"""Retrieval evaluation metrics.

Implements standard retrieval metrics including:
- Recall@K (standard and both-support for multi-hop)
- MRR (Mean Reciprocal Rank)
- NDCG@K (Normalized Discounted Cumulative Gain)
"""

from typing import List, Set, Union
import numpy as np


def recall_at_k(
    retrieved: List[int], 
    relevant: Set[int], 
    k: int
) -> float:
    """Compute Recall@K.
    
    Recall@K = |relevant ∩ retrieved[:k]| / |relevant|
    
    Args:
        retrieved: List of retrieved document IDs (ranked).
        relevant: Set of relevant document IDs.
        k: Cutoff.
        
    Returns:
        Recall@K score.
    """
    if not relevant:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    return len(relevant & retrieved_at_k) / len(relevant)


def both_support_recall(
    retrieved: List[int],
    support_docs: List[int],
    k: int
) -> float:
    """Compute both-support Recall@K for multi-hop QA.
    
    This metric checks if ALL supporting documents are retrieved
    within top-K. Used in HotpotQA evaluation.
    
    Args:
        retrieved: List of retrieved document IDs.
        support_docs: List of gold supporting document IDs (usually 2).
        k: Cutoff.
        
    Returns:
        1.0 if all support docs in top-K, else 0.0.
    """
    if not support_docs:
        return 0.0
    
    retrieved_at_k = set(retrieved[:k])
    support_set = set(support_docs)
    
    return 1.0 if support_set.issubset(retrieved_at_k) else 0.0


def mrr(
    retrieved: List[int], 
    relevant: Set[int]
) -> float:
    """Compute Mean Reciprocal Rank.
    
    MRR = 1 / rank of first relevant document
    
    Args:
        retrieved: List of retrieved document IDs.
        relevant: Set of relevant document IDs.
        
    Returns:
        MRR score.
    """
    for rank, doc_id in enumerate(retrieved, 1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: List[int],
    relevance_scores: dict,
    k: int
) -> float:
    """Compute NDCG@K.
    
    NDCG@K = DCG@K / IDCG@K
    DCG@K = sum(rel_i / log2(i+1)) for i in 1..k
    
    Args:
        retrieved: List of retrieved document IDs.
        relevance_scores: Dict mapping doc_id to relevance score.
        k: Cutoff.
        
    Returns:
        NDCG@K score.
    """
    # Compute DCG
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        rel = relevance_scores.get(doc_id, 0)
        dcg += rel / np.log2(i + 2)  # i+2 because log2(1) = 0
    
    # Compute ideal DCG
    ideal_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_rels))
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def compute_retrieval_metrics(
    all_retrieved: List[List[int]],
    all_relevant: List[Set[int]],
    ks: List[int] = [5, 10, 20, 50, 100]
) -> dict:
    """Compute retrieval metrics across a dataset.
    
    Args:
        all_retrieved: List of retrieved rankings for each query.
        all_relevant: List of relevant doc sets for each query.
        ks: List of K values for Recall@K.
        
    Returns:
        Dictionary of metrics.
    """
    metrics = {}
    
    for k in ks:
        recalls = [
            recall_at_k(ret, rel, k) 
            for ret, rel in zip(all_retrieved, all_relevant)
        ]
        metrics[f"recall@{k}"] = np.mean(recalls)
        metrics[f"recall@{k}_std"] = np.std(recalls)
    
    mrrs = [mrr(ret, rel) for ret, rel in zip(all_retrieved, all_relevant)]
    metrics["mrr"] = np.mean(mrrs)
    metrics["mrr_std"] = np.std(mrrs)
    
    return metrics


def compute_hotpotqa_metrics(
    all_retrieved: List[List[int]],
    all_support_pairs: List[List[int]],
    ks: List[int] = [5, 10, 20, 50, 100]
) -> dict:
    """Compute HotpotQA-specific retrieval metrics.
    
    Args:
        all_retrieved: Retrieved rankings for each query.
        all_support_pairs: Gold support doc pairs for each query.
        ks: K values.
        
    Returns:
        Dictionary of metrics including both-support recall.
    """
    metrics = {}
    
    for k in ks:
        # Standard recall (treating both docs as relevant)
        recalls = [
            recall_at_k(ret, set(sup), k)
            for ret, sup in zip(all_retrieved, all_support_pairs)
        ]
        metrics[f"recall@{k}"] = np.mean(recalls)
        
        # Both-support recall (stricter: both must be in top-K)
        both_recalls = [
            both_support_recall(ret, sup, k)
            for ret, sup in zip(all_retrieved, all_support_pairs)
        ]
        metrics[f"both@{k}"] = np.mean(both_recalls)
    
    return metrics


def bootstrap_ci(
    values: List[float],
    n_bootstrap: int = 1000,
    confidence: float = 0.95
) -> tuple:
    """Compute bootstrap confidence interval.
    
    Args:
        values: List of metric values.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    values = np.array(values)
    n = len(values)
    
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(values, size=n, replace=True)
        bootstrap_means.append(np.mean(sample))
    
    bootstrap_means = np.array(bootstrap_means)
    
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, 100 * alpha / 2)
    upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
    
    return float(np.mean(values)), float(lower), float(upper)
