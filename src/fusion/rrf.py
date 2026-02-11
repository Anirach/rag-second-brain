"""Reciprocal Rank Fusion (RRF) for combining rankings.

RRF is a simple but effective method for combining multiple ranked lists.
It's parameter-free (except for k) and doesn't require score normalization.

Reference: Cormack, G. V., Clarke, C. L., & Buettcher, S. (2009).
Reciprocal rank fusion outperforms condorcet and individual rank learning methods.
"""

from typing import List, Tuple, Dict
from collections import defaultdict


def rrf_fusion(
    rankings: List[List[Tuple[int, float]]], 
    k: int = 60,
    top_n: Optional[int] = None
) -> List[Tuple[int, float]]:
    """Combine multiple rankings using Reciprocal Rank Fusion.
    
    RRF score = sum(1 / (k + rank)) for each ranking
    
    Args:
        rankings: List of rankings, where each ranking is a list of
                 (document_id, score) tuples sorted by score descending.
        k: RRF parameter (default 60, as in original paper).
        top_n: Optional limit on output size.
        
    Returns:
        Combined ranking as list of (document_id, rrf_score) tuples.
    
    Example:
        >>> ranking1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
        >>> ranking2 = [(2, 0.95), (1, 0.85), (4, 0.75)]
        >>> rrf_fusion([ranking1, ranking2])
        [(2, 0.0322...), (1, 0.0319...), ...]
    """
    scores: Dict[int, float] = defaultdict(float)
    
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    
    # Sort by RRF score
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    
    if top_n:
        sorted_docs = sorted_docs[:top_n]
    
    return sorted_docs


def weighted_rrf_fusion(
    rankings: List[List[Tuple[int, float]]],
    weights: List[float],
    k: int = 60,
    top_n: Optional[int] = None
) -> List[Tuple[int, float]]:
    """Weighted RRF fusion with per-source weights.
    
    Args:
        rankings: List of rankings from different sources.
        weights: Weight for each source (should sum to 1 for interpretability).
        k: RRF parameter.
        top_n: Optional limit on output size.
        
    Returns:
        Combined ranking.
    """
    if len(rankings) != len(weights):
        raise ValueError("Number of rankings must match number of weights")
    
    scores: Dict[int, float] = defaultdict(float)
    
    for ranking, weight in zip(rankings, weights):
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] += weight / (k + rank + 1)
    
    sorted_docs = sorted(scores.items(), key=lambda x: -x[1])
    
    if top_n:
        sorted_docs = sorted_docs[:top_n]
    
    return sorted_docs


# Import Optional for type hints
from typing import Optional
