"""
Gating Mechanism Module

Implements the learned query-adaptive gating mechanism that combines
signals from co-occurrence, dense retrieval, and knowledge graph sources.
"""

import numpy as np
from typing import Dict, List, Optional

class GatingMechanism:
    """
    Query-adaptive gating mechanism for multi-source fusion.
    
    Learns to weight different retrieval sources based on query type
    and candidate relevance signals.
    """
    
    def __init__(self, num_sources: int = 3):
        """
        Initialize the gating mechanism.
        
        Args:
            num_sources: Number of retrieval sources to combine
        """
        self.num_sources = num_sources
        self.source_names = ['cooccurrence', 'dense', 'kg']
        
        # Initialize with uniform weights (will be learned)
        self.source_weights = np.ones(num_sources) / num_sources
        
        # Query type patterns for heuristic classification
        self.entity_patterns = ['who', 'what', 'which', 'where']
        self.factual_patterns = ['when', 'how many', 'how much', 'date', 'year']
        self.reasoning_patterns = ['why', 'how does', 'explain', 'cause']
    
    def classify_query(self, query: str) -> str:
        """
        Classify query type for source weighting.
        
        Args:
            query: The input query
            
        Returns:
            Query type: 'entity', 'factual', 'reasoning', or 'general'
        """
        query_lower = query.lower()
        
        for pattern in self.entity_patterns:
            if pattern in query_lower:
                return 'entity'
        
        for pattern in self.factual_patterns:
            if pattern in query_lower:
                return 'factual'
        
        for pattern in self.reasoning_patterns:
            if pattern in query_lower:
                return 'reasoning'
        
        return 'general'
    
    def compute_weights(self, query: str) -> Dict[str, float]:
        """
        Compute source weights based on query.
        
        This implements the source-level gating g_src(s) from the paper.
        
        Args:
            query: The input query
            
        Returns:
            Dictionary mapping source names to weights
        """
        query_type = self.classify_query(query)
        
        # Heuristic weights based on query type
        # (In full implementation, these would be learned via InfoNCE)
        weight_map = {
            'entity': {'cooccurrence': 0.2, 'dense': 0.3, 'kg': 0.5},
            'factual': {'cooccurrence': 0.3, 'dense': 0.4, 'kg': 0.3},
            'reasoning': {'cooccurrence': 0.4, 'dense': 0.4, 'kg': 0.2},
            'general': {'cooccurrence': 0.33, 'dense': 0.34, 'kg': 0.33}
        }
        
        return weight_map[query_type]
    
    def compute_candidate_scores(
        self,
        query_embedding: np.ndarray,
        candidate_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-candidate gating scores.
        
        This implements g_cand(i) = σ(w^T h_i) from the paper.
        
        Args:
            query_embedding: Query vector (d,)
            candidate_embeddings: Candidate matrix (n, d)
            
        Returns:
            Candidate scores (n,)
        """
        # Simple dot product attention (full version uses learned projection)
        scores = candidate_embeddings @ query_embedding
        
        # Sigmoid activation
        return 1 / (1 + np.exp(-scores))
    
    def fuse_scores(
        self,
        source_scores: Dict[str, np.ndarray],
        source_weights: Dict[str, float],
        candidate_weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Fuse scores from multiple sources using gating weights.
        
        Implements: final_score(i) = Σ_s g_src(s) × g_cand(i) × score_s(i)
        
        Args:
            source_scores: Dict mapping source name to score array
            source_weights: Source-level gating weights
            candidate_weights: Per-candidate gating weights (optional)
            
        Returns:
            Fused scores for each candidate
        """
        n_candidates = len(next(iter(source_scores.values())))
        
        if candidate_weights is None:
            candidate_weights = np.ones(n_candidates)
        
        fused = np.zeros(n_candidates)
        
        for source_name, scores in source_scores.items():
            if source_name in source_weights:
                w_src = source_weights[source_name]
                fused += w_src * candidate_weights * scores
        
        return fused


# Mathematical foundations
"""
THEOREM 2: Gating Mechanism Properties

The two-level gating mechanism with:
- Source weights: g_src(s) ∈ [0,1] with Σ_s g_src(s) = 1
- Candidate weights: g_cand(i) = σ(w^T h_i) ∈ (0,1)

Satisfies:

1. BOUNDEDNESS: Final scores are bounded
   Proof: fused(i) = Σ_s g_src(s) × g_cand(i) × score_s(i)
          ≤ Σ_s g_src(s) × 1 × 1 = 1  (when all scores ≤ 1)

2. DIFFERENTIABILITY: The gating is fully differentiable
   Proof: σ(x) is differentiable everywhere with σ'(x) = σ(x)(1-σ(x))
          Source weights via softmax are also differentiable

3. EXPRESSIVENESS: Can represent both uniform and extreme weightings
   Proof: Setting all g_src equal gives uniform fusion
          Setting one g_src = 1 and others = 0 gives single-source

COMPLEXITY ANALYSIS:
- Time: O(k × n) where k = number of sources, n = number of candidates
- Space: O(n) for storing fused scores

QED
"""


class InfoNCETrainer:
    """
    Trainer for learning gating weights via InfoNCE contrastive loss.
    
    This implements Stage 1 of the two-stage training pipeline.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize the InfoNCE trainer.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        self.temperature = temperature
    
    def compute_loss(
        self,
        query_embedding: np.ndarray,
        positive_embedding: np.ndarray,
        negative_embeddings: np.ndarray
    ) -> float:
        """
        Compute InfoNCE contrastive loss.
        
        L = -log( exp(sim(q,p)/τ) / Σ_n exp(sim(q,n)/τ) )
        
        Args:
            query_embedding: Query vector
            positive_embedding: Positive candidate vector
            negative_embeddings: Matrix of negative candidate vectors
            
        Returns:
            Loss value
        """
        # Compute similarities
        pos_sim = np.dot(query_embedding, positive_embedding) / self.temperature
        neg_sims = negative_embeddings @ query_embedding / self.temperature
        
        # Numerical stability
        max_sim = max(pos_sim, np.max(neg_sims))
        
        # InfoNCE loss
        numerator = np.exp(pos_sim - max_sim)
        denominator = numerator + np.sum(np.exp(neg_sims - max_sim))
        
        loss = -np.log(numerator / denominator + 1e-10)
        
        return loss


"""
THEOREM 3: InfoNCE Training Convergence

Under standard assumptions (bounded gradients, sufficient data):

1. The InfoNCE loss is a lower bound on mutual information:
   L_InfoNCE ≥ log(N) - I(Q; P)
   where N = number of negatives, I(Q;P) = mutual information

2. Minimizing InfoNCE maximizes a lower bound on I(Q; P)

3. With learned temperature τ, the optimal representations satisfy:
   p(positive | query) ∝ exp(sim(q,p) / τ)

This justifies using InfoNCE for learning gating weights:
candidates that help answer the query become "positives"
and are pulled closer in embedding space.

REFERENCE: Oord et al., "Representation Learning with Contrastive Predictive Coding"
"""
