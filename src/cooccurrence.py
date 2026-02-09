"""
Co-occurrence Module

Implements semantic similarity scoring based on word co-occurrence statistics
using GloVe-style PPMI factorization.
"""

import numpy as np
from typing import List, Optional

class CooccurrenceScorer:
    """
    Co-occurrence based semantic similarity scorer.
    
    Uses pre-trained word vectors (GloVe-style) to compute
    semantic overlap between query and candidate passages.
    """
    
    def __init__(self, model_name: str = "glove-wiki-gigaword-100"):
        """
        Initialize the co-occurrence scorer.
        
        Args:
            model_name: Name of pre-trained word vectors to use
        """
        self.model_name = model_name
        self.vectors = None
        self._load_vectors()
    
    def _load_vectors(self):
        """Load pre-trained word vectors."""
        try:
            # Try to load from gensim
            import gensim.downloader as api
            print(f"Loading word vectors: {self.model_name}...")
            self.vectors = api.load(self.model_name)
            print(f"Loaded {len(self.vectors)} word vectors")
        except Exception as e:
            print(f"Could not load vectors: {e}")
            print("Using fallback random vectors for demo")
            self.vectors = None
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """Convert text to vector by averaging word vectors."""
        words = text.lower().split()
        
        if self.vectors is None:
            # Fallback: random but deterministic vectors
            np.random.seed(hash(text) % 2**32)
            return np.random.randn(100)
        
        vectors = []
        for word in words:
            if word in self.vectors:
                vectors.append(self.vectors[word])
        
        if not vectors:
            return np.zeros(self.vectors.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def score(self, query: str, candidates: List[str]) -> List[float]:
        """
        Score candidates based on co-occurrence similarity with query.
        
        Args:
            query: The search query
            candidates: List of candidate passages
            
        Returns:
            List of similarity scores (0-1)
        """
        query_vec = self._text_to_vector(query)
        query_norm = np.linalg.norm(query_vec)
        
        if query_norm == 0:
            return [0.0] * len(candidates)
        
        scores = []
        for candidate in candidates:
            cand_vec = self._text_to_vector(candidate)
            cand_norm = np.linalg.norm(cand_vec)
            
            if cand_norm == 0:
                scores.append(0.0)
            else:
                # Cosine similarity
                similarity = np.dot(query_vec, cand_vec) / (query_norm * cand_norm)
                # Normalize to 0-1
                scores.append((similarity + 1) / 2)
        
        return scores
    
    def retrieve(self, query: str, corpus: List[str], top_k: int = 10) -> List[tuple]:
        """
        Retrieve top-k candidates from corpus.
        
        Args:
            query: The search query
            corpus: List of all passages
            top_k: Number of results to return
            
        Returns:
            List of (passage, score) tuples
        """
        scores = self.score(query, corpus)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in indexed_scores[:top_k]:
            results.append((corpus[idx], score))
        
        return results


# Mathematical proof of properties
"""
THEOREM 1: Co-occurrence Scoring Convergence

Given:
- Word vectors W ∈ ℝ^{V×d} from PPMI factorization
- Query q with word set Q
- Candidate c with word set C

The co-occurrence score S(q,c) = cos(avg(W[Q]), avg(W[C])) satisfies:

1. Boundedness: S(q,c) ∈ [-1, 1]
2. Symmetry: S(q,c) = S(c,q)  
3. Self-similarity: S(q,q) = 1

PROOF:
1. Boundedness follows from cosine similarity definition
2. Symmetry: cos(a,b) = cos(b,a) by commutativity of dot product
3. Self-similarity: cos(a,a) = ||a||²/||a||² = 1

QED
"""
