#!/usr/bin/env python3
"""
Learned Gating Module for Multi-Source RAG Fusion

Implements:
- Sigmoid gating: Logistic regression on query features
- Softmax gating: Multi-class classification for source selection
- Comparison with RRF baseline

Author: RAG Second Brain Team
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import json
import re
import random

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


class QueryFeatureExtractor:
    """Extract features from queries for learned gating."""
    
    # Question type patterns
    WHAT_PATTERN = re.compile(r'\bwhat\b', re.IGNORECASE)
    WHO_PATTERN = re.compile(r'\bwho\b', re.IGNORECASE)
    WHEN_PATTERN = re.compile(r'\bwhen\b', re.IGNORECASE)
    WHERE_PATTERN = re.compile(r'\bwhere\b', re.IGNORECASE)
    HOW_PATTERN = re.compile(r'\bhow\b', re.IGNORECASE)
    WHY_PATTERN = re.compile(r'\bwhy\b', re.IGNORECASE)
    WHICH_PATTERN = re.compile(r'\bwhich\b', re.IGNORECASE)
    
    # Entity patterns (simplified NER)
    CAPITALIZED_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b')
    
    def extract(self, query: str) -> np.ndarray:
        """
        Extract feature vector from query.
        
        Features:
        0: Query length (normalized)
        1: Entity count (capitalized phrases)
        2: is_what question
        3: is_who question
        4: is_when question
        5: is_where question
        6: is_how question
        7: is_why question
        8: is_which question
        9: has_comparison words (vs, compare, between, difference)
        10: has_multi_hop indicators (and, also, both, after, before)
        
        Returns:
            11-dimensional feature vector
        """
        features = np.zeros(11, dtype=np.float32)
        
        # Length (normalized by typical query length ~50 chars)
        features[0] = min(len(query) / 100.0, 2.0)
        
        # Entity count
        entities = self.CAPITALIZED_PATTERN.findall(query)
        features[1] = min(len(entities) / 5.0, 2.0)
        
        # Question types
        features[2] = 1.0 if self.WHAT_PATTERN.search(query) else 0.0
        features[3] = 1.0 if self.WHO_PATTERN.search(query) else 0.0
        features[4] = 1.0 if self.WHEN_PATTERN.search(query) else 0.0
        features[5] = 1.0 if self.WHERE_PATTERN.search(query) else 0.0
        features[6] = 1.0 if self.HOW_PATTERN.search(query) else 0.0
        features[7] = 1.0 if self.WHY_PATTERN.search(query) else 0.0
        features[8] = 1.0 if self.WHICH_PATTERN.search(query) else 0.0
        
        # Comparison indicators
        comparison_words = ['vs', 'versus', 'compare', 'comparison', 'between', 'difference']
        features[9] = 1.0 if any(w in query.lower() for w in comparison_words) else 0.0
        
        # Multi-hop indicators
        multi_hop_words = ['and', 'also', 'both', 'after', 'before', 'then', 'while']
        features[10] = 1.0 if any(w in query.lower() for w in multi_hop_words) else 0.0
        
        return features


class SigmoidGating:
    """
    Sigmoid gating: Learn per-source weights via logistic regression.
    
    For each source s, we learn weights w_s such that:
    gate_s(query) = sigmoid(w_s^T * features(query))
    
    Final fusion: sum_s gate_s(query) * score_s(doc)
    """
    
    def __init__(self, n_features: int = 11, n_sources: int = 3, learning_rate: float = 0.01):
        self.n_features = n_features
        self.n_sources = n_sources
        self.learning_rate = learning_rate
        
        # Initialize weights (small random values)
        np.random.seed(RANDOM_SEED)
        self.weights = np.random.randn(n_sources, n_features) * 0.01
        self.bias = np.zeros(n_sources)
        
        self.feature_extractor = QueryFeatureExtractor()
        self.source_names = ['dense', 'bm25', 'entity']
    
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid."""
        return np.where(x >= 0,
                       1 / (1 + np.exp(-x)),
                       np.exp(x) / (1 + np.exp(x)))
    
    def compute_gates(self, query: str) -> np.ndarray:
        """Compute gating weights for each source."""
        features = self.feature_extractor.extract(query)
        logits = self.weights @ features + self.bias
        gates = self.sigmoid(logits)
        return gates
    
    def fuse_scores(self, query: str, source_scores: Dict[str, List[float]]) -> List[float]:
        """
        Fuse scores from multiple sources using learned gates.
        """
        gates = self.compute_gates(query)
        n_docs = len(next(iter(source_scores.values())))
        fused = np.zeros(n_docs)
        
        for i, source in enumerate(self.source_names):
            if source in source_scores:
                fused += gates[i] * np.array(source_scores[source])
        
        return fused.tolist()
    
    def train(self, 
              queries: List[str],
              source_scores: List[Dict[str, List[float]]],
              gold_docs: List[set],
              n_epochs: int = 100) -> Dict[str, float]:
        """Train gating weights using gradient descent."""
        history = {'loss': [], 'recall@10': []}
        
        for epoch in range(n_epochs):
            total_loss = 0.0
            
            for query, scores, gold in zip(queries, source_scores, gold_docs):
                features = self.feature_extractor.extract(query)
                gates = self.compute_gates(query)
                fused = self.fuse_scores(query, scores)
                
                loss, grads = self._compute_gradients(features, gates, scores, fused, gold)
                total_loss += loss
                
                self.weights -= self.learning_rate * grads['weights']
                self.bias -= self.learning_rate * grads['bias']
            
            avg_loss = total_loss / len(queries)
            recall = self._evaluate(queries, source_scores, gold_docs)
            
            history['loss'].append(avg_loss)
            history['recall@10'].append(recall)
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss={avg_loss:.4f}, R@10={recall:.4f}")
        
        return history
    
    def _compute_gradients(self, features, gates, scores, fused, gold):
        """Compute gradients for ranking loss."""
        n_docs = len(fused)
        loss = 0.0
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.zeros_like(self.bias)
        fused_arr = np.array(fused)
        
        for pos_idx in gold:
            if pos_idx >= n_docs:
                continue
            for neg_idx in range(n_docs):
                if neg_idx in gold or neg_idx >= n_docs:
                    continue
                
                margin = 1.0 - (fused_arr[pos_idx] - fused_arr[neg_idx])
                if margin > 0:
                    loss += margin
                    
                    for i, source in enumerate(self.source_names):
                        if source not in scores:
                            continue
                        score_diff = scores[source][neg_idx] - scores[source][pos_idx]
                        gate_grad = gates[i] * (1 - gates[i])
                        grad_weights[i] += score_diff * gate_grad * features
                        grad_bias[i] += score_diff * gate_grad
        
        return loss, {'weights': grad_weights, 'bias': grad_bias}
    
    def _evaluate(self, queries, source_scores, gold_docs, k: int = 10) -> float:
        recalls = []
        for query, scores, gold in zip(queries, source_scores, gold_docs):
            fused = self.fuse_scores(query, scores)
            top_k = set(np.argsort(fused)[-k:][::-1])
            recall = len(top_k & gold) / len(gold) if gold else 0.0
            recalls.append(recall)
        return np.mean(recalls)


class SoftmaxGating:
    """Softmax gating: Multi-class classification for source selection."""
    
    def __init__(self, n_features: int = 11, n_sources: int = 3, learning_rate: float = 0.01):
        self.n_features = n_features
        self.n_sources = n_sources
        self.learning_rate = learning_rate
        
        np.random.seed(RANDOM_SEED)
        self.weights = np.random.randn(n_sources, n_features) * 0.01
        self.bias = np.zeros(n_sources)
        
        self.feature_extractor = QueryFeatureExtractor()
        self.source_names = ['dense', 'bm25', 'entity']
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum()
    
    def compute_gates(self, query: str) -> np.ndarray:
        features = self.feature_extractor.extract(query)
        logits = self.weights @ features + self.bias
        return self.softmax(logits)
    
    def fuse_scores(self, query: str, source_scores: Dict[str, List[float]]) -> List[float]:
        gates = self.compute_gates(query)
        n_docs = len(next(iter(source_scores.values())))
        fused = np.zeros(n_docs)
        
        for i, source in enumerate(self.source_names):
            if source in source_scores:
                fused += gates[i] * np.array(source_scores[source])
        
        return fused.tolist()


class RRFBaseline:
    """Reciprocal Rank Fusion baseline."""
    
    def __init__(self, k: int = 60):
        self.k = k
    
    def fuse_scores(self, query: str, source_rankings: Dict[str, List[int]]):
        scores = defaultdict(float)
        for source, ranking in source_rankings.items():
            for rank, doc_id in enumerate(ranking):
                scores[doc_id] += 1.0 / (self.k + rank + 1)
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def generate_synthetic_data(n_queries: int = 500, n_docs: int = 100):
    """Generate synthetic training data."""
    np.random.seed(RANDOM_SEED)
    
    templates = [
        "What is the capital of {entity}?",
        "Who founded {entity}?",
        "When was {entity} established?",
        "Compare {entity} and {entity2}",
    ]
    
    entities = ["France", "Microsoft", "Einstein", "Tesla", "Amazon", "Google"]
    
    queries, source_scores, gold_docs = [], [], []
    
    for _ in range(n_queries):
        template = random.choice(templates)
        entity = random.choice(entities)
        entity2 = random.choice([e for e in entities if e != entity])
        query = template.format(entity=entity, entity2=entity2)
        queries.append(query)
        
        dense = np.random.rand(n_docs)
        bm25 = np.random.rand(n_docs)
        entity_scores = np.random.rand(n_docs)
        
        gold = set(random.sample(range(n_docs), random.randint(1, 3)))
        for g in gold:
            if random.random() < 0.7: dense[g] += 1.5
            if random.random() < 0.5: bm25[g] += 1.5
            if random.random() < 0.4: entity_scores[g] += 1.5
        
        source_scores.append({
            'dense': dense.tolist(),
            'bm25': bm25.tolist(),
            'entity': entity_scores.tolist()
        })
        gold_docs.append(gold)
    
    return queries, source_scores, gold_docs


if __name__ == "__main__":
    print("Learned Gating Module")
    print("Run run_v15_experiments.py for full evaluation")
