#!/usr/bin/env python3
"""
End-to-End QA Evaluation Module

Implements:
- Exact Match (EM) metric
- Token-level F1 score
- Full QA pipeline: retrieve → generate → evaluate
- Per-query-type breakdown (bridge vs comparison)

Author: RAG Second Brain Team
"""

import re
import string
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import Counter
import json
from pathlib import Path

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)


def normalize_answer(text: str) -> str:
    """
    Normalize answer text for comparison.
    
    Follows SQuAD evaluation protocol:
    - Lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Remove extra whitespace
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    return ' '.join(remove_articles(remove_punctuation(text.lower())).split())


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match score."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if len(gold_tokens) == 0 or len(pred_tokens) == 0:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        return 0.0
    
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_em_f1_batch(
    predictions: List[str],
    ground_truths: List[str]
) -> Dict[str, float]:
    """Compute EM and F1 over a batch of predictions."""
    assert len(predictions) == len(ground_truths)
    
    ems = [compute_exact_match(p, g) for p, g in zip(predictions, ground_truths)]
    f1s = [compute_f1(p, g) for p, g in zip(predictions, ground_truths)]
    
    return {
        'em': np.mean(ems),
        'f1': np.mean(f1s),
        'em_std': np.std(ems),
        'f1_std': np.std(f1s),
        'per_query_em': ems,
        'per_query_f1': f1s
    }


class QueryTypeClassifier:
    """Classify HotpotQA questions into types."""
    
    BRIDGE_PATTERNS = [
        r'\b(and|also|both)\b.*\b(who|what|where)\b',
        r'.*\b(that|which)\b.*\b(was|is|are)\b.*\b(founded|born|created)\b',
    ]
    
    COMPARISON_PATTERNS = [
        r'\bcompare\b', r'\bvs\.?\b', r'\bversus\b', r'\bdifference\b',
        r'\bmore\b.*\bthan\b', r'\bless\b.*\bthan\b',
        r'\bwhich\b.*\b(one|is)\b.*\b(more|less|larger|smaller)\b',
    ]
    
    def __init__(self):
        self.bridge_re = [re.compile(p, re.IGNORECASE) for p in self.BRIDGE_PATTERNS]
        self.comparison_re = [re.compile(p, re.IGNORECASE) for p in self.COMPARISON_PATTERNS]
    
    def classify(self, query: str) -> str:
        """Classify query type: 'bridge', 'comparison', or 'other'"""
        for pattern in self.comparison_re:
            if pattern.search(query):
                return 'comparison'
        
        for pattern in self.bridge_re:
            if pattern.search(query):
                return 'bridge'
        
        return 'other'


class QAEvaluator:
    """Full QA evaluation pipeline."""
    
    def __init__(self):
        self.query_classifier = QueryTypeClassifier()
    
    def evaluate_retrieval(
        self,
        queries: List[str],
        retrieved_docs: List[List[str]],
        gold_docs: List[Set[int]],
        k_values: List[int] = [5, 10, 20]
    ) -> Dict[str, float]:
        """Evaluate retrieval quality."""
        results = {}
        
        for k in k_values:
            recalls = []
            for retrieved, gold in zip(retrieved_docs, gold_docs):
                top_k = set(range(min(k, len(retrieved))))
                recall = len(top_k & gold) / len(gold) if gold else 0.0
                recalls.append(recall)
            
            results[f'recall@{k}'] = {
                'mean': np.mean(recalls),
                'std': np.std(recalls)
            }
        
        return results
    
    def evaluate_qa(
        self,
        queries: List[str],
        predictions: List[str],
        gold_answers: List[str]
    ) -> Dict:
        """Evaluate end-to-end QA performance."""
        metrics = compute_em_f1_batch(predictions, gold_answers)
        
        # Per-query-type breakdown
        type_results = {
            'bridge': {'em': [], 'f1': []},
            'comparison': {'em': [], 'f1': []},
            'other': {'em': [], 'f1': []}
        }
        
        for query, pred, gold in zip(queries, predictions, gold_answers):
            qtype = self.query_classifier.classify(query)
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            
            type_results[qtype]['em'].append(em)
            type_results[qtype]['f1'].append(f1)
        
        metrics['per_type'] = {}
        for qtype, scores in type_results.items():
            if scores['em']:
                metrics['per_type'][qtype] = {
                    'count': len(scores['em']),
                    'em': np.mean(scores['em']),
                    'f1': np.mean(scores['f1']),
                    'em_std': np.std(scores['em']),
                    'f1_std': np.std(scores['f1'])
                }
        
        return metrics


if __name__ == "__main__":
    print("QA Evaluation Module")
    print("Run run_v15_experiments.py for full evaluation")
