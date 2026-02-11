"""Question Answering evaluation metrics.

Implements standard QA metrics:
- Exact Match (EM)
- F1 Score (token-level)
- Supporting Fact metrics (for HotpotQA)
"""

from typing import List, Tuple, Set
import re
import string
from collections import Counter


def normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.
    
    Lowercases, removes articles, punctuation, and extra whitespace.
    
    Args:
        s: Answer string.
        
    Returns:
        Normalized string.
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match(prediction: str, ground_truth: str) -> float:
    """Compute Exact Match score.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Gold answer.
        
    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 score.
    
    Args:
        prediction: Predicted answer.
        ground_truth: Gold answer.
        
    Returns:
        F1 score.
    """
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()
    
    if not pred_tokens or not gold_tokens:
        return float(pred_tokens == gold_tokens)
    
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    
    if num_common == 0:
        return 0.0
    
    precision = num_common / len(pred_tokens)
    recall = num_common / len(gold_tokens)
    
    return 2 * precision * recall / (precision + recall)


def compute_em_f1(
    prediction: str, 
    ground_truths: List[str]
) -> Tuple[float, float]:
    """Compute EM and F1 against multiple ground truths.
    
    Takes the max score across all ground truths.
    
    Args:
        prediction: Predicted answer.
        ground_truths: List of acceptable gold answers.
        
    Returns:
        Tuple of (max_em, max_f1).
    """
    em_scores = [exact_match(prediction, gt) for gt in ground_truths]
    f1_scores = [f1_score(prediction, gt) for gt in ground_truths]
    
    return max(em_scores), max(f1_scores)


def supporting_fact_em(
    pred_facts: List[Tuple[str, int]],
    gold_facts: List[Tuple[str, int]]
) -> float:
    """Compute Supporting Fact Exact Match.
    
    For HotpotQA: checks if predicted supporting facts exactly
    match the gold supporting facts.
    
    Args:
        pred_facts: Predicted (title, sentence_idx) pairs.
        gold_facts: Gold (title, sentence_idx) pairs.
        
    Returns:
        1.0 if exact match, 0.0 otherwise.
    """
    return float(set(pred_facts) == set(gold_facts))


def supporting_fact_f1(
    pred_facts: List[Tuple[str, int]],
    gold_facts: List[Tuple[str, int]]
) -> Tuple[float, float, float]:
    """Compute Supporting Fact Precision, Recall, F1.
    
    Args:
        pred_facts: Predicted supporting facts.
        gold_facts: Gold supporting facts.
        
    Returns:
        Tuple of (precision, recall, f1).
    """
    pred_set = set(pred_facts)
    gold_set = set(gold_facts)
    
    if not pred_set:
        precision = 0.0
    else:
        precision = len(pred_set & gold_set) / len(pred_set)
    
    if not gold_set:
        recall = 0.0
    else:
        recall = len(pred_set & gold_set) / len(gold_set)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    
    return precision, recall, f1


def compute_qa_metrics(
    predictions: List[str],
    ground_truths: List[List[str]]
) -> dict:
    """Compute QA metrics across a dataset.
    
    Args:
        predictions: List of predicted answers.
        ground_truths: List of gold answer lists (multiple acceptable per question).
        
    Returns:
        Dictionary of metrics.
    """
    ems = []
    f1s = []
    
    for pred, gts in zip(predictions, ground_truths):
        em, f1 = compute_em_f1(pred, gts)
        ems.append(em)
        f1s.append(f1)
    
    import numpy as np
    
    return {
        "em": np.mean(ems),
        "em_std": np.std(ems),
        "f1": np.mean(f1s),
        "f1_std": np.std(f1s),
        "n_samples": len(predictions)
    }


def compute_hotpotqa_full_metrics(
    answer_predictions: List[str],
    answer_golds: List[str],
    sp_predictions: List[List[Tuple[str, int]]],
    sp_golds: List[List[Tuple[str, int]]]
) -> dict:
    """Compute full HotpotQA metrics (Answer + Supporting Facts).
    
    Args:
        answer_predictions: Predicted answers.
        answer_golds: Gold answers.
        sp_predictions: Predicted supporting facts.
        sp_golds: Gold supporting facts.
        
    Returns:
        Full metrics dictionary.
    """
    import numpy as np
    
    # Answer metrics
    ans_ems = [exact_match(p, g) for p, g in zip(answer_predictions, answer_golds)]
    ans_f1s = [f1_score(p, g) for p, g in zip(answer_predictions, answer_golds)]
    
    # Supporting fact metrics
    sp_ems = [supporting_fact_em(p, g) for p, g in zip(sp_predictions, sp_golds)]
    sp_f1s = [supporting_fact_f1(p, g)[2] for p, g in zip(sp_predictions, sp_golds)]
    
    # Joint metrics (both answer AND supporting facts correct)
    joint_ems = [
        float(ae == 1.0 and se == 1.0) 
        for ae, se in zip(ans_ems, sp_ems)
    ]
    joint_f1s = [
        af * sf for af, sf in zip(ans_f1s, sp_f1s)
    ]
    
    return {
        "answer_em": np.mean(ans_ems),
        "answer_f1": np.mean(ans_f1s),
        "sp_em": np.mean(sp_ems),
        "sp_f1": np.mean(sp_f1s),
        "joint_em": np.mean(joint_ems),
        "joint_f1": np.mean(joint_f1s),
    }
