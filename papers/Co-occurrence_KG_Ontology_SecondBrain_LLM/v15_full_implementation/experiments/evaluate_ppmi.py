"""
Evaluate PPMI retriever on HotpotQA and compare with BM25/Dense baselines.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from tqdm import tqdm
from collections import defaultdict
import time
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ppmi_retriever import CooccurrenceMatrix, PPMIRetriever


def load_hotpotqa(path: str, n_samples: int = 1000) -> List[Dict]:
    """Load HotpotQA dev set."""
    with open(path) as f:
        data = json.load(f)
    return data[:n_samples]


def get_gold_passages(item: Dict) -> set:
    """Get gold supporting passage titles."""
    return set([fact[0] for fact in item.get("supporting_facts", [])])


def evaluate_retriever(
    retriever_fn,
    data: List[Dict],
    k_values: List[int] = [5, 10, 20]
) -> Dict[str, float]:
    """
    Evaluate retriever on HotpotQA.
    retriever_fn: function(query, passages) -> ranked_indices
    """
    recalls = {k: [] for k in k_values}
    latencies = []
    
    for item in tqdm(data, desc="Evaluating"):
        question = item["question"]
        context = item["context"]  # List of (title, sentences)
        gold_titles = get_gold_passages(item)
        
        # Flatten passages
        passages = []
        titles = []
        for title, sentences in context:
            passages.append(" ".join(sentences))
            titles.append(title)
        
        # Retrieve
        start = time.time()
        ranked_indices = retriever_fn(question, passages)
        latencies.append((time.time() - start) * 1000)  # ms
        
        # Compute recall@k
        for k in k_values:
            top_k_titles = set([titles[i] for i in ranked_indices[:k]])
            recall = len(top_k_titles & gold_titles) / len(gold_titles) if gold_titles else 0
            recalls[k].append(recall)
    
    results = {
        f"recall@{k}": np.mean(recalls[k]) for k in k_values
    }
    results["latency_ms"] = np.mean(latencies)
    return results


class DenseRetriever:
    """Dense retriever using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        
    def retrieve(self, query: str, passages: List[str]) -> List[int]:
        """Return passage indices ranked by similarity."""
        query_emb = self.model.encode([query])
        passage_embs = self.model.encode(passages)
        
        similarities = cosine_similarity(query_emb, passage_embs)[0]
        ranked = np.argsort(-similarities)
        return ranked.tolist()


class BM25Retriever:
    """BM25 retriever."""
    
    def retrieve(self, query: str, passages: List[str]) -> List[int]:
        """Return passage indices ranked by BM25 score."""
        tokenized = [p.lower().split() for p in passages]
        bm25 = BM25Okapi(tokenized)
        
        query_tokens = query.lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = np.argsort(-scores)
        return ranked.tolist()


class PPMIRetrieverWrapper:
    """Wrapper for PPMI retriever."""
    
    def __init__(self, ppmi_matrix: CooccurrenceMatrix):
        self.retriever = PPMIRetriever(ppmi_matrix)
        self.ppmi = ppmi_matrix
        
    def retrieve(self, query: str, passages: List[str]) -> List[int]:
        """Return passage indices ranked by PPMI score."""
        results = self.retriever.retrieve(query, passages, top_k=len(passages))
        return [idx for idx, _ in results]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/tmp/hotpotqa_dev.json", help="HotpotQA data path")
    parser.add_argument("--ppmi-path", default="./ppmi_model", help="PPMI model path")
    parser.add_argument("--n-samples", type=int, default=1000, help="Number of samples")
    parser.add_argument("--build-ppmi", action="store_true", help="Build PPMI from scratch")
    parser.add_argument("--output", default="./results/ppmi_comparison.json", help="Output path")
    args = parser.parse_args()
    
    # Load data
    print(f"Loading HotpotQA from {args.data}...")
    data = load_hotpotqa(args.data, args.n_samples)
    print(f"Loaded {len(data)} samples")
    
    # Build or load PPMI
    ppmi = CooccurrenceMatrix()
    if args.build_ppmi or not Path(args.ppmi_path).exists():
        print("Building PPMI matrix from corpus...")
        # Extract all passages for PPMI training
        all_passages = []
        for item in data:
            for title, sentences in item["context"]:
                all_passages.append(" ".join(sentences))
        
        ppmi.build_vocab(all_passages, max_vocab=20000)
        ppmi.build_cooccurrence(all_passages)
        ppmi.compute_ppmi()
        ppmi.sparsify_top_n()
        ppmi.save(args.ppmi_path)
    else:
        ppmi.load(args.ppmi_path)
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    dense = DenseRetriever()
    bm25 = BM25Retriever()
    ppmi_ret = PPMIRetrieverWrapper(ppmi)
    
    # Evaluate each
    results = {}
    
    print("\n=== Evaluating Dense ===")
    results["dense"] = evaluate_retriever(dense.retrieve, data)
    print(f"Results: {results['dense']}")
    
    print("\n=== Evaluating BM25 ===")
    results["bm25"] = evaluate_retriever(bm25.retrieve, data)
    print(f"Results: {results['bm25']}")
    
    print("\n=== Evaluating PPMI ===")
    results["ppmi"] = evaluate_retriever(ppmi_ret.retrieve, data)
    print(f"Results: {results['ppmi']}")
    
    # Fusion: RRF
    print("\n=== Evaluating RRF Fusion (Dense + BM25 + PPMI) ===")
    def rrf_fusion(query: str, passages: List[str], k: int = 60) -> List[int]:
        dense_ranks = dense.retrieve(query, passages)
        bm25_ranks = bm25.retrieve(query, passages)
        ppmi_ranks = ppmi_ret.retrieve(query, passages)
        
        scores = defaultdict(float)
        for ranks in [dense_ranks, bm25_ranks, ppmi_ranks]:
            for rank, idx in enumerate(ranks):
                scores[idx] += 1.0 / (k + rank + 1)
        
        sorted_indices = sorted(scores.keys(), key=lambda x: -scores[x])
        return sorted_indices
    
    results["rrf_fusion"] = evaluate_retriever(rrf_fusion, data)
    print(f"Results: {results['rrf_fusion']}")
    
    # Save results
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")
    
    # Print comparison table
    print("\n" + "="*60)
    print("COMPARISON TABLE")
    print("="*60)
    print(f"{'Method':<15} {'R@5':>8} {'R@10':>8} {'R@20':>8} {'Latency':>10}")
    print("-"*60)
    for method, res in results.items():
        print(f"{method:<15} {res['recall@5']:>8.3f} {res['recall@10']:>8.3f} {res['recall@20']:>8.3f} {res['latency_ms']:>8.1f}ms")
    
    # Calculate improvements
    best_single = max(results["dense"]["recall@10"], results["bm25"]["recall@10"], results["ppmi"]["recall@10"])
    fusion_r10 = results["rrf_fusion"]["recall@10"]
    improvement = (fusion_r10 - best_single) / best_single * 100
    print(f"\nFusion improvement over best single: {improvement:.1f}%")


if __name__ == "__main__":
    main()
