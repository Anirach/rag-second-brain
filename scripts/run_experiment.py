#!/usr/bin/env python3
"""Main experiment script for RAG Second Brain v16.

This script runs the full experiment pipeline:
1. Load HotpotQA data
2. Build all retrievers (Dense, PPMI, KG)
3. Run fusion experiments (RRF, Learned Gating)
4. Evaluate and report results
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Set
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.retrieval import DenseRetriever, PPMIRetriever, KnowledgeGraphRetriever
from src.fusion import rrf_fusion
from src.evaluation.retrieval import (
    recall_at_k, both_support_recall, 
    compute_retrieval_metrics, compute_hotpotqa_metrics,
    bootstrap_ci
)
from src.utils.data_loader import load_hotpotqa


def run_experiment(
    n_samples: int = 500,
    ks: List[int] = [5, 10, 20, 50],
    output_dir: str = "experiments/results"
):
    """Run the full experiment.
    
    Args:
        n_samples: Number of samples to evaluate (for faster testing).
        ks: K values for Recall@K.
        output_dir: Directory for results.
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {"config": {"n_samples": n_samples, "ks": ks}}
    
    # ========================================
    # Step 1: Load Data
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 1: Loading HotpotQA data...")
    logger.info("=" * 50)
    
    start = time.time()
    questions, documents, doc_id_to_idx = load_hotpotqa(
        split="validation", 
        subset="distractor"
    )
    load_time = time.time() - start
    
    logger.info(f"Loaded {len(questions)} questions, {len(documents)} documents in {load_time:.1f}s")
    
    # Subsample for faster experiments
    if n_samples and n_samples < len(questions):
        np.random.seed(42)
        indices = np.random.choice(len(questions), n_samples, replace=False)
        questions = [questions[i] for i in indices]
        logger.info(f"Subsampled to {len(questions)} questions")
    
    results["data"] = {
        "n_questions": len(questions),
        "n_documents": len(documents),
        "load_time": load_time
    }
    
    # ========================================
    # Step 2: Build Dense Retriever
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 2: Building Dense Retriever...")
    logger.info("=" * 50)
    
    start = time.time()
    dense = DenseRetriever(model_name="intfloat/e5-base-v2")  # Use base for speed
    dense.build_index(documents, batch_size=32)
    dense_build_time = time.time() - start
    
    logger.info(f"Dense index built in {dense_build_time:.1f}s")
    results["dense"] = {"build_time": dense_build_time}
    
    # ========================================
    # Step 3: Build PPMI Retriever
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 3: Building PPMI Retriever...")
    logger.info("=" * 50)
    
    start = time.time()
    ppmi = PPMIRetriever(window_size=5, min_count=3)
    ppmi.build_from_corpus(documents)
    ppmi_build_time = time.time() - start
    
    logger.info(f"PPMI index built in {ppmi_build_time:.1f}s")
    logger.info(f"Vocabulary size: {len(ppmi.vocab)}")
    results["ppmi"] = {
        "build_time": ppmi_build_time,
        "vocab_size": len(ppmi.vocab)
    }
    
    # ========================================
    # Step 4: Build KG Retriever
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 4: Building KG Retriever...")
    logger.info("=" * 50)
    
    start = time.time()
    kg = KnowledgeGraphRetriever()
    kg.build_from_documents(documents)
    kg_build_time = time.time() - start
    
    logger.info(f"KG built in {kg_build_time:.1f}s")
    logger.info(f"Entities: {kg.graph.number_of_nodes()}, Edges: {kg.graph.number_of_edges()}")
    results["kg"] = {
        "build_time": kg_build_time,
        "n_entities": kg.graph.number_of_nodes(),
        "n_edges": kg.graph.number_of_edges()
    }
    
    # ========================================
    # Step 5: Run Retrieval
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 5: Running Retrieval...")
    logger.info("=" * 50)
    
    max_k = max(ks)
    
    all_dense_retrieved = []
    all_ppmi_retrieved = []
    all_kg_retrieved = []
    all_rrf_retrieved = []
    all_support_pairs = []
    
    start = time.time()
    for i, q in enumerate(questions):
        if (i + 1) % 100 == 0:
            logger.info(f"Processing query {i+1}/{len(questions)}")
        
        query = q["question"]
        support_docs = q["support_doc_indices"]
        all_support_pairs.append(support_docs)
        
        # Dense retrieval
        dense_results = dense.retrieve(query, k=max_k)
        all_dense_retrieved.append([doc_id for doc_id, _ in dense_results])
        
        # PPMI retrieval
        ppmi_results = ppmi.retrieve(query, k=max_k)
        all_ppmi_retrieved.append([doc_id for doc_id, _ in ppmi_results])
        
        # KG retrieval
        kg_results = kg.retrieve(query, k=max_k)
        all_kg_retrieved.append([doc_id for doc_id, _ in kg_results])
        
        # RRF fusion
        rrf_results = rrf_fusion([dense_results, ppmi_results, kg_results], top_n=max_k)
        all_rrf_retrieved.append([doc_id for doc_id, _ in rrf_results])
    
    retrieval_time = time.time() - start
    logger.info(f"Retrieval completed in {retrieval_time:.1f}s")
    
    # ========================================
    # Step 6: Evaluate
    # ========================================
    logger.info("=" * 50)
    logger.info("Step 6: Evaluating...")
    logger.info("=" * 50)
    
    def evaluate_method(name: str, retrieved: List[List[int]]) -> Dict:
        """Evaluate a retrieval method."""
        metrics = {}
        for k in ks:
            recalls = []
            both_recalls = []
            for ret, sup in zip(retrieved, all_support_pairs):
                recalls.append(recall_at_k(ret, set(sup), k))
                both_recalls.append(both_support_recall(ret, sup, k))
            
            mean, lower, upper = bootstrap_ci(recalls)
            metrics[f"recall@{k}"] = mean
            metrics[f"recall@{k}_ci"] = [lower, upper]
            
            both_mean, both_lower, both_upper = bootstrap_ci(both_recalls)
            metrics[f"both@{k}"] = both_mean
            metrics[f"both@{k}_ci"] = [both_lower, both_upper]
        
        return metrics
    
    results["metrics"] = {
        "dense": evaluate_method("Dense", all_dense_retrieved),
        "ppmi": evaluate_method("PPMI", all_ppmi_retrieved),
        "kg": evaluate_method("KG", all_kg_retrieved),
        "rrf": evaluate_method("RRF", all_rrf_retrieved),
    }
    
    # ========================================
    # Step 7: Report Results
    # ========================================
    logger.info("=" * 50)
    logger.info("Results")
    logger.info("=" * 50)
    
    print("\n" + "=" * 70)
    print(f"{'Method':<12} | {'R@5':>8} | {'R@10':>8} | {'R@20':>8} | {'Both@10':>8}")
    print("-" * 70)
    
    for method in ["dense", "ppmi", "kg", "rrf"]:
        m = results["metrics"][method]
        r5 = m.get("recall@5", 0)
        r10 = m.get("recall@10", 0)
        r20 = m.get("recall@20", 0)
        b10 = m.get("both@10", 0)
        print(f"{method.upper():<12} | {r5:>8.3f} | {r10:>8.3f} | {r20:>8.3f} | {b10:>8.3f}")
    
    print("=" * 70)
    
    # Calculate improvements
    dense_r10 = results["metrics"]["dense"]["recall@10"]
    rrf_r10 = results["metrics"]["rrf"]["recall@10"]
    improvement = ((rrf_r10 - dense_r10) / dense_r10) * 100
    
    print(f"\n📊 RRF improves over Dense by {improvement:.1f}% relative at R@10")
    
    # Save results
    output_path = Path(output_dir) / "experiment_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {output_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run RAG Second Brain experiment")
    parser.add_argument("--n_samples", type=int, default=500, 
                       help="Number of samples (default: 500)")
    parser.add_argument("--output_dir", type=str, default="experiments/results",
                       help="Output directory")
    args = parser.parse_args()
    
    run_experiment(
        n_samples=args.n_samples,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
