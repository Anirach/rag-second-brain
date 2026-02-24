#!/usr/bin/env python3
"""Main orchestrator for DDXPlus differential diagnosis experiments.

Usage:
    python run_experiments.py [--n-vignettes N] [--n-per-group N] [--conditions COND1,COND2,...]
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

# Setup path
sys.path.insert(0, str(Path(__file__).parent))

from data.loader import (
    load_ddxplus, sample_vignettes, split_vignettes,
    get_all_pathologies, get_training_data, ALL_KNOWN_PATHOLOGIES,
    PATHOLOGY_GROUPS, PATHOLOGY_TO_GROUP,
)
from kg.builder import KnowledgeGraph
from retrieval.dense import DenseRetriever
from retrieval.statistical import StatisticalRetriever
from retrieval.kg import KGRetriever
from retrieval.fusion import GatingFusion
from experiments.conditions import (
    run_b1_llm_only, run_b2_dense_rag, run_b3_bm25, run_proposed, CONDITIONS,
)
from eval.metrics import (
    compute_all_metrics, aggregate_metrics, bootstrap_ci,
    mcnemar_test, format_results_table,
)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("results/experiment.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)


def run_condition(
    condition: str,
    vignette: dict,
    dense_retriever,
    stat_retriever,
    kg_retriever,
    fusion,
) -> list:
    """Run a single experimental condition on a vignette."""
    if condition == "B1_LLM_only":
        return run_b1_llm_only(vignette)
    elif condition == "B2_Dense_RAG":
        return run_b2_dense_rag(vignette, dense_retriever)
    elif condition == "B3_BM25":
        return run_b3_bm25(vignette, stat_retriever)
    elif condition == "Proposed":
        return run_proposed(vignette, dense_retriever, stat_retriever, kg_retriever, fusion, use_gating=True)
    elif condition == "A1_no_dense":
        return run_proposed(vignette, dense_retriever, stat_retriever, kg_retriever, fusion, exclude_sources=["dense"])
    elif condition == "A2_no_statistical":
        return run_proposed(vignette, dense_retriever, stat_retriever, kg_retriever, fusion, exclude_sources=["statistical"])
    elif condition == "A3_no_kg":
        return run_proposed(vignette, dense_retriever, stat_retriever, kg_retriever, fusion, exclude_sources=["kg"])
    elif condition == "A4_no_gating":
        return run_proposed(vignette, dense_retriever, stat_retriever, kg_retriever, fusion, use_gating=False)
    else:
        raise ValueError(f"Unknown condition: {condition}")


def main():
    parser = argparse.ArgumentParser(description="DDXPlus Experiment Pipeline")
    parser.add_argument("--n-per-group", type=int, default=300, help="Vignettes per disease group")
    parser.add_argument("--n-vignettes", type=int, default=None, help="Override: total vignettes (for quick test)")
    parser.add_argument("--conditions", type=str, default=None, help="Comma-separated conditions to run")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM calls (for testing retrieval only)")
    args = parser.parse_args()

    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("DDXPlus Experiment Pipeline")
    logger.info("Multi-Source KG as Second Brain for LLM Differential Diagnosis")
    logger.info("=" * 60)

    # Step 1: Load data
    logger.info("\n[Step 1] Loading DDXPlus dataset...")
    t0 = time.time()
    data = load_ddxplus()
    df = data["data"]
    logger.info(f"Loaded {len(df)} samples in {time.time()-t0:.1f}s")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Pathologies: {df['PATHOLOGY'].nunique()} unique")

    # Check pathology mapping coverage
    all_pathologies = get_all_pathologies(df)
    mapped = sum(1 for p in all_pathologies if p in PATHOLOGY_TO_GROUP)
    logger.info(f"Pathology mapping: {mapped}/{len(all_pathologies)} mapped to groups")
    unmapped = [p for p in all_pathologies if p not in PATHOLOGY_TO_GROUP]
    if unmapped:
        logger.info(f"Unmapped pathologies: {unmapped[:20]}")

    # Step 2: Sample vignettes
    logger.info("\n[Step 2] Sampling vignettes...")
    n_per_group = args.n_per_group
    if args.n_vignettes:
        n_per_group = args.n_vignettes // 3

    vignettes = sample_vignettes(df, n_per_group=n_per_group)

    if args.n_vignettes and len(vignettes) > args.n_vignettes:
        vignettes = vignettes[:args.n_vignettes]

    test_set, val_set = split_vignettes(vignettes)
    logger.info(f"Test: {len(test_set)}, Validation: {len(val_set)}")

    # Step 3: Build KG
    logger.info("\n[Step 3] Building Knowledge Graph...")
    t0 = time.time()
    train_df = get_training_data(df)
    kg = KnowledgeGraph(db_path="cache/kg.db")
    kg_stats = kg.build_from_dataframe(train_df)
    logger.info(f"KG built in {time.time()-t0:.1f}s")
    logger.info(f"KG stats: {json.dumps(kg_stats, indent=2)}")

    # Save KG stats
    with open(results_dir / "kg_statistics.json", "w") as f:
        json.dump(kg_stats, f, indent=2)

    # Step 4: Build retrieval indices
    logger.info("\n[Step 4] Building retrieval indices...")

    dense_retriever = DenseRetriever()
    dense_retriever.build_index(kg)

    stat_retriever = StatisticalRetriever()
    stat_retriever.build_index(kg)

    kg_retriever = KGRetriever()
    kg_retriever.build_index(kg)

    fusion = GatingFusion()

    # Step 5: Run experiments
    logger.info("\n[Step 5] Running experimental conditions...")

    conditions_to_run = list(CONDITIONS.keys())
    if args.conditions:
        conditions_to_run = args.conditions.split(",")

    valid_pathologies = all_pathologies

    all_results = {}
    all_predictions = {}
    condition_correct = {}  # for McNemar's test

    for condition in conditions_to_run:
        logger.info(f"\n--- Running {condition}: {CONDITIONS[condition]['name']} ---")
        metrics_list = []
        predictions = []
        correct_list = []

        for i, vignette in enumerate(test_set):
            if args.skip_llm:
                # Just test retrieval
                predicted = []
                if condition in ["Proposed", "A1_no_dense", "A2_no_statistical", "A3_no_kg", "A4_no_gating"]:
                    symptom_list = vignette.get("symptoms", [])
                    dense_r = dense_retriever.retrieve(symptom_list, top_k=10)
                    stat_r = stat_retriever.retrieve(symptom_list, top_k=10)
                    kg_r = kg_retriever.retrieve(symptom_list, top_k=10)
                    fused = fusion.fuse(dense_r, stat_r, kg_r, symptom_list)
                    predicted = [d for d, _ in fused]
                else:
                    predicted = [d for d, _ in dense_retriever.retrieve(vignette.get("symptoms", []), top_k=10)]
            else:
                try:
                    predicted = run_condition(
                        condition, vignette,
                        dense_retriever, stat_retriever, kg_retriever, fusion,
                    )
                except Exception as e:
                    logger.error(f"Error on vignette {i}: {e}")
                    predicted = []

            m = compute_all_metrics(predicted, vignette, valid_pathologies)
            metrics_list.append(m)
            predictions.append({"vignette_id": vignette["id"], "predicted": predicted, "ground_truth": vignette["pathology"]})
            correct_list.append(m["top1_acc"] > 0)

            if (i + 1) % 10 == 0:
                running_top1 = np.mean([m["top1_acc"] for m in metrics_list])
                logger.info(f"  [{i+1}/{len(test_set)}] Running Top-1: {running_top1:.3f}")

        agg = aggregate_metrics(metrics_list)

        # Bootstrap CI for top-1 accuracy
        top1_values = [m["top1_acc"] for m in metrics_list]
        if top1_values:
            ci_low, ci_high = bootstrap_ci(top1_values)
            agg["top1_acc_ci_lower"] = ci_low
            agg["top1_acc_ci_upper"] = ci_high

        all_results[condition] = agg
        all_predictions[condition] = predictions
        condition_correct[condition] = correct_list

        logger.info(f"  Results: Top-1={agg.get('top1_acc_mean',0):.3f}, Top-3={agg.get('top3_acc_mean',0):.3f}, "
                     f"Top-5={agg.get('top5_acc_mean',0):.3f}, NDCG@5={agg.get('ndcg5_mean',0):.3f}, "
                     f"F1={agg.get('f1_mean',0):.3f}, Halluc={agg.get('hallucination_rate_mean',0):.3f}")

    # Step 6: Statistical tests
    logger.info("\n[Step 6] Statistical tests...")
    stat_tests = {}
    if "Proposed" in condition_correct:
        for cond in condition_correct:
            if cond != "Proposed" and len(condition_correct[cond]) == len(condition_correct["Proposed"]):
                p = mcnemar_test(condition_correct[cond], condition_correct["Proposed"])
                stat_tests[f"Proposed_vs_{cond}"] = p
                logger.info(f"  McNemar Proposed vs {cond}: p={p:.4f}")

    # Step 7: Save results
    logger.info("\n[Step 7] Saving results...")

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    with open(results_dir / "predictions.json", "w") as f:
        json.dump(all_predictions, f, indent=2)

    with open(results_dir / "statistical_tests.json", "w") as f:
        json.dump(stat_tests, f, indent=2)

    # Text table
    table_text = format_results_table(all_results, latex=False)
    with open(results_dir / "results_table.txt", "w") as f:
        f.write(table_text)
    logger.info(f"\n{table_text}")

    # LaTeX table
    table_latex = format_results_table(all_results, latex=True)
    with open(results_dir / "results_table.tex", "w") as f:
        f.write(table_latex)

    # Summary
    summary = {
        "dataset_size": len(df),
        "n_vignettes": len(vignettes),
        "n_test": len(test_set),
        "n_val": len(val_set),
        "kg_statistics": kg_stats,
        "conditions_run": conditions_to_run,
        "results": all_results,
        "statistical_tests": stat_tests,
    }
    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "=" * 60)
    logger.info("Experiment complete! Results saved to results/")
    logger.info("=" * 60)

    kg.close()


if __name__ == "__main__":
    main()
