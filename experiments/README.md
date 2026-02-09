# HotpotQA Retrieval Experiments

This directory contains experiments validating our multi-source RAG approach on HotpotQA.

## Results Summary

| Method | R@5 | R@10 | R@20 | Latency (ms) |
|--------|-----|------|------|--------------|
| Dense | 0.586 | 0.703 | 0.797 | 42.4 |
| BM25 | 0.503 | 0.625 | 0.742 | 10.4 |
| Entity | 0.340 | 0.447 | 0.561 | 8.7 |
| **RRF Fusion** | **0.625** | **0.762** | **0.850** | 67.8 |

### Key Findings

1. **RRF fusion +8.5%** over best single source (Dense)
2. **Dense most critical** (13.9% drop when removed)
3. **Each source contributes unique signal**

## Ablation Study

| Configuration | R@10 | Δ from Full |
|---------------|------|-------------|
| Full (all sources) | 0.762 | - |
| - Dense | 0.656 | -13.9% |
| - BM25 | 0.727 | -4.6% |
| - Entity | 0.742 | -2.7% |

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Running Experiments

```bash
# Run full experiment
python hotpotqa_eval.py --samples 1000

# Run simulated (uses pre-computed results)
python run_simulated.py
```

## Directory Structure

```
experiments/
├── retrievers/
│   ├── dense.py       # Sentence-transformer retriever
│   ├── bm25.py        # BM25 lexical retriever
│   └── entity.py      # Entity-based KG proxy
├── fusion/
│   └── rrf.py         # Reciprocal Rank Fusion
├── data/
│   └── download_hotpotqa.py
├── results/
│   └── results.json   # Experiment results
└── hotpotqa_eval.py   # Main evaluation script
```
