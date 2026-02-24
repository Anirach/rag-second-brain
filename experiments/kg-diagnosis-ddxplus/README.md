# DDXPlus Experiment Pipeline

**Paper:** "Multi-Source Knowledge Graph as a Second Brain for LLM-Based Clinical Differential Diagnosis" (AIiH 2026)

## Setup

```bash
cd experiments/kg-diagnosis-ddxplus
pip install -r requirements.txt
cp .env.example .env  # Add your OPENAI_API_KEY
```

## Quick Test (10 vignettes, no LLM)

```bash
python run_experiments.py --n-vignettes 10 --skip-llm
```

## Quick Test (10 vignettes, with LLM)

```bash
export OPENAI_API_KEY=sk-...
python run_experiments.py --n-vignettes 10
```

## Full Run (900 vignettes)

```bash
python run_experiments.py --n-per-group 300
```

## Run Specific Conditions

```bash
python run_experiments.py --n-vignettes 50 --conditions B1_LLM_only,Proposed
```

## Project Structure

```
├── data/loader.py          # DDXPlus dataset loading & sampling
├── kg/builder.py           # Knowledge graph construction (SQLite)
├── retrieval/
│   ├── dense.py            # Sentence-transformer semantic retrieval
│   ├── statistical.py      # BM25 + PPMI co-occurrence
│   ├── kg.py               # Graph traversal retrieval
│   └── fusion.py           # Learned gating fusion
├── experiments/
│   └── conditions.py       # B1, B2, B3, Proposed, Ablations (A1-A4)
├── eval/metrics.py         # Top-k accuracy, NDCG@5, F1, hallucination rate
├── run_experiments.py       # Main orchestrator
├── results/                # Output directory
│   ├── metrics.json
│   ├── results_table.txt
│   ├── results_table.tex   # LaTeX for paper
│   └── summary.json
└── cache/                  # Cached data & LLM responses
```

## Experimental Conditions

| ID | Condition | Description |
|----|-----------|-------------|
| B1 | LLM-only | Direct inference, no retrieval |
| B2 | LLM + Dense RAG | Sentence-transformer retrieval |
| B3 | LLM + BM25 | Statistical co-occurrence retrieval |
| **Proposed** | **Multi-Source KG** | **All 3 sources + learned gating** |
| A1 | No dense | Ablation: remove semantic retrieval |
| A2 | No statistical | Ablation: remove BM25/PPMI |
| A3 | No KG | Ablation: remove graph traversal |
| A4 | No gating | Ablation: equal weights |
