# 🧠 RAG Second Brain — Multi-Source Retrieval Demo

A working web application demonstrating multi-source Retrieval-Augmented Generation (RAG) for Personal Knowledge Management, as described in our paper:

> **Multi-Source RAG for PKM: Integrating Co-occurrence Statistics, Knowledge Graphs, and Ontological Reasoning via Learned Gating**

## Features

- **🔵 Dense Vector Retrieval** — Semantic similarity via sentence-transformers
- **🟢 Statistical Retrieval** — BM25 + PPMI co-occurrence matrix
- **🟣 Knowledge Graph Retrieval** — Entity extraction + graph traversal
- **⚡ Sigmoid Gating** — Query-adaptive fusion weights
- **🕸️ Interactive KG Visualization** — vis.js entity graph
- **📄 Document Upload** — Add your own documents to the Second Brain
- **📊 Transparent Scoring** — See how each source contributes per result

## Quick Start

### Docker
```bash
docker compose up --build
# Open http://localhost:8000
```

### Manual
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python app.py
# Open http://localhost:8000
```

## Architecture

```
Query → [Query Encoder]
         ↓          ↓          ↓
    [Dense]    [BM25+PPMI]   [KG]
         ↓          ↓          ↓
       [Sigmoid Gating Network]
                 ↓
          [Fused Results]
```

## Pre-loaded Data

20 AI/ML paper abstracts covering: RAG, BERT, Transformers, Knowledge Graphs, Dense Retrieval, Multi-hop QA, NER, PPMI, OWL Ontology, FAISS, MoE, and more.

## Paper Results

| Benchmark | Fusion Gain | Gating vs RRF |
|-----------|-------------|---------------|
| HotpotQA  | +8.4% R@10  | +18.4%        |
| MuSiQue   | +13.8% R@10 | +25.7%        |

## License

MIT
