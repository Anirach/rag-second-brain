# RAG Second Brain

**Co-occurrence, Sequence, and Knowledge Graph with Ontology as a Second Brain for AI-LLM**

A proof-of-concept implementation of a multi-source RAG framework that combines:
- **Co-occurrence statistics** (GloVe/PPMI-based semantic similarity)
- **Sequential dense retrieval** (Sentence transformers)
- **Knowledge Graph retrieval** (Entity linking + graph traversal)
- **Ontology reasoning** (OWL 2 RL materialization)

## Status

🚧 **Work in Progress** — This is a conceptual implementation for research validation.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the menu-driven demo
python main.py
```

## Menu Options

```
=== RAG Second Brain ===
1. Co-occurrence Module Demo
2. Dense Retrieval Demo
3. Knowledge Graph Demo
4. Ontology Reasoning Demo
5. Gating Mechanism Demo
6. Full Pipeline Demo
7. Run Proof-of-Concept Tests
8. Exit
```

## Project Structure

```
rag-second-brain/
├── main.py                 # Menu-driven entry point
├── src/
│   ├── cooccurrence.py     # Co-occurrence scoring
│   ├── dense_retrieval.py  # Sentence-T5 retrieval
│   ├── kg_retrieval.py     # Knowledge graph module
│   ├── ontology.py         # OWL 2 RL reasoning
│   ├── gating.py           # Learned gating mechanism
│   └── pipeline.py         # Full RAG pipeline
├── tests/
│   └── test_components.py  # Unit tests
├── data/
│   └── sample/             # Small sample datasets
├── paper/                  # LaTeX source
└── requirements.txt
```

## Paper

This code accompanies the paper: *"Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"*

## License

MIT

## Citation

```bibtex
@article{mingkhwan2026secondbrain,
  title={Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM},
  author={Mingkhwan, Anirach},
  year={2026}
}
```
