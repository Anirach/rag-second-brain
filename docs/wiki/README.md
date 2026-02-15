# RAG Second Brain - Code Wiki

## 📖 Project Overview

**RAG Second Brain** is a proof-of-concept implementation of a multi-source retrieval-augmented generation (RAG) framework that combines statistical, dense, and knowledge-based retrieval methods with ontology reasoning to create an AI "second brain" system.

### 🎯 Core Concept

The system implements a novel approach to information retrieval by combining:
- **Co-occurrence Statistics** - GloVe/PPMI-based semantic similarity
- **Dense Retrieval** - Sentence transformer embeddings  
- **Knowledge Graph Traversal** - Entity linking and graph-based retrieval
- **Ontology Reasoning** - OWL 2 RL materialization for logical inference
- **Learned Gating** - Neural mechanisms to balance different retrieval sources

### 🚀 Quick Navigation

| Document | Description |
|----------|-------------|
| [🏗️ Architecture](ARCHITECTURE.md) | System design and high-level architecture |
| [📦 Modules](MODULES.md) | Detailed module breakdown with code references |
| [🔌 API Reference](API_REFERENCE.md) | Complete function and class documentation |
| [🌊 Data Flow](DATA_FLOW.md) | Data flow patterns and sequence diagrams |
| [⚙️ Setup](SETUP.md) | Development environment and deployment guide |
| [🔧 Configuration](CONFIGURATION.md) | Configuration options and environment variables |
| [📚 Dependencies](DEPENDENCIES.md) | External dependencies and rationale |

## 🛠️ Technology Stack

### Core ML & AI
- **PyTorch** `>=2.0.0` - Deep learning framework
- **Transformers** `>=4.36.0` - Hugging Face transformer models
- **Sentence Transformers** `>=2.2.0` - Semantic embeddings

### Knowledge Representation
- **OWLready2** `>=0.45` - Ontology processing and reasoning
- **NetworkX** `>=3.0` - Graph data structures and algorithms
- **RDFlib** `>=7.0.0` - RDF processing

### Vector Search & Retrieval
- **FAISS** `>=1.7.4` - Efficient similarity search
- **Scikit-learn** `>=1.3.0` - Machine learning utilities

### Web Application
- **Flask** - Web framework (in app/)
- **HTML/CSS/JS** - Frontend interface
- **Docker** - Containerization

## 📊 Project Statistics

### Codebase Metrics
```
Total Python Files: 25+
Core Source Files: 7 main modules
Lines of Code: ~1,165 (src/ only)
Test Coverage: Unit tests in tests/
Documentation: Comprehensive inline docs
```

### Repository Structure
```
rag-second-brain/
├── 📁 src/           # Core implementation (1,165 LOC)
├── 📁 app/           # Flask web application  
├── 📁 tests/         # Unit testing suite
├── 📁 experiments/   # Research experiments
├── 📁 notebooks/     # Jupyter analysis
├── 📁 scripts/       # Utility scripts
└── 📝 main.py       # Menu-driven demo
```

### Module Breakdown
| Module | LOC | Purpose |
|--------|-----|---------|
| `pipeline.py` | 324 | Main RAG pipeline orchestration |
| `gating.py` | 245 | Learned gating mechanisms |
| `ontology.py` | 166 | OWL 2 RL reasoning engine |
| `kg_retrieval.py` | 161 | Knowledge graph retrieval |
| `cooccurrence.py` | 138 | Statistical co-occurrence scoring |
| `dense_retrieval.py` | 123 | Dense embedding retrieval |

## 🔬 Research Context

This implementation accompanies the research paper:
> *"Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"*
> Author: Mingkhwan, Anirach (2026)

### Key Innovations
1. **Multi-Modal Retrieval Fusion** - Combines statistical, dense, and symbolic retrieval
2. **Ontology-Enhanced RAG** - Uses OWL 2 RL reasoning for logical inference
3. **Learned Gating** - Neural mechanisms to balance retrieval sources
4. **Proof-of-Concept Architecture** - Modular design for research validation

## 🎮 Demo & Testing

The project includes a menu-driven demo (`main.py`) with options:
1. Co-occurrence Module Demo
2. Dense Retrieval Demo  
3. Knowledge Graph Demo
4. Ontology Reasoning Demo
5. Gating Mechanism Demo
6. Full Pipeline Demo
7. Proof-of-Concept Tests

## 🌐 Web Application

A Flask-based web interface (`app/`) provides:
- Document upload and indexing
- Interactive search interface
- Knowledge graph visualization
- Retrieval result comparison

## 📄 License & Citation

**License:** MIT

**Citation:**
```bibtex
@article{mingkhwan2026secondbrain,
  title={Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM},
  author={Mingkhwan, Anirach},
  year={2026}
}
```

---

**Status:** 🚧 Work in Progress - Research proof-of-concept implementation

**Repository:** [github.com/Anirach/rag-second-brain](https://github.com/Anirach/rag-second-brain)

**Last Updated:** February 2026