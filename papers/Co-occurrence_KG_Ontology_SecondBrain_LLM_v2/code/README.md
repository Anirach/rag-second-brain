# Hybrid Memory System for LLM Augmentation

## "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

This repository contains the implementation code for the research paper presenting a hybrid memory architecture that serves as a "Second Brain" for Large Language Models.

## Overview

The system integrates three complementary knowledge representation approaches:

1. **Co-occurrence Pattern Analysis**: Statistical modeling of word associations using PMI
2. **Sequence Modeling**: Transformer-based contextual retrieval
3. **Knowledge Graphs with Ontology**: Structured knowledge representation with logical reasoning

## Repository Structure

```
code/
├── hybrid_memory.py      # Core implementation
├── evaluation.py         # Evaluation metrics and benchmarks
├── visualizations.py     # Figure generation for paper
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Installation

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Requirements

```
numpy>=1.21.0
scipy>=1.7.0
torch>=1.10.0
matplotlib>=3.5.0
seaborn>=0.11.0
```

## Quick Start

### 1. Run the Demo

```python
from hybrid_memory import create_demo_system, main

# Run interactive demo
main()

# Or create system programmatically
system = create_demo_system()

# Query the hybrid memory
result = system.query("What is machine learning?")
print(result.explanation)
```

### 2. Evaluate the System

```python
from evaluation import run_evaluation_demo, ResultsGenerator

# Run evaluation demo
run_evaluation_demo()

# Generate paper results
generator = ResultsGenerator(seed=42)
main_results = generator.generate_main_results()
```

### 3. Generate Figures

```python
from visualizations import generate_all_figures

# Generate all paper figures
generate_all_figures(output_dir="figures")
```

## Module Documentation

### hybrid_memory.py

Core classes:
- `CooccurrenceAnalyzer`: Builds co-occurrence matrix, computes PPMI, generates embeddings
- `SequenceEncoder`: Transformer-based sequence encoder
- `SequenceIndexer`: Dense retrieval with ANN search
- `KnowledgeGraph`: Graph storage with TransE embeddings
- `OntologyReasoner`: Class hierarchies, transitivity, consistency checking
- `HybridMemorySystem`: Unified query interface with gating mechanism

### evaluation.py

Evaluation classes:
- `BenchmarkDataset`: Base class for evaluation datasets
- `SyntheticQADataset`: Demo QA dataset
- `HotpotQALike`: Multi-hop reasoning benchmark
- `MetricsCalculator`: EM, F1, factual consistency, hallucination detection
- `Evaluator`: Main evaluation pipeline with ablation studies
- `ResultsGenerator`: Generate paper-style results

### visualizations.py

Figure generation:
- `plot_main_results_comparison()`: Benchmark comparison bar chart
- `plot_multihop_accuracy()`: Multi-hop reasoning line plot
- `plot_ablation_study()`: Component contribution analysis
- `plot_factual_consistency()`: Consistency and hallucination metrics
- `plot_gating_weights_heatmap()`: Adaptive gating visualization
- `plot_efficiency_comparison()`: Latency-accuracy trade-off
- `plot_architecture_diagram()`: System architecture overview

## Mathematical Foundation

### Co-occurrence Analysis

```
PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j)))
PPMI(w_i, w_j) = max(0, PMI(w_i, w_j))
```

### Knowledge Graph Embedding (TransE)

```
h + r ≈ t
Score: f(h, r, t) = -||h + r - t||
```

### Hybrid Integration

```
e_hybrid(c) = α * e_cooc(c) + β * e_seq(c) + γ * e_kg(c)
[α, β, γ] = softmax(W_gate * [e_cooc; e_seq; e_kg])
```

## Data Sources

The system is designed to work with:
- **WikiData**: Structured knowledge base
- **ConceptNet**: Commonsense knowledge graph
- **Wikipedia**: Text corpus for co-occurrence
- **Schema.org**: Ontology definitions

For demonstration, synthetic data is included.

## Extending the System

### Adding New Knowledge Sources

```python
from hybrid_memory import KnowledgeGraph, Entity, Triple

kg = KnowledgeGraph()

# Add entities
kg.add_entity(Entity(id="new_entity", name="New Entity", entity_type="Type"))

# Add triples
kg.add_triple(Triple(head="entity1", relation="relates_to", tail="entity2"))
```

### Custom Ontology Rules

```python
from hybrid_memory import OntologyReasoner

onto = OntologyReasoner()

# Add class hierarchy
onto.add_subclass("Subclass", "Superclass")

# Add transitive property
onto.add_property("part_of", transitive=True)
```

## Citation

If you use this code, please cite:

```bibtex
@article{hybrid_memory_2024,
  title={Co-occurrence, Sequence and Knowledge Graph with Ontology 
         as a Second Brain for AI-LLM},
  author={Research Team},
  journal={IEEE Conference Proceedings},
  year={2024}
}
```

## License

MIT License

## Acknowledgments

This work builds upon:
- TransE (Bordes et al., 2013)
- Transformers (Vaswani et al., 2017)
- RAG (Lewis et al., 2020)
- ConceptNet (Speer et al., 2017)
