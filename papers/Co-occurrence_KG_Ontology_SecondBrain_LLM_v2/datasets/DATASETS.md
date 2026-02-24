# Datasets Information

This document describes the datasets used in the research paper.

## Open Datasets Used

### 1. WikiData
- **Source**: https://www.wikidata.org/
- **Description**: Free collaborative knowledge base with 100M+ entities and 1.4B+ statements
- **Usage**: Knowledge graph construction, entity linking, fact verification
- **License**: CC0 1.0 Universal (Public Domain)
- **Download**: https://dumps.wikimedia.org/wikidatawiki/entities/

### 2. ConceptNet
- **Source**: https://conceptnet.io/
- **Description**: Commonsense knowledge graph with 21M+ edges
- **Usage**: Commonsense reasoning evaluation, relation extraction
- **License**: CC BY-SA 4.0
- **Download**: https://github.com/commonsense/conceptnet5/wiki/Downloads

### 3. Wikipedia (English)
- **Source**: https://dumps.wikimedia.org/
- **Description**: Text corpus for co-occurrence analysis
- **Usage**: Building co-occurrence matrix, sequence indexing
- **License**: CC BY-SA 3.0
- **Download**: https://dumps.wikimedia.org/enwiki/

### 4. Schema.org
- **Source**: https://schema.org/
- **Description**: Collaborative vocabulary for structured data
- **Usage**: Ontology definitions, class hierarchies
- **License**: CC BY-SA 3.0
- **Download**: https://schema.org/docs/developers.html

## Evaluation Benchmarks

### 1. Natural Questions (NQ)
- **Paper**: Kwiatkowski et al., 2019
- **Description**: Open-domain QA from Google search queries
- **URL**: https://ai.google.com/research/NaturalQuestions

### 2. TriviaQA
- **Paper**: Joshi et al., 2017
- **Description**: Reading comprehension with trivia questions
- **URL**: http://nlp.cs.washington.edu/triviaqa/

### 3. HotpotQA
- **Paper**: Yang et al., 2018
- **Description**: Multi-hop reasoning QA
- **URL**: https://hotpotqa.github.io/

### 4. FEVER
- **Paper**: Thorne et al., 2018
- **Description**: Fact extraction and verification
- **URL**: https://fever.ai/

### 5. TruthfulQA
- **Paper**: Lin et al., 2022
- **Description**: Measuring model truthfulness
- **URL**: https://github.com/sylinrl/TruthfulQA

## Data Processing

### Co-occurrence Matrix
```python
# Window size: 5
# Min word count: 5
# Vocabulary filtering: top 50K words
# PPMI smoothing: α = 0.75
```

### Knowledge Graph
```python
# Entity types: Person, Organization, Location, Concept
# Relations: ~500 distinct types
# Filtering: confidence > 0.8
```

### Sequence Index
```python
# Max sequence length: 512 tokens
# Encoding: sentence-transformers/all-MiniLM-L6-v2
# Index: FAISS IVF with PQ compression
```

## Synthetic Demo Data

For demonstration purposes, the code includes synthetic data:
- Small vocabulary (~100 words)
- Sample KG with AI/ML domain entities
- Basic ontology with class hierarchy

To use real datasets, replace the demo data loading with actual data loaders.

## Data Statistics

| Dataset | Entities | Relations | Triples | Size |
|---------|----------|-----------|---------|------|
| WikiData (sample) | 1M | 500 | 10M | 5 GB |
| ConceptNet | 8M concepts | 40 | 21M | 2 GB |
| Wikipedia (processed) | - | - | - | 20 GB |

## Reproducibility

To reproduce results:
1. Download datasets from URLs above
2. Run preprocessing scripts
3. Build indices
4. Execute evaluation scripts

See `code/README.md` for detailed instructions.
