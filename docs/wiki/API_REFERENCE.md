# 🔌 API Reference

Complete reference for all classes, methods, and endpoints in the RAG Second Brain system.

## 📋 API Overview

The system provides both **Python API** (for programmatic access) and **Web API** (via Flask) interfaces.

```mermaid
graph TD
    subgraph "Python API"
        PIPE[RAGPipeline]
        COOC[CooccurrenceScorer]
        DENSE[DenseRetriever] 
        KG[KnowledgeGraphRetriever]
        ONTO[OntologyReasoner]
        GATE[GatingMechanism]
    end
    
    subgraph "Web API"
        SEARCH[/search]
        UPLOAD[/upload]
        GRAPH[/graph]
        STATUS[/status]
    end
    
    subgraph "Demo API"
        MENU[Menu Interface]
        DEMOS[Component Demos]
    end
    
    PIPE --> SEARCH
    MENU --> PIPE
    
    classDef python fill:#e1f5fe
    classDef web fill:#f3e5f5
    classDef demo fill:#e8f5e8
    
    class PIPE,COOC,DENSE,KG,ONTO,GATE python
    class SEARCH,UPLOAD,GRAPH,STATUS web
    class MENU,DEMOS demo
```

---

## 🐍 Python API

### Core Pipeline API

#### `RAGPipeline` Class

**Location:** `src/pipeline.py:34-324`

```python
class RAGPipeline:
    """Main orchestration class for multi-modal retrieval."""
    
    def __init__(self, config: Optional[Dict] = None) -> None
    def retrieve_and_rank(self, query: str, candidates: List[str], 
                         top_k: int = 10) -> List[Tuple[str, float]]
    def evaluate_pipeline(self, test_queries: List[str], 
                         ground_truth: List[List[str]]) -> Dict[str, float]
    def save_pipeline(self, path: str) -> None
    def load_pipeline(self, path: str) -> None
```

**Constructor Parameters:**
```python
def __init__(self, config: Optional[Dict] = None) -> None:
    """
    Initialize RAG pipeline with optional configuration.
    
    Args:
        config: Optional configuration dictionary with keys:
            - cooccurrence: Dict for CooccurrenceScorer config
            - dense: Dict for DenseRetriever config  
            - kg: Dict for KnowledgeGraphRetriever config
            - ontology: Dict for OntologyReasoner config
            - gating: Dict for GatingMechanism config
    
    Returns:
        None
        
    Example:
        >>> config = {
        ...     'dense': {'model_name': 'all-MiniLM-L6-v2'},
        ...     'gating': {'learning_rate': 1e-4}
        ... }
        >>> pipeline = RAGPipeline(config)
    """
```

**Main Retrieval Method:**
```python
def retrieve_and_rank(self, query: str, candidates: List[str], 
                     top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Main retrieval interface combining all modalities.
    
    Args:
        query: Input query string
        candidates: List of candidate documents to rank
        top_k: Number of top results to return
    
    Returns:
        List of (document, score) tuples sorted by relevance
        
    Raises:
        ValueError: If pipeline components not properly initialized
        
    Example:
        >>> pipeline = RAGPipeline()
        >>> query = "What is machine learning?"
        >>> candidates = ["ML is a subset of AI...", "Python is a language..."]
        >>> results = pipeline.retrieve_and_rank(query, candidates, top_k=5)
        >>> print(f"Best match: {results[0][0]} (score: {results[0][1]:.3f})")
    """
```

**Evaluation Method:**
```python
def evaluate_pipeline(self, test_queries: List[str], 
                     ground_truth: List[List[str]]) -> Dict[str, float]:
    """
    Evaluate pipeline performance on test dataset.
    
    Args:
        test_queries: List of test query strings
        ground_truth: List of relevant documents for each query
    
    Returns:
        Dict with evaluation metrics:
            - 'map': Mean Average Precision
            - 'ndcg@10': Normalized Discounted Cumulative Gain
            - 'mrr': Mean Reciprocal Rank
            - 'recall@10': Recall at 10
            
    Example:
        >>> queries = ["What is AI?", "How does ML work?"]
        >>> ground_truth = [["doc1", "doc2"], ["doc3", "doc4"]]
        >>> metrics = pipeline.evaluate_pipeline(queries, ground_truth)
        >>> print(f"MAP: {metrics['map']:.3f}")
    """
```

---

### Retrieval Component APIs

#### `CooccurrenceScorer` Class

**Location:** `src/cooccurrence.py:15-138`

```python
class CooccurrenceScorer:
    """Statistical co-occurrence and PPMI-based scoring."""
    
    def __init__(self, window_size: int = 5, min_count: int = 2) -> None
    def build_cooccurrence_matrix(self, documents: List[str]) -> None  
    def score_candidates(self, query: str, candidates: List[str]) -> List[float]
    def get_similar_words(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]
```

**Configuration:**
```python
def __init__(self, window_size: int = 5, min_count: int = 2) -> None:
    """
    Initialize co-occurrence scorer.
    
    Args:
        window_size: Context window size for co-occurrence counting
        min_count: Minimum word frequency threshold
        
    Example:
        >>> scorer = CooccurrenceScorer(window_size=3, min_count=5)
    """
```

**Matrix Building:**
```python
def build_cooccurrence_matrix(self, documents: List[str]) -> None:
    """
    Build PPMI matrix from document corpus.
    
    Args:
        documents: List of document strings
        
    Raises:
        ValueError: If documents list is empty
        MemoryError: If corpus too large for available memory
        
    Side Effects:
        - Sets self.ppmi_matrix
        - Sets self.vocab_to_idx mapping
        - Sets self.built = True
        
    Example:
        >>> docs = ["machine learning is fun", "deep learning works well"]
        >>> scorer.build_cooccurrence_matrix(docs)
        >>> print(f"Vocabulary size: {len(scorer.vocab_to_idx)}")
    """
```

**Candidate Scoring:**
```python
def score_candidates(self, query: str, candidates: List[str]) -> List[float]:
    """
    Score candidates using PPMI-based similarity.
    
    Args:
        query: Query string
        candidates: List of candidate documents
        
    Returns:
        List of similarity scores (0.0 to 1.0)
        
    Raises:
        ValueError: If build_cooccurrence_matrix() not called first
        
    Example:
        >>> scores = scorer.score_candidates("machine learning", candidates)
        >>> best_idx = scores.index(max(scores))
        >>> print(f"Best candidate: {candidates[best_idx]}")
    """
```

#### `DenseRetriever` Class

**Location:** `src/dense_retrieval.py:12-123`

```python
class DenseRetriever:
    """Semantic embedding-based retrieval with FAISS indexing."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None
    def build_index(self, documents: List[str]) -> None
    def retrieve(self, query: str, candidates: List[str] = None, 
                top_k: int = 10) -> Union[List[Tuple[str, float]], List[float]]
    def get_embeddings(self, texts: List[str]) -> np.ndarray
```

**Initialization:**
```python
def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
    """
    Initialize dense retriever with sentence transformer model.
    
    Args:
        model_name: HuggingFace model identifier
        
    Supported Models:
        - sentence-transformers/all-MiniLM-L6-v2 (default, 384 dim)
        - sentence-transformers/all-mpnet-base-v2 (768 dim)
        - sentence-transformers/all-distilroberta-v1 (768 dim)
        
    Example:
        >>> retriever = DenseRetriever("all-mpnet-base-v2")
        >>> print(f"Model: {retriever.encoder.get_sentence_embedding_dimension()}d")
    """
```

**Index Construction:**
```python
def build_index(self, documents: List[str]) -> None:
    """
    Build FAISS index for efficient similarity search.
    
    Args:
        documents: List of documents to index
        
    Process:
        1. Encode documents to embeddings
        2. Normalize embeddings for cosine similarity
        3. Build FAISS IndexFlatIP index
        
    Memory Usage:
        - ~4 bytes per dimension per document
        - Example: 10k docs × 384 dims = ~15MB
        
    Example:
        >>> docs = load_corpus()
        >>> retriever.build_index(docs)
        >>> print(f"Indexed {len(docs)} documents")
    """
```

**Retrieval Interface:**
```python
def retrieve(self, query: str, candidates: List[str] = None, 
            top_k: int = 10) -> Union[List[Tuple[str, float]], List[float]]:
    """
    Perform semantic retrieval.
    
    Args:
        query: Query string
        candidates: Optional candidate list (if None, searches full index)
        top_k: Number of results (ignored if candidates provided)
        
    Returns:
        If candidates=None: List of (document, score) tuples
        If candidates provided: List of similarity scores
        
    Example:
        >>> # Index search mode
        >>> results = retriever.retrieve("machine learning", top_k=5)
        >>> print(f"Top result: {results[0][0]} (score: {results[0][1]:.3f})")
        
        >>> # Candidate scoring mode  
        >>> scores = retriever.retrieve("AI ethics", candidates=candidate_docs)
        >>> ranked = sorted(zip(candidate_docs, scores), key=lambda x: x[1], reverse=True)
    """
```

#### `KnowledgeGraphRetriever` Class

**Location:** `src/kg_retrieval.py:18-161`

```python
class KnowledgeGraphRetriever:
    """Knowledge graph construction and entity-based retrieval."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None
    def build_knowledge_graph(self, documents: List[str]) -> None
    def retrieve_with_reasoning(self, query: str, candidates: List[str]) -> List[float]
    def get_graph_stats(self) -> Dict[str, int]
    def visualize_subgraph(self, entities: List[str]) -> None
```

**Graph Construction:**
```python
def build_knowledge_graph(self, documents: List[str]) -> None:
    """
    Build knowledge graph from document corpus.
    
    Args:
        documents: List of document strings
        
    Process:
        1. Extract entities using NER
        2. Compute co-occurrence weights  
        3. Build NetworkX graph with entities as nodes
        4. Generate entity embeddings
        
    Graph Properties:
        - Nodes: Named entities (PERSON, ORG, GPE, etc.)
        - Edges: Co-occurrence relationships with weights
        - Attributes: Entity types, embeddings, document frequencies
        
    Example:
        >>> kg = KnowledgeGraphRetriever()
        >>> kg.build_knowledge_graph(document_corpus)
        >>> stats = kg.get_graph_stats()
        >>> print(f"Graph: {stats['nodes']} nodes, {stats['edges']} edges")
    """
```

**Reasoning-based Retrieval:**
```python
def retrieve_with_reasoning(self, query: str, candidates: List[str]) -> List[float]:
    """
    Graph-based retrieval with multi-hop reasoning.
    
    Args:
        query: Query string
        candidates: List of candidate documents
        
    Returns:
        List of similarity scores based on graph structure
        
    Algorithm:
        1. Extract entities from query and candidates
        2. Find shortest paths in knowledge graph
        3. Compute path-based similarity scores
        4. Combine with semantic similarity
        
    Scoring Formula:
        score = α × graph_score + β × semantic_score
        where α=0.7, β=0.3 (configurable)
        
    Example:
        >>> scores = kg.retrieve_with_reasoning("Apple Inc", candidates)
        >>> # High scores for candidates mentioning related entities
        >>> # (e.g., "iPhone", "Tim Cook", "Cupertino")
    """
```

#### `OntologyReasoner` Class

**Location:** `src/ontology.py:23-166`

```python
class OntologyReasoner:
    """OWL 2 RL ontology reasoning and semantic similarity."""
    
    def __init__(self, ontology_path: Optional[str] = None) -> None
    def materialize_inferences(self) -> None
    def semantic_scoring(self, query: str, candidates: List[str]) -> List[float]
    def add_axioms(self, axioms: List[str]) -> None
    def query_ontology(self, sparql_query: str) -> List[Dict]
```

**Reasoning Engine:**
```python
def materialize_inferences(self) -> None:
    """
    Perform OWL 2 RL materialization using Pellet reasoner.
    
    Process:
        1. Load ontology into OWLready2 world
        2. Run Pellet reasoner for inference
        3. Materialize implicit knowledge
        
    Side Effects:
        - Populates inferred class memberships
        - Adds derived property assertions
        - Sets self._materialized = True
        
    Error Handling:
        - Falls back to RDFS reasoning if Pellet fails
        - Logs reasoning statistics
        
    Example:
        >>> reasoner = OntologyReasoner("domain_ontology.owl")
        >>> reasoner.materialize_inferences()
        >>> # Now implicit relationships are explicit
    """
```

**Semantic Scoring:**
```python
def semantic_scoring(self, query: str, candidates: List[str]) -> List[float]:
    """
    Score candidates using ontological semantic similarity.
    
    Args:
        query: Query string
        candidates: List of candidate documents
        
    Returns:
        List of semantic similarity scores (0.0 to 1.0)
        
    Methodology:
        1. Extract entities from query and candidates
        2. Map entities to ontological concepts
        3. Compute concept similarity via class hierarchy
        4. Aggregate similarities across entity pairs
        
    Similarity Measures:
        - Wu-Palmer similarity (path-based)
        - Information content similarity
        - Lowest Common Subsumer (LCS) based
        
    Example:
        >>> scores = reasoner.semantic_scoring("medical diagnosis", candidates)
        >>> # High scores for docs about "disease", "symptom", "treatment"
    """
```

#### `GatingMechanism` Class

**Location:** `src/gating.py:15-245`

```python
class GatingMechanism:
    """Neural gating network for dynamic retrieval source weighting."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None
    def compute_weights(self, query: str) -> torch.Tensor
    def train_step(self, batch_queries: List[str], 
                   target_weights: torch.Tensor) -> float
    def save_model(self, path: str) -> None
    def load_model(self, path: str) -> None
```

**Weight Computation:**
```python
def compute_weights(self, query: str) -> torch.Tensor:
    """
    Compute dynamic weights for retrieval sources.
    
    Args:
        query: Input query string
        
    Returns:
        Tensor of shape (4,) with weights for [cooccurrence, dense, kg, ontology]
        Weights sum to 1.0 (softmax normalized)
        
    Process:
        1. Encode query using sentence transformer
        2. Pass through gating network
        3. Apply softmax for normalization
        
    Example:
        >>> gating = GatingMechanism()
        >>> weights = gating.compute_weights("What is deep learning?")
        >>> print(f"Weights: {weights.tolist()}")
        >>> # Example output: [0.1, 0.6, 0.2, 0.1] - favors dense retrieval
    """
```

**Training Interface:**
```python
def train_step(self, batch_queries: List[str], 
               target_weights: torch.Tensor) -> float:
    """
    Single training step with backpropagation.
    
    Args:
        batch_queries: List of training queries
        target_weights: Target weight tensor of shape (batch_size, 4)
        
    Returns:
        Training loss (MSE between predicted and target weights)
        
    Process:
        1. Encode queries to embeddings
        2. Forward pass through gating network
        3. Compute MSE loss against targets
        4. Backpropagation and optimizer step
        
    Example:
        >>> queries = ["query1", "query2"]
        >>> targets = torch.tensor([[0.2, 0.5, 0.2, 0.1], [0.1, 0.3, 0.4, 0.2]])
        >>> loss = gating.train_step(queries, targets)
        >>> print(f"Training loss: {loss:.4f}")
    """
```

---

## 🌐 Web API Endpoints

**Base URL:** `http://localhost:5000` (development)

### Search Endpoint

```http
POST /search
Content-Type: application/json

{
    "query": "machine learning applications",
    "top_k": 10,
    "sources": ["dense", "cooccurrence", "kg", "ontology"]
}
```

**Response:**
```json
{
    "status": "success",
    "query": "machine learning applications",
    "results": [
        {
            "document": "Machine learning has numerous applications...",
            "score": 0.87,
            "source_scores": {
                "dense": 0.92,
                "cooccurrence": 0.78,
                "kg": 0.85,
                "ontology": 0.93
            },
            "gating_weights": [0.15, 0.35, 0.25, 0.25]
        }
    ],
    "total_results": 156,
    "processing_time_ms": 234
}
```

### Document Upload Endpoint

```http
POST /upload
Content-Type: multipart/form-data

file: document.pdf
title: "Optional document title"
metadata: {"author": "John Doe", "category": "research"}
```

**Response:**
```json
{
    "status": "success", 
    "document_id": "doc_12345",
    "title": "Extracted or provided title",
    "text_length": 5432,
    "entities_extracted": 23,
    "indexed": true,
    "processing_time_ms": 1200
}
```

### Knowledge Graph Visualization

```http
GET /graph?entities=Apple,iPhone,Technology&depth=2
```

**Response:**
```json
{
    "nodes": [
        {"id": "Apple", "type": "ORG", "size": 10},
        {"id": "iPhone", "type": "PRODUCT", "size": 8},
        {"id": "Technology", "type": "CONCEPT", "size": 15}
    ],
    "edges": [
        {"source": "Apple", "target": "iPhone", "weight": 0.85, "type": "develops"},
        {"source": "Apple", "target": "Technology", "weight": 0.72, "type": "operates_in"}
    ],
    "layout": "force_directed"
}
```

### System Status Endpoint

```http
GET /status
```

**Response:**
```json
{
    "status": "healthy",
    "components": {
        "pipeline": "ready",
        "cooccurrence": {"status": "ready", "vocab_size": 15420},
        "dense_retrieval": {"status": "ready", "index_size": 10000, "model": "all-MiniLM-L6-v2"},
        "knowledge_graph": {"status": "ready", "nodes": 3421, "edges": 8934},
        "ontology": {"status": "ready", "classes": 145, "materialized": true}
    },
    "memory_usage_mb": 1024,
    "uptime_seconds": 3600
}
```

---

## 🎮 Demo Interface API

**Location:** `main.py` - Menu-driven demonstration interface

### Menu Options

```python
def main():
    """Main demo interface with interactive menu."""
    options = {
        1: demo_cooccurrence,      # Co-occurrence module demo
        2: demo_dense_retrieval,   # Dense retrieval demo  
        3: demo_knowledge_graph,   # Knowledge graph demo
        4: demo_ontology,          # Ontology reasoning demo
        5: demo_gating,            # Gating mechanism demo
        6: demo_full_pipeline,     # Complete pipeline demo
        7: run_tests,              # Proof-of-concept tests
        8: show_configuration,     # Configuration display
        9: show_about,             # System information
        0: exit                    # Exit program
    }
```

### Demo Functions

```python
def demo_cooccurrence() -> None:
    """
    Interactive co-occurrence scoring demonstration.
    
    Process:
        1. Load sample corpus or accept user input
        2. Build co-occurrence matrix  
        3. Accept query from user
        4. Display top similar documents with scores
        5. Show vocabulary statistics
        
    Example Output:
        Query: "machine learning"
        
        Results:
        1. "Deep learning is a subset of ML..." (Score: 0.87)
        2. "Neural networks enable machines..." (Score: 0.73)
        3. "Algorithms can learn patterns..." (Score: 0.68)
        
        Vocabulary: 1,245 terms | Matrix density: 0.03%
    """
```

```python
def demo_full_pipeline() -> None:
    """
    Complete pipeline demonstration with all components.
    
    Features:
        - Multi-modal retrieval comparison
        - Gating weight visualization
        - Performance timing
        - Score breakdown by source
        
    Interactive Elements:
        - Custom query input
        - Source selection (enable/disable components)
        - Top-k configuration
        - Detailed vs. summary output modes
        
    Example Session:
        >>> Enter query: "artificial intelligence ethics"
        >>> Enable sources [1=cooccur, 2=dense, 3=kg, 4=ontology]: 1,2,3,4
        >>> Top-k results: 5
        
        Processing with gating weights: [0.12, 0.41, 0.28, 0.19]
        
        Results (245ms):
        1. "AI ethics frameworks consider..." (Score: 0.91)
        2. "Ethical considerations in ML..." (Score: 0.87)
        ...
    """
```

---

## 🔧 Configuration API

### Environment Variables

```bash
# Model Configuration
DENSE_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
KG_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

# Retrieval Parameters  
COOCCUR_WINDOW_SIZE=5
COOCCUR_MIN_COUNT=2
DENSE_INDEX_TYPE="IndexFlatIP"
KG_ENTITY_THRESHOLD=0.8

# Ontology Configuration
ONTOLOGY_PATH="data/ontologies/domain.owl"
REASONING_DEPTH=3
USE_PELLET_REASONER=true

# Gating Network
GATING_HIDDEN_DIM=128
GATING_LEARNING_RATE=0.0001
GATING_DROPOUT=0.1

# Web Application
FLASK_HOST="0.0.0.0"
FLASK_PORT=5000
FLASK_DEBUG=false
MAX_UPLOAD_SIZE=10MB

# Performance
BATCH_SIZE=32
NUM_WORKERS=4
CACHE_SIZE=1000
```

### Configuration Loading

```python
# src/pipeline.py:15-32
def _default_config() -> Dict:
    """Load configuration from environment variables with defaults."""
    return {
        'cooccurrence': {
            'window_size': int(os.getenv('COOCCUR_WINDOW_SIZE', '5')),
            'min_count': int(os.getenv('COOCCUR_MIN_COUNT', '2'))
        },
        'dense': {
            'model_name': os.getenv('DENSE_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
            'index_type': os.getenv('DENSE_INDEX_TYPE', 'IndexFlatIP')
        },
        'kg': {
            'model_name': os.getenv('KG_MODEL_NAME', 'sentence-transformers/all-MiniLM-L6-v2'),
            'entity_threshold': float(os.getenv('KG_ENTITY_THRESHOLD', '0.8'))
        },
        'ontology': {
            'ontology_path': os.getenv('ONTOLOGY_PATH'),
            'reasoning_depth': int(os.getenv('REASONING_DEPTH', '3'))
        },
        'gating': {
            'hidden_dim': int(os.getenv('GATING_HIDDEN_DIM', '128')),
            'learning_rate': float(os.getenv('GATING_LEARNING_RATE', '1e-4'))
        }
    }
```

---

## 📊 Error Handling

### Exception Types

```python
class RAGPipelineError(Exception):
    """Base exception for RAG pipeline errors."""
    pass

class ComponentNotBuiltError(RAGPipelineError):
    """Raised when trying to use uninitialized components."""
    pass

class ModelLoadError(RAGPipelineError):
    """Raised when model loading fails."""
    pass

class InsufficientDataError(RAGPipelineError):
    """Raised when not enough training data provided."""
    pass
```

### HTTP Error Responses

```json
{
    "status": "error",
    "error_code": "COMPONENT_NOT_READY",
    "message": "Dense retrieval index not built. Call build_index() first.",
    "details": {
        "component": "dense_retrieval",
        "required_action": "build_index",
        "suggestion": "Upload documents or call /build_index endpoint"
    }
}
```

---

**Next:** [🌊 Data Flow](DATA_FLOW.md) | **Previous:** [📦 Modules](MODULES.md)