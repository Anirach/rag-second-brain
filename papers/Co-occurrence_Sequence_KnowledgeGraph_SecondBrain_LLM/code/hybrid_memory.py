"""
Hybrid Memory System for LLM Augmentation
==========================================

This module implements the "Second Brain" architecture combining:
1. Co-occurrence Pattern Analysis (PMI-based)
2. Sequence Modeling (Transformer-based retrieval)
3. Knowledge Graph with Ontology (RDF/OWL-based)

Author: Research Team
License: MIT
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import svds
import json
import logging
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Triple:
    """Represents a knowledge graph triple."""
    head: str
    relation: str
    tail: str
    confidence: float = 1.0
    
    def __hash__(self):
        return hash((self.head, self.relation, self.tail))


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class QueryResult:
    """Container for hybrid query results."""
    cooc_results: List[Tuple[str, float]]
    seq_results: List[Tuple[str, float]]
    kg_results: List[Triple]
    combined_score: float
    explanation: str


# =============================================================================
# Co-occurrence Analysis Module
# =============================================================================

class CooccurrenceAnalyzer:
    """
    Builds and queries co-occurrence statistics with PMI computation.
    
    Mathematical foundation:
    - Co-occurrence matrix: M[i,j] = count of (w_i, w_j) within window k
    - PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j)))
    - PPMI(w_i, w_j) = max(0, PMI(w_i, w_j))
    """
    
    def __init__(self, window_size: int = 5, min_count: int = 5, 
                 embedding_dim: int = 300):
        self.window_size = window_size
        self.min_count = min_count
        self.embedding_dim = embedding_dim
        
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.cooc_matrix: Optional[csr_matrix] = None
        self.ppmi_matrix: Optional[csr_matrix] = None
        self.embeddings: Optional[np.ndarray] = None
        self.word_counts: Dict[str, int] = defaultdict(int)
        
    def build_vocabulary(self, corpus: List[List[str]]) -> None:
        """Build vocabulary from tokenized corpus."""
        logger.info("Building vocabulary...")
        
        # Count word frequencies
        for doc in corpus:
            for word in doc:
                self.word_counts[word] += 1
        
        # Filter by min_count and create mappings
        idx = 0
        for word, count in self.word_counts.items():
            if count >= self.min_count:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
                
        logger.info(f"Vocabulary size: {len(self.word2idx)}")
        
    def build_cooccurrence_matrix(self, corpus: List[List[str]]) -> None:
        """
        Build co-occurrence matrix from corpus.
        
        Algorithm complexity: O(D * k) where D = total tokens, k = window size
        Space complexity: O(|V|^2) dense, O(nnz) sparse
        """
        logger.info("Building co-occurrence matrix...")
        vocab_size = len(self.word2idx)
        
        # Use sparse matrix for efficiency
        cooc = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        for doc in corpus:
            doc_indices = [self.word2idx.get(w) for w in doc]
            
            for center_pos, center_idx in enumerate(doc_indices):
                if center_idx is None:
                    continue
                    
                # Define context window
                start = max(0, center_pos - self.window_size)
                end = min(len(doc_indices), center_pos + self.window_size + 1)
                
                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue
                    context_idx = doc_indices[context_pos]
                    if context_idx is not None:
                        # Weight by distance (optional)
                        distance = abs(context_pos - center_pos)
                        weight = 1.0 / distance  # Harmonic weighting
                        cooc[center_idx, context_idx] += weight
        
        self.cooc_matrix = cooc.tocsr()
        logger.info(f"Co-occurrence matrix built: {self.cooc_matrix.nnz} non-zero entries")
        
    def compute_ppmi(self, alpha: float = 0.75) -> None:
        """
        Compute Positive Pointwise Mutual Information matrix.
        
        PMI(w_i, w_j) = log(P(w_i, w_j) / (P(w_i) * P(w_j)))
        PPMI = max(0, PMI)
        
        Uses context distribution smoothing with alpha parameter.
        """
        logger.info("Computing PPMI matrix...")
        
        if self.cooc_matrix is None:
            raise ValueError("Co-occurrence matrix not built yet")
            
        # Total count
        total = self.cooc_matrix.sum()
        
        # Row and column sums
        row_sums = np.array(self.cooc_matrix.sum(axis=1)).flatten()
        col_sums = np.array(self.cooc_matrix.sum(axis=0)).flatten()
        
        # Apply context distribution smoothing
        col_sums_smoothed = np.power(col_sums, alpha)
        col_sums_smoothed_sum = col_sums_smoothed.sum()
        
        # Convert to LIL for efficient modification
        ppmi = lil_matrix(self.cooc_matrix.shape, dtype=np.float32)
        
        # Compute PPMI for non-zero entries
        cx = self.cooc_matrix.tocoo()
        for i, j, v in zip(cx.row, cx.col, cx.data):
            if v > 0 and row_sums[i] > 0 and col_sums_smoothed[j] > 0:
                # PMI calculation with smoothing
                p_ij = v / total
                p_i = row_sums[i] / total
                p_j_smoothed = col_sums_smoothed[j] / col_sums_smoothed_sum
                
                pmi = np.log2(p_ij / (p_i * p_j_smoothed) + 1e-10)
                ppmi[i, j] = max(0.0, pmi)
        
        self.ppmi_matrix = ppmi.tocsr()
        logger.info(f"PPMI matrix computed: {self.ppmi_matrix.nnz} positive entries")
        
    def compute_embeddings(self) -> None:
        """
        Compute word embeddings via truncated SVD of PPMI matrix.
        
        M_PPMI ≈ U_d * Σ_d * V_d^T
        Embeddings: e(w_i) = U_d[i,:] * Σ_d^0.5
        """
        logger.info(f"Computing {self.embedding_dim}-dimensional embeddings...")
        
        if self.ppmi_matrix is None:
            raise ValueError("PPMI matrix not computed yet")
        
        # Truncated SVD
        k = min(self.embedding_dim, min(self.ppmi_matrix.shape) - 1)
        U, S, Vt = svds(self.ppmi_matrix.astype(np.float64), k=k)
        
        # Sort by singular values (descending)
        idx = np.argsort(-S)
        U = U[:, idx]
        S = S[idx]
        
        # Word embeddings: U * sqrt(S)
        self.embeddings = U * np.sqrt(S)
        
        logger.info(f"Embeddings computed: shape {self.embeddings.shape}")
        
    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """Get embedding for a word."""
        if word not in self.word2idx:
            return None
        idx = self.word2idx[word]
        return self.embeddings[idx]
    
    def find_similar(self, word: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words by cosine similarity."""
        emb = self.get_embedding(word)
        if emb is None:
            return []
            
        # Compute cosine similarities
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normalized = self.embeddings / norms
        
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
        similarities = normalized @ emb_norm
        
        # Get top-k (excluding the query word itself)
        top_indices = np.argsort(-similarities)[:top_k + 1]
        
        results = []
        for idx in top_indices:
            if self.idx2word[idx] != word:
                results.append((self.idx2word[idx], float(similarities[idx])))
                if len(results) >= top_k:
                    break
                    
        return results
    
    def get_ppmi_score(self, word1: str, word2: str) -> float:
        """Get PPMI score between two words."""
        if word1 not in self.word2idx or word2 not in self.word2idx:
            return 0.0
        i, j = self.word2idx[word1], self.word2idx[word2]
        return float(self.ppmi_matrix[i, j])


# =============================================================================
# Sequence Modeling Module
# =============================================================================

class SequenceEncoder(nn.Module):
    """
    Transformer-based sequence encoder for passage retrieval.
    
    Uses self-attention to create context-aware representations.
    """
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, 
                 num_heads: int = 8, num_layers: int = 4,
                 max_seq_length: int = 512):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Token and position embeddings
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection head for retrieval
        self.projection = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode sequences to dense vectors.
        
        Args:
            input_ids: Token IDs [batch_size, seq_length]
            attention_mask: Mask for padding [batch_size, seq_length]
            
        Returns:
            Sequence embeddings [batch_size, embed_dim]
        """
        batch_size, seq_length = input_ids.shape
        
        # Create position indices
        positions = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Combine token and position embeddings
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Apply transformer
        if attention_mask is not None:
            # Convert to proper mask format (True = masked)
            src_key_padding_mask = ~attention_mask.bool()
        else:
            src_key_padding_mask = None
            
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # Mean pooling over sequence
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        else:
            x = x.mean(dim=1)
            
        # Project to retrieval space
        x = self.projection(x)
        x = F.normalize(x, p=2, dim=-1)
        
        return x


class SequenceIndexer:
    """
    Manages sequence encoding and retrieval with FAISS-style indexing.
    
    For demonstration, uses numpy-based approximate nearest neighbor.
    In production, use FAISS or ScaNN for efficiency.
    """
    
    def __init__(self, encoder: SequenceEncoder, device: str = 'cpu'):
        self.encoder = encoder.to(device)
        self.device = device
        self.index: Optional[np.ndarray] = None
        self.documents: List[str] = []
        self.document_ids: List[str] = []
        
    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None,
                      tokenizer=None, batch_size: int = 32) -> None:
        """Index documents for retrieval."""
        logger.info(f"Indexing {len(documents)} documents...")
        
        self.documents = documents
        self.document_ids = doc_ids or [str(i) for i in range(len(documents))]
        
        embeddings = []
        self.encoder.eval()
        
        with torch.no_grad():
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i + batch_size]
                
                # Simple tokenization (replace with proper tokenizer in production)
                if tokenizer is None:
                    # Dummy tokenization for demonstration
                    max_len = self.encoder.max_seq_length
                    input_ids = torch.zeros((len(batch_docs), max_len), dtype=torch.long)
                    attention_mask = torch.zeros((len(batch_docs), max_len), dtype=torch.long)
                    
                    for j, doc in enumerate(batch_docs):
                        tokens = doc.lower().split()[:max_len]
                        for k, token in enumerate(tokens):
                            # Simple hash-based token ID
                            input_ids[j, k] = hash(token) % (self.encoder.token_embedding.num_embeddings - 1) + 1
                            attention_mask[j, k] = 1
                else:
                    encoded = tokenizer(batch_docs, padding=True, truncation=True,
                                        max_length=self.encoder.max_seq_length,
                                        return_tensors='pt')
                    input_ids = encoded['input_ids']
                    attention_mask = encoded['attention_mask']
                
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                
                batch_emb = self.encoder(input_ids, attention_mask)
                embeddings.append(batch_emb.cpu().numpy())
        
        self.index = np.vstack(embeddings)
        logger.info(f"Index built: {self.index.shape}")
        
    def search(self, query: str, top_k: int = 10, tokenizer=None) -> List[Tuple[str, str, float]]:
        """Search for relevant documents."""
        if self.index is None:
            return []
            
        self.encoder.eval()
        
        with torch.no_grad():
            # Encode query
            if tokenizer is None:
                max_len = self.encoder.max_seq_length
                input_ids = torch.zeros((1, max_len), dtype=torch.long)
                attention_mask = torch.zeros((1, max_len), dtype=torch.long)
                
                tokens = query.lower().split()[:max_len]
                for k, token in enumerate(tokens):
                    input_ids[0, k] = hash(token) % (self.encoder.token_embedding.num_embeddings - 1) + 1
                    attention_mask[0, k] = 1
            else:
                encoded = tokenizer([query], padding=True, truncation=True,
                                    max_length=self.encoder.max_seq_length,
                                    return_tensors='pt')
                input_ids = encoded['input_ids']
                attention_mask = encoded['attention_mask']
            
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            
            query_emb = self.encoder(input_ids, attention_mask).cpu().numpy()
        
        # Compute similarities
        similarities = self.index @ query_emb.T
        similarities = similarities.flatten()
        
        # Get top-k
        top_indices = np.argsort(-similarities)[:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.document_ids[idx],
                self.documents[idx],
                float(similarities[idx])
            ))
            
        return results


# =============================================================================
# Knowledge Graph Module
# =============================================================================

class KnowledgeGraph:
    """
    Knowledge graph storage and querying with embedding support.
    
    G = (V, E, R) where:
    - V: set of entities
    - E: set of triples (h, r, t)
    - R: set of relations
    """
    
    def __init__(self, embedding_dim: int = 100):
        self.embedding_dim = embedding_dim
        
        self.entities: Dict[str, Entity] = {}
        self.triples: Set[Triple] = set()
        self.relations: Set[str] = set()
        
        # Index structures for efficient querying
        self.head_index: Dict[str, List[Triple]] = defaultdict(list)
        self.tail_index: Dict[str, List[Triple]] = defaultdict(list)
        self.relation_index: Dict[str, List[Triple]] = defaultdict(list)
        
        # Embeddings
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        
    def add_entity(self, entity: Entity) -> None:
        """Add an entity to the graph."""
        self.entities[entity.id] = entity
        
    def add_triple(self, triple: Triple) -> None:
        """Add a triple to the graph."""
        self.triples.add(triple)
        self.relations.add(triple.relation)
        
        # Update indices
        self.head_index[triple.head].append(triple)
        self.tail_index[triple.tail].append(triple)
        self.relation_index[triple.relation].append(triple)
        
    def get_neighbors(self, entity_id: str, relation: Optional[str] = None,
                      direction: str = 'out') -> List[Triple]:
        """Get neighboring triples for an entity."""
        if direction == 'out':
            triples = self.head_index.get(entity_id, [])
        else:
            triples = self.tail_index.get(entity_id, [])
            
        if relation is not None:
            triples = [t for t in triples if t.relation == relation]
            
        return triples
    
    def multi_hop_query(self, start_entity: str, path: List[str], 
                        max_results: int = 100) -> List[List[str]]:
        """
        Execute multi-hop query following a path of relations.
        
        Args:
            start_entity: Starting entity ID
            path: List of relations to follow
            max_results: Maximum number of result paths
            
        Returns:
            List of entity paths
        """
        current_entities = [[start_entity]]
        
        for relation in path:
            next_entities = []
            for entity_path in current_entities:
                current = entity_path[-1]
                for triple in self.get_neighbors(current, relation, 'out'):
                    new_path = entity_path + [triple.tail]
                    next_entities.append(new_path)
                    if len(next_entities) >= max_results:
                        break
                if len(next_entities) >= max_results:
                    break
            current_entities = next_entities
            
        return current_entities
    
    def compute_embeddings_transe(self, epochs: int = 100, lr: float = 0.01,
                                   margin: float = 1.0) -> None:
        """
        Compute TransE embeddings for entities and relations.
        
        TransE: h + r ≈ t
        Loss: max(0, margin + ||h + r - t|| - ||h' + r - t'||)
        """
        logger.info("Computing TransE embeddings...")
        
        # Initialize embeddings
        for entity_id in self.entities:
            self.entity_embeddings[entity_id] = np.random.randn(self.embedding_dim) * 0.1
            
        for relation in self.relations:
            self.relation_embeddings[relation] = np.random.randn(self.embedding_dim) * 0.1
            
        # Normalize
        for entity_id in self.entity_embeddings:
            self.entity_embeddings[entity_id] /= np.linalg.norm(
                self.entity_embeddings[entity_id]) + 1e-10
                
        triple_list = list(self.triples)
        entity_list = list(self.entities.keys())
        
        for epoch in range(epochs):
            total_loss = 0.0
            np.random.shuffle(triple_list)
            
            for triple in triple_list:
                h = self.entity_embeddings.get(triple.head)
                r = self.relation_embeddings.get(triple.relation)
                t = self.entity_embeddings.get(triple.tail)
                
                if h is None or r is None or t is None:
                    continue
                
                # Positive score
                pos_score = np.linalg.norm(h + r - t)
                
                # Generate negative sample
                if np.random.random() < 0.5:
                    # Corrupt head
                    neg_entity = np.random.choice(entity_list)
                    h_neg = self.entity_embeddings[neg_entity]
                    neg_score = np.linalg.norm(h_neg + r - t)
                else:
                    # Corrupt tail
                    neg_entity = np.random.choice(entity_list)
                    t_neg = self.entity_embeddings[neg_entity]
                    neg_score = np.linalg.norm(h + r - t_neg)
                
                # Margin-based loss
                loss = max(0, margin + pos_score - neg_score)
                total_loss += loss
                
                if loss > 0:
                    # Gradient update
                    grad = (h + r - t) / (pos_score + 1e-10)
                    
                    self.entity_embeddings[triple.head] -= lr * grad
                    self.relation_embeddings[triple.relation] -= lr * grad
                    self.entity_embeddings[triple.tail] += lr * grad
                    
            if epoch % 20 == 0:
                logger.info(f"Epoch {epoch}: Loss = {total_loss / len(triple_list):.4f}")
                
        logger.info("TransE embeddings computed")
        
    def find_similar_entities(self, entity_id: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Find similar entities by embedding distance."""
        if entity_id not in self.entity_embeddings:
            return []
            
        query_emb = self.entity_embeddings[entity_id]
        
        similarities = []
        for other_id, other_emb in self.entity_embeddings.items():
            if other_id != entity_id:
                sim = -np.linalg.norm(query_emb - other_emb)  # Negative distance = similarity
                similarities.append((other_id, sim))
                
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# =============================================================================
# Ontology Reasoning Module
# =============================================================================

class OntologyReasoner:
    """
    Ontology-based reasoning with class hierarchies and inference rules.
    
    Supports:
    - Subsumption (subclass relationships)
    - Transitivity
    - Domain/range constraints
    """
    
    def __init__(self):
        # Class hierarchy
        self.classes: Set[str] = set()
        self.subclass_of: Dict[str, Set[str]] = defaultdict(set)  # child -> parents
        
        # Property definitions
        self.properties: Set[str] = set()
        self.transitive_properties: Set[str] = set()
        self.property_domain: Dict[str, str] = {}
        self.property_range: Dict[str, str] = {}
        
        # Instance types
        self.instance_types: Dict[str, Set[str]] = defaultdict(set)
        
    def add_class(self, class_name: str) -> None:
        """Add a class to the ontology."""
        self.classes.add(class_name)
        
    def add_subclass(self, subclass: str, superclass: str) -> None:
        """Add subclass relationship."""
        self.classes.add(subclass)
        self.classes.add(superclass)
        self.subclass_of[subclass].add(superclass)
        
    def add_property(self, prop: str, domain: Optional[str] = None,
                     range_: Optional[str] = None, transitive: bool = False) -> None:
        """Add property definition."""
        self.properties.add(prop)
        if domain:
            self.property_domain[prop] = domain
        if range_:
            self.property_range[prop] = range_
        if transitive:
            self.transitive_properties.add(prop)
            
    def add_instance_type(self, instance: str, class_name: str) -> None:
        """Assert that an instance belongs to a class."""
        self.instance_types[instance].add(class_name)
        
    def get_all_superclasses(self, class_name: str) -> Set[str]:
        """Get all superclasses (transitive closure)."""
        result = set()
        to_process = [class_name]
        
        while to_process:
            current = to_process.pop()
            parents = self.subclass_of.get(current, set())
            for parent in parents:
                if parent not in result:
                    result.add(parent)
                    to_process.append(parent)
                    
        return result
    
    def infer_types(self, instance: str) -> Set[str]:
        """Infer all types for an instance including through hierarchy."""
        direct_types = self.instance_types.get(instance, set())
        inferred = set(direct_types)
        
        for t in direct_types:
            inferred.update(self.get_all_superclasses(t))
            
        return inferred
    
    def check_consistency(self, kg: KnowledgeGraph) -> List[str]:
        """
        Check knowledge graph consistency against ontology.
        
        Returns list of violations.
        """
        violations = []
        
        for triple in kg.triples:
            # Check domain constraint
            if triple.relation in self.property_domain:
                expected_domain = self.property_domain[triple.relation]
                head_types = self.infer_types(triple.head)
                
                if expected_domain not in head_types and expected_domain not in self.get_all_superclasses(expected_domain):
                    valid = False
                    for t in head_types:
                        if expected_domain in self.get_all_superclasses(t):
                            valid = True
                            break
                        if t == expected_domain:
                            valid = True
                            break
                    if not valid and head_types:
                        violations.append(
                            f"Domain violation: {triple.head} with types {head_types} "
                            f"used as head of {triple.relation} (expected {expected_domain})"
                        )
                        
            # Check range constraint
            if triple.relation in self.property_range:
                expected_range = self.property_range[triple.relation]
                tail_types = self.infer_types(triple.tail)
                
                if expected_range not in tail_types:
                    valid = False
                    for t in tail_types:
                        if expected_range in self.get_all_superclasses(t):
                            valid = True
                            break
                        if t == expected_range:
                            valid = True
                            break
                    if not valid and tail_types:
                        violations.append(
                            f"Range violation: {triple.tail} with types {tail_types} "
                            f"used as tail of {triple.relation} (expected {expected_range})"
                        )
                        
        return violations
    
    def apply_transitive_closure(self, kg: KnowledgeGraph) -> List[Triple]:
        """
        Apply transitivity rules to infer new triples.
        
        If r is transitive and r(a,b) ∧ r(b,c), then r(a,c).
        """
        inferred = []
        
        for prop in self.transitive_properties:
            triples_with_prop = kg.relation_index.get(prop, [])
            
            # Build adjacency for this relation
            adjacency: Dict[str, Set[str]] = defaultdict(set)
            for triple in triples_with_prop:
                adjacency[triple.head].add(triple.tail)
            
            # Compute transitive closure
            changed = True
            while changed:
                changed = False
                for head in list(adjacency.keys()):
                    tails = list(adjacency[head])
                    for tail in tails:
                        for next_tail in adjacency.get(tail, set()):
                            if next_tail not in adjacency[head]:
                                adjacency[head].add(next_tail)
                                new_triple = Triple(head, prop, next_tail, confidence=0.9)
                                if new_triple not in kg.triples:
                                    inferred.append(new_triple)
                                changed = True
                                
        return inferred


# =============================================================================
# Hybrid Memory System
# =============================================================================

class HybridMemorySystem:
    """
    Unified hybrid memory system combining all three components.
    
    Query processing:
    1. Extract key concepts from query
    2. Retrieve from each component in parallel
    3. Fuse results using attention-based gating
    4. Return ranked, aggregated results
    """
    
    def __init__(self, cooc_analyzer: CooccurrenceAnalyzer,
                 seq_indexer: SequenceIndexer,
                 knowledge_graph: KnowledgeGraph,
                 ontology: OntologyReasoner):
        self.cooc = cooc_analyzer
        self.seq = seq_indexer
        self.kg = knowledge_graph
        self.onto = ontology
        
        # Gating weights (can be learned)
        self.gate_cooc = 0.25
        self.gate_seq = 0.35
        self.gate_kg = 0.40
        
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query (simplified)."""
        # In production, use NER model
        words = query.lower().split()
        entities = []
        
        for word in words:
            if word in self.kg.entities:
                entities.append(word)
                
        # Also check multi-word entities
        for entity_id in self.kg.entities:
            if entity_id.lower() in query.lower():
                if entity_id not in entities:
                    entities.append(entity_id)
                    
        return entities
    
    def query(self, query_text: str, top_k: int = 10) -> QueryResult:
        """
        Process query through hybrid memory system.
        
        Combines evidence from:
        - Co-occurrence: semantic associations
        - Sequence: contextual passages
        - Knowledge graph: structured facts
        """
        # 1. Co-occurrence retrieval
        query_words = query_text.lower().split()
        cooc_results = []
        
        for word in query_words:
            similar = self.cooc.find_similar(word, top_k=5)
            cooc_results.extend(similar)
            
        # Deduplicate and sort
        cooc_dict = {}
        for word, score in cooc_results:
            if word not in cooc_dict or score > cooc_dict[word]:
                cooc_dict[word] = score
        cooc_results = sorted(cooc_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        # 2. Sequence retrieval
        seq_results_raw = self.seq.search(query_text, top_k=top_k)
        seq_results = [(doc_id, score) for doc_id, doc, score in seq_results_raw]
        
        # 3. Knowledge graph query
        entities = self.extract_entities(query_text)
        kg_results = []
        
        for entity in entities:
            # Get outgoing triples
            triples = self.kg.get_neighbors(entity, direction='out')
            kg_results.extend(triples[:5])
            
            # Get incoming triples
            triples = self.kg.get_neighbors(entity, direction='in')
            kg_results.extend(triples[:5])
            
        kg_results = kg_results[:top_k]
        
        # 4. Compute combined score
        combined_score = (
            self.gate_cooc * (sum(s for _, s in cooc_results[:3]) / 3 if cooc_results else 0) +
            self.gate_seq * (sum(s for _, s in seq_results[:3]) / 3 if seq_results else 0) +
            self.gate_kg * (len(kg_results) / top_k if kg_results else 0)
        )
        
        # 5. Generate explanation
        explanation = self._generate_explanation(query_text, cooc_results, 
                                                  seq_results_raw, kg_results)
        
        return QueryResult(
            cooc_results=cooc_results,
            seq_results=seq_results,
            kg_results=kg_results,
            combined_score=combined_score,
            explanation=explanation
        )
    
    def _generate_explanation(self, query: str, cooc: List, seq: List, kg: List) -> str:
        """Generate human-readable explanation of results."""
        parts = []
        
        if cooc:
            top_cooc = [w for w, _ in cooc[:3]]
            parts.append(f"Related concepts: {', '.join(top_cooc)}")
            
        if seq:
            parts.append(f"Found {len(seq)} relevant passages")
            
        if kg:
            parts.append(f"Found {len(kg)} knowledge graph facts")
            if kg:
                sample = kg[0]
                parts.append(f"Example: {sample.head} --{sample.relation}--> {sample.tail}")
                
        return "; ".join(parts) if parts else "No relevant information found"
    
    def update_gates(self, alpha: float, beta: float, gamma: float) -> None:
        """Update gating weights (should sum to 1)."""
        total = alpha + beta + gamma
        self.gate_cooc = alpha / total
        self.gate_seq = beta / total
        self.gate_kg = gamma / total


# =============================================================================
# Demonstration and Testing
# =============================================================================

def create_demo_system() -> HybridMemorySystem:
    """Create a demonstration hybrid memory system."""
    
    # 1. Create co-occurrence analyzer with sample data
    cooc = CooccurrenceAnalyzer(window_size=5, min_count=1, embedding_dim=50)
    
    sample_corpus = [
        "artificial intelligence machine learning deep neural networks".split(),
        "knowledge graph entity relation triple reasoning".split(),
        "natural language processing text understanding generation".split(),
        "machine learning models training data features".split(),
        "neural networks layers activation functions backpropagation".split(),
        "knowledge base ontology semantic web reasoning".split(),
        "language models transformers attention mechanism".split(),
        "graph neural networks node embedding message passing".split(),
        "text classification sentiment analysis named entity recognition".split(),
        "question answering reading comprehension information retrieval".split(),
    ]
    
    cooc.build_vocabulary(sample_corpus)
    cooc.build_cooccurrence_matrix(sample_corpus)
    cooc.compute_ppmi()
    cooc.compute_embeddings()
    
    # 2. Create sequence indexer
    encoder = SequenceEncoder(vocab_size=10000, embed_dim=128, num_heads=4, num_layers=2)
    seq_indexer = SequenceIndexer(encoder)
    
    sample_documents = [
        "Artificial intelligence and machine learning are transforming industries.",
        "Knowledge graphs provide structured representations of entities and relations.",
        "Natural language processing enables computers to understand human language.",
        "Deep learning uses neural networks with multiple layers.",
        "Ontologies define formal specifications of shared conceptualizations.",
        "Transformers use self-attention mechanisms for sequence modeling.",
        "Graph neural networks can reason over relational data structures.",
        "Large language models are trained on massive text corpora.",
    ]
    
    seq_indexer.add_documents(sample_documents)
    
    # 3. Create knowledge graph
    kg = KnowledgeGraph(embedding_dim=50)
    
    # Add entities
    entities_data = [
        ("AI", "Artificial Intelligence", "Technology"),
        ("ML", "Machine Learning", "Technology"),
        ("DL", "Deep Learning", "Technology"),
        ("NLP", "Natural Language Processing", "Field"),
        ("KG", "Knowledge Graph", "DataStructure"),
        ("Transformer", "Transformer Architecture", "Model"),
        ("GPT", "Generative Pre-trained Transformer", "Model"),
        ("BERT", "Bidirectional Encoder Representations", "Model"),
    ]
    
    for eid, name, etype in entities_data:
        kg.add_entity(Entity(id=eid, name=name, entity_type=etype))
    
    # Add triples
    triples_data = [
        ("ML", "subfield_of", "AI"),
        ("DL", "subfield_of", "ML"),
        ("NLP", "uses", "ML"),
        ("NLP", "uses", "DL"),
        ("Transformer", "used_in", "NLP"),
        ("GPT", "is_a", "Transformer"),
        ("BERT", "is_a", "Transformer"),
        ("KG", "stores", "knowledge"),
        ("GPT", "is_a", "LLM"),
        ("BERT", "is_a", "LLM"),
    ]
    
    for h, r, t in triples_data:
        kg.add_triple(Triple(head=h, relation=r, tail=t))
    
    kg.compute_embeddings_transe(epochs=50)
    
    # 4. Create ontology
    onto = OntologyReasoner()
    
    onto.add_class("Thing")
    onto.add_class("Technology")
    onto.add_class("Model")
    onto.add_class("Field")
    onto.add_subclass("Technology", "Thing")
    onto.add_subclass("Model", "Thing")
    onto.add_subclass("Field", "Thing")
    
    onto.add_property("subfield_of", domain="Field", range_="Field", transitive=True)
    onto.add_property("uses", domain="Field", range_="Technology")
    onto.add_property("is_a", domain="Model", range_="Model", transitive=True)
    
    # Type assertions
    onto.add_instance_type("AI", "Technology")
    onto.add_instance_type("ML", "Technology")
    onto.add_instance_type("DL", "Technology")
    onto.add_instance_type("NLP", "Field")
    onto.add_instance_type("Transformer", "Model")
    onto.add_instance_type("GPT", "Model")
    onto.add_instance_type("BERT", "Model")
    
    # 5. Create hybrid system
    hybrid = HybridMemorySystem(cooc, seq_indexer, kg, onto)
    
    return hybrid


def main():
    """Demonstration of the hybrid memory system."""
    print("=" * 60)
    print("Hybrid Memory System for LLM Augmentation - Demo")
    print("=" * 60)
    
    # Create system
    print("\nInitializing hybrid memory system...")
    system = create_demo_system()
    
    # Test queries
    queries = [
        "What is machine learning?",
        "How do transformers work?",
        "What is the relationship between AI and deep learning?",
    ]
    
    for query in queries:
        print(f"\n{'=' * 60}")
        print(f"Query: {query}")
        print("-" * 60)
        
        result = system.query(query)
        
        print(f"\nCombined Score: {result.combined_score:.4f}")
        print(f"\nExplanation: {result.explanation}")
        
        if result.cooc_results:
            print(f"\nCo-occurrence Results (top 5):")
            for word, score in result.cooc_results[:5]:
                print(f"  - {word}: {score:.4f}")
                
        if result.kg_results:
            print(f"\nKnowledge Graph Results:")
            for triple in result.kg_results[:5]:
                print(f"  - {triple.head} --{triple.relation}--> {triple.tail}")
                
    print("\n" + "=" * 60)
    print("Demo complete!")
    

if __name__ == "__main__":
    main()
