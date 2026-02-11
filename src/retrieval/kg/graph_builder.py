"""Knowledge Graph retrieval with PPR traversal.

This module implements knowledge graph-based retrieval using:
1. Entity extraction and linking from documents
2. Graph construction from Wikidata or custom KG
3. Personalized PageRank (PPR) for entity expansion
4. IDF-weighted entity-to-document scoring
5. Optional OWL reasoning for ontological inference
"""

from typing import List, Tuple, Dict, Set, Optional, Any
from collections import defaultdict
import logging
import networkx as nx
import numpy as np

logger = logging.getLogger(__name__)


class EntityLinker:
    """Base class for entity linking.
    
    Override this class to implement custom entity linking.
    """
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract and link entities from text.
        
        Args:
            text: Input text.
            
        Returns:
            List of entity dicts with keys: id, label, types, span.
        """
        raise NotImplementedError


class SpacyEntityLinker(EntityLinker):
    """Entity linking using spaCy NER.
    
    This is a simple entity linker that uses spaCy's named entity
    recognition. For better results, use a dedicated entity linking
    system like REL, BLINK, or GENRE.
    """
    
    def __init__(self, model_name: str = "en_core_web_lg"):
        """Initialize spaCy entity linker.
        
        Args:
            model_name: spaCy model name.
        """
        try:
            import spacy
            self.nlp = spacy.load(model_name)
        except ImportError:
            raise ImportError("spacy is required. Install with: pip install spacy")
        except OSError:
            raise OSError(f"Model {model_name} not found. Download with: python -m spacy download {model_name}")
    
    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract entities using spaCy NER."""
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Create a simple ID from the entity text
            entity_id = f"{ent.label_}:{ent.text.lower().replace(' ', '_')}"
            
            entities.append({
                "id": entity_id,
                "label": ent.text,
                "types": [ent.label_],
                "span": (ent.start_char, ent.end_char)
            })
        
        return entities


class KnowledgeGraphRetriever:
    """Knowledge graph retriever with PPR and IDF scoring.
    
    This retriever:
    1. Builds a knowledge graph from document entities
    2. Links entities using an entity linker
    3. Traverses the graph using Personalized PageRank
    4. Scores documents using IDF-weighted entity coverage
    
    Attributes:
        graph: NetworkX directed graph.
        entity_to_docs: Mapping from entity IDs to document indices.
        entity_linker: Entity linking component.
        owl_reasoner: Optional OWL reasoner for ontological inference.
    """
    
    def __init__(
        self, 
        entity_linker: Optional[EntityLinker] = None,
        owl_reasoner: Optional['OWLReasoner'] = None
    ):
        """Initialize knowledge graph retriever.
        
        Args:
            entity_linker: Entity linking component.
            owl_reasoner: Optional OWL reasoner.
        """
        self.graph = nx.DiGraph()
        self.entity_to_docs: Dict[str, Set[int]] = defaultdict(set)
        self.doc_to_entities: Dict[int, Set[str]] = defaultdict(set)
        self.entity_linker = entity_linker
        self.owl_reasoner = owl_reasoner
        self.documents: List[str] = []
        self.entity_idf: Dict[str, float] = {}
    
    def build_from_documents(
        self, 
        documents: List[str],
        show_progress: bool = True
    ) -> None:
        """Build knowledge graph from documents.
        
        Args:
            documents: List of document texts.
            show_progress: Whether to show progress.
        """
        if self.entity_linker is None:
            logger.warning("No entity linker provided. Using SpacyEntityLinker.")
            self.entity_linker = SpacyEntityLinker()
        
        self.documents = documents
        
        iterator = enumerate(documents)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(list(iterator), desc="Extracting entities")
            except ImportError:
                iterator = enumerate(documents)
        
        for doc_id, doc in iterator:
            entities = self.entity_linker.extract(doc)
            
            for entity in entities:
                eid = entity["id"]
                
                # Map entity to document
                self.entity_to_docs[eid].add(doc_id)
                self.doc_to_entities[doc_id].add(eid)
                
                # Add entity to graph
                if eid not in self.graph:
                    self.graph.add_node(
                        eid,
                        label=entity["label"],
                        types=entity.get("types", [])
                    )
                
                # Add edges between co-occurring entities in same document
                for other_eid in self.doc_to_entities[doc_id]:
                    if other_eid != eid:
                        # Bidirectional co-occurrence edge
                        if self.graph.has_edge(eid, other_eid):
                            self.graph[eid][other_eid]["weight"] += 1
                        else:
                            self.graph.add_edge(eid, other_eid, weight=1, relation="co_occurs")
                        
                        if self.graph.has_edge(other_eid, eid):
                            self.graph[other_eid][eid]["weight"] += 1
                        else:
                            self.graph.add_edge(other_eid, eid, weight=1, relation="co_occurs")
        
        # Compute entity IDF
        n_docs = len(documents)
        for eid, docs in self.entity_to_docs.items():
            self.entity_idf[eid] = np.log(n_docs / (len(docs) + 1))
        
        logger.info(f"Built KG with {self.graph.number_of_nodes()} entities, "
                   f"{self.graph.number_of_edges()} edges")
    
    def add_external_relations(
        self, 
        relations: List[Tuple[str, str, str]]
    ) -> None:
        """Add external relations to the graph (e.g., from Wikidata).
        
        Args:
            relations: List of (subject_id, predicate, object_id) tuples.
        """
        for subj, pred, obj in relations:
            if subj in self.graph and obj in self.graph:
                self.graph.add_edge(subj, obj, relation=pred, weight=1)
    
    def ppr_expand(
        self, 
        seed_entities: List[str], 
        alpha: float = 0.85,
        top_k: int = 20
    ) -> Dict[str, float]:
        """Expand seed entities using Personalized PageRank.
        
        Args:
            seed_entities: List of seed entity IDs.
            alpha: Damping factor for PPR.
            top_k: Number of top entities to return.
            
        Returns:
            Dictionary mapping entity IDs to PPR scores.
        """
        if not seed_entities:
            return {}
        
        # Filter to entities in graph
        valid_seeds = [e for e in seed_entities if e in self.graph]
        
        if not valid_seeds:
            return {e: 1.0 for e in seed_entities if e in self.entity_to_docs}
        
        # Create personalization dict
        personalization = {eid: 1.0 / len(valid_seeds) for eid in valid_seeds}
        
        try:
            ppr_scores = nx.pagerank(
                self.graph,
                alpha=alpha,
                personalization=personalization,
                max_iter=100
            )
        except nx.PowerIterationFailedConvergence:
            logger.warning("PPR failed to converge, using seed entities only")
            return {e: 1.0 for e in valid_seeds}
        
        # Return top-k
        sorted_entities = sorted(ppr_scores.items(), key=lambda x: -x[1])[:top_k]
        return dict(sorted_entities)
    
    def owl_materialize(self, entity_ids: List[str]) -> List[str]:
        """Expand entities using OWL reasoning.
        
        Args:
            entity_ids: List of entity IDs.
            
        Returns:
            Expanded list of entity IDs.
        """
        if self.owl_reasoner is None:
            return entity_ids
        
        return self.owl_reasoner.materialize(entity_ids)
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        use_ppr: bool = True,
        use_owl: bool = True,
        ppr_alpha: float = 0.85,
        ppr_top_k: int = 20
    ) -> List[Tuple[int, float]]:
        """Retrieve documents via knowledge graph traversal.
        
        Args:
            query: Query string.
            k: Number of documents to retrieve.
            use_ppr: Whether to use PPR expansion.
            use_owl: Whether to use OWL materialization.
            ppr_alpha: PPR damping factor.
            ppr_top_k: Number of entities to consider from PPR.
            
        Returns:
            List of (document_index, score) tuples.
        """
        if self.entity_linker is None:
            return []
        
        # Step 1: Extract query entities
        query_entities = self.entity_linker.extract(query)
        seed_ids = [e["id"] for e in query_entities]
        
        if not seed_ids:
            return []
        
        # Step 2: OWL materialization (optional)
        if use_owl and self.owl_reasoner:
            seed_ids = self.owl_materialize(seed_ids)
        
        # Step 3: PPR expansion (optional)
        if use_ppr:
            entity_scores = self.ppr_expand(seed_ids, ppr_alpha, ppr_top_k)
        else:
            entity_scores = {eid: 1.0 for eid in seed_ids}
        
        # Step 4: Score documents using IDF-weighted entity coverage
        doc_scores: Dict[int, float] = defaultdict(float)
        
        for eid, ppr_score in entity_scores.items():
            if eid in self.entity_to_docs:
                idf = self.entity_idf.get(eid, 1.0)
                for doc_id in self.entity_to_docs[eid]:
                    doc_scores[doc_id] += ppr_score * idf
        
        # Return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:k]
        return sorted_docs
    
    def retrieve_batch(
        self, 
        queries: List[str], 
        k: int = 10,
        **kwargs
    ) -> List[List[Tuple[int, float]]]:
        """Retrieve documents for multiple queries."""
        return [self.retrieve(q, k, **kwargs) for q in queries]
    
    def get_entity_neighbors(
        self, 
        entity_id: str, 
        top_k: int = 10
    ) -> List[Tuple[str, str, float]]:
        """Get neighboring entities in the graph.
        
        Args:
            entity_id: Entity ID.
            top_k: Number of neighbors.
            
        Returns:
            List of (neighbor_id, relation, weight) tuples.
        """
        if entity_id not in self.graph:
            return []
        
        neighbors = []
        for neighbor in self.graph.neighbors(entity_id):
            edge_data = self.graph[entity_id][neighbor]
            neighbors.append((
                neighbor,
                edge_data.get("relation", "unknown"),
                edge_data.get("weight", 1.0)
            ))
        
        neighbors.sort(key=lambda x: -x[2])
        return neighbors[:top_k]
