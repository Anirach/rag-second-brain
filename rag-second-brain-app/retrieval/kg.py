"""Knowledge Graph retrieval with entity extraction."""
import re
from collections import defaultdict
from typing import List, Tuple, Dict, Set

# Domain-specific AI/ML entities for extraction
DOMAIN_ENTITIES = {
    "transformer", "attention", "bert", "gpt", "llm", "rag", "retrieval",
    "knowledge graph", "ontology", "owl", "embedding", "vector", "faiss",
    "bm25", "tf-idf", "ner", "named entity", "question answering", "qa",
    "multi-hop", "reasoning", "inference", "fine-tuning", "pre-training",
    "seq2seq", "encoder", "decoder", "cross-attention", "self-attention",
    "dense retrieval", "sparse retrieval", "passage retrieval", "dpr",
    "pmi", "ppmi", "co-occurrence", "semantic similarity", "cosine similarity",
    "neural network", "deep learning", "machine learning", "nlp",
    "language model", "hallucination", "grounding", "chain-of-thought",
    "mixture of experts", "gating", "fusion", "information fusion",
    "pagerank", "graph traversal", "entity linking", "relation extraction",
    "knowledge base", "triple store", "sparql", "bart", "roberta", "sbert",
    "sentence-bert", "wikipedia", "second brain", "pkm"
}

class KGRetriever:
    def __init__(self):
        self.entities: Dict[str, Set[int]] = defaultdict(set)  # entity -> doc_ids
        self.doc_entities: Dict[int, Set[str]] = defaultdict(set)  # doc_id -> entities
        self.relations: List[dict] = []  # {source, target, doc_id, type}
        self.doc_ids: List[int] = []
        self.nlp = None

    def _load_spacy(self):
        if self.nlp is None:
            try:
                import spacy
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                self.nlp = None

    def _extract_entities(self, text: str) -> Set[str]:
        """Extract entities using domain patterns + spaCy NER."""
        found = set()
        text_lower = text.lower()
        # Domain entity matching
        for entity in DOMAIN_ENTITIES:
            if entity in text_lower:
                found.add(entity)
        # spaCy NER
        self._load_spacy()
        if self.nlp:
            doc = self.nlp(text[:10000])  # limit for performance
            for ent in doc.ents:
                if ent.label_ in ("PERSON", "ORG", "PRODUCT", "WORK_OF_ART"):
                    name = ent.text.strip()
                    if len(name) > 2:
                        found.add(name.lower())
        return found

    def index(self, documents: List[dict]):
        """Build knowledge graph from documents."""
        self.entities.clear()
        self.doc_entities.clear()
        self.relations.clear()
        self.doc_ids = [d.get("id", i) for i, d in enumerate(documents)]

        for doc in documents:
            doc_id = doc.get("id", 0)
            text = f"{doc['title']}. {doc['abstract']}"
            ents = self._extract_entities(text)
            self.doc_entities[doc_id] = ents
            for e in ents:
                self.entities[e].add(doc_id)

        # Build co-occurrence relations (entities appearing in same document)
        for doc_id, ents in self.doc_entities.items():
            ent_list = sorted(ents)
            for i, e1 in enumerate(ent_list):
                for e2 in ent_list[i+1:]:
                    self.relations.append({
                        "source": e1, "target": e2,
                        "doc_id": doc_id, "type": "co-occurs"
                    })

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find documents sharing entities with query."""
        query_ents = self._extract_entities(query)
        if not query_ents:
            return []
        doc_scores = defaultdict(float)
        for ent in query_ents:
            if ent in self.entities:
                # IDF-like weighting: rarer entities score higher
                idf = 1.0 / (len(self.entities[ent]) + 1)
                for doc_id in self.entities[ent]:
                    doc_scores[doc_id] += 1.0 + idf
        ranked = sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
        return [(doc_id, score) for doc_id, score in ranked]

    def get_graph_data(self) -> dict:
        """Return graph data for visualization."""
        nodes = []
        node_set = set()
        for entity, doc_ids in self.entities.items():
            if len(doc_ids) >= 1:  # include all entities
                nodes.append({
                    "id": entity,
                    "label": entity,
                    "size": len(doc_ids),
                    "type": "entity"
                })
                node_set.add(entity)
        # Deduplicate edges
        edge_set = set()
        edges = []
        for rel in self.relations:
            key = (rel["source"], rel["target"])
            if key not in edge_set and rel["source"] in node_set and rel["target"] in node_set:
                edge_set.add(key)
                edges.append({"from": rel["source"], "to": rel["target"]})
        return {"nodes": nodes, "edges": edges}

    def get_query_entities(self, query: str) -> List[str]:
        return sorted(self._extract_entities(query))
