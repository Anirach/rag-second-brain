"""
Knowledge Graph Retrieval Module

Implements entity linking and knowledge graph traversal
for structured knowledge retrieval.
"""

from typing import List, Dict, Tuple, Set
import re

class KGRetriever:
    """
    Knowledge graph retrieval with entity linking.
    """
    
    def __init__(self):
        """Initialize the KG retriever."""
        # Sample knowledge graph for demonstration
        self.kg = self._load_sample_kg()
        self.entity_aliases = self._load_entity_aliases()
    
    def _load_sample_kg(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load a sample knowledge graph for demonstration."""
        return {
            "Microsoft": [
                ("founded_by", "Bill Gates"),
                ("founded_by", "Paul Allen"),
                ("founded_in", "1975"),
                ("headquarters", "Redmond, Washington"),
                ("type", "Technology Company")
            ],
            "Bill Gates": [
                ("born_in", "1955"),
                ("occupation", "Businessman"),
                ("co-founded", "Microsoft"),
                ("spouse", "Melinda Gates")
            ],
            "Paris": [
                ("capital_of", "France"),
                ("population", "2.1 million"),
                ("contains", "Eiffel Tower"),
                ("type", "City")
            ],
            "France": [
                ("capital", "Paris"),
                ("type", "Country"),
                ("located_in", "Europe"),
                ("language", "French")
            ],
            "Einstein": [
                ("full_name", "Albert Einstein"),
                ("born_in", "1879"),
                ("died_in", "1955"),
                ("known_for", "Theory of Relativity"),
                ("occupation", "Physicist")
            ]
        }
    
    def _load_entity_aliases(self) -> Dict[str, str]:
        """Load entity name aliases."""
        return {
            "gates": "Bill Gates",
            "bill gates": "Bill Gates",
            "microsoft corporation": "Microsoft",
            "albert einstein": "Einstein",
            "eiffel tower": "Eiffel Tower",
            "french capital": "Paris"
        }
    
    def link_entities(self, query: str) -> List[str]:
        """
        Link mentions in query to KG entities.
        
        Args:
            query: Input query text
            
        Returns:
            List of linked entity IDs
        """
        query_lower = query.lower()
        entities = []
        
        # Check direct matches
        for entity in self.kg.keys():
            if entity.lower() in query_lower:
                entities.append(entity)
        
        # Check aliases
        for alias, entity in self.entity_aliases.items():
            if alias in query_lower and entity not in entities:
                entities.append(entity)
        
        return entities
    
    def retrieve_triples(
        self,
        entities: List[str],
        hops: int = 2
    ) -> List[Tuple[str, str, str]]:
        """
        Retrieve triples from KG starting from entities.
        
        Args:
            entities: Starting entities
            hops: Number of hops to traverse
            
        Returns:
            List of (subject, predicate, object) triples
        """
        triples = []
        visited: Set[str] = set()
        frontier = set(entities)
        
        for hop in range(hops):
            next_frontier: Set[str] = set()
            
            for entity in frontier:
                if entity in visited:
                    continue
                visited.add(entity)
                
                if entity in self.kg:
                    for predicate, obj in self.kg[entity]:
                        triples.append((entity, predicate, obj))
                        if obj in self.kg:
                            next_frontier.add(obj)
            
            frontier = next_frontier
        
        return triples
    
    def triples_to_text(self, triples: List[Tuple[str, str, str]]) -> List[str]:
        """Convert triples to natural language passages."""
        texts = []
        for subj, pred, obj in triples:
            # Simple template-based verbalization
            pred_text = pred.replace("_", " ")
            texts.append(f"{subj} {pred_text} {obj}.")
        return texts


"""
ALGORITHM: Entity Linking with Confidence

Input: Query Q, Entity Dictionary E, Threshold θ
Output: Linked entities with confidence scores

1. Tokenize Q into spans S = {s_1, ..., s_n}
2. For each span s_i:
   a. Compute similarity to all entities: sim(s_i, e) for e ∈ E
   b. If max(sim) > θ: add (e*, sim(s_i, e*)) to results
3. Apply non-maximum suppression to remove overlapping mentions
4. Return linked entities with confidence scores

COMPLEXITY:
- Time: O(|S| × |E|) for naive matching
- With trie/BPE: O(|Q| × log|E|)

LIMITATION: This PoC uses simple string matching.
Production systems should use BLINK or similar neural entity linkers.
"""
