"""Knowledge Graph traversal retrieval."""
import logging
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class KGRetriever:
    """Graph-based retrieval via multi-hop traversal."""

    def __init__(self):
        self.kg = None

    def build_index(self, kg):
        """Store reference to KG."""
        self.kg = kg
        logger.info("KG retriever initialized")

    def retrieve(self, symptoms: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve diseases via graph traversal."""
        if self.kg is None:
            return []

        scores = self.kg.graph_traverse(symptoms, max_hops=2)

        # Normalize
        max_score = max(scores.values()) if scores else 1.0
        results = [(d, s / max_score) for d, s in scores.items()]
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
