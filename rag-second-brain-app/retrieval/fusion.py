"""Sigmoid-based gating mechanism for multi-source fusion."""
import math
import re
from typing import List, Tuple, Dict

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20, min(20, x))))

class GatingFusion:
    """Learned sigmoid gating that weights retrieval sources based on query features."""

    # Pretrained-style weights (simulate learned gating behavior)
    # These produce reasonable gating for different query types
    WEIGHTS = {
        "dense": {"length": 0.3, "entity_count": -0.2, "is_what": 0.5,
                  "is_how": 0.6, "is_comparison": -0.3, "is_structural": -0.5,
                  "has_entities": -0.1, "bias": 1.2},
        "statistical": {"length": 0.1, "entity_count": -0.1, "is_what": 0.2,
                        "is_how": 0.3, "is_comparison": 0.5, "is_structural": -0.2,
                        "has_entities": -0.1, "bias": -0.3},
        "kg": {"length": -0.2, "entity_count": 0.8, "is_what": -0.1,
               "is_how": -0.3, "is_comparison": 0.2, "is_structural": 0.9,
               "has_entities": 0.7, "bias": -0.5},
    }

    def extract_features(self, query: str, entity_count: int = 0) -> Dict[str, float]:
        """Extract query features for gating."""
        q_lower = query.lower().strip()
        words = re.findall(r'\b\w+\b', q_lower)
        return {
            "length": min(len(words) / 20.0, 1.0),
            "entity_count": min(entity_count / 5.0, 1.0),
            "is_what": 1.0 if q_lower.startswith(("what", "which")) else 0.0,
            "is_how": 1.0 if q_lower.startswith(("how", "why")) else 0.0,
            "is_comparison": 1.0 if any(w in q_lower for w in ["compare", "difference", "between", "vs", "both"]) else 0.0,
            "is_structural": 1.0 if any(w in q_lower for w in ["cite", "connect", "relate", "link", "graph", "relationship"]) else 0.0,
            "has_entities": 1.0 if entity_count > 0 else 0.0,
        }

    def compute_gates(self, query: str, entity_count: int = 0) -> Dict[str, float]:
        """Compute gating weights for each source."""
        features = self.extract_features(query, entity_count)
        gates = {}
        for source, weights in self.WEIGHTS.items():
            z = weights["bias"]
            for feat, val in features.items():
                if feat in weights:
                    z += weights[feat] * val
            gates[source] = round(sigmoid(z), 3)
        return gates

    def fuse(self, results: Dict[str, List[Tuple[int, float]]],
             gates: Dict[str, float], top_k: int = 10) -> List[dict]:
        """Fuse results from multiple sources using gating weights."""
        # Collect all doc_ids
        all_docs = set()
        for source_results in results.values():
            for doc_id, _ in source_results:
                all_docs.add(doc_id)

        # Normalize scores per source (min-max)
        normalized = {}
        for source, source_results in results.items():
            if not source_results:
                normalized[source] = {}
                continue
            scores = [s for _, s in source_results]
            min_s, max_s = min(scores), max(scores)
            range_s = max_s - min_s if max_s > min_s else 1.0
            normalized[source] = {
                doc_id: (score - min_s) / range_s
                for doc_id, score in source_results
            }

        # Compute fused scores
        fused = []
        for doc_id in all_docs:
            source_scores = {}
            weighted_sum = 0.0
            gate_sum = 0.0
            for source in ["dense", "statistical", "kg"]:
                score = normalized.get(source, {}).get(doc_id, 0.0)
                source_scores[source] = round(score, 3)
                weighted_sum += gates.get(source, 0.0) * score
                gate_sum += gates.get(source, 0.0)
            fused_score = weighted_sum / (gate_sum + 1e-10)
            fused.append({
                "doc_id": doc_id,
                "fused_score": round(fused_score, 4),
                "source_scores": source_scores,
            })

        fused.sort(key=lambda x: -x["fused_score"])
        return fused[:top_k]
