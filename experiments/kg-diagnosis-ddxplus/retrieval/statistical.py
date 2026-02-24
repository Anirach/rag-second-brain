"""BM25 + PPMI statistical retrieval."""
import logging
from collections import defaultdict
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


class StatisticalRetriever:
    """BM25 + PPMI co-occurrence based retrieval."""

    def __init__(self):
        self.bm25 = None
        self.disease_docs: List[str] = []
        self.disease_names: List[str] = []
        self.kg = None

    def build_index(self, kg):
        """Build BM25 index from KG disease-symptom associations."""
        from rank_bm25 import BM25Okapi

        self.kg = kg
        diseases = kg.get_all_diseases()
        self.disease_names = []
        corpus = []

        for disease in diseases:
            symptoms = kg.get_symptoms_for_disease(disease)
            tokens = [s[0].lower().replace(" ", "_") for s in symptoms]
            if tokens:
                corpus.append(tokens)
                self.disease_names.append(disease)

        if corpus:
            self.bm25 = BM25Okapi(corpus)
            logger.info(f"Built BM25 index for {len(corpus)} diseases")
        else:
            logger.warning("No disease documents for BM25 index")

    def retrieve(self, symptoms: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve diseases using BM25 + PPMI."""
        if self.bm25 is None or not self.disease_names:
            return []

        # BM25 scores
        query_tokens = [s.lower().replace(" ", "_") for s in symptoms]
        bm25_scores = self.bm25.get_scores(query_tokens)

        # PPMI scores
        ppmi_scores = defaultdict(float)
        if self.kg:
            for symptom in symptoms:
                for disease in self.disease_names:
                    ppmi = self.kg.get_ppmi(symptom, disease)
                    if ppmi > 0:
                        ppmi_scores[disease] += ppmi

        # Combine: normalize and weight
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        max_ppmi = max(ppmi_scores.values()) if ppmi_scores else 1.0

        combined = []
        for i, disease in enumerate(self.disease_names):
            bm25_norm = bm25_scores[i] / max_bm25
            ppmi_norm = ppmi_scores.get(disease, 0) / max_ppmi
            score = 0.5 * bm25_norm + 0.5 * ppmi_norm
            combined.append((disease, score))

        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]
