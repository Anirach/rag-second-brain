"""Learned gating fusion of multiple retrieval sources."""
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class GatingFusion:
    """Sigmoid-based learned gating mechanism for multi-source fusion."""

    def __init__(self):
        # Learnable parameters (can be tuned on validation set)
        self.gate_weights = np.array([0.4, 0.3, 0.3])  # dense, statistical, kg
        self.bias = np.array([0.0, 0.0, 0.0])

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -10, 10)))

    def _compute_query_features(self, symptoms: List[str]) -> np.ndarray:
        """Compute query-dependent features for gating."""
        n_symptoms = len(symptoms)
        avg_symptom_len = np.mean([len(s.split()) for s in symptoms]) if symptoms else 1.0
        specificity = 1.0 / max(n_symptoms, 1)  # fewer symptoms = more specific
        return np.array([n_symptoms / 10.0, avg_symptom_len / 5.0, specificity])

    def fuse(
        self,
        dense_results: List[Tuple[str, float]],
        stat_results: List[Tuple[str, float]],
        kg_results: List[Tuple[str, float]],
        symptoms: List[str],
        use_gating: bool = True,
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Fuse results from multiple retrieval sources."""
        if use_gating:
            features = self._compute_query_features(symptoms)
            gates = self._sigmoid(self.gate_weights * features + self.bias)
        else:
            gates = np.array([1.0, 1.0, 1.0]) / 3.0

        # Normalize gates
        gates = gates / (gates.sum() + 1e-8)

        # Merge all results
        all_diseases = set()
        dense_dict = {d: s for d, s in dense_results}
        stat_dict = {d: s for d, s in stat_results}
        kg_dict = {d: s for d, s in kg_results}

        all_diseases.update(dense_dict.keys())
        all_diseases.update(stat_dict.keys())
        all_diseases.update(kg_dict.keys())

        fused = []
        for disease in all_diseases:
            score = (
                gates[0] * dense_dict.get(disease, 0.0)
                + gates[1] * stat_dict.get(disease, 0.0)
                + gates[2] * kg_dict.get(disease, 0.0)
            )
            fused.append((disease, float(score)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]

    def fuse_subset(
        self,
        results_dict: Dict[str, List[Tuple[str, float]]],
        symptoms: List[str],
        top_k: int = 10,
    ) -> List[Tuple[str, float]]:
        """Fuse arbitrary subset of retrieval sources (for ablations)."""
        all_diseases = set()
        dicts = {}
        for name, results in results_dict.items():
            d = {disease: score for disease, score in results}
            dicts[name] = d
            all_diseases.update(d.keys())

        n_sources = max(len(dicts), 1)
        fused = []
        for disease in all_diseases:
            score = sum(d.get(disease, 0.0) for d in dicts.values()) / n_sources
            fused.append((disease, float(score)))

        fused.sort(key=lambda x: x[1], reverse=True)
        return fused[:top_k]
