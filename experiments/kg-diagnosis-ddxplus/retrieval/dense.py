"""Dense vector retrieval using OpenAI embeddings (or TF-IDF fallback)."""
import logging
import os
from typing import Dict, List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


class DenseRetriever:
    """Semantic similarity retrieval. Uses OpenAI embeddings if available, else TF-IDF."""

    def __init__(self, use_openai: bool = False):
        self.use_openai = use_openai and bool(os.environ.get("OPENAI_API_KEY"))
        self.vectorizer = None
        self.disease_matrix = None
        self.disease_names: List[str] = []
        self.disease_descriptions: Dict[str, str] = {}

    def build_index(self, kg):
        """Build disease description embeddings from KG."""
        diseases = kg.get_all_diseases()
        descriptions = []

        for disease in diseases:
            symptoms = kg.get_symptoms_for_disease(disease)
            top_symptoms = sorted(symptoms, key=lambda x: x[1], reverse=True)[:15]
            symptom_text = ", ".join([s[0] for s in top_symptoms])
            desc = f"{disease}: commonly presents with {symptom_text}" if symptom_text else disease
            descriptions.append(desc)
            self.disease_names.append(disease)
            self.disease_descriptions[disease] = desc

        if not descriptions:
            return

        if self.use_openai:
            self._build_openai_index(descriptions)
        else:
            self._build_tfidf_index(descriptions)

        logger.info(f"Built dense index for {len(self.disease_names)} diseases (mode={'openai' if self.use_openai else 'tfidf'})")

    def _build_tfidf_index(self, descriptions: List[str]):
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
        self.disease_matrix = self.vectorizer.fit_transform(descriptions)

    def _build_openai_index(self, descriptions: List[str]):
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(input=descriptions, model="text-embedding-3-small")
        self.disease_matrix = np.array([e.embedding for e in resp.data])

    def retrieve(self, symptoms: List[str], top_k: int = 10) -> List[Tuple[str, float]]:
        """Retrieve most similar diseases for given symptoms."""
        if self.disease_matrix is None or not self.disease_names:
            return []

        query = "Patient presents with: " + ", ".join(symptoms)

        if self.use_openai:
            from openai import OpenAI
            client = OpenAI()
            resp = client.embeddings.create(input=[query], model="text-embedding-3-small")
            query_vec = np.array(resp.data[0].embedding).reshape(1, -1)
            sims = cosine_similarity(query_vec, self.disease_matrix)[0]
        else:
            query_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(query_vec, self.disease_matrix)[0]

        indices = np.argsort(sims)[::-1][:top_k]
        return [(self.disease_names[i], float(sims[i])) for i in indices]
