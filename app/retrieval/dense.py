"""Dense vector retrieval using sentence-transformers."""
import numpy as np
from typing import List, Tuple, Optional

class DenseRetriever:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self.doc_embeddings: Optional[np.ndarray] = None
        self.doc_ids: List[int] = []

    def _load_model(self):
        if self.model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.model = SentenceTransformer(self.model_name)
            except Exception:
                self.model = None

    def index(self, documents: List[dict]):
        """Index documents by computing embeddings."""
        self._load_model()
        if self.model is None:
            # Fallback: TF-IDF based embeddings
            self._index_fallback(documents)
            return
        texts = [f"{d['title']}. {d['abstract']}" for d in documents]
        self.doc_embeddings = self.model.encode(texts, normalize_embeddings=True)
        self.doc_ids = [d.get("id", i) for i, d in enumerate(documents)]

    def _index_fallback(self, documents: List[dict]):
        """Fallback using sklearn TF-IDF when sentence-transformers unavailable."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        texts = [f"{d['title']}. {d['abstract']}" for d in documents]
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.doc_embeddings = self.vectorizer.fit_transform(texts).toarray()
        self.doc_ids = [d.get("id", i) for i, d in enumerate(documents)]

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """Search for documents similar to query. Returns [(doc_id, score)]."""
        if self.doc_embeddings is None:
            return []
        self._load_model()
        if self.model is not None:
            q_emb = self.model.encode([query], normalize_embeddings=True)
            scores = (q_emb @ self.doc_embeddings.T).flatten()
        elif hasattr(self, 'vectorizer'):
            q_emb = self.vectorizer.transform([query]).toarray()
            norms_q = np.linalg.norm(q_emb, axis=1, keepdims=True) + 1e-10
            norms_d = np.linalg.norm(self.doc_embeddings, axis=1, keepdims=True) + 1e-10
            scores = (q_emb / norms_q @ (self.doc_embeddings / norms_d.T).T).flatten()
        else:
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices]
