"""
Dense Retrieval Module

Implements semantic retrieval using dense embeddings from
sentence transformers (Sentence-T5, Contriever, etc.)
"""

import numpy as np
from typing import List, Tuple, Optional

class DenseRetriever:
    """
    Dense passage retrieval using sentence embeddings.
    """
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the dense retriever.
        
        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.corpus_embeddings = None
        self.corpus = []
        self._load_model()
    
    def _load_model(self):
        """Load the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading model: {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print("Model loaded successfully")
        except ImportError:
            print("sentence-transformers not installed. Using fallback.")
            self.model = None
    
    def _encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts to embeddings."""
        if self.model is None:
            # Fallback: random embeddings for demo
            return np.random.randn(len(texts), 384)
        
        return self.model.encode(texts, show_progress_bar=False)
    
    def index_corpus(self, corpus: List[str]):
        """
        Index a corpus for retrieval.
        
        Args:
            corpus: List of passages to index
        """
        self.corpus = corpus
        print(f"Indexing {len(corpus)} passages...")
        self.corpus_embeddings = self._encode(corpus)
        print("Indexing complete")
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        corpus: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k passages for a query.
        
        Args:
            query: Search query
            top_k: Number of results
            corpus: Optional corpus (uses indexed if not provided)
            
        Returns:
            List of (passage, score) tuples
        """
        if corpus is not None:
            self.index_corpus(corpus)
        
        if self.corpus_embeddings is None:
            # Demo with sample corpus
            sample_corpus = [
                "Climate change is driven by greenhouse gas emissions.",
                "The greenhouse effect traps heat in Earth's atmosphere.",
                "Carbon dioxide levels have increased significantly.",
                "Renewable energy can help reduce emissions.",
                "Global temperatures are rising each decade."
            ]
            self.index_corpus(sample_corpus)
        
        query_embedding = self._encode([query])[0]
        
        # Cosine similarity
        similarities = self.corpus_embeddings @ query_embedding
        similarities /= (
            np.linalg.norm(self.corpus_embeddings, axis=1) * 
            np.linalg.norm(query_embedding) + 1e-10
        )
        
        # Get top-k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((self.corpus[idx], float(similarities[idx])))
        
        return results


"""
COMPLEXITY ANALYSIS: Dense Retrieval

Indexing:
- Time: O(n × d × L) where n = corpus size, d = embedding dim, L = avg length
- Space: O(n × d) for storing embeddings

Retrieval:
- Naive: O(n × d) for computing all similarities
- With FAISS/ScaNN: O(log(n) × d) approximate search

For this PoC, we use naive search which is sufficient for small corpora
(< 100k passages). Production systems should use approximate search.
"""
