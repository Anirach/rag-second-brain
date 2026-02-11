"""Dense retrieval using sentence transformers and FAISS.

This module implements dense passage retrieval using pre-trained
sentence transformers (E5, BGE) with FAISS indexing for efficient
similarity search.
"""

from typing import List, Tuple, Optional
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None


class DenseRetriever:
    """Dense passage retriever using sentence transformers.
    
    Attributes:
        model_name: Name of the sentence transformer model.
        model: The loaded sentence transformer model.
        index: FAISS index for similarity search.
        documents: List of indexed documents.
    """
    
    def __init__(self, model_name: str = "intfloat/e5-large-v2"):
        """Initialize the dense retriever.
        
        Args:
            model_name: HuggingFace model name for sentence embeddings.
                       Recommended: "intfloat/e5-large-v2" or "BAAI/bge-large-en-v1.5"
        """
        if SentenceTransformer is None:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.Index] = None
        self.documents: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
    
    def build_index(
        self, 
        documents: List[str], 
        batch_size: int = 32,
        show_progress: bool = True
    ) -> None:
        """Build FAISS index from documents.
        
        Args:
            documents: List of document texts to index.
            batch_size: Batch size for encoding.
            show_progress: Whether to show progress bar.
        """
        if faiss is None:
            raise ImportError("faiss is required. Install with: pip install faiss-cpu")
        
        self.documents = documents
        
        # Encode documents
        self.embeddings = self.model.encode(
            documents,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        # Build FAISS index (Inner Product for cosine similarity with normalized vectors)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings.astype(np.float32))
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10
    ) -> List[Tuple[int, float]]:
        """Retrieve top-k documents for a query.
        
        Args:
            query: Query string.
            k: Number of documents to retrieve.
            
        Returns:
            List of (document_index, score) tuples, sorted by score descending.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode query
        query_emb = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_emb, k)
        
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0])]
    
    def retrieve_batch(
        self, 
        queries: List[str], 
        k: int = 10,
        batch_size: int = 32
    ) -> List[List[Tuple[int, float]]]:
        """Retrieve top-k documents for multiple queries.
        
        Args:
            queries: List of query strings.
            k: Number of documents to retrieve per query.
            batch_size: Batch size for encoding.
            
        Returns:
            List of retrieval results, one per query.
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Encode queries
        query_embs = self.model.encode(
            queries,
            batch_size=batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True
        ).astype(np.float32)
        
        # Search
        scores, indices = self.index.search(query_embs, k)
        
        results = []
        for i in range(len(queries)):
            results.append([
                (int(idx), float(score)) 
                for idx, score in zip(indices[i], scores[i])
            ])
        
        return results
    
    def get_document(self, index: int) -> str:
        """Get document by index.
        
        Args:
            index: Document index.
            
        Returns:
            Document text.
        """
        return self.documents[index]
    
    def save_index(self, path: str) -> None:
        """Save FAISS index to disk.
        
        Args:
            path: Path to save the index.
        """
        if self.index is None:
            raise ValueError("No index to save.")
        faiss.write_index(self.index, path)
    
    def load_index(self, path: str, documents: List[str]) -> None:
        """Load FAISS index from disk.
        
        Args:
            path: Path to the saved index.
            documents: List of documents (must match indexed documents).
        """
        if faiss is None:
            raise ImportError("faiss is required.")
        self.index = faiss.read_index(path)
        self.documents = documents
