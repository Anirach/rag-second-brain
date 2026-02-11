"""PPMI-based co-occurrence retrieval module.

This module implements retrieval based on Positive Pointwise Mutual Information
(PPMI) co-occurrence statistics. Unlike BM25, PPMI captures semantic relationships
between terms based on their co-occurrence patterns in the corpus.

Key features:
- Builds word co-occurrence matrix from corpus
- Computes PPMI weights for term relationships
- Supports query expansion using PPMI-weighted related terms
- Provides complementary signal to dense retrieval
"""

from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import re
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
import logging

logger = logging.getLogger(__name__)


class PPMIRetriever:
    """PPMI-based co-occurrence retriever.
    
    This retriever uses Positive Pointwise Mutual Information to capture
    statistical co-occurrence patterns between terms. It provides complementary
    retrieval signal to dense embeddings by capturing corpus-specific
    term associations.
    
    Attributes:
        window_size: Size of the co-occurrence window.
        min_count: Minimum term frequency to include in vocabulary.
        vocab: Mapping from terms to indices.
        inv_vocab: Mapping from indices to terms.
        ppmi_matrix: Sparse PPMI co-occurrence matrix.
        doc_term_matrix: Sparse document-term matrix.
    """
    
    def __init__(
        self, 
        window_size: int = 5, 
        min_count: int = 5,
        max_vocab_size: Optional[int] = 100000
    ):
        """Initialize the PPMI retriever.
        
        Args:
            window_size: Size of sliding window for co-occurrence.
            min_count: Minimum term frequency to include in vocabulary.
            max_vocab_size: Maximum vocabulary size (most frequent terms kept).
        """
        self.window_size = window_size
        self.min_count = min_count
        self.max_vocab_size = max_vocab_size
        
        self.vocab: Dict[str, int] = {}
        self.inv_vocab: Dict[int, str] = {}
        self.ppmi_matrix: Optional[csr_matrix] = None
        self.doc_term_matrix: Optional[csr_matrix] = None
        self.documents: List[str] = []
        self.idf: Optional[np.ndarray] = None
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization.
        
        Args:
            text: Input text.
            
        Returns:
            List of lowercase tokens.
        """
        # Simple whitespace + punctuation tokenization
        text = text.lower()
        tokens = re.findall(r'\b[a-z]+\b', text)
        return tokens
    
    def build_from_corpus(
        self, 
        documents: List[str],
        show_progress: bool = True
    ) -> None:
        """Build PPMI matrix and document-term matrix from corpus.
        
        Args:
            documents: List of document texts.
            show_progress: Whether to show progress.
        """
        self.documents = documents
        
        logger.info("Building vocabulary...")
        self._build_vocabulary(documents)
        
        logger.info(f"Vocabulary size: {len(self.vocab)}")
        
        logger.info("Building co-occurrence matrix...")
        cooc_matrix = self._build_cooccurrence_matrix(documents, show_progress)
        
        logger.info("Computing PPMI...")
        self.ppmi_matrix = self._compute_ppmi(cooc_matrix)
        
        logger.info("Building document-term matrix...")
        self._build_doc_term_matrix(documents)
        
        logger.info("PPMI retriever built successfully.")
    
    def _build_vocabulary(self, documents: List[str]) -> None:
        """Build vocabulary from documents."""
        word_counts: Dict[str, int] = defaultdict(int)
        
        for doc in documents:
            for token in self._tokenize(doc):
                word_counts[token] += 1
        
        # Filter by min_count and sort by frequency
        filtered = [
            (word, count) for word, count in word_counts.items()
            if count >= self.min_count
        ]
        filtered.sort(key=lambda x: -x[1])
        
        # Limit vocabulary size
        if self.max_vocab_size and len(filtered) > self.max_vocab_size:
            filtered = filtered[:self.max_vocab_size]
        
        # Build vocab mappings
        self.vocab = {word: idx for idx, (word, _) in enumerate(filtered)}
        self.inv_vocab = {idx: word for word, idx in self.vocab.items()}
    
    def _build_cooccurrence_matrix(
        self, 
        documents: List[str],
        show_progress: bool = True
    ) -> lil_matrix:
        """Build word co-occurrence matrix."""
        vocab_size = len(self.vocab)
        cooc = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        iterator = documents
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(documents, desc="Building co-occurrence")
            except ImportError:
                pass
        
        for doc in iterator:
            tokens = [t for t in self._tokenize(doc) if t in self.vocab]
            token_ids = [self.vocab[t] for t in tokens]
            
            for i, w1_id in enumerate(token_ids):
                # Window around current word
                start = max(0, i - self.window_size)
                end = min(len(token_ids), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        w2_id = token_ids[j]
                        # Weight by distance (closer = higher weight)
                        distance = abs(i - j)
                        weight = 1.0 / distance
                        cooc[w1_id, w2_id] += weight
        
        return cooc
    
    def _compute_ppmi(self, cooc: lil_matrix) -> csr_matrix:
        """Compute Positive PMI from co-occurrence matrix.
        
        PMI(w1, w2) = log2(P(w1, w2) / (P(w1) * P(w2)))
        PPMI = max(0, PMI)
        
        Args:
            cooc: Co-occurrence matrix.
            
        Returns:
            Sparse PPMI matrix.
        """
        cooc_csr = cooc.tocsr()
        
        # Total co-occurrences
        total = cooc_csr.sum()
        if total == 0:
            return csr_matrix(cooc_csr.shape)
        
        # Row sums (word frequencies in co-occurrence context)
        row_sums = np.array(cooc_csr.sum(axis=1)).flatten()
        col_sums = np.array(cooc_csr.sum(axis=0)).flatten()
        
        # Build PPMI matrix
        ppmi = lil_matrix(cooc_csr.shape, dtype=np.float32)
        
        # Convert to COO for efficient iteration
        cooc_coo = cooc_csr.tocoo()
        
        for i, j, count in zip(cooc_coo.row, cooc_coo.col, cooc_coo.data):
            if count > 0:
                # PMI = log2((count * total) / (row_sum * col_sum))
                pmi = np.log2(
                    (count * total) / (row_sums[i] * col_sums[j] + 1e-10)
                )
                # Positive PMI
                if pmi > 0:
                    ppmi[i, j] = pmi
        
        return ppmi.tocsr()
    
    def _build_doc_term_matrix(self, documents: List[str]) -> None:
        """Build document-term matrix for retrieval."""
        n_docs = len(documents)
        vocab_size = len(self.vocab)
        
        doc_term = lil_matrix((n_docs, vocab_size), dtype=np.float32)
        doc_freqs = np.zeros(vocab_size)
        
        for doc_id, doc in enumerate(documents):
            term_counts: Dict[int, int] = defaultdict(int)
            for token in self._tokenize(doc):
                if token in self.vocab:
                    term_counts[self.vocab[token]] += 1
            
            for term_id, count in term_counts.items():
                doc_term[doc_id, term_id] = count
                doc_freqs[term_id] += 1
        
        self.doc_term_matrix = doc_term.tocsr()
        
        # Compute IDF
        self.idf = np.log(n_docs / (doc_freqs + 1))
    
    def expand_query(
        self, 
        query_terms: List[str], 
        top_k_expansion: int = 10,
        expansion_weight: float = 0.5
    ) -> Dict[int, float]:
        """Expand query using PPMI-weighted related terms.
        
        Args:
            query_terms: List of query terms.
            top_k_expansion: Number of related terms to add per query term.
            expansion_weight: Weight for expanded terms relative to original.
            
        Returns:
            Dictionary mapping term indices to weights.
        """
        query_weights: Dict[int, float] = defaultdict(float)
        
        for term in query_terms:
            if term not in self.vocab:
                continue
                
            term_id = self.vocab[term]
            query_weights[term_id] += 1.0
            
            # Get PPMI-related terms
            if self.ppmi_matrix is not None:
                ppmi_row = self.ppmi_matrix[term_id].toarray().flatten()
                top_related = np.argsort(ppmi_row)[-top_k_expansion:]
                
                for related_id in top_related:
                    if ppmi_row[related_id] > 0:
                        query_weights[related_id] += expansion_weight * ppmi_row[related_id]
        
        return dict(query_weights)
    
    def retrieve(
        self, 
        query: str, 
        k: int = 10,
        use_expansion: bool = True,
        top_k_expansion: int = 10,
        expansion_weight: float = 0.5
    ) -> List[Tuple[int, float]]:
        """Retrieve documents using PPMI-weighted scoring.
        
        Args:
            query: Query string.
            k: Number of documents to retrieve.
            use_expansion: Whether to use PPMI-based query expansion.
            top_k_expansion: Number of expansion terms per query term.
            expansion_weight: Weight for expanded terms.
            
        Returns:
            List of (document_index, score) tuples.
        """
        if self.doc_term_matrix is None:
            raise ValueError("Retriever not built. Call build_from_corpus() first.")
        
        query_terms = [t for t in self._tokenize(query) if t in self.vocab]
        
        if not query_terms:
            return []
        
        if use_expansion:
            query_weights = self.expand_query(
                query_terms, top_k_expansion, expansion_weight
            )
        else:
            query_weights = {self.vocab[t]: 1.0 for t in query_terms}
        
        # Build query vector
        query_vec = np.zeros(len(self.vocab))
        for term_id, weight in query_weights.items():
            query_vec[term_id] = weight * self.idf[term_id]
        
        # Score documents
        doc_scores = self.doc_term_matrix.dot(query_vec)
        
        # Get top-k
        top_k_indices = np.argsort(doc_scores)[-k:][::-1]
        
        return [
            (int(idx), float(doc_scores[idx])) 
            for idx in top_k_indices 
            if doc_scores[idx] > 0
        ]
    
    def retrieve_batch(
        self, 
        queries: List[str], 
        k: int = 10,
        **kwargs
    ) -> List[List[Tuple[int, float]]]:
        """Retrieve documents for multiple queries.
        
        Args:
            queries: List of query strings.
            k: Number of documents per query.
            **kwargs: Additional arguments for retrieve().
            
        Returns:
            List of retrieval results.
        """
        return [self.retrieve(q, k, **kwargs) for q in queries]
    
    def get_related_terms(
        self, 
        term: str, 
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """Get PPMI-related terms for a given term.
        
        Args:
            term: Input term.
            top_k: Number of related terms.
            
        Returns:
            List of (term, ppmi_score) tuples.
        """
        if term not in self.vocab or self.ppmi_matrix is None:
            return []
        
        term_id = self.vocab[term]
        ppmi_row = self.ppmi_matrix[term_id].toarray().flatten()
        top_indices = np.argsort(ppmi_row)[-top_k:][::-1]
        
        return [
            (self.inv_vocab[idx], float(ppmi_row[idx]))
            for idx in top_indices
            if ppmi_row[idx] > 0
        ]
