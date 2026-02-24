"""
PPMI-based Co-occurrence Retrieval Module
Implements actual PPMI retrieval (not BM25 proxy) for RAG paper validation.
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, save_npz, load_npz
from collections import defaultdict, Counter
import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import re
from tqdm import tqdm
import math


class CooccurrenceMatrix:
    """Build and store PPMI co-occurrence matrix."""
    
    def __init__(self, window_size: int = 5, min_count: int = 5, top_n: int = 100):
        self.window_size = window_size
        self.min_count = min_count
        self.top_n = top_n  # Keep top N neighbors per term
        self.vocab = {}
        self.inv_vocab = {}
        self.ppmi_matrix = None
        self.word_counts = Counter()
        self.total_pairs = 0
        
    def tokenize(self, text: str) -> List[str]:
        """Simple whitespace + lowercase tokenization."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()
        return [t for t in tokens if len(t) > 1]
    
    def build_vocab(self, documents: List[str], max_vocab: int = 50000):
        """Build vocabulary from documents."""
        print("Building vocabulary...")
        for doc in tqdm(documents):
            tokens = self.tokenize(doc)
            self.word_counts.update(tokens)
        
        # Filter by min_count and take top max_vocab
        filtered = [(w, c) for w, c in self.word_counts.items() if c >= self.min_count]
        filtered.sort(key=lambda x: -x[1])
        filtered = filtered[:max_vocab]
        
        self.vocab = {w: i for i, (w, _) in enumerate(filtered)}
        self.inv_vocab = {i: w for w, i in self.vocab.items()}
        print(f"Vocabulary size: {len(self.vocab)}")
        
    def build_cooccurrence(self, documents: List[str]):
        """Build co-occurrence counts with sliding window."""
        print("Building co-occurrence matrix...")
        vocab_size = len(self.vocab)
        cooc = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        for doc in tqdm(documents):
            tokens = self.tokenize(doc)
            token_ids = [self.vocab[t] for t in tokens if t in self.vocab]
            
            for i, center_id in enumerate(token_ids):
                start = max(0, i - self.window_size)
                end = min(len(token_ids), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_id = token_ids[j]
                        cooc[center_id, context_id] += 1
                        self.total_pairs += 1
        
        self.cooc_matrix = cooc.tocsr()
        print(f"Total co-occurrence pairs: {self.total_pairs}")
        
    def compute_ppmi(self):
        """Compute PPMI from co-occurrence counts."""
        print("Computing PPMI...")
        vocab_size = len(self.vocab)
        
        # Row sums (word counts in context)
        row_sums = np.array(self.cooc_matrix.sum(axis=1)).flatten()
        col_sums = np.array(self.cooc_matrix.sum(axis=0)).flatten()
        total = self.cooc_matrix.sum()
        
        # Convert to probabilities and compute PPMI
        ppmi = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        cx = self.cooc_matrix.tocoo()
        for i, j, count in tqdm(zip(cx.row, cx.col, cx.data), total=len(cx.data)):
            if count > 0 and row_sums[i] > 0 and col_sums[j] > 0:
                # PMI = log(P(w,c) / (P(w) * P(c)))
                p_wc = count / total
                p_w = row_sums[i] / total
                p_c = col_sums[j] / total
                pmi = math.log(p_wc / (p_w * p_c))
                ppmi[i, j] = max(0, pmi)  # PPMI: clip negative values
        
        self.ppmi_matrix = ppmi.tocsr()
        print("PPMI computation complete.")
        
    def sparsify_top_n(self):
        """Keep only top-N neighbors per term for efficiency."""
        print(f"Sparsifying to top-{self.top_n} neighbors...")
        vocab_size = len(self.vocab)
        sparse_ppmi = lil_matrix((vocab_size, vocab_size), dtype=np.float32)
        
        for i in tqdm(range(vocab_size)):
            row = self.ppmi_matrix.getrow(i).toarray().flatten()
            if row.sum() > 0:
                # Get top N indices
                top_indices = np.argsort(row)[-self.top_n:]
                for j in top_indices:
                    if row[j] > 0:
                        sparse_ppmi[i, j] = row[j]
        
        self.ppmi_matrix = sparse_ppmi.tocsr()
        print(f"Sparsified matrix: {self.ppmi_matrix.nnz} non-zero entries")
        
    def save(self, path: str):
        """Save matrix and vocab to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        save_npz(path / "ppmi_matrix.npz", self.ppmi_matrix)
        with open(path / "vocab.json", "w") as f:
            json.dump(self.vocab, f)
        with open(path / "config.json", "w") as f:
            json.dump({
                "window_size": self.window_size,
                "min_count": self.min_count,
                "top_n": self.top_n,
                "vocab_size": len(self.vocab)
            }, f)
        print(f"Saved to {path}")
        
    def load(self, path: str):
        """Load matrix and vocab from disk."""
        path = Path(path)
        self.ppmi_matrix = load_npz(path / "ppmi_matrix.npz")
        with open(path / "vocab.json") as f:
            self.vocab = json.load(f)
        self.inv_vocab = {int(i): w for w, i in self.vocab.items()}
        with open(path / "config.json") as f:
            config = json.load(f)
            self.window_size = config["window_size"]
            self.min_count = config["min_count"]
            self.top_n = config["top_n"]
        print(f"Loaded from {path}, vocab size: {len(self.vocab)}")


class PPMIRetriever:
    """PPMI-based document retriever."""
    
    def __init__(self, ppmi_matrix: CooccurrenceMatrix):
        self.ppmi = ppmi_matrix
        
    def score_document(self, query_tokens: List[str], doc_tokens: List[str]) -> float:
        """
        Score a document against a query using PPMI.
        score(q,d) = Σ_i Σ_j PPMI(q_i, d_j) * (1 + log(tf(d_j)))
        """
        # Get token IDs
        q_ids = [self.ppmi.vocab[t] for t in query_tokens if t in self.ppmi.vocab]
        
        # Count doc tokens (for TF)
        doc_tf = Counter(doc_tokens)
        
        score = 0.0
        for q_id in q_ids:
            ppmi_row = self.ppmi.ppmi_matrix.getrow(q_id).toarray().flatten()
            
            for d_token, tf in doc_tf.items():
                if d_token in self.ppmi.vocab:
                    d_id = self.ppmi.vocab[d_token]
                    ppmi_val = ppmi_row[d_id]
                    if ppmi_val > 0:
                        # Sublinear TF weighting
                        tf_weight = 1 + math.log(tf) if tf > 0 else 0
                        score += ppmi_val * tf_weight
        
        return score
    
    def retrieve(self, query: str, documents: List[str], top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve top-k documents for a query.
        Returns list of (doc_index, score) tuples.
        """
        query_tokens = self.ppmi.tokenize(query)
        
        scores = []
        for i, doc in enumerate(documents):
            doc_tokens = self.ppmi.tokenize(doc)
            score = self.score_document(query_tokens, doc_tokens)
            scores.append((i, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]


def build_ppmi_from_hotpotqa(data_path: str, output_path: str):
    """Build PPMI matrix from HotpotQA corpus."""
    import json
    
    print(f"Loading HotpotQA from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)
    
    # Extract all passages
    documents = []
    for item in data:
        for title, sentences in item.get("context", []):
            doc = " ".join(sentences)
            documents.append(doc)
    
    print(f"Extracted {len(documents)} passages")
    
    # Build matrix
    ppmi = CooccurrenceMatrix(window_size=5, min_count=3, top_n=100)
    ppmi.build_vocab(documents, max_vocab=30000)
    ppmi.build_cooccurrence(documents)
    ppmi.compute_ppmi()
    ppmi.sparsify_top_n()
    ppmi.save(output_path)
    
    return ppmi


if __name__ == "__main__":
    # Test with sample data
    print("Testing PPMI retriever...")
    
    # Sample documents
    docs = [
        "The cat sat on the mat.",
        "Dogs are loyal pets.",
        "Cats and dogs are popular pets.",
        "The mat was soft and comfortable.",
        "Pets bring joy to families."
    ]
    
    # Build small PPMI matrix
    ppmi = CooccurrenceMatrix(window_size=2, min_count=1, top_n=10)
    ppmi.build_vocab(docs)
    ppmi.build_cooccurrence(docs)
    ppmi.compute_ppmi()
    
    # Test retrieval
    retriever = PPMIRetriever(ppmi)
    query = "cat pet"
    results = retriever.retrieve(query, docs, top_k=3)
    
    print(f"\nQuery: {query}")
    print("Results:")
    for idx, score in results:
        print(f"  [{score:.4f}] {docs[idx]}")
