"""BM25 + PPMI co-occurrence retrieval."""
import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

class StatisticalRetriever:
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.bm25 = None
        self.doc_ids: List[int] = []
        self.ppmi_matrix: Dict[str, Dict[str, float]] = {}
        self.tokenized_corpus: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\b[a-zA-Z]{2,}\b', text.lower())

    def index(self, documents: List[dict]):
        """Build BM25 index and PPMI matrix."""
        from rank_bm25 import BM25Okapi
        texts = [f"{d['title']}. {d['abstract']}" for d in documents]
        self.tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.doc_ids = [d.get("id", i) for i, d in enumerate(documents)]
        self._build_ppmi(texts)

    def _build_ppmi(self, texts: List[str]):
        """Build PPMI co-occurrence matrix from corpus."""
        cooccur = defaultdict(Counter)
        word_count = Counter()
        total_windows = 0
        for text in texts:
            tokens = self._tokenize(text)
            for i, w in enumerate(tokens):
                word_count[w] += 1
                window = tokens[max(0, i - self.window_size):i] + tokens[i+1:i+1+self.window_size]
                for ctx in window:
                    if ctx != w:
                        cooccur[w][ctx] += 1
                        total_windows += 1
        if total_windows == 0:
            return
        total_words = sum(word_count.values())
        self.ppmi_matrix = {}
        for w in cooccur:
            self.ppmi_matrix[w] = {}
            for ctx in cooccur[w]:
                p_wc = cooccur[w][ctx] / total_windows
                p_w = word_count[w] / total_words
                p_c = word_count[ctx] / total_words
                pmi = math.log2(p_wc / (p_w * p_c + 1e-15) + 1e-15)
                if pmi > 0:
                    self.ppmi_matrix[w][ctx] = round(pmi, 3)
            # Keep top 20 associations per word
            if len(self.ppmi_matrix[w]) > 20:
                top = sorted(self.ppmi_matrix[w].items(), key=lambda x: -x[1])[:20]
                self.ppmi_matrix[w] = dict(top)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """BM25 search. Returns [(doc_id, score)]."""
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]
        return [(self.doc_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

    def get_expanded_terms(self, query: str, top_n: int = 8) -> List[Tuple[str, float]]:
        """Get PPMI-expanded terms for a query."""
        tokens = self._tokenize(query)
        expanded = Counter()
        for t in tokens:
            if t in self.ppmi_matrix:
                for ctx, score in self.ppmi_matrix[t].items():
                    if ctx not in tokens:
                        expanded[ctx] += score
        return expanded.most_common(top_n)
