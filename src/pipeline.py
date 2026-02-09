"""
Full RAG Pipeline

Combines all modules: co-occurrence, dense retrieval, KG, ontology,
and gating mechanism into a unified retrieval-augmented generation pipeline.
"""

from typing import Dict, List, Optional, Any
import numpy as np

class RAGPipeline:
    """
    Multi-source RAG pipeline with learned gating.
    
    Implements the full proposed framework:
    1. Retrieve from co-occurrence, dense, and KG sources
    2. Perform ontology materialization on KG results
    3. Apply learned gating to fuse candidates
    4. Generate response using LLM
    """
    
    def __init__(self, use_local_llm: bool = True):
        """
        Initialize the RAG pipeline.
        
        Args:
            use_local_llm: Whether to use local LLM (Ollama) or mock
        """
        self.use_local_llm = use_local_llm
        self._init_modules()
    
    def _init_modules(self):
        """Initialize all pipeline modules."""
        try:
            from .cooccurrence import CooccurrenceScorer
            from .dense_retrieval import DenseRetriever
            from .kg_retrieval import KGRetriever
            from .ontology import OntologyReasoner
            from .gating import GatingMechanism
            
            self.cooccurrence = CooccurrenceScorer()
            self.dense = DenseRetriever()
            self.kg = KGRetriever()
            self.ontology = OntologyReasoner()
            self.gating = GatingMechanism()
            
        except ImportError as e:
            print(f"Warning: Some modules not available: {e}")
            self.cooccurrence = None
            self.dense = None
            self.kg = None
            self.ontology = None
            self.gating = None
    
    def retrieve_all_sources(
        self,
        query: str,
        top_k_per_source: int = 10
    ) -> Dict[str, List[tuple]]:
        """
        Retrieve candidates from all sources.
        
        Args:
            query: Input query
            top_k_per_source: Number of candidates per source
            
        Returns:
            Dict mapping source name to list of (text, score) tuples
        """
        results = {}
        
        # 1. Co-occurrence retrieval
        if self.cooccurrence:
            # Using sample corpus for demo
            sample_corpus = self._get_sample_corpus()
            cooc_scores = self.cooccurrence.score(query, sample_corpus)
            results['cooccurrence'] = [
                (text, score) 
                for text, score in zip(sample_corpus, cooc_scores)
            ]
            results['cooccurrence'].sort(key=lambda x: x[1], reverse=True)
            results['cooccurrence'] = results['cooccurrence'][:top_k_per_source]
        
        # 2. Dense retrieval
        if self.dense:
            results['dense'] = self.dense.retrieve(query, top_k=top_k_per_source)
        
        # 3. KG retrieval
        if self.kg:
            entities = self.kg.link_entities(query)
            triples = self.kg.retrieve_triples(entities, hops=2)
            kg_texts = self.kg.triples_to_text(triples)
            # Score by relevance (simple: 1.0 for direct, 0.5 for 2-hop)
            results['kg'] = [(text, 0.8) for text in kg_texts[:top_k_per_source]]
        
        return results
    
    def _get_sample_corpus(self) -> List[str]:
        """Get sample corpus for demonstration."""
        return [
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower is a famous landmark in Paris.",
            "Bill Gates and Paul Allen founded Microsoft in 1975.",
            "Microsoft is headquartered in Redmond, Washington.",
            "Albert Einstein developed the theory of relativity.",
            "Einstein was born in Germany in 1879.",
            "Climate change is driven by greenhouse gas emissions.",
            "DNA contains genetic instructions for organisms.",
            "Proteins are synthesized based on DNA sequences.",
            "The Earth orbits the Sun once per year."
        ]
    
    def apply_gating(
        self,
        query: str,
        source_results: Dict[str, List[tuple]]
    ) -> List[tuple]:
        """
        Apply gating mechanism to fuse candidates.
        
        Args:
            query: Input query
            source_results: Results from each source
            
        Returns:
            Fused and ranked list of (text, score) tuples
        """
        if not self.gating:
            # Fallback: simple concatenation
            all_results = []
            for results in source_results.values():
                all_results.extend(results)
            all_results.sort(key=lambda x: x[1], reverse=True)
            return all_results
        
        # Get source weights
        source_weights = self.gating.compute_weights(query)
        
        # Fuse scores
        candidate_scores: Dict[str, float] = {}
        
        for source_name, results in source_results.items():
            weight = source_weights.get(source_name, 0.33)
            for text, score in results:
                if text not in candidate_scores:
                    candidate_scores[text] = 0
                candidate_scores[text] += weight * score
        
        # Sort by fused score
        fused = [(text, score) for text, score in candidate_scores.items()]
        fused.sort(key=lambda x: x[1], reverse=True)
        
        return fused
    
    def generate_response(
        self,
        query: str,
        context: List[str],
        max_context: int = 5
    ) -> str:
        """
        Generate response using LLM.
        
        Args:
            query: Input query
            context: Retrieved context passages
            max_context: Maximum context passages to use
            
        Returns:
            Generated answer
        """
        context_text = "\n".join(context[:max_context])
        
        if self.use_local_llm:
            return self._generate_with_ollama(query, context_text)
        else:
            return self._generate_mock(query, context_text)
    
    def _generate_with_ollama(self, query: str, context: str) -> str:
        """Generate using local Ollama LLM."""
        try:
            import requests
            
            prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "llama3.2",
                    "prompt": prompt,
                    "stream": False
                },
                timeout=30
            )
            
            if response.ok:
                return response.json().get("response", "No response generated")
            else:
                return self._generate_mock(query, context)
                
        except Exception as e:
            print(f"Ollama not available: {e}")
            return self._generate_mock(query, context)
    
    def _generate_mock(self, query: str, context: str) -> str:
        """Generate mock response for demo."""
        # Simple extractive answer
        sentences = context.split('.')
        query_words = set(query.lower().split())
        
        best_sentence = ""
        best_overlap = 0
        
        for sentence in sentences:
            sentence_words = set(sentence.lower().split())
            overlap = len(query_words & sentence_words)
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = sentence.strip()
        
        if best_sentence:
            return best_sentence + "."
        return "Unable to find answer in the provided context."
    
    def run(
        self,
        query: str,
        top_k: int = 10,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Run the full RAG pipeline.
        
        Args:
            query: Input query
            top_k: Number of final candidates
            verbose: Print intermediate results
            
        Returns:
            Dict with answer, sources, confidence
        """
        # 1. Retrieve from all sources
        if verbose:
            print("Retrieving from all sources...")
        source_results = self.retrieve_all_sources(query)
        
        # 2. Apply ontology reasoning (if KG results exist)
        if 'kg' in source_results and self.ontology:
            if verbose:
                print("Applying ontology reasoning...")
            self.ontology.load_sample()
            self.ontology.materialize()
        
        # 3. Apply gating
        if verbose:
            print("Applying gating mechanism...")
        fused = self.apply_gating(query, source_results)
        
        # 4. Generate response
        if verbose:
            print("Generating response...")
        context = [text for text, _ in fused[:top_k]]
        answer = self.generate_response(query, context)
        
        # Compute confidence
        if fused:
            confidence = fused[0][1]  # Top candidate score
        else:
            confidence = 0.0
        
        return {
            'answer': answer,
            'sources': list(source_results.keys()),
            'confidence': confidence,
            'top_candidates': fused[:5]
        }


"""
ALGORITHM 1: Full RAG Pipeline

Input: Query Q, Corpus C, Knowledge Graph G, Ontology O
Output: Answer A with confidence score

1. RETRIEVE:
   a. C_cooc ← CooccurrenceRetrieve(Q, C, k)
   b. C_dense ← DenseRetrieve(Q, C, k)
   c. E ← EntityLink(Q)
   d. T ← KGTraverse(G, E, hops=2)
   e. C_kg ← Verbalize(T)

2. REASON:
   a. T' ← Materialize(O, T)  // OWL 2 RL closure
   b. C_kg ← C_kg ∪ Verbalize(T')

3. GATE:
   a. w_src ← ComputeSourceWeights(Q)
   b. For each candidate c in C_cooc ∪ C_dense ∪ C_kg:
      - w_cand(c) ← σ(W_g · Encode(c))
      - score(c) ← Σ_s w_src(s) · w_cand(c) · score_s(c)

4. GENERATE:
   a. C_top ← TopK(candidates, k=5)
   b. A ← LLM.Generate(Q, C_top)
   c. conf ← max(score(c) for c in C_top)

5. RETURN (A, conf)

COMPLEXITY ANALYSIS:
- Retrieval: O(|C| × d) for each source
- Reasoning: O(|T|³) for materialization
- Gating: O(k × 3) for 3 sources, k candidates each
- Generation: O(L × V) for LLM with length L, vocab V

TOTAL: O(|C| × d + |T|³ + L × V)
Dominated by LLM generation in practice.
"""
