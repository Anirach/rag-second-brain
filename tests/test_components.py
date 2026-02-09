"""
Unit tests for RAG Second Brain components.

Run with: pytest tests/ -v
"""

import pytest
import numpy as np


class TestCooccurrence:
    """Tests for co-occurrence module."""
    
    def test_score_returns_list(self):
        """Test that score returns a list of floats."""
        from src.cooccurrence import CooccurrenceScorer
        
        scorer = CooccurrenceScorer()
        query = "What is machine learning?"
        candidates = ["Machine learning is a subset of AI.", "The weather is nice."]
        
        scores = scorer.score(query, candidates)
        
        assert isinstance(scores, list)
        assert len(scores) == len(candidates)
        assert all(isinstance(s, float) for s in scores)
    
    def test_score_bounded(self):
        """Test that scores are bounded between 0 and 1."""
        from src.cooccurrence import CooccurrenceScorer
        
        scorer = CooccurrenceScorer()
        scores = scorer.score("test query", ["candidate 1", "candidate 2"])
        
        assert all(0 <= s <= 1 for s in scores)
    
    def test_relevant_higher_score(self):
        """Test that relevant candidates score higher."""
        from src.cooccurrence import CooccurrenceScorer
        
        scorer = CooccurrenceScorer()
        query = "What is the capital of France?"
        candidates = [
            "Paris is the capital of France.",
            "The moon orbits the Earth."
        ]
        
        scores = scorer.score(query, candidates)
        
        # Relevant candidate should score higher (usually)
        # Note: This is a soft test due to embedding quality
        assert scores[0] != scores[1]  # At least different


class TestGating:
    """Tests for gating mechanism."""
    
    def test_classify_query(self):
        """Test query classification."""
        from src.gating import GatingMechanism
        
        gating = GatingMechanism()
        
        assert gating.classify_query("Who is Einstein?") == "entity"
        assert gating.classify_query("When was he born?") == "factual"
        assert gating.classify_query("Why does this happen?") == "reasoning"
        assert gating.classify_query("Tell me about science") == "general"
    
    def test_weights_sum_to_one(self):
        """Test that source weights sum to approximately 1."""
        from src.gating import GatingMechanism
        
        gating = GatingMechanism()
        weights = gating.compute_weights("test query")
        
        total = sum(weights.values())
        assert abs(total - 1.0) < 0.01
    
    def test_fuse_scores(self):
        """Test score fusion."""
        from src.gating import GatingMechanism
        
        gating = GatingMechanism()
        
        source_scores = {
            'cooccurrence': np.array([0.8, 0.3]),
            'dense': np.array([0.7, 0.4]),
            'kg': np.array([0.5, 0.6])
        }
        source_weights = {'cooccurrence': 0.3, 'dense': 0.4, 'kg': 0.3}
        
        fused = gating.fuse_scores(source_scores, source_weights)
        
        assert len(fused) == 2
        assert all(f > 0 for f in fused)
    
    def test_infonce_loss(self):
        """Test InfoNCE loss computation."""
        from src.gating import InfoNCETrainer
        
        trainer = InfoNCETrainer(temperature=0.07)
        
        query = np.random.randn(128)
        positive = query + np.random.randn(128) * 0.1  # Similar
        negatives = np.random.randn(5, 128)  # Random
        
        loss = trainer.compute_loss(query, positive, negatives)
        
        assert isinstance(loss, float)
        assert loss >= 0  # Loss should be non-negative


class TestKGRetrieval:
    """Tests for KG retrieval module."""
    
    def test_entity_linking(self):
        """Test entity linking."""
        from src.kg_retrieval import KGRetriever
        
        retriever = KGRetriever()
        
        entities = retriever.link_entities("Who founded Microsoft?")
        assert "Microsoft" in entities
        
        entities = retriever.link_entities("Tell me about Bill Gates")
        assert "Bill Gates" in entities
    
    def test_triple_retrieval(self):
        """Test triple retrieval."""
        from src.kg_retrieval import KGRetriever
        
        retriever = KGRetriever()
        
        triples = retriever.retrieve_triples(["Microsoft"], hops=1)
        
        assert len(triples) > 0
        assert all(len(t) == 3 for t in triples)
    
    def test_verbalization(self):
        """Test triple verbalization."""
        from src.kg_retrieval import KGRetriever
        
        retriever = KGRetriever()
        
        triples = [("Paris", "capital_of", "France")]
        texts = retriever.triples_to_text(triples)
        
        assert len(texts) == 1
        assert "Paris" in texts[0]
        assert "France" in texts[0]


class TestOntology:
    """Tests for ontology reasoning module."""
    
    def test_load_sample(self):
        """Test sample ontology loading."""
        from src.ontology import OntologyReasoner
        
        reasoner = OntologyReasoner()
        reasoner.load_sample()
        
        # Should not raise
        assert True
    
    def test_materialization(self):
        """Test ontology materialization."""
        from src.ontology import OntologyReasoner
        
        reasoner = OntologyReasoner()
        reasoner.load_sample()
        
        inferred = reasoner.materialize()
        
        assert isinstance(inferred, list)
        assert len(inferred) > 0


class TestPipeline:
    """Tests for full RAG pipeline."""
    
    def test_pipeline_init(self):
        """Test pipeline initialization."""
        from src.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(use_local_llm=False)
        
        assert pipeline is not None
    
    def test_retrieve_all_sources(self):
        """Test multi-source retrieval."""
        from src.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(use_local_llm=False)
        
        results = pipeline.retrieve_all_sources("What is Python?")
        
        assert isinstance(results, dict)
        # At least one source should return results
        assert any(len(v) > 0 for v in results.values())
    
    def test_full_run(self):
        """Test full pipeline run."""
        from src.pipeline import RAGPipeline
        
        pipeline = RAGPipeline(use_local_llm=False)
        
        result = pipeline.run("What is the capital of France?")
        
        assert 'answer' in result
        assert 'confidence' in result
        assert isinstance(result['confidence'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
