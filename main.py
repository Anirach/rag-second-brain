#!/usr/bin/env python3
"""
RAG Second Brain - Menu-Driven Proof of Concept

A multi-source RAG framework combining:
- Co-occurrence statistics
- Dense retrieval
- Knowledge graph
- Ontology reasoning
"""

import sys
from typing import Optional

def print_banner():
    """Print the application banner."""
    print("\n" + "=" * 50)
    print("  RAG Second Brain - Proof of Concept")
    print("  Multi-Source Retrieval with Learned Gating")
    print("=" * 50)

def print_menu():
    """Print the main menu."""
    print("\n--- Main Menu ---")
    print("1. Co-occurrence Module Demo")
    print("2. Dense Retrieval Demo")
    print("3. Knowledge Graph Demo")
    print("4. Ontology Reasoning Demo")
    print("5. Gating Mechanism Demo")
    print("6. Full Pipeline Demo")
    print("7. Run Proof-of-Concept Tests")
    print("8. Configuration")
    print("9. About")
    print("0. Exit")
    print("-" * 20)

def demo_cooccurrence():
    """Demonstrate co-occurrence module."""
    print("\n[Co-occurrence Module Demo]")
    try:
        from src.cooccurrence import CooccurrenceScorer
        scorer = CooccurrenceScorer()
        
        # Sample demonstration
        query = input("Enter a query (or press Enter for default): ").strip()
        if not query:
            query = "What is the capital of France?"
        
        candidates = [
            "Paris is the capital and largest city of France.",
            "The Eiffel Tower is located in Paris.",
            "London is the capital of the United Kingdom.",
            "France is a country in Western Europe."
        ]
        
        print(f"\nQuery: {query}")
        print("\nCandidates and scores:")
        scores = scorer.score(query, candidates)
        for candidate, score in zip(candidates, scores):
            print(f"  [{score:.4f}] {candidate[:60]}...")
            
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def demo_dense_retrieval():
    """Demonstrate dense retrieval module."""
    print("\n[Dense Retrieval Demo]")
    try:
        from src.dense_retrieval import DenseRetriever
        retriever = DenseRetriever()
        
        query = input("Enter a query (or press Enter for default): ").strip()
        if not query:
            query = "What causes climate change?"
        
        print(f"\nQuery: {query}")
        results = retriever.retrieve(query, top_k=5)
        print("\nTop-5 retrieved passages:")
        for i, (passage, score) in enumerate(results, 1):
            print(f"  {i}. [{score:.4f}] {passage[:60]}...")
            
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def demo_kg_retrieval():
    """Demonstrate knowledge graph retrieval."""
    print("\n[Knowledge Graph Demo]")
    try:
        from src.kg_retrieval import KGRetriever
        retriever = KGRetriever()
        
        query = input("Enter a query (or press Enter for default): ").strip()
        if not query:
            query = "Who founded Microsoft?"
        
        print(f"\nQuery: {query}")
        print("\nEntity linking...")
        entities = retriever.link_entities(query)
        print(f"Linked entities: {entities}")
        
        print("\nRetrieving from KG...")
        triples = retriever.retrieve_triples(entities, hops=2)
        print(f"Retrieved {len(triples)} triples")
        for triple in triples[:5]:
            print(f"  {triple}")
            
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def demo_ontology():
    """Demonstrate ontology reasoning."""
    print("\n[Ontology Reasoning Demo]")
    try:
        from src.ontology import OntologyReasoner
        reasoner = OntologyReasoner()
        
        print("\nLoading sample ontology...")
        reasoner.load_sample()
        
        print("\nRunning OWL 2 RL materialization...")
        inferred = reasoner.materialize()
        print(f"Inferred {len(inferred)} new triples")
        
        for triple in inferred[:5]:
            print(f"  INFERRED: {triple}")
            
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def demo_gating():
    """Demonstrate gating mechanism."""
    print("\n[Gating Mechanism Demo]")
    try:
        from src.gating import GatingMechanism
        gating = GatingMechanism()
        
        query = input("Enter a query (or press Enter for default): ").strip()
        if not query:
            query = "What year was Einstein born?"
        
        print(f"\nQuery: {query}")
        print("\nComputing source weights...")
        
        # Simulate candidates from each source
        weights = gating.compute_weights(query)
        print(f"\nSource weights:")
        print(f"  Co-occurrence: {weights['cooccurrence']:.3f}")
        print(f"  Dense:         {weights['dense']:.3f}")
        print(f"  KG:            {weights['kg']:.3f}")
        
        print(f"\nQuery type classification: {gating.classify_query(query)}")
        
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def demo_full_pipeline():
    """Demonstrate full RAG pipeline."""
    print("\n[Full Pipeline Demo]")
    try:
        from src.pipeline import RAGPipeline
        pipeline = RAGPipeline()
        
        query = input("Enter a query (or press Enter for default): ").strip()
        if not query:
            query = "What is the relationship between DNA and proteins?"
        
        print(f"\nQuery: {query}")
        print("\nRunning full pipeline...")
        print("  1. Co-occurrence retrieval...")
        print("  2. Dense retrieval...")
        print("  3. KG retrieval...")
        print("  4. Ontology materialization...")
        print("  5. Computing gating weights...")
        print("  6. Fusing candidates...")
        print("  7. Generating response...")
        
        result = pipeline.run(query)
        print(f"\n--- Result ---")
        print(f"Answer: {result['answer']}")
        print(f"Sources used: {result['sources']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
    except ImportError as e:
        print(f"Module not yet implemented: {e}")
        print("This will be added in future iterations.")

def run_tests():
    """Run proof-of-concept tests."""
    print("\n[Running Proof-of-Concept Tests]")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"],
            capture_output=True,
            text=True
        )
        print(result.stdout)
        if result.returncode != 0:
            print(result.stderr)
    except Exception as e:
        print(f"Error running tests: {e}")
        print("Make sure pytest is installed: pip install pytest")

def show_config():
    """Show and modify configuration."""
    print("\n[Configuration]")
    print("Current settings:")
    print("  - Embedding model: sentence-transformers/all-MiniLM-L6-v2")
    print("  - KG source: Wikidata (sample)")
    print("  - Ontology: OWL 2 RL profile")
    print("  - Gating: Learned (InfoNCE)")
    print("\n(Configuration editing will be added in future iterations)")

def show_about():
    """Show about information."""
    print("\n[About]")
    print("RAG Second Brain v0.1.0")
    print("\nA proof-of-concept implementation for the paper:")
    print("'Co-occurrence, Sequence and Knowledge Graph with")
    print(" Ontology as a Second Brain for AI-LLM'")
    print("\nAuthor: Anirach Mingkhwan")
    print("Repository: https://github.com/Anirach/rag-second-brain")
    print("\nThis is a conceptual implementation demonstrating")
    print("the proposed multi-source RAG framework.")

def main():
    """Main entry point."""
    print_banner()
    
    while True:
        print_menu()
        choice = input("Select option: ").strip()
        
        if choice == "1":
            demo_cooccurrence()
        elif choice == "2":
            demo_dense_retrieval()
        elif choice == "3":
            demo_kg_retrieval()
        elif choice == "4":
            demo_ontology()
        elif choice == "5":
            demo_gating()
        elif choice == "6":
            demo_full_pipeline()
        elif choice == "7":
            run_tests()
        elif choice == "8":
            show_config()
        elif choice == "9":
            show_about()
        elif choice == "0":
            print("\nGoodbye!")
            break
        else:
            print("\nInvalid option. Please try again.")

if __name__ == "__main__":
    main()
