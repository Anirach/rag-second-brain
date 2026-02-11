#!/usr/bin/env python3
"""Test script to verify the setup is working correctly."""

import sys

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    errors = []
    
    # Core modules
    try:
        from src.retrieval import DenseRetriever, PPMIRetriever, KnowledgeGraphRetriever
        print("  ✓ Retrieval modules")
    except ImportError as e:
        errors.append(f"  ✗ Retrieval modules: {e}")
    
    try:
        from src.fusion import rrf_fusion, LearnedGating, CrossAttentionFusion
        print("  ✓ Fusion modules")
    except ImportError as e:
        errors.append(f"  ✗ Fusion modules: {e}")
    
    try:
        from src.evaluation import recall_at_k, exact_match, f1_score
        print("  ✓ Evaluation modules")
    except ImportError as e:
        errors.append(f"  ✗ Evaluation modules: {e}")
    
    try:
        from src.utils import load_hotpotqa, load_documents
        print("  ✓ Utils modules")
    except ImportError as e:
        errors.append(f"  ✗ Utils modules: {e}")
    
    return errors


def test_dependencies():
    """Test that key dependencies are installed."""
    print("\nTesting dependencies...")
    
    deps = {
        "torch": "PyTorch",
        "transformers": "Transformers",
        "sentence_transformers": "Sentence Transformers",
        "faiss": "FAISS",
        "owlready2": "OWLReady2",
        "networkx": "NetworkX",
        "scipy": "SciPy",
        "datasets": "HuggingFace Datasets",
        "numpy": "NumPy",
        "spacy": "spaCy",
    }
    
    installed = []
    missing = []
    
    for module, name in deps.items():
        try:
            __import__(module)
            installed.append(name)
            print(f"  ✓ {name}")
        except ImportError:
            missing.append(name)
            print(f"  ✗ {name} (not installed)")
    
    return missing


def test_basic_functionality():
    """Test basic functionality of key components."""
    print("\nTesting basic functionality...")
    
    errors = []
    
    # Test RRF fusion
    try:
        from src.fusion import rrf_fusion
        ranking1 = [(1, 0.9), (2, 0.8), (3, 0.7)]
        ranking2 = [(2, 0.95), (1, 0.85), (4, 0.75)]
        result = rrf_fusion([ranking1, ranking2])
        assert len(result) == 4
        print("  ✓ RRF fusion")
    except Exception as e:
        errors.append(f"  ✗ RRF fusion: {e}")
    
    # Test evaluation metrics
    try:
        from src.evaluation import recall_at_k, exact_match, f1_score
        
        # Recall@K
        retrieved = [1, 2, 3, 4, 5]
        relevant = {1, 3, 6}
        r = recall_at_k(retrieved, relevant, k=5)
        assert r == 2/3
        
        # Exact match
        assert exact_match("the answer", "The Answer") == 1.0
        assert exact_match("the answer", "another answer") == 0.0
        
        # F1 score
        f1 = f1_score("the quick brown fox", "the fast brown fox")
        assert f1 > 0.5
        
        print("  ✓ Evaluation metrics")
    except Exception as e:
        errors.append(f"  ✗ Evaluation metrics: {e}")
    
    # Test PPMI retriever initialization
    try:
        from src.retrieval import PPMIRetriever
        ppmi = PPMIRetriever(window_size=5, min_count=2)
        assert ppmi.window_size == 5
        print("  ✓ PPMI retriever initialization")
    except Exception as e:
        errors.append(f"  ✗ PPMI retriever: {e}")
    
    return errors


def main():
    """Run all tests."""
    print("=" * 50)
    print("RAG Second Brain v16 - Setup Test")
    print("=" * 50)
    
    all_errors = []
    
    # Test imports
    import_errors = test_imports()
    all_errors.extend(import_errors)
    
    # Test dependencies
    missing_deps = test_dependencies()
    
    # Test functionality
    func_errors = test_basic_functionality()
    all_errors.extend(func_errors)
    
    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    
    if all_errors:
        print("\n⚠️  Some tests failed:")
        for err in all_errors:
            print(err)
    else:
        print("\n✅ All tests passed!")
    
    if missing_deps:
        print(f"\n📦 Missing dependencies: {', '.join(missing_deps)}")
        print("   Install with: pip install -r requirements.txt")
    
    return 0 if not all_errors else 1


if __name__ == "__main__":
    sys.exit(main())
