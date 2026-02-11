"""Data loading utilities for RAG experiments."""

from typing import List, Dict, Tuple, Optional, Any
import json
import logging

logger = logging.getLogger(__name__)


def load_hotpotqa(
    split: str = "validation",
    subset: str = "distractor"
) -> Tuple[List[Dict], List[str], Dict[str, int]]:
    """Load HotpotQA dataset.
    
    Args:
        split: Dataset split ("train" or "validation").
        subset: "distractor" or "fullwiki".
        
    Returns:
        Tuple of (questions, documents, doc_id_to_idx).
        - questions: List of question dicts with keys:
            - question, answer, supporting_facts, context
        - documents: List of document texts (title + sentences)
        - doc_id_to_idx: Mapping from doc title to index
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("datasets is required. Install with: pip install datasets")
    
    logger.info(f"Loading HotpotQA {subset} {split}...")
    
    dataset = load_dataset("hotpot_qa", subset, split=split)
    
    # Build document corpus from contexts
    documents = []
    doc_id_to_idx = {}
    
    for item in dataset:
        context = item["context"]
        titles = context["title"]
        sentences_list = context["sentences"]
        
        for title, sentences in zip(titles, sentences_list):
            if title not in doc_id_to_idx:
                # Combine title and sentences into document
                doc_text = f"{title}\n" + " ".join(sentences)
                doc_id_to_idx[title] = len(documents)
                documents.append(doc_text)
    
    # Build questions list
    questions = []
    for item in dataset:
        # Get supporting fact doc indices
        sf_titles = item["supporting_facts"]["title"]
        sf_sent_ids = item["supporting_facts"]["sent_id"]
        
        support_doc_indices = []
        for title in set(sf_titles):
            if title in doc_id_to_idx:
                support_doc_indices.append(doc_id_to_idx[title])
        
        questions.append({
            "id": item["id"],
            "question": item["question"],
            "answer": item["answer"],
            "type": item["type"],
            "level": item["level"],
            "supporting_facts": list(zip(sf_titles, sf_sent_ids)),
            "support_doc_indices": support_doc_indices,
        })
    
    logger.info(f"Loaded {len(questions)} questions, {len(documents)} documents")
    
    return questions, documents, doc_id_to_idx


def load_documents(path: str, format: str = "jsonl") -> List[str]:
    """Load documents from file.
    
    Args:
        path: Path to document file.
        format: File format ("jsonl", "txt", "json").
        
    Returns:
        List of document texts.
    """
    documents = []
    
    if format == "jsonl":
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                text = item.get("text") or item.get("content") or item.get("passage")
                if text:
                    documents.append(text)
    
    elif format == "txt":
        with open(path, "r") as f:
            # Each line is a document
            documents = [line.strip() for line in f if line.strip()]
    
    elif format == "json":
        with open(path, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        documents.append(item)
                    elif isinstance(item, dict):
                        text = item.get("text") or item.get("content")
                        if text:
                            documents.append(text)
    
    logger.info(f"Loaded {len(documents)} documents from {path}")
    return documents


def create_synthetic_pkm(
    n_docs: int = 1000,
    topics: Optional[List[str]] = None
) -> Tuple[List[str], List[Dict]]:
    """Create synthetic PKM dataset for testing.
    
    Args:
        n_docs: Number of documents to generate.
        topics: List of topics to include.
        
    Returns:
        Tuple of (documents, queries).
    """
    import random
    
    if topics is None:
        topics = [
            "machine learning", "deep learning", "natural language processing",
            "computer vision", "reinforcement learning", "knowledge graphs",
            "retrieval augmented generation", "large language models",
            "neural networks", "transformers", "attention mechanisms",
            "embedding models", "semantic search", "question answering"
        ]
    
    templates = [
        "This note discusses {topic} and its applications in {related}.",
        "Research on {topic}: key findings about {related}.",
        "Meeting notes: discussed {topic} with focus on {related}.",
        "Ideas for {topic} project involving {related}.",
        "{topic} tutorial: understanding {related} concepts.",
    ]
    
    documents = []
    for i in range(n_docs):
        topic = random.choice(topics)
        related = random.choice([t for t in topics if t != topic])
        template = random.choice(templates)
        doc = template.format(topic=topic, related=related)
        documents.append(doc)
    
    # Create some queries
    queries = [
        {"question": f"What are the key concepts in {t}?", "topic": t}
        for t in topics[:5]
    ]
    
    return documents, queries
