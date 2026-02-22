#!/usr/bin/env python3
"""
ChromaDB Helper for OpenClaw Memory Enhancement
Provides semantic search over memory files and documents.
"""

import chromadb
from chromadb.config import Settings
import os
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

# Default paths
CHROMA_PATH = os.path.expanduser("~/.openclaw/chroma_db")
MEMORY_PATH = os.path.expanduser("~/clawd/memory")

def get_client(persist_directory: str = CHROMA_PATH) -> chromadb.Client:
    """Get or create a persistent ChromaDB client."""
    os.makedirs(persist_directory, exist_ok=True)
    return chromadb.PersistentClient(path=persist_directory)

def get_or_create_collection(client: chromadb.Client, name: str = "memory"):
    """Get or create a collection."""
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"}
    )

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start = end - overlap
    return chunks

def hash_content(content: str) -> str:
    """Generate a hash for content deduplication."""
    return hashlib.md5(content.encode()).hexdigest()[:12]

def index_file(collection, filepath: str, source_type: str = "memory") -> int:
    """
    Index a file into ChromaDB.
    
    Returns number of chunks indexed.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    chunks = chunk_text(content)
    filename = os.path.basename(filepath)
    
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(chunks):
        chunk_id = f"{hash_content(filepath)}_{i}"
        ids.append(chunk_id)
        documents.append(chunk)
        metadatas.append({
            "source": filepath,
            "filename": filename,
            "type": source_type,
            "chunk_index": i
        })
    
    if ids:
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
    
    return len(ids)

def index_memory_directory(memory_path: str = MEMORY_PATH) -> Dict:
    """Index all markdown files in memory directory."""
    client = get_client()
    collection = get_or_create_collection(client)
    
    results = {"files": 0, "chunks": 0}
    
    for filepath in Path(memory_path).glob("**/*.md"):
        try:
            chunks = index_file(collection, str(filepath), "memory")
            results["files"] += 1
            results["chunks"] += chunks
            print(f"Indexed: {filepath.name} ({chunks} chunks)")
        except Exception as e:
            print(f"Error indexing {filepath}: {e}")
    
    return results

def search(query: str, n_results: int = 5, collection_name: str = "memory") -> List[Dict]:
    """
    Search for similar content.
    
    Returns list of results with content, source, and score.
    """
    client = get_client()
    collection = get_or_create_collection(client, collection_name)
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    
    formatted = []
    if results and results['documents']:
        for i, doc in enumerate(results['documents'][0]):
            metadata = results['metadatas'][0][i] if results['metadatas'] else {}
            distance = results['distances'][0][i] if results['distances'] else None
            
            formatted.append({
                "content": doc,
                "source": metadata.get("source", "unknown"),
                "filename": metadata.get("filename", "unknown"),
                "score": 1 - distance if distance else None  # Convert distance to similarity
            })
    
    return formatted

def add_document(content: str, metadata: Dict, collection_name: str = "memory") -> str:
    """Add a single document to the collection."""
    client = get_client()
    collection = get_or_create_collection(client, collection_name)
    
    doc_id = hash_content(content)
    
    collection.upsert(
        ids=[doc_id],
        documents=[content],
        metadatas=[metadata]
    )
    
    return doc_id

# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("ChromaDB Helper")
        print("Usage:")
        print("  python chroma_helper.py index     - Index memory files")
        print("  python chroma_helper.py search <query>  - Search")
        print("  python chroma_helper.py status    - Show collection status")
        sys.exit(0)
    
    command = sys.argv[1]
    
    if command == "index":
        print("Indexing memory files...")
        results = index_memory_directory()
        print(f"\nIndexed {results['files']} files, {results['chunks']} chunks total.")
    
    elif command == "search" and len(sys.argv) > 2:
        query = " ".join(sys.argv[2:])
        print(f"Searching for: {query}\n")
        results = search(query, n_results=5)
        
        if not results:
            print("No results found.")
        else:
            for i, r in enumerate(results, 1):
                print(f"{i}. [{r['filename']}] (score: {r['score']:.3f})")
                print(f"   {r['content'][:200]}...")
                print()
    
    elif command == "status":
        client = get_client()
        collection = get_or_create_collection(client)
        print(f"Collection: {collection.name}")
        print(f"Documents: {collection.count()}")
    
    else:
        print("Unknown command. Use: index, search, or status")
