#!/usr/bin/env python3
"""
Obsidian Semantic Search with ChromaDB

Index and search Obsidian vault notes using embeddings.
Supports incremental updates, frontmatter parsing, and link extraction.

Usage:
    python obsidian_search.py index /path/to/vault
    python obsidian_search.py search "your query"
    python obsidian_search.py status
    python obsidian_search.py reindex /path/to/vault
"""

import os
import sys
import json
import hashlib
import re
from pathlib import Path
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.config import Settings

# Configuration
CHROMA_DIR = os.path.expanduser("~/.openclaw/chroma_obsidian")
COLLECTION_NAME = "obsidian_notes"
CHUNK_SIZE = 1000  # characters per chunk
CHUNK_OVERLAP = 200
STATE_FILE = os.path.join(CHROMA_DIR, "index_state.json")


def get_client():
    """Get ChromaDB client."""
    os.makedirs(CHROMA_DIR, exist_ok=True)
    return chromadb.PersistentClient(path=CHROMA_DIR)


def get_collection(client):
    """Get or create the Obsidian collection."""
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Obsidian vault semantic search"}
    )


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown."""
    frontmatter = {}
    body = content
    
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                import yaml
                frontmatter = yaml.safe_load(parts[1]) or {}
            except:
                pass
            body = parts[2].strip()
    
    return frontmatter, body


def extract_links(content: str) -> list[str]:
    """Extract [[wikilinks]] and [markdown](links)."""
    wikilinks = re.findall(r'\[\[([^\]|]+)(?:\|[^\]]+)?\]\]', content)
    mdlinks = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
    return wikilinks + [url for _, url in mdlinks if not url.startswith('http')]


def extract_tags(content: str) -> list[str]:
    """Extract #tags from content."""
    return re.findall(r'(?:^|\s)#([a-zA-Z][a-zA-Z0-9_/-]*)', content)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        
        # Try to break at paragraph or sentence
        if end < len(text):
            for sep in ['\n\n', '\n', '. ', ' ']:
                last_sep = chunk.rfind(sep)
                if last_sep > chunk_size // 2:
                    chunk = chunk[:last_sep + len(sep)]
                    end = start + len(chunk)
                    break
        
        if chunk.strip():
            chunks.append(chunk.strip())
        
        start = end - overlap
        if start >= len(text):
            break
    
    return chunks


def file_hash(filepath: Path) -> str:
    """Get hash of file for change detection."""
    stat = filepath.stat()
    return hashlib.md5(f"{filepath}:{stat.st_mtime}:{stat.st_size}".encode()).hexdigest()


def load_state() -> dict:
    """Load indexing state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"files": {}, "vault_path": None, "last_indexed": None}


def save_state(state: dict):
    """Save indexing state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def index_vault(vault_path: str, force: bool = False):
    """Index or update Obsidian vault."""
    vault = Path(vault_path).expanduser().resolve()
    
    if not vault.exists():
        print(f"❌ Vault not found: {vault}")
        return
    
    print(f"📚 Indexing vault: {vault}")
    
    client = get_client()
    collection = get_collection(client)
    state = load_state()
    
    if force:
        # Clear existing data
        try:
            client.delete_collection(COLLECTION_NAME)
        except:
            pass
        collection = get_collection(client)
        state = {"files": {}, "vault_path": str(vault), "last_indexed": None}
    
    # Find all markdown files
    md_files = list(vault.rglob("*.md"))
    
    # Filter out hidden folders and common excludes
    md_files = [f for f in md_files if not any(
        part.startswith('.') or part in ['node_modules', '.trash']
        for part in f.parts
    )]
    
    print(f"📄 Found {len(md_files)} markdown files")
    
    indexed = 0
    skipped = 0
    updated = 0
    
    for filepath in md_files:
        rel_path = str(filepath.relative_to(vault))
        current_hash = file_hash(filepath)
        
        # Check if file changed
        if not force and rel_path in state["files"]:
            if state["files"][rel_path] == current_hash:
                skipped += 1
                continue
            else:
                # File changed - delete old chunks
                try:
                    old_ids = collection.get(where={"source": rel_path})["ids"]
                    if old_ids:
                        collection.delete(ids=old_ids)
                except:
                    pass
                updated += 1
        else:
            indexed += 1
        
        # Read and parse file
        try:
            content = filepath.read_text(encoding="utf-8")
        except:
            print(f"  ⚠️ Could not read: {rel_path}")
            continue
        
        frontmatter, body = parse_frontmatter(content)
        tags = extract_tags(content)
        links = extract_links(content)
        
        # Chunk the content
        chunks = chunk_text(body)
        
        if not chunks:
            continue
        
        # Prepare metadata
        title = frontmatter.get("title", filepath.stem)
        
        # Add to collection
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::{i}"
            metadata = {
                "source": rel_path,
                "title": title,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "tags": ",".join(tags[:10]) if tags else "",
                "links": ",".join(links[:10]) if links else "",
                "folder": str(filepath.parent.relative_to(vault)),
            }
            
            # Add frontmatter fields
            for key in ["date", "created", "modified", "type", "status"]:
                if key in frontmatter:
                    val = frontmatter[key]
                    if isinstance(val, (str, int, float, bool)):
                        metadata[key] = str(val)
            
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                metadatas=[metadata]
            )
        
        state["files"][rel_path] = current_hash
    
    state["vault_path"] = str(vault)
    state["last_indexed"] = datetime.now().isoformat()
    save_state(state)
    
    total_chunks = collection.count()
    print(f"\n✅ Indexing complete!")
    print(f"   New files: {indexed}")
    print(f"   Updated: {updated}")
    print(f"   Unchanged: {skipped}")
    print(f"   Total chunks: {total_chunks}")


def search(query: str, n_results: int = 5, folder: Optional[str] = None, tags: Optional[str] = None):
    """Search the indexed vault."""
    client = get_client()
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except:
        print("❌ No index found. Run 'index' first.")
        return []
    
    # Build filter
    where = None
    if folder:
        where = {"folder": {"$eq": folder}}
    if tags:
        where = where or {}
        where["tags"] = {"$contains": tags}
    
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results["ids"][0]:
        print("No results found.")
        return []
    
    print(f"\n🔍 Results for: \"{query}\"\n")
    
    seen_sources = set()
    formatted = []
    
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        source = meta["source"]
        score = 1 - dist  # Convert distance to similarity
        
        # Dedupe by source (show best chunk per file)
        if source in seen_sources:
            continue
        seen_sources.add(source)
        
        print(f"{'─' * 60}")
        print(f"📄 {meta['title']}")
        print(f"   Source: {source}")
        print(f"   Score: {score:.3f}")
        if meta.get("tags"):
            print(f"   Tags: #{meta['tags'].replace(',', ' #')}")
        print()
        
        # Show preview
        preview = doc[:300] + "..." if len(doc) > 300 else doc
        print(f"   {preview}\n")
        
        formatted.append({
            "source": source,
            "title": meta["title"],
            "score": score,
            "preview": preview,
            "metadata": meta
        })
    
    return formatted


def status():
    """Show index status."""
    client = get_client()
    state = load_state()
    
    try:
        collection = client.get_collection(COLLECTION_NAME)
        count = collection.count()
    except:
        count = 0
    
    print("📊 Obsidian Search Index Status")
    print(f"   Vault: {state.get('vault_path', 'Not set')}")
    print(f"   Files indexed: {len(state.get('files', {}))}")
    print(f"   Total chunks: {count}")
    print(f"   Last indexed: {state.get('last_indexed', 'Never')}")
    print(f"   DB location: {CHROMA_DIR}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    
    if cmd == "index":
        if len(sys.argv) < 3:
            print("Usage: obsidian_search.py index /path/to/vault")
            sys.exit(1)
        index_vault(sys.argv[2])
    
    elif cmd == "reindex":
        if len(sys.argv) < 3:
            print("Usage: obsidian_search.py reindex /path/to/vault")
            sys.exit(1)
        index_vault(sys.argv[2], force=True)
    
    elif cmd == "search":
        if len(sys.argv) < 3:
            print("Usage: obsidian_search.py search \"query\"")
            sys.exit(1)
        query = " ".join(sys.argv[2:])
        search(query)
    
    elif cmd == "status":
        status()
    
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
