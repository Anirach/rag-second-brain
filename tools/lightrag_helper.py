#!/usr/bin/env python3
"""
LightRAG Knowledge Graph Helper for Arthur's Memory Layer 4.
Indexes Obsidian vault into a knowledge graph for relationship queries.

Usage:
    python3 tools/lightrag_helper.py index          # Index new/changed files
    python3 tools/lightrag_helper.py index --full    # Full re-index
    python3 tools/lightrag_helper.py query "question" [--mode hybrid]
    python3 tools/lightrag_helper.py stats           # Show graph stats
"""

import asyncio
import os
import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

# Paths
LIGHTRAG_DIR = os.path.expanduser("~/clawd/tools/lightrag_data")
VAULT_PATH = os.path.expanduser("~/obsidian-vault")
STATE_FILE = os.path.join(LIGHTRAG_DIR, "index_state.json")
GOOGLE_API_KEY = "AIzaSyC1BLzV7El8nNV5hqeCdo4R32Cd2HydyNk"

# Directories to skip
SKIP_DIRS = {".obsidian", ".git", ".trash", "node_modules", ".stversions"}
# File extensions to index
INDEX_EXTENSIONS = {".md"}
# Max file size (skip huge files)
MAX_FILE_SIZE = 100_000  # 100KB


def load_state():
    """Load indexing state (file hashes)."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"indexed_files": {}, "last_run": None}


def save_state(state):
    """Save indexing state."""
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def file_hash(filepath):
    """Get MD5 hash of file contents."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def collect_files(vault_path, full=False, state=None):
    """Collect files that need indexing."""
    files_to_index = []
    current_hashes = {}

    for root, dirs, files in os.walk(vault_path):
        # Skip hidden/excluded dirs
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for fname in files:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in INDEX_EXTENSIONS:
                continue

            filepath = os.path.join(root, fname)
            rel_path = os.path.relpath(filepath, vault_path)

            # Skip large files
            try:
                if os.path.getsize(filepath) > MAX_FILE_SIZE:
                    continue
            except OSError:
                continue

            h = file_hash(filepath)
            current_hashes[rel_path] = h

            if full:
                files_to_index.append(filepath)
            elif state and state["indexed_files"].get(rel_path) != h:
                files_to_index.append(filepath)
            else:
                files_to_index.append(filepath) if not state else None

    return files_to_index, current_hashes


def read_file_with_metadata(filepath, vault_path):
    """Read file and prepend metadata header for better KG extraction."""
    rel_path = os.path.relpath(filepath, vault_path)
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
    except Exception:
        return None

    if not content.strip():
        return None

    # Add source metadata
    header = f"[Source: {rel_path}]\n"
    return header + content


async def init_rag():
    """Initialize LightRAG instance with Gemini."""
    from lightrag import LightRAG
    from lightrag.llm.gemini import gemini_model_complete
    from lightrag.utils import EmbeddingFunc
    from functools import partial
    import numpy as np

    os.makedirs(LIGHTRAG_DIR, exist_ok=True)
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY

    # Custom embedding function that wraps Gemini embedding API directly
    async def _gemini_embed(texts: list[str]) -> np.ndarray:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = await client.aio.models.embed_content(
            model="gemini-embedding-001",
            contents=texts,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT",
                output_dimensionality=768,  # smaller dim for efficiency
            ),
        )
        embeddings = np.array(
            [np.array(e.values, dtype=np.float32) for e in response.embeddings]
        )
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return embeddings / norms

    embedding_func = EmbeddingFunc(
        embedding_dim=768,
        func=_gemini_embed,
        max_token_size=2048,
    )

    # Use gemini_model_complete which pulls model from global config
    llm_func = gemini_model_complete

    rag = LightRAG(
        working_dir=LIGHTRAG_DIR,
        llm_model_func=llm_func,
        llm_model_name="gemini-2.5-flash",
        embedding_func=embedding_func,
        chunk_token_size=1200,
        chunk_overlap_token_size=100,
        max_parallel_insert=2,
        llm_model_max_async=2,
        embedding_func_max_async=4,
    )

    await rag.initialize_storages()
    return rag


async def do_index(full=False):
    """Index vault files into LightRAG."""
    state = load_state()
    files_to_index, current_hashes = collect_files(VAULT_PATH, full=full, state=state)

    if not files_to_index:
        print("No new or changed files to index.")
        return

    print(f"Indexing {len(files_to_index)} files...")
    rag = await init_rag()

    # Batch insert documents
    batch_size = 10
    indexed = 0
    errors = 0

    for i in range(0, len(files_to_index), batch_size):
        batch = files_to_index[i : i + batch_size]
        docs = []
        for fp in batch:
            content = read_file_with_metadata(fp, VAULT_PATH)
            if content:
                docs.append(content)

        if docs:
            try:
                await rag.ainsert(docs)
                indexed += len(docs)
                print(f"  Indexed {indexed}/{len(files_to_index)}...")
            except Exception as e:
                errors += len(docs)
                print(f"  Error indexing batch: {e}")

    # Update state
    state["indexed_files"] = current_hashes
    state["last_run"] = datetime.now().isoformat()
    state["total_files"] = len(current_hashes)
    state["last_indexed_count"] = indexed
    state["last_errors"] = errors
    save_state(state)

    print(f"\nDone! Indexed: {indexed}, Errors: {errors}, Total tracked: {len(current_hashes)}")


async def do_query(question, mode="hybrid"):
    """Query the knowledge graph."""
    from lightrag import QueryParam

    rag = await init_rag()

    valid_modes = ["naive", "local", "global", "hybrid"]
    if mode not in valid_modes:
        print(f"Invalid mode '{mode}'. Use: {valid_modes}")
        return

    try:
        result = await rag.aquery(question, param=QueryParam(mode=mode))
        print(result)
    except Exception as e:
        print(f"Query error: {e}")


async def do_stats():
    """Show knowledge graph statistics."""
    state = load_state()
    print("=== LightRAG Knowledge Graph Stats ===")
    print(f"Last run: {state.get('last_run', 'never')}")
    print(f"Total tracked files: {state.get('total_files', 0)}")
    print(f"Last indexed: {state.get('last_indexed_count', 0)}")
    print(f"Last errors: {state.get('last_errors', 0)}")

    # Check storage sizes
    for fname in os.listdir(LIGHTRAG_DIR):
        fpath = os.path.join(LIGHTRAG_DIR, fname)
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            if size > 1024:
                print(f"  {fname}: {size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="LightRAG Knowledge Graph Helper")
    subparsers = parser.add_subparsers(dest="command")

    # Index command
    idx_parser = subparsers.add_parser("index", help="Index vault files")
    idx_parser.add_argument("--full", action="store_true", help="Full re-index")

    # Query command
    q_parser = subparsers.add_parser("query", help="Query knowledge graph")
    q_parser.add_argument("question", help="Question to ask")
    q_parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["naive", "local", "global", "hybrid"],
        help="Query mode",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show graph stats")

    args = parser.parse_args()

    if args.command == "index":
        asyncio.run(do_index(full=args.full))
    elif args.command == "query":
        asyncio.run(do_query(args.question, mode=args.mode))
    elif args.command == "stats":
        asyncio.run(do_stats())
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
