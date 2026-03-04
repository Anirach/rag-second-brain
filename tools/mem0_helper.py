#!/usr/bin/env python3
"""
Mem0 Relationship Memory Helper for Arthur's Memory Layer 5.
Stores decisions, preferences, and relationships as structured memories.

Usage:
    python3 tools/mem0_helper.py add "Anirach prefers dark theme for PPTX slides"
    python3 tools/mem0_helper.py search "RAG preferences"
    python3 tools/mem0_helper.py ingest                    # Ingest today's daily log
    python3 tools/mem0_helper.py ingest 2026-03-03         # Ingest specific date
    python3 tools/mem0_helper.py list                      # List recent memories
    python3 tools/mem0_helper.py stats                     # Show stats
"""

import os
import sys
import json
import argparse
from datetime import datetime, timedelta
from pathlib import Path

# Use OpenAI for Mem0 (it uses OpenAI by default)
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

MEM0_DIR = os.path.expanduser("~/clawd/tools/mem0_data")
MEMORY_PATH = os.path.expanduser("~/clawd/memory")
USER_ID = "anirach"
GOOGLE_API_KEY = "AIzaSyC1BLzV7El8nNV5hqeCdo4R32Cd2HydyNk"


def get_mem0():
    """Initialize Mem0 client with Gemini 2.5 Flash + local HF embeddings."""
    from mem0 import Memory
    
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    config = {
        "version": "v1.1",
        "vector_store": {
            "provider": "chroma",
            "config": {
                "collection_name": "arthur_memories",
                "path": MEM0_DIR,
            },
        },
        "llm": {
            "provider": "gemini",
            "config": {
                "model": "gemini-2.5-flash",
                "temperature": 0.1,
                "api_key": GOOGLE_API_KEY,
            },
        },
        "embedder": {
            "provider": "huggingface",
            "config": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
        },
    }
    
    os.makedirs(MEM0_DIR, exist_ok=True)
    return Memory.from_config(config)


def do_add(text, metadata=None):
    """Add a memory."""
    m = get_mem0()
    result = m.add(text, user_id=USER_ID, metadata=metadata or {})
    print(f"Added memory: {json.dumps(result, indent=2, default=str)}")
    return result


def do_search(query, limit=10):
    """Search memories."""
    m = get_mem0()
    results = m.search(query, user_id=USER_ID, limit=limit)
    
    if not results.get("results"):
        print("No matching memories found.")
        return
    
    print(f"Found {len(results['results'])} memories:\n")
    for i, mem in enumerate(results["results"], 1):
        score = mem.get("score", 0)
        memory_text = mem.get("memory", "")
        created = mem.get("created_at", "")
        print(f"  {i}. [{score:.3f}] {memory_text}")
        if created:
            print(f"     Created: {created}")
        print()


def do_list(limit=20):
    """List all memories."""
    m = get_mem0()
    results = m.get_all(user_id=USER_ID)
    
    memories = results.get("results", [])
    if not memories:
        print("No memories stored yet.")
        return
    
    print(f"Total memories: {len(memories)}\n")
    for i, mem in enumerate(memories[:limit], 1):
        memory_text = mem.get("memory", "")
        created = mem.get("created_at", "")
        print(f"  {i}. {memory_text}")
        if created:
            print(f"     Created: {created}")
        print()
    
    if len(memories) > limit:
        print(f"  ... and {len(memories) - limit} more")


def do_ingest(date_str=None):
    """Ingest a daily log file, extracting decisions and preferences."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    log_file = os.path.join(MEMORY_PATH, f"{date_str}.md")
    if not os.path.exists(log_file):
        print(f"No daily log found: {log_file}")
        return
    
    with open(log_file, "r", encoding="utf-8") as f:
        content = f.read()
    
    if not content.strip():
        print(f"Daily log is empty: {log_file}")
        return
    
    print(f"Ingesting daily log: {date_str} ({len(content)} chars)")
    
    m = get_mem0()
    
    # Feed the whole daily log to mem0 - it will extract relevant memories
    messages = [
        {
            "role": "user",
            "content": f"Daily log for {date_str}:\n\n{content}",
        }
    ]
    
    result = m.add(
        messages,
        user_id=USER_ID,
        metadata={"source": f"daily_log_{date_str}", "date": date_str},
    )
    
    added = result.get("results", [])
    new_count = sum(1 for r in added if r.get("event") == "ADD")
    updated_count = sum(1 for r in added if r.get("event") == "UPDATE")
    
    print(f"Done! New: {new_count}, Updated: {updated_count}")
    
    # Show what was extracted
    for r in added:
        if r.get("event") in ("ADD", "UPDATE"):
            print(f"  [{r['event']}] {r.get('memory', 'N/A')}")


def do_stats():
    """Show Mem0 stats."""
    m = get_mem0()
    results = m.get_all(user_id=USER_ID)
    memories = results.get("results", [])
    
    print("=== Mem0 Memory Stats ===")
    print(f"Total memories: {len(memories)}")
    
    if memories:
        dates = [m.get("created_at", "") for m in memories if m.get("created_at")]
        if dates:
            print(f"Oldest: {min(dates)}")
            print(f"Newest: {max(dates)}")
    
    # Storage size
    total_size = 0
    for root, dirs, files in os.walk(MEM0_DIR):
        for f in files:
            total_size += os.path.getsize(os.path.join(root, f))
    print(f"Storage: {total_size / 1024:.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="Mem0 Relationship Memory Helper")
    subparsers = parser.add_subparsers(dest="command")
    
    # Add command
    add_p = subparsers.add_parser("add", help="Add a memory")
    add_p.add_argument("text", help="Memory text")
    
    # Search command
    search_p = subparsers.add_parser("search", help="Search memories")
    search_p.add_argument("query", help="Search query")
    search_p.add_argument("--limit", type=int, default=10, help="Max results")
    
    # List command
    list_p = subparsers.add_parser("list", help="List memories")
    list_p.add_argument("--limit", type=int, default=20, help="Max results")
    
    # Ingest command
    ingest_p = subparsers.add_parser("ingest", help="Ingest daily log")
    ingest_p.add_argument("date", nargs="?", help="Date (YYYY-MM-DD), default today")
    
    # Stats command
    subparsers.add_parser("stats", help="Show stats")
    
    args = parser.parse_args()
    
    if args.command == "add":
        do_add(args.text)
    elif args.command == "search":
        do_search(args.query, limit=args.limit)
    elif args.command == "list":
        do_list(limit=args.limit)
    elif args.command == "ingest":
        do_ingest(args.date)
    elif args.command == "stats":
        do_stats()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
