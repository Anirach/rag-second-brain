#!/usr/bin/env python3
"""
Knowledge Graph Query Interface
Usage:
  python3 kg_query.py search "term"
  python3 kg_query.py connections "person:anirach"
  python3 kg_query.py projects [--active]
  python3 kg_query.py timeline [--days N]
  python3 kg_query.py context "topic phrase"
  python3 kg_query.py stats
"""

import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# Detect path
if os.path.exists("/home/clawdbot/clawd/tools/knowledge_graph.json"):
    KG_FILE = "/home/clawdbot/clawd/tools/knowledge_graph.json"
else:
    KG_FILE = "/workspace/tools/knowledge_graph.json"


def load_kg():
    try:
        with open(KG_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: {KG_FILE} not found. Run kg_builder.py first.")
        sys.exit(1)


def search(kg, term):
    """Search nodes and edges matching term."""
    term_lower = term.lower()
    results = {"nodes": [], "edges": []}

    for node in kg["nodes"]:
        searchable = json.dumps(node, ensure_ascii=False).lower()
        if term_lower in searchable:
            results["nodes"].append(node)
            nid = node["id"]
            for edge in kg["edges"]:
                if edge["source"] == nid or edge["target"] == nid:
                    if edge not in results["edges"]:
                        results["edges"].append(edge)

    if not results["nodes"]:
        print(f"No results for '{term}'")
        return

    print(f"=== Search: '{term}' ===\n")
    print(f"Nodes ({len(results['nodes'])}):")
    for n in results["nodes"]:
        attrs = ", ".join(f"{k}={v}" for k, v in n["attributes"].items() if v)
        print(f"  [{n['type']}] {n['name']} ({n['id']}){f' — {attrs}' if attrs else ''}")

    if results["edges"]:
        print(f"\nEdges ({len(results['edges'])}):")
        for e in results["edges"]:
            extra = {k: v for k, v in e.items() if k not in ("source", "target", "relation")}
            extra_str = f" {extra}" if extra else ""
            print(f"  {e['source']} —[{e['relation']}]→ {e['target']}{extra_str}")


def connections(kg, node_id):
    """Show all connections for a node."""
    node = next((n for n in kg["nodes"] if n["id"] == node_id), None)
    if not node:
        # Try partial match
        matches = [n for n in kg["nodes"] if node_id in n["id"]]
        if len(matches) == 1:
            node = matches[0]
            node_id = node["id"]
        elif matches:
            print(f"Multiple matches for '{node_id}':")
            for m in matches:
                print(f"  {m['id']} — {m['name']}")
            return
        else:
            print(f"Node '{node_id}' not found")
            return

    node_map = {n["id"]: n for n in kg["nodes"]}
    print(f"=== Connections: {node['name']} ({node['type']}) ===\n")

    outgoing, incoming = [], []
    for e in kg["edges"]:
        if e["source"] == node_id:
            target = node_map.get(e["target"], {"name": e["target"], "type": "?"})
            outgoing.append((e["relation"], target))
        elif e["target"] == node_id:
            source = node_map.get(e["source"], {"name": e["source"], "type": "?"})
            incoming.append((e["relation"], source))

    if outgoing:
        print("Outgoing:")
        for rel, t in outgoing:
            print(f"  → [{rel}] {t['name']} ({t['type']})")
    if incoming:
        print("Incoming:")
        for rel, s in incoming:
            print(f"  ← [{rel}] {s['name']} ({s['type']})")
    if not outgoing and not incoming:
        print("  (no connections)")


def projects(kg, active_only=False):
    """List projects with their people."""
    node_map = {n["id"]: n for n in kg["nodes"]}
    proj_nodes = [n for n in kg["nodes"] if n["type"] == "Project"]

    if active_only:
        proj_nodes = [n for n in proj_nodes if n["attributes"].get("status") in ("active", "in-progress")]

    print(f"=== Projects{' (active)' if active_only else ''} ===\n")
    for p in proj_nodes:
        status = p["attributes"].get("status", "unknown")
        print(f"📁 {p['name']} [{status}]")
        for k, v in p["attributes"].items():
            if k != "status" and v:
                print(f"   {k}: {v}")
        # Find connected people
        people = []
        for e in kg["edges"]:
            if e["target"] == p["id"] and e["source"].startswith("person:"):
                person = node_map.get(e["source"])
                if person:
                    people.append(f"{person['name']} ({e['relation']})")
            elif e["source"] == p["id"] and e["target"].startswith("person:"):
                person = node_map.get(e["target"])
                if person:
                    people.append(f"{person['name']} ({e['relation']})")
        if people:
            print(f"   People: {', '.join(people)}")
        # Topics
        topics = []
        for e in kg["edges"]:
            if e["source"] == p["id"] and e["relation"] == "related_to":
                t = node_map.get(e["target"])
                if t:
                    topics.append(t["name"])
        if topics:
            print(f"   Topics: {', '.join(topics)}")
        print()


def timeline(kg, days=7):
    """Show recent activity from metadata."""
    print(f"=== Timeline (last {days} days) ===\n")
    print(f"Graph last updated: {kg['metadata'].get('last_updated', 'unknown')}")
    print(f"Total: {kg['metadata']['node_count']} nodes, {kg['metadata']['edge_count']} edges")
    print()

    # Group by type
    tc = defaultdict(list)
    for n in kg["nodes"]:
        tc[n["type"]].append(n)
    for t in sorted(tc.keys()):
        print(f"{t} ({len(tc[t])}):")
        for n in tc[t][:10]:
            print(f"  - {n['name']}")
        if len(tc[t]) > 10:
            print(f"  ... and {len(tc[t])-10} more")
        print()


def context(kg, phrase):
    """Pull all relevant context for a phrase."""
    term_lower = phrase.lower()
    node_map = {n["id"]: n for n in kg["nodes"]}

    # Find matching nodes
    matched = set()
    for n in kg["nodes"]:
        if term_lower in json.dumps(n, ensure_ascii=False).lower():
            matched.add(n["id"])

    # Expand to connected nodes (1 hop)
    expanded = set(matched)
    for e in kg["edges"]:
        if e["source"] in matched:
            expanded.add(e["target"])
        if e["target"] in matched:
            expanded.add(e["source"])

    if not expanded:
        print(f"No context found for '{phrase}'")
        return

    print(f"=== Context: '{phrase}' ===\n")
    print(f"Direct matches: {len(matched)}, Total (1-hop): {len(expanded)}\n")

    # Group by type
    by_type = defaultdict(list)
    for nid in expanded:
        if nid in node_map:
            by_type[node_map[nid]["type"]].append(node_map[nid])

    for t in ["Person", "Project", "Organization", "Topic", "Event", "Document"]:
        if t in by_type:
            print(f"{t}:")
            for n in by_type[t]:
                marker = "★" if n["id"] in matched else "·"
                print(f"  {marker} {n['name']}")
            print()

    # Relevant edges
    rel_edges = [e for e in kg["edges"] if e["source"] in expanded and e["target"] in expanded]
    if rel_edges:
        print("Relationships:")
        for e in rel_edges:
            s = node_map.get(e["source"], {"name": e["source"]})
            t = node_map.get(e["target"], {"name": e["target"]})
            print(f"  {s['name']} —[{e['relation']}]→ {t['name']}")


def stats(kg):
    """Show graph statistics."""
    print("=== Knowledge Graph Statistics ===\n")
    print(f"Last updated: {kg['metadata'].get('last_updated', 'unknown')}")
    print(f"Nodes: {kg['metadata']['node_count']}")
    print(f"Edges: {kg['metadata']['edge_count']}")

    tc = defaultdict(int)
    for n in kg["nodes"]:
        tc[n["type"]] += 1
    print("\nNodes by type:")
    for t, c in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {t}: {c}")

    rc = defaultdict(int)
    for e in kg["edges"]:
        rc[e["relation"]] += 1
    print("\nEdges by relation:")
    for r, c in sorted(rc.items(), key=lambda x: -x[1]):
        print(f"  {r}: {c}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    kg = load_kg()
    cmd = sys.argv[1]

    if cmd == "search" and len(sys.argv) >= 3:
        search(kg, " ".join(sys.argv[2:]))
    elif cmd == "connections" and len(sys.argv) >= 3:
        connections(kg, sys.argv[2])
    elif cmd == "projects":
        active_only = "--active" in sys.argv
        projects(kg, active_only)
    elif cmd == "timeline":
        days = 7
        if "--days" in sys.argv:
            idx = sys.argv.index("--days")
            if idx + 1 < len(sys.argv):
                days = int(sys.argv[idx + 1])
        timeline(kg, days)
    elif cmd == "context" and len(sys.argv) >= 3:
        context(kg, " ".join(sys.argv[2:]))
    elif cmd == "stats":
        stats(kg)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
