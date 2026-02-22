#!/usr/bin/env python3
"""
Knowledge Graph Builder for Arthur (OpenClaw Agent)
Extracts entities and relationships from memory files, emails, and calendar.
Outputs: knowledge_graph.json + Obsidian notes
"""

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# Paths — detect environment
if os.path.exists("/home/clawdbot/clawd/MEMORY.md"):
    WORKSPACE = "/home/clawdbot/clawd"
    OBSIDIAN_BASE = "/home/clawdbot/obsidian-vault"
else:
    WORKSPACE = "/workspace"
    OBSIDIAN_BASE = "/workspace/obsidian-vault"

MEMORY_DIR = os.path.join(WORKSPACE, "memory")
MEMORY_FILE = os.path.join(WORKSPACE, "MEMORY.md")
KG_FILE = os.path.join(WORKSPACE, "tools", "knowledge_graph.json")
OBSIDIAN_KG = os.path.join(OBSIDIAN_BASE, "KnowledgeGraph")

BKK = timezone(timedelta(hours=7))

# ─── Known Entities (seeds) ───

KNOWN_PEOPLE = {
    "Anirach": {"full_name": "Anirach Mingkhwan", "org": "KMUTNB", "role": "lecturer"},
    "Naphatsara": {"full_name": "Naphatsara", "org": "KMUTNB", "role": "collaborator"},
    "DuckMan": {"full_name": "DuckMan", "org": None, "role": "collaborator"},
}

KNOWN_ORGS = {
    "KMUTNB": "King Mongkut's University of Technology North Bangkok",
    "FITM": "Faculty of Information Technology and Digital Innovation, KMUTNB",
    "Imperial College": "Imperial College London",
    "Springer": "Springer Publishing",
    "IEEE": "IEEE",
    "ACM": "ACM",
    "OpenClaw": "OpenClaw Platform",
    "Anthropic": "Anthropic",
    "Google": "Google",
    "OpenAI": "OpenAI",
    "Stanford": "Stanford University",
}

KNOWN_PROJECTS = {
    "RAG Second Brain": {"status": "active", "target": "Information Fusion", "github": "https://github.com/Anirach/rag-second-brain"},
    "Three Old Men": {"status": "complete", "genre": "literary fiction", "github": "https://github.com/Anirach/three-old-men"},
    "NCD-CIE": {"status": "active", "target": "AIiH 2026", "github": "https://github.com/Anirach/ncd-cie"},
    "ChartSense AI": {"status": "active", "target": "Thai hospitals", "github": "https://github.com/Anirach/chartsense-ai"},
    "Bookshelf App": {"status": "in-progress", "stack": "Next.js + Prisma"},
    "Helpdesk Booking": {"status": "existing", "github": "https://github.com/Anirach/helpdesk-booking"},
}

TOPIC_KEYWORDS = [
    "RAG", "knowledge graph", "ontology", "causal inference", "NCD",
    "machine learning", "deep learning", "AI", "LLM", "GPT",
    "education technology", "longevity", "biomedical", "healthcare",
    "data warehouse", "second brain", "clinical decision support",
    "ICD-10", "GraphRAG", "translation", "Thai NLP",
    "Docker", "Next.js", "FastAPI", "PostgreSQL", "Neo4j",
    "arXiv", "peer review", "Obsidian", "ChromaDB",
    "agent architecture", "multi-agent", "prompt engineering",
]


class KnowledgeGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = []
        self._edge_set = set()

    def add_node(self, node_id, node_type, name, **attrs):
        clean_attrs = {k: v for k, v in attrs.items() if v is not None}
        if node_id not in self.nodes:
            self.nodes[node_id] = {"id": node_id, "type": node_type, "name": name, "attributes": clean_attrs}
        else:
            self.nodes[node_id]["attributes"].update(clean_attrs)

    def add_edge(self, source, target, relation, **attrs):
        key = (source, target, relation)
        if key not in self._edge_set and source in self.nodes and target in self.nodes:
            self._edge_set.add(key)
            edge = {"source": source, "target": target, "relation": relation}
            edge.update({k: v for k, v in attrs.items() if v is not None})
            self.edges.append(edge)

    def to_dict(self):
        return {
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
            "metadata": {
                "last_updated": datetime.now(BKK).isoformat(),
                "node_count": len(self.nodes),
                "edge_count": len(self.edges),
            },
        }

    def save(self, path=None):
        path = path or KG_FILE
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"[KG] Saved {len(self.nodes)} nodes, {len(self.edges)} edges → {path}")


def slugify(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def read_file(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except (FileNotFoundError, PermissionError):
        return ""


def read_memory_files():
    texts = {}
    content = read_file(MEMORY_FILE)
    if content:
        texts["MEMORY.md"] = content
    if os.path.isdir(MEMORY_DIR):
        for fn in sorted(os.listdir(MEMORY_DIR)):
            if fn.endswith(".md"):
                c = read_file(os.path.join(MEMORY_DIR, fn))
                if c:
                    texts[fn] = c
    return texts


def run_command(cmd, timeout=90):
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        print(f"[KG] Command timed out after {timeout}s: {cmd[:80]}", file=sys.stderr)
        return ""
    except Exception as e:
        print(f"[KG] Command error: {e}", file=sys.stderr)
        return ""


def fetch_emails():
    env = "GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT=anirach.m@fitm.kmutnb.ac.th"
    return run_command(f"{env} gog gmail search 'newer_than:30d' --max 50 --plain", 120)


def fetch_calendar():
    env = "GOG_KEYRING_PASSWORD=openclaw GOG_ACCOUNT=anirach.m@fitm.kmutnb.ac.th"
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    future = (datetime.now(timezone.utc) + timedelta(days=30)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return run_command(f"{env} gog calendar events primary --from {now} --to {future} --max 30 --plain", 120)


# ─── Extractors ───

def extract_people(text, kg):
    for name, info in KNOWN_PEOPLE.items():
        if name.lower() in text.lower():
            kg.add_node(f"person:{slugify(name)}", "Person", info["full_name"],
                        org=info.get("org"), role=info.get("role"))
    # Titled names
    for m in re.finditer(r"(?:Dr\.|Prof\.|Assoc\.?\s*Prof\.?|Asst\.?\s*Prof\.?)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})", text):
        name = m.group(1).strip()
        if len(name) > 2:
            kg.add_node(f"person:{slugify(name)}", "Person", name)
    # Email senders
    for m in re.finditer(r"From:\s*([^<\n]+?)(?:\s*<|\n|$)", text):
        name = m.group(1).strip().strip('"')
        if name and len(name) > 2 and "@" not in name and "github" not in name.lower() and "/" not in name:
            kg.add_node(f"person:{slugify(name)}", "Person", name)


def extract_orgs(text, kg):
    for name, full in KNOWN_ORGS.items():
        if name in text:
            kg.add_node(f"org:{slugify(name)}", "Organization", full)


def extract_projects(text, kg):
    text_norm = text.lower().replace(" ", "").replace("-", "")
    for name, info in KNOWN_PROJECTS.items():
        if name.lower().replace(" ", "").replace("-", "") in text_norm:
            kg.add_node(f"project:{slugify(name)}", "Project", name, **info)
    # GitHub repos
    for m in re.finditer(r"github\.com/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", text):
        repo = m.group(2)
        nid = f"project:{slugify(repo)}"
        if nid not in kg.nodes:
            kg.add_node(nid, "Project", repo, github=f"https://github.com/{m.group(1)}/{repo}")


def extract_topics(text, kg):
    tl = text.lower()
    for topic in TOPIC_KEYWORDS:
        if topic.lower() in tl:
            kg.add_node(f"topic:{slugify(topic)}", "Topic", topic)


def extract_events(text, kg):
    # Known events
    if "AIiH" in text or "aiih" in text.lower():
        kg.add_node("event:aiih-2026", "Event", "AIiH 2026",
                     venue="Imperial College London", date="2026-08-26/28", type="conference")
    # Calendar event patterns
    for m in re.finditer(r"(?:Summary|Event|Title):\s*(.+?)(?:\n|$)", text):
        ename = m.group(1).strip()
        if len(ename) > 3:
            kg.add_node(f"event:{slugify(ename)}", "Event", ename)


def extract_documents(text, kg):
    # Google Drive docs
    for m in re.finditer(r"\[([^\]]+)\]\(https://docs\.google\.com/[^)]+\)", text):
        name = m.group(1).strip()
        if name and name != "Link" and len(name) > 2:
            kg.add_node(f"doc:{slugify(name[:50])}", "Document", name)
    # Quoted paper/document titles (filter out conversational text)
    skip_starts = ("yes", "no", "can", "add", "what", "how", "why", "do", "is", "are", "the ")
    for m in re.finditer(r'"([A-Z][^"]{15,100})"', text):
        title = m.group(1).strip()
        if not any(title.lower().startswith(s) for s in skip_starts):
            kg.add_node(f"doc:{slugify(title[:50])}", "Document", title)


def extract_relationships(text, kg):
    text_lower = text.lower()
    anirach = "person:anirach"

    if anirach in kg.nodes:
        # Anirach ↔ Projects
        for nid, node in kg.nodes.items():
            if node["type"] == "Project":
                kg.add_edge(anirach, nid, "works_on")
            if node["type"] == "Organization" and "kmutnb" in nid:
                kg.add_edge(anirach, nid, "affiliated_with")

    # Collaborator ↔ Project hints
    collab_map = {
        "person:naphatsara": ["data-warehouse"],
        "person:duckman": ["chartsense-ai"],
    }
    for pid, hints in collab_map.items():
        if pid in kg.nodes:
            for hint in hints:
                for nid in kg.nodes:
                    if kg.nodes[nid]["type"] == "Project" and hint in nid:
                        kg.add_edge(pid, nid, "collaborates_on")

    # Project ↔ Topic
    proj_topics = {
        "project:rag-second-brain": ["rag", "knowledge-graph", "ontology", "llm", "second-brain", "multi-agent"],
        "project:ncd-cie": ["ncd", "causal-inference", "knowledge-graph", "healthcare", "icd-10"],
        "project:chartsense-ai": ["clinical-decision-support", "graphrag", "healthcare", "icd-10", "next-js", "fastapi"],
        "project:three-old-men": ["translation", "thai-nlp"],
        "project:bookshelf-app": ["next-js", "postgresql"],
    }
    for pid, tslugs in proj_topics.items():
        if pid in kg.nodes:
            for ts in tslugs:
                tid = f"topic:{ts}"
                if tid in kg.nodes:
                    kg.add_edge(pid, tid, "related_to")

    # Project ↔ Event
    if "project:ncd-cie" in kg.nodes and "event:aiih-2026" in kg.nodes:
        kg.add_edge("project:ncd-cie", "event:aiih-2026", "submitted_to")

    # Project ↔ Org
    for pid in kg.nodes:
        if kg.nodes[pid]["type"] == "Project" and "org:kmutnb" in kg.nodes:
            kg.add_edge(pid, "org:kmutnb", "developed_at")


def generate_obsidian_notes(kg):
    type_dirs = {
        "Person": "People", "Project": "Projects", "Topic": "Topics",
        "Organization": "Organizations", "Event": "Events", "Document": "Documents",
    }
    for d in type_dirs.values():
        os.makedirs(os.path.join(OBSIDIAN_KG, d), exist_ok=True)

    for nid, node in kg.nodes.items():
        subdir = type_dirs.get(node["type"], node["type"])
        safe_name = re.sub(r'[/\\:*?"<>|]', '_', node["name"])
        safe_name = safe_name[:80]  # cap filename length
        filepath = os.path.join(OBSIDIAN_KG, subdir, f"{safe_name}.md")

        lines = ["---"]
        lines.append(f'type: "{node["type"]}"')
        lines.append(f'id: "{nid}"')
        tags = [node["type"].lower()]
        if node["attributes"].get("status"):
            tags.append(node["attributes"]["status"])
        lines.append(f"tags: [{', '.join(tags)}]")
        lines.append(f'date: "{datetime.now(BKK).strftime("%Y-%m-%d")}"')
        for k, v in node["attributes"].items():
            if v is not None:
                lines.append(f'{k}: "{v}"')
        lines.append("---")
        lines.append(f"\n# {node['name']}\n")

        # Connections via wikilinks
        connections = []
        for edge in kg.edges:
            other_id, rel, direction = None, edge["relation"], ""
            if edge["source"] == nid:
                other_id, direction = edge["target"], "→"
            elif edge["target"] == nid:
                other_id, direction = edge["source"], "←"
            if other_id and other_id in kg.nodes:
                other = kg.nodes[other_id]
                safe_other = re.sub(r'[/\\:*?"<>|]', '_', other["name"])
                other_dir = type_dirs.get(other["type"], other["type"])
                connections.append(f"- {direction} **{rel}**: [[{other_dir}/{safe_other}|{other['name']}]]")

        if connections:
            lines.append("## Connections\n")
            lines.extend(connections)
            lines.append("")

        if node["attributes"]:
            lines.append("## Attributes\n")
            for k, v in node["attributes"].items():
                if v is not None:
                    lines.append(f"- **{k}:** {v}")
            lines.append("")

        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    print(f"[KG] Generated Obsidian notes → {OBSIDIAN_KG}")


def build():
    kg = KnowledgeGraph()

    print("[KG] Reading memory files...")
    memory_texts = read_memory_files()
    all_text = "\n".join(memory_texts.values())
    print(f"[KG] Loaded {len(memory_texts)} files ({len(all_text)} chars)")

    print("[KG] Fetching emails...")
    email_text = fetch_emails()
    if email_text:
        all_text += "\n" + email_text
        print(f"[KG] Got {len(email_text)} chars of email data")
    else:
        print("[KG] No email data available")

    print("[KG] Fetching calendar...")
    cal_text = fetch_calendar()
    if cal_text:
        all_text += "\n" + cal_text
        print(f"[KG] Got {len(cal_text)} chars of calendar data")
    else:
        print("[KG] No calendar data available")

    print("[KG] Extracting entities...")
    extract_people(all_text, kg)
    extract_orgs(all_text, kg)
    extract_projects(all_text, kg)
    extract_topics(all_text, kg)
    extract_events(all_text, kg)
    extract_documents(all_text, kg)

    print("[KG] Extracting relationships...")
    extract_relationships(all_text, kg)

    kg.save()

    print("[KG] Generating Obsidian notes...")
    generate_obsidian_notes(kg)

    return kg


if __name__ == "__main__":
    kg = build()
    data = kg.to_dict()
    print(f"\n{'='*40}")
    print(f"Knowledge Graph Summary")
    print(f"{'='*40}")
    print(f"Nodes: {data['metadata']['node_count']}")
    print(f"Edges: {data['metadata']['edge_count']}")
    print(f"\nBy type:")
    tc = defaultdict(int)
    for n in data["nodes"]:
        tc[n["type"]] += 1
    for t, c in sorted(tc.items()):
        print(f"  {t}: {c}")
    print(f"\nRelationships:")
    rc = defaultdict(int)
    for e in data["edges"]:
        rc[e["relation"]] += 1
    for r, c in sorted(rc.items()):
        print(f"  {r}: {c}")
