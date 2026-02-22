#!/usr/bin/env python3
"""
Knowledge Graph Query Tool - Obsidian Vault Edition
====================================================
Queries the Obsidian vault directly via frontmatter + wikilink parsing.
No external dependencies - pure Python stdlib.

Usage:
  python3 kg_query_vault.py decisions ["term"]         List decisions (optionally filtered)
  python3 kg_query_vault.py people ["term"]            List people and their roles
  python3 kg_query_vault.py projects [--status active] List projects (filter by status)
  python3 kg_query_vault.py topics ["term"]            Find notes related to a topic
  python3 kg_query_vault.py events ["term"]            List events
  python3 kg_query_vault.py timeline "2026-02"         What happened in a month/year
  python3 kg_query_vault.py links "Note Title"         Show all backlinks to a note
  python3 kg_query_vault.py search "term"              Full-text search across all notes
  python3 kg_query_vault.py stats                      Vault statistics

Flags:
  --json          Output as JSON
  --status VALUE  Filter by status field (for projects)
  --type VALUE    Filter by note type

Examples:
  python3 kg_query_vault.py decisions "RAG paper"
  python3 kg_query_vault.py people "collaborators"
  python3 kg_query_vault.py projects --status active
  python3 kg_query_vault.py topics "longevity"
  python3 kg_query_vault.py timeline "2026-02"
  python3 kg_query_vault.py links "RAG Second Brain"
"""

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# ─── Vault path detection ────────────────────────────────────────────────────
VAULT_PATHS = [
    "/home/clawdbot/obsidian-vault",
    "/workspace/obsidian-vault",
]
VAULT_ROOT = next((p for p in VAULT_PATHS if os.path.isdir(p)), None)

# Category → directory mappings (relative to vault root)
CATEGORY_DIRS = {
    "decisions":     ["KnowledgeGraph/Decisions"],
    "people":        ["KnowledgeGraph/People"],
    "projects":      ["KnowledgeGraph/Projects"],
    "topics":        ["KnowledgeGraph/Topics"],
    "events":        ["KnowledgeGraph/Events"],
    "actions":       ["KnowledgeGraph/Action-Items"],
    "documents":     ["KnowledgeGraph/Documents"],
    "organizations": ["KnowledgeGraph/Organizations"],
    "daily":         ["Daily"],
    "agents":        ["Agents"],
}


# ─── Parsing utilities ────────────────────────────────────────────────────────

def parse_frontmatter(text):
    """Extract YAML frontmatter dict and body from markdown text."""
    fm = {}
    body = text
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            yaml_block = text[3:end].strip()
            body = text[end + 4:].strip()
            for line in yaml_block.splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    # Handle list values like [tag1, tag2]
                    if val.startswith("[") and val.endswith("]"):
                        val = [v.strip().strip('"').strip("'")
                               for v in val[1:-1].split(",") if v.strip()]
                    fm[key] = val
    return fm, body


def extract_wikilinks(text):
    """Extract [[wikilink]] targets from markdown text."""
    raw = re.findall(r'\[\[([^\]]+)\]\]', text)
    results = []
    for link in raw:
        if "|" in link:
            link = link.split("|")[0]
        link = link.split("/")[-1]
        results.append(link.strip())
    return results


def load_all_notes(vault_root):
    """Load all markdown notes from vault, returning structured dicts."""
    notes = []
    for md_file in sorted(Path(vault_root).rglob("*.md")):
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = parse_frontmatter(text)
        rel_path = str(md_file.relative_to(vault_root))
        parts = rel_path.replace("\\", "/").split("/")
        if len(parts) > 2:
            category = parts[1]  # e.g. KnowledgeGraph/People/X.md -> People
        elif len(parts) > 1:
            category = parts[0]
        else:
            category = "Root"

        note = {
            "path": rel_path,
            "abs_path": str(md_file),
            "title": md_file.stem,
            "category": category,
            "frontmatter": fm,
            "body": body,
            "wikilinks": extract_wikilinks(text),
            "mtime": md_file.stat().st_mtime,
        }
        notes.append(note)
    return notes


def build_backlink_index(notes):
    """Build a map: note_title -> [titles of notes that link to it]."""
    index = defaultdict(list)
    for note in notes:
        for link in note["wikilinks"]:
            if note["title"] not in index[link]:
                index[link].append(note["title"])
    return dict(index)


# ─── Fuzzy / partial matching ─────────────────────────────────────────────────

def fuzzy_match(query, candidates, threshold=55):
    """Simple fuzzy match using token overlap (no external deps)."""
    q = query.lower()
    results = []
    for c in candidates:
        cl = c.lower()
        if q in cl or cl in q:
            results.append(c)
            continue
        q_tokens = set(re.findall(r'\w+', q))
        c_tokens = set(re.findall(r'\w+', cl))
        if q_tokens and c_tokens:
            overlap = len(q_tokens & c_tokens) / max(len(q_tokens), len(c_tokens)) * 100
            if overlap >= threshold:
                results.append(c)
    return results


def filter_notes(notes, term="", category_dirs=None):
    """Filter notes by category directories and optional search term."""
    if category_dirs:
        notes = [n for n in notes if any(
            n["path"].replace("\\", "/").startswith(d + "/") or
            n["path"].replace("\\", "/") == d
            for d in category_dirs
        )]
    if term:
        term_l = term.lower()
        filtered = []
        for n in notes:
            searchable = (
                n["title"].lower() + " "
                + " ".join(str(v).lower() for v in n["frontmatter"].values()
                           if isinstance(v, str)) + " "
                + n["body"].lower()
            )
            if term_l in searchable:
                filtered.append(n)
        if not filtered:
            titles = [n["title"] for n in notes]
            matched_titles = set(fuzzy_match(term, titles))
            filtered = [n for n in notes if n["title"] in matched_titles]
        return filtered
    return notes


# ─── Command implementations ──────────────────────────────────────────────────

def cmd_decisions(notes, backlinks, term, as_json):
    dirs = CATEGORY_DIRS.get("decisions", [])
    results = filter_notes(notes, term, dirs)
    if not results:
        results = [n for n in notes if
                   n["frontmatter"].get("type", "").lower() == "decision"
                   or "decision" in n["path"].lower()]
        if term:
            results = filter_notes(results, term)

    if as_json:
        print(json.dumps([{"title": n["title"], "path": n["path"],
            "frontmatter": n["frontmatter"], "wikilinks": n["wikilinks"]}
            for n in results], indent=2, ensure_ascii=False))
        return

    label = 'Decisions: "{}"'.format(term) if term else "Decisions"
    print("=== {} ({} found) ===\n".format(label, len(results)))
    if not results:
        print("  No decisions found.")
        return
    for n in results:
        fm = n["frontmatter"]
        print("[{}] {}".format(n["category"], n["title"]))
        for key in ("status", "date", "type"):
            if fm.get(key): print("   {}: {}".format(key, fm[key]))
        body_lines = [l for l in n["body"].splitlines() if l.strip() and not l.startswith("#")]
        if body_lines:
            print("   >> {}".format(" ".join(body_lines[:2])[:180]))
        bl = backlinks.get(n["title"], [])
        if bl: print("   linked by: {}".format(", ".join(bl[:4])))
        print()


def cmd_people(notes, backlinks, term, as_json):
    dirs = CATEGORY_DIRS.get("people", [])
    results = filter_notes(notes, term, dirs)

    if as_json:
        print(json.dumps([{
            "title": n["title"], "role": n["frontmatter"].get("role", ""),
            "org": n["frontmatter"].get("org", ""), "frontmatter": n["frontmatter"],
            "links_to": n["wikilinks"], "linked_by": backlinks.get(n["title"], [])}
            for n in results], indent=2, ensure_ascii=False))
        return

    label = 'People: "{}"'.format(term) if term else "People"
    print("=== {} ({} found) ===\n".format(label, len(results)))
    if not results:
        print("  No people found.")
        return
    for n in results:
        fm = n["frontmatter"]
        role = fm.get("role", "unknown role")
        org  = fm.get("org", "")
        print("Person: {}  --  {}{}".format(n["title"], role, " @ " + org if org else ""))
        if n["wikilinks"]:
            print("   Works on/with: {}".format(", ".join(n["wikilinks"][:8])))
        bl = backlinks.get(n["title"], [])
        if bl: print("   Referenced by: {}".format(", ".join(bl[:5])))
        print()


def cmd_projects(notes, backlinks, term, status_filter, as_json):
    dirs = CATEGORY_DIRS.get("projects", [])
    results = filter_notes(notes, term, dirs)
    if status_filter:
        results = [n for n in results
                   if n["frontmatter"].get("status", "").lower() == status_filter.lower()]

    if as_json:
        print(json.dumps([{
            "title": n["title"], "status": n["frontmatter"].get("status", ""),
            "github": n["frontmatter"].get("github", ""), "frontmatter": n["frontmatter"],
            "links_to": n["wikilinks"], "linked_by": backlinks.get(n["title"], [])}
            for n in results], indent=2, ensure_ascii=False))
        return

    label = "Projects"
    if status_filter: label += " [{}]".format(status_filter)
    if term: label += ': "{}"'.format(term)
    print("=== {} ({} found) ===\n".format(label, len(results)))
    if not results:
        print("  No projects found.")
        return
    status_icons = {"active": "[ACTIVE]", "complete": "[DONE]",
                    "paused": "[PAUSED]", "cancelled": "[CANCELLED]"}
    for n in results:
        fm = n["frontmatter"]
        status = fm.get("status", "unknown")
        icon = status_icons.get(status.lower(), "[{}]".format(status.upper()))
        print("{} {}".format(icon, n["title"]))
        for key in ("target", "github", "venue"):
            if fm.get(key): print("   {}: {}".format(key, fm[key]))
        if n["wikilinks"]:
            print("   Related: {}".format(", ".join(n["wikilinks"][:6])))
        bl = backlinks.get(n["title"], [])
        if bl: print("   Referenced by: {}".format(", ".join(bl[:5])))
        print()


def cmd_topics(notes, backlinks, term, as_json):
    dirs = CATEGORY_DIRS.get("topics", [])
    topic_notes = filter_notes(notes, term, dirs)
    related = []
    if term:
        term_l = term.lower()
        for n in notes:
            if n in topic_notes:
                continue
            link_text = " ".join(n["wikilinks"]).lower()
            if term_l in link_text or term_l in n["body"].lower() or term_l in n["title"].lower():
                related.append(n)

    if as_json:
        print(json.dumps({
            "topic_notes": [{"title": n["title"], "path": n["path"],
                "frontmatter": n["frontmatter"]} for n in topic_notes],
            "related_notes": [{"title": n["title"], "path": n["path"],
                "category": n["category"]} for n in related]},
            indent=2, ensure_ascii=False))
        return

    label = 'Topics: "{}"'.format(term) if term else "Topics"
    print("=== {} ===\n".format(label))
    if topic_notes:
        print("-- Topic Entries ({}) --".format(len(topic_notes)))
        for n in topic_notes:
            bl = backlinks.get(n["title"], [])
            print("  [TOPIC] {}".format(n["title"]))
            if bl:
                suffix = " ..." if len(bl) > 8 else ""
                print("   Mentioned in: {}{}".format(", ".join(bl[:8]), suffix))
            print()
    if related:
        print("-- Notes Mentioning '{}' ({}) --".format(term, len(related)))
        for n in related:
            print("   [{}] {}".format(n["category"], n["title"]))
        print()
    if not topic_notes and not related:
        print("  No notes found related to '{}'.".format(term))


def cmd_events(notes, backlinks, term, as_json):
    dirs = CATEGORY_DIRS.get("events", [])
    results = filter_notes(notes, term, dirs)

    if as_json:
        print(json.dumps([{
            "title": n["title"],
            "date": n["frontmatter"].get("event_date", n["frontmatter"].get("date", "")),
            "venue": n["frontmatter"].get("venue", ""), "frontmatter": n["frontmatter"]}
            for n in results], indent=2, ensure_ascii=False))
        return

    label = 'Events: "{}"'.format(term) if term else "Events"
    print("=== {} ({} found) ===\n".format(label, len(results)))
    for n in results:
        fm = n["frontmatter"]
        date_str = fm.get("event_date", fm.get("date", ""))
        venue = fm.get("venue", "")
        print("[EVENT] {}".format(n["title"]))
        if date_str: print("   Date: {}".format(date_str))
        if venue:    print("   Venue: {}".format(venue))
        bl = backlinks.get(n["title"], [])
        if bl:       print("   Referenced by: {}".format(", ".join(bl)))
        print()


def cmd_timeline(notes, backlinks, period, as_json):
    """Show notes whose date/mtime matches a period like '2026-02' or '2026'."""
    matched = []
    for n in notes:
        fm_date = n["frontmatter"].get("date", "")
        mtime_str = datetime.fromtimestamp(n["mtime"]).strftime("%Y-%m-%d")
        if fm_date.startswith(period) or mtime_str.startswith(period):
            matched.append(n)

    if as_json:
        print(json.dumps([{
            "title": n["title"], "category": n["category"],
            "date": n["frontmatter"].get("date", ""),
            "mtime": datetime.fromtimestamp(n["mtime"]).strftime("%Y-%m-%d %H:%M")}
            for n in matched], indent=2, ensure_ascii=False))
        return

    print("=== Timeline: {} ({} notes) ===\n".format(period, len(matched)))
    if not matched:
        print("  No notes dated in '{}'.".format(period))
        return
    by_cat = defaultdict(list)
    for n in matched:
        by_cat[n["category"]].append(n)
    for cat in sorted(by_cat.keys()):
        print("-- {} --".format(cat))
        for n in by_cat[cat]:
            date_str = n["frontmatter"].get(
                "date", datetime.fromtimestamp(n["mtime"]).strftime("%Y-%m-%d"))
            print("  [{}] {}".format(date_str, n["title"]))
        print()


def cmd_links(notes, backlinks, title_query, as_json):
    """Show all notes that link to a given note (backlinks)."""
    all_titles = [n["title"] for n in notes]
    matches = fuzzy_match(title_query, all_titles)
    if not matches:
        matches = [t for t in all_titles if title_query.lower() in t.lower()]
    if not matches:
        print("No note matching '{}' found.".format(title_query))
        print("Available: {}".format(", ".join(all_titles[:20])))
        return

    target_title = matches[0]
    bl = list(backlinks.get(target_title, []))

    # Also catch notes linking with partial path match
    for n in notes:
        for link in n["wikilinks"]:
            if (target_title.lower() in link.lower() or link.lower() in target_title.lower()):
                if n["title"] not in bl and n["title"] != target_title:
                    bl.append(n["title"])

    if as_json:
        target_note = next((n for n in notes if n["title"] == target_title), None)
        print(json.dumps({
            "target": target_title,
            "backlinks": bl,
            "outgoing_links": target_note.get("wikilinks", []) if target_note else []},
            indent=2, ensure_ascii=False))
        return

    print("=== Links: '{}' ===\n".format(target_title))
    target_note = next((n for n in notes if n["title"] == target_title), None)
    if target_note:
        print("[NOTE] {}  [{}]".format(target_title, target_note["category"]))
        out = target_note["wikilinks"]
        if out:
            print("   -> Links OUT ({}): {}".format(len(out), ", ".join(out)))
        print()

    if bl:
        print("<- Backlinks ({} notes link to this):".format(len(bl)))
        for title in bl:
            linker = next((n for n in notes if n["title"] == title), None)
            cat = linker["category"] if linker else "?"
            print("   [{}] {}".format(cat, title))
    else:
        print("  No backlinks found (this note is an orphan).")
    print()


def cmd_search(notes, backlinks, term, as_json):
    """Full-text search across all notes."""
    term_l = term.lower()
    results = []
    for n in notes:
        score = 0
        title_l = n["title"].lower()
        body_l = n["body"].lower()
        fm_str = json.dumps(n["frontmatter"]).lower()
        if term_l == title_l:     score += 100
        elif term_l in title_l:   score += 50
        if term_l in fm_str:      score += 20
        count = body_l.count(term_l)
        score += min(count * 5, 30)
        if score > 0:
            results.append((score, n))

    results.sort(key=lambda x: -x[0])

    if as_json:
        print(json.dumps([{"score": s, "title": n["title"],
            "path": n["path"], "category": n["category"]}
            for s, n in results], indent=2, ensure_ascii=False))
        return

    print("=== Search: '{}' ({} results) ===\n".format(term, len(results)))
    for score, n in results[:20]:
        print("[{:3d}] [{}] {}".format(score, n["category"], n["title"]))
        for line in n["body"].splitlines():
            if term_l in line.lower() and line.strip():
                print("       >> {}".format(line.strip()[:120]))
                break
    if not results:
        print("  No results found.")


def cmd_stats(notes, backlinks, as_json):
    """Quick vault stats overview."""
    total = len(notes)
    by_cat = defaultdict(int)
    for n in notes: by_cat[n["category"]] += 1

    orphans = [n for n in notes
               if not n["wikilinks"] and not backlinks.get(n["title"])]
    missing_fm = [n for n in notes if not n["frontmatter"]]
    hub_scores = [(len(backlinks.get(n["title"], [])) + len(n["wikilinks"]), n)
                  for n in notes]
    hub_scores.sort(key=lambda x: -x[0])
    now = datetime.now().timestamp()
    stale = [n for n in notes if (now - n["mtime"]) > 30 * 24 * 3600]
    recent = [n for n in notes if (now - n["mtime"]) < 7 * 24 * 3600]

    if as_json:
        print(json.dumps({
            "total_notes": total, "by_category": dict(by_cat),
            "orphan_count": len(orphans), "orphans": [n["title"] for n in orphans],
            "missing_frontmatter": [n["title"] for n in missing_fm],
            "hub_nodes": [{"title": n["title"], "score": s} for s, n in hub_scores[:10]],
            "stale_notes": [n["title"] for n in stale],
            "recent_notes": [n["title"] for n in recent]},
            indent=2, ensure_ascii=False))
        return

    print("=" * 52)
    print("     Obsidian Vault Health Dashboard")
    print("=" * 52)
    print("\nTotal Notes:          {}".format(total))
    print("Total Wikilinks:      {}".format(sum(len(n["wikilinks"]) for n in notes)))
    print("Orphan Notes:         {}".format(len(orphans)))
    print("Missing Frontmatter:  {}".format(len(missing_fm)))

    print("\n-- Notes by Category --")
    for cat, count in sorted(by_cat.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 20)
        print("  {:<30} {:3d}  {}".format(cat, count, bar))

    print("\n-- Hub Nodes (Most Connected) --")
    for score, n in hub_scores[:10]:
        if score > 0:
            bl_cnt = len(backlinks.get(n["title"], []))
            out_cnt = len(n["wikilinks"])
            print("  {:<35} in:{} out:{} (total:{})".format(
                n["title"], bl_cnt, out_cnt, score))

    print("\n-- Recent Notes (last 7 days): {} --".format(len(recent)))
    for n in recent[:10]:
        dt = datetime.fromtimestamp(n["mtime"]).strftime("%Y-%m-%d")
        print("  [{}] {}".format(dt, n["title"]))

    if orphans:
        print("\n-- Orphan Notes ({}) --".format(len(orphans)))
        for n in orphans[:15]:
            print("  [ORPHAN] {}  [{}]".format(n["title"], n["category"]))
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if VAULT_ROOT is None:
        print("ERROR: Obsidian vault not found. Checked:")
        for p in VAULT_PATHS: print("  {}".format(p))
        sys.exit(1)

    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(0)

    as_json = "--json" in args
    args = [a for a in args if a != "--json"]

    status_filter = None
    if "--status" in args:
        idx = args.index("--status")
        if idx + 1 < len(args):
            status_filter = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    if "--type" in args:
        idx = args.index("--type")
        if idx + 1 < len(args):
            args = args[:idx] + args[idx + 2:]

    cmd = args[0].lower() if args else "help"
    term = " ".join(args[1:]) if len(args) > 1 else ""

    notes = load_all_notes(VAULT_ROOT)
    backlinks = build_backlink_index(notes)

    dispatch = {
        "decisions": lambda: cmd_decisions(notes, backlinks, term, as_json),
        "decision":  lambda: cmd_decisions(notes, backlinks, term, as_json),
        "people":    lambda: cmd_people(notes, backlinks, term, as_json),
        "person":    lambda: cmd_people(notes, backlinks, term, as_json),
        "projects":  lambda: cmd_projects(notes, backlinks, term, status_filter, as_json),
        "project":   lambda: cmd_projects(notes, backlinks, term, status_filter, as_json),
        "topics":    lambda: cmd_topics(notes, backlinks, term, as_json),
        "topic":     lambda: cmd_topics(notes, backlinks, term, as_json),
        "events":    lambda: cmd_events(notes, backlinks, term, as_json),
        "event":     lambda: cmd_events(notes, backlinks, term, as_json),
        "timeline":  lambda: cmd_timeline(notes, backlinks, term, as_json),
        "links":     lambda: cmd_links(notes, backlinks, term, as_json) if term else (
                        print("Usage: kg_query_vault.py links \"Note Title\"") or sys.exit(1)),
        "backlinks": lambda: cmd_links(notes, backlinks, term, as_json) if term else (
                        print("Usage: kg_query_vault.py backlinks \"Note Title\"") or sys.exit(1)),
        "search":    lambda: cmd_search(notes, backlinks, term, as_json) if term else (
                        print("Usage: kg_query_vault.py search \"term\"") or sys.exit(1)),
        "stats":     lambda: cmd_stats(notes, backlinks, as_json),
        "help":      lambda: print(__doc__),
        "--help":    lambda: print(__doc__),
        "-h":        lambda: print(__doc__),
    }

    if cmd in dispatch:
        dispatch[cmd]()
    else:
        print("Unknown command: {}".format(cmd))
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
