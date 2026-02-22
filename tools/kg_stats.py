#!/usr/bin/env python3
"""
kg_stats.py - Obsidian Vault Health Dashboard
==============================================
Comprehensive vault health report: note counts, link graph, orphans,
hub nodes, staleness, frontmatter issues, and broken links.

Usage:
  python3 kg_stats.py [--json] [--full] [--category CATEGORY]

Flags:
  --json            Output raw JSON
  --full            Include full orphan/broken link lists
  --category CAT    Show stats for a specific category only

Examples:
  python3 kg_stats.py
  python3 kg_stats.py --full
  python3 kg_stats.py --json
  python3 kg_stats.py --category Projects
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

REQUIRED_FM_FIELDS = ["type", "date"]
STALE_DAYS = 30
RECENT_DAYS = 7


# ─── Parsing utilities ────────────────────────────────────────────────────────

def parse_frontmatter(text):
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
                    if val.startswith("[") and val.endswith("]"):
                        val = [v.strip().strip('"').strip("'")
                               for v in val[1:-1].split(",") if v.strip()]
                    fm[key] = val
    return fm, body


def extract_wikilinks(text):
    raw = re.findall(r'\[\[([^\]]+)\]\]', text)
    results = []
    for link in raw:
        if "|" in link:
            link = link.split("|")[0]
        link = link.split("/")[-1]
        results.append(link.strip())
    return results


def load_all_notes(vault_root):
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
            category = parts[1]
        elif len(parts) > 1:
            category = parts[0]
        else:
            category = "Root"

        notes.append({
            "path": rel_path,
            "abs_path": str(md_file),
            "title": md_file.stem,
            "category": category,
            "frontmatter": fm,
            "body": body,
            "wikilinks": extract_wikilinks(text),
            "mtime": md_file.stat().st_mtime,
            "size": md_file.stat().st_size,
        })
    return notes


# ─── Analysis ─────────────────────────────────────────────────────────────────

def analyze_vault(notes):
    """Run full vault analysis, return structured report dict."""
    now = datetime.now().timestamp()
    all_titles = {n["title"] for n in notes}

    # Backlink index
    backlinks = defaultdict(list)
    for note in notes:
        for link in note["wikilinks"]:
            if note["title"] not in backlinks[link]:
                backlinks[link].append(note["title"])

    # Per-note metrics
    note_data = []
    for n in notes:
        bl_count = len(backlinks.get(n["title"], []))
        out_count = len(n["wikilinks"])
        age_days = (now - n["mtime"]) / 86400

        # Find broken links (links to notes that don't exist in vault)
        broken = [l for l in n["wikilinks"] if l not in all_titles and l != n["title"]]

        # Missing required frontmatter
        missing_fm = [f for f in REQUIRED_FM_FIELDS if f not in n["frontmatter"] or not n["frontmatter"][f]]

        note_data.append({
            "title": n["title"],
            "path": n["path"],
            "category": n["category"],
            "backlinks": backlinks.get(n["title"], []),
            "backlink_count": bl_count,
            "outgoing_count": out_count,
            "total_connections": bl_count + out_count,
            "is_orphan": bl_count == 0 and out_count == 0,
            "age_days": round(age_days, 1),
            "is_stale": age_days > STALE_DAYS,
            "is_recent": age_days < RECENT_DAYS,
            "broken_links": broken,
            "missing_frontmatter": missing_fm,
            "has_issues": bool(broken or missing_fm),
            "body_words": len(n["body"].split()),
            "size_bytes": n["size"],
        })

    # Category stats
    by_cat = defaultdict(lambda: {"count": 0, "total_links": 0, "orphans": 0, "stale": 0})
    for nd in note_data:
        c = nd["category"]
        by_cat[c]["count"] += 1
        by_cat[c]["total_links"] += nd["total_connections"]
        if nd["is_orphan"]: by_cat[c]["orphans"] += 1
        if nd["is_stale"]:  by_cat[c]["stale"] += 1

    # Sort by connectivity for hub nodes
    hubs = sorted(note_data, key=lambda x: -x["total_connections"])

    # Orphans (no links in OR out)
    orphans = [nd for nd in note_data if nd["is_orphan"]]

    # Notes with broken links
    broken_notes = [nd for nd in note_data if nd["broken_links"]]

    # Notes missing frontmatter
    fm_issues = [nd for nd in note_data if nd["missing_frontmatter"]]

    # Timeline summary
    recent = [nd for nd in note_data if nd["is_recent"]]
    stale  = [nd for nd in note_data if nd["is_stale"]]

    # Graph density: actual links / max possible
    n_notes = len(notes)
    max_links = n_notes * (n_notes - 1)
    total_links = sum(nd["outgoing_count"] for nd in note_data)
    density = (total_links / max_links * 100) if max_links > 0 else 0

    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vault_root": VAULT_ROOT,
        "summary": {
            "total_notes": n_notes,
            "total_wikilinks": total_links,
            "unique_linked_targets": len(backlinks),
            "orphan_notes": len(orphans),
            "stale_notes": len(stale),
            "recent_notes": len(recent),
            "broken_link_notes": len(broken_notes),
            "fm_issue_notes": len(fm_issues),
            "graph_density_pct": round(density, 3),
        },
        "by_category": {k: dict(v) for k, v in by_cat.items()},
        "hub_nodes": hubs[:15],
        "orphans": orphans,
        "stale_notes": stale,
        "recent_notes": recent,
        "broken_link_notes": broken_notes,
        "fm_issue_notes": fm_issues,
        "all_notes": note_data,
    }


# ─── Report formatting ─────────────────────────────────────────────────────────

def health_score(report):
    """Compute a simple health score 0-100."""
    s = report["summary"]
    total = s["total_notes"]
    if total == 0:
        return 100
    orphan_pct   = s["orphan_notes"] / total
    stale_pct    = s["stale_notes"] / total
    broken_pct   = s["broken_link_notes"] / total
    fm_issue_pct = s["fm_issue_notes"] / total
    score = 100 - (orphan_pct * 30 + stale_pct * 20 + broken_pct * 30 + fm_issue_pct * 20)
    return max(0, round(score))


def score_badge(score):
    if score >= 90: return "EXCELLENT"
    if score >= 75: return "GOOD"
    if score >= 60: return "FAIR"
    if score >= 40: return "NEEDS WORK"
    return "CRITICAL"


def print_report(report, full=False, category_filter=None):
    s = report["summary"]
    score = health_score(report)
    badge = score_badge(score)

    width = 60
    print("=" * width)
    print("  OBSIDIAN VAULT HEALTH REPORT")
    print("  Generated: {}".format(report["generated_at"]))
    print("=" * width)

    print("\n  HEALTH SCORE: {} / 100  [{}]".format(score, badge))
    print("  Vault: {}".format(report["vault_root"]))

    print("\n" + "-" * width)
    print("  OVERVIEW")
    print("-" * width)
    print("  Total Notes:            {:>6}".format(s["total_notes"]))
    print("  Total Wikilinks:        {:>6}".format(s["total_wikilinks"]))
    print("  Unique Link Targets:    {:>6}".format(s["unique_linked_targets"]))
    print("  Graph Density:          {:>5.2f}%".format(s["graph_density_pct"]))
    print()
    print("  Orphan Notes:           {:>6}  {}".format(
        s["orphan_notes"],
        "[WARNING]" if s["orphan_notes"] > 0 else "[OK]"))
    print("  Stale Notes (>{}d):     {:>6}  {}".format(
        STALE_DAYS, s["stale_notes"],
        "[WARNING]" if s["stale_notes"] > 3 else "[OK]"))
    print("  Recent Notes (<{}d):     {:>6}".format(RECENT_DAYS, s["recent_notes"]))
    print("  Broken Link Notes:      {:>6}  {}".format(
        s["broken_link_notes"],
        "[WARNING]" if s["broken_link_notes"] > 0 else "[OK]"))
    print("  Frontmatter Issues:     {:>6}  {}".format(
        s["fm_issue_notes"],
        "[WARNING]" if s["fm_issue_notes"] > 0 else "[OK]"))

    # By category
    print("\n" + "-" * width)
    print("  NOTES BY CATEGORY")
    print("-" * width)
    by_cat = report["by_category"]
    if category_filter:
        by_cat = {k: v for k, v in by_cat.items() if k.lower() == category_filter.lower()}
    for cat, stats in sorted(by_cat.items(), key=lambda x: -x[1]["count"]):
        bar = "#" * min(stats["count"], 25)
        orphan_warn = " [!{} orphans]".format(stats["orphans"]) if stats["orphans"] else ""
        stale_warn  = " [!{} stale]".format(stats["stale"]) if stats["stale"] else ""
        print("  {:<25} {:3d}  {}{}{}".format(
            cat, stats["count"], bar, orphan_warn, stale_warn))

    # Hub nodes
    print("\n" + "-" * width)
    print("  HUB NODES (Most Connected)")
    print("-" * width)
    for nd in report["hub_nodes"][:10]:
        if nd["total_connections"] == 0:
            break
        bar = "#" * min(nd["total_connections"], 20)
        print("  {:<35} in:{:2d} out:{:2d}  {}".format(
            nd["title"][:35], nd["backlink_count"], nd["outgoing_count"], bar))

    # Recent notes
    recent = report["recent_notes"]
    if recent:
        print("\n" + "-" * width)
        print("  RECENTLY UPDATED ({} notes in last {} days)".format(
            len(recent), RECENT_DAYS))
        print("-" * width)
        for nd in sorted(recent, key=lambda x: x["age_days"])[:10]:
            age_str = "{:.0f}h ago".format(nd["age_days"] * 24) if nd["age_days"] < 1 else \
                      "{:.0f}d ago".format(nd["age_days"])
            print("  [{:>8}] [{}] {}".format(age_str, nd["category"], nd["title"]))

    # Orphan notes
    orphans = report["orphans"]
    if orphans:
        limit = len(orphans) if full else min(10, len(orphans))
        print("\n" + "-" * width)
        print("  ORPHAN NOTES ({}) - No links in OR out".format(len(orphans)))
        print("-" * width)
        for nd in orphans[:limit]:
            print("  [ORPHAN] [{}] {}".format(nd["category"], nd["title"]))
        if not full and len(orphans) > 10:
            print("  ... and {} more (use --full to see all)".format(len(orphans) - 10))

    # Broken links
    broken = report["broken_link_notes"]
    if broken:
        limit = len(broken) if full else min(10, len(broken))
        print("\n" + "-" * width)
        print("  BROKEN LINKS ({} notes with missing targets)".format(len(broken)))
        print("-" * width)
        for nd in broken[:limit]:
            print("  [{}] {}".format(nd["category"], nd["title"]))
            for bl in nd["broken_links"][:3]:
                print("    -> [[{}]] (not found)".format(bl))
            if len(nd["broken_links"]) > 3:
                print("    ... and {} more".format(len(nd["broken_links"]) - 3))
        if not full and len(broken) > 10:
            print("  ... and {} more (use --full to see all)".format(len(broken) - 10))

    # Frontmatter issues
    fm_issues = report["fm_issue_notes"]
    if fm_issues:
        limit = len(fm_issues) if full else min(10, len(fm_issues))
        print("\n" + "-" * width)
        print("  FRONTMATTER ISSUES ({} notes)".format(len(fm_issues)))
        print("-" * width)
        for nd in fm_issues[:limit]:
            print("  [{}] {}".format(nd["category"], nd["title"]))
            print("    Missing: {}".format(", ".join(nd["missing_frontmatter"])))
        if not full and len(fm_issues) > 10:
            print("  ... and {} more (use --full to see all)".format(len(fm_issues) - 10))

    # Stale notes
    stale = report["stale_notes"]
    if stale and full:
        print("\n" + "-" * width)
        print("  STALE NOTES (not updated in >{}d)  -- {} total".format(
            STALE_DAYS, len(stale)))
        print("-" * width)
        for nd in sorted(stale, key=lambda x: -x["age_days"])[:15]:
            print("  [{:4.0f}d] [{}] {}".format(nd["age_days"], nd["category"], nd["title"]))

    print("\n" + "=" * width)
    print("  Tip: Run 'python3 kg_auto_link.py --dry-run' to find missing links")
    print("  Tip: Run 'python3 kg_query_vault.py stats' for quick overview")
    print("=" * width)
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if VAULT_ROOT is None:
        print("ERROR: Obsidian vault not found. Checked:")
        for p in VAULT_PATHS: print("  {}".format(p))
        sys.exit(1)

    args = sys.argv[1:]
    as_json = "--json" in args
    full = "--full" in args

    category_filter = None
    if "--category" in args:
        idx = args.index("--category")
        if idx + 1 < len(args):
            category_filter = args[idx + 1]

    print("Loading vault from {}...".format(VAULT_ROOT), file=sys.stderr)
    notes = load_all_notes(VAULT_ROOT)
    print("Analyzing {} notes...".format(len(notes)), file=sys.stderr)
    report = analyze_vault(notes)

    if as_json:
        # For JSON output, exclude the large all_notes array unless --full
        output = {k: v for k, v in report.items() if k != "all_notes"}
        if full:
            output["all_notes"] = report["all_notes"]
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print_report(report, full=full, category_filter=category_filter)


if __name__ == "__main__":
    main()
