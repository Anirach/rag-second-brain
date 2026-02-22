#!/usr/bin/env python3
"""
kg_auto_link.py - Knowledge Graph Auto-Linker for Obsidian Vault
=================================================================
Scans notes for unlinked mentions of known entities and suggests or
automatically adds [[wikilinks]]. Also updates MOC files.

Usage:
  python3 kg_auto_link.py [--dry-run] [--auto] [--since DAYS] [--moc]

Flags:
  --dry-run     Show what would be changed (default: show suggestions only)
  --auto        Actually apply the changes (write files)
  --since N     Only process notes modified in last N days (default: 7)
  --all         Process all notes (not just recent)
  --moc         Update MOC (Map of Content) files

Examples:
  python3 kg_auto_link.py --dry-run          # Show suggestions
  python3 kg_auto_link.py --auto             # Apply changes
  python3 kg_auto_link.py --auto --all       # Process everything
  python3 kg_auto_link.py --moc --dry-run    # Preview MOC updates
"""

import os
import re
import sys
import json
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

# ─── Vault path detection ────────────────────────────────────────────────────
VAULT_PATHS = [
    "/home/clawdbot/obsidian-vault",
    "/workspace/obsidian-vault",
]
VAULT_ROOT = next((p for p in VAULT_PATHS if os.path.isdir(p)), None)

# Category dirs to scan for known entities
ENTITY_DIRS = {
    "People":        "KnowledgeGraph/People",
    "Projects":      "KnowledgeGraph/Projects",
    "Topics":        "KnowledgeGraph/Topics",
    "Events":        "KnowledgeGraph/Events",
    "Organizations": "KnowledgeGraph/Organizations",
    "Documents":     "KnowledgeGraph/Documents",
    "Decisions":     "KnowledgeGraph/Decisions",
}

# MOC templates for each category
MOC_TEMPLATE = """---
type: "MOC"
category: "{category}"
updated: "{date}"
---

# Map of Content: {category}

Auto-generated index of all {category} notes.

## Notes

{entries}
"""


# ─── Core utilities ───────────────────────────────────────────────────────────

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


def get_existing_wikilinks(text):
    """Return set of already-linked titles."""
    linked = set()
    for m in re.finditer(r'\[\[([^\]]+)\]\]', text):
        link = m.group(1)
        if "|" in link:
            link = link.split("|")[0]
        linked.add(link.split("/")[-1].strip())
    return linked


def load_entities(vault_root):
    """Load all known entities from KnowledgeGraph subdirs."""
    entities = []  # list of (title, category, rel_path, aliases)
    for category, subdir in ENTITY_DIRS.items():
        dir_path = Path(vault_root) / subdir
        if not dir_path.exists():
            continue
        for md_file in sorted(dir_path.glob("*.md")):
            try:
                text = md_file.read_text(encoding="utf-8")
            except Exception:
                continue
            fm, _ = parse_frontmatter(text)
            title = md_file.stem
            # Collect aliases from frontmatter
            aliases = [title]
            if "aliases" in fm:
                if isinstance(fm["aliases"], list):
                    aliases.extend(fm["aliases"])
                elif isinstance(fm["aliases"], str):
                    aliases.append(fm["aliases"])
            # Also add short forms (first word for multi-word titles)
            words = title.split()
            if len(words) > 1 and len(words[0]) > 3:
                # Only add short form if it's distinctive enough
                pass
            rel_path = str(md_file.relative_to(vault_root))
            entities.append({
                "title": title,
                "category": category,
                "path": rel_path,
                "aliases": aliases,
            })
    return entities


def load_all_notes(vault_root, since_days=None):
    """Load all notes, optionally filtered by modification time."""
    notes = []
    cutoff = None
    if since_days is not None:
        cutoff = datetime.now().timestamp() - since_days * 86400

    for md_file in sorted(Path(vault_root).rglob("*.md")):
        mtime = md_file.stat().st_mtime
        if cutoff and mtime < cutoff:
            continue
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception:
            continue
        fm, body = parse_frontmatter(text)
        rel_path = str(md_file.relative_to(vault_root))
        notes.append({
            "path": rel_path,
            "abs_path": str(md_file),
            "title": md_file.stem,
            "text": text,
            "frontmatter": fm,
            "body": body,
            "mtime": mtime,
        })
    return notes


def suggest_links_for_note(note, entities, min_length=4):
    """
    Find entity mentions in a note that aren't already wikilinked.
    Returns list of (entity_title, mention_text, position) suggestions.
    """
    text = note["text"]
    note_title = note["title"]
    existing_links = get_existing_wikilinks(text)
    suggestions = []
    already_suggested = set()

    for entity in entities:
        # Don't link a note to itself
        if entity["title"] == note_title:
            continue
        # Don't suggest if already linked
        if entity["title"] in existing_links:
            continue
        # Don't suggest if entity path contains this note's path (prevent circular)
        if note["path"].replace("\\", "/").startswith(
                entity["path"].replace("\\", "/").rsplit("/", 1)[0]):
            continue

        for alias in entity["aliases"]:
            if len(alias) < min_length:
                continue
            if alias in already_suggested:
                continue

            # Search for standalone mention (word boundary)
            pattern = r'(?<!\[\[)(?<!\w)' + re.escape(alias) + r'(?!\w)(?!\]\])'
            try:
                match = re.search(pattern, text, re.IGNORECASE)
            except re.error:
                continue

            if match:
                # Make sure it's not inside an existing [[...]]
                pos = match.start()
                # Check if inside wikilink brackets
                before = text[:pos]
                open_brackets = before.count("[[") - before.count("]]")
                if open_brackets > 0:
                    continue

                suggestions.append({
                    "entity_title": entity["title"],
                    "entity_path": entity["path"],
                    "category": entity["category"],
                    "mention": match.group(0),
                    "position": pos,
                    "context": text[max(0, pos-30):pos+len(alias)+30].strip(),
                })
                already_suggested.add(alias)
                break  # Only suggest once per entity

    return suggestions


def apply_links_to_note(text, suggestions):
    """
    Apply wikilink suggestions to note text.
    Replaces first occurrence of each entity mention with [[wikilink]].
    """
    # Sort by position (descending) to replace from end to avoid position shifts
    to_replace = []
    for s in suggestions:
        alias = s["mention"]
        title = s["entity_title"]
        pattern = r'(?<!\[\[)(?<!\w)' + re.escape(alias) + r'(?!\w)(?!\]\])'
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                to_replace.append((match.start(), match.end(), "[[{}]]".format(title)))
        except re.error:
            pass

    # Sort by start position descending to replace without shifting
    to_replace.sort(key=lambda x: -x[0])
    for start, end, replacement in to_replace:
        text = text[:start] + replacement + text[end:]
    return text


def update_moc(vault_root, category, subdir, notes_in_category, dry_run=True):
    """Create or update a MOC file for a category."""
    moc_dir = Path(vault_root) / "KnowledgeGraph"
    moc_path = moc_dir / "MOC-{}.md".format(category)

    entries = []
    for n in sorted(notes_in_category, key=lambda x: x["title"]):
        fm = n.get("frontmatter", {})
        status = fm.get("status", "")
        role = fm.get("role", "")
        extra = " [{}]".format(status) if status else ""
        extra += " - {}".format(role) if role else ""
        entries.append("- [[{}/{}|{}]]{}".format(
            subdir, n["title"], n["title"], extra))

    content = MOC_TEMPLATE.format(
        category=category,
        date=datetime.now().strftime("%Y-%m-%d"),
        entries="\n".join(entries) if entries else "_No notes yet._"
    )

    if dry_run:
        action = "UPDATE" if moc_path.exists() else "CREATE"
        print("[DRY-RUN] Would {} MOC: {}".format(action, moc_path))
        print("   Entries: {}".format(len(entries)))
        return False

    try:
        moc_path.write_text(content, encoding="utf-8")
        action = "Updated" if moc_path.exists() else "Created"
        print("[MOC] {} {}".format(action, moc_path))
        return True
    except PermissionError:
        print("[ERROR] Cannot write MOC (permission denied): {}".format(moc_path))
        return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if VAULT_ROOT is None:
        print("ERROR: Obsidian vault not found. Checked:")
        for p in VAULT_PATHS: print("  {}".format(p))
        sys.exit(1)

    args = sys.argv[1:]
    dry_run = "--auto" not in args  # dry-run is DEFAULT; need --auto to write
    do_moc  = "--moc" in args
    all_notes = "--all" in args

    since_days = 7
    if "--since" in args:
        idx = args.index("--since")
        if idx + 1 < len(args):
            try:
                since_days = int(args[idx + 1])
            except ValueError:
                pass

    if all_notes:
        since_days = None  # load all notes

    print("=" * 60)
    print("  Knowledge Graph Auto-Linker")
    print("=" * 60)
    print("Vault:     {}".format(VAULT_ROOT))
    print("Mode:      {}".format("DRY-RUN (preview only)" if dry_run else "AUTO (writing changes)"))
    print("Scope:     {}".format("All notes" if since_days is None
                                 else "Notes modified in last {} days".format(since_days)))
    print()

    # Load entities (the "known" things to link to)
    print("Loading entities...")
    entities = load_entities(VAULT_ROOT)
    print("  Found {} entities across {} categories".format(
        len(entities), len(ENTITY_DIRS)))
    for cat, count in sorted(
            {e["category"]: 0 for e in entities}.items()):
        cat_count = sum(1 for e in entities if e["category"] == cat)
        print("    {}: {}".format(cat, cat_count))
    print()

    # Load notes to scan
    print("Loading notes to scan...")
    notes = load_all_notes(VAULT_ROOT, since_days)
    print("  Found {} notes to scan".format(len(notes)))
    print()

    # Analyze and apply
    total_suggestions = 0
    total_applied = 0
    changed_files = []

    for note in notes:
        suggestions = suggest_links_for_note(note, entities)
        if not suggestions:
            continue

        total_suggestions += len(suggestions)
        print("[NOTE] {}  ({} suggestions)".format(note["title"], len(suggestions)))
        for s in suggestions:
            ctx = s["context"].replace("\n", " ")
            print("  + {} [{}]".format(s["entity_title"], s["category"]))
            print("    context: \"...{}...\"".format(ctx[:80]))

        if not dry_run:
            new_text = apply_links_to_note(note["text"], suggestions)
            try:
                Path(note["abs_path"]).write_text(new_text, encoding="utf-8")
                print("  [APPLIED] {} links added".format(len(suggestions)))
                total_applied += len(suggestions)
                changed_files.append(note["path"])
            except PermissionError:
                print("  [ERROR] Cannot write (permission denied): {}".format(note["abs_path"]))
        print()

    # MOC updates
    if do_moc:
        print("\n" + "=" * 40)
        print("  Updating MOC Files")
        print("=" * 40)
        all_notes_full = load_all_notes(VAULT_ROOT, since_days=None)
        for category, subdir in ENTITY_DIRS.items():
            cat_notes = [n for n in all_notes_full
                         if n["path"].replace("\\", "/").startswith(subdir + "/")]
            if cat_notes or True:  # Always create MOC even if empty
                update_moc(VAULT_ROOT, category, subdir, cat_notes, dry_run)

    # Summary
    print("=" * 60)
    print("Summary:")
    print("  Notes scanned:         {}".format(len(notes)))
    print("  Link suggestions:      {}".format(total_suggestions))
    if not dry_run:
        print("  Links applied:         {}".format(total_applied))
        print("  Files modified:        {}".format(len(changed_files)))
    else:
        print("  (Run with --auto to apply changes)")
    print()

    if total_suggestions == 0:
        print("No unlinked entity mentions found. Vault is well-linked!")


if __name__ == "__main__":
    main()
