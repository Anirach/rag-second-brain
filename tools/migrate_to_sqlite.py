#!/usr/bin/env python3
"""
migrate_to_sqlite.py — Migrate existing memory files into memory.db (SQLite + FTS5)
Sources:
  1. Obsidian vault  (/home/clawdbot/clawd/obsidian-vault or /workspace/obsidian-vault)
  2. Daily memory logs (/home/clawdbot/clawd/memory/*.md)
  3. MEMORY.md — long-term memory file

Features:
  - MD5 dedup: safe to re-run multiple times
  - --dry-run: show what would be imported without writing
  - --verbose: detailed per-file logging
"""

import sqlite3
import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

# ─── Paths ─────────────────────────────────────────────────────────────────────

SCRIPT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
WORKSPACE  = SCRIPT_DIR.parent  # /home/clawdbot/clawd or /workspace

# Try both sandbox and host paths
VAULT_CANDIDATES = [
    WORKSPACE / "obsidian-vault",
    Path("/home/clawdbot/obsidian-vault"),
]
MEMORY_DIR    = WORKSPACE / "memory"
MEMORY_MD     = WORKSPACE / "MEMORY.md"
DB_PATH       = WORKSPACE / "memory.db"

VALID_TYPES = {"fact", "decision", "todo", "event", "person", "project", "lesson", "preference", "note"}

# Obsidian folder → memory type mapping
FOLDER_TYPE_MAP = {
    "People":       "person",
    "Projects":     "project",
    "Decisions":    "decision",
    "Topics":       "note",
    "Action-Items": "todo",
    "Events":       "event",
    "Organizations": "note",
    "Documents":    "fact",
    "Lessons":      "lesson",
}

# MEMORY.md section keyword → type mapping
SECTION_TYPE_MAP = {
    "todo": "todo",
    "task": "todo",
    "action": "todo",
    "decision": "decision",
    "chose": "decision",
    "lesson": "lesson",
    "learned": "lesson",
    "preference": "preference",
    "prefer": "preference",
    "person": "person",
    "project": "project",
    "setup": "fact",
    "config": "fact",
    "api": "fact",
    "tool": "fact",
    "fact": "fact",
    "note": "note",
    "memory": "note",
    "event": "event",
    "milestone": "event",
}

# ─── DB helpers ────────────────────────────────────────────────────────────────

def get_db(path):
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db(conn):
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT NOT NULL,
            title       TEXT NOT NULL,
            content     TEXT NOT NULL,
            tags        TEXT,
            source      TEXT,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            archived    BOOLEAN DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS migration_hashes (
            hash TEXT PRIMARY KEY,
            imported_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
            title, content, tags, type,
            content=memories,
            content_rowid=id
        );

        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memories_fts(rowid, title, content, tags, type)
            VALUES (new.id, new.title, new.content, COALESCE(new.tags,''), new.type);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, type)
            VALUES ('delete', old.id, old.title, old.content, COALESCE(old.tags,''), old.type);
            INSERT INTO memories_fts(rowid, title, content, tags, type)
            VALUES (new.id, new.title, new.content, COALESCE(new.tags,''), new.type);
        END;

        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memories_fts(memories_fts, rowid, title, content, tags, type)
            VALUES ('delete', old.id, old.title, old.content, COALESCE(old.tags,''), old.type);
        END;
    """)
    conn.commit()

def content_hash(mtype, title, content):
    key = f"{mtype}||{title}||{content[:500]}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def is_duplicate(conn, mtype, title, content):
    h = content_hash(mtype, title, content)
    row = conn.execute("SELECT 1 FROM migration_hashes WHERE hash=?", (h,)).fetchone()
    return bool(row), h

def insert_memory(conn, mtype, title, content, tags="", source="", created_at=None, dry_run=False):
    """Returns (inserted: bool, duplicate: bool)"""
    title   = (title or "Untitled").strip()[:500]
    content = (content or "").strip()
    tags    = (tags or "").strip()
    if not content:
        return False, False

    is_dup, h = is_duplicate(conn, mtype, title, content)
    if is_dup:
        return False, True

    if dry_run:
        return True, False

    dt = created_at or datetime.now().isoformat()
    conn.execute(
        "INSERT INTO memories (type, title, content, tags, source, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (mtype, title, content, tags, source or "", dt, dt)
    )
    conn.execute("INSERT OR IGNORE INTO migration_hashes (hash) VALUES (?)", (h,))
    conn.commit()
    return True, False

# ─── Parsing helpers ───────────────────────────────────────────────────────────

def parse_frontmatter(text):
    """Return (frontmatter dict, body text)."""
    fm = {}
    body = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            fm_text = text[3:end].strip()
            body = text[end+3:].strip()
            for line in fm_text.splitlines():
                if ":" in line:
                    k, _, v = line.partition(":")
                    fm[k.strip().lower()] = v.strip().strip('"').strip("'")
    return fm, body

def extract_tags_from_text(text):
    """Find #tags or yaml tags list in text."""
    tags = set()
    # YAML list: tags: [a, b, c]
    m = re.search(r'tags:\s*\[([^\]]+)\]', text)
    if m:
        for t in m.group(1).split(","):
            t = t.strip().strip('"').strip("'")
            if t:
                tags.add(t)
    # Inline #hashtags
    for t in re.findall(r'#(\w[\w-]*)', text):
        tags.add(t)
    return ", ".join(sorted(tags))

def guess_date(path: Path, fallback=None):
    """Try to extract a date from filename like 2026-02-21.md"""
    m = re.search(r'(\d{4}-\d{2}-\d{2})', path.stem)
    if m:
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").isoformat()
        except ValueError:
            pass
    return fallback or datetime.now().isoformat()

def guess_type_from_section_title(title: str) -> str:
    low = title.lower()
    for kw, t in SECTION_TYPE_MAP.items():
        if kw in low:
            return t
    return "note"

# ─── Source 1: Obsidian Vault ──────────────────────────────────────────────────

def find_vault():
    for p in VAULT_CANDIDATES:
        if p.exists():
            return p
    return None

def migrate_vault(conn, vault: Path, dry_run=False, verbose=False):
    stats = {"files": 0, "inserted": 0, "skipped": 0, "dupes": 0}
    kg_dir = vault / "KnowledgeGraph"
    if not kg_dir.exists():
        if verbose:
            print(f"  [vault] KnowledgeGraph not found at {kg_dir}")
        return stats

    for md_file in sorted(kg_dir.rglob("*.md")):
        # Determine type from parent folder
        rel = md_file.relative_to(kg_dir)
        folder = rel.parts[0] if len(rel.parts) > 1 else ""
        mtype = FOLDER_TYPE_MAP.get(folder, "note")

        text = md_file.read_text(encoding="utf-8", errors="replace")
        fm, body = parse_frontmatter(text)

        # Title: frontmatter id, or H1 heading, or filename
        title = fm.get("title", "")
        if not title:
            h1 = re.search(r'^#\s+(.+)$', body, re.MULTILINE)
            title = h1.group(1).strip() if h1 else md_file.stem

        tags_raw = fm.get("tags", "")
        if isinstance(tags_raw, list):
            tags = ", ".join(tags_raw)
        else:
            tags = extract_tags_from_text(text)

        source = str(md_file.relative_to(WORKSPACE))
        created_at = fm.get("date", guess_date(md_file))
        if created_at and len(created_at) == 10:
            created_at += "T00:00:00"

        stats["files"] += 1
        ok, dup = insert_memory(conn, mtype, title, body, tags=tags, source=source,
                                created_at=created_at, dry_run=dry_run)
        if ok:
            stats["inserted"] += 1
            if verbose:
                print(f"  [vault] +{mtype:8s} {title[:60]}")
        elif dup:
            stats["dupes"] += 1
            if verbose:
                print(f"  [vault] ~{mtype:8s} {title[:60]} (dup)")
        else:
            stats["skipped"] += 1

    return stats

# ─── Source 2: Daily Memory Logs ──────────────────────────────────────────────

def split_sections(text, heading_level="###"):
    """Split markdown by ### headings, return list of (title, body)."""
    pattern = re.compile(rf'^{re.escape(heading_level)}\s+(.+)$', re.MULTILINE)
    matches = list(pattern.finditer(text))
    sections = []
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(text)
        title = m.group(1).strip()
        body = text[start:end].strip()
        if body:
            sections.append((title, body))
    return sections

def migrate_daily_logs(conn, memory_dir: Path, dry_run=False, verbose=False):
    stats = {"files": 0, "inserted": 0, "skipped": 0, "dupes": 0}
    if not memory_dir.exists():
        if verbose:
            print(f"  [daily] directory not found: {memory_dir}")
        return stats

    for md_file in sorted(memory_dir.glob("*.md")):
        if md_file.name == "README.md":
            continue
        text = md_file.read_text(encoding="utf-8", errors="replace")
        created_at_base = guess_date(md_file)
        source = str(md_file.relative_to(WORKSPACE))
        stats["files"] += 1

        # Try splitting by ### headings first
        sections = split_sections(text, "###")

        # Fall back to ## headings
        if not sections:
            sections = split_sections(text, "##")

        # If no headings, treat entire file as one note
        if not sections:
            title = md_file.stem
            mtype = "note"
            ok, dup = insert_memory(conn, mtype, title, text.strip(),
                                    source=source, created_at=created_at_base, dry_run=dry_run)
            if ok:
                stats["inserted"] += 1
                if verbose:
                    print(f"  [daily] +{mtype:8s} {title[:60]}")
            elif dup:
                stats["dupes"] += 1
            continue

        for title, body in sections:
            if len(body) < 10:
                continue
            mtype = guess_type_from_section_title(title)
            tags = extract_tags_from_text(body)
            ok, dup = insert_memory(conn, mtype, title, body, tags=tags,
                                    source=source, created_at=created_at_base, dry_run=dry_run)
            if ok:
                stats["inserted"] += 1
                if verbose:
                    print(f"  [daily] +{mtype:8s} {title[:60]}")
            elif dup:
                stats["dupes"] += 1
            else:
                stats["skipped"] += 1

    return stats

# ─── Source 3: MEMORY.md ──────────────────────────────────────────────────────

def migrate_memory_md(conn, memory_md: Path, dry_run=False, verbose=False):
    stats = {"files": 0, "inserted": 0, "skipped": 0, "dupes": 0}
    if not memory_md.exists():
        if verbose:
            print(f"  [MEMORY.md] not found: {memory_md}")
        return stats

    text = memory_md.read_text(encoding="utf-8", errors="replace")
    stats["files"] = 1
    source = str(memory_md.relative_to(WORKSPACE))

    # Parse by ## sections (H2 = major sections)
    sections_h2 = split_sections(text, "##")
    if not sections_h2:
        # Treat whole file as one fact
        ok, dup = insert_memory(conn, "fact", "MEMORY.md", text.strip(),
                                source=source, dry_run=dry_run)
        if ok:
            stats["inserted"] += 1
        elif dup:
            stats["dupes"] += 1
        return stats

    for section_title, section_body in sections_h2:
        # Try to split further by ### subsections
        subsections = split_sections(section_body, "###")
        if subsections:
            for sub_title, sub_body in subsections:
                if len(sub_body) < 15:
                    continue
                full_title = f"{section_title} — {sub_title}"
                mtype = guess_type_from_section_title(full_title)
                tags = extract_tags_from_text(sub_body)
                ok, dup = insert_memory(conn, mtype, full_title, sub_body,
                                        tags=tags, source=source, dry_run=dry_run)
                if ok:
                    stats["inserted"] += 1
                    if verbose:
                        print(f"  [MEMORY] +{mtype:8s} {full_title[:60]}")
                elif dup:
                    stats["dupes"] += 1
                else:
                    stats["skipped"] += 1
        else:
            # Use whole section
            if len(section_body) < 15:
                continue
            mtype = guess_type_from_section_title(section_title)
            tags = extract_tags_from_text(section_body)
            ok, dup = insert_memory(conn, mtype, section_title, section_body,
                                    tags=tags, source=source, dry_run=dry_run)
            if ok:
                stats["inserted"] += 1
                if verbose:
                    print(f"  [MEMORY] +{mtype:8s} {section_title[:60]}")
            elif dup:
                stats["dupes"] += 1
            else:
                stats["skipped"] += 1

    return stats

# ─── Main ──────────────────────────────────────────────────────────────────────

def merge_stats(a, b):
    return {k: a.get(k, 0) + b.get(k, 0) for k in set(a) | set(b)}

def main():
    parser = argparse.ArgumentParser(
        prog="migrate_to_sqlite.py",
        description="Migrate Obsidian vault + memory files into memory.db"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be imported without writing")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print each imported item")
    parser.add_argument("--db", default=str(DB_PATH), help=f"DB path (default: {DB_PATH})")
    parser.add_argument("--source", choices=["vault", "daily", "memory", "all"], default="all",
                        help="Which sources to migrate (default: all)")
    args = parser.parse_args()

    if args.dry_run:
        print("  [DRY RUN — nothing will be written]\n")

    conn = get_db(args.db)
    init_db(conn)

    total_stats = {"files": 0, "inserted": 0, "skipped": 0, "dupes": 0}

    # 1. Obsidian Vault
    if args.source in ("vault", "all"):
        vault = find_vault()
        if vault:
            print(f"\n📂 Obsidian Vault: {vault}")
            s = migrate_vault(conn, vault, dry_run=args.dry_run, verbose=args.verbose)
            print(f"   Files: {s['files']}  Inserted: {s['inserted']}  Dupes: {s['dupes']}  Skipped: {s['skipped']}")
            total_stats = merge_stats(total_stats, s)
        else:
            print("\n⚠️  Obsidian vault not found (checked: " + ", ".join(str(p) for p in VAULT_CANDIDATES) + ")")

    # 2. Daily Memory Logs
    if args.source in ("daily", "all"):
        print(f"\n📝 Daily Memory Logs: {MEMORY_DIR}")
        s = migrate_daily_logs(conn, MEMORY_DIR, dry_run=args.dry_run, verbose=args.verbose)
        print(f"   Files: {s['files']}  Inserted: {s['inserted']}  Dupes: {s['dupes']}  Skipped: {s['skipped']}")
        total_stats = merge_stats(total_stats, s)

    # 3. MEMORY.md
    if args.source in ("memory", "all"):
        print(f"\n🧠 MEMORY.md: {MEMORY_MD}")
        s = migrate_memory_md(conn, MEMORY_MD, dry_run=args.dry_run, verbose=args.verbose)
        print(f"   Files: {s['files']}  Inserted: {s['inserted']}  Dupes: {s['dupes']}  Skipped: {s['skipped']}")
        total_stats = merge_stats(total_stats, s)

    # Rebuild FTS
    if not args.dry_run:
        try:
            conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
            conn.commit()
        except Exception:
            pass  # ignore if nothing inserted

    print(f"\n{'='*50}")
    print(f"  TOTAL  Files: {total_stats['files']}  "
          f"Inserted: {total_stats['inserted']}  "
          f"Dupes: {total_stats['dupes']}  "
          f"Skipped: {total_stats['skipped']}")
    if not args.dry_run:
        total_active = conn.execute("SELECT COUNT(*) FROM memories WHERE archived=0").fetchone()[0]
        print(f"  DB now has {total_active} active memories in {args.db}")
    print()
    conn.close()

if __name__ == "__main__":
    main()
