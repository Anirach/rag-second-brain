#!/usr/bin/env python3
"""
memory_db.py — SQLite + FTS5 Memory CLI Tool
DB: /home/clawdbot/clawd/memory.db
Usage: python3 tools/memory_db.py <command> [options]
"""

import sqlite3
import argparse
import json
import sys
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "memory.db")

VALID_TYPES = ["fact", "decision", "todo", "event", "person", "project", "lesson", "preference", "note"]

# ─── DB Setup ──────────────────────────────────────────────────────────────────

def get_db(path=None):
    db_path = path or DB_PATH
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
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

# ─── Formatting ────────────────────────────────────────────────────────────────

def truncate(s, n):
    if not s:
        return ""
    s = str(s).replace("\n", " ")
    return s[:n-1] + "…" if len(s) > n else s

def fmt_date(dt_str):
    if not dt_str:
        return ""
    try:
        dt = datetime.fromisoformat(dt_str)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(dt_str)[:16]

def print_table(rows, columns, col_widths, output_json=False):
    if output_json:
        out = []
        for row in rows:
            d = {}
            for col in columns:
                d[col] = row[col] if col in row.keys() else None
            out.append(d)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if not rows:
        print("  (no results)")
        return

    # Header
    header = " | ".join(col.upper().ljust(col_widths[i]) for i, col in enumerate(columns))
    sep = "-+-".join("-" * w for w in col_widths)
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for i, col in enumerate(columns):
            val = row[col] if col in row.keys() else ""
            if col in ("created_at", "updated_at"):
                val = fmt_date(val)
            cells.append(truncate(str(val) if val is not None else "", col_widths[i]).ljust(col_widths[i]))
        print(" | ".join(cells))

def print_single(row, output_json=False):
    if output_json:
        print(json.dumps(dict(row), ensure_ascii=False, indent=2))
        return
    print()
    for k in row.keys():
        val = row[k]
        if k in ("created_at", "updated_at"):
            val = fmt_date(val)
        label = k.upper().ljust(12)
        if k == "content":
            print(f"  {label}: {val}")
        else:
            print(f"  {label}: {val}")
    print()

# ─── Commands ──────────────────────────────────────────────────────────────────

def cmd_add(args, conn):
    mtype = args.type.lower()
    if mtype not in VALID_TYPES:
        print(f"Error: type must be one of: {', '.join(VALID_TYPES)}", file=sys.stderr)
        sys.exit(1)
    cur = conn.execute(
        "INSERT INTO memories (type, title, content, tags, source) VALUES (?,?,?,?,?)",
        (mtype, args.title, args.content, args.tags, args.source)
    )
    conn.commit()
    row_id = cur.lastrowid
    if args.json:
        print(json.dumps({"id": row_id, "status": "added"}))
    else:
        print(f"✓ Added memory #{row_id}: [{mtype}] {args.title}")

def cmd_search(args, conn):
    query = args.query
    params = [query]
    base_sql = """
        SELECT m.id, m.type, m.title, m.tags, m.created_at
        FROM memories_fts f
        JOIN memories m ON m.id = f.rowid
        WHERE memories_fts MATCH ?
          AND m.archived = 0
    """
    if args.type:
        base_sql += " AND m.type = ?"
        params.append(args.type)
    if args.after:
        base_sql += " AND m.created_at >= ?"
        params.append(args.after)
    if args.before:
        base_sql += " AND m.created_at <= ?"
        params.append(args.before)
    base_sql += " ORDER BY rank LIMIT ?"
    params.append(args.limit)

    try:
        rows = conn.execute(base_sql, params).fetchall()
    except sqlite3.OperationalError as e:
        # FTS might need rebuild
        print(f"Search error: {e}\nTry: python3 tools/memory_db.py rebuild-fts", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print_table(rows, ["id", "type", "title", "tags", "created_at"], [], output_json=True)
    else:
        print(f"\n  Search: '{query}'" + (f"  type={args.type}" if args.type else "") + f"  — {len(rows)} result(s)\n")
        print_table(rows,
                    ["id", "type", "title", "tags", "created_at"],
                    [5, 10, 45, 25, 16])

def cmd_recent(args, conn):
    rows = conn.execute(
        "SELECT id, type, title, tags, created_at FROM memories WHERE archived=0 ORDER BY created_at DESC LIMIT ?",
        (args.limit,)
    ).fetchall()
    if args.json:
        print_table(rows, ["id", "type", "title", "tags", "created_at"], [], output_json=True)
    else:
        print(f"\n  Recent memories (limit={args.limit}):\n")
        print_table(rows, ["id", "type", "title", "tags", "created_at"], [5, 10, 45, 25, 16])

def cmd_list(args, conn):
    sql = "SELECT id, type, title, tags, created_at FROM memories WHERE archived=0"
    params = []
    if args.type:
        sql += " AND type = ?"
        params.append(args.type)
    sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(args.limit)
    rows = conn.execute(sql, params).fetchall()
    if args.json:
        print_table(rows, ["id", "type", "title", "tags", "created_at"], [], output_json=True)
    else:
        label = f"type={args.type}" if args.type else "all types"
        print(f"\n  Memories ({label}, limit={args.limit}):\n")
        print_table(rows, ["id", "type", "title", "tags", "created_at"], [5, 10, 45, 25, 16])

def cmd_get(args, conn):
    row = conn.execute("SELECT * FROM memories WHERE id=?", (args.id,)).fetchone()
    if not row:
        print(f"Error: memory #{args.id} not found", file=sys.stderr)
        sys.exit(1)
    print_single(row, output_json=args.json)

def cmd_update(args, conn):
    row = conn.execute("SELECT * FROM memories WHERE id=?", (args.id,)).fetchone()
    if not row:
        print(f"Error: memory #{args.id} not found", file=sys.stderr)
        sys.exit(1)
    updates = []
    params = []
    if args.title is not None:
        updates.append("title=?"); params.append(args.title)
    if args.content is not None:
        updates.append("content=?"); params.append(args.content)
    if args.tags is not None:
        updates.append("tags=?"); params.append(args.tags)
    if args.type is not None:
        updates.append("type=?"); params.append(args.type)
    if args.source is not None:
        updates.append("source=?"); params.append(args.source)
    if not updates:
        print("Nothing to update.", file=sys.stderr)
        sys.exit(1)
    updates.append("updated_at=CURRENT_TIMESTAMP")
    params.append(args.id)
    conn.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id=?", params)
    conn.commit()
    if args.json:
        print(json.dumps({"id": args.id, "status": "updated"}))
    else:
        print(f"✓ Updated memory #{args.id}")

def cmd_archive(args, conn):
    row = conn.execute("SELECT id FROM memories WHERE id=?", (args.id,)).fetchone()
    if not row:
        print(f"Error: memory #{args.id} not found", file=sys.stderr)
        sys.exit(1)
    conn.execute("UPDATE memories SET archived=1, updated_at=CURRENT_TIMESTAMP WHERE id=?", (args.id,))
    conn.commit()
    if args.json:
        print(json.dumps({"id": args.id, "status": "archived"}))
    else:
        print(f"✓ Archived memory #{args.id}")

def cmd_stats(args, conn):
    total = conn.execute("SELECT COUNT(*) FROM memories WHERE archived=0").fetchone()[0]
    archived = conn.execute("SELECT COUNT(*) FROM memories WHERE archived=1").fetchone()[0]
    by_type = conn.execute(
        "SELECT type, COUNT(*) as cnt FROM memories WHERE archived=0 GROUP BY type ORDER BY cnt DESC"
    ).fetchall()

    if args.json:
        data = {
            "total": total,
            "archived": archived,
            "by_type": {r["type"]: r["cnt"] for r in by_type}
        }
        print(json.dumps(data, indent=2))
        return

    print(f"\n  Memory Database Stats")
    print(f"  {'─'*30}")
    print(f"  Active memories  : {total}")
    print(f"  Archived         : {archived}")
    print(f"  Total            : {total + archived}")
    print(f"\n  {'TYPE':<15} {'COUNT':>6}")
    print(f"  {'─'*22}")
    for r in by_type:
        print(f"  {r['type']:<15} {r['cnt']:>6}")
    print()

def cmd_export(args, conn):
    rows = conn.execute(
        "SELECT * FROM memories WHERE archived=0 ORDER BY type, created_at"
    ).fetchall()
    if args.json:
        out = [dict(r) for r in rows]
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print(f"# Memory Export — {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
    current_type = None
    for row in rows:
        if row["type"] != current_type:
            current_type = row["type"]
            print(f"\n## {current_type.upper()}\n")
        print(f"### [{row['id']}] {row['title']}")
        print(f"> **Tags:** {row['tags'] or '—'}  **Created:** {fmt_date(row['created_at'])}")
        if row['source']:
            print(f"> **Source:** {row['source']}")
        print()
        print(row['content'])
        print()

def cmd_rebuild_fts(args, conn):
    print("Rebuilding FTS index...")
    conn.execute("INSERT INTO memories_fts(memories_fts) VALUES('rebuild')")
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    if args.json:
        print(json.dumps({"status": "rebuilt", "rows": count}))
    else:
        print(f"✓ FTS index rebuilt ({count} rows)")

# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="memory_db.py",
        description="SQLite + FTS5 memory store for Claude",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--db", default=DB_PATH, help=f"DB path (default: {DB_PATH})")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    sub = parser.add_subparsers(dest="command", required=True)

    # add
    p_add = sub.add_parser("add", help="Add a new memory")
    p_add.add_argument("--type", required=True, choices=VALID_TYPES, help="Memory type")
    p_add.add_argument("--title", required=True, help="Short title")
    p_add.add_argument("--content", required=True, help="Full content")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--source", default="", help="Source file or URL")

    # search
    p_search = sub.add_parser("search", help="Full-text search memories")
    p_search.add_argument("query", help="Search query (FTS5 syntax supported)")
    p_search.add_argument("--type", choices=VALID_TYPES, help="Filter by type")
    p_search.add_argument("--after", metavar="YYYY-MM-DD", help="Only show memories after date")
    p_search.add_argument("--before", metavar="YYYY-MM-DD", help="Only show memories before date")
    p_search.add_argument("--limit", type=int, default=20, help="Max results (default: 20)")

    # recent
    p_recent = sub.add_parser("recent", help="List recent memories")
    p_recent.add_argument("--limit", type=int, default=20, help="How many to show")

    # list
    p_list = sub.add_parser("list", help="List memories (optionally filtered by type)")
    p_list.add_argument("--type", choices=VALID_TYPES, help="Filter by type")
    p_list.add_argument("--limit", type=int, default=50, help="Max results")

    # get
    p_get = sub.add_parser("get", help="Get full details of a memory by ID")
    p_get.add_argument("id", type=int)

    # update
    p_update = sub.add_parser("update", help="Update a memory")
    p_update.add_argument("id", type=int)
    p_update.add_argument("--title", help="New title")
    p_update.add_argument("--content", help="New content")
    p_update.add_argument("--tags", help="New tags")
    p_update.add_argument("--type", choices=VALID_TYPES, help="New type")
    p_update.add_argument("--source", help="New source")

    # archive
    p_archive = sub.add_parser("archive", help="Archive (soft-delete) a memory")
    p_archive.add_argument("id", type=int)

    # stats
    sub.add_parser("stats", help="Show database statistics")

    # export
    sub.add_parser("export", help="Export all memories as Markdown (redirect to file)")

    # rebuild-fts
    sub.add_parser("rebuild-fts", help="Rebuild FTS5 index")

    args = parser.parse_args()

    conn = get_db(args.db)
    init_db(conn)

    dispatch = {
        "add": cmd_add,
        "search": cmd_search,
        "recent": cmd_recent,
        "list": cmd_list,
        "get": cmd_get,
        "update": cmd_update,
        "archive": cmd_archive,
        "stats": cmd_stats,
        "export": cmd_export,
        "rebuild-fts": cmd_rebuild_fts,
    }

    fn = dispatch.get(args.command)
    if fn:
        fn(args, conn)
    else:
        parser.print_help()
        sys.exit(1)

    conn.close()

if __name__ == "__main__":
    main()
