from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import Settings
from .db import connect
from .markdown import read_markdown_body


def ensure_fts(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS objects_fts USING fts5(object_id UNINDEXED, title, body)"
        )
        conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(chunk_id UNINDEXED, source_id UNINDEXED, text)"
        )
        return True
    except sqlite3.OperationalError:
        return False


def refresh_fts(settings: Settings) -> bool:
    with connect(settings.database) as conn:
        try:
            conn.execute("DROP TABLE IF EXISTS objects_fts")
            conn.execute("DROP TABLE IF EXISTS chunks_fts")
            conn.commit()
        except sqlite3.OperationalError:
            return False
        enabled = ensure_fts(conn)
        if not enabled:
            return False

        rows = conn.execute("SELECT object_id, title, path FROM objects ORDER BY object_id")
        for row in rows:
            body = read_markdown_body(settings.root / row["path"])
            conn.execute(
                "INSERT INTO objects_fts (object_id, title, body) VALUES (?, ?, ?)",
                (row["object_id"], row["title"], body),
            )

        rows = conn.execute("SELECT chunk_id, source_id, text FROM chunks ORDER BY chunk_id")
        for row in rows:
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, source_id, text) VALUES (?, ?, ?)",
                (row["chunk_id"], row["source_id"], row["text"]),
            )
        conn.commit()
        return True
