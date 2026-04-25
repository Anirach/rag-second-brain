from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field

from .config import Settings
from .db import connect
from .fts import ensure_fts
from .markdown import read_markdown_body


@dataclass
class QueryHit:
    kind: str
    identifier: str
    title: str
    score: float
    snippet: str
    source_ids: list[str] = field(default_factory=list)
    evidence_ids: list[str] = field(default_factory=list)
    status: str | None = None
    path: str | None = None
    retrieval: str = "scan"


def search(settings: Settings, query: str) -> list[QueryHit]:
    q = query.strip()
    if not q:
        return []
    max_objects = int(settings.retrieval.get("maxObjects", 8))
    max_chunks = int(settings.retrieval.get("maxEvidenceChunks", 12))

    with connect(settings.database) as conn:
        object_hits = _search_objects(conn, settings, q, max_objects)
        seen_object_ids = {hit.identifier for hit in object_hits}
        chunk_hits = _search_chunks(conn, q, max_chunks, seen_object_ids)

    return object_hits + chunk_hits


def format_results(results: list[QueryHit]) -> str:
    if not results:
        return "No matches found."
    lines: list[str] = []
    for hit in results:
        lines.append(f"[{hit.kind}] {hit.title} ({hit.identifier}) | score={hit.score:.2f} | via={hit.retrieval}")
        meta: list[str] = []
        if hit.status:
            meta.append(f"status={hit.status}")
        if hit.source_ids:
            meta.append("sources=" + ", ".join(hit.source_ids))
        if hit.evidence_ids:
            meta.append("evidence=" + ", ".join(hit.evidence_ids[:4]))
        if hit.path:
            meta.append(f"path={hit.path}")
        if meta:
            lines.append("  " + " | ".join(meta))
        lines.append(f"  {hit.snippet}")
    return "\n".join(lines)


def _search_objects(conn: sqlite3.Connection, settings: Settings, query: str, limit: int) -> list[QueryHit]:
    if ensure_fts(conn):
        hits = _search_objects_fts(conn, settings, query, limit)
        if hits:
            return hits
    return _search_objects_scan(conn, settings, query, limit)


def _search_chunks(conn: sqlite3.Connection, query: str, limit: int, seen_object_ids: set[str]) -> list[QueryHit]:
    if ensure_fts(conn):
        hits = _search_chunks_fts(conn, query, limit, seen_object_ids)
        if hits:
            return hits
    return _search_chunks_scan(conn, query, limit, seen_object_ids)


def _search_objects_fts(conn: sqlite3.Connection, settings: Settings, query: str, limit: int) -> list[QueryHit]:
    hits: list[QueryHit] = []
    try:
        rows = conn.execute(
            """
            SELECT o.object_id, o.type, o.title, o.status, o.path, o.confidence,
                   bm25(objects_fts, 8.0, 3.0) AS rank
            FROM objects_fts
            JOIN objects o ON o.object_id = objects_fts.object_id
            WHERE objects_fts MATCH ? AND o.status != 'deprecated'
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), limit * 3),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    for row in rows:
        body = read_markdown_body(settings.root / row["path"])
        evidence_ids = _evidence_ids(conn, row["object_id"])
        source_ids = _source_ids(conn, row["object_id"])
        lexical = _score_text(f"{row['title']}\n{body}", query)
        title_bonus = 4.0 if query.lower() in row["title"].lower() else 0.0
        confidence_bonus = float(row["confidence"] or 0.0)
        status_bonus = _status_bonus(row["status"])
        score = lexical + title_bonus + confidence_bonus + status_bonus + max(0.0, 3.0 - float(row["rank"]))
        hits.append(
            QueryHit(
                kind=row["type"],
                identifier=row["object_id"],
                title=row["title"],
                score=score,
                snippet=_snippet(body, query),
                source_ids=source_ids,
                evidence_ids=evidence_ids,
                status=row["status"],
                path=row["path"],
                retrieval="fts",
            )
        )

    hits.sort(key=lambda h: (-h.score, _kind_rank(h.kind), h.title.lower()))
    return hits[:limit]


def _search_objects_scan(conn: sqlite3.Connection, settings: Settings, query: str, limit: int) -> list[QueryHit]:
    hits: list[QueryHit] = []
    for row in conn.execute("select object_id, title, path, status, type, confidence from objects where status != 'deprecated'"):
        body = read_markdown_body(settings.root / row["path"])
        score = _score_text(f"{row['title']}\n{body}", query)
        if score <= 0:
            continue
        score += 4.0 if query.lower() in row["title"].lower() else 0.0
        score += float(row["confidence"] or 0.0)
        score += _status_bonus(row["status"])
        hits.append(
            QueryHit(
                kind=row["type"],
                identifier=row["object_id"],
                title=row["title"],
                score=score,
                snippet=_snippet(body, query),
                source_ids=_source_ids(conn, row["object_id"]),
                evidence_ids=_evidence_ids(conn, row["object_id"]),
                status=row["status"],
                path=row["path"],
                retrieval="scan",
            )
        )
    hits.sort(key=lambda h: (-h.score, _kind_rank(h.kind), h.title.lower()))
    return hits[:limit]


def _search_chunks_fts(conn: sqlite3.Connection, query: str, limit: int, seen_object_ids: set[str]) -> list[QueryHit]:
    hits: list[QueryHit] = []
    try:
        rows = conn.execute(
            """
            SELECT c.chunk_id, c.source_id, c.text, bm25(chunks_fts, 5.0) AS rank
            FROM chunks_fts
            JOIN chunks c ON c.chunk_id = chunks_fts.chunk_id
            WHERE chunks_fts MATCH ?
            ORDER BY rank
            LIMIT ?
            """,
            (_fts_query(query), limit * 2),
        ).fetchall()
    except sqlite3.OperationalError:
        return []

    for row in rows:
        linked = conn.execute(
            "select object_id from evidence_links where chunk_id = ? order by object_id",
            (row["chunk_id"],),
        ).fetchall()
        linked_ids = [r[0] for r in linked]
        penalty = 1.0 if any(obj_id in seen_object_ids for obj_id in linked_ids) else 0.0
        lexical = _score_text(row["text"], query)
        score = lexical + max(0.0, 2.0 - float(row["rank"])) - penalty
        if score <= 0:
            continue
        hits.append(
            QueryHit(
                kind="chunk",
                identifier=row["chunk_id"],
                title=f"Chunk from {row['source_id']}",
                score=score,
                snippet=_snippet(row["text"], query),
                source_ids=[row["source_id"]],
                evidence_ids=[row["chunk_id"]],
                retrieval="fts",
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:limit]


def _search_chunks_scan(conn: sqlite3.Connection, query: str, limit: int, seen_object_ids: set[str]) -> list[QueryHit]:
    hits: list[QueryHit] = []
    for row in conn.execute("select chunk_id, source_id, text from chunks"):
        lexical = _score_text(row["text"], query)
        if lexical <= 0:
            continue
        linked = conn.execute(
            "select object_id from evidence_links where chunk_id = ? order by object_id",
            (row["chunk_id"],),
        ).fetchall()
        penalty = 1.0 if any(r[0] in seen_object_ids for r in linked) else 0.0
        score = lexical - penalty
        if score <= 0:
            continue
        hits.append(
            QueryHit(
                kind="chunk",
                identifier=row["chunk_id"],
                title=f"Chunk from {row['source_id']}",
                score=score,
                snippet=_snippet(row["text"], query),
                source_ids=[row["source_id"]],
                evidence_ids=[row["chunk_id"]],
                retrieval="scan",
            )
        )
    hits.sort(key=lambda h: (-h.score, h.title.lower()))
    return hits[:limit]


def _source_ids(conn: sqlite3.Connection, object_id: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT c.source_id
        FROM evidence_links e
        JOIN chunks c ON c.chunk_id = e.chunk_id
        WHERE e.object_id = ?
        ORDER BY c.source_id
        """,
        (object_id,),
    ).fetchall()
    return [row[0] for row in rows]


def _evidence_ids(conn: sqlite3.Connection, object_id: str) -> list[str]:
    rows = conn.execute(
        "SELECT chunk_id FROM evidence_links WHERE object_id = ? ORDER BY chunk_id",
        (object_id,),
    ).fetchall()
    return [row[0] for row in rows]


def _score_text(text: str, query: str) -> float:
    lower = text.lower()
    q = query.lower()
    terms = [term for term in re.split(r"\s+", q) if term]
    if not terms:
        return 0.0
    score = 0.0
    if q in lower:
        score += 5.0
    for term in terms:
        count = lower.count(term)
        if count:
            score += 1.5 * count
    return score


def _snippet(text: str, query: str, window: int = 180) -> str:
    lower = text.lower()
    q = query.lower()
    idx = lower.find(q)
    if idx == -1:
        terms = [term for term in re.split(r"\s+", q) if term]
        idx = min((lower.find(term) for term in terms if lower.find(term) != -1), default=0)
    start = max(0, idx - window // 2)
    end = min(len(text), idx + len(query) + window // 2)
    snippet = re.sub(r"\s+", " ", text[start:end]).strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(text):
        snippet += "..."
    return snippet


def _fts_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[A-Za-z0-9_-]+", query.lower()) if token]
    if not tokens:
        return '"' + query.replace('"', ' ') + '"'
    return " OR ".join(tokens)


def _status_bonus(status: str | None) -> float:
    bonuses = {
        "trusted": 3.0,
        "reviewed": 1.5,
        "synthesized": 0.5,
        "draft": -0.5,
        "deprecated": -3.0,
    }
    return bonuses.get(status or "", 0.0)


def _kind_rank(kind: str) -> int:
    order = {"entity": 0, "concept": 1, "synthesis": 2, "question": 3, "chunk": 4}
    return order.get(kind, 9)
