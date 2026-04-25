from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import connect
from .status import promote_object_status


@dataclass
class ReviewItem:
    review_id: str
    target_type: str
    target_id: str
    reason: str
    severity: str
    status: str
    created_at: str
    resolved_at: str | None


@dataclass
class ReviewScanResult:
    created: list[ReviewItem]
    existing_open: list[ReviewItem]
    reports: list[Path]


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def scan_reviews(settings: Settings) -> ReviewScanResult:
    created: list[ReviewItem] = []
    existing_open: list[ReviewItem] = []
    reports: list[Path] = []

    with connect(settings.database) as conn:
        candidates = []
        candidates.extend(_draft_object_candidates(conn))
        candidates.extend(_overlap_candidates(conn))

        seen_keys = set()
        for candidate in candidates:
            key = (candidate["target_type"], candidate["target_id"], candidate["reason"])
            if key in seen_keys:
                continue
            seen_keys.add(key)

            existing = conn.execute(
                """
                SELECT review_id, target_type, target_id, reason, severity, status, created_at, resolved_at
                FROM review_items
                WHERE target_type = ? AND target_id = ? AND reason = ? AND status = 'open'
                LIMIT 1
                """,
                (candidate["target_type"], candidate["target_id"], candidate["reason"]),
            ).fetchone()
            if existing:
                existing_open.append(_row_to_review_item(existing))
                continue

            review_id = f"rev_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(created)+1:03d}"
            created_at = now_iso()
            conn.execute(
                """
                INSERT INTO review_items (review_id, target_type, target_id, reason, severity, status, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, 'open', ?, NULL)
                """,
                (review_id, candidate["target_type"], candidate["target_id"], candidate["reason"], candidate["severity"], created_at),
            )
            item = ReviewItem(review_id, candidate["target_type"], candidate["target_id"], candidate["reason"], candidate["severity"], "open", created_at, None)
            created.append(item)

            if candidate.get("report"):
                report_path = _write_conflict_report(settings, review_id, candidate["report"])
                reports.append(report_path)

        conn.commit()

    _append_queue(settings, created)
    return ReviewScanResult(created=created, existing_open=existing_open, reports=reports)


def list_reviews(settings: Settings, status: str = "open") -> list[ReviewItem]:
    with connect(settings.database) as conn:
        rows = conn.execute(
            """
            SELECT review_id, target_type, target_id, reason, severity, status, created_at, resolved_at
            FROM review_items
            WHERE status = ?
            ORDER BY created_at DESC
            """,
            (status,),
        ).fetchall()
    return [_row_to_review_item(row) for row in rows]


def resolve_review(settings: Settings, review_id: str, decision: str, notes: str = "", promote: bool = False, promote_target: str | None = None) -> ReviewItem | None:
    if decision not in {"accepted", "rejected", "deferred"}:
        raise ValueError("decision must be accepted, rejected, or deferred")

    resolved_at = now_iso()
    with connect(settings.database) as conn:
        row = conn.execute(
            """
            SELECT review_id, target_type, target_id, reason, severity, status, created_at, resolved_at
            FROM review_items
            WHERE review_id = ?
            LIMIT 1
            """,
            (review_id,),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE review_items SET status = ?, resolved_at = ? WHERE review_id = ?",
            (decision, resolved_at, review_id),
        )
        conn.commit()
        item = ReviewItem(row["review_id"], row["target_type"], row["target_id"], row["reason"], row["severity"], decision, row["created_at"], resolved_at)

    if promote and decision == "accepted" and item.target_type in {"entity", "concept", "synthesis", "question", "claim"}:
        promote_object_status(settings, item.target_id, target=promote_target, reason=f"review:{item.review_id} accepted {notes}".strip())

    _append_decision(settings, item, decision, notes)
    return item


def format_review_scan(result: ReviewScanResult) -> str:
    lines = ["Review scan complete."]
    lines.append(f"Created: {len(result.created)}")
    lines.append(f"Already open: {len(result.existing_open)}")
    if result.reports:
        lines.append("Reports:")
        for path in result.reports:
            lines.append(f"- {path}")
    if result.created:
        lines.append("New review items:")
        for item in result.created:
            lines.append(f"- {item.review_id} | {item.target_type}:{item.target_id} | {item.severity} | {item.reason}")
    return "\n".join(lines)


def format_reviews(items: list[ReviewItem]) -> str:
    if not items:
        return "No review items found."
    lines = []
    for item in items:
        lines.append(
            f"{item.review_id} | {item.status} | {item.severity} | {item.target_type}:{item.target_id} | {item.reason}"
        )
    return "\n".join(lines)


def _draft_object_candidates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT object_id, type, title, status FROM objects WHERE status IN ('draft', 'synthesized') ORDER BY updated_at DESC"
    ).fetchall()
    candidates = []
    for row in rows:
        candidates.append(
            {
                "target_type": row["type"],
                "target_id": row["object_id"],
                "severity": "low" if row["status"] == "synthesized" else "medium",
                "reason": f"Object '{row['title']}' is {row['status']} and should be reviewed before being treated as trusted knowledge.",
            }
        )
    return candidates


def _overlap_candidates(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT e.chunk_id,
               GROUP_CONCAT(e.object_id, ',') AS object_ids,
               COUNT(DISTINCT e.object_id) AS object_count,
               GROUP_CONCAT(DISTINCT o.title) AS titles,
               GROUP_CONCAT(DISTINCT o.type) AS types,
               c.source_id
        FROM evidence_links e
        JOIN objects o ON o.object_id = e.object_id
        JOIN chunks c ON c.chunk_id = e.chunk_id
        GROUP BY e.chunk_id
        HAVING COUNT(DISTINCT e.object_id) >= 2
        ORDER BY object_count DESC, e.chunk_id
        """
    ).fetchall()
    candidates = []
    for row in rows:
        object_ids = row["object_ids"].split(",")
        report = {
            "chunk_id": row["chunk_id"],
            "source_id": row["source_id"],
            "object_ids": object_ids,
            "titles": row["titles"].split(",") if row["titles"] else [],
            "types": row["types"].split(",") if row["types"] else [],
            "object_count": row["object_count"],
            "reason": "Multiple objects share the same evidence chunk; review whether they are redundant, conflicting, or appropriately distinct.",
        }
        candidates.append(
            {
                "target_type": "chunk-overlap",
                "target_id": row["chunk_id"],
                "severity": "medium" if row["object_count"] <= 2 else "high",
                "reason": "Multiple knowledge objects are anchored to the same evidence chunk; review for overlap or conflict.",
                "report": report,
            }
        )
    return candidates


def _write_conflict_report(settings: Settings, review_id: str, report: dict) -> Path:
    path = settings.root / "reviews" / "conflict-reports" / f"{review_id}.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def _append_queue(settings: Settings, items: list[ReviewItem]) -> None:
    if not items:
        return
    with settings.review_queue.open("a", encoding="utf-8") as fh:
        for item in items:
            fh.write(json.dumps(item.__dict__) + "\n")


def _append_decision(settings: Settings, item: ReviewItem, decision: str, notes: str) -> None:
    path = settings.root / "reviews" / "decisions.jsonl"
    record = {
        "review_id": item.review_id,
        "target_type": item.target_type,
        "target_id": item.target_id,
        "decision": decision,
        "notes": notes,
        "created_at": item.resolved_at,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _row_to_review_item(row: sqlite3.Row) -> ReviewItem:
    return ReviewItem(
        review_id=row["review_id"],
        target_type=row["target_type"],
        target_id=row["target_id"],
        reason=row["reason"],
        severity=row["severity"],
        status=row["status"],
        created_at=row["created_at"],
        resolved_at=row["resolved_at"],
    )
