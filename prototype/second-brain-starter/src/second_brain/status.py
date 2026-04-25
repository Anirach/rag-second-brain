from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .db import connect

VALID_STATUSES = {"draft", "synthesized", "reviewed", "trusted", "deprecated"}
PROMOTION_ORDER = ["draft", "synthesized", "reviewed", "trusted"]


@dataclass
class ObjectStatus:
    object_id: str
    object_type: str
    title: str
    status: str
    path: str
    updated_at: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def get_object_status(settings: Settings, object_id: str) -> ObjectStatus | None:
    with connect(settings.database) as conn:
        row = conn.execute(
            "SELECT object_id, type, title, status, path, updated_at FROM objects WHERE object_id = ? LIMIT 1",
            (object_id,),
        ).fetchone()
    if not row:
        return None
    return ObjectStatus(row["object_id"], row["type"], row["title"], row["status"], row["path"], row["updated_at"])


def list_object_statuses(settings: Settings, status: str | None = None) -> list[ObjectStatus]:
    sql = "SELECT object_id, type, title, status, path, updated_at FROM objects"
    params: tuple[str, ...] = ()
    if status:
        sql += " WHERE status = ?"
        params = (status,)
    sql += " ORDER BY updated_at DESC, object_id"
    with connect(settings.database) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [ObjectStatus(r["object_id"], r["type"], r["title"], r["status"], r["path"], r["updated_at"]) for r in rows]


def set_object_status(settings: Settings, object_id: str, new_status: str, reason: str = "") -> ObjectStatus | None:
    if new_status not in VALID_STATUSES:
        raise ValueError(f"invalid status: {new_status}")
    timestamp = now_iso()
    with connect(settings.database) as conn:
        row = conn.execute(
            "SELECT object_id, type, title, status, path FROM objects WHERE object_id = ? LIMIT 1",
            (object_id,),
        ).fetchone()
        if not row:
            return None
        conn.execute(
            "UPDATE objects SET status = ?, updated_at = ? WHERE object_id = ?",
            (new_status, timestamp, object_id),
        )
        conn.commit()
    status_obj = ObjectStatus(row["object_id"], row["type"], row["title"], new_status, row["path"], timestamp)
    _sync_markdown_status(settings, status_obj)
    _append_status_history(settings, status_obj, reason)
    return status_obj


def promote_object_status(settings: Settings, object_id: str, target: str | None = None, reason: str = "") -> ObjectStatus | None:
    current = get_object_status(settings, object_id)
    if not current:
        return None
    if target is None:
        try:
            idx = PROMOTION_ORDER.index(current.status)
        except ValueError:
            idx = -1
        if idx == -1 or idx >= len(PROMOTION_ORDER) - 1:
            target = current.status
        else:
            target = PROMOTION_ORDER[idx + 1]
    return set_object_status(settings, object_id, target, reason=reason)


def format_object_statuses(items: list[ObjectStatus]) -> str:
    if not items:
        return "No objects found."
    return "\n".join(
        f"{item.object_id} | {item.status} | {item.object_type} | {item.title} | {item.path}" for item in items
    )


def _sync_markdown_status(settings: Settings, item: ObjectStatus) -> None:
    path = settings.root / item.path
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    if text.startswith("---\n"):
        parts = text.split("---\n", 2)
        if len(parts) == 3:
            frontmatter = parts[1]
            body = parts[2]
            frontmatter, count = _replace_frontmatter_field(frontmatter, "status", item.status)
            frontmatter, _ = _replace_frontmatter_field(frontmatter, "updated_at", item.updated_at, add_if_missing=True)
            path.write_text("---\n" + frontmatter + "---\n" + body, encoding="utf-8")
            return
    # fallback if no frontmatter
    path.write_text(f"---\nstatus: {item.status}\nupdated_at: {item.updated_at}\n---\n\n{text}", encoding="utf-8")


def _replace_frontmatter_field(frontmatter: str, field: str, value: str, add_if_missing: bool = False) -> tuple[str, int]:
    lines = frontmatter.splitlines()
    replaced = 0
    for idx, line in enumerate(lines):
        if line.startswith(f"{field}:"):
            lines[idx] = f"{field}: {value}"
            replaced += 1
            break
    if replaced == 0 and add_if_missing:
        lines.append(f"{field}: {value}")
    return "\n".join(lines) + ("\n" if lines else ""), replaced


def _append_status_history(settings: Settings, item: ObjectStatus, reason: str) -> None:
    path = settings.root / "reviews" / "status-history.jsonl"
    record = {
        "object_id": item.object_id,
        "status": item.status,
        "reason": reason,
        "updated_at": item.updated_at,
    }
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")
