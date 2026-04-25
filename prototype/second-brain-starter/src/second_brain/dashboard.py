from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import Settings
from .db import connect


@dataclass
class DashboardData:
    source_count: int
    object_count: int
    review_count_open: int
    review_count_total: int
    status_counts: dict[str, int]
    type_counts: dict[str, int]
    recent_objects: list[dict]
    recent_reviews: list[dict]
    recent_status_changes: list[dict]


def build_dashboard(settings: Settings) -> DashboardData:
    with connect(settings.database) as conn:
        source_count = conn.execute('select count(*) from sources').fetchone()[0]
        object_count = conn.execute("select count(*) from objects where status != 'deprecated'").fetchone()[0]
        review_count_open = conn.execute("select count(*) from review_items where status = 'open'").fetchone()[0]
        review_count_total = conn.execute('select count(*) from review_items').fetchone()[0]

        status_counts = {row[0]: row[1] for row in conn.execute("select status, count(*) from objects group by status")}
        type_counts = {row[0]: row[1] for row in conn.execute("select type, count(*) from objects where status != 'deprecated' group by type")}

        recent_objects = [
            {
                'object_id': row['object_id'],
                'title': row['title'],
                'type': row['type'],
                'status': row['status'],
                'updated_at': row['updated_at'],
            }
            for row in conn.execute(
                "select object_id, title, type, status, updated_at from objects order by updated_at desc limit 8"
            ).fetchall()
        ]

        recent_reviews = [
            {
                'review_id': row['review_id'],
                'target_type': row['target_type'],
                'target_id': row['target_id'],
                'status': row['status'],
                'severity': row['severity'],
                'created_at': row['created_at'],
            }
            for row in conn.execute(
                'select review_id, target_type, target_id, status, severity, created_at from review_items order by created_at desc limit 8'
            ).fetchall()
        ]

    recent_status_changes = _read_recent_jsonl(settings.root / 'reviews' / 'status-history.jsonl', limit=8)
    return DashboardData(
        source_count=source_count,
        object_count=object_count,
        review_count_open=review_count_open,
        review_count_total=review_count_total,
        status_counts=status_counts,
        type_counts=type_counts,
        recent_objects=recent_objects,
        recent_reviews=recent_reviews,
        recent_status_changes=recent_status_changes,
    )


def dashboard_payload(settings: Settings) -> dict:
    return asdict(build_dashboard(settings))


def _read_recent_jsonl(path: Path, limit: int = 8) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows[-limit:][::-1]
