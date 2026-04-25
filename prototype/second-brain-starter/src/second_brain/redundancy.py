from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from .config import Settings
from .db import connect
from .fts import refresh_fts
from .status import set_object_status


@dataclass
class RedundancyCandidate:
    source_object_id: str
    target_object_id: str
    object_type: str
    reason: str
    score: float


def scan_redundancy(settings: Settings) -> list[RedundancyCandidate]:
    with connect(settings.database) as conn:
        rows = conn.execute(
            "SELECT object_id, type, title, status FROM objects WHERE status != 'deprecated' ORDER BY type, title"
        ).fetchall()
        evidence = {
            row[0]: {r[0] for r in conn.execute('SELECT chunk_id FROM evidence_links WHERE object_id = ?', (row[0],)).fetchall()}
            for row in rows
        }
        sources = {
            row[0]: {r[0] for r in conn.execute("""
                SELECT DISTINCT c.source_id
                FROM evidence_links e
                JOIN chunks c ON c.chunk_id = e.chunk_id
                WHERE e.object_id = ?
            """, (row[0],)).fetchall()}
            for row in rows
        }

    candidates: list[RedundancyCandidate] = []
    by_type: dict[str, list[sqlite3.Row]] = {}
    for row in rows:
        by_type.setdefault(row['type'], []).append(row)

    for object_type, items in by_type.items():
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                a = items[i]
                b = items[j]
                score, reasons = _redundancy_score(
                    a['title'],
                    b['title'],
                    object_type,
                    evidence.get(a['object_id'], set()),
                    evidence.get(b['object_id'], set()),
                    sources.get(a['object_id'], set()),
                    sources.get(b['object_id'], set()),
                )
                if score <= 0 or not reasons:
                    continue
                keep, merge = _choose_target(a, b)
                candidates.append(RedundancyCandidate(merge['object_id'], keep['object_id'], object_type, ' + '.join(reasons), score))

    candidates.sort(key=lambda c: (-c.score, c.object_type, c.source_object_id, c.target_object_id))
    return candidates


def merge_objects(settings: Settings, source_object_id: str, target_object_id: str, reason: str = '') -> RedundancyCandidate:
    if source_object_id == target_object_id:
        raise ValueError('source and target must be different')

    with connect(settings.database) as conn:
        source = conn.execute('SELECT object_id, type, title, path, status FROM objects WHERE object_id = ?', (source_object_id,)).fetchone()
        target = conn.execute('SELECT object_id, type, title, path, status FROM objects WHERE object_id = ?', (target_object_id,)).fetchone()
        if not source or not target:
            raise ValueError('source or target object not found')
        if source['type'] != target['type']:
            raise ValueError('merge currently requires matching object types')

        # Move evidence links to target
        for row in conn.execute('SELECT chunk_id, relation FROM evidence_links WHERE object_id = ?', (source_object_id,)).fetchall():
            conn.execute(
                'INSERT OR IGNORE INTO evidence_links (object_id, chunk_id, relation) VALUES (?, ?, ?)',
                (target_object_id, row['chunk_id'], row['relation']),
            )
        conn.execute('DELETE FROM evidence_links WHERE object_id = ?', (source_object_id,))

        # Preserve link provenance
        conn.execute(
            'INSERT OR IGNORE INTO object_links (from_object_id, to_object_id, relation) VALUES (?, ?, ?)',
            (source_object_id, target_object_id, 'merged_into'),
        )

        # Rewire object links
        conn.execute('UPDATE OR IGNORE object_links SET from_object_id = ? WHERE from_object_id = ?', (target_object_id, source_object_id))
        conn.execute('UPDATE OR IGNORE object_links SET to_object_id = ? WHERE to_object_id = ?', (target_object_id, source_object_id))

        # Point open reviews at the winner when appropriate
        conn.execute(
            "UPDATE review_items SET target_id = ? WHERE target_id = ? AND target_type = ? AND status = 'open'",
            (target_object_id, source_object_id, source['type']),
        )
        conn.commit()

    set_object_status(settings, source_object_id, 'deprecated', reason=f'merged into {target_object_id} {reason}'.strip())
    _mark_markdown_merged(settings, source_object_id, target_object_id)
    refresh_fts(settings)
    return RedundancyCandidate(source_object_id, target_object_id, source['type'], reason or 'manual merge', 1.0)


def format_redundancy_candidates(items: list[RedundancyCandidate]) -> str:
    if not items:
        return 'No likely redundant objects found.'
    return '\n'.join(
        f"{item.object_type} | merge {item.source_object_id} -> {item.target_object_id} | score={item.score:.2f} | {item.reason}"
        for item in items
    )


def _normalized_title(title: str, object_type: str) -> str:
    value = re.sub(r'\s+', ' ', title).strip().lower()
    value = re.sub(r'^(the|a|an)\s+', '', value)
    value = re.sub(r'[^a-z0-9 ]+', ' ', value)
    if object_type == 'synthesis':
        value = re.sub(r'\s+overview$', '', value)
    return re.sub(r'\s+', ' ', value).strip()


def _title_tokens(title: str, object_type: str) -> set[str]:
    return {tok for tok in _normalized_title(title, object_type).split() if tok and tok not in {'overview'}}


def _redundancy_score(
    title_a: str,
    title_b: str,
    object_type: str,
    evidence_a: set[str],
    evidence_b: set[str],
    sources_a: set[str],
    sources_b: set[str],
) -> tuple[float, list[str]]:
    norm_a = _normalized_title(title_a, object_type)
    norm_b = _normalized_title(title_b, object_type)
    token_a = _title_tokens(title_a, object_type)
    token_b = _title_tokens(title_b, object_type)
    title_similarity = SequenceMatcher(None, norm_a, norm_b).ratio() if norm_a and norm_b else 0.0
    token_overlap = _jaccard(token_a, token_b)
    evidence_overlap = _jaccard(evidence_a, evidence_b)
    source_overlap = _jaccard(sources_a, sources_b)

    reasons: list[str] = []
    score = 0.0

    if norm_a == norm_b and norm_a:
        reasons.append('same normalized title')
        score += 1.0
    elif title_similarity >= 0.96:
        reasons.append(f'very similar title ({title_similarity:.2f})')
        score += 0.9
    elif token_overlap >= 0.8 and (source_overlap > 0 or evidence_overlap > 0):
        reasons.append(f'high token overlap ({token_overlap:.2f})')
        score += 0.75
    elif object_type == 'synthesis' and token_overlap >= 0.67 and source_overlap >= 0.5:
        reasons.append(f'similar synthesis topic ({token_overlap:.2f})')
        score += 0.65

    if evidence_overlap > 0:
        reasons.append('shared evidence')
        score += 0.5 * evidence_overlap
    elif source_overlap >= 0.5 and token_overlap >= 0.5:
        reasons.append('same source base')
        score += 0.25 * source_overlap

    # Stay conservative: title similarity alone below this threshold is not enough.
    if score < 1.0:
        return 0.0, []
    return score, reasons


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _choose_target(a: sqlite3.Row, b: sqlite3.Row) -> tuple[sqlite3.Row, sqlite3.Row]:
    rank = {'trusted': 4, 'reviewed': 3, 'synthesized': 2, 'draft': 1, 'deprecated': 0}
    ra = rank.get(a['status'], 0)
    rb = rank.get(b['status'], 0)
    if ra != rb:
        return (a, b) if ra > rb else (b, a)
    # prefer shorter cleaner title, then stable lexical order
    if len(a['title']) != len(b['title']):
        return (a, b) if len(a['title']) < len(b['title']) else (b, a)
    return (a, b) if a['object_id'] < b['object_id'] else (b, a)


def _mark_markdown_merged(settings: Settings, source_object_id: str, target_object_id: str) -> None:
    with connect(settings.database) as conn:
        row = conn.execute('SELECT path FROM objects WHERE object_id = ?', (source_object_id,)).fetchone()
    if not row:
        return
    path = settings.root / row['path']
    if not path.exists():
        return
    text = path.read_text(encoding='utf-8')
    if text.startswith('---\n'):
        parts = text.split('---\n', 2)
        if len(parts) == 3:
            fm, body = parts[1], parts[2]
            if 'merged_into:' not in fm:
                fm += f'merged_into: {target_object_id}\n'
            body = body.rstrip() + f"\n\n> Deprecated: merged into `{target_object_id}`.\n"
            path.write_text('---\n' + fm + '---\n' + body, encoding='utf-8')
            return
    path.write_text(text.rstrip() + f"\n\n> Deprecated: merged into `{target_object_id}`.\n", encoding='utf-8')
