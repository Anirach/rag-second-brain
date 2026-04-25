from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import Settings
from .db import connect
from .markdown import slugify
from .fts import refresh_fts
from .extractors import copy_original, extract_document
from .llm_extract import maybe_llm_extract


@dataclass
class IngestResult:
    source_id: str
    chunk_ids: list[str]
    entity_ids: list[str]
    concept_id: str
    synthesis_id: str


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def ingest_file(
    settings: Settings,
    file_path: str | Path,
    title: str | None = None,
    origin: str = "cli-file",
) -> IngestResult:
    extracted = extract_document(file_path, title=title)
    result = ingest_text(
        settings,
        title=extracted.title,
        text=extracted.text,
        source_type=extracted.source_type,
        origin=origin,
        original_path=str(Path(file_path).resolve()),
        raw_filename=Path(file_path).name,
    )
    source_dir = settings.raw_root / result.source_id
    copy_original(file_path, source_dir)
    return result


def make_source_id(text: str) -> str:
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
    return f"src_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{digest}"


def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", normalized)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                chunks.append(current)
            current = sentence
    if current:
        chunks.append(current)
    return chunks


def extract_entities(title: str, text: str) -> list[str]:
    title_norm = _normalize_entity_candidate(title)
    title_words = title_norm.split()
    candidates: list[str] = []

    # Keep short, meaningful one-word titles as possible entities.
    if len(title_words) == 1 and len(title_norm) >= 3 and title_norm.lower() not in {"note", "example", "document"}:
        candidates.append(title_norm)

    # Acronyms like RAG, PDF, DOCX
    for match in re.findall(r"\b[A-Z][A-Z0-9]{1,9}\b", text):
        candidates.append(match)

    # Multi-word/title-case phrases, line-safe and punctuation-trimmed
    phrase_pattern = re.compile(
        r"\b(?:[A-Z][a-z0-9]+(?:[-/][A-Z]?[a-z0-9]+)?|[A-Z]{2,})(?:\s+(?:[A-Z][a-z0-9]+(?:[-/][A-Z]?[a-z0-9]+)?|[A-Z]{2,})){0,3}\b"
    )
    flattened = text.replace("\n", " ")
    for match in phrase_pattern.findall(flattened):
        candidates.append(match)

    cleaned: list[str] = []
    seen_norm: set[str] = set()
    for candidate in candidates:
        normalized = _normalize_entity_candidate(candidate)
        if not normalized:
            continue
        norm_key = normalized.lower()
        if norm_key in seen_norm:
            continue
        if len(title_words) != 1 and norm_key == title_norm.lower():
            continue
        if _is_bad_entity_candidate(normalized, flattened, title_norm):
            continue
        seen_norm.add(norm_key)
        cleaned.append(normalized)

    cleaned.sort(key=lambda value: (_entity_priority(value), value.lower()))
    return cleaned[:8]


def _normalize_entity_candidate(value: str) -> str:
    value = re.sub(r"\s+", " ", value).strip(" \n\t.,;:!?-–—_()[]{}\"'")
    value = re.sub(r"\s*(?:[:;,.!?])\s*$", "", value)
    value = re.sub(r"^(?:The|A|An)\s+", "", value)
    return value.strip()


def _is_bad_entity_candidate(value: str, text: str, title_norm: str) -> bool:
    lower = value.lower()
    words = value.split()
    if len(value) < 2:
        return True
    if len(words) > 4:
        return True
    if any(token in {"this", "that", "these", "those", "should", "note", "example", "another"} for token in lower.split()):
        return True
    if re.search(r"\b(is|are|was|were|should|remain|discusses|combines)\b", lower):
        return True
    if value.count(" ") >= 2 and any(token.isupper() and len(token) <= 4 for token in value.split()[:-1]):
        return True
    if re.search(r"[^A-Za-z0-9\-/ ]", value):
        return True
    if len(words) == 1 and not value.isupper():
        occurrences = len(re.findall(rf"\b{re.escape(value)}\b", text))
        if lower != title_norm.lower() and occurrences < 2:
            return True
        if lower in {"persistent", "clinical", "source", "answer", "workflow", "knowledge"}:
            return True
    return False


def _entity_priority(value: str) -> tuple[int, int]:
    # prefer useful acronyms and short proper names over long phrases
    words = value.split()
    is_acronym = int(not value.isupper())
    return (is_acronym, len(words))


def summarize(text: str, max_sentences: int = 2) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", text).strip())
    return " ".join([s for s in sentences if s][:max_sentences]).strip()


def ingest_text(
    settings: Settings,
    title: str,
    text: str,
    source_type: str = "note",
    origin: str = "cli",
    original_path: str | None = None,
    raw_filename: str = "source.txt",
) -> IngestResult:
    text = text.strip()
    if not text:
        raise ValueError("text is empty")

    source_id = make_source_id(text)
    checksum = f"sha256:{hashlib.sha256(text.encode('utf-8')).hexdigest()}"
    timestamp = now_iso()
    source_dir = settings.raw_root / source_id
    source_dir.mkdir(parents=True, exist_ok=True)
    (source_dir / "source.txt").write_text(text + "\n", encoding="utf-8")
    metadata = {
        "source_id": source_id,
        "title": title,
        "source_type": source_type,
        "origin": origin,
        "original_path": original_path,
        "normalized_text_path": "source.txt",
        "checksum": checksum,
        "created_at": timestamp,
        "updated_at": timestamp,
        "language": settings.default_output_language,
        "authors": [],
        "url": None,
        "tags": [],
    }
    (source_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    chunks = chunk_text(text)
    if not chunks:
        chunks = [text]
    chunk_ids: list[str] = []
    chunk_rows: list[dict] = []
    cursor = 0
    for idx, chunk in enumerate(chunks, start=1):
        start = text.find(chunk[:20].strip(), cursor)
        if start < 0:
            start = cursor
        end = start + len(chunk)
        cursor = end
        chunk_id = f"chk_{source_id}_{idx:04d}"
        chunk_ids.append(chunk_id)
        chunk_rows.append({
            "chunk_id": chunk_id,
            "source_id": source_id,
            "section": f"Chunk {idx}",
            "text": chunk,
            "char_start": start,
            "char_end": end,
            "page_start": 1,
            "page_end": 1,
            "embedding_ref": None,
            "checksum": f"sha256:{hashlib.sha256(chunk.encode('utf-8')).hexdigest()}",
        })
    chunk_path = settings.extract_root / "chunks" / f"{source_id}.jsonl"
    chunk_path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in chunk_rows) + "\n", encoding="utf-8")

    llm_extraction = maybe_llm_extract(settings, title, text)
    entity_names = llm_extraction.entities if llm_extraction and llm_extraction.entities else extract_entities(title, text)
    entity_records = []
    entity_specs: list[tuple[str, str, Path]] = []
    entity_ids: list[str] = []

    summary = llm_extraction.summary if llm_extraction and llm_extraction.summary else summarize(text)
    with connect(settings.database) as conn:
        concept_slug = _ensure_unique_slug(conn, slugify(title), "concept", allow_existing=False)
        concept_id = f"cpt_{concept_slug}"
        concept_path = settings.knowledge_root / "concepts" / f"{concept_slug}.md"

        for name in entity_names:
            base_slug = slugify(name)
            entity_slug = _ensure_unique_slug(conn, base_slug, "entity", allow_existing=True)
            entity_id = f"ent_{entity_slug}"
            entity_ids.append(entity_id)
            entity_specs.append((entity_id, entity_slug, settings.knowledge_root / "entities" / f"{entity_slug}.md"))
            entity_records.append({
                "entity_id": entity_id,
                "name": name,
                "type": "entity",
                "aliases": [],
                "source_id": source_id,
                "evidence": [chunk_ids[0]],
                "confidence": 0.6,
            })

        synthesis_slug = _ensure_unique_slug(conn, f"{concept_slug}-overview", "synthesis", allow_existing=False)
        synthesis_id = f"syn_{synthesis_slug}"
        synthesis_path = settings.knowledge_root / "synthesis" / f"{synthesis_slug}.md"

    for (entity_id, entity_slug, entity_md), name in zip(entity_specs, entity_names):
        if not entity_md.exists():
            entity_md.write_text(
                f"---\nobject_id: {entity_id}\ntype: entity\ntitle: {name}\nstatus: synthesized\nconfidence: 0.6\nsource_ids:\n  - {source_id}\nevidence_ids:\n  - {chunk_ids[0]}\nupdated_at: {timestamp}\n---\n\n# {name}\n\nAuto-generated entity page from ingested source.\n\n## Evidence\n- [{chunk_ids[0]}]\n",
                encoding="utf-8",
            )
    (settings.extract_root / "entities" / f"{source_id}.json").write_text(json.dumps(entity_records, indent=2), encoding="utf-8")

    concept_path = settings.knowledge_root / "concepts" / f"{concept_slug}.md"
    concept_path.write_text(
        f"---\nobject_id: {concept_id}\ntype: concept\ntitle: {title}\nstatus: synthesized\nconfidence: 0.7\nsource_ids:\n  - {source_id}\nevidence_ids:\n  - {chunk_ids[0]}\nupdated_at: {timestamp}\n---\n\n# {title}\n\n{summary}\n\n## Source\n- {source_id}\n",
        encoding="utf-8",
    )

    synthesis_path.write_text(
        f"---\nobject_id: {synthesis_id}\ntype: synthesis\ntitle: {title} Overview\nstatus: draft\nsource_ids:\n  - {source_id}\nupdated_at: {timestamp}\n---\n\n# {title} Overview\n\n{summary}\n\n## Related entities\n" + "\n".join(f"- [[{slug}]]" for _, slug, _ in entity_specs[:5]) + "\n",
        encoding="utf-8",
    )

    (settings.extract_root / "summaries" / f"{source_id}.md").write_text(f"# Source Summary\n\n{summary}\n", encoding="utf-8")
    (settings.extract_root / "claims" / f"{source_id}.json").write_text("[]\n", encoding="utf-8")
    (settings.extract_root / "relations" / f"{source_id}.json").write_text("[]\n", encoding="utf-8")

    with connect(settings.database) as conn:
        _insert_source(conn, metadata)
        for row in chunk_rows:
            conn.execute(
                """
                INSERT INTO chunks (chunk_id, source_id, section, text, char_start, char_end, page_start, page_end, checksum)
                VALUES (:chunk_id, :source_id, :section, :text, :char_start, :char_end, :page_start, :page_end, :checksum)
                """,
                row,
            )
        _upsert_object(conn, concept_id, "concept", title, concept_slug, "synthesized", 0.7, concept_path, timestamp)
        _upsert_object(conn, synthesis_id, "synthesis", f"{title} Overview", synthesis_slug, "draft", None, synthesis_path, timestamp)
        conn.execute(
            "INSERT OR IGNORE INTO evidence_links (object_id, chunk_id, relation) VALUES (?, ?, 'supports')",
            (concept_id, chunk_ids[0]),
        )
        for (entity_id, entity_slug, entity_path), name in zip(entity_specs, entity_names):
            _upsert_object(conn, entity_id, "entity", name, entity_slug, "synthesized", 0.6, entity_path, timestamp)
            conn.execute(
                "INSERT OR IGNORE INTO evidence_links (object_id, chunk_id, relation) VALUES (?, ?, 'supports')",
                (entity_id, chunk_ids[0]),
            )
        conn.commit()

    _update_index(settings)
    refresh_fts(settings)
    return IngestResult(source_id, chunk_ids, entity_ids, concept_id, synthesis_id)


def _ensure_unique_slug(conn: sqlite3.Connection, preferred_slug: str, object_type: str, allow_existing: bool = True) -> str:
    slug = preferred_slug
    suffix = 2
    while True:
        row = conn.execute("SELECT object_id, type FROM objects WHERE slug = ?", (slug,)).fetchone()
        if row is None:
            return slug
        if allow_existing and row["type"] == object_type:
            return slug
        slug = f"{preferred_slug}-{object_type}-{suffix}"
        suffix += 1


def _insert_source(conn: sqlite3.Connection, metadata: dict) -> None:
    conn.execute(
        """
        INSERT INTO sources (source_id, title, source_type, origin, url, checksum, language, created_at, updated_at, metadata_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            metadata["source_id"],
            metadata["title"],
            metadata["source_type"],
            metadata["origin"],
            metadata.get("url"),
            metadata["checksum"],
            metadata["language"],
            metadata["created_at"],
            metadata["updated_at"],
            json.dumps(metadata),
        ),
    )


def _upsert_object(conn: sqlite3.Connection, object_id: str, object_type: str, title: str, slug: str, status: str, confidence: float | None, path: Path, timestamp: str) -> None:
    conn.execute(
        """
        INSERT INTO objects (object_id, type, title, slug, status, confidence, path, created_at, updated_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(object_id) DO UPDATE SET
          type=excluded.type,
          title=excluded.title,
          slug=excluded.slug,
          status=excluded.status,
          confidence=excluded.confidence,
          path=excluded.path,
          updated_at=excluded.updated_at
        """,
        (object_id, object_type, title, slug, status, confidence, str(path.relative_to(path.parents[2])), timestamp, timestamp),
    )


def _update_index(settings: Settings) -> None:
    sections = {
        "Entities": settings.knowledge_root / "entities",
        "Concepts": settings.knowledge_root / "concepts",
        "Synthesis": settings.knowledge_root / "synthesis",
        "Questions": settings.knowledge_root / "questions",
    }
    lines = ["# Knowledge Index", ""]
    for label, folder in sections.items():
        lines.append(f"## {label}")
        files = sorted(folder.glob("*.md"))
        if not files:
            lines.append("- (none)")
        else:
            for file in files:
                lines.append(f"- [[{file.stem}]]")
        lines.append("")
    (settings.knowledge_root / "indexes" / "index.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
