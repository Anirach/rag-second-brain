from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Settings
from .query import QueryHit, search


@dataclass
class CitedPoint:
    text: str
    citations: list[str]
    source_ids: list[str]
    evidence_ids: list[str]
    object_id: str | None = None


@dataclass
class AnswerBundle:
    query: str
    answer: str
    summary_points: list[CitedPoint]
    object_hits: list[QueryHit]
    chunk_hits: list[QueryHit]
    caveats: list[str]
    llm_used: bool = False


def answer_query(settings: Settings, query: str, use_llm: bool = False) -> AnswerBundle:
    hits = search(settings, query)
    object_hits = [hit for hit in hits if hit.kind != "chunk"]
    chunk_hits = [hit for hit in hits if hit.kind == "chunk"]
    summary_points = _summary_points(object_hits, chunk_hits)
    caveats = _caveats(object_hits, chunk_hits)
    deterministic_answer = _deterministic_answer(query, summary_points, caveats)

    if use_llm:
        llm_answer = _llm_answer(settings, query, object_hits, chunk_hits, caveats)
        if llm_answer:
            return AnswerBundle(
                query=query,
                answer=llm_answer,
                summary_points=summary_points,
                object_hits=object_hits,
                chunk_hits=chunk_hits,
                caveats=caveats,
                llm_used=True,
            )
        caveats = caveats + ["LLM synthesis requested, but no usable LLM configuration was available."]

    return AnswerBundle(
        query=query,
        answer=deterministic_answer,
        summary_points=summary_points,
        object_hits=object_hits,
        chunk_hits=chunk_hits,
        caveats=caveats,
        llm_used=False,
    )


def format_answer(bundle: AnswerBundle) -> str:
    lines: list[str] = []
    lines.append(f"Query: {bundle.query}")
    lines.append(f"Mode: {'llm' if bundle.llm_used else 'deterministic'}")
    lines.append("")
    lines.append("Answer")
    lines.append(bundle.answer)

    if bundle.summary_points:
        lines.append("")
        lines.append("Key points")
        for point in bundle.summary_points:
            lines.append(f"- {point.text} {_format_human_citation(point)}".rstrip())

    if bundle.object_hits:
        lines.append("")
        lines.append("Knowledge objects")
        for hit in bundle.object_hits[:5]:
            meta = []
            if hit.status:
                meta.append(f"status={hit.status}")
            if hit.source_ids:
                meta.append("sources=" + ", ".join(hit.source_ids))
            if hit.evidence_ids:
                meta.append("evidence=" + ", ".join(hit.evidence_ids[:3]))
            lines.append(f"- {hit.title} ({hit.identifier})" + (f" | {' | '.join(meta)}" if meta else ""))

    if bundle.chunk_hits:
        lines.append("")
        lines.append("Evidence")
        for hit in bundle.chunk_hits[:4]:
            lines.append(f"- {hit.identifier} from {', '.join(hit.source_ids)}: {hit.snippet}")

    if bundle.summary_points:
        lines.append("")
        lines.append("Citation map")
        for idx, point in enumerate(bundle.summary_points, start=1):
            raw = _format_raw_citations(point.citations)
            lines.append(f"- Point {idx}: {raw}")

    if bundle.caveats:
        lines.append("")
        lines.append("Caveats")
        for caveat in bundle.caveats:
            lines.append(f"- {caveat}")

    return "\n".join(lines)


def _summary_points(object_hits: list[QueryHit], chunk_hits: list[QueryHit]) -> list[CitedPoint]:
    points: list[CitedPoint] = []
    for hit in object_hits[:3]:
        sentence = _clean_snippet(hit.snippet)
        if sentence:
            citations = [hit.identifier, *hit.evidence_ids[:2], *hit.source_ids[:1]]
            points.append(
                CitedPoint(
                    text=f"{hit.title}: {sentence}",
                    citations=_dedupe(citations),
                    source_ids=_dedupe(hit.source_ids[:1]),
                    evidence_ids=_dedupe(hit.evidence_ids[:2]),
                    object_id=hit.identifier,
                )
            )
    if not points:
        for hit in chunk_hits[:2]:
            citations = [hit.identifier, *hit.source_ids[:1]]
            points.append(
                CitedPoint(
                    text=f"Evidence from {', '.join(hit.source_ids)}: {_clean_snippet(hit.snippet)}",
                    citations=_dedupe(citations),
                    source_ids=_dedupe(hit.source_ids[:1]),
                    evidence_ids=_dedupe(hit.evidence_ids[:1]),
                    object_id=None,
                )
            )
    return points


def _caveats(object_hits: list[QueryHit], chunk_hits: list[QueryHit]) -> list[str]:
    caveats: list[str] = []
    if not object_hits and not chunk_hits:
        caveats.append("No matching knowledge objects or evidence chunks were found.")
        return caveats
    if any(hit.status in {"draft", "synthesized"} for hit in object_hits):
        caveats.append("Some top knowledge objects are not yet reviewed or trusted.")
    if object_hits and all(hit.status == "trusted" for hit in object_hits):
        caveats.append("Top knowledge objects are trusted, but you should still check the cited evidence for high-stakes use.")
    if not chunk_hits:
        caveats.append("No direct evidence chunks matched strongly enough to include as support.")
    if len({src for hit in object_hits + chunk_hits for src in hit.source_ids}) <= 1:
        caveats.append("Answer is supported by a narrow source base.")
    return caveats


def _deterministic_answer(query: str, summary_points: list[CitedPoint], caveats: list[str]) -> str:
    if not summary_points:
        return f"I couldn't assemble a grounded answer for '{query}' from the current knowledge base."
    first = summary_points[0]
    answer = f"Best grounded answer for '{query}': {first.text} {_format_human_citation(first)}".rstrip()
    if len(summary_points) > 1:
        support = "; ".join(f"{point.text} {_format_human_citation(point)}".rstrip() for point in summary_points[1:3])
        answer += f" Additional support: {support}."
    if caveats:
        answer += f" Caveat: {caveats[0]}"
    return answer


def _llm_answer(settings: Settings, query: str, object_hits: list[QueryHit], chunk_hits: list[QueryHit], caveats: list[str]) -> str | None:
    cfg = settings.llm
    if not cfg or not cfg.get("enabled"):
        return None
    api_key_env = cfg.get("apiKeyEnv")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    endpoint = cfg.get("endpoint")
    model = cfg.get("model")
    if not endpoint or not model or not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are synthesizing an answer from retrieved knowledge objects and evidence chunks. "
                    "Stay grounded in the provided context. Use the supplied citation labels inline where relevant. "
                    "If evidence is weak, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": _build_llm_prompt(query, object_hits, chunk_hits, caveats),
            },
        ],
        "temperature": 0.2,
    }

    req = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    return _extract_openai_chat_text(data)


def _build_llm_prompt(query: str, object_hits: list[QueryHit], chunk_hits: list[QueryHit], caveats: list[str]) -> str:
    lines = [f"Query: {query}", "", "Knowledge objects:"]
    for hit in object_hits[:5]:
        lines.append(f"- {hit.title} ({hit.kind}, status={hit.status}, score={hit.score:.2f}) {_format_prompt_citation(hit)}")
        lines.append(f"  sources: {', '.join(hit.source_ids) or '(none)'}")
        lines.append(f"  evidence: {', '.join(hit.evidence_ids) or '(none)'}")
        lines.append(f"  snippet: {_clean_snippet(hit.snippet)}")
    lines.append("")
    lines.append("Evidence chunks:")
    for hit in chunk_hits[:5]:
        lines.append(f"- {hit.identifier} from {', '.join(hit.source_ids)} [source: {', '.join(hit.source_ids[:1]) or 'unknown'}; evidence: {hit.identifier}]: {_clean_snippet(hit.snippet)}")
    if caveats:
        lines.append("")
        lines.append("Caveats:")
        for caveat in caveats:
            lines.append(f"- {caveat}")
    lines.append("")
    lines.append("Write a concise answer grounded only in this context. Keep human-readable citations inline, e.g. [source: src_x; evidence: chk_y; object: cpt_z].")
    return "\n".join(lines)


def _extract_openai_chat_text(data: dict[str, Any]) -> str | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content.strip()
    return None


def _clean_snippet(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned.rstrip(". ")


def _format_human_citation(point: CitedPoint) -> str:
    parts: list[str] = []
    if point.source_ids:
        parts.append(f"source: {', '.join(point.source_ids)}")
    if point.evidence_ids:
        parts.append(f"evidence: {', '.join(point.evidence_ids)}")
    if point.object_id:
        parts.append(f"object: {point.object_id}")
    return f"[{' ; '.join(parts)}]" if parts else ""


def _format_prompt_citation(hit: QueryHit) -> str:
    parts: list[str] = []
    if hit.source_ids:
        parts.append(f"source: {', '.join(hit.source_ids[:1])}")
    if hit.evidence_ids:
        parts.append(f"evidence: {', '.join(hit.evidence_ids[:2])}")
    parts.append(f"object: {hit.identifier}")
    return f"[{' ; '.join(parts)}]"


def _format_raw_citations(citations: list[str]) -> str:
    citations = _dedupe([c for c in citations if c])
    return f"[{' ; '.join(citations)}]" if citations else ""


def _dedupe(items: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out
