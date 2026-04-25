from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .config import Settings


@dataclass
class LlmExtraction:
    entities: list[str]
    summary: str | None = None


def maybe_llm_extract(settings: Settings, title: str, text: str) -> LlmExtraction | None:
    cfg = settings.llm
    if not cfg or not cfg.get("enabled") or not cfg.get("extractionEnabled"):
        return None
    api_key_env = cfg.get("apiKeyEnv")
    api_key = os.environ.get(api_key_env) if api_key_env else None
    endpoint = cfg.get("endpoint")
    model = cfg.get("extractionModel") or cfg.get("model")
    if not endpoint or not model or not api_key:
        return None

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Extract concise entities and a short summary from the source. "
                    "Return strict JSON with keys: entities (array of short strings) and summary (string). "
                    "Entities should be canonical, short, and non-redundant."
                ),
            },
            {
                "role": "user",
                "content": _build_prompt(title, text),
            },
        ],
        "temperature": 0.1,
        "response_format": {"type": "json_object"},
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
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    parsed = _extract_json_content(data)
    if not parsed:
        return None

    entities = parsed.get("entities")
    summary = parsed.get("summary")
    if not isinstance(entities, list):
        return None
    clean_entities = []
    seen = set()
    for item in entities:
        if not isinstance(item, str):
            continue
        value = re.sub(r"\s+", " ", item).strip()
        if not value:
            continue
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        clean_entities.append(value)
    clean_summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
    return LlmExtraction(entities=clean_entities[:8], summary=clean_summary)


def _build_prompt(title: str, text: str) -> str:
    excerpt = text[:5000]
    return (
        f"Title: {title}\n\n"
        f"Source text:\n{excerpt}\n\n"
        "Return JSON only. Example: {\"entities\": [\"Transformer\", \"RAG\"], \"summary\": \"...\"}"
    )


def _extract_json_content(data: dict[str, Any]) -> dict[str, Any] | None:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    message = choices[0].get("message")
    if not isinstance(message, dict):
        return None
    content = message.get("content")
    if isinstance(content, str):
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                texts.append(part["text"])
        joined = "".join(texts).strip()
        if joined:
            try:
                return json.loads(joined)
            except json.JSONDecodeError:
                return None
    return None
