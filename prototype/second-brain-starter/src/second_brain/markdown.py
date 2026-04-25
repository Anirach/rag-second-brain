from __future__ import annotations

from pathlib import Path
import re


def read_markdown_body(path: str | Path) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if text.startswith("---\n"):
        parts = text.split("---\n", 2)
        if len(parts) == 3:
            return parts[2].strip()
    return text.strip()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "untitled"
