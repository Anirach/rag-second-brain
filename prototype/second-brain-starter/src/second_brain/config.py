from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root: Path
    database: Path
    knowledge_root: Path
    raw_root: Path
    extract_root: Path
    review_queue: Path
    default_output_language: str
    retrieval: dict
    llm: dict


def load_settings(root: str | Path) -> Settings:
    root_path = Path(root).resolve()
    settings = json.loads((root_path / "config" / "settings.json").read_text())
    return Settings(
        root=root_path,
        database=root_path / settings["database"],
        knowledge_root=root_path / settings["knowledgeRoot"],
        raw_root=root_path / settings["rawRoot"],
        extract_root=root_path / settings["extractRoot"],
        review_queue=root_path / settings["reviewQueue"],
        default_output_language=settings.get("defaultOutputLanguage", "en"),
        retrieval=settings.get("retrieval", {}),
        llm=settings.get("llm", {}),
    )
