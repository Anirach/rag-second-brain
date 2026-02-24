#!/usr/bin/env python3
"""
weekly_synthesis.py — Aggregate weekly paper summaries and generate research synthesis.

Collects all paper notes from the current (or specified) week, identifies
trends, surfaces research opportunities, and saves a synthesis note to the
Obsidian vault.

Usage:
    python3 weekly_synthesis.py                  # Current ISO week
    python3 weekly_synthesis.py --week 2026-W08  # Specific week
    python3 weekly_synthesis.py --days 7         # Last N days
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [weekly_synthesis] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
RESEARCH_DIR = Path(__file__).parent
VAULT_ROOT = Path("/home/clawdbot/obsidian-vault")
PAPERS_DIR = RESEARCH_DIR / "papers"
NOTES_DIR = VAULT_ROOT / "KnowledgeGraph/Papers"
DOCS_DIR = VAULT_ROOT / "KnowledgeGraph/Documents"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# ---------------------------------------------------------------------------
# Research gap / opportunity definitions
# ---------------------------------------------------------------------------
RESEARCH_GAPS: List[Dict] = [
    {
        "gap": "Thai-language RAG systems",
        "description": "RAG systems optimised for Thai language / low-resource settings",
        "keywords": ["thai", "low-resource", "multilingual", "rag", "retrieval"],
        "relevance": "RAG Second Brain + KMUTNB context",
    },
    {
        "gap": "Longevity biomarkers + AI interpretation",
        "description": "AI models that interpret multi-omics aging biomarkers",
        "keywords": ["biomarker", "aging", "longevity", "omics", "deep learning"],
        "relevance": "Longevity Healthcare research direction",
    },
    {
        "gap": "Knowledge Graph + LLM fusion for EHR",
        "description": "Combining KGs with LLMs for clinical decision support",
        "keywords": ["knowledge graph", "llm", "ehr", "clinical", "decision support"],
        "relevance": "RAG Second Brain / NCD-CIE intersection",
    },
    {
        "gap": "Adaptive learning personalisation at scale",
        "description": "LLM-driven personalised tutoring with real-time feedback",
        "keywords": ["adaptive", "personalized", "tutoring", "education", "llm"],
        "relevance": "Education Technology stream",
    },
    {
        "gap": "NCD risk stratification with graph methods",
        "description": "Graph neural networks for NCD risk prediction from EHR data",
        "keywords": ["ncd", "graph", "risk", "stratification", "ehr", "classification"],
        "relevance": "NCD-CIE project",
    },
]

# Papers to watch for citation opportunities
CITATION_TARGETS: Dict[str, List[str]] = {
    "RAG Second Brain": [
        "rag", "retrieval augmented", "second brain", "knowledge graph", "pkm",
        "personal knowledge", "multi-graph", "llm knowledge",
    ],
    "NCD-CIE": [
        "non-communicable disease", "ncd", "icd", "clinical coding", "causal",
        "cie", "disease classification", "health informatics",
    ],
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------
def get_week_dates(week_str: Optional[str] = None, days_back: int = 7) -> Tuple[date, date]:
    """
    Return (start_date, end_date) for the target week.

    Args:
        week_str: ISO week string like "2026-W08". If None, uses current week.
        days_back: Alternative: last N days from today.
    """
    if week_str:
        # Parse "YYYY-WNN"
        year, week_num = week_str.split("-W")
        start = datetime.strptime(f"{year} {week_num} 1", "%Y %W %w").date()
        end = start + timedelta(days=6)
    else:
        today = date.today()
        start = today - timedelta(days=days_back - 1)
        end = today
    return start, end


def load_papers_for_week(start_date: date, end_date: date) -> List[Dict]:
    """Load all paper JSON files covering the target week."""
    all_papers: List[Dict] = []
    seen_ids: set = set()

    for json_file in sorted(PAPERS_DIR.glob("papers_*.json")):
        try:
            # Extract date from filename: papers_YYYYMMDD_HHMMSS.json
            parts = json_file.stem.split("_")
            if len(parts) >= 2:
                file_date = datetime.strptime(parts[1], "%Y%m%d").date()
                if start_date <= file_date <= end_date:
                    with open(json_file) as fh:
                        data = json.load(fh)
                    papers = data.get("relevant_papers", data) if isinstance(data, dict) else data
                    for p in papers:
                        pid = p.get("arxiv_id", p.get("id", p.get("title", "")))
                        if pid and pid not in seen_ids:
                            seen_ids.add(pid)
                            p["_source_date"] = file_date.isoformat()
                            all_papers.append(p)
        except Exception as exc:
            log.warning(f"Could not parse {json_file.name}: {exc}")

    log.info(f"Loaded {len(all_papers)} unique papers for {start_date} → {end_date}")
    return all_papers


def load_existing_notes(start_date: date, end_date: date) -> List[Dict]:
    """Load existing Obsidian paper notes from the week (parse frontmatter)."""
    notes = []
    if not NOTES_DIR.exists():
        return notes

    for md_file in NOTES_DIR.glob("*.md"):
        try:
            # Filename format: YYYY-MM-DD-{short-title}.md
            parts = md_file.stem.split("-", 3)
            if len(parts) >= 3:
                file_date = date(int(parts[0]), int(parts[1]), int(parts[2]))
                if start_date <= file_date <= end_date:
                    text = md_file.read_text(encoding="utf-8")
                    fm = _parse_frontmatter(text)
                    fm["_file"] = md_file.name
                    fm["_text"] = text
                    notes.append(fm)
        except Exception:
            pass

    log.info(f"Found {len(notes)} existing notes for the week")
    return notes


def _parse_frontmatter(text: str) -> Dict:
    """Simple YAML frontmatter parser."""
    result: Dict = {}
    if not text.startswith("---"):
        return result
    parts = text.split("---", 2)
    if len(parts) < 3:
        return result
    for line in parts[1].splitlines():
        if ":" in line:
            k, _, v = line.partition(":")
            result[k.strip().lower()] = v.strip().strip('"').strip("'")
    return result


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------
def analyse_trends(papers: List[Dict], existing_notes: List[Dict]) -> Dict:
    """
    Identify trending topics, methods, and patterns across the week's papers.
    """
    # Keyword frequency across all abstracts + titles
    keyword_freq: Counter = Counter()
    category_freq: Counter = Counter()
    area_freq: Counter = Counter()

    # Interest areas from ranked papers
    for p in papers:
        text = (p.get("title", "") + " " + p.get("summary", "")).lower()
        # Count common AI/ML terms
        for kw in [
            "transformer", "llm", "rag", "knowledge graph", "fine-tuning",
            "reinforcement learning", "diffusion", "multimodal", "graph neural",
            "retrieval", "embedding", "attention", "clinical", "biomedical",
            "education", "adaptive", "aging", "longevity",
        ]:
            if kw in text:
                keyword_freq[kw] += 1
        category_freq[p.get("category", "unknown")] += 1

    # Top areas from notes
    for note in existing_notes:
        areas_str = note.get("top_areas", "")
        for area in re.findall(r'"([^"]+)"', areas_str):
            area_freq[area] += 1

    return {
        "top_keywords": keyword_freq.most_common(15),
        "top_categories": category_freq.most_common(10),
        "top_areas": area_freq.most_common(8),
        "paper_count": len(papers),
        "note_count": len(existing_notes),
    }


def find_gap_matches(papers: List[Dict]) -> List[Dict]:
    """Match papers against defined research gaps."""
    matched_gaps = []
    for gap in RESEARCH_GAPS:
        matching_papers = []
        for p in papers:
            text = (p.get("title", "") + " " + p.get("summary", "")).lower()
            matches = sum(1 for kw in gap["keywords"] if kw in text)
            if matches >= 2:
                matching_papers.append({
                    "title": p.get("title", "Unknown"),
                    "arxiv_id": p.get("arxiv_id", ""),
                    "url": p.get("url", ""),
                    "match_score": matches,
                })
        if matching_papers:
            matched_gaps.append({
                **gap,
                "matching_papers": sorted(
                    matching_papers, key=lambda x: x["match_score"], reverse=True
                )[:3],
            })
    return matched_gaps


def find_citation_candidates(papers: List[Dict]) -> Dict[str, List[Dict]]:
    """Find papers that could be cited in Anirach's ongoing work."""
    candidates: Dict[str, List[Dict]] = {proj: [] for proj in CITATION_TARGETS}

    for p in papers:
        text = (p.get("title", "") + " " + p.get("summary", "")).lower()
        for project, keywords in CITATION_TARGETS.items():
            matches = sum(1 for kw in keywords if kw in text)
            if matches >= 2:
                candidates[project].append({
                    "title": p.get("title", "Unknown"),
                    "arxiv_id": p.get("arxiv_id", ""),
                    "url": p.get("url", ""),
                    "authors": p.get("authors", ""),
                    "published": p.get("published", ""),
                    "match_score": matches,
                })

    # Sort by match score and deduplicate
    for proj in candidates:
        candidates[proj] = sorted(
            candidates[proj], key=lambda x: x["match_score"], reverse=True
        )[:5]

    return candidates


def find_collaboration_targets(papers: List[Dict]) -> List[Dict]:
    """Identify high-value collaboration targets from this week's papers."""
    targets = []
    thai_keywords = [
        "thailand", "thai", "kmutnb", "chulalongkorn", "mahidol", "kmitl",
        "kasetsart", "thammasat", "nectec", "nstda", "chiang mai",
    ]
    similar_topic_keywords = [
        "rag", "knowledge graph", "second brain", "adaptive learning",
        "biomedical ai", "longevity", "aging ai", "clinical nlp",
    ]

    for p in papers:
        authors_raw = p.get("authors", "")
        if isinstance(authors_raw, list):
            author_names = [
                a.get("name", a) if isinstance(a, dict) else str(a)
                for a in authors_raw
            ]
        else:
            author_names = [s.strip() for s in str(authors_raw).split(",")]

        text = (p.get("title", "") + " " + p.get("summary", "")).lower()
        similar_topics = [kw for kw in similar_topic_keywords if kw in text]

        # Check for Thai authors
        authors_text = " ".join(author_names).lower()
        thai_matches = [kw for kw in thai_keywords if kw in authors_text]

        if thai_matches or similar_topics:
            targets.append({
                "paper_title": p.get("title", "Unknown"),
                "arxiv_id": p.get("arxiv_id", ""),
                "url": p.get("url", ""),
                "authors": author_names[:5],
                "thai_connection": bool(thai_matches),
                "similar_topics": similar_topics,
                "relevance": "🇹🇭 Thai connection" if thai_matches else f"📚 {', '.join(similar_topics[:3])}",
            })

    return targets[:10]


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------
def generate_synthesis_note(
    week_label: str,
    start_date: date,
    end_date: date,
    papers: List[Dict],
    existing_notes: List[Dict],
    trends: Dict,
    gap_matches: List[Dict],
    citation_candidates: Dict[str, List[Dict]],
    collab_targets: List[Dict],
) -> str:
    """Generate the weekly synthesis Obsidian note."""
    today = date.today().isoformat()

    # Top keywords section
    kw_lines = [f"  - `{kw}` ({count})" for kw, count in trends["top_keywords"][:10]]
    cat_lines = [f"  - `{cat}` ({count})" for cat, count in trends["top_categories"][:5]]
    area_lines = [f"  - **{area}** ({count} papers)" for area, count in trends["top_areas"][:5]]

    # Gap matches section
    gap_sections = ""
    for gap in gap_matches:
        gap_sections += f"\n### 🔍 {gap['gap']}\n"
        gap_sections += f"_{gap['description']}_\n"
        gap_sections += f"**Why relevant:** {gap['relevance']}\n\n"
        gap_sections += "**Matching papers this week:**\n"
        for mp in gap["matching_papers"]:
            gap_sections += f"- [{mp['title'][:70]}]({mp['url']}) (score: {mp['match_score']})\n"

    # Citation candidates section
    cite_sections = ""
    for project, candidates in citation_candidates.items():
        if candidates:
            cite_sections += f"\n### 📄 {project}\n\n"
            for c in candidates:
                authors = c["authors"]
                if isinstance(authors, list):
                    authors_str = ", ".join(str(a) for a in authors[:3])
                else:
                    authors_str = str(authors)[:80]
                cite_sections += (
                    f"- **[{c['title'][:70]}]({c['url']})**\n"
                    f"  _{authors_str}_ ({c['published']}) — Match: {c['match_score']}\n"
                )

    # Collaboration targets section
    collab_lines = ""
    for t in collab_targets[:6]:
        authors_str = ", ".join(t["authors"][:3])
        collab_lines += (
            f"- **[{t['paper_title'][:60]}]({t['url']})**\n"
            f"  Authors: _{authors_str}_\n"
            f"  {t['relevance']}\n\n"
        )

    # Note links section — link to individual paper notes from this week
    note_links = ""
    for note in existing_notes[:10]:
        filename = note.get("_file", "")
        stem = filename.replace(".md", "") if filename else ""
        if stem:
            title = note.get("title", stem)[:70]
            score = note.get("relevance_score", "?")
            note_links += f"- [[Papers/{stem}|{title}]] — {score}/10\n"

    NL = chr(10)
    note = f"""---
type: weekly-synthesis
date: "{today}"
week: "{week_label}"
period: "{start_date} to {end_date}"
tags: ["research", "weekly-synthesis", "trends"]
papers_analysed: {trends["paper_count"]}
notes_created: {trends["note_count"]}
---

# Weekly Research Synthesis — {week_label}

**Period:** {start_date} → {end_date}
**Papers analysed:** {trends["paper_count"]}
**Notes created:** {trends["note_count"]}

_Auto-generated by `weekly_synthesis.py`_

---

## 📊 This Week's Trends

### Top Keywords
{NL.join(kw_lines) if kw_lines else "  - No data"}

### arXiv Categories
{NL.join(cat_lines) if cat_lines else "  - No data"}

### Research Areas (from scored papers)
{NL.join(area_lines) if area_lines else "  - No area data"}

---

## 🎯 Research Opportunities

These are gaps that match Anirach's expertise, surfaced from this week's literature:
{gap_sections if gap_sections else "_No strong gap matches found this week._"}

---

## 📚 Papers to Cite in Ongoing Work
{cite_sections if cite_sections else "_No strong citation candidates found this week._"}

---

## 🤝 Potential Collaboration Targets

{collab_lines if collab_lines else "_No collaboration targets flagged this week._"}

---

## 📋 Notes Created This Week

{note_links if note_links else "_No notes created yet. Run auto_summarize.py first._"}

---

## 🔗 Related

- [[People/Anirach Mingkhwan]]
- [[Projects/RAG Second Brain]]
- [[Projects/NCD-CIE]]
- [[Documents/Daily-Research-{start_date}]]
- [[Documents/Daily-Research-{end_date}]]

---

## 🤖 Generation Notes

- **Generated:** {today}
- **Script:** `research/weekly_synthesis.py`
- **Papers source:** `research/papers/`
- **Notes source:** `KnowledgeGraph/Papers/`
"""

    return note


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def run(
    week_str: Optional[str] = None,
    days_back: int = 7,
    include_gap_analysis: bool = True,
) -> Optional[Path]:
    """
    Main entry point.

    Args:
        week_str: ISO week like "2026-W08". None = current week.
        days_back: How many days to look back (used if week_str is None).
        include_gap_analysis: Whether to run gap/opportunity analysis.

    Returns:
        Path to the created synthesis note, or None on failure.
    """
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from interest_matcher import InterestMatcher
    except ImportError:
        log.error("Could not import interest_matcher.")
        raise

    start_date, end_date = get_week_dates(week_str, days_back)
    week_label = (
        week_str
        if week_str
        else f"{start_date.year}-W{start_date.isocalendar()[1]:02d}"
    )
    log.info(f"Synthesising week {week_label}: {start_date} → {end_date}")

    # Load papers
    papers = load_papers_for_week(start_date, end_date)
    if not papers:
        log.warning("No papers found for this week. Check that daily monitor has run.")

    # Load existing notes
    existing_notes = load_existing_notes(start_date, end_date)

    # Rank papers with interest matcher
    matcher = InterestMatcher()
    if papers:
        ranked = matcher.rank_papers(papers, min_score=0.3)
        area_trends = matcher.get_area_trend(ranked)
        log.info(f"Top areas: {list(area_trends.keys())[:3]}")
    else:
        ranked = []

    # Analyse trends
    trends = analyse_trends(papers, existing_notes)

    # Research opportunities
    gap_matches = find_gap_matches(papers) if include_gap_analysis else []

    # Citation candidates
    citation_candidates = find_citation_candidates(papers)

    # Collaboration targets
    collab_targets = find_collaboration_targets(papers)

    # Generate note
    synthesis_note = generate_synthesis_note(
        week_label=week_label,
        start_date=start_date,
        end_date=end_date,
        papers=papers,
        existing_notes=existing_notes,
        trends=trends,
        gap_matches=gap_matches,
        citation_candidates=citation_candidates,
        collab_targets=collab_targets,
    )

    # Save
    try:
        DOCS_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        log.error(f"Cannot create docs directory: {exc}")
        return None

    output_path = DOCS_DIR / f"Weekly-Research-{week_label}.md"
    try:
        output_path.write_text(synthesis_note, encoding="utf-8")
        log.info(f"✅ Weekly synthesis saved: {output_path}")
        return output_path
    except Exception as exc:
        log.error(f"Failed to save synthesis note: {exc}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate weekly research synthesis note for Obsidian vault"
    )
    parser.add_argument(
        "--week", "-w",
        help="ISO week to synthesise (e.g. 2026-W08). Default: current week",
    )
    parser.add_argument(
        "--days", "-d",
        type=int,
        default=7,
        help="Number of days to look back (default: 7). Used when --week not set.",
    )
    parser.add_argument(
        "--no-gaps",
        action="store_true",
        help="Skip research gap analysis",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    result = run(
        week_str=args.week,
        days_back=args.days,
        include_gap_analysis=not args.no_gaps,
    )

    if result:
        print(f"\n✅ Weekly synthesis: {result}")
    else:
        print("❌ Synthesis failed. Check logs.")
        sys.exit(1)
