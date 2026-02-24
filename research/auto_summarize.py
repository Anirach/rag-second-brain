#!/usr/bin/env python3
"""
auto_summarize.py — Auto-summarize top arXiv papers and create Obsidian notes.

Reads the latest arXiv paper JSON from the daily monitor, picks the top-5 by
interest score, generates structured summaries, and saves them as Obsidian notes
with frontmatter, wikilinks, and collaborator flags.

Usage:
    python3 auto_summarize.py                       # Use latest papers JSON
    python3 auto_summarize.py papers_20260221.json  # Use specific file
    python3 auto_summarize.py --top-n 10            # Process top 10 papers
"""

import argparse
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [auto_summarize] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# Paths
RESEARCH_DIR = Path(__file__).parent
VAULT_ROOT = Path("/home/clawdbot/obsidian-vault")
PAPERS_DIR = RESEARCH_DIR / "papers"
NOTES_DIR = VAULT_ROOT / "KnowledgeGraph/Papers"
SEMANTIC_SCHOLAR_BASE = "https://api.semanticscholar.org/graph/v1"

# Anirach's Semantic Scholar author name for co-authorship lookup
ANIRACH_SS_QUERY = "Anirach Mingkhwan"
ANIRACH_KNOWN_PAPER_TITLES: List[str] = [
    "multi-graph neural architecture",
    "rag second brain",
    "knowledge graph",
    "ncd cie",
]

# Interest areas → wikilink topics mapping
AREA_WIKILINKS: Dict[str, List[str]] = {
    "AI/ML": ["[[Topics/Artificial Intelligence]]", "[[Topics/Machine Learning]]",
              "[[Topics/Deep Learning]]", "[[Topics/Neural Networks]]"],
    "Education Technology": ["[[Topics/Education Technology]]", "[[Topics/Adaptive Learning]]",
                             "[[Topics/AI in Education]]"],
    "Longevity Healthcare": ["[[Topics/Longevity]]", "[[Topics/Aging Research]]",
                             "[[Topics/Healthcare AI]]"],
    "Biomedical AI": ["[[Topics/Biomedical AI]]", "[[Topics/Clinical NLP]]",
                      "[[Topics/Digital Health]]"],
    "RAG": ["[[Topics/RAG]]", "[[Topics/Retrieval Augmented Generation]]",
            "[[Projects/RAG Second Brain]]"],
    "Knowledge Graphs": ["[[Topics/Knowledge Graphs]]", "[[Topics/Ontology]]",
                         "[[Projects/RAG Second Brain]]"],
    "NLP": ["[[Topics/Natural Language Processing]]", "[[Topics/Large Language Models]]"],
    "Computer Vision": ["[[Topics/Computer Vision]]", "[[Topics/Vision Transformers]]"],
}

# Project relevance triggers
PROJECT_RELEVANCE: Dict[str, List[str]] = {
    "RAG Second Brain": ["rag", "retrieval", "knowledge graph", "second brain",
                         "vector database", "embedding", "semantic search"],
    "NCD-CIE": ["ncd", "non-communicable disease", "classification", "icd",
                 "clinical", "healthcare", "medical ai", "diagnosis"],
}


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------
def ss_search_author(name: str) -> Optional[str]:
    """Search Semantic Scholar for an author by name, return author ID."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/author/search"
    params = {"query": name, "fields": "name,affiliations,paperCount,hIndex", "limit": 3}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            if data:
                return data[0].get("authorId")
    except Exception as exc:
        log.warning(f"SS author search failed for '{name}': {exc}")
    return None


def ss_get_author_papers(author_id: str, limit: int = 20) -> List[str]:
    """Return list of paper titles for a given SS author ID."""
    url = f"{SEMANTIC_SCHOLAR_BASE}/author/{author_id}/papers"
    params = {"fields": "title", "limit": limit}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return [p.get("title", "") for p in resp.json().get("data", [])]
    except Exception as exc:
        log.warning(f"SS papers fetch failed for author {author_id}: {exc}")
    return []


def ss_search_paper(arxiv_id: str) -> Optional[Dict]:
    """Fetch paper metadata from Semantic Scholar by arXiv ID."""
    clean_id = arxiv_id.replace("v1", "").replace("v2", "").replace("v3", "")
    url = f"{SEMANTIC_SCHOLAR_BASE}/paper/arXiv:{clean_id}"
    params = {"fields": "title,authors,year,abstract,citationCount,references,externalIds"}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as exc:
        log.warning(f"SS paper fetch failed for arXiv:{clean_id}: {exc}")
    return None


def check_coauthorship(author_name: str, anirach_papers: List[str]) -> bool:
    """Check if author has published with Anirach (via title overlap)."""
    name_lower = author_name.lower()
    # Check their SS papers
    author_id = ss_search_author(author_name)
    if not author_id:
        return False
    author_papers = ss_get_author_papers(author_id, limit=30)
    # Cross-reference with Anirach's known papers
    for ap in author_papers:
        ap_lower = ap.lower()
        for known in ANIRACH_KNOWN_PAPER_TITLES:
            if known in ap_lower:
                return True
    return False


def detect_thai_affiliation(author_name: str) -> bool:
    """Use Semantic Scholar to check if author has Thai affiliation."""
    thai_keywords = [
        "thailand", "thai", "kmutnb", "chulalongkorn", "mahidol", "kmitl",
        "kasetsart", "thammasat", "nectec", "nstda", "chiang mai",
    ]
    url = f"{SEMANTIC_SCHOLAR_BASE}/author/search"
    params = {"query": author_name, "fields": "name,affiliations", "limit": 3}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            for author in resp.json().get("data", []):
                affs = author.get("affiliations", [])
                aff_text = " ".join(affs).lower() if affs else ""
                if any(kw in aff_text for kw in thai_keywords):
                    return True
    except Exception as exc:
        log.debug(f"Thai affiliation check failed for '{author_name}': {exc}")
    return False


# ---------------------------------------------------------------------------
# Note generation
# ---------------------------------------------------------------------------
def make_short_title(title: str, max_len: int = 50) -> str:
    """Create a filesystem-safe short title slug."""
    # Remove special chars, replace spaces with hyphens
    slug = re.sub(r"[^a-zA-Z0-9\s\-]", "", title)
    slug = re.sub(r"\s+", "-", slug.strip())
    slug = slug[:max_len].rstrip("-")
    return slug


def extract_problem_method_results(abstract: str) -> Tuple[str, str, str]:
    """
    Heuristically extract Problem / Method / Results from an abstract.
    Falls back to splitting the abstract into thirds.
    """
    sentences = re.split(r"(?<=[.!?])\s+", abstract.strip())

    # Simple heuristic: first ~30% = problem, middle ~40% = method, last ~30% = results
    n = len(sentences)
    if n < 3:
        return abstract, "See abstract.", "See abstract."

    p_end = max(1, n // 3)
    m_end = max(p_end + 1, (2 * n) // 3)

    problem = " ".join(sentences[:p_end])
    method = " ".join(sentences[p_end:m_end])
    results = " ".join(sentences[m_end:])

    return problem or abstract, method or "N/A", results or "N/A"


def score_project_relevance(paper: Dict) -> Dict[str, int]:
    """Score paper relevance to ongoing projects (1-10)."""
    text = (paper.get("title", "") + " " + paper.get("summary", "")).lower()
    scores = {}
    for project, keywords in PROJECT_RELEVANCE.items():
        matches = sum(1 for kw in keywords if kw in text)
        scores[project] = min(10, matches * 2)
    return scores


def generate_wikilinks(top_areas: List[str]) -> List[str]:
    """Generate wikilinks based on matching interest areas."""
    links = set()
    for area in top_areas:
        links.update(AREA_WIKILINKS.get(area, []))
    return sorted(links)


def check_collaborators(paper: Dict, anirach_papers: List[str]) -> List[Dict]:
    """
    Analyse authors for collaboration potential.
    Returns list of {name, thai_affiliation, coauthored_with_anirach, flag}
    """
    authors_raw = paper.get("authors", "")
    if isinstance(authors_raw, list):
        author_names = [
            a.get("name", a) if isinstance(a, dict) else str(a)
            for a in authors_raw
        ]
    else:
        author_names = [s.strip() for s in str(authors_raw).split(",")]

    # Remove "et al." suffix
    author_names = [n for n in author_names if n and "et al" not in n.lower()]

    collaborators = []
    for name in author_names[:6]:  # Limit API calls to first 6 authors
        col = {"name": name, "thai_affiliation": False, "coauthored_with_anirach": False, "flags": []}

        # Check Thai affiliation (quick text check first)
        thai_kws = ["thailand", "thai", "kmutnb", "chulalongkorn", "mahidol",
                    "nectec", "kasetsart", "thammasat"]
        name_lower = name.lower()
        if any(kw in name_lower for kw in thai_kws):
            col["thai_affiliation"] = True
            col["flags"].append("🇹🇭 Thai institution")
        else:
            # Do SS lookup (rate-limited)
            time.sleep(0.3)
            if detect_thai_affiliation(name):
                col["thai_affiliation"] = True
                col["flags"].append("🇹🇭 Thai institution")

        # Check co-authorship with Anirach
        if check_coauthorship(name, anirach_papers):
            col["coauthored_with_anirach"] = True
            col["flags"].append("🤝 Prior co-author")

        collaborators.append(col)

    return collaborators


def generate_obsidian_note(
    paper: Dict,
    match: Dict,
    collaborators: List[Dict],
    anirach_papers: List[str],
    today: str,
) -> str:
    """Generate a complete Obsidian markdown note for a paper."""
    title = paper.get("title", "Unknown Title")
    arxiv_id = paper.get("arxiv_id", paper.get("id", "unknown"))
    authors_raw = paper.get("authors", "")
    if isinstance(authors_raw, list):
        authors_list = [
            a.get("name", a) if isinstance(a, dict) else str(a)
            for a in authors_raw
        ]
        authors_str = ", ".join(authors_list[:5])
        if len(authors_list) > 5:
            authors_str += " et al."
    else:
        authors_str = str(authors_raw)

    abstract = paper.get("summary", paper.get("abstract", "No abstract available."))
    url = paper.get("url", f"https://arxiv.org/abs/{arxiv_id}")
    published = paper.get("published", today)
    overall_score = match.get("overall_score", 0)
    top_areas = match.get("top_areas", [])
    area_scores = match.get("area_scores", {})
    flags = match.get("flags", [])

    # Generate structured summary sections
    problem, method, results = extract_problem_method_results(abstract)

    # Wikilinks
    wikilinks = generate_wikilinks(top_areas)

    # Project relevance
    project_scores = score_project_relevance(paper)

    # Tags
    tags = ["paper", "arxiv"]
    for area in top_areas:
        tags.append(area.lower().replace("/", "-").replace(" ", "-"))
    tags_yaml = "[" + ", ".join(f'"{t}"' for t in tags) + "]"

    # Area scores section
    area_lines = []
    for area, score in sorted(area_scores.items(), key=lambda x: x[1], reverse=True):
        if score > 0:
            bar = "█" * int(score) + "░" * (10 - int(score))
            area_lines.append(f"  - {area}: {bar} {score:.1f}/10")

    # Collaborator section
    collab_lines = []
    for col in collaborators:
        flag_str = " ".join(col["flags"]) if col["flags"] else "—"
        collab_lines.append(f"  - **{col['name']}** {flag_str}")

    # Build note
    note = f"""---
type: paper
date: "{today}"
tags: {tags_yaml}
arxiv_id: "{arxiv_id}"
authors: "{authors_str}"
published: "{published}"
relevance_score: {overall_score}
top_areas: [{", ".join(f'"{a}"' for a in top_areas)}]
url: "{url}"
---

# {title}

> **arXiv:** [{arxiv_id}]({url})
> **Published:** {published}
> **Authors:** {authors_str}

## 🔍 Structured Summary

### Problem
{problem}

### Method
{method}

### Results & Contributions
{results}

### Relevance to Anirach's Research
**Overall Score:** {overall_score}/10

{chr(10).join(area_lines) if area_lines else "No strong matches."}

**Project Relevance:**
"""

    for proj, score in project_scores.items():
        if score > 0:
            note += f"  - **{proj}**: {score}/10\n"

    note += f"""
## 🔗 Related Topics

{chr(10).join(wikilinks) if wikilinks else "No strong topic matches."}

## 👥 Potential Collaborators

{chr(10).join(collab_lines) if collab_lines else "  - No flagged collaborators."}

## 📎 Full Abstract

{abstract}

## 🏷️ Source

- **Source:** arXiv
- **Category:** {paper.get("category", "N/A")}
- **Fetched:** {today}
- **Monitor:** [[Documents/Daily-Research-Monitor]]
"""

    if flags:
        note += f"\n## ⚠️ Flags\n\n"
        for flag in flags:
            note += f"- {flag}\n"

    return note


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def find_latest_papers_file() -> Optional[Path]:
    """Find the most recent papers_*.json file."""
    files = sorted(PAPERS_DIR.glob("papers_*.json"), reverse=True)
    return files[0] if files else None


def load_papers(papers_file: Path) -> List[Dict]:
    """Load papers from JSON file."""
    with open(papers_file) as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        papers = data.get("relevant_papers", [])
        log.info(f"Loaded {len(papers)} papers from {papers_file.name} "
                 f"(timestamp: {data.get('timestamp', 'N/A')})")
    else:
        papers = data
        log.info(f"Loaded {len(papers)} papers from {papers_file.name}")
    return papers


def ensure_notes_dir() -> Path:
    """Create KnowledgeGraph/Papers directory if it doesn't exist."""
    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    return NOTES_DIR


def run(papers_file: Optional[Path] = None, top_n: int = 5) -> List[Path]:
    """
    Main entry point. Returns list of created note paths.

    Args:
        papers_file: Path to papers JSON. Uses latest if None.
        top_n: Number of top papers to summarise.
    """
    # Import here to avoid circular imports if used as module
    sys.path.insert(0, str(Path(__file__).parent))
    try:
        from interest_matcher import InterestMatcher
    except ImportError:
        log.error("Could not import interest_matcher. Ensure it's in the same directory.")
        raise

    today = date.today().isoformat()

    # Find papers file
    if papers_file is None:
        papers_file = find_latest_papers_file()
        if papers_file is None:
            log.error(f"No papers files found in {PAPERS_DIR}")
            return []
    log.info(f"Using papers file: {papers_file}")

    # Load and rank papers
    papers = load_papers(papers_file)
    if not papers:
        log.warning("No papers found in file.")
        return []

    matcher = InterestMatcher()
    ranked = matcher.rank_papers(papers, top_n=top_n, min_score=0.5)
    log.info(f"Processing top {len(ranked)} papers")

    # Pre-fetch Anirach's paper list (for co-authorship detection)
    anirach_papers: List[str] = []
    try:
        anirach_id = ss_search_author(ANIRACH_SS_QUERY)
        if anirach_id:
            anirach_papers = ss_get_author_papers(anirach_id, limit=50)
            log.info(f"Found {len(anirach_papers)} Anirach papers on Semantic Scholar")
    except Exception as exc:
        log.warning(f"Could not fetch Anirach's SS papers: {exc}")

    # Create notes directory
    try:
        notes_dir = ensure_notes_dir()
    except PermissionError as exc:
        log.error(f"Cannot create notes directory: {exc}")
        log.error("Run as host user or pre-create the directory.")
        return []

    # Generate notes
    created_paths: List[Path] = []
    for i, paper in enumerate(ranked, 1):
        title = paper.get("title", f"Paper-{i}")
        arxiv_id = paper.get("arxiv_id", "unknown")
        match = paper.get("interest_match", {})

        log.info(f"[{i}/{len(ranked)}] Processing: {title[:60]}...")

        # Detect collaborators (with SS lookups)
        try:
            collaborators = check_collaborators(paper, anirach_papers)
        except Exception as exc:
            log.warning(f"Collaborator check failed: {exc}")
            collaborators = []

        # Generate note content
        note_content = generate_obsidian_note(
            paper=paper,
            match=match,
            collaborators=collaborators,
            anirach_papers=anirach_papers,
            today=today,
        )

        # Save note
        short_title = make_short_title(title)
        note_filename = f"{today}-{short_title}.md"
        note_path = notes_dir / note_filename

        try:
            note_path.write_text(note_content, encoding="utf-8")
            log.info(f"  ✓ Saved: {note_path.name}")
            created_paths.append(note_path)
        except Exception as exc:
            log.error(f"  ✗ Failed to save {note_filename}: {exc}")

        # Rate limit for API calls
        time.sleep(0.5)

    log.info(f"Created {len(created_paths)} Obsidian notes in {notes_dir}")

    # Generate summary index note
    if created_paths:
        _save_daily_index(today, ranked, created_paths)

    return created_paths


def _save_daily_index(today: str, ranked_papers: List[Dict], note_paths: List[Path]):
    """Save a daily index note listing all processed papers."""
    index_content = f"""---
type: daily-research-index
date: "{today}"
tags: ["research", "daily-monitor", "index"]
---

# Research Papers — {today}

Auto-generated by `auto_summarize.py`. Top {len(ranked_papers)} papers from today's arXiv monitor.

## Papers Processed

"""
    for i, (paper, path) in enumerate(zip(ranked_papers, note_paths), 1):
        title = paper.get("title", "Unknown")
        score = paper.get("interest_match", {}).get("overall_score", 0)
        areas = paper.get("interest_match", {}).get("top_areas", [])
        short_title = path.stem
        index_content += f"{i}. [[Papers/{short_title}|{title[:70]}]] — Score: {score}/10 | {', '.join(areas)}\n"

    index_content += f"""
## Links

- [[Documents/Weekly-Research-{datetime.now().strftime('%Y-W%V')}|This Week's Synthesis]]
- [[Projects/RAG Second Brain]]
- [[Projects/NCD-CIE]]
"""

    index_path = VAULT_ROOT / "KnowledgeGraph/Documents" / f"Daily-Research-{today}.md"
    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(index_content, encoding="utf-8")
        log.info(f"Daily index saved: {index_path.name}")
    except Exception as exc:
        log.warning(f"Could not save daily index: {exc}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-summarize arXiv papers to Obsidian notes")
    parser.add_argument("papers_file", nargs="?", type=Path,
                        help="Path to papers JSON file (default: latest)")
    parser.add_argument("--top-n", type=int, default=5,
                        help="Number of top papers to process (default: 5)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    created = run(papers_file=args.papers_file, top_n=args.top_n)
    if created:
        print(f"\n✅ Created {len(created)} Obsidian notes:")
        for p in created:
            print(f"  {p}")
    else:
        print("❌ No notes created. Check logs for errors.")
        sys.exit(1)
