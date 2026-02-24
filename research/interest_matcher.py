#!/usr/bin/env python3
"""
interest_matcher.py — Match arXiv papers against Anirach's research interests.

Loads the researcher profile from the Obsidian vault and scores papers
using keyword matching and TF-IDF-style term weighting.

Usage (module):
    from interest_matcher import InterestMatcher
    matcher = InterestMatcher()
    ranked = matcher.rank_papers(papers)

Usage (CLI):
    python3 interest_matcher.py papers_YYYYMMDD_HHMMSS.json
"""

import json
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [interest_matcher] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VAULT_ROOT = Path("/home/clawdbot/obsidian-vault")
PROFILE_PATH = VAULT_ROOT / "KnowledgeGraph/People/Anirach-Mingkhwan.md"
PROFILE_PATH_ALT = VAULT_ROOT / "KnowledgeGraph/People/Anirach Mingkhwan.md"
CONFIG_PATH = Path(__file__).parent / "config.json"

# ---------------------------------------------------------------------------
# Research-interest taxonomy
# ---------------------------------------------------------------------------
INTEREST_TAXONOMY: List[Dict] = [
    {
        "area": "AI/ML",
        "weight": 1.0,
        "keywords": [
            "machine learning", "deep learning", "neural network", "artificial intelligence",
            "reinforcement learning", "supervised learning", "unsupervised learning",
            "transformer", "attention mechanism", "self-supervised", "foundation model",
            "large language model", "llm", "gpt", "bert", "diffusion model",
            "generative ai", "multimodal", "vision-language", "embedding",
        ],
    },
    {
        "area": "Education Technology",
        "weight": 0.9,
        "keywords": [
            "education", "e-learning", "adaptive learning", "intelligent tutoring",
            "learning analytics", "educational ai", "pedagogical", "student",
            "curriculum", "teaching", "assessment", "personalized learning",
            "edtech", "mooc", "classroom", "instructional design",
        ],
    },
    {
        "area": "Longevity Healthcare",
        "weight": 0.85,
        "keywords": [
            "longevity", "aging", "lifespan", "healthspan", "anti-aging",
            "senescence", "gerontology", "life extension", "biogerontology",
            "age-related", "elderly", "geriatric", "biomarkers of aging",
            "hallmarks of aging", "epigenetic clock", "telomere",
        ],
    },
    {
        "area": "Biomedical AI",
        "weight": 0.85,
        "keywords": [
            "biomedical", "clinical", "medical ai", "healthcare ai", "diagnosis",
            "medical imaging", "radiology", "pathology", "drug discovery",
            "genomics", "proteomics", "metabolomics", "bioinformatics",
            "electronic health record", "ehr", "clinical nlp", "precision medicine",
            "personalized medicine", "digital health",
        ],
    },
    {
        "area": "RAG",
        "weight": 0.9,
        "keywords": [
            "retrieval augmented generation", "rag", "retrieval-augmented",
            "dense retrieval", "document retrieval", "semantic search",
            "vector database", "embedding search", "open domain qa",
            "question answering", "knowledge retrieval", "hybrid search",
            "re-ranking", "contextual retrieval",
        ],
    },
    {
        "area": "Knowledge Graphs",
        "weight": 0.85,
        "keywords": [
            "knowledge graph", "knowledge base", "ontology", "entity linking",
            "relation extraction", "named entity recognition", "information extraction",
            "graph neural network", "gnn", "graph attention", "property graph",
            "linked data", "rdf", "sparql", "second brain",
            "personal knowledge management", "pkm",
        ],
    },
    {
        "area": "NLP",
        "weight": 0.8,
        "keywords": [
            "natural language processing", "nlp", "text classification",
            "sentiment analysis", "summarization", "machine translation",
            "dialogue", "conversational ai", "chatbot", "prompt engineering",
            "fine-tuning", "instruction tuning", "rlhf", "text generation",
        ],
    },
    {
        "area": "Computer Vision",
        "weight": 0.6,
        "keywords": [
            "computer vision", "image recognition", "object detection",
            "semantic segmentation", "image generation", "video understanding",
            "convolutional", "vision transformer", "vit",
        ],
    },
]

THAI_INSTITUTION_KEYWORDS = [
    "thailand", "thai", "kmutnb", "chulalongkorn", "mahidol", "kmitl",
    "kasetsart", "thammasat", "prince of songkla", "suranaree", "naresuan",
    "nectec", "nstda", "bangkok", "chiang mai university", "cmu", "kku",
    "ubon ratchathani", "buu",
]

ANIRACH_NAME_VARIANTS = [
    "anirach mingkhwan",
    "a. mingkhwan",
    "mingkhwan",
]


class InterestMatcher:
    """Match arXiv papers against Anirach's research profile."""

    def __init__(
        self,
        profile_path: Path = None,
        config_path: Path = CONFIG_PATH,
        taxonomy: List[Dict] = None,
    ):
        if profile_path is None:
            if PROFILE_PATH.exists():
                profile_path = PROFILE_PATH
            elif PROFILE_PATH_ALT.exists():
                profile_path = PROFILE_PATH_ALT
            else:
                profile_path = PROFILE_PATH
        self.profile_path = profile_path
        self.config_path = config_path
        self.taxonomy = taxonomy if taxonomy is not None else INTEREST_TAXONOMY
        self.profile = self._load_profile()
        self.config = self._load_config()
        self._kw_index = self._build_keyword_index()

    def _load_profile(self) -> Dict:
        profile: Dict = {
            "name": "Anirach Mingkhwan",
            "interests": [],
            "projects": [],
            "institution": "KMUTNB",
        }
        if self.profile_path.exists():
            try:
                text = self.profile_path.read_text(encoding="utf-8")
                log.info(f"Loaded profile from {self.profile_path}")
                fm = self._parse_frontmatter(text)
                profile.update(fm)
                profile["_raw"] = text
            except Exception as exc:
                log.warning(f"Could not read profile: {exc}")
        else:
            log.warning(f"Profile not found at {self.profile_path}, using defaults")
        return profile

    def _load_config(self) -> Dict:
        if self.config_path.exists():
            try:
                with open(self.config_path) as fh:
                    return json.load(fh)
            except Exception as exc:
                log.warning(f"Could not read config: {exc}")
        return {}

    @staticmethod
    def _parse_frontmatter(text: str) -> Dict:
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

    def _build_keyword_index(self) -> Dict[str, List[Dict]]:
        index: Dict[str, List[Dict]] = defaultdict(list)
        for entry in self.taxonomy:
            for kw in entry["keywords"]:
                index[kw.lower()].append({"area": entry["area"], "weight": entry["weight"]})
        return index

    @staticmethod
    def _normalise(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^a-z0-9\s\-]", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    def _score_text(self, text: str) -> Dict[str, float]:
        norm = self._normalise(text)
        area_scores: Dict[str, float] = defaultdict(float)
        for kw, entries in self._kw_index.items():
            pattern = r"\b" + re.escape(kw) + r"\b"
            matches = len(re.findall(pattern, norm))
            if matches:
                for e in entries:
                    area_scores[e["area"]] += e["weight"] * (1 + math.log1p(matches))
        return dict(area_scores)

    def score_paper(self, paper: Dict) -> Dict:
        """Score a paper against all interest areas. Returns match dict."""
        title = paper.get("title", "")
        abstract = paper.get("summary", paper.get("abstract", ""))
        combined = f"{title} {title} {title} {abstract}"  # title 3x weight

        raw = self._score_text(combined)
        max_possible = 15.0
        area_scores = {}
        for area_info in self.taxonomy:
            area = area_info["area"]
            area_scores[area] = round(min(raw.get(area, 0.0) / max_possible * 10, 10.0), 2)

        sorted_areas = sorted(area_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [a for a, s in sorted_areas if s > 0][:3]
        overall = round(sum(area_scores[a] for a in top3) / len(top3), 2) if top3 else 0.0

        return {
            "area_scores": area_scores,
            "overall_score": overall,
            "top_areas": top3,
            "flags": self._detect_flags(paper),
        }

    def _detect_flags(self, paper: Dict) -> List[str]:
        flags: List[str] = []
        authors_raw = paper.get("authors", "")
        if isinstance(authors_raw, list):
            author_names = [
                a.get("name", a) if isinstance(a, dict) else str(a)
                for a in authors_raw
            ]
        else:
            author_names = [s.strip() for s in str(authors_raw).split(",")]
        authors_text = " ".join(author_names).lower()

        if any(kw in authors_text for kw in THAI_INSTITUTION_KEYWORDS):
            flags.append("🇹🇭 Thai institution author")
        if any(v in authors_text for v in ANIRACH_NAME_VARIANTS):
            flags.append("👤 Anirach is co-author")
        return flags

    def rank_papers(
        self,
        papers: List[Dict],
        top_n: Optional[int] = None,
        min_score: float = 0.0,
    ) -> List[Dict]:
        """Score and rank papers, returning list augmented with 'interest_match'."""
        results = []
        for paper in papers:
            match = self.score_paper(paper)
            if match["overall_score"] >= min_score:
                results.append({**paper, "interest_match": match})
        results.sort(key=lambda p: p["interest_match"]["overall_score"], reverse=True)
        if top_n is not None:
            results = results[:top_n]
        log.info(f"Ranked {len(papers)} papers → {len(results)} above threshold (min={min_score})")
        return results

    def get_area_trend(self, ranked_papers: List[Dict]) -> Dict[str, float]:
        """Aggregate area scores across all ranked papers."""
        totals: Dict[str, float] = defaultdict(float)
        for p in ranked_papers:
            for area, score in p["interest_match"]["area_scores"].items():
                totals[area] += score
        return dict(sorted(totals.items(), key=lambda x: x[1], reverse=True))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 interest_matcher.py <papers_YYYYMMDD.json>")
        sys.exit(1)
    papers_file = Path(sys.argv[1])
    if not papers_file.exists():
        print(f"File not found: {papers_file}")
        sys.exit(1)
    with open(papers_file) as fh:
        data = json.load(fh)
    papers = data.get("relevant_papers", data) if isinstance(data, dict) else data
    matcher = InterestMatcher()
    ranked = matcher.rank_papers(papers, top_n=10)
    print(f"\n{'='*60}\nTOP {len(ranked)} PAPERS BY INTEREST MATCH\n{'='*60}")
    for i, p in enumerate(ranked, 1):
        m = p["interest_match"]
        print(f"\n{i}. {p.get('title', 'Unknown')[:80]}")
        print(f"   Overall: {m['overall_score']}/10  Areas: {', '.join(m['top_areas'])}")
        if m["flags"]:
            print(f"   Flags: {', '.join(m['flags'])}")
