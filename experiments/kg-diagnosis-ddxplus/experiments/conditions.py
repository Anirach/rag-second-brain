"""Experimental conditions: baselines, proposed method, and ablations."""
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

logger = logging.getLogger(__name__)

# LLM response cache
CACHE_DIR = Path("cache/llm_responses")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_key(prompt: str, model: str) -> str:
    return hashlib.sha256(f"{model}:{prompt}".encode()).hexdigest()


def _get_cached(prompt: str, model: str) -> Optional[str]:
    key = _cache_key(prompt, model)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)["response"]
    return None


def _set_cache(prompt: str, model: str, response: str):
    key = _cache_key(prompt, model)
    path = CACHE_DIR / f"{key}.json"
    with open(path, "w") as f:
        json.dump({"prompt": prompt[:200], "model": model, "response": response}, f)


def call_llm(
    prompt: str,
    model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Call OpenAI LLM with caching and retry logic."""
    cached = _get_cached(prompt, model)
    if cached is not None:
        return cached

    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
            )
            result = response.choices[0].message.content.strip()
            _set_cache(prompt, model, result)
            return result
        except Exception as e:
            logger.warning(f"LLM call attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def _format_symptoms(vignette: Dict) -> str:
    """Format vignette symptoms for LLM prompt."""
    parts = []
    if vignette.get("age"):
        parts.append(f"Age: {vignette['age']}")
    if vignette.get("sex"):
        parts.append(f"Sex: {vignette['sex']}")
    if vignette.get("initial_evidence"):
        parts.append(f"Chief complaint: {vignette['initial_evidence']}")
    if vignette.get("symptoms"):
        parts.append(f"Symptoms/findings: {', '.join(vignette['symptoms'][:20])}")
    return "\n".join(parts)


def _parse_llm_diagnoses(response: str) -> List[str]:
    """Parse LLM response to extract list of diagnoses."""
    diagnoses = []
    for line in response.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering like "1.", "1)", "- "
        for prefix in ["- ", "* "]:
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            # Remove "1. " or "1) "
            for sep in [". ", ") ", ": "]:
                if sep in line:
                    line = line.split(sep, 1)[1]
                    break
        line = line.strip().strip("*").strip()
        if line and len(line) > 2:
            diagnoses.append(line)
    return diagnoses[:10]


SYSTEM_PROMPT = """You are a clinical decision support system. Given patient information, provide a differential diagnosis list.
Return ONLY a numbered list of up to 10 possible diagnoses, ordered from most likely to least likely.
Each line should contain only the diagnosis name, nothing else."""


def run_b1_llm_only(vignette: Dict) -> List[str]:
    """B1: LLM-only baseline."""
    symptoms = _format_symptoms(vignette)
    prompt = f"""{SYSTEM_PROMPT}

Patient information:
{symptoms}

Differential diagnosis (most likely first):"""
    response = call_llm(prompt)
    return _parse_llm_diagnoses(response)


def run_b2_dense_rag(vignette: Dict, dense_retriever) -> List[str]:
    """B2: LLM + Dense RAG baseline."""
    symptoms = _format_symptoms(vignette)
    retrieved = dense_retriever.retrieve(vignette.get("symptoms", []), top_k=10)
    context = "\n".join([f"- {d} (similarity: {s:.3f})" for d, s in retrieved])

    prompt = f"""{SYSTEM_PROMPT}

Patient information:
{symptoms}

Retrieved similar conditions (from semantic search):
{context}

Based on the patient information and retrieved conditions, provide your differential diagnosis (most likely first):"""
    response = call_llm(prompt)
    return _parse_llm_diagnoses(response)


def run_b3_bm25(vignette: Dict, stat_retriever) -> List[str]:
    """B3: LLM + BM25 baseline."""
    symptoms = _format_symptoms(vignette)
    retrieved = stat_retriever.retrieve(vignette.get("symptoms", []), top_k=10)
    context = "\n".join([f"- {d} (score: {s:.3f})" for d, s in retrieved])

    prompt = f"""{SYSTEM_PROMPT}

Patient information:
{symptoms}

Retrieved conditions (from statistical co-occurrence analysis):
{context}

Based on the patient information and retrieved conditions, provide your differential diagnosis (most likely first):"""
    response = call_llm(prompt)
    return _parse_llm_diagnoses(response)


def run_proposed(
    vignette: Dict,
    dense_retriever,
    stat_retriever,
    kg_retriever,
    fusion,
    use_gating: bool = True,
    exclude_sources: Optional[List[str]] = None,
) -> List[str]:
    """Proposed: LLM + Multi-Source KG Second Brain."""
    symptom_list = vignette.get("symptoms", [])
    symptoms_text = _format_symptoms(vignette)

    # Get results from each source
    sources = {}
    if not exclude_sources or "dense" not in exclude_sources:
        dense_results = dense_retriever.retrieve(symptom_list, top_k=10)
        sources["dense"] = dense_results
    else:
        dense_results = []

    if not exclude_sources or "statistical" not in exclude_sources:
        stat_results = stat_retriever.retrieve(symptom_list, top_k=10)
        sources["statistical"] = stat_results
    else:
        stat_results = []

    if not exclude_sources or "kg" not in exclude_sources:
        kg_results = kg_retriever.retrieve(symptom_list, top_k=10)
        sources["kg"] = kg_results
    else:
        kg_results = []

    # Fuse
    if len(sources) == 3 and use_gating:
        fused = fusion.fuse(dense_results, stat_results, kg_results, symptom_list, use_gating=True)
    else:
        fused = fusion.fuse_subset(sources, symptom_list)

    # Format context
    context_parts = []
    if dense_results:
        context_parts.append("Semantic retrieval: " + ", ".join([d for d, _ in dense_results[:5]]))
    if stat_results:
        context_parts.append("Statistical co-occurrence: " + ", ".join([d for d, _ in stat_results[:5]]))
    if kg_results:
        context_parts.append("Knowledge graph traversal: " + ", ".join([d for d, _ in kg_results[:5]]))
    context_parts.append("Fused ranking: " + ", ".join([f"{d} ({s:.3f})" for d, s in fused[:10]]))

    context = "\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

Patient information:
{symptoms_text}

Multi-source knowledge graph analysis:
{context}

Based on the patient information and multi-source knowledge graph analysis, provide your differential diagnosis (most likely first):"""
    response = call_llm(prompt)
    return _parse_llm_diagnoses(response)


# Condition registry
CONDITIONS = {
    "B1_LLM_only": {
        "name": "B1: LLM-only",
        "description": "Direct LLM inference without retrieval",
    },
    "B2_Dense_RAG": {
        "name": "B2: LLM + Dense RAG",
        "description": "Standard RAG with dense retrieval",
    },
    "B3_BM25": {
        "name": "B3: LLM + BM25",
        "description": "Statistical retrieval only",
    },
    "Proposed": {
        "name": "Proposed: Multi-Source KG",
        "description": "All 3 sources with learned gating",
    },
    "A1_no_dense": {
        "name": "A1: No dense retrieval",
        "description": "Ablation: remove dense/semantic retrieval",
    },
    "A2_no_statistical": {
        "name": "A2: No statistical retrieval",
        "description": "Ablation: remove BM25/PPMI retrieval",
    },
    "A3_no_kg": {
        "name": "A3: No KG retrieval",
        "description": "Ablation: remove graph traversal",
    },
    "A4_no_gating": {
        "name": "A4: No gating (equal weights)",
        "description": "Ablation: equal weights instead of learned gating",
    },
}
