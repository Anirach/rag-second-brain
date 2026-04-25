"""Retrieval modules: Dense, PPMI, and KG+OWL.

Keep optional heavy dependencies from breaking lightweight imports.
"""

from .dense.encoder import DenseRetriever
from .ppmi.builder import PPMIRetriever

try:
    from .kg.graph_builder import KnowledgeGraphRetriever
except ImportError:  # optional dependency path (e.g. networkx unavailable)
    KnowledgeGraphRetriever = None

__all__ = ["DenseRetriever", "PPMIRetriever", "KnowledgeGraphRetriever"]
