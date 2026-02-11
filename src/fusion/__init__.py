"""Fusion methods for combining multiple retrieval sources."""

from .rrf import rrf_fusion
from .gating import LearnedGating, GatingTrainer
from .cross_attention import CrossAttentionFusion

__all__ = ["rrf_fusion", "LearnedGating", "GatingTrainer", "CrossAttentionFusion"]
