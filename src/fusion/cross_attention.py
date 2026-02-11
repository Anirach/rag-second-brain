"""Cross-attention fusion for multi-source retrieval.

This module implements cross-attention between documents from different
retrieval sources, allowing information exchange before final scoring.
"""

from typing import List, Tuple, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class CrossAttentionFusion(nn.Module):
        """Cross-attention fusion between retrieval sources.
        
        This module allows documents from different sources to exchange
        information via cross-attention, potentially capturing complementary
        signals before final scoring.
        
        Attributes:
            doc_dim: Document embedding dimension.
            num_heads: Number of attention heads.
            num_sources: Number of retrieval sources.
        """
        
        def __init__(
            self, 
            doc_dim: int = 768, 
            num_heads: int = 8,
            num_sources: int = 3,
            dropout: float = 0.1
        ):
            """Initialize cross-attention fusion.
            
            Args:
                doc_dim: Document embedding dimension.
                num_heads: Number of attention heads.
                num_sources: Number of retrieval sources.
                dropout: Dropout rate.
            """
            super().__init__()
            
            self.doc_dim = doc_dim
            self.num_heads = num_heads
            self.num_sources = num_sources
            
            # Cross-attention between sources
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=doc_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            
            # Layer norm
            self.layer_norm = nn.LayerNorm(doc_dim)
            
            # Feed-forward for final scoring
            self.score_mlp = nn.Sequential(
                nn.Linear(doc_dim, doc_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(doc_dim // 2, 1)
            )
            
            # Query-document attention for final scoring
            self.query_doc_attention = nn.MultiheadAttention(
                embed_dim=doc_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
        
        def forward(
            self, 
            query_emb: torch.Tensor,
            doc_embs_per_source: torch.Tensor,
            source_mask: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Forward pass.
            
            Args:
                query_emb: Query embeddings [batch, query_dim].
                doc_embs_per_source: Document embeddings from each source
                    [batch, num_sources, num_docs_per_source, doc_dim].
                source_mask: Optional mask for padding [batch, num_sources, num_docs].
                
            Returns:
                Tuple of (scores, attention_weights).
                scores: [batch, total_docs]
                attention_weights: Cross-attention weights
            """
            batch_size = query_emb.shape[0]
            num_sources = doc_embs_per_source.shape[1]
            num_docs_per_source = doc_embs_per_source.shape[2]
            
            # Flatten sources for cross-attention
            # [batch, num_sources * num_docs, doc_dim]
            all_docs = doc_embs_per_source.view(
                batch_size, num_sources * num_docs_per_source, -1
            )
            
            # Cross-attention: docs attend to each other
            attended_docs, attn_weights = self.cross_attention(
                all_docs, all_docs, all_docs
            )
            
            # Residual connection + layer norm
            attended_docs = self.layer_norm(all_docs + attended_docs)
            
            # Query-document attention
            query_emb_expanded = query_emb.unsqueeze(1)  # [batch, 1, dim]
            
            query_attended, _ = self.query_doc_attention(
                query_emb_expanded,
                attended_docs,
                attended_docs
            )
            
            # Score each document
            # Combine query-attended representation with doc embeddings
            combined = attended_docs * query_attended.expand_as(attended_docs)
            scores = self.score_mlp(combined).squeeze(-1)  # [batch, total_docs]
            
            return scores, attn_weights
        
        def fuse_with_gating(
            self,
            query_emb: torch.Tensor,
            doc_embs_per_source: torch.Tensor,
            gating_weights: torch.Tensor,
            alpha: float = 0.5
        ) -> torch.Tensor:
            """Combine cross-attention scores with gating weights.
            
            Args:
                query_emb: Query embeddings.
                doc_embs_per_source: Document embeddings per source.
                gating_weights: Weights from learned gating [batch, num_sources].
                alpha: Interpolation between cross-attn and gating.
                
            Returns:
                Final fused scores [batch, total_docs].
            """
            # Cross-attention scores
            cross_attn_scores, _ = self.forward(query_emb, doc_embs_per_source)
            
            # Apply gating weights to source-wise scores
            batch_size = query_emb.shape[0]
            num_sources = doc_embs_per_source.shape[1]
            num_docs = doc_embs_per_source.shape[2]
            
            # Reshape for per-source weighting
            scores_per_source = cross_attn_scores.view(batch_size, num_sources, num_docs)
            
            # Weight by gating
            weighted_scores = scores_per_source * gating_weights.unsqueeze(-1)
            gated_scores = weighted_scores.sum(dim=1)  # [batch, num_docs]
            
            # Also get raw cross-attention aggregate
            raw_scores = cross_attn_scores.view(batch_size, num_sources, num_docs).mean(dim=1)
            
            # Interpolate
            final_scores = alpha * gated_scores + (1 - alpha) * raw_scores
            
            return final_scores

else:
    class CrossAttentionFusion:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for CrossAttentionFusion")
