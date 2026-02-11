"""Learned gating for adaptive source weighting.

This module implements query-dependent learned gating to adaptively
weight different retrieval sources based on query characteristics.

Two gating variants:
1. Sigmoid gating: Independent weights per source
2. Softmax gating: Normalized weights that sum to 1
"""

from typing import List, Tuple, Dict, Optional
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


if HAS_TORCH:
    class LearnedGating(nn.Module):
        """Query-dependent learned gating network.
        
        This network learns to weight different retrieval sources based
        on the query embedding. It can use either sigmoid (independent)
        or softmax (normalized) gating.
        
        Attributes:
            num_sources: Number of retrieval sources to combine.
            hidden_dim: Hidden layer dimension.
            gating_type: "sigmoid" or "softmax".
        """
        
        def __init__(
            self, 
            query_dim: int = 768,
            num_sources: int = 3,
            hidden_dim: int = 64,
            gating_type: str = "sigmoid",
            dropout: float = 0.1
        ):
            """Initialize gating network.
            
            Args:
                query_dim: Dimension of query embeddings.
                num_sources: Number of retrieval sources.
                hidden_dim: Hidden layer dimension.
                gating_type: "sigmoid" or "softmax".
                dropout: Dropout rate.
            """
            super().__init__()
            
            self.num_sources = num_sources
            self.gating_type = gating_type
            
            self.network = nn.Sequential(
                nn.Linear(query_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, num_sources)
            )
            
            # Initialize with small weights for stable training
            self._init_weights()
        
        def _init_weights(self):
            """Initialize weights for stable training."""
            for module in self.network:
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    nn.init.zeros_(module.bias)
        
        def forward(
            self, 
            query_emb: torch.Tensor,
            source_scores: Optional[torch.Tensor] = None
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """Compute gating weights and optionally apply to scores.
            
            Args:
                query_emb: Query embeddings [batch_size, query_dim].
                source_scores: Optional source scores [batch_size, num_docs, num_sources].
                
            Returns:
                Tuple of (weights, fused_scores or None).
                weights: [batch_size, num_sources]
                fused_scores: [batch_size, num_docs] if source_scores provided
            """
            # Compute raw gating logits
            logits = self.network(query_emb)  # [batch, num_sources]
            
            # Apply gating function
            if self.gating_type == "sigmoid":
                weights = torch.sigmoid(logits)
                # Normalize to sum to 1 for interpretability
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
            else:  # softmax
                weights = F.softmax(logits, dim=-1)
            
            # Apply weights to scores if provided
            if source_scores is not None:
                # source_scores: [batch, num_docs, num_sources]
                # weights: [batch, num_sources]
                fused = (source_scores * weights.unsqueeze(1)).sum(dim=-1)
                return weights, fused
            
            return weights, None
        
        def get_weights_for_query(
            self, 
            query_emb: np.ndarray
        ) -> np.ndarray:
            """Get gating weights for a single query (numpy interface).
            
            Args:
                query_emb: Query embedding [query_dim].
                
            Returns:
                Weights [num_sources].
            """
            self.eval()
            with torch.no_grad():
                query_tensor = torch.tensor(query_emb).unsqueeze(0).float()
                weights, _ = self.forward(query_tensor)
                return weights.squeeze(0).numpy()


    class GatingTrainer:
        """Trainer for learned gating network.
        
        Uses contrastive learning with positive/negative document pairs
        to train the gating network to rank relevant documents higher.
        """
        
        def __init__(
            self, 
            model: LearnedGating,
            lr: float = 1e-4,
            weight_decay: float = 0.01,
            margin: float = 0.1
        ):
            """Initialize trainer.
            
            Args:
                model: LearnedGating model.
                lr: Learning rate.
                weight_decay: L2 regularization.
                margin: Margin for ranking loss.
            """
            self.model = model
            self.optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=lr, 
                weight_decay=weight_decay
            )
            self.loss_fn = nn.MarginRankingLoss(margin=margin)
            self.history: List[Dict] = []
        
        def train_step(
            self, 
            query_emb: torch.Tensor,
            source_scores: torch.Tensor,
            positive_mask: torch.Tensor,
            negative_mask: torch.Tensor
        ) -> Dict[str, float]:
            """Single training step.
            
            Args:
                query_emb: Query embeddings [batch, query_dim].
                source_scores: Source scores [batch, num_docs, num_sources].
                positive_mask: Mask for positive docs [batch, num_docs].
                negative_mask: Mask for negative docs [batch, num_docs].
                
            Returns:
                Dictionary with loss and weight statistics.
            """
            self.model.train()
            self.optimizer.zero_grad()
            
            # Forward pass
            weights, fused_scores = self.model(query_emb, source_scores)
            
            # Compute ranking loss
            # Get mean score for positive and negative docs
            pos_scores = (fused_scores * positive_mask).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-8)
            neg_scores = (fused_scores * negative_mask).sum(dim=1) / (negative_mask.sum(dim=1) + 1e-8)
            
            loss = self.loss_fn(
                pos_scores, 
                neg_scores, 
                torch.ones_like(pos_scores)
            )
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            
            # Statistics
            stats = {
                "loss": loss.item(),
                "weight_dense": weights[:, 0].mean().item(),
                "weight_ppmi": weights[:, 1].mean().item() if weights.shape[1] > 1 else 0,
                "weight_kg": weights[:, 2].mean().item() if weights.shape[1] > 2 else 0,
            }
            self.history.append(stats)
            
            return stats
        
        def evaluate(
            self,
            query_embs: torch.Tensor,
            source_scores: torch.Tensor,
            labels: torch.Tensor,
            k: int = 10
        ) -> Dict[str, float]:
            """Evaluate model on a batch.
            
            Args:
                query_embs: Query embeddings.
                source_scores: Source scores.
                labels: Binary relevance labels [batch, num_docs].
                k: Cutoff for Recall@K.
                
            Returns:
                Evaluation metrics.
            """
            self.model.eval()
            
            with torch.no_grad():
                weights, fused_scores = self.model(query_embs, source_scores)
                
                # Compute Recall@K
                _, top_k_indices = fused_scores.topk(k, dim=1)
                
                recalls = []
                for i in range(len(labels)):
                    relevant = labels[i].nonzero().squeeze(-1).tolist()
                    if not isinstance(relevant, list):
                        relevant = [relevant]
                    retrieved = top_k_indices[i].tolist()
                    
                    if relevant:
                        recall = len(set(relevant) & set(retrieved)) / len(relevant)
                        recalls.append(recall)
                
                return {
                    "recall@k": np.mean(recalls) if recalls else 0,
                    "mean_weight_dense": weights[:, 0].mean().item(),
                    "mean_weight_ppmi": weights[:, 1].mean().item() if weights.shape[1] > 1 else 0,
                    "mean_weight_kg": weights[:, 2].mean().item() if weights.shape[1] > 2 else 0,
                }

else:
    # Stub classes when PyTorch not available
    class LearnedGating:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for LearnedGating")
    
    class GatingTrainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GatingTrainer")
