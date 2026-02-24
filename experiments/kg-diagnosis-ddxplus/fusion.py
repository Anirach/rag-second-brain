"""
Uncertainty-Aware Bayesian Adaptive Fusion (UBAF)

A query-adaptive multi-source retrieval fusion mechanism that combines:
1. Per-query confidence descriptors from each retrieval source
2. Inter-source agreement features (overlap + Rank-Biased Overlap)
3. A lightweight gating network to predict adaptive source weights
4. Bayesian score aggregation via weighted Product of Experts

Designed for the Multi-Source RAG Second Brain paper.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import warnings
import json

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SourceScores:
    """Scores from a single retrieval source for one query."""
    scores: np.ndarray          # shape (N,) — score per candidate disease
    candidate_ids: np.ndarray   # shape (N,) — disease IDs corresponding to scores

    @property
    def ranked_ids(self) -> np.ndarray:
        """Candidate IDs sorted by descending score."""
        order = np.argsort(-self.scores)
        return self.candidate_ids[order]

    @property
    def sorted_scores(self) -> np.ndarray:
        return np.sort(self.scores)[::-1]


@dataclass
class QueryScores:
    """Scores from all three sources for one query."""
    dense: SourceScores
    sparse: SourceScores
    kg: SourceScores

    def sources(self) -> List[Tuple[str, SourceScores]]:
        return [("dense", self.dense), ("sparse", self.sparse), ("kg", self.kg)]


@dataclass
class TrainSample:
    """One training example: query scores + ground-truth disease ID."""
    query_scores: QueryScores
    label: int  # ground-truth disease index (into the common candidate set)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x_scaled = x / max(temperature, 1e-8)
    x_scaled -= x_scaled.max()
    e = np.exp(x_scaled)
    return e / (e.sum() + 1e-12)


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 1e-12]
    return -float(np.sum(p * np.log(p)))


def _rbo(list1: np.ndarray, list2: np.ndarray, p: float = 0.9, depth: int = 20) -> float:
    """Rank-Biased Overlap between two ranked lists (Webber et al., 2010)."""
    depth = min(depth, len(list1), len(list2))
    if depth == 0:
        return 0.0
    agreement = 0.0
    rbo_val = 0.0
    set1, set2 = set(), set()
    for d in range(1, depth + 1):
        set1.add(list1[d - 1])
        set2.add(list2[d - 1])
        agreement = len(set1 & set2) / d
        rbo_val += (p ** (d - 1)) * agreement
    rbo_val *= (1 - p)
    return float(rbo_val)


def extract_source_confidence(src: SourceScores, tau: float = 1.0, theta: float = 0.0) -> np.ndarray:
    """Extract 4-dim confidence descriptor for one source.
    
    Returns: [max_score, gap, entropy, n_above_threshold]
    """
    s = src.sorted_scores
    max_score = float(s[0]) if len(s) > 0 else 0.0
    gap = float(s[0] - s[1]) if len(s) > 1 else max_score
    probs = _softmax(src.scores, temperature=tau)
    ent = _entropy(probs)
    kappa = float(np.sum(src.scores > theta))
    return np.array([max_score, gap, ent, kappa], dtype=np.float64)


def extract_query_features(qs: QueryScores, top_k: int = 10,
                           tau: float = 1.0, theta: float = 0.0) -> np.ndarray:
    """Extract 18-dim feature vector for a query.
    
    Layout: [conf_dense(4), conf_sparse(4), conf_kg(4),
             overlap_ds, overlap_dk, overlap_sk,
             rbo_ds, rbo_dk, rbo_sk]
    """
    sources = [qs.dense, qs.sparse, qs.kg]
    
    # Per-source confidence (4 × 3 = 12)
    confs = [extract_source_confidence(s, tau=tau, theta=theta) for s in sources]
    
    # Top-K sets and ranked lists
    ranked = [s.ranked_ids[:top_k] for s in sources]
    sets = [set(r.tolist()) for r in ranked]
    
    # Pairwise overlap (3)
    pairs = [(0, 1), (0, 2), (1, 2)]
    overlaps = [len(sets[i] & sets[j]) / max(top_k, 1) for i, j in pairs]
    
    # Pairwise RBO (3)
    rbos = [_rbo(ranked[i], ranked[j]) for i, j in pairs]
    
    return np.concatenate(confs + [np.array(overlaps), np.array(rbos)])


# ---------------------------------------------------------------------------
# Gating network (PyTorch)
# ---------------------------------------------------------------------------

if HAS_TORCH:
    class GatingNetwork(nn.Module):
        """Lightweight MLP: R^18 -> softmax(R^3) for per-query source weights."""
        
        def __init__(self, input_dim: int = 18, hidden_dim: int = 32, n_sources: int = 3):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, n_sources),
            )
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Returns source weights in simplex (B, 3)."""
            return torch.softmax(self.net(x), dim=-1)


# ---------------------------------------------------------------------------
# Main fusion class
# ---------------------------------------------------------------------------

class AdaptiveFusion:
    """Uncertainty-Aware Bayesian Adaptive Fusion (UBAF).
    
    Combines per-query adaptive gating with weighted Product-of-Experts
    aggregation and learned source temperatures.
    
    Usage:
        fusion = AdaptiveFusion()
        fusion.fit(train_data)           # list of TrainSample
        result = fusion.fuse(query_scores)  # QueryScores -> dict
    """
    
    def __init__(self,
                 top_k: int = 10,
                 tau: float = 1.0,
                 theta: float = 0.0,
                 hidden_dim: int = 32,
                 lr: float = 1e-3,
                 weight_decay: float = 1e-4,
                 n_epochs: int = 100,
                 batch_size: int = 64,
                 prior: Optional[np.ndarray] = None,
                 backend: str = "auto"):
        """
        Args:
            top_k: Number of top candidates for overlap/RBO features.
            tau: Temperature for softmax in entropy computation.
            theta: Threshold for specificity feature.
            hidden_dim: Hidden layer size for gating MLP.
            lr: Learning rate.
            weight_decay: L2 regularization.
            n_epochs: Training epochs.
            batch_size: Mini-batch size.
            prior: Prior disease probabilities (N,). None = uniform.
            backend: "torch", "sklearn", or "auto".
        """
        self.top_k = top_k
        self.tau = tau
        self.theta = theta
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.prior = prior
        
        # Decide backend
        if backend == "auto":
            self.backend = "torch" if HAS_TORCH else ("sklearn" if HAS_SKLEARN else "uniform")
        else:
            self.backend = backend
        
        # Learned parameters
        self._gating = None
        self._log_temperatures = None  # log(sigma_m) for 3 sources
        self._scaler = None
        self._fitted = False
        
        # Default temperatures (will be learned)
        self._temperatures = np.ones(3, dtype=np.float64)
    
    def _extract_features(self, qs: QueryScores) -> np.ndarray:
        return extract_query_features(qs, top_k=self.top_k, tau=self.tau, theta=self.theta)
    
    def _build_score_matrix(self, qs: QueryScores) -> Tuple[np.ndarray, np.ndarray]:
        """Build aligned score matrix (N, 3) over the union of candidates.
        
        Returns: (score_matrix, candidate_ids)
        """
        all_ids = set()
        for _, src in qs.sources():
            all_ids.update(src.candidate_ids.tolist())
        all_ids = sorted(all_ids)
        id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
        N = len(all_ids)
        
        S = np.zeros((N, 3), dtype=np.float64)
        for m, (_, src) in enumerate(qs.sources()):
            for cid, score in zip(src.candidate_ids, src.scores):
                S[id_to_idx[int(cid)], m] = score
        
        return S, np.array(all_ids)
    
    # ------- Fitting -------
    
    def fit(self, train_data: List[TrainSample]) -> "AdaptiveFusion":
        """Train the gating network and source temperatures.
        
        Args:
            train_data: List of TrainSample with query scores and labels.
        """
        if len(train_data) == 0:
            warnings.warn("Empty training data; using uniform weights.")
            self._fitted = True
            return self
        
        # Extract features and build score matrices
        features = []
        score_matrices = []
        labels = []
        candidate_id_lists = []
        
        for sample in train_data:
            feat = self._extract_features(sample.query_scores)
            S, cids = self._build_score_matrix(sample.query_scores)
            features.append(feat)
            score_matrices.append(S)
            labels.append(sample.label)
            candidate_id_lists.append(cids)
        
        X = np.stack(features)  # (T, 18)
        
        if self.backend == "torch" and HAS_TORCH:
            self._fit_torch(X, score_matrices, labels, candidate_id_lists)
        elif self.backend == "sklearn" and HAS_SKLEARN:
            self._fit_sklearn(X, score_matrices, labels, candidate_id_lists)
        else:
            # Uniform fallback — just learn temperatures via grid search
            self._fit_uniform(score_matrices, labels, candidate_id_lists)
        
        self._fitted = True
        return self
    
    def _fit_torch(self, X: np.ndarray, score_matrices: List[np.ndarray],
                   labels: List[int], candidate_ids: List[np.ndarray]):
        T = len(labels)
        
        # Normalize features
        self._scaler_mean = X.mean(axis=0)
        self._scaler_std = X.std(axis=0) + 1e-8
        X_norm = (X - self._scaler_mean) / self._scaler_std
        
        X_t = torch.tensor(X_norm, dtype=torch.float32)
        
        # Build gating network
        self._gating = GatingNetwork(input_dim=18, hidden_dim=self.hidden_dim)
        
        # Learnable log-temperatures
        self._log_temperatures = nn.Parameter(torch.zeros(3))
        
        optimizer = optim.Adam(
            list(self._gating.parameters()) + [self._log_temperatures],
            lr=self.lr, weight_decay=self.weight_decay
        )
        
        best_loss = float("inf")
        patience_counter = 0
        
        for epoch in range(self.n_epochs):
            # Shuffle
            perm = np.random.permutation(T)
            epoch_loss = 0.0
            
            for start in range(0, T, self.batch_size):
                batch_idx = perm[start:start + self.batch_size]
                x_batch = X_t[batch_idx]
                
                # Forward: get weights
                alphas = self._gating(x_batch)  # (B, 3)
                temps = torch.exp(self._log_temperatures)  # (3,)
                
                # Compute log-posteriors per sample
                batch_loss = torch.tensor(0.0)
                for i, idx in enumerate(batch_idx):
                    S = torch.tensor(score_matrices[idx], dtype=torch.float32)  # (N, 3)
                    label = labels[idx]
                    
                    # Find label index in candidate set
                    cids = candidate_ids[idx]
                    label_positions = np.where(cids == label)[0]
                    if len(label_positions) == 0:
                        continue
                    label_idx = label_positions[0]
                    
                    # Weighted log-scores: sum_m alpha_m * s_m_i / sigma_m
                    w = alphas[i]  # (3,)
                    log_scores = (S / temps.unsqueeze(0)) * w.unsqueeze(0)  # (N, 3)
                    logits = log_scores.sum(dim=1)  # (N,)
                    
                    # Cross-entropy
                    log_probs = logits - torch.logsumexp(logits, dim=0)
                    batch_loss = batch_loss - log_probs[label_idx]
                
                batch_loss = batch_loss / len(batch_idx)
                
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                
                epoch_loss += batch_loss.item() * len(batch_idx)
            
            epoch_loss /= T
            
            if epoch_loss < best_loss - 1e-4:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter > 15:
                    break
        
        self._gating.eval()
        with torch.no_grad():
            self._temperatures = torch.exp(self._log_temperatures).numpy()
    
    def _fit_sklearn(self, X: np.ndarray, score_matrices: List[np.ndarray],
                     labels: List[int], candidate_ids: List[np.ndarray]):
        """Fallback: learn a logistic regression over discrete weight buckets."""
        from sklearn.preprocessing import StandardScaler
        
        self._scaler = StandardScaler()
        X_norm = self._scaler.fit_transform(X)
        
        # For sklearn fallback: grid-search temperatures, then learn gating
        # as classification over {best source} per query
        best_source_per_query = []
        for i in range(len(labels)):
            S = score_matrices[i]
            cids = candidate_ids[i]
            label_pos = np.where(cids == labels[i])[0]
            if len(label_pos) == 0:
                best_source_per_query.append(0)
                continue
            li = label_pos[0]
            # Which source ranks the label highest?
            source_ranks = []
            for m in range(3):
                rank = int(np.sum(S[:, m] > S[li, m])) + 1
                source_ranks.append(rank)
            best_source_per_query.append(int(np.argmin(source_ranks)))
        
        y = np.array(best_source_per_query)
        
        self._sklearn_model = LogisticRegression(
            multi_class="multinomial", max_iter=500, C=1.0
        )
        self._sklearn_model.fit(X_norm, y)
        
        # Learn temperatures via simple optimization
        self._temperatures = self._optimize_temperatures(
            score_matrices, labels, candidate_ids
        )
    
    def _fit_uniform(self, score_matrices, labels, candidate_ids):
        self._temperatures = self._optimize_temperatures(
            score_matrices, labels, candidate_ids
        )
    
    def _optimize_temperatures(self, score_matrices, labels, candidate_ids,
                               n_grid: int = 10) -> np.ndarray:
        """Grid search over temperatures."""
        best_temps = np.ones(3)
        best_acc = -1.0
        
        candidates = np.logspace(-1, 1, n_grid)
        # Coarse grid (sample 200 combos)
        rng = np.random.RandomState(42)
        for _ in range(200):
            temps = rng.choice(candidates, size=3)
            correct = 0
            for i in range(len(labels)):
                S = score_matrices[i]
                cids = candidate_ids[i]
                label_pos = np.where(cids == labels[i])[0]
                if len(label_pos) == 0:
                    continue
                logits = np.sum(S / temps[None, :] / 3.0, axis=1)
                if np.argmax(logits) == label_pos[0]:
                    correct += 1
            acc = correct / max(len(labels), 1)
            if acc > best_acc:
                best_acc = acc
                best_temps = temps.copy()
        
        return best_temps
    
    # ------- Inference -------
    
    def _predict_weights(self, feat: np.ndarray) -> np.ndarray:
        """Predict per-query source weights (3,)."""
        if self.backend == "torch" and self._gating is not None:
            feat_norm = (feat - self._scaler_mean) / self._scaler_std
            with torch.no_grad():
                x = torch.tensor(feat_norm, dtype=torch.float32).unsqueeze(0)
                w = self._gating(x).squeeze(0).numpy()
            return w
        elif self.backend == "sklearn" and self._sklearn_model is not None:
            feat_norm = self._scaler.transform(feat.reshape(1, -1))
            proba = self._sklearn_model.predict_proba(feat_norm)[0]
            # Map class probabilities to soft weights
            n_classes = len(self._sklearn_model.classes_)
            weights = np.ones(3) / 3.0
            for ci, cls in enumerate(self._sklearn_model.classes_):
                weights[cls] = proba[ci]
            weights /= weights.sum()
            return weights
        else:
            return np.ones(3) / 3.0
    
    def fuse(self, query_scores: QueryScores) -> Dict[str, Any]:
        """Fuse multi-source scores for a single query.
        
        Args:
            query_scores: QueryScores with dense, sparse, kg source scores.
            
        Returns:
            dict with keys:
                - "posterior": np.ndarray (N,) — posterior disease probabilities
                - "candidate_ids": np.ndarray (N,) — disease IDs
                - "ranked_ids": np.ndarray (N,) — disease IDs sorted by posterior (desc)
                - "weights": np.ndarray (3,) — adaptive source weights used
                - "temperatures": np.ndarray (3,) — source temperatures
                - "uncertainty": float — posterior entropy
                - "features": np.ndarray (18,) — query feature vector
        """
        feat = self._extract_features(query_scores)
        weights = self._predict_weights(feat)
        S, cids = self._build_score_matrix(query_scores)
        
        temps = self._temperatures
        
        # Weighted Product of Experts
        logits = np.sum(weights[None, :] * S / temps[None, :], axis=1)  # (N,)
        
        # Add log-prior
        if self.prior is not None:
            # Align prior to candidate IDs
            log_prior = np.zeros(len(cids))
            for i, cid in enumerate(cids):
                if cid < len(self.prior):
                    log_prior[i] = np.log(self.prior[cid] + 1e-12)
            logits += log_prior
        
        # Posterior via softmax
        logits -= logits.max()
        posterior = np.exp(logits)
        posterior /= posterior.sum() + 1e-12
        
        # Uncertainty
        uncertainty = _entropy(posterior)
        
        # Ranking
        order = np.argsort(-posterior)
        
        return {
            "posterior": posterior,
            "candidate_ids": cids,
            "ranked_ids": cids[order],
            "ranked_scores": posterior[order],
            "weights": weights,
            "temperatures": temps,
            "uncertainty": uncertainty,
            "features": feat,
        }
    
    def fuse_batch(self, queries: List[QueryScores]) -> List[Dict[str, Any]]:
        """Fuse a batch of queries."""
        return [self.fuse(q) for q in queries]
    
    # ------- Baselines for comparison -------
    
    @staticmethod
    def rrf_baseline(query_scores: QueryScores, k: int = 60) -> Dict[str, Any]:
        """Standard Reciprocal Rank Fusion baseline."""
        all_ids = set()
        for _, src in query_scores.sources():
            all_ids.update(src.candidate_ids.tolist())
        all_ids = sorted(all_ids)
        
        rrf_scores = np.zeros(len(all_ids))
        id_to_idx = {cid: i for i, cid in enumerate(all_ids)}
        
        for _, src in query_scores.sources():
            ranked = src.ranked_ids
            for rank, cid in enumerate(ranked):
                rrf_scores[id_to_idx[int(cid)]] += 1.0 / (k + rank + 1)
        
        cids = np.array(all_ids)
        order = np.argsort(-rrf_scores)
        
        return {
            "scores": rrf_scores,
            "candidate_ids": cids,
            "ranked_ids": cids[order],
            "weights": np.ones(3) / 3.0,
        }
    
    # ------- Serialization -------
    
    def save(self, path: str):
        """Save model state."""
        state = {
            "backend": self.backend,
            "temperatures": self._temperatures.tolist(),
            "top_k": self.top_k,
            "tau": self.tau,
            "theta": self.theta,
            "fitted": self._fitted,
        }
        if self.backend == "torch" and self._gating is not None:
            state["gating_state"] = {
                k: v.tolist() for k, v in self._gating.state_dict().items()
            }
            state["scaler_mean"] = self._scaler_mean.tolist()
            state["scaler_std"] = self._scaler_std.tolist()
        
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: str) -> "AdaptiveFusion":
        """Load model state."""
        with open(path) as f:
            state = json.load(f)
        
        self._temperatures = np.array(state["temperatures"])
        self._fitted = state["fitted"]
        self.backend = state["backend"]
        
        if self.backend == "torch" and HAS_TORCH and "gating_state" in state:
            self._gating = GatingNetwork(input_dim=18, hidden_dim=self.hidden_dim)
            torch_state = {
                k: torch.tensor(v) for k, v in state["gating_state"].items()
            }
            self._gating.load_state_dict(torch_state)
            self._gating.eval()
            self._scaler_mean = np.array(state["scaler_mean"])
            self._scaler_std = np.array(state["scaler_std"])
        
        return self


# ---------------------------------------------------------------------------
# Quick demo / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    N = 50  # diseases
    
    # Synthetic data
    def make_query(true_label: int) -> QueryScores:
        def make_source(noise: float) -> SourceScores:
            scores = np.random.randn(N) * noise
            scores[true_label] += 2.0  # boost true label
            return SourceScores(scores=scores, candidate_ids=np.arange(N))
        return QueryScores(
            dense=make_source(0.5),
            sparse=make_source(0.8),
            kg=make_source(1.0),
        )
    
    # Generate training data
    train = [TrainSample(make_query(i % N), i % N) for i in range(200)]
    test_queries = [make_query(i % N) for i in range(20)]
    test_labels = [i % N for i in range(20)]
    
    # Train and evaluate
    fusion = AdaptiveFusion(n_epochs=50, backend="auto")
    fusion.fit(train)
    
    correct = 0
    for qs, label in zip(test_queries, test_labels):
        result = fusion.fuse(qs)
        pred = result["ranked_ids"][0]
        if pred == label:
            correct += 1
        
    print(f"UBAF Accuracy: {correct}/{len(test_labels)} = {correct/len(test_labels):.1%}")
    print(f"Learned temperatures: {fusion._temperatures}")
    
    # Compare with RRF
    correct_rrf = 0
    for qs, label in zip(test_queries, test_labels):
        result = AdaptiveFusion.rrf_baseline(qs)
        if result["ranked_ids"][0] == label:
            correct_rrf += 1
    print(f"RRF Accuracy:  {correct_rrf}/{len(test_labels)} = {correct_rrf/len(test_labels):.1%}")
    
    # Show adaptive weights for a sample query
    sample = fusion.fuse(test_queries[0])
    print(f"\nSample query weights: dense={sample['weights'][0]:.3f}, "
          f"sparse={sample['weights'][1]:.3f}, kg={sample['weights'][2]:.3f}")
    print(f"Uncertainty (entropy): {sample['uncertainty']:.3f}")
