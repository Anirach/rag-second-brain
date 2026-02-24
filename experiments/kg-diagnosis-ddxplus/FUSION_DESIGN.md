# Uncertainty-Aware Bayesian Adaptive Fusion (UBAF)

## 1. Motivation

Standard multi-source retrieval fusion uses static combination rules — Reciprocal Rank Fusion (RRF) with fixed $k$, or linear combinations with uniform weights. These approaches treat all queries identically, ignoring that **source reliability varies per query**. A symptom-heavy query may favor BM25+PPMI lexical matching, while a rare-disease query benefits from KG traversal. UBAF adapts fusion weights per-query by modeling each source's confidence as a signal of its local reliability, then aggregating via Bayesian posterior inference.

## 2. Problem Setup

Given a query $q$, three retrieval sources produce candidate score vectors over $N$ diseases:

- **Dense retrieval**: $\mathbf{s}^{(d)} \in \mathbb{R}^N$ (cosine similarities from bi-encoder)
- **Sparse retrieval (BM25+PPMI)**: $\mathbf{s}^{(s)} \in \mathbb{R}^N$ (BM25 scores with PPMI-weighted expansion)
- **KG traversal**: $\mathbf{s}^{(k)} \in \mathbb{R}^N$ (aggregated edge weights from multi-hop paths)

We seek a fused score $\mathbf{s}^* \in \mathbb{R}^N$ that is **query-adaptive** and **uncertainty-calibrated**.

## 3. Mathematical Formulation

### 3.1 Source Confidence Estimation

For each source $m \in \{d, s, k\}$ and query $q$, we compute a confidence descriptor $\mathbf{c}^{(m)}_q \in \mathbb{R}^4$:

$$\mathbf{c}^{(m)}_q = \left[ \max(\mathbf{s}^{(m)}), \; \text{gap}^{(m)}, \; H^{(m)}, \; \kappa^{(m)} \right]$$

where:
- $\max(\mathbf{s}^{(m)})$ = top retrieval score (peak confidence)
- $\text{gap}^{(m)} = s^{(m)}_{[1]} - s^{(m)}_{[2]}$ = score gap between rank-1 and rank-2 (decisiveness)
- $H^{(m)} = -\sum_i \tilde{s}^{(m)}_i \log \tilde{s}^{(m)}_i$ = entropy of normalized scores (uncertainty), where $\tilde{s}^{(m)} = \text{softmax}(\mathbf{s}^{(m)} / \tau)$
- $\kappa^{(m)} = |\{i : s^{(m)}_i > \theta\}|$ = number of candidates above threshold (specificity)

### 3.2 Inter-Source Agreement Features

We compute pairwise agreement between sources' top-$K$ candidate sets $\mathcal{T}^{(m)}$:

$$\text{overlap}_{ij} = \frac{|\mathcal{T}^{(i)} \cap \mathcal{T}^{(j)}|}{K}, \quad \rho_{ij} = \text{RBO}(\pi^{(i)}, \pi^{(j)})$$

where $\text{RBO}$ is Rank-Biased Overlap (Webber et al., 2010) between the ranked lists $\pi^{(i)}, \pi^{(j)}$.

The full query feature vector is:

$$\mathbf{x}_q = [\mathbf{c}^{(d)}_q; \; \mathbf{c}^{(s)}_q; \; \mathbf{c}^{(k)}_q; \; \text{overlap}_{ds}; \; \text{overlap}_{dk}; \; \text{overlap}_{sk}; \; \rho_{ds}; \; \rho_{dk}; \; \rho_{sk}] \in \mathbb{R}^{18}$$

### 3.3 Adaptive Weight Prediction

A lightweight gating network $g_\phi$ predicts per-query source weights:

$$\boldsymbol{\alpha}_q = \text{softmax}(g_\phi(\mathbf{x}_q)) \in \Delta^2$$

where $g_\phi$ is a 2-layer MLP: $\mathbb{R}^{18} \to \mathbb{R}^{32} \to \mathbb{R}^3$ with ReLU activation.

### 3.4 Bayesian Score Aggregation (Weighted Product of Experts)

Rather than a simple weighted sum, we interpret each source's normalized scores as a **likelihood** for disease relevance. Given predicted weights $\boldsymbol{\alpha}_q = (\alpha_d, \alpha_s, \alpha_k)$:

**Prior:** Uniform $P(D_i) = 1/N$ (or optionally prevalence-informed).

**Likelihood per source:** We model source $m$'s score for disease $i$ as:

$$P(\mathbf{s}^{(m)} | D_i, \text{relevant}) \propto \exp\left(\frac{s^{(m)}_i}{\sigma_m}\right)$$

where $\sigma_m$ is a learned temperature per source.

**Weighted log-posterior:**

$$\log P(D_i | q) \propto \sum_{m \in \{d,s,k\}} \alpha^{(m)}_q \cdot \frac{s^{(m)}_i}{\sigma_m} + \log P(D_i)$$

$$P(D_i | q) = \frac{\exp\left(\sum_m \alpha^{(m)}_q \cdot s^{(m)}_i / \sigma_m + \log P(D_i)\right)}{\sum_j \exp\left(\sum_m \alpha^{(m)}_q \cdot s^{(m)}_j / \sigma_m + \log P(D_j)\right)}$$

This is a **Product of Experts** (Hinton, 2002) with adaptive expert weights — each source is an "expert" whose influence is modulated by its estimated reliability for the current query.

### 3.5 Uncertainty Quantification

The posterior $P(D_i | q)$ naturally provides calibrated uncertainty:

$$\mathcal{U}(q) = H[P(D | q)] = -\sum_i P(D_i | q) \log P(D_i | q)$$

High $\mathcal{U}(q)$ signals that sources disagree or are individually uncertain → the system can **abstain or request clarification**, a clinically valuable property.

### 3.6 Training Objective

Given training set $\{(q_t, y_t)\}$ where $y_t$ is the ground-truth disease, we minimize:

$$\mathcal{L} = -\frac{1}{T}\sum_{t=1}^T \log P(D_{y_t} | q_t) + \lambda \|\phi\|^2$$

This jointly optimizes the gating network parameters $\phi$ and source temperatures $\sigma_m$.

## 4. Novelty Claims

| Aspect | Standard RRF / Linear | **UBAF (Ours)** |
|---|---|---|
| Weights | Fixed across all queries | Per-query adaptive via gating network |
| Aggregation | Reciprocal rank or weighted sum | Bayesian posterior (Product of Experts) |
| Uncertainty | None | Posterior entropy for abstention |
| Source modeling | Rank-only (ordinal) | Score-calibrated via learned temperatures |
| Features | None | 18-dim confidence + agreement descriptor |

**Key novelty statement:** "We propose UBAF, a query-adaptive fusion mechanism that dynamically modulates source influence based on per-query confidence descriptors and inter-source agreement, aggregating via a weighted Product of Experts with learned temperatures. Unlike static RRF or fixed-weight fusion, UBAF yields calibrated posterior probabilities with natural uncertainty quantification for clinical abstention."

### Comparison to Related Work

- **RRF (Cormack et al., 2009):** Purely rank-based, no score calibration, no adaptivity. $\text{RRF}_i = \sum_m 1/(k + r^{(m)}_i)$.
- **Linear CombSUM/CombMNZ:** Static weights, no uncertainty, no per-query adaptation.
- **Learning-to-Rank (LambdaMART, etc.):** Learns a single global ranking function, not source-weight adaptation.
- **Mixture of Experts (Shazeer et al., 2017):** Our gating mechanism is inspired by MoE but applied to retrieval source fusion rather than neural network sub-modules. The Bayesian interpretation is novel.
- **Bayesian rank aggregation (Klementiev et al., 2008):** Uses Bayesian inference but with fixed source reliability; we make it query-adaptive.

## 5. Ablation Study Design

To validate each component's contribution:

1. **UBAF-full**: Complete model
2. **UBAF-uniform**: $\alpha_m = 1/3$ (no gating) but keep Bayesian aggregation + temperatures
3. **UBAF-no-temp**: Adaptive gating but $\sigma_m = 1$ (no learned temperatures)
4. **UBAF-no-agreement**: Remove inter-source agreement features from $\mathbf{x}_q$
5. **UBAF-no-entropy**: Remove entropy features from confidence descriptors
6. **RRF baseline**: Standard $k=60$
7. **Weighted-sum oracle**: Grid-search optimal fixed weights on dev set

## 6. Implementation

See `fusion.py` for the complete implementation.
