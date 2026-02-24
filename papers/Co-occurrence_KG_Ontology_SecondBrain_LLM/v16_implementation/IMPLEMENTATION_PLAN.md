# RAG Second Brain v16 — Full Implementation Plan

> **Goal:** Implement real PPMI + KG+OWL components, run rigorous experiments, address all reviewer concerns
> **Timeline:** 8-10 weeks
> **Cost:** ~$0-50 (using open-source)

---

## Phase 1: Infrastructure Setup (Week 1)

### 1.1 Repository Structure
```
rag-second-brain/
├── src/
│   ├── retrieval/
│   │   ├── dense/           # E5/BGE embeddings
│   │   ├── ppmi/            # Real PPMI co-occurrence
│   │   └── kg/              # KG+OWL retrieval
│   ├── fusion/
│   │   ├── rrf.py           # Reciprocal Rank Fusion
│   │   ├── gating.py        # Learned sigmoid/softmax gating
│   │   └── cross_attention.py
│   ├── evaluation/
│   │   ├── retrieval.py     # Recall@K metrics
│   │   └── qa.py            # EM/F1 end-to-end
│   └── utils/
├── data/
│   ├── hotpotqa/
│   ├── nq/
│   └── pkm_synthetic/
├── experiments/
│   ├── configs/
│   └── results/
├── notebooks/
└── paper/
```

### 1.2 Environment
```bash
# Core dependencies
pip install torch transformers sentence-transformers
pip install faiss-cpu  # or faiss-gpu
pip install owlready2  # OWL reasoning
pip install networkx   # Graph operations
pip install scipy      # PPMI computation
pip install datasets   # HuggingFace datasets
pip install wandb      # Experiment tracking
```

---

## Phase 2: Dense Retrieval Module (Week 1-2)

### Implementation
```python
# src/retrieval/dense/encoder.py
from sentence_transformers import SentenceTransformer

class DenseRetriever:
    def __init__(self, model_name="intfloat/e5-large-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None  # FAISS index
    
    def build_index(self, documents: List[str]):
        embeddings = self.model.encode(documents, normalize_embeddings=True)
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], normalize_embeddings=True)
        scores, indices = self.index.search(q_emb, k)
        return list(zip(indices[0], scores[0]))
```

### Models to Test
| Model | Dim | Speed | Quality |
|-------|-----|-------|---------|
| E5-large-v2 | 1024 | Medium | High |
| BGE-large-en-v1.5 | 1024 | Medium | High |
| E5-base-v2 | 768 | Fast | Good |

---

## Phase 3: PPMI Co-occurrence Module (Week 2-3) ⭐ NEW

### 3.1 Build Co-occurrence Matrix
```python
# src/retrieval/ppmi/builder.py
from collections import defaultdict
from scipy.sparse import csr_matrix
import numpy as np

class PPMIRetriever:
    def __init__(self, window_size=5, min_count=5):
        self.window_size = window_size
        self.min_count = min_count
        self.vocab = {}
        self.ppmi_matrix = None
        self.doc_term_matrix = None
    
    def build_from_corpus(self, documents: List[str]):
        # Step 1: Build vocabulary
        word_counts = defaultdict(int)
        for doc in documents:
            for word in self._tokenize(doc):
                word_counts[word] += 1
        
        self.vocab = {w: i for i, (w, c) in enumerate(
            [(w, c) for w, c in word_counts.items() if c >= self.min_count]
        )}
        
        # Step 2: Build co-occurrence matrix
        cooc = defaultdict(lambda: defaultdict(int))
        for doc in documents:
            tokens = [t for t in self._tokenize(doc) if t in self.vocab]
            for i, w1 in enumerate(tokens):
                for j in range(max(0, i-self.window_size), 
                               min(len(tokens), i+self.window_size+1)):
                    if i != j:
                        cooc[self.vocab[w1]][self.vocab[tokens[j]]] += 1
        
        # Step 3: Convert to PPMI
        self.ppmi_matrix = self._compute_ppmi(cooc)
        
        # Step 4: Build document-term matrix for retrieval
        self._build_doc_term_matrix(documents)
    
    def _compute_ppmi(self, cooc) -> csr_matrix:
        """Positive Pointwise Mutual Information"""
        total = sum(sum(row.values()) for row in cooc.values())
        word_totals = defaultdict(int)
        for w1, row in cooc.items():
            for w2, count in row.items():
                word_totals[w1] += count
                word_totals[w2] += count
        
        rows, cols, data = [], [], []
        for w1, row in cooc.items():
            for w2, count in row.items():
                pmi = np.log2((count * total) / 
                             (word_totals[w1] * word_totals[w2]) + 1e-10)
                ppmi = max(0, pmi)  # Positive PMI
                if ppmi > 0:
                    rows.append(w1)
                    cols.append(w2)
                    data.append(ppmi)
        
        return csr_matrix((data, (rows, cols)), 
                         shape=(len(self.vocab), len(self.vocab)))
    
    def retrieve(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Retrieve documents using PPMI-weighted query expansion"""
        query_tokens = [t for t in self._tokenize(query) if t in self.vocab]
        query_indices = [self.vocab[t] for t in query_tokens]
        
        # Expand query using PPMI-weighted related terms
        expanded_weights = np.zeros(len(self.vocab))
        for idx in query_indices:
            expanded_weights += self.ppmi_matrix[idx].toarray().flatten()
            expanded_weights[idx] += 1.0  # Original term weight
        
        # Score documents
        doc_scores = self.doc_term_matrix.dot(expanded_weights)
        top_k = np.argsort(doc_scores)[-k:][::-1]
        
        return [(int(idx), float(doc_scores[idx])) for idx in top_k]
```

### 3.2 Evaluation Metrics
- Vocabulary size vs coverage trade-off
- Window size ablation (3, 5, 7, 10)
- PPMI vs PMI vs raw co-occurrence

---

## Phase 4: KG + OWL Module (Week 3-5) ⭐ NEW

### 4.1 Knowledge Graph Construction
```python
# src/retrieval/kg/graph_builder.py
import networkx as nx
from owlready2 import *

class KnowledgeGraphRetriever:
    def __init__(self, ontology_path: str = None):
        self.graph = nx.DiGraph()
        self.entity_to_docs = defaultdict(set)
        self.ontology = None
        
        if ontology_path:
            self.ontology = get_ontology(ontology_path).load()
    
    def add_entities_from_documents(self, documents: List[str], 
                                     entity_linker):
        """Extract and link entities from documents"""
        for doc_id, doc in enumerate(documents):
            entities = entity_linker.extract(doc)
            for entity in entities:
                self.entity_to_docs[entity['id']].add(doc_id)
                # Add to graph with Wikidata relations
                self._add_entity_to_graph(entity)
    
    def _add_entity_to_graph(self, entity):
        """Add entity and its Wikidata relations to graph"""
        self.graph.add_node(entity['id'], 
                          label=entity['label'],
                          types=entity.get('types', []))
        
        # Fetch relations from Wikidata (cached)
        relations = self._fetch_wikidata_relations(entity['id'])
        for rel, target in relations:
            self.graph.add_edge(entity['id'], target, relation=rel)
    
    def owl_materialize(self, entity_ids: List[str]) -> List[str]:
        """Use OWL reasoning to infer additional entities"""
        if not self.ontology:
            return entity_ids
        
        inferred = set(entity_ids)
        with self.ontology:
            # Run reasoner
            sync_reasoner_pellet(infer_property_values=True)
            
            for eid in entity_ids:
                # Get entity's classes
                entity = self.ontology.search_one(iri=f"*{eid}")
                if entity:
                    # Add superclasses (subsumption)
                    for cls in entity.is_a:
                        inferred.update(self._get_instances(cls))
                    
                    # Add related via object properties
                    for prop in self.ontology.object_properties():
                        related = getattr(entity, prop.python_name, [])
                        inferred.update(r.name for r in related)
        
        return list(inferred)
    
    def retrieve(self, query: str, entity_linker, k: int = 10,
                 use_ppr: bool = True, use_owl: bool = True) -> List[Tuple[int, float]]:
        """Retrieve documents via KG traversal"""
        # Step 1: Extract query entities
        query_entities = entity_linker.extract(query)
        seed_ids = [e['id'] for e in query_entities]
        
        # Step 2: OWL materialization (optional)
        if use_owl and self.ontology:
            seed_ids = self.owl_materialize(seed_ids)
        
        # Step 3: PPR traversal
        if use_ppr and len(seed_ids) > 0:
            personalization = {eid: 1.0/len(seed_ids) for eid in seed_ids 
                              if eid in self.graph}
            if personalization:
                ppr_scores = nx.pagerank(self.graph, 
                                        personalization=personalization,
                                        alpha=0.85)
            else:
                ppr_scores = {eid: 1.0 for eid in seed_ids}
        else:
            ppr_scores = {eid: 1.0 for eid in seed_ids}
        
        # Step 4: Entity-to-document scoring with IDF weighting
        doc_scores = defaultdict(float)
        total_docs = len(set.union(*self.entity_to_docs.values())) if self.entity_to_docs else 1
        
        for eid, ppr_score in ppr_scores.items():
            if eid in self.entity_to_docs:
                docs = self.entity_to_docs[eid]
                idf = np.log(total_docs / (len(docs) + 1))
                for doc_id in docs:
                    doc_scores[doc_id] += ppr_score * idf
        
        # Return top-k
        sorted_docs = sorted(doc_scores.items(), key=lambda x: -x[1])[:k]
        return sorted_docs
```

### 4.2 Entity Linking Options
| Tool | Speed | Quality | Cost |
|------|-------|---------|------|
| spaCy + Wikidata | Fast | Good | Free |
| REL (Radboud) | Medium | High | Free |
| BLINK (Facebook) | Slow | Highest | Free |

### 4.3 Ontology Setup
```python
# Use schema.org + custom PKM ontology
from owlready2 import *

onto = get_ontology("http://example.org/pkm.owl")

with onto:
    class Document(Thing): pass
    class Person(Thing): pass
    class Concept(Thing): pass
    class Project(Thing): pass
    
    class mentions(Document >> Person): pass
    class relatedTo(Concept >> Concept): pass
    class partOf(Document >> Project): pass
    
    # Reasoning rules
    class TransitiveRelation(relatedTo):
        is_a = [TransitiveProperty]
```

---

## Phase 5: Fusion Methods (Week 5-6)

### 5.1 RRF Baseline
```python
def rrf_fusion(rankings: List[List[Tuple[int, float]]], k: int = 60) -> List[Tuple[int, float]]:
    """Reciprocal Rank Fusion"""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])
```

### 5.2 Learned Gating ⭐
```python
# src/fusion/gating.py
import torch
import torch.nn as nn

class LearnedGating(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64):
        super().__init__()
        self.query_encoder = nn.Sequential(
            nn.Linear(768, hidden_dim),  # Query embedding dim
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, query_emb, source_scores):
        """
        query_emb: [batch, 768] - query embedding
        source_scores: [batch, num_docs, 3] - scores from 3 sources
        """
        # Compute query-dependent weights
        weights = torch.sigmoid(self.query_encoder(query_emb))  # [batch, 3]
        weights = weights / weights.sum(dim=-1, keepdim=True)   # Normalize
        
        # Apply weights to source scores
        fused = (source_scores * weights.unsqueeze(1)).sum(dim=-1)  # [batch, num_docs]
        return fused, weights


class GatingTrainer:
    def __init__(self, model, lr=1e-4):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.loss_fn = nn.MarginRankingLoss(margin=0.1)
    
    def train_step(self, query_emb, source_scores, positive_idx, negative_idx):
        """Contrastive training with positive/negative document pairs"""
        self.optimizer.zero_grad()
        
        fused, weights = self.model(query_emb, source_scores)
        
        pos_scores = fused[:, positive_idx]
        neg_scores = fused[:, negative_idx]
        
        loss = self.loss_fn(pos_scores, neg_scores, 
                           torch.ones_like(pos_scores))
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), weights.detach()
```

### 5.3 Cross-Attention Fusion
```python
# src/fusion/cross_attention.py
class CrossAttentionFusion(nn.Module):
    def __init__(self, doc_dim=768, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(doc_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(doc_dim, doc_dim),
            nn.ReLU(),
            nn.Linear(doc_dim, 1)
        )
    
    def forward(self, query_emb, doc_embs_per_source):
        """
        query_emb: [1, 768]
        doc_embs_per_source: [3, num_docs, 768] - embeddings from each source
        """
        # Cross-attention between sources
        attended, _ = self.cross_attn(
            doc_embs_per_source, 
            doc_embs_per_source, 
            doc_embs_per_source
        )
        
        # Query-document scoring
        scores = self.mlp(attended * query_emb).squeeze(-1)
        return scores.mean(dim=0)  # Aggregate across sources
```

---

## Phase 6: Experiments (Week 6-8)

### 6.1 Datasets

| Dataset | Type | Size | Purpose |
|---------|------|------|---------|
| HotpotQA (distractor) | Multi-hop QA | 113K docs | Main benchmark |
| Natural Questions | Single-hop | 21M docs | Generalization |
| Synthetic PKM | Personal KB | ~10K docs | PKM validation |

### 6.2 Baselines to Implement

| Baseline | Paper | Implementation |
|----------|-------|----------------|
| DPR | Karpukhin 2020 | HuggingFace |
| ColBERTv2 | Santhanam 2022 | Official repo |
| MDR | Xiong 2021 | Official repo |
| BM25+DPR Hybrid | - | Custom |
| HiRAG | Wu 2024 | Custom |

### 6.3 Metrics

**Retrieval:**
- Recall@K (K=5,10,20,50,100)
- **Both-support Recall@K** (HotpotQA-specific)
- MRR
- NDCG@10

**End-to-End QA:**
- Answer EM
- Answer F1
- Supporting Fact EM
- Supporting Fact F1

### 6.4 Experiment Grid

```yaml
# experiments/configs/main.yaml
experiments:
  - name: "ablation_sources"
    sources: [[dense], [ppmi], [kg], [dense,ppmi], [dense,kg], [ppmi,kg], [dense,ppmi,kg]]
    fusion: [rrf, learned_gating]
    
  - name: "gating_variants"
    gating: [sigmoid, softmax, learned_weights]
    
  - name: "owl_ablation"
    kg_config:
      use_owl: [true, false]
      use_ppr: [true, false]
      
  - name: "cross_attention"
    fusion: [gating_only, cross_attn_only, gating+cross_attn]
```

### 6.5 Training Protocol (Reviewer Requirement)

```yaml
training:
  # Data splits
  train_split: "train[:80%]"
  val_split: "train[80%:90%]"
  test_split: "validation"  # HotpotQA validation as test
  
  # Gating training
  gating:
    epochs: 20
    batch_size: 32
    lr: 1e-4
    weight_decay: 0.01
    early_stopping_patience: 3
    
  # Cross-validation
  cv_folds: 5
  
  # Seeds for reproducibility
  seeds: [42, 123, 456, 789, 1000]
  
  # Hardware
  gpu: "NVIDIA RTX 3090" or "Google Colab T4"
  
  # Hyperparameters to report
  faiss:
    index_type: "IVF4096,Flat"
    nprobe: 64
  bm25:
    k1: 1.2
    b: 0.75
  ppmi:
    window_size: 5
    min_count: 5
```

---

## Phase 7: Paper Revision (Week 8-10)

### 7.1 New Results Tables

**Table 2: Main Results (HotpotQA Distractor)**
| Method | R@10 | R@20 | Both@10 | Both@20 | Ans EM | Ans F1 |
|--------|------|------|---------|---------|--------|--------|
| BM25 | - | - | - | - | - | - |
| DPR | - | - | - | - | - | - |
| ColBERTv2 | - | - | - | - | - | - |
| MDR | - | - | - | - | - | - |
| **Ours (RRF)** | - | - | - | - | - | - |
| **Ours (Gating)** | - | - | - | - | - | - |

### 7.2 Sections to Rewrite

1. **Section 4: Implementation** — Add real PPMI & KG+OWL details
2. **Section 5: Experiments** — New protocol, clear corpus spec
3. **Section 6: Results** — Real numbers with CIs
4. **Section 7: Analysis** — Cross-attention, OWL ablation

### 7.3 Reviewer Response Checklist

- [ ] Corpus: "Full HotpotQA distractor setting (113,662 passages)"
- [ ] Recall@K: "Both-support Recall@K (both gold paragraphs in top-K)"
- [ ] Table consistency: Audit all tables
- [ ] Training details: Full protocol in Appendix
- [ ] Hyperparameters: All settings documented
- [ ] Seeds: Report mean ± std over 5 seeds
- [ ] PPMI implemented: Real co-occurrence, not BM25 proxy
- [ ] KG+OWL implemented: Real entity linking + reasoning
- [ ] Cross-attention tested: Ablation included
- [ ] Baselines: MDR, ColBERTv2, hybrid
- [ ] End-to-end QA: EM/F1 reported

---

## Timeline Summary

| Week | Phase | Deliverable |
|------|-------|-------------|
| 1 | Setup | Repo structure, environment |
| 1-2 | Dense | E5/BGE retriever + FAISS |
| 2-3 | PPMI | Real co-occurrence module |
| 3-5 | KG+OWL | Entity linking + reasoning |
| 5-6 | Fusion | Gating + cross-attention |
| 6-8 | Experiments | Full benchmark suite |
| 8-10 | Paper | v16 revision |

---

## Quick Start Commands

```bash
# Clone and setup
git clone https://github.com/Anirach/rag-second-brain
cd rag-second-brain
pip install -r requirements.txt

# Download HotpotQA
python scripts/download_data.py --dataset hotpotqa

# Build indices
python scripts/build_dense_index.py --model e5-large-v2
python scripts/build_ppmi_index.py --window 5
python scripts/build_kg_index.py --entity-linker rel

# Run experiments
python experiments/run_all.py --config configs/main.yaml

# Generate paper tables
python scripts/generate_tables.py --results experiments/results/
```

---

**Ready to start Phase 1?** 🚀
