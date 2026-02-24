# Response to Reviewers

## Paper: "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

**Revision Date:** February 9, 2026  
**Original Score:** 5.2/10 (ICLR scale)  
**Decision:** Revise and Resubmit

---

## Summary of Major Changes

We thank the reviewers for their thorough and constructive feedback. This revision addresses all major concerns:

1. **Technical Clarity:** Added explicit novelty statement clarifying that our contribution is the principled integration through learned gating, not the individual components (Abstract, Introduction).

2. **Scalability:** Added Section 3.1.1 detailing three scalability strategies: subword tokenization, sparse representations, and block-sparse SVD.

3. **Ontology Specifics:** Section 3.4 now specifies OWL 2 RL profile and RDFox reasoner with runtime benchmarks.

4. **Experimental Completeness:** Added comprehensive details on LLM backbone (GPT-3.5-turbo-0613), prompt templates (Appendix A), retrieval parameters, formal metric definitions, data leakage mitigation, baseline configurations, and hardware specs.

5. **Related Work:** Added Section 2.2 on GraphRAG methods, Section 2.4 on PMI-GSVD/ROOT-CA variants, Section 2.5 on retrofitting techniques, and comparison table.

6. **Reproducibility:** Added GitHub repository URL and complete configuration details.

---

## Detailed Responses

### Weakness 1: Technical Limitations

> "Many components are standard (PPMI/SVD, Transformer attention, KG embeddings, R-GCN) — novelty is primarily in combination"

**Response:** We agree and have made this explicit in the revised paper. The Abstract now states: *"Our primary contribution is not the individual components—which draw from established techniques—but their principled integration through a learned gating mechanism that dynamically weights each memory source based on query characteristics."*

The Introduction adds a Novelty Statement paragraph explicitly clarifying our three contributions: (1) principled integration via attention-based gating, (2) bidirectional cross-attention, and (3) formal ontological constraints.

**Changes:** Abstract (lines 8-12), Introduction (page 1, Novelty Statement paragraph)

---

> "Scalability of O(|V|²) co-occurrence matrix not resolved — need practical strategies"

**Response:** We added Section 3.1.1 "Scalability Strategies" detailing three approaches:

1. **Subword Tokenization:** BPE/SentencePiece reduces vocabulary from >1M to 32K-50K, yielding ~400× size reduction.
2. **Sparse Representation:** CSR format with typical <5% density reduces from O(|V|²) to O(nnz).
3. **Block-Sparse SVD:** For vocabularies >100K, we use randomized SVD [Halko et al., 2011] achieving O(|V|·k·d) complexity.

**Changes:** Section 3.1.1 (page 3)

---

> "Ontology reasoning too abstract — need concrete DL fragment, reasoner, configuration details"

**Response:** Section 3.4 now specifies:
- **OWL Profile:** OWL 2 RL (decidable, polynomial-time)
- **Reasoner:** RDFox [Nenov et al., 2015] for rule-based materialization
- **Runtime:** 47 seconds for full materialization on 1.2M entities / 4.8M triples
- **Supported Constructors:** Class intersection (⊓), existential/universal restrictions (∃, ∀), transitive properties, subsumption

**Changes:** Section 3.4 (page 4)

---

### Weakness 2: Experimental Gaps

> "Missing LLM backbone details (which model?)"

**Response:** Section 4.1.1 now specifies: **GPT-3.5-turbo** (version `gpt-3.5-turbo-0613`), accessed via OpenAI API, with temperature 0.0, context length 4,096 tokens (3,500 for retrieval after prompt overhead), and max output 256 tokens. Appendix B includes comparison with GPT-4-turbo and Llama-2-70B-chat.

**Changes:** Section 4.1.1 (page 4), Appendix B (page 9)

---

> "Missing prompt templates"

**Response:** Appendix A provides complete prompt templates for:
- Question answering (with system prompt and retrieved context injection)
- Fact verification (SUPPORTS/REFUTES/NOT ENOUGH INFO)

**Changes:** Appendix A (page 9)

---

> "Missing context length specifications"

**Response:** Section 4.1.1 specifies: 4,096 total tokens, 3,500 allocated to retrieved context.

**Changes:** Section 4.1.1

---

> "Missing retrieval parameters"

**Response:** Section 4.1.2 provides:
- Top-k: 5 passages
- ANN Index: FAISS IVF-PQ (1024 centroids, 64 sub-quantizers)
- Embedding Model: Sentence-T5-XL (768-dim)
- KG Traversal Depth: Maximum 3 hops

**Changes:** Section 4.1.2 (page 4)

---

> "Missing KG construction scope"

**Response:** Section 4.1.3 specifies:
- WikiData subset: 1.2M entities, 4.8M triples (covering entities in evaluation datasets)
- ConceptNet 5.7: 850K commonsense triples
- Ontology: Schema.org + custom domain (1,247 classes, 892 properties)

**Changes:** Section 4.1.3 (page 5)

---

> "Missing gating network training details"

**Response:** Section 3.5.1 now provides complete training details:
- **Objective:** Contrastive loss with temperature τ=0.07
- **Negatives:** In-batch negatives + hard negatives from BM25
- **Optimizer:** AdamW (lr=5×10⁻⁵)
- **Batch Size:** 128
- **Epochs:** 10
- **Training Data:** 50K query-answer pairs from Natural Questions training set

**Changes:** Section 3.5.1 (page 4)

---

> "Factual consistency and hallucination metrics undefined"

**Response:** Section 4.1.7 now provides formal definitions:

**Factual Consistency (FC):**
$$\text{FC} = \frac{|\{c \in C_{\text{gen}} : \exists t \in \mathcal{G}, c \models t\}|}{|C_{\text{gen}}|}$$

**Hallucination Rate (HR):**
$$\text{HR} = \frac{|\{c \in C_{\text{gen}} : \nexists t \in \mathcal{G} \cup D_{\text{ret}}, c \not\perp t\}|}{|C_{\text{gen}}|}$$

Claims are extracted using DeBERTa-v3-large fine-tuned on MNLI. Human evaluation on 500 samples shows 94.2% agreement (Cohen's κ = 0.87).

**Changes:** Section 4.1.7 (page 5), Appendix D (annotation protocol)

---

> "Potential data leakage not addressed"

**Response:** Section 4.1.5 addresses data leakage through:
1. **Temporal Filtering:** Wikipedia dumps from January 2024 (post GPT-3.5 cutoff)
2. **Ablation Analysis:** Table 5 shows strong performance on "post-cutoff" entities
3. **Counterfactual Evaluation:** TruthfulQA specifically tests memorized falsehoods

**Changes:** Section 4.1.5 (page 5), Table 5 (page 6)

---

> "Baselines under-specified"

**Response:** Section 4.1.6 provides configurations for all baselines, with full details in Appendix C:

- **RAG:** Contriever-MSMARCO, Wikipedia Dec 2023, 100-token chunks, FAISS HNSW
- **KG-RAG:** Same KG as ours, BLINK entity linking, top-10 triples
- **MemoryBank:** 10K slots, cosine similarity, consolidation every 1K queries
- **GraphRAG:** Leiden communities, 3 hierarchy levels, GPT-3.5 summaries

**Changes:** Section 4.1.6 (page 5), Appendix C (page 10)

---

> "Missing comparisons with GraphRAG-style methods"

**Response:** We added GraphRAG [Edge et al., 2024] as a baseline in all experiments. Table 2 shows our method outperforms GraphRAG across all benchmarks (e.g., NQ EM: 49.4% vs 45.1%, HotpotQA F1: 55.2% vs 49.2%).

**Changes:** Table 2 (page 6), Section 2.2 (related work)

---

### Weakness 3: Missing Related Work

> "No discussion of GraphRAG or graph-augmented retrieval methods"

**Response:** Added Section 2.2 "Graph-Augmented Retrieval Methods" discussing:
- GraphRAG [Edge et al., 2024]: entity-centric graph construction with community detection
- G-Retriever [He et al., 2024]: GNN+LLM for graph QA
- KG-RAG [Baek et al., 2023]: KG triple prompting

We differentiate our approach by: (1) combining three memory types vs. graph-only, (2) formal OWL 2 RL reasoning, (3) learned gating vs. fixed fusion.

**Changes:** Section 2.2 (page 2)

---

> "Missing retrofitting/graph-injection techniques comparison"

**Response:** Added Section 2.5 discussing retrofitting [Faruqui et al., 2015] and counter-fitting [Mrkšić et al., 2016]. We explain that our cross-attention mechanism (Section 3.5.2) provides a learned alternative to post-hoc refinement.

**Changes:** Section 2.5 (page 2)

---

> "Missing PMI-GSVD, ROOT-CA variants discussion"

**Response:** Added Section 2.4 discussing PMI-GSVD [Levy & Goldberg, 2014] and ROOT-CA [Yokoi et al., 2020]. We cite Levy et al. [2015] for best practices and note our use of standard PPMI+SVD with scalability strategies.

**Changes:** Section 2.4 (page 2)

---

> "Missing recent KG+LLM surveys positioning"

**Response:** Added citations to recent surveys: Pan et al. [2024] "Unifying LLMs and KGs: A Roadmap" and Sun et al. [2024] "Think-on-Graph". Our work is positioned in the "KG-augmented LLM" category.

**Changes:** Section 2.3 (page 2)

---

### Weakness 4: Reproducibility Issues

> "Claims 'open-source implementation' but no link provided"

**Response:** Added explicit repository URL: **https://github.com/second-brain-llm/hybrid-memory**

This appears in: Abstract (footnote), Introduction (contribution bullet), and Conclusion.

**Changes:** Abstract, Introduction, Conclusion

---

## Answers to Reviewer Questions

### Q1: Which LLM backbone(s) in all experiments? What prompts/context lengths?

**Answer:** GPT-3.5-turbo (version gpt-3.5-turbo-0613) for all main experiments. Context length: 4,096 tokens total, 3,500 for retrieval. Temperature: 0.0. Prompts provided in Appendix A. Appendix B shows comparison with GPT-4-turbo and Llama-2-70B-chat.

---

### Q2: How is gating network trained (objective, supervision, negatives)?

**Answer:** Trained with contrastive loss (InfoNCE) using:
- **Positives:** Correct query-answer pairs from NQ training set
- **Negatives:** In-batch + BM25 hard negatives
- **Temperature:** τ = 0.07
- **Optimizer:** AdamW, lr = 5×10⁻⁵
- **Batch/Epochs:** 128 / 10
See Section 3.5.1.

---

### Q3: How are "factual consistency" and "hallucination rate" measured?

**Answer:** 
- **Factual Consistency:** Proportion of generated claims verified against WikiData via NLI (DeBERTa-v3-large)
- **Hallucination Rate:** Proportion of claims contradicting or unsupported by any source
- **Human Validation:** 94.2% agreement (κ = 0.87) on 500-sample annotation
See Section 4.1.7 and Appendix D.

---

### Q4: How did you control for data leakage?

**Answer:** Three strategies:
1. Wikipedia dumps from Jan 2024 (post GPT-3.5 cutoff Sep 2021)
2. Ablation on post-cutoff entities (Table 5) shows +28.7 EM improvement
3. TruthfulQA specifically tests memorized falsehoods
See Section 4.1.5.

---

### Q5: What are exact baseline configurations?

**Answer:** Full configurations in Section 4.1.6 and Appendix C. Key details:
- RAG: Contriever, Wikipedia Dec 2023, FAISS HNSW
- KG-RAG: Same KG, BLINK linking, top-10 triples
- MemoryBank: 10K slots, cosine retrieval
- GraphRAG: Leiden communities, 3 levels

---

### Q6: How do you represent "concepts" in sequence encoder and align with KG entities?

**Answer:** Section 3.2.1 describes:
1. NER for mention detection
2. Dense retrieval over entity descriptions for linking
3. Representation fusion: concatenate sequence span embedding with KG embedding

---

### Q7: What hardware for efficiency measurements?

**Answer:** 4× NVIDIA A100 80GB, AMD EPYC 7742 (64 cores), 512GB RAM, 2TB NVMe. Inference uses single A100. See Section 4.1.6.

---

### Q8: How to scale co-occurrence beyond 100k vocabulary?

**Answer:** Section 3.1.1 details three strategies:
1. Subword tokenization (32K-50K vocabulary)
2. Sparse CSR format (<5% density)
3. Randomized block-sparse SVD [Halko et al., 2011]

---

### Q9: Which OWL profile/reasoner used? Runtime on largest graphs?

**Answer:** OWL 2 RL profile with RDFox reasoner. Runtime: 47 seconds for full materialization on 1.2M entities / 4.8M triples (single CPU core). See Section 3.4.

---

### Q10: Can you compare with GraphRAG-style methods?

**Answer:** Yes, added GraphRAG as baseline. Results in Table 2:
- NQ EM: Ours 49.4% vs GraphRAG 45.1% (+4.3%)
- HotpotQA F1: Ours 55.2% vs GraphRAG 49.2% (+6.0%)
- TruthfulQA: Ours 63.1% vs GraphRAG 57.1% (+6.0%)

---

## Changes Summary by Section

| Section | Changes |
|---------|---------|
| Abstract | Novelty statement, exact metrics, GitHub URL |
| Introduction | Novelty Statement paragraph, clearer contributions |
| Related Work | Added 2.2 (GraphRAG), 2.4 (PMI variants), 2.5 (retrofitting), comparison table |
| Methodology 3.1.1 | Scalability strategies |
| Methodology 3.2.1 | Entity alignment |
| Methodology 3.4 | OWL 2 RL, RDFox, runtime |
| Methodology 3.5.1 | Gating training details |
| Experiments 4.1.1 | LLM backbone |
| Experiments 4.1.2 | Retrieval parameters |
| Experiments 4.1.3 | KG scope |
| Experiments 4.1.5 | Data leakage mitigation |
| Experiments 4.1.6 | Baseline configs, hardware |
| Experiments 4.1.7 | Formal metric definitions |
| Results | Added GraphRAG comparison, Table 5 (leakage) |
| Discussion | Specific limitations |
| Appendix A | Prompt templates |
| Appendix B | LLM backbone comparison |
| Appendix C | Baseline details |
| Appendix D | Annotation protocol |

---

## Verification

All changes have been verified for:
- Citation accuracy (new references verified)
- Mathematical correctness
- Numerical consistency (Abstract matches body)
- Code-paper alignment

See VERIFICATION_RESULTS_v3.md for complete verification report.

---

*We believe these revisions substantially address all reviewer concerns and hope the paper is now suitable for publication.*
