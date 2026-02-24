# Revision Log: v2 → v3

## Paper: "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"

**Revision Date:** February 9, 2026  
**Previous Version:** v2  
**New Version:** v3

---

## Summary Statistics

| Metric | v2 | v3 | Change |
|--------|----|----|--------|
| Page Count | ~16 | ~18 | +2 pages |
| References | 43 | 54 | +11 new |
| Tables | 6 | 8 | +2 new |
| Appendix Sections | 0 | 4 | +4 new |

---

## Detailed Change Log

### Abstract
| Line | Change Type | Description |
|------|-------------|-------------|
| 8-12 | ADDED | Explicit novelty statement: "Our primary contribution is not the individual components..." |
| 15 | MODIFIED | Added "GPT-3.5-turbo as the backbone LLM" specification |
| 18 | MODIFIED | Changed "15.2% reduction" to "61.5% relative reduction" for clarity |
| 19 | ADDED | GitHub repository URL |

### Introduction (Section 1)
| Location | Change Type | Description |
|----------|-------------|-------------|
| Page 1, para 4 | ADDED | "Novelty Statement" paragraph explicitly differentiating our contributions from prior work |
| Contribution bullet 4 | MODIFIED | Changed "Open-source implementation" to include actual URL |

### Related Work (Section 2)
| Section | Change Type | Description |
|---------|-------------|-------------|
| 2.2 | ADDED | New section "Graph-Augmented Retrieval Methods" (~250 words) |
| 2.2 | ADDED | Discussion of GraphRAG, G-Retriever, KG-RAG |
| 2.3 | ADDED | Citations to Pan et al. 2024, Sun et al. 2024 surveys |
| 2.4 | ADDED | New section on PMI-GSVD, ROOT-CA variants |
| 2.5 | ADDED | New section on retrofitting techniques |
| 2.6 | ADDED | Comparison table (Table 1) |

### Methodology (Section 3)
| Section | Change Type | Description |
|---------|-------------|-------------|
| 3.1.1 | ADDED | "Scalability Strategies" subsection with 3 approaches |
| 3.2.1 | ADDED | "Concept Representation and Entity Alignment" subsection |
| 3.4 | MODIFIED | Added OWL 2 RL specification, RDFox reasoner, runtime benchmarks |
| 3.5.1 | MODIFIED | Added complete gating network training details (objective, optimizer, hyperparameters) |

### Experiments (Section 4)
| Section | Change Type | Description |
|---------|-------------|-------------|
| 4.1.1 | ADDED | LLM backbone specification (GPT-3.5-turbo-0613, context length, temperature) |
| 4.1.2 | ADDED | Retrieval parameters (top-k, ANN index, embedding model) |
| 4.1.3 | ADDED | KG construction scope (entity counts, ontology size) |
| 4.1.5 | ADDED | Data leakage mitigation subsection |
| 4.1.6 | MODIFIED | Expanded baseline configurations with hyperparameters |
| 4.1.6 | ADDED | Hardware specifications |
| 4.1.7 | ADDED | Formal metric definitions with equations |
| Table 2 | MODIFIED | Added GraphRAG baseline row |
| Table 5 | ADDED | New table for post-cutoff entity performance |

### Discussion (Section 5)
| Location | Change Type | Description |
|----------|-------------|-------------|
| 5.1 item 2 | MODIFIED | Expanded knowledge freshness limitation |
| 5.1 item 3 | ADDED | New limitation on gating opacity |
| 5.1 item 4 | ADDED | New limitation on domain transfer |

### Appendices
| Section | Change Type | Description |
|---------|-------------|-------------|
| Appendix A | ADDED | Complete prompt templates for QA and fact verification |
| Appendix B | ADDED | LLM backbone comparison table (GPT-3.5, GPT-4, Llama-2) |
| Appendix C | ADDED | Detailed baseline configurations |
| Appendix D | ADDED | Human annotation protocol |

### References
| Citation | Type | Description |
|----------|------|-------------|
| edge2024graphrag | ADDED | GraphRAG paper |
| he2024gretriever | ADDED | G-Retriever paper |
| pan2024unifying | ADDED | KG+LLM survey |
| sun2024thinkongraph | ADDED | Think-on-Graph paper |
| levy2014neural | ADDED | PMI-GSVD paper |
| yokoi2020word | ADDED | ROOT-CA paper |
| levy2015improving | ADDED | Word embedding best practices |
| faruqui2015retrofitting | ADDED | Retrofitting paper |
| mrksic2016counterfitting | ADDED | Counter-fitting paper |
| motik2012owl | ADDED | OWL 2 profiles spec |
| nenov2015rdfox | ADDED | RDFox paper |
| halko2011finding | ADDED | Randomized SVD paper |
| shaw2018selfattention | ADDED | Relative positional encoding |

---

## Files Modified

| File | Status | Description |
|------|--------|-------------|
| main_v3.tex | MODIFIED | All revisions above |
| references.bib | MODIFIED | Added 11 new references |
| code/README.md | MODIFIED | Added GitHub URL |

---

## Reviewer Concern Mapping

| Reviewer Concern | Addressed In |
|------------------|--------------|
| Standard components | Abstract, Intro novelty statement |
| Scalability O(|V|²) | Section 3.1.1 |
| Ontology details | Section 3.4 |
| LLM backbone | Section 4.1.1 |
| Prompt templates | Appendix A |
| Context length | Section 4.1.1 |
| Retrieval parameters | Section 4.1.2 |
| KG scope | Section 4.1.3 |
| Gating training | Section 3.5.1 |
| Metric definitions | Section 4.1.7 |
| Data leakage | Section 4.1.5 |
| Baseline configs | Section 4.1.6, Appendix C |
| GraphRAG comparison | Table 2, Section 2.2 |
| PMI variants | Section 2.4 |
| Retrofitting | Section 2.5 |
| KG+LLM surveys | Section 2.3 |
| Code repository | Abstract, Intro, Conclusion |

---

## Verification Status

All changes verified:
- [x] New citations exist and are accurate
- [x] New equations are mathematically correct
- [x] Numbers in abstract match body
- [x] Table references are correct
- [x] No broken cross-references

See VERIFICATION_RESULTS_v3.md for complete verification report.
