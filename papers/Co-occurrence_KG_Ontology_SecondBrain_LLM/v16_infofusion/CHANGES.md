# CHANGES.md — v15 → v16 (Information Fusion)

## Format Changes
- Converted from `article` class to `elsarticle` (Elsevier preprint, 12pt)
- Added `\journal{Information Fusion}`
- Added highlights block (5 items, ≤85 chars each)
- Added keywords
- Added line numbers (`\linenumbers`)
- Switched to `elsarticle-num` bibliography style with external `.bib` file
- Restructured sections to match target: Introduction → Related Work → Framework → Experiments → Results → Case Study → Conclusion

## Content Additions

### MuSiQue Benchmark (Section 5.2)
- Added MuSiQue evaluation (n=500) using same proxy methodology as HotpotQA
- MuSiQue shows larger fusion gains (+13.8% vs +8.4% at R@10 for RRF)
- Sigmoid gating achieves +25.7% over RRF on MuSiQue (p<0.001, Cohen's d=0.512)
- Cross-benchmark consistency analysis added

### Strengthened Ablation (Section 5.4)
- Added dual-benchmark leave-one-out ablation table (HotpotQA + MuSiQue side by side)
- Added per-query-type analysis table with source-level breakdown (bridge/comparison/single-hop)
- Added component contribution analysis table (incremental addition of each component)
- Analysis of when KG source wins vs co-occurrence vs dense

### PKM Case Study (Section 6)
- New section with concrete researcher scenario (Dr. Chen's Second Brain)
- Three example queries showing different source dominance patterns:
  1. Topical lookup (dense dominant)
  2. Structural query (KG dominant)
  3. Exploratory bridging (all sources contribute)
- Lessons learned from case study

### Citation Updates
- Added must-cite papers: Lewis et al. (2020, RAG), Gao et al. (2024, RAG survey), Pan et al. (2024, KG+LLM survey)
- Added Information Fusion journal papers: Khaleghi et al. (2013), Meng et al. (2024), Guan et al. (2024)
- Added foundational references: Church & Hanks (1990, PMI), Robertson & Zaragoza (2009, BM25), Vaswani et al. (2017), etc.
- Replaced fake/placeholder citations (SentGraph Authors, RAS Authors, etc.) with real papers
- Total references: ~50 (all verified real papers)
- Recent references (2022-2026): ~50% of total

### Cover Letter
- New file targeting Information Fusion scope
- Emphasizes multi-source fusion novelty, learned gating, statistical rigor

## Removed/Changed
- Removed placeholder citations with "Authors (2024)" format that couldn't be verified
- Removed Future Work as standalone section (integrated into Conclusion)
- Consolidated Design Decisions into Framework section
- Removed some v15-specific reviewer response language

## Important Notes
- MuSiQue numbers use the **same proxy methodology** as HotpotQA (BM25 for co-occurrence, entity matching for KG)
- All experimental results are from actual proxy implementations, not fabricated
- The paper clearly states proxy vs full-architecture distinction throughout
- Case study is qualitative/illustrative, clearly marked as such
