# Revision Log: v5 → v5.1

**Date:** February 2026
**Focus:** Final polish addressing remaining reviewer concerns

## Changes Made

### 1. Faithfulness Metrics Added to All Results Tables
- Main results table (Table 2): Added Faith. column for all baselines
- Recent systems table (Table 7): Added Faith. column
- Shows +5.6 improvement over GraphRAG, +2.9 over FAIR-RAG

### 2. BLEU-4 Removed
- Removed from metrics list (not standard for QA, per reviewer)
- Focus now on EM/F1 and Faithfulness

### 3. FE2H Extractive Pipeline Comparison
- Added discussion in Related Work (Section 2.2)
- Contrasts RAG-centric vs extractive approaches
- Citation: tu2022fe2h

### 4. Ethics and Broader Impact Section
- New Section 7 added:
  - KG sources and licensing (Wikidata CC0, ConceptNet CC-BY-SA)
  - Potential biases (coverage, temporal, ontology design)
  - Maintenance considerations

### 5. Line Numbers Disabled
- Commented out `\usepackage{lineno}` and `\linenumbers`
- Clean submission-ready PDF

## Files Modified
- `main_v5.1.tex`: All changes above
- `references_v5.1.bib`: Added FE2H citation
