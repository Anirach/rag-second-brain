# Academic Paper Team — Top-Tier Journal Publication Pipeline

## 🎯 Mission

Produce publication-ready academic papers meeting the standards of:
- **PubMed** (biomedical/life sciences)
- **IEEE** (engineering/computer science)
- **ACM** (computing)
- **Nature/Science** tier journals

Every paper must be rigorously researched, fact-checked, properly cited, and formatted to journal specifications.

## 🚀 Full Auto Mode (with Verification)

One spawn → Complete publication-ready paper with full verification.

```
[Research Topic + Target Journal] ──► paper-architect ──► [Publication-Ready Paper]
                                            │
                                            ├── 1. Define research questions & scope
                                            ├── 2. Comprehensive literature review
                                            ├── 3. Methodology design
                                            ├── 4. Analysis & synthesis
                                            ├── 5. Draft all sections
                                            ├── 6. 🗣️ STRUCTURED ARGUMENTATION CONSENSUS
                                            │       ├── 3 reviewers assess independently
                                            │       ├── Position statements with evidence
                                            │       ├── Rebuttal & convergence rounds
                                            │       └── Argumentation log produced
                                            ├── 7. Fact-check & verify citations
                                            ├── 8. Format to journal specs
                                            ├── 9. ⭐ VERIFICATION PHASE ⭐
                                            │       ├── Verify ALL citations exist
                                            │       ├── Test ALL code (if any)
                                            │       ├── Reproduce ALL results
                                            │       └── Check math formulas
                                            ├── 10. 🔍 NOVELTY VERIFICATION GATE
                                            │       ├── Extract contributions
                                            │       ├── Multi-database prior art search
                                            │       └── Similarity assessment per claim
                                            ├── 11. 🔴 ADVERSARIAL RED-TEAM REVIEW
                                            │       ├── 6 attack vectors
                                            │       ├── Fatal flaw detection
                                            │       └── Must SURVIVE to proceed
                                            ├── 12. 📐 CALIBRATION CHECK
                                            │       └── Record predictions in CALIBRATION_DB.md
                                            ├── 13. Fix any issues found
                                            ├── 14. Re-verify until ALL PASS
                                            └── 15. Upload to Google Drive (subfolder)
```

**⚠️ CRITICAL: Do NOT deliver until verification passes!**

### 📐 Paper Structure Template (MANDATORY)

**Every paper MUST follow the structure defined in:**
```
/home/clawdbot/clawd/templates/ACADEMIC_PAPER_PROMPT.md
```

This template defines exact section structure, math notation rules, quality gates, anti-patterns to avoid, and style rules. Read it before starting any paper. All agents in the academic team must comply with its rules — especially:
- Number verification protocol (no unverified quantitative claims)
- Citation completeness (real, verifiable, complete bibliographic data)
- Statistical evidence requirements (mean + std dev + run count)
- Abstract–body consistency check
- Anti-pattern avoidance list (10 items from production experience)

**To produce a paper:**
```
Arthur, write an academic paper on [TOPIC].
Target journal: [IEEE/ACM/PubMed/Nature format]
Research focus: [Specific angle/hypothesis]
```

---

## 🔄 Human Collaboration Mode (NEW)

Iterative back-and-forth workflow for co-authoring papers with human oversight.

```
┌─────────────────────────────────────────────────────────────┐
│              HUMAN COLLABORATION WORKFLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User provides: Topic + Target Journal + Hypothesis       │
│         │                                                    │
│         ▼                                                    │
│  2. paper-architect creates RESEARCH PLAN                   │
│     ├── Research questions                                  │
│     ├── Methodology outline                                 │
│     └── Target contributions                                │
│         │                                                    │
│         ▼                                                    │
│  3. 👤 HUMAN REVIEW — Approve research direction            │
│         │                                                    │
│         ├── Adjust scope → Revise plan → Step 3             │
│         │                                                    │
│         ▼ Approved                                           │
│  4. literature-lead conducts LIT REVIEW                     │
│         │                                                    │
│         ▼                                                    │
│  5. 👤 HUMAN REVIEW — Check coverage, key papers            │
│         │                                                    │
│         ├── Missing papers → Add citations → Step 5         │
│         │                                                    │
│         ▼ Approved                                           │
│  6. methodology-expert designs EXPERIMENTS                  │
│         │                                                    │
│         ▼                                                    │
│  7. 👤 HUMAN REVIEW — Approve methodology                   │
│         │                                                    │
│         ├── Changes needed → Revise → Step 7                │
│         │                                                    │
│         ▼ Approved                                           │
│  8. [Run experiments — may involve Coding Team]             │
│         │                                                    │
│         ▼                                                    │
│  9. data-analyst processes RESULTS                          │
│         │                                                    │
│         ▼                                                    │
│  10. 👤 HUMAN REVIEW — Verify results, interpretations      │
│         │                                                    │
│         ├── Re-run needed → Back to Step 8                  │
│         │                                                    │
│         ▼ Approved                                           │
│  11. technical-writer drafts SECTIONS                       │
│         │                                                    │
│         ▼                                                    │
│  12. 👤 HUMAN REVIEW — Per-section feedback                 │
│         │                                                    │
│         ├── Revisions → Rewrite → Step 12                   │
│         │                                                    │
│         ▼ Approved                                           │
│  13. peer-reviewer INTERNAL REVIEW                          │
│         │                                                    │
│         ▼                                                    │
│  14. 👤 HUMAN REVIEW — Address internal feedback            │
│         │                                                    │
│         ▼ Approved                                           │
│  15. ethics-reviewer checks COMPLIANCE                      │
│         │                                                    │
│         ▼                                                    │
│  16. format-editor FORMATS for journal                      │
│         │                                                    │
│         ▼                                                    │
│  17. 👤 FINAL REVIEW                                        │
│         │                                                    │
│         ▼ Approved                                           │
│  18. ✅ COMPLETE — Ready for submission                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**To start collaborative mode:**
```
Arthur, let's write a paper together on [TOPIC].
Target: [Journal]
I want to review each stage before proceeding.
```

**Feedback commands:**
- `"Approved, continue"` → Move to next stage
- `"Add citation for X"` → Include specific paper
- `"Strengthen the Y section"` → Expand/improve
- `"This claim needs evidence"` → Add support
- `"Revise methodology"` → Change approach
- `"Run additional experiment"` → More data

**Stage deliverables for review:**
| Stage | Deliverable | Review Focus |
|-------|-------------|--------------|
| Research Plan | `RESEARCH_PLAN.md` | Scope, feasibility |
| Lit Review | `LITERATURE_REVIEW.md` | Coverage, gaps |
| Methodology | `METHODOLOGY.md` | Rigor, reproducibility |
| Results | Tables, figures, stats | Accuracy, significance |
| Draft Sections | Per-section `.md` | Clarity, claims |
| Full Paper | `paper.pdf` | Overall quality |

## 👥 Team Roster (Enhanced with Skills)

| Agent | Role | Model | Expertise | **🆕 Key Skills** |
|-------|------|-------|-----------|-------------------|
| **paper-architect** | Principal Investigator | Opus | Research design, coordination, quality control | **agentarxiv**, academic-deep-research, agent-content-pipeline |
| **literature-lead** | Literature Review | Opus | Systematic review, gap analysis, citation mining | **literature-review**, **literature-search**, **scholargraph**, academic-deep-research |
| **methodology-expert** | Research Methods | Opus | Experimental design, statistical rigor, reproducibility | **statistics**, academic-deep-research, data-analysis |
| **technical-writer** | Section Drafting | Sonnet | Academic prose, clear exposition, technical accuracy | **latex**, academic-writing-refiner, writing |
| **peer-reviewer** | Internal Review | Opus | Critical analysis, fact-checking, weakness identification | **peer-review**, **empirical-paper-analysis-skill**, academic-deep-research |
| **format-editor** | Journal Compliance | Sonnet | Citation formatting, style guides, submission prep | **latex**, **typetex**, markdown-converter, word-docx |
| **data-analyst** | Statistical Analysis | Opus | Stats, visualization, reproducibility | **data-analysis**, **statistics**, diagram, mermaid-architect |
| **ethics-reviewer** | Ethics & Compliance | Opus | IRB, data privacy, ethical considerations | literature-search, academic-deep-research |
| **journal-scout** | Journal Intelligence | Opus | Journal matching, acceptance prediction, writing culture analysis | **literature-search**, **literature-review**, **academic-writing-refiner**, web_search |
| **red-team-reviewer** | Adversarial Review | Opus | Attack-oriented critique, finding fatal flaws, reproducibility challenges | **peer-review**, **empirical-paper-analysis-skill**, academic-deep-research |

---

## 🔍 Journal Intelligence System (journal-scout)

**Trigger:** "find journal for", "where should I submit", "journal recommendation", "journal matching"

**Input:** Title + Abstract (minimum) — full paper optional

### Phase 1: Journal Discovery & Ranking (~5 min)
- Web search 60-80 sources (Scopus, WoS, journal websites, recent publications)
- 8-dimension scoring:
  1. **Topic fit** (0-25) — keyword/methodology match with recent publications
  2. **Methodology match** (0-15) — does journal publish this type of research?
  3. **Impact Factor** (0-10) — IF relative to field average
  4. **Acceptance rate** (0-10) — historical acceptance data
  5. **Review speed** (0-10) — average time to first decision
  6. **Open Access options** (0-10) — OA availability and APC costs
  7. **Prestige/Indexing** (0-10) — Scopus/WoS/PubMed indexing
  8. **Precedent papers** (0-10) — similar published work in last 2 years
- Output: **Top 5 journals** with scores, probability estimates, and reasoning

### Phase 2: Decision Refinement (~10 min)
- Head-to-head comparison of top candidates
- Narrow to **2 best options**
- Submission roadmap: **Plan A → B → C** (with cascade strategy if rejected)
- Timeline estimation per journal

### Phase 3: Writing Culture Benchmarking (~15 min)
- Analyze 5-10 recent papers from target journal
- Extract: section structure, theory emphasis %, hedging style, active/passive ratio
- Identify **must-cite papers** and citation patterns
- Generate: section templates, writing guidelines, pre-submission checklist

**Spawn command:**
```
Arthur, find the best journal for my paper.
Title: [paper title]
Abstract: [abstract text]
Field: [e.g., AI, education, biomedical]
```

---

## 🔬 Skills Integration for Academic Excellence

The Academic Team leverages advanced academic skills to produce top-tier research meeting publication standards of Nature, Science, IEEE, ACM, and PubMed journals.

### Skills Integration Matrix

| Agent | Skill | When to Use | How It Helps |
|-------|-------|-------------|--------------|
| **paper-architect** | **agentarxiv** | arXiv research & publishing | Search arXiv papers, replicate studies, prepare submissions |
| **paper-architect** | academic-deep-research | Initial research design | Academic-focused deep research with proper APA citations |
| **paper-architect** | agent-content-pipeline | Multi-stage paper workflow | Orchestrate research → analysis → writing → review → publication |
| **literature-lead** | **literature-review** | Systematic literature reviews | Multi-engine search (S2, OA, CR, PM) with deduplication |
| **literature-lead** | **literature-search** | Cross-database research | Search Google Scholar, PubMed, arXiv, IEEE, ACM, Semantic Scholar |
| **literature-lead** | **scholargraph** | Citation network analysis | Explore citation graphs and paper relationships |
| **literature-lead** | academic-deep-research | Academic source validation | Verify academic claims and source authority |
| **methodology-expert** | **statistics** | Statistical method reference | Access statistical methods library and best practices |
| **methodology-expert** | data-analysis | Statistical analysis patterns | Implement robust statistical analysis workflows |
| **technical-writer** | **latex** | Academic document formatting | LaTeX compilation with proper academic formatting |
| **technical-writer** | academic-writing-refiner | Scholarly prose | Enhance academic writing style and clarity |
| **peer-reviewer** | **peer-review** | Multi-model peer review | Simulate peer review process with multiple perspectives |
| **peer-reviewer** | **empirical-paper-analysis-skill** | Paper structure analysis | Systematically analyze empirical papers for strengths/weaknesses |
| **format-editor** | **latex** | Journal-specific formatting | LaTeX templates for IEEE, ACM, Nature, Science formats |
| **format-editor** | **typetex** | Advanced typesetting | LaTeX + Typst → PDF compilation API |
| **format-editor** | markdown-converter | Format flexibility | Convert between markdown, LaTeX, Word as needed |
| **data-analyst** | **data-analysis** | Statistical computation | Advanced statistical analysis with reproducible results |
| **data-analyst** | **statistics** | Statistical validation | Validate statistical approaches and interpretations |
| **data-analyst** | diagram | Data visualization | Create publication-quality figures and charts |

### Cross-Team Skills Available

| Skill | From Team | Academic Use Case |
|-------|-----------|-------------------|
| deep-research-pro | Writing | Comprehensive background research |
| content-quality-auditor | Writing | Content quality assessment for papers |
| mermaid-architect | General | Methodology flowcharts and system diagrams |
| clean-code-review | Coding | Code quality for computational research |
| edge-tts | Translation/Course | Research presentation audio |

### Enhanced Quality Standards with Skills

#### Research Integrity (Skill-Enhanced)
- ✅ **literature-review**: Systematic multi-database literature search
- ✅ **scholargraph**: Citation network validation 
- ✅ **academic-deep-research**: Source authority verification
- ✅ **agentarxiv**: Preprint validation and replication
- ✅ **peer-review**: Multi-model internal review process

#### Statistical Rigor (Skill-Enhanced)
- ✅ **statistics**: Statistical method validation
- ✅ **data-analysis**: Reproducible computational analysis
- ✅ **empirical-paper-analysis-skill**: Methodology review
- ✅ **diagram**: Publication-quality visualizations

#### Citation Requirements (Skill-Enhanced)
- ✅ **literature-search**: Multi-source citation discovery
- ✅ **scholargraph**: Citation completeness verification
- ✅ **literature-review**: Systematic citation management
- ✅ **latex**: Proper academic citation formatting

#### Format Standards (Skill-Enhanced)
- ✅ **latex**: Journal-specific LaTeX templates
- ✅ **typetex**: Advanced PDF compilation
- ✅ **markdown-converter**: Multi-format compatibility
- ✅ **word-docx**: Publisher-specific DOCX requirements

### Enhanced Academic Workflow

```
🔬 SKILL-POWERED ACADEMIC PAPER PIPELINE

1. paper-architect + agentarxiv + academic-deep-research
   ├── arXiv literature discovery
   ├── Academic-focused deep research
   └── Research gap identification

2. literature-lead + literature-review + literature-search + scholargraph
   ├── Multi-engine systematic review (S2, OA, CR, PM)
   ├── Automated deduplication
   ├── Citation network analysis
   └── Comprehensive literature mapping

3. methodology-expert + statistics + data-analysis
   ├── Statistical method validation
   ├── Reproducible analysis design
   └── Statistical rigor verification

4. technical-writer + latex + academic-writing-refiner
   ├── Journal-specific LaTeX formatting
   ├── Academic prose optimization
   └── Technical accuracy enhancement

5. peer-reviewer + methodology-expert + technical-writer (STRUCTURED ARGUMENTATION)
   ├── Independent assessment with evidence citations
   ├── Position statements with confidence levels
   ├── Rebuttal & convergence rounds
   ├── Argumentation log documenting resolution
   └── Consensus or escalation to human

6. 🆕 NOVELTY VERIFICATION GATE
   ├── Contribution extraction
   ├── Multi-database prior art search
   ├── Similarity assessment per contribution
   └── Differentiation strengthening if needed

7. 🆕 red-team-reviewer (ADVERSARIAL RED-TEAM)
   ├── 6 attack vectors (claim, methodology, baseline, reproducibility, novelty, statistics)
   ├── Red-Team Report with severity ratings
   └── Fatal flaws must be fixed before submission

8. data-analyst + data-analysis + statistics + diagram
   ├── Advanced statistical computation
   ├── Statistical validation
   └── Publication-quality visualizations

9. format-editor + latex + typetex + markdown-converter
   ├── Multi-journal format preparation
   ├── Advanced PDF compilation
   └── Format conversion flexibility

10. 🆕 CALIBRATION CHECK
    ├── Record internal predictions in CALIBRATION_DB.md
    ├── Compare against historical accuracy
    └── Flag if predictions diverge from calibrated expectations
```

### Research-to-Publication Commands

```bash
# Literature-enhanced paper
"Arthur, write a paper on [TOPIC] using literature-review for systematic literature search"

# Methodology-validated research
"Arthur, design a study using statistics skill for statistical method validation"

# Multi-format academic output
"Arthur, write a paper and prepare it for multiple journals using latex skill"

# Peer-review enhanced quality
"Arthur, write a paper and run peer-review skill for internal quality assessment"

# arXiv research integration
"Arthur, research [TOPIC] using agentarxiv for preprint analysis and replication"

# Citation network analysis
"Arthur, analyze the citation network for [TOPIC] using scholargraph"
```

---

## 📋 Quality Standards

### Research Integrity
- ✅ All claims backed by peer-reviewed sources
- ✅ Primary sources preferred over secondary
- ✅ Recency check (prioritize last 5 years, note seminal older works)
- ✅ No predatory journal citations
- ✅ Conflict of interest awareness
- ✅ Reproducibility considerations

### Citation Requirements
- ✅ Every factual claim cited
- ✅ DOI links for all papers when available
- ✅ Proper attribution (no missing citations)
- ✅ Citation format matches target journal
- ✅ Reference list verified against in-text citations
- ✅ No self-plagiarism or excessive self-citation

### Fact-Checking Protocol
```
Level 1: Source exists and is accessible
Level 2: Claim accurately represents source
Level 3: Source is authoritative (journal impact, author credentials)
Level 4: Claim is current (not superseded by newer research)
Level 5: Statistical claims verified
```

### Writing Standards
- ✅ Active voice preferred (per modern style guides)
- ✅ Precise technical terminology
- ✅ No hedge words without justification
- ✅ Clear logical flow between sections
- ✅ Figures/tables properly referenced
- ✅ Abstract accurately summarizes content

## 📄 Paper Structure (Standard)

```
1. Title (concise, specific, searchable keywords)
2. Abstract (structured: Background, Methods, Results, Conclusions)
3. Keywords (5-7, MeSH terms for biomedical)
4. Introduction
   - Context & significance
   - Gap in knowledge
   - Research questions/hypotheses
   - Contributions of this work
5. Related Work / Literature Review
   - Systematic coverage
   - Critical analysis (not just summary)
   - Clear positioning of current work
6. Methodology
   - Reproducible detail
   - Justification for choices
   - Limitations acknowledged
7. Results / Findings
   - Clear presentation
   - Statistical significance noted
   - Visualizations where helpful
8. Discussion
   - Interpretation of results
   - Comparison with prior work
   - Implications
   - Limitations
9. Conclusion
   - Summary of contributions
   - Future work
10. References (journal-appropriate format)
11. Appendices (supplementary material)
```

## 🔬 Journal-Specific Formats

### IEEE Format
- Two-column layout
- IEEE citation style [1], [2-4]
- Abstract ≤ 200 words
- Keywords: Index Terms
- Section numbering: I, II, III...

### ACM Format
- ACM Reference Format
- CCS Concepts included
- Author keywords + ACM keywords
- Structured abstract

### PubMed/Biomedical (IMRAD)
- Introduction, Methods, Results, and Discussion
- Structured abstract (Background, Methods, Results, Conclusions)
- MeSH keywords
- Vancouver citation style
- Ethics statement if applicable

### Nature/Science Style
- No section numbers
- Methods often at end or supplementary
- Brief, high-impact abstract
- Reference style per journal

## ⚠️ Red Flags to Catch

### Content Issues
- 🚩 Unreproducible claims
- 🚩 Overclaiming ("proves" vs "suggests")
- 🚩 Cherry-picked data/citations
- 🚩 Missing limitations section
- 🚩 Correlation presented as causation
- 🚩 P-hacking indicators

### Citation Issues
- 🚩 [citation needed] placeholders remaining
- 🚩 Broken DOI links
- 🚩 Retracted papers cited
- 🚩 Predatory journal sources
- 🚩 Misquoted statistics

### Format Issues
- 🚩 Inconsistent citation style
- 🚩 Missing figure/table references
- 🚩 Word count violations
- 🚩 Incorrect heading hierarchy

## ✅ SKILL-ENHANCED VERIFICATION PHASE

**Before any paper is delivered, ALL skill-powered verification checks must PASS.**

### Enhanced Verification Checklist with Skills

#### Literature & Citations (literature-review + literature-search + scholargraph)
```
□ literature-review: Multi-database literature search completed
□ literature-search: Cross-verification across Google Scholar, PubMed, arXiv, IEEE, ACM
□ scholargraph: Citation network analysis confirms completeness
□ DOIs verified accessible through literature databases
□ No retracted papers identified through database checks
□ Citation gaps identified and filled through systematic search
```

#### Statistical Rigor (statistics + data-analysis)
```
□ statistics: All statistical methods validated against best practices
□ data-analysis: Reproducible analysis pipeline implemented
□ Statistical assumptions verified and documented
□ Effect sizes and confidence intervals reported
□ Multiple comparisons corrections applied where appropriate
□ Bootstrap/resampling validation completed where applicable
```

#### Peer Review Quality (peer-review + empirical-paper-analysis-skill)
```
□ peer-review: Multi-model internal review completed
□ empirical-paper-analysis-skill: Structural analysis passed
□ All peer review recommendations addressed
□ Internal review consensus achieved
□ Critical weaknesses identified and resolved
□ Methodological soundness confirmed
```

#### Format & Presentation (latex + typetex + academic-writing-refiner)
```
□ latex: Journal-specific formatting verified
□ typetex: Advanced PDF compilation successful
□ academic-writing-refiner: Academic prose optimized
□ All equations properly formatted and numbered
□ Figures and tables meet journal standards
□ Reference formatting matches target journal exactly
```

#### Research Integrity (academic-deep-research + agentarxiv)
```
□ academic-deep-research: Source authority verification completed
□ agentarxiv: Preprint landscape analyzed for novelty
□ No research integrity concerns identified
□ Proper attribution of all previous work
□ Original contributions clearly distinguished
□ Replication potential documented
```

#### Code & Data (If Applicable - with Coding Team skills)
```
□ clean-code-review: Code quality standards met
□ All code files validated through clean-code patterns
□ Dependencies managed and documented
□ Reproducibility verified on clean environment
□ Data availability and ethics compliance confirmed
□ Version control and change documentation complete
```

### Advanced Verification Workflow

```
🔍 SKILL-POWERED VERIFICATION SEQUENCE

Phase 1: Literature Verification
├── literature-review: Systematic literature completeness
├── scholargraph: Citation network validation
└── literature-search: Cross-database verification

Phase 2: Content Quality Assessment
├── peer-review: Multi-model review simulation
├── empirical-paper-analysis-skill: Structural analysis
└── academic-deep-research: Academic integrity check

Phase 3: Statistical & Data Validation
├── statistics: Method validation
├── data-analysis: Computational verification
└── Coding Team (if needed): Code quality review

Phase 4: Format & Presentation
├── latex: Journal formatting compliance
├── typetex: PDF compilation quality
└── academic-writing-refiner: Prose optimization

Phase 5: 🆕 Novelty Verification Gate
├── Contribution extraction from Introduction
├── Multi-database prior art search (S2, arXiv, Scholar)
├── Similarity assessment per contribution
├── Differentiation strengthening if incremental
└── Novelty verdict: NOVEL / INCREMENTAL / NOT NOVEL

Phase 6: 🆕 Adversarial Red-Team Review
├── red-team-reviewer: 6 attack vectors
├── Claim destruction attempts
├── Methodology stress test
├── Baseline challenge (search for stronger baselines)
├── Statistical adversary analysis
└── Red-Team verdict: SURVIVED / NEEDS WORK / FATAL FLAW

Phase 7: Final Integration Check
├── agentarxiv: Positioning verification
├── Cross-reference all verification outputs
├── Argumentation log from consensus
└── Generate comprehensive verification report
```

### Skill-Enhanced Verification Report Template

```markdown
# Academic Paper Verification Report (Skill-Enhanced)

## Literature Review Verification (literature-review + scholargraph)
- **Database Coverage**: ✅ Searched S2, OA, CR, PM
- **Citation Completeness**: ✅ [X] papers, [Y] unique DOIs
- **Network Analysis**: ✅ Citation gaps identified and filled
- **Status**: VERIFIED

## Peer Review Assessment (peer-review + empirical-paper-analysis-skill)
- **Multi-Model Review**: ✅ [N] review perspectives
- **Structural Analysis**: ✅ All components validated
- **Critical Issues**: ✅ [X] issues identified and resolved
- **Status**: APPROVED

## Statistical Validation (statistics + data-analysis)
- **Method Validation**: ✅ All approaches verified
- **Reproducibility**: ✅ Analysis pipeline documented
- **Statistical Rigor**: ✅ Best practices followed
- **Status**: VALIDATED

## Format Compliance (latex + typetex)
- **Journal Format**: ✅ [Journal] standards met
- **PDF Quality**: ✅ Publication-ready compilation
- **Reference Format**: ✅ Citation style verified
- **Status**: COMPLIANT

## Research Integrity (academic-deep-research + agentarxiv)
- **Source Verification**: ✅ All sources validated
- **Novelty Check**: ✅ Original contributions confirmed
- **Ethics Compliance**: ✅ All requirements met
- **Status**: INTEGRITY_VERIFIED

## Final Verification: ✅ ALL SYSTEMS PASS
```

### Required Output Files

Every paper project MUST include:

| File | Purpose | Required |
|------|---------|----------|
| `paper.pdf` | Final paper | ✅ Yes |
| `paper.docx` | Editable version | ✅ Yes |
| `main.tex` | LaTeX source (if applicable) | Optional |
| `README.md` | Project overview | ✅ Yes |
| `PROCESS_DOCUMENT.md` | Full methodology & checklist | ✅ Yes |
| `VERIFICATION_RESULTS.md` | Verification report | ✅ Yes |
| `code/*.py` | Implementation (if applicable) | As needed |
| `references.bib` | Bibliography | ✅ Yes |

### Verification Workflow

```
1. Complete paper draft
         ↓
2. Run verification checks
         ↓
3. Generate VERIFICATION_RESULTS.md
         ↓
    ┌─────────────────┐
    │ All checks PASS? │
    └────────┬────────┘
             │
      ┌──────┴──────┐
      ↓ NO          ↓ YES
   Fix issues    Deliver to
   and re-verify  Google Drive
      │               │
      └───────────────┘
```

### Verification Results Template

```markdown
# VERIFICATION_RESULTS.md

## Overall Status: [✅ PASS / ⚠️ PARTIAL / ❌ FAIL]

### Citation Verification: [X/Y verified]
### Math Verification: [X/Y verified]  
### Code Verification: [PASS/FAIL/N/A]
### Results Verification: [PASS/FAIL]
### Consistency Check: [PASS/FAIL]

### Issues Found: [list]
### Actions Taken: [list]
### Final Status: [READY FOR DELIVERY / NEEDS WORK]
```

## 📊 Pre-Submission Checklist

```
□ Research questions clearly stated and answered
□ All claims have citations
□ All citations verified accessible
□ Statistics double-checked
□ Figures high-resolution, properly labeled
□ Tables formatted per journal style
□ Abstract within word limit
□ Keywords optimized for discoverability
□ Author contributions defined
□ Conflicts of interest declared
□ Data availability statement
□ Supplementary materials organized
□ Cover letter drafted
□ Suggested reviewers identified
□ ⭐ VERIFICATION_RESULTS.md shows ALL PASS
```

## 📁 Output Structure

**Google Drive:** `ArthurBotData/[Paper-Title-Folder]/`

⚠️ **ALWAYS create a dedicated subfolder** — never dump files in root!

```
[Paper-Title-Folder]/
├── paper.pdf                    ⭐ Final paper
├── paper.docx                   ⭐ Editable version
├── main.tex                     LaTeX source
├── README.md                    ⭐ Project overview
├── PROCESS_DOCUMENT.md          ⭐ Full methodology
├── VERIFICATION_RESULTS.md      ⭐ Verification report
├── references.bib               Bibliography
├── code/
│   ├── *.py                     Implementation
│   ├── requirements.txt         Dependencies
│   └── README.md                Code documentation
├── figures/
│   └── [figure files]
└── data/
    └── [datasets or links]
```

**Upload command:**
```bash
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py <file> --folder "<Paper-Title>"
```

## 🔄 Review Cycles

### Internal Review (peer-reviewer)
1. **Structural Review** — Is the argument coherent?
2. **Technical Review** — Are methods sound?
3. **Evidence Review** — Are claims supported?
4. **Clarity Review** — Is it understandable?
5. **Format Review** — Does it meet journal specs?

### Revision Protocol
- All reviewer comments addressed
- Changes tracked and documented
- Response letter drafted for each concern

---

## 📝 JOURNAL REVIEW & REVISION WORKFLOW

### Complete Publication Lifecycle

```
┌─────────────────────────────────────────────────────────────┐
│                    PUBLICATION LIFECYCLE                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Paper Complete ──► 2. Submit to Journal                  │
│         ✅                     📤                            │
│                                 │                            │
│                                 ▼                            │
│                    ┌───────────────────────┐                │
│                    │   Journal Review      │                │
│                    │   (Wait for decision) │                │
│                    └───────────┬───────────┘                │
│                                │                            │
│           ┌────────────────────┼────────────────────┐       │
│           ▼                    ▼                    ▼       │
│      ┌─────────┐        ┌───────────┐        ┌─────────┐   │
│      │ ACCEPT  │        │  REVISE   │        │ REJECT  │   │
│      │   🎉    │        │  📝       │        │   ❌    │   │
│      └────┬────┘        └─────┬─────┘        └────┬────┘   │
│           │                   │                   │        │
│           ▼                   ▼                   ▼        │
│      Published!         User shares          Consider      │
│                        reviewer comments    other journal  │
│                              │                             │
│                              ▼                             │
│                    ┌─────────────────────┐                │
│                    │  Academic Team      │                │
│                    │  Revision Agent     │                │
│                    │                     │                │
│                    │  • Parse reviews    │                │
│                    │  • Address each     │                │
│                    │    comment          │                │
│                    │  • Revise paper     │                │
│                    │  • Generate         │                │
│                    │    Response Letter  │                │
│                    │  • Re-verify        │                │
│                    └──────────┬──────────┘                │
│                               │                           │
│                               ▼                           │
│                    ┌─────────────────────┐                │
│                    │  Deliverables:      │                │
│                    │  • Revised paper    │                │
│                    │  • Response letter  │                │
│                    │  • Diff/track       │                │
│                    │    changes          │                │
│                    └──────────┬──────────┘                │
│                               │                           │
│                               ▼                           │
│                         Re-submit                         │
│                         (Loop back to Review)             │
│                                                           │
└───────────────────────────────────────────────────────────┘
```

### How to Trigger Revision

When you receive journal reviews, send me:

```
Arthur, revise the paper "[Paper Title]" based on these reviewer comments:

[Paste reviewer comments here or attach the review document]

Paper folder: [Google Drive folder name]
```

### Revision Process

1. **Parse Reviews**
   - Extract each reviewer comment
   - Categorize: Major/Minor/Editorial
   - Identify required changes

2. **Address Each Comment**
   - For each point, decide: Accept/Partially Accept/Rebut
   - Make corresponding changes in paper
   - Document rationale

3. **Revise Paper**
   - Update text, figures, tables as needed
   - Add new experiments if required
   - Ensure consistency after changes
   - **🔧 If reviewers request code/implementation:**
     - Spawn Coding Team agents for implementation work
     - `code-builder` for new features/experiments
     - `test-writer` for test coverage requirements
     - `code-reviewer` for audit requests
     - Coordinate outputs with paper updates

4. **Generate Response Letter**
   - Format: "Response to Reviewers"
   - Quote each comment
   - Explain changes made (with page/line references)
   - Justify any rebuttals professionally

5. **Re-verify**
   - Run full verification again
   - Ensure no new errors introduced
   - Check all citations still valid

### Response Letter Format

```markdown
# Response to Reviewers

**Paper:** [Title]
**Manuscript ID:** [If applicable]
**Date:** [Date]

---

## Response to Reviewer 1

### Comment 1.1
> [Quoted reviewer comment]

**Response:** [Our response]

**Changes Made:** [Description of changes, with page/line references]

### Comment 1.2
> [Quoted reviewer comment]

**Response:** [Our response]

**Changes Made:** [Description] OR **Rebuttal:** [Professional explanation why no change]

---

## Response to Reviewer 2

[Same format...]

---

## Summary of Changes

| Section | Change | Reason |
|---------|--------|--------|
| Abstract | Updated results | Reviewer 1, Comment 3 |
| Section 3.2 | Added clarification | Reviewer 2, Comment 1 |
| Figure 4 | Improved resolution | Reviewer 1, Comment 5 |

---

*We thank the reviewers for their valuable feedback.*
```

### Cross-Team Coordination

**When to involve the Coding Team:**

| Reviewer Request | Coding Team Agent | Action |
|------------------|-------------------|--------|
| "Add baseline comparison" | `code-builder` | Implement new baseline |
| "Run additional experiments" | `code-builder` | New experiment scripts |
| "Provide ablation study" | `code-builder` + `test-writer` | Ablation code + tests |
| "Show reproducibility" | `test-writer` | Unit tests, CI/CD |
| "Code review concerns" | `code-reviewer` | Audit & improve code |
| "Performance optimization" | `code-builder` | Optimize implementation |

**Coordination flow:**
```
paper-architect ──► Identifies code-related reviewer requests
        │
        ▼
   Spawns appropriate Coding Team agent(s)
        │
        ▼
   code-builder/test-writer delivers implementation
        │
        ▼
   paper-architect integrates results into paper revision
```

### Revision Deliverables

| File | Description |
|------|-------------|
| `paper_v[N].pdf` | Revised paper (clean) |
| `paper_v[N]_tracked.pdf` | With track changes |
| `paper_v[N].docx` | Editable version |
| `RESPONSE_TO_REVIEWERS.md` | Response letter |
| `REVISION_LOG.md` | All changes documented |
| `VERIFICATION_RESULTS_v[N].md` | Re-verification |
| `POSITIVE_FEEDBACK.md` | ⭐ Accumulated praise (updated) |
| `QUALITY_LOCKS.md` | ⭐ Locked improvements (updated) |
| `code/*.py` | New/updated implementations (if requested) |

### Version Control

```
📁 Paper_Folder/
├── v1/                      # Initial submission
│   ├── paper_v1.pdf
│   └── VERIFICATION_RESULTS_v1.md
├── v2/                      # After Round 1 review
│   ├── paper_v2.pdf
│   ├── paper_v2_tracked.pdf
│   ├── RESPONSE_TO_REVIEWERS_R1.md
│   └── VERIFICATION_RESULTS_v2.md
├── v3/                      # After Round 2 review
│   └── ...
├── POSITIVE_FEEDBACK.md     # ⭐ Accumulated praise registry
├── QUALITY_LOCKS.md         # ⭐ Locked improvements
└── FINAL/                   # Accepted version
    └── paper_FINAL.pdf
```

---

## 🔒 QUALITY ACCUMULATION SYSTEM

### Purpose

Ensure paper quality **only improves** across revisions. Positive reviewer feedback and improvements are "locked in" and must be preserved in all future versions.

### Positive Feedback Registry

Every review round, extract and accumulate:

**`POSITIVE_FEEDBACK.md`** template:
```markdown
# Positive Feedback Registry

## Purpose
Track all positive reviewer comments to ensure we PRESERVE these strengths.

---

## Round 1 (v2)

### Reviewer 1 Praise
- "The two-stage training decomposition is clean and well-motivated"
- "Ablations are thorough"

### Reviewer 2 Praise  
- "Good engagement with ontology literature"
- "Clear writing in methodology section"

### Overall Strengths Identified
- [x] Two-stage training (LOCKED - do not change core design)
- [x] Ablation coverage (LOCKED - maintain or expand)
- [x] Ontology integration (LOCKED - preserve this contribution)

---

## Round 2 (v3)

### New Praise
- "Compute breakdown is now clear and reproducible"
- "FRAMES benchmark addition strengthens empirical claims"

### Cumulative Locked Strengths
- [x] Two-stage training
- [x] Ablation coverage  
- [x] Ontology integration
- [x] Compute transparency (NEW LOCK)
- [x] Multi-benchmark evaluation (NEW LOCK)

---
```

### Quality Locks

**`QUALITY_LOCKS.md`** — Immutable improvements:

⚠️ **CRITICAL: Verify Before Locking!**

A lock is only as good as its verification. Before locking ANY numerical claim:

```
□ Cost calculations: Verify against actual API pricing (input vs output rates)
□ Compute claims: Have realistic basis (cite benchmarks, show wall-clock logs)
□ Performance numbers: Cross-check against reported baselines in original papers
□ Dataset statistics: Verify against official documentation
```

**Lesson learned (2026-02-09)**: v8 locked "$135 for 2.7M GPT-3.5 calls" — actual cost ~$135,000. This single error dropped the score from 5.6 to 4.4. **Precise-looking but wrong numbers hurt MORE than vague statements.**

```markdown
# Quality Locks — DO NOT REGRESS

## How This Works
Once an improvement is verified and praised, it becomes a LOCK.
Future revisions MUST preserve locked items. Any change that would 
remove or weaken a lock requires explicit justification.

---

## Current Locks

| Version | Lock ID | Description | Source |
|---------|---------|-------------|--------|
| v2 | LOCK-001 | Two-stage training pipeline | R1 praise |
| v2 | LOCK-002 | Comprehensive ablations (Tables 5-12) | R1 praise |
| v3 | LOCK-003 | Faithfulness metric defined (DeBERTa NLI) | R2 request resolved |
| v3 | LOCK-004 | Compute reconciled (44 GPU-hours) | R2 request resolved |
| v4 | LOCK-005 | Adaptive RAG comparison table | R3 request resolved |

---

## Pre-Revision Checklist

Before creating version N+1, verify:

□ All locks from version N are preserved
□ No locked content has been removed
□ No locked content has been weakened
□ New improvements are candidates for new locks

## Lock Violation Protocol

If a revision would violate a lock:
1. STOP — Do not proceed
2. Document the conflict
3. Seek explicit approval to unlock
4. If approved, document rationale in REVISION_LOG
```

### Regression Prevention Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              REVISION WITH QUALITY ACCUMULATION              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Receive reviewer feedback                                │
│         │                                                    │
│         ▼                                                    │
│  2. EXTRACT POSITIVE FEEDBACK ◄──── Add to POSITIVE_FEEDBACK.md
│         │                                                    │
│         ▼                                                    │
│  3. CREATE NEW LOCKS ◄──────────── Add to QUALITY_LOCKS.md  │
│     (from praise + resolved issues)                          │
│         │                                                    │
│         ▼                                                    │
│  4. Address criticism (fix issues)                           │
│         │                                                    │
│         ▼                                                    │
│  5. LOCK CHECK ◄───────────────── Verify ALL locks preserved│
│         │                                                    │
│         ├─── FAIL ───► Fix regression before proceeding     │
│         │                                                    │
│         ▼ PASS                                               │
│  6. Generate new version                                     │
│         │                                                    │
│         ▼                                                    │
│  7. Update score estimate                                    │
│     (must be ≥ previous version)                            │
│         │                                                    │
│         ├─── Score dropped? ───► Review what was lost       │
│         │                                                    │
│         ▼                                                    │
│  8. Deliver with confidence                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Mandatory Revision Checklist

**Before delivering any new version:**

```markdown
## Pre-Delivery Lock Verification

Paper: [Title]
From Version: vN → vN+1

### Lock Preservation Check

| Lock ID | Description | Status | Notes |
|---------|-------------|--------|-------|
| LOCK-001 | Two-stage training | ✅ PRESERVED | |
| LOCK-002 | Ablation tables | ✅ PRESERVED | |
| LOCK-003 | Faithfulness metric | ✅ PRESERVED | |
| ... | ... | ... | |

### New Locks Added This Version

| Lock ID | Description | Source |
|---------|-------------|--------|
| LOCK-00X | [New improvement] | [Reviewer request/praise] |

### Quality Score Trend

| Version | Score | Trend |
|---------|-------|-------|
| v1 | 5.0/10 | — |
| v2 | 5.2/10 | ↑ +0.2 |
| v3 | 6.1/10 | ↑ +0.9 |
| v4 | 7.3/10 | ↑ +1.2 |
| v5 | X.X/10 | ↑ MUST IMPROVE |

⚠️ If score would decrease, STOP and investigate.

### Certification

- [ ] All locks preserved
- [ ] No regressions detected  
- [ ] Score trend is positive
- [ ] Ready for delivery
```

### Revision Loop Commands

```bash
# User triggers revision
"Arthur, revise paper for Round 1 reviews: [paste reviews]"

# Agent delivers revised paper + response letter

# User re-submits to journal

# Repeat until:
"Arthur, the paper was ACCEPTED! 🎉"
```

### Acceptance Celebration 🎉

When paper is accepted:
1. Archive all versions
2. Update with camera-ready requirements
3. Generate final publication-ready version
4. Log in memory as completed publication

## 🔴 ADVERSARIAL RED-TEAM REVIEWER (NEW)

### Purpose

A dedicated agent whose **sole job is to break the paper**. Unlike the peer-reviewer (who assesses overall quality), the red-team-reviewer actively tries to find fatal flaws, logical holes, and reproducibility failures. This is inspired by red-teaming practices in cybersecurity and AI safety.

### Philosophy

> "A paper that survives adversarial attack is a paper worth publishing."

The red-team-reviewer assumes the paper is **wrong** and tries to prove it. If they fail to find fatal flaws, the paper is genuinely strong.

### Red-Team Attack Vectors

```
┌─────────────────────────────────────────────────────────────┐
│              RED-TEAM ATTACK PROTOCOL                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Attack 1: CLAIM DESTRUCTION                                │
│  ├── Find the strongest claim in the paper                  │
│  ├── Search for contradicting evidence                      │
│  ├── Check if claim survives worst-case assumptions         │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
│  Attack 2: METHODOLOGY STRESS TEST                          │
│  ├── Identify methodological assumptions                    │
│  ├── Violate each assumption — does conclusion hold?        │
│  ├── Check for confounding variables not addressed          │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
│  Attack 3: BASELINE CHALLENGE                               │
│  ├── Are baselines truly state-of-the-art?                  │
│  ├── Search for stronger baselines published recently       │
│  ├── Would a simpler approach achieve similar results?      │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
│  Attack 4: REPRODUCIBILITY AUDIT                            │
│  ├── Can results be reproduced from paper alone?            │
│  ├── Are hyperparameters, seeds, data splits specified?     │
│  ├── Is compute budget realistic and verified?              │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
│  Attack 5: NOVELTY CHALLENGE                                │
│  ├── Search arXiv/Scholar for extremely similar prior work  │
│  ├── Is the contribution genuinely new or incremental?      │
│  ├── Could reviewers see this as "well-known approach"?     │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
│  Attack 6: STATISTICAL ADVERSARY                            │
│  ├── Are p-values near threshold (p=0.04x)?                 │
│  ├── Would different test choices change conclusions?       │
│  ├── Is there evidence of p-hacking or cherry-picking?      │
│  └── Score: Fatal / Weakening / Survived                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Red-Team Report Template

```markdown
# 🔴 RED-TEAM REPORT

## Paper: [Title]
## Red-Team Reviewer: red-team-reviewer (Opus)
## Date: [Date]

### Overall Verdict: [SURVIVED / NEEDS WORK / FATAL FLAW FOUND]

### Attack Results

| # | Attack Vector | Severity | Finding |
|---|---------------|----------|---------|
| 1 | Claim Destruction | ✅ Survived | Strongest claim holds under scrutiny |
| 2 | Methodology Stress | ⚠️ Weakening | Assumption X not addressed |
| 3 | Baseline Challenge | ❌ Fatal | Baseline Y (2025) outperforms |
| 4 | Reproducibility | ✅ Survived | All params specified |
| 5 | Novelty Challenge | ⚠️ Weakening | Similar to [Paper Z] |
| 6 | Statistical Adversary | ✅ Survived | Results robust |

### Fatal Flaws (must fix before submission)
1. [Description + evidence + fix recommendation]

### Weaknesses (should fix, not fatal)
1. [Description + evidence + fix recommendation]

### Survived Attacks (strengths to highlight)
1. [What makes the paper resilient]
```

### Integration in Pipeline

```
Standard Review Pipeline:
peer-reviewer + methodology-expert + technical-writer → Consensus

THEN (after consensus):

Red-Team Phase:
red-team-reviewer → Adversarial Attack Report
   │
   ├── No fatal flaws → Proceed to submission
   │
   └── Fatal flaw found → Back to revision
       (must fix before re-entering consensus)
```

### When to Deploy

- ✅ **Always** before first submission to any venue
- ✅ **Always** before camera-ready submission
- ✅ After major revisions (new claims or methodology changes)
- ⬚ Optional for minor editorial revisions

---

## 🗣️ STRUCTURED ARGUMENTATION IN CONSENSUS (NEW)

### Purpose

Replace simple "vote and average" consensus with **evidence-based argumentation**. Each reviewer must cite specific evidence for their position, and disagreements are resolved through structured debate — not averaging.

### Why This Matters

Research shows that structured argumentation reduces false positives by ~40% compared to majority voting (Tariq et al., 2025 — HIKMA framework). Simple scoring can mask fundamental disagreements.

### Argumentation Protocol

```
┌─────────────────────────────────────────────────────────────┐
│           STRUCTURED ARGUMENTATION WORKFLOW                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ROUND 1: Independent Assessment (~5 min each)              │
│  ├── peer-reviewer: Score + Evidence (cited paragraphs)     │
│  ├── methodology-expert: Score + Evidence                   │
│  └── technical-writer: Score + Evidence                     │
│                                                              │
│  ROUND 2: Position Statements (~3 min each)                 │
│  Each reviewer writes a structured position:                 │
│  ├── "I rate this [SCORE] because..."                       │
│  ├── "Evidence supporting my position: [specific text]"     │
│  ├── "I disagree with [other reviewer] on [point] because.."│
│  └── "My confidence level: [High/Medium/Low]"               │
│                                                              │
│  ROUND 3: Rebuttal & Convergence (~5 min)                   │
│  ├── Each reviewer responds to challenges                   │
│  ├── Update positions based on new arguments                │
│  └── Final score with justification                         │
│                                                              │
│  RESOLUTION:                                                 │
│  ├── If unanimous → Accept consensus                        │
│  ├── If 2-1 split → Minority must rebut or concede          │
│  └── If 3-way split → Escalate to human (Anirach)           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Position Statement Template

```markdown
## Reviewer Position: [peer-reviewer / methodology-expert / technical-writer]

### Overall Assessment: [ACCEPT / WEAK ACCEPT / WEAK REJECT / REJECT]
### Confidence: [High / Medium / Low]

### Key Arguments (with evidence)

**Argument 1: [Claim]**
> Evidence: [Quote from paper, section reference]
> Supporting literature: [External citation if applicable]

**Argument 2: [Claim]**
> Evidence: [Quote from paper, section reference]

### Points of Agreement with Other Reviewers
- [List shared assessments]

### Points of Disagreement
- **I disagree with [reviewer] on [point]:**
  > Their claim: [what they said]
  > My counter-evidence: [specific reference]
  > Why this matters: [impact on overall assessment]

### Non-Negotiables (issues that MUST be addressed)
1. [Issue + specific fix required]

### Nice-to-Haves (improvements that would strengthen but aren't blocking)
1. [Suggestion]
```

### Consensus Resolution Rules

| Scenario | Action | Rationale |
|----------|--------|-----------|
| 3-0 ACCEPT | Proceed | Strong consensus |
| 2-1 ACCEPT (minority WEAK REJECT) | Minority writes rebuttal; if addressed → proceed | Ensure minority concern is real |
| 2-1 ACCEPT (minority REJECT) | Escalate to human | Fundamental disagreement needs judgment |
| 3-way split | Escalate to human | No consensus achievable |
| Any REJECT with "Fatal" tag | Must address before proceeding | Fatal flaws are non-negotiable |
| Confidence mismatch (High vs Low) | Weight toward high-confidence reviewer | Domain expertise matters |

### Argumentation Log

Every consensus round produces:

```markdown
# Argumentation Log — [Paper Title] v[N]

## Round 1 Scores
| Reviewer | Score | Confidence |
|----------|-------|------------|
| peer-reviewer | WEAK ACCEPT | High |
| methodology-expert | ACCEPT | Medium |
| technical-writer | ACCEPT | High |

## Key Disagreements
1. [Issue]: peer-reviewer vs methodology-expert
   - Resolution: [How resolved + evidence]

## Final Consensus: ACCEPT
## Resolved by: Round 2 argumentation (no escalation needed)
```

---

## 🔍 NOVELTY VERIFICATION GATE (NEW)

### Purpose

Before submission, systematically verify that the paper's contributions are genuinely novel — not already published or simultaneously submitted elsewhere. This catches the "novelty challenge" that reviewers frequently raise.

### Why This Matters

Industry data shows AI-generated papers score lowest on **Novelty (2.1-2.9/5)** in human evaluations. A dedicated novelty gate catches this before reviewers do.

### Novelty Verification Protocol

```
┌─────────────────────────────────────────────────────────────┐
│              NOVELTY VERIFICATION GATE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: CONTRIBUTION EXTRACTION                            │
│  ├── List all claimed contributions (from Introduction)     │
│  ├── For each: extract the specific novel element           │
│  └── Classify: Conceptual / Methodological / Empirical      │
│                                                              │
│  Step 2: PRIOR ART SEARCH (per contribution)                │
│  ├── Semantic Scholar: similarity search on contribution    │
│  ├── arXiv: recent preprints in same area (last 12 months) │
│  ├── Google Scholar: exact phrase + paraphrase search       │
│  └── Connected Papers: citation graph exploration           │
│                                                              │
│  Step 3: SIMILARITY ASSESSMENT                              │
│  For each prior work found:                                  │
│  ├── Overlap percentage (Low < 30% / Med 30-60% / High 60%+)│
│  ├── What's different in our approach?                       │
│  ├── Is the difference meaningful or cosmetic?              │
│  └── Would a reviewer see this as "already done"?           │
│                                                              │
│  Step 4: NOVELTY VERDICT                                    │
│  ├── ✅ NOVEL: No significant overlap found                 │
│  ├── ⚠️ INCREMENTAL: Similar work exists but clear delta    │
│  ├── ❌ NOT NOVEL: Substantially duplicated                 │
│  └── 🔄 CONCURRENT: Similar work submitted simultaneously  │
│                                                              │
│  Step 5: DIFFERENTIATION STRENGTHENING                      │
│  If ⚠️ INCREMENTAL:                                        │
│  ├── Explicitly cite the similar work                       │
│  ├── Add comparison table showing differences               │
│  ├── Strengthen the "what's new" argument                   │
│  └── Consider repositioning the contribution                │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Novelty Report Template

```markdown
# 🔍 NOVELTY VERIFICATION REPORT

## Paper: [Title]
## Date: [Date]
## Databases Searched: Semantic Scholar, arXiv, Google Scholar, Connected Papers

### Claimed Contributions

| # | Contribution | Type | Novelty Status |
|---|-------------|------|----------------|
| 1 | [Description] | Conceptual | ✅ NOVEL |
| 2 | [Description] | Methodological | ⚠️ INCREMENTAL |
| 3 | [Description] | Empirical | ✅ NOVEL |

### Detailed Analysis

#### Contribution 1: [Description]
- **Prior art search:** [X] papers examined
- **Closest match:** [Paper title, authors, year]
- **Overlap:** Low (< 30%)
- **Our differentiation:** [What makes ours different]
- **Verdict:** ✅ NOVEL — No prior work combines [A] with [B]

#### Contribution 2: [Description]
- **Prior art search:** [X] papers examined
- **Closest match:** [Paper title, authors, year]
- **Overlap:** Medium (45%)
- **Our differentiation:** [Specific differences]
- **Verdict:** ⚠️ INCREMENTAL — Must explicitly cite [Paper] and clarify delta
- **Action required:** Add comparison table in Section 2.3

### Concurrent Work Alert
- [Paper on arXiv from last 3 months with similar approach]
- **Risk level:** [Low/Medium/High]
- **Mitigation:** [Cite and differentiate]

### Overall Novelty Score: [Strong / Adequate / Weak]
### Gate Decision: [PASS / PASS WITH CHANGES / FAIL]
```

### Integration in Pipeline

```
Verification Phase (existing):
├── Citation Verification ✅
├── Math Verification ✅
├── Code Verification ✅
├── Reproducibility Check ✅
├── 🆕 NOVELTY VERIFICATION ← new gate
│   ├── Contribution extraction
│   ├── Multi-database prior art search
│   ├── Similarity assessment
│   └── Differentiation strengthening (if needed)
└── Red-Team Review ✅
```

### Tools Used

| Tool | Purpose | How |
|------|---------|-----|
| `python3 tools/semantic_scholar.py` | Semantic similarity search | Query each contribution |
| `agentarxiv` skill | arXiv preprint search | Recent papers in area |
| `scholargraph` skill | Citation graph exploration | Connected Papers analysis |
| `literature-search` skill | Multi-database search | Google Scholar, IEEE, ACM |

---

## 📐 REVIEWER SCORE CALIBRATION SYSTEM (NEW)

### Purpose

Track the correlation between our internal review scores and actual conference outcomes. Over time, this calibrates our reviewers to predict real acceptance/rejection more accurately.

### Why This Matters

Without calibration, internal scores are meaningless numbers. A "WEAK ACCEPT" from our peer-reviewer should correlate with actual conference outcomes. If our internal "ACCEPT" papers get rejected, or our "WEAK REJECT" papers get accepted, the scoring system needs adjustment.

### Calibration Database

**Location:** `/home/clawdbot/clawd/papers/CALIBRATION_DB.md`

```markdown
# Reviewer Score Calibration Database

## Completed Papers

| Paper | Version | Internal Score | Venue | Actual Outcome | Delta |
|-------|---------|---------------|-------|----------------|-------|
| RAG Second Brain | v23.1 | 3x ACCEPT | AIiH 2026 | ACCEPT (all 3) | 0 ✅ |
| NCD-CIE | v20 | 3x WEAK ACCEPT | AIiH 2026 | Pending | — |

## Calibration Metrics

### Score-to-Outcome Mapping (updated after each decision)

| Internal Score | Expected Outcome | Actual Outcome | Accuracy |
|---------------|-------------------|----------------|----------|
| 3x ACCEPT | Accept | [track] | [%] |
| 2x ACCEPT + 1x WA | Accept | [track] | [%] |
| 3x WEAK ACCEPT | Borderline | [track] | [%] |
| Mixed (any REJECT) | Revise/Reject | [track] | [%] |

### Per-Reviewer Accuracy

| Reviewer Agent | Predictions | Correct | Accuracy | Bias |
|---------------|-------------|---------|----------|------|
| peer-reviewer | [N] | [N] | [%] | [optimistic/pessimistic/neutral] |
| methodology-expert | [N] | [N] | [%] | [bias] |
| technical-writer | [N] | [N] | [%] | [bias] |
| red-team-reviewer | [N] | [N] | [%] | [bias] |

### Calibration Adjustments

Based on accumulated data:
- If peer-reviewer is consistently optimistic → weight down by [X]
- If methodology-expert catches issues others miss → weight up for methodology papers
- If red-team "fatal flaws" correlate with rejections → trust red-team more
```

### Calibration Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              CALIBRATION LOOP                                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Paper goes through full review pipeline                  │
│     ├── Each reviewer records prediction + confidence        │
│     └── Saved to CALIBRATION_DB.md                          │
│                                                              │
│  2. Paper submitted to venue                                 │
│                                                              │
│  3. When outcome received:                                   │
│     ├── Record actual reviewer scores                        │
│     ├── Record accept/reject decision                        │
│     ├── Compare against internal predictions                 │
│     └── Update calibration metrics                          │
│                                                              │
│  4. After 5+ data points:                                    │
│     ├── Calculate per-reviewer accuracy                      │
│     ├── Identify systematic biases                          │
│     ├── Adjust weighting in consensus                       │
│     └── Update reviewer prompts if needed                   │
│                                                              │
│  5. After 10+ data points:                                   │
│     ├── Statistical significance test on biases             │
│     ├── Formal calibration report                           │
│     └── Adjust score thresholds for submission decisions    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Post-Decision Review Template

```markdown
# Post-Decision Calibration — [Paper Title]

## Venue: [Conference/Journal]
## Submitted Version: v[N]
## Decision: [ACCEPT / REVISE / REJECT]

## Internal vs Actual Comparison

| Reviewer | Internal Prediction | Actual Reviewer Score | Delta |
|----------|--------------------|-----------------------|-------|
| peer-reviewer | ACCEPT (conf: High) | Reviewer 1: Accept | ✅ Match |
| methodology-expert | WEAK ACCEPT (conf: Med) | Reviewer 2: Weak Accept | ✅ Match |
| technical-writer | ACCEPT (conf: High) | Reviewer 3: Reject | ❌ Miss |

## What We Got Right
- [Strengths that reviewers also praised]

## What We Missed
- [Issues reviewers raised that we didn't catch]
- [Why our reviewer missed this — prompt gap? knowledge gap?]

## Calibration Actions
- [ ] Update peer-reviewer prompt to check for [missed issue]
- [ ] Adjust methodology-expert weighting: [up/down/same]
- [ ] Add new attack vector to red-team: [new attack type]
- [ ] Update QUALITY_LOCKS with new lesson
```

### Automatic Calibration Trigger

After receiving any venue decision:
```
Arthur, we received the decision for "[Paper Title]":
[Paste reviewer comments and decision]

Please update the calibration database.
```

Arthur will:
1. Parse the actual reviews
2. Compare against internal predictions
3. Update CALIBRATION_DB.md
4. Identify missed issues
5. Recommend reviewer prompt adjustments

---

## 💰 Cost Optimization

- **Opus** for critical thinking (architect, literature, methodology, peer-review, red-team)
- **Sonnet** for execution (drafting, formatting)
- Literature search batched efficiently
- Parallel section drafting when possible

## 🛠️ Tools & Resources

### Literature Search
- **Semantic Scholar API** — citation graphs, relevance
- **PubMed/MEDLINE** — biomedical
- **arXiv** — preprints (CS, physics, math)
- **Google Scholar** — broad coverage
- **IEEE Xplore** — engineering
- **ACM Digital Library** — computing

### Citation Management
- BibTeX format for all references
- DOI resolution for verification
- CrossRef for metadata

### Formatting
- Pandoc for format conversion
- LaTeX templates for IEEE/ACM
- Word templates for journals requiring .docx

---

## 📊 Data Analyst Agent (NEW)

**Purpose:** Statistical analysis, visualization, and reproducibility verification.

### Capabilities

| Task | Description |
|------|-------------|
| **Statistical Analysis** | t-tests, ANOVA, regression, Bayesian |
| **Visualization** | Matplotlib, Seaborn, publication-quality |
| **Reproducibility** | Fixed seeds, environment specs |
| **Results Tables** | Auto-generate LaTeX tables |
| **Ablation Studies** | Systematic component analysis |

### Statistical Rigor Checklist
```markdown
□ Sample size justified (power analysis)
□ Normality tested (if parametric)
□ Multiple comparisons corrected (Bonferroni/FDR)
□ Effect sizes reported (Cohen's d, etc.)
□ Confidence intervals included
□ Bootstrap for non-parametric
□ Random seeds fixed and documented
```

### Output Files
- `results/tables/*.tex` — LaTeX tables
- `results/figures/*.pdf` — Publication figures
- `results/stats/STATISTICAL_ANALYSIS.md` — Full analysis
- `results/reproducibility/REPRO_CHECKLIST.md` — Reproducibility

### Integration with Experiments
```
methodology-expert ──► Designs experiment
        │
        ▼
   Coding Team ──► Implements & runs
        │
        ▼
   data-analyst ──► Analyzes results
        │
        ▼
   technical-writer ──► Writes Results section
```

---

## ⚖️ Ethics Reviewer Agent (NEW)

**Purpose:** Ensure ethical compliance, data privacy, and research integrity.

### Review Areas

| Area | Checks |
|------|--------|
| **IRB Compliance** | Human subjects approval documented |
| **Data Privacy** | GDPR, HIPAA compliance if applicable |
| **Consent** | Informed consent obtained |
| **Conflicts of Interest** | Disclosed appropriately |
| **Data Handling** | Secure storage, anonymization |
| **Authorship** | Proper attribution, no ghost authors |

### Ethics Statement Template
```markdown
## Ethics Statement

### Human Subjects
This study [did/did not] involve human subjects.
[If yes: IRB approval number: XXX-XXXX]

### Data Privacy
All data was [anonymized/de-identified].
No personally identifiable information (PII) was collected.

### Consent
All participants provided informed consent.

### Conflicts of Interest
The authors declare [no conflicts / the following conflicts: ...]

### Data Availability
Data is available at [repository URL] under [license].
```

### Red Flags to Catch
- 🚩 Missing IRB for human subjects
- 🚩 PII in published data
- 🚩 Unclear consent process
- 🚩 Undisclosed funding
- 🚩 Ghost authorship
- 🚩 Duplicate publication

---

## 🧪 Experiment Automation (NEW)

Automated experiment pipeline for reproducible results.

### Workflow
```
┌─────────────────────────────────────────────────────────────┐
│              EXPERIMENT AUTOMATION PIPELINE                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. methodology-expert designs experiment                    │
│         │                                                    │
│         ▼                                                    │
│  2. Spawn code-builder (Coding Team)                        │
│     ├── Create experiment scripts                           │
│     ├── Set up dependencies                                 │
│     └── Document environment                                │
│         │                                                    │
│         ▼                                                    │
│  3. Run experiments (Colab/local/cloud)                     │
│     ├── Log all parameters                                  │
│     ├── Save intermediate results                           │
│     └── Track compute time/cost                             │
│         │                                                    │
│         ▼                                                    │
│  4. data-analyst processes results                          │
│     ├── Statistical tests                                   │
│     ├── Generate figures                                    │
│     └── Create tables                                       │
│         │                                                    │
│         ▼                                                    │
│  5. paper-architect integrates into paper                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Experiment Config Template
```yaml
experiment:
  name: "RAG Fusion Comparison"
  version: "v16"
  
seeds:
  - 42
  - 123
  - 456
  
datasets:
  - name: "HotpotQA"
    split: "dev"
    samples: 1000
    
models:
  - name: "E5-base"
    path: "intfloat/e5-base-v2"
    
baselines:
  - "BM25"
  - "DPR"
  - "RRF"
  
metrics:
  - "Recall@5"
  - "Recall@10"
  - "EM"
  - "F1"
  
compute:
  platform: "colab"
  gpu: "T4"
  estimated_hours: 4
```

---

## 📚 Citation Intelligence (NEW)

Smart citation management beyond basic reference tracking.

### Features

| Feature | Description |
|---------|-------------|
| **Related Paper Discovery** | Auto-find similar papers via Semantic Scholar |
| **Citation Gap Analysis** | Identify missing citations in your field |
| **Predatory Journal Detection** | Flag questionable sources |
| **Citation Trend Tracking** | Which papers are gaining citations |
| **Self-Citation Check** | Ensure balanced self-citation |

### Citation Health Report
```markdown
## Citation Health Report

### Summary
- Total citations: 45
- Unique sources: 42
- Self-citations: 3 (6.7%) ✅ Acceptable
- Predatory journals: 0 ✅
- Retracted papers: 0 ✅
- Broken DOIs: 2 ⚠️

### Recency Distribution
- 2024-2026: 25 (56%) ✅
- 2021-2023: 15 (33%) ✅
- Pre-2020: 5 (11%) — Classic references ✅

### Missing Citations (Suggested)
Based on your topic, consider citing:
1. [Paper 1] — Highly cited, relevant
2. [Paper 2] — Recent breakthrough
3. [Paper 3] — Standard baseline

### Issues to Fix
1. DOI not found: [ref 23]
2. DOI not found: [ref 41]
```

---

## 🎯 Multi-Journal Targeting (NEW)

Prepare paper for multiple journals simultaneously.

### Workflow
```
1. Write paper in "neutral" format
2. Generate journal-specific versions:
   - IEEE format
   - ACM format  
   - Nature format
   - arXiv preprint

3. Track submissions:
   | Journal | Status | Submitted | Decision |
   |---------|--------|-----------|----------|
   | IEEE TPAMI | Under Review | 2026-02-10 | Pending |
   | Backup: ACM | Ready | — | — |
```

### Rejection → Reformat Pipeline
```
If rejected from Journal A:
1. Review feedback
2. Address major issues
3. Auto-reformat for Journal B
4. Update submission tracker
5. Submit to Journal B
```

---

## 📤 Preprint & Archive Integration (NEW)

Auto-submit to preprint servers and data repositories.

### Supported Platforms
- **arXiv** — CS, physics, math preprints
- **bioRxiv** — Biology preprints
- **medRxiv** — Medical preprints
- **Zenodo** — Data & code archival
- **GitHub** — Code repository
- **HuggingFace** — Model hosting

### Archive Checklist
```markdown
□ arXiv submission prepared
□ Code uploaded to GitHub
□ Data uploaded to Zenodo
□ Models uploaded to HuggingFace
□ DOIs minted for all artifacts
□ Paper links to all repositories
```

---

---

## 🎭 VENUE-SPECIFIC REVIEWER PERSONAS (#1)

### Purpose

Instead of generic review criteria, reviewers adopt the actual reviewing style, scoring rubric, and pet peeves of the target venue. This dramatically improves review accuracy.

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│              VENUE-AWARE REVIEW PIPELINE                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Identify target venue (LNCS, NeurIPS, IEEE, ACM, ICML) │
│         │                                                    │
│         ▼                                                    │
│  2. Load venue profile from:                                │
│     templates/venue-profiles/{VENUE}.md                      │
│         │                                                    │
│         ▼                                                    │
│  3. All reviewer agents ADOPT venue persona:                │
│     ├── peer-reviewer → uses venue scoring rubric            │
│     ├── methodology-expert → applies venue rigor standards   │
│     ├── technical-writer → checks venue format/style norms   │
│     └── red-team-reviewer → targets venue-specific red flags │
│         │                                                    │
│         ▼                                                    │
│  4. Reviews scored on VENUE CRITERIA, not generic            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Available Venue Profiles

| Venue | Profile | Key Reviewer Focus |
|-------|---------|-------------------|
| LNCS (Springer) | `templates/venue-profiles/LNCS.md` | Page limits, novelty framing, format compliance |
| NeurIPS | `templates/venue-profiles/NeurIPS.md` | Technical novelty, theory, reproducibility |
| IEEE | `templates/venue-profiles/IEEE.md` | Engineering rigor, practical applicability |
| ACM | `templates/venue-profiles/ACM.md` | User impact, artifacts, reproducibility |
| ICML | `templates/venue-profiles/ICML.md` | Methodology, theoretical depth |

### Reviewer Prompt Injection

When reviewing, each agent prepends:

```
You are reviewing this paper AS IF you are a {VENUE} reviewer.
Load and follow the scoring rubric, acceptance criteria, and red flags from:
{venue_profile_path}

Score using the venue's actual criteria. Flag venue-specific issues.
Your review should read like a real {VENUE} review, not a generic assessment.
```

### Adding New Venue Profiles

Create `templates/venue-profiles/{VENUE}.md` with:
- Reviewer persona & mindset
- Scoring rubric with weights
- Common rejection reasons (venue-specific)
- Hedging/style expectations
- Citation patterns
- Format requirements
- Red flags

---

## 🗺️ AUTOMATED RELATED WORK POSITIONING MAP (#2)

### Purpose

Auto-generate a visual diagram showing where our paper sits relative to existing work. Catches "how is this different from X?" before reviewers ask.

### Tool

```bash
# Generate sample config
python3 tools/academic/positioning_map.py --sample --output positioning.json

# Generate positioning map from config
python3 tools/academic/positioning_map.py --config positioning.json --output POSITIONING_MAP.md
```

### Output Includes

1. **Quadrant Chart** — Mermaid diagram plotting our paper vs related work on two key dimensions
2. **Threat Analysis** — Which related works are closest (🔴 high / 🟡 medium / 🟢 low threat)
3. **Novelty Map** — Contribution-level overlap visualization
4. **Comparison Matrix** — Detailed feature comparison table
5. **Preemptive Q&A** — "How is this different from X?" answers ready for rebuttal

### Integration in Pipeline

```
After literature-lead completes literature review:
├── Extract key related works + their approaches
├── Classify threat level (high/medium/low overlap)
├── Generate positioning config JSON
├── Run positioning_map.py
├── Include map in Related Work section planning
└── Use preemptive Q&A in paper framing
```

### When to Generate

- ✅ After literature review is complete
- ✅ Before writing Related Work section
- ✅ After finding a new high-threat related work
- ✅ Before submission (final check)

---

## 🔍 WRITING STYLE FORENSICS (#3)

### Purpose

Analyze our paper against accepted papers from the target venue. Check sentence length, passive voice, hedging, section proportions, and AI-writing signals. Flag deviations.

### Tool

```bash
# Analyze paper against LNCS norms
python3 tools/academic/style_forensics.py paper.md --venue LNCS

# Analyze against NeurIPS norms, save report
python3 tools/academic/style_forensics.py paper.tex --venue NeurIPS --output forensics_report.md

# Get raw JSON for programmatic use
python3 tools/academic/style_forensics.py paper.md --venue IEEE --json
```

### What It Checks

| Check | Why It Matters |
|-------|---------------|
| Sentence length distribution | AI writes uniform lengths; humans vary |
| Passive voice ratio | Venues differ: IEEE tolerates more, NeurIPS less |
| Hedging frequency | Overclaiming vs under-claiming detection |
| Section proportions | Methods too short? Results too long? |
| AI-writing signals | Repetitive starters, filler phrases, suspicious uniformity |
| Paragraph length variation | AI paragraphs are suspiciously uniform |

### AI-Detection Signals

The tool flags patterns that signal "AI-written" to reviewers:

- 🔴 **Repetitive transition phrases** — "Furthermore," "Moreover," "Additionally," used >3 times
- 🔴 **Filler phrases** — "plays a crucial role," "of paramount importance," "delve into"
- 🔴 **Suspicious uniformity** — Sentence length std/mean < 0.15 (humans: 0.35-0.55)
- 🟡 **Paragraph uniformity** — All paragraphs same length

### Integration in Pipeline

```
Before final submission:
├── Run style_forensics.py on complete paper
├── Fix any 🔴 high-severity signals
├── Adjust style to match venue norms
├── Re-run to confirm fixes
└── Include forensics_report.md in VERIFICATION_RESULTS
```

---

## 🔄 MULTI-ROUND SELF-IMPROVEMENT LOOP (#4)

### Purpose

After red-team finds weaknesses, automatically revise and re-run the full review pipeline (up to 3 rounds) without human intervention. Currently one-shot — this makes it iterative.

### Protocol

```
┌─────────────────────────────────────────────────────────────┐
│           MULTI-ROUND SELF-IMPROVEMENT LOOP                 │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ROUND 1: Standard Pipeline                                 │
│  ├── Full review (peer + methodology + technical)           │
│  ├── Consensus via structured argumentation                 │
│  ├── Novelty verification                                   │
│  ├── Red-team review                                        │
│  └── Style forensics                                        │
│         │                                                    │
│         ▼                                                    │
│  Improvement Analysis:                                       │
│  ├── Collect ALL issues (reviews + red-team + forensics)    │
│  ├── Categorize: Fatal / Major / Minor / Style              │
│  ├── If no Fatal/Major issues → EXIT (paper ready)          │
│  └── If issues found → AUTO-REVISE                          │
│         │                                                    │
│         ▼                                                    │
│  ROUND 2: Targeted Revision                                 │
│  ├── Address Fatal issues first                             │
│  ├── Address Major issues                                   │
│  ├── Lock all Round 1 praise (QUALITY_LOCKS)                │
│  ├── Re-run ONLY affected review checks                     │
│  ├── Re-run red-team on changed sections                    │
│  └── Re-run style forensics                                 │
│         │                                                    │
│         ▼                                                    │
│  Improvement Check:                                          │
│  ├── Score trend MUST be positive                           │
│  ├── No new Fatal issues introduced                         │
│  ├── All Round 1 locks preserved                            │
│  └── If still issues → ROUND 3 (final)                     │
│         │                                                    │
│         ▼                                                    │
│  ROUND 3: Final Polish (if needed)                          │
│  ├── Address remaining Minor issues                         │
│  ├── Style alignment fixes                                  │
│  ├── Final comprehensive review                             │
│  └── EXIT — deliver regardless (with any remaining issues   │
│       documented in KNOWN_ISSUES.md)                        │
│                                                              │
│  ⚠️ HARD LIMIT: 3 rounds max to prevent infinite loops     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Round Tracking

```markdown
# IMPROVEMENT_ROUNDS.md

## Round 1
- Issues found: [N Fatal, M Major, K Minor]
- Auto-revisions made: [list]
- Score: [X/10]

## Round 2
- Issues from Round 1 addressed: [list]
- New issues: [any?]
- Locks preserved: ✅/❌
- Score: [X/10] (must be > Round 1)

## Round 3 (if needed)
- Remaining issues: [list]
- Final score: [X/10]
- Known issues documented: [Y/N]
```

### Exit Conditions

| Condition | Action |
|-----------|--------|
| No Fatal/Major after Round 1 | Exit — paper ready |
| All Fixed after Round 2 | Exit — paper ready |
| Round 3 complete | Exit — deliver with KNOWN_ISSUES.md |
| Score drops between rounds | STOP — investigate regression |
| New Fatal issue introduced | STOP — human review needed |

### Spawn Command

```
Arthur, write a paper on [TOPIC] with self-improvement loop enabled.
Max rounds: 3
```

---

## 📋 REJECTION AUTOPSY PROTOCOL (#5)

### Purpose

Structured post-mortem when a paper gets rejected. What did we miss? Feed findings back into ALL reviewer agents as permanent lessons.

### Location

```
papers/CROSS_PAPER_KB/AUTOPSY_LOG.md
```

### Trigger

```
Arthur, paper "[Title]" was rejected from [Venue]. Here are the reviews:
[paste reviews]

Run rejection autopsy.
```

### Autopsy Process

```
┌─────────────────────────────────────────────────────────────┐
│              REJECTION AUTOPSY PROTOCOL                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Step 1: PARSE actual reviewer comments                     │
│  ├── Extract each concern                                   │
│  ├── Categorize severity                                    │
│  └── Identify the kill shot (primary rejection reason)      │
│                                                              │
│  Step 2: COMPARE with internal review                       │
│  ├── Which concerns did we catch?                           │
│  ├── Which did we miss entirely?                            │
│  └── Which did we see but underweight?                      │
│                                                              │
│  Step 3: ROOT CAUSE ANALYSIS (5 Whys)                       │
│  ├── Why was it rejected?                                   │
│  ├── Why did that happen?                                   │
│  ├── Why wasn't it caught?                                  │
│  ├── Why does the pipeline have that gap?                   │
│  └── What's the systemic fix?                               │
│                                                              │
│  Step 4: CALIBRATION UPDATE                                 │
│  ├── Update CALIBRATION_DB.md                               │
│  ├── Adjust per-reviewer accuracy scores                    │
│  └── Identify bias patterns                                 │
│                                                              │
│  Step 5: KNOWLEDGE TRANSFER                                 │
│  ├── Add new patterns to CROSS_PAPER_KB                     │
│  ├── Update venue profile if venue-specific                 │
│  ├── Add new red-team attack vector if applicable           │
│  └── Update reviewer agent prompts with new lessons         │
│                                                              │
│  Step 6: RESUBMISSION PLAN                                  │
│  ├── Same venue revision or pivot?                          │
│  ├── Key changes needed (prioritized)                       │
│  └── Timeline estimate                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Deliverables

| File | Content |
|------|---------|
| `AUTOPSY_LOG.md` (updated) | Full post-mortem |
| `KNOWLEDGE_BASE.md` (updated) | New patterns + anti-patterns |
| `CALIBRATION_DB.md` (updated) | Prediction accuracy |
| Venue profile (updated) | Any venue-specific lessons |
| Resubmission plan | If applicable |

---

## 🧠 CROSS-PAPER KNOWLEDGE TRANSFER (#6)

### Purpose

Lessons from Paper A's review process inform Paper B's writing from the start. Accumulated wisdom across all papers.

### Knowledge Base

```
papers/CROSS_PAPER_KB/
├── KNOWLEDGE_BASE.md    # Accumulated lessons, patterns, anti-patterns
├── AUTOPSY_LOG.md       # Rejection post-mortems
└── [future additions]
```

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│           CROSS-PAPER KNOWLEDGE TRANSFER                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  WHEN STARTING ANY NEW PAPER:                               │
│                                                              │
│  1. paper-architect reads KNOWLEDGE_BASE.md                 │
│     ├── Reviewer prediction patterns → preemptive defense    │
│     ├── Writing patterns that work → incorporate             │
│     ├── Writing patterns that fail → avoid                   │
│     ├── Venue-specific lessons → apply                       │
│     └── Tool/process lessons → follow                        │
│                                                              │
│  2. All reviewer agents loaded with accumulated lessons     │
│     ├── Known reviewer concerns → check proactively         │
│     ├── Known anti-patterns → flag immediately              │
│     └── Calibration data → adjust scoring                   │
│                                                              │
│  AFTER EACH REVIEW CYCLE:                                   │
│                                                              │
│  3. Extract new lessons → add to KNOWLEDGE_BASE.md          │
│     ├── New reviewer patterns                               │
│     ├── What worked / what failed                           │
│     ├── Venue-specific discoveries                          │
│     └── Tool/process improvements                           │
│                                                              │
│  4. If rejected → run Rejection Autopsy (#5)                │
│     └── Findings feed into KB automatically                 │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Current Knowledge (from RAG-KB + NCD-CIE)

Already populated with:
- 6 reviewer prediction patterns
- 5 proven writing strategies  
- 5 known anti-patterns
- LNCS + AIiH venue lessons
- 5 tool/process lessons

See: `papers/CROSS_PAPER_KB/KNOWLEDGE_BASE.md`

---

## 🔮 REVIEWER COMMENT PREDICTION (#7)

### Purpose

Before submission, predict the top 5 likely reviewer comments. Allows preemptive strengthening of weak areas.

### Protocol

```
┌─────────────────────────────────────────────────────────────┐
│           REVIEWER COMMENT PREDICTION                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  INPUTS:                                                     │
│  ├── Complete paper draft                                   │
│  ├── Target venue profile                                   │
│  ├── CROSS_PAPER_KB patterns                                │
│  ├── Red-team findings                                      │
│  └── Style forensics results                                │
│                                                              │
│  PREDICTION PROCESS:                                         │
│                                                              │
│  Phase 1: Pattern Matching                                   │
│  ├── Match paper characteristics against KNOWLEDGE_BASE     │
│  ├── Check for known trigger patterns                       │
│  └── Weight by venue-specific tendencies                    │
│                                                              │
│  Phase 2: Simulated Review                                   │
│  ├── peer-reviewer: "What would I ask?"                     │
│  ├── methodology-expert: "What would I challenge?"          │
│  ├── red-team: "What's the weakest link?"                   │
│  └── Each generates top 3 predicted comments                │
│                                                              │
│  Phase 3: Consolidation                                      │
│  ├── Merge & deduplicate predictions                        │
│  ├── Rank by likelihood (from KB patterns + review)         │
│  ├── Select top 5 most likely comments                      │
│  └── Generate preemptive responses/fixes                    │
│                                                              │
│  OUTPUT: PREDICTED_COMMENTS.md                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Prediction Report Template

```markdown
# 🔮 Predicted Reviewer Comments

**Paper:** [Title]
**Venue:** [Target]
**Prediction Date:** [Date]
**Based on:** [N] historical patterns + simulated review

## Top 5 Predicted Comments

### 1. [Most Likely Comment] — Confidence: HIGH (85%)
> "The comparison with [recent baseline] is missing..."

**Why predicted:** Pattern from KNOWLEDGE_BASE + missing from Related Work
**Preemptive fix:** Add comparison in Table 3, cite [Paper], 2 paragraphs in Section 2.3
**Status:** [ ] Fixed / [ ] Intentionally not addressed (with justification)

### 2. [Second Most Likely] — Confidence: HIGH (80%)
> "The evaluation is limited to..."

**Why predicted:** Single-dataset evaluation, venue expects multi-benchmark
**Preemptive fix:** Add [Benchmark B] results in Section 5
**Status:** [ ] Fixed / [ ] Intentionally not addressed

### 3. [Third Most Likely] — Confidence: MEDIUM (65%)
> "How does this differ from [Similar Paper]?"

**Why predicted:** Positioning map shows high overlap with [Paper]
**Preemptive fix:** Add explicit differentiation paragraph in Section 2
**Status:** [ ] Fixed / [ ] Intentionally not addressed

### 4. [Fourth] — Confidence: MEDIUM (55%)
> "Reproducibility concerns..."

**Why predicted:** Missing hyperparameter details in methodology
**Preemptive fix:** Add Table 4 with all hyperparameters
**Status:** [ ] Fixed / [ ] Intentionally not addressed

### 5. [Fifth] — Confidence: LOW (40%)
> "The broader impact discussion is insufficient..."

**Why predicted:** Venue requires detailed ethics discussion
**Preemptive fix:** Expand Section 7 with concrete societal implications
**Status:** [ ] Fixed / [ ] Intentionally not addressed

## Confidence Calibration
- Historical accuracy of predictions: [X]% (from CALIBRATION_DB)
- Predictions that matched actual comments: [Y/Z]
```

### Integration in Pipeline

```
Pre-submission checklist (FINAL STEP before delivery):
├── Full review pipeline complete ✅
├── Self-improvement rounds complete ✅  
├── Run reviewer comment prediction
├── Address high-confidence predictions
├── Document any intentionally unaddressed predictions
└── Submit with confidence
```

### Post-Submission Calibration

After receiving actual reviews:
```
1. Compare predicted comments vs actual comments
2. Score prediction accuracy
3. Update CALIBRATION_DB.md
4. Refine prediction patterns in KNOWLEDGE_BASE.md
5. Adjust prediction confidence thresholds
```

---

*"If I have seen further, it is by standing on the shoulders of Giants."* — Isaac Newton

---

## 📓 Obsidian Workspace

All Academic Team work is documented in Obsidian:

**Location:** `/home/clawdbot/obsidian-vault/Agents/Academic/`

```
Academic/
├── Projects/           # Active papers
│   └── [Paper-Title].md
├── Notes/              # Literature & research
│   ├── Literature/     # Paper reviews
│   ├── Methodology/    # Methods notes
│   └── Data/           # Data analysis notes
└── Templates/          # Paper templates
    └── Paper Project.md
```

**For each paper, create:**
1. `Projects/[Paper-Title].md` — Overview, status, verification
2. `Notes/Literature/[Paper-Title]/` — Literature review notes
3. Link citations with `[[Author Year]]`

**Tags:** `#paper`, `#academic`, `#literature`, `#methodology`, `#verified`
