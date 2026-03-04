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
                                            ├── 6. Internal peer review
                                            ├── 7. Fact-check & verify citations
                                            ├── 8. Format to journal specs
                                            ├── 9. ⭐ VERIFICATION PHASE ⭐
                                            │       ├── Verify ALL citations exist
                                            │       ├── Test ALL code (if any)
                                            │       ├── Reproduce ALL results
                                            │       ├── Check math formulas
                                            │       └── Generate VERIFICATION_RESULTS.md
                                            ├── 10. Fix any issues found
                                            ├── 11. Re-verify until ALL PASS
                                            └── 12. Upload to Google Drive (subfolder)
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

5. peer-reviewer + peer-review + empirical-paper-analysis-skill
   ├── Multi-model peer review simulation
   ├── Structural analysis of empirical content
   └── Weakness identification and remediation

6. data-analyst + data-analysis + statistics + diagram
   ├── Advanced statistical computation
   ├── Statistical validation
   └── Publication-quality visualizations

7. format-editor + latex + typetex + markdown-converter
   ├── Multi-journal format preparation
   ├── Advanced PDF compilation
   └── Format conversion flexibility
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

Phase 5: Final Integration Check
├── agentarxiv: Novelty and positioning verification
├── Cross-reference all skill outputs
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

## 💰 Cost Optimization

- **Opus** for critical thinking (architect, literature, methodology, peer-review)
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
