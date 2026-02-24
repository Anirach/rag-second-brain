# Writing Team - Book Production Pipeline

## 🚀 FULL AUTO MODE (Default)

One spawn → Complete book. No approvals needed.

```
[Topic + Description] ──► book-architect ──► [Complete 120-200 page Book]
                              │
                              ├── Research (web search)
                              ├── Outline (7-10 chapters)
                              ├── Write all chapters
                              ├── Self-edit for quality
                              ├── Assemble + format
                              └── Upload to Google Drive
```

**To produce a book:**
```
Arthur, write a book about [TOPIC]. Target audience: [WHO]. Focus on: [ANGLE].
```

## Team Roster (Specialized Agents)

| Agent | Role | Model | Workspace | **🆕 Key Skills** |
|-------|------|-------|-----------|-------------------|
| **book-architect** | Full Auto Producer | Opus | `~/.openclaw/workspace-book-architect` | agent-content-pipeline, deep-research-pro, ai-pdf-builder |
| **research-lead** | Deep Research | Opus | `~/.openclaw/workspace-research-lead` | **deep-research-pro**, deep-research, literature-search |
| **chapter-writer** | Chapter Drafts | Sonnet | `~/.openclaw/workspace-chapter-writer` | writing, academic-writing-refiner, copywriting |
| **dev-editor** | Coherence Review | Opus | `~/.openclaw/workspace-dev-editor` | **content-quality-auditor**, writing, style-guide-generator |
| **copy-editor** | Polish & Style | Sonnet | `~/.openclaw/workspace-copy-editor` | **copywriting**, writing, academic-writing-refiner |
| **publisher** | Assembly & Format | Opus | `~/.openclaw/workspace-publisher` | **content-repurposing-engine**, **markdown-converter**, **wordpress-publishing-skill-for-claude**, word-docx, ai-pdf-builder |
| **fact-checker** | Verification | Opus | `~/.openclaw/workspace-fact-checker` | **content-quality-auditor**, deep-research, literature-search |
| **engagement-analyst** | Reader Analytics | Sonnet | `~/.openclaw/workspace-engagement-analyst` | **content-repurposing-engine**, **seo-content-writer**, style-guide-generator |

---

## 🔧 Skills Integration

The Writing Team leverages newly installed skills to maximize content quality and output efficiency. Each agent integrates specific skills into their workflow for enhanced capabilities.

### Skills Integration Matrix

| Agent | Skill | When to Use | How It Helps |
|-------|-------|-------------|--------------|
| **book-architect** | agent-content-pipeline | Multi-stage book production | Orchestrates content workflow from research → writing → editing → publishing |
| **book-architect** | deep-research-pro | Initial research phase | Comprehensive multi-source research with cited findings |
| **book-architect** | ai-pdf-builder | Final assembly | Professional PDF generation with proper formatting |
| **research-lead** | **deep-research-pro** | Primary research workflow | Multi-source web research with DuckDuckGo, synthesized reports |
| **research-lead** | deep-research | Legacy research support | Fallback research capabilities |
| **research-lead** | literature-search | Academic content | Multi-database academic paper search (S2, OA, CR, PM) |
| **chapter-writer** | writing | Core writing | Enhanced prose generation and style |
| **chapter-writer** | copywriting | Persuasive content | Professional copywriting patterns and techniques |
| **chapter-writer** | academic-writing-refiner | Academic style | Scholarly tone and structure refinement |
| **dev-editor** | **content-quality-auditor** | Quality verification | 80-item CORE-EEAT content quality assessment |
| **dev-editor** | style-guide-generator | Consistency checks | Ensure consistent voice and style throughout |
| **copy-editor** | **copywriting** | Final polish | Professional copywriting patterns and optimization |
| **copy-editor** | academic-writing-refiner | Academic refinement | Scholarly writing enhancement |
| **publisher** | **content-repurposing-engine** | Multi-format output | Convert content to multiple formats (blog posts, social, etc.) |
| **publisher** | **markdown-converter** | Format conversion | Convert between markdown, HTML, DOCX seamlessly |
| **publisher** | **wordpress-publishing-skill-for-claude** | Direct publishing | Automated WordPress publishing workflow |
| **publisher** | word-docx | Document generation | Professional DOCX creation with formatting |
| **fact-checker** | **content-quality-auditor** | Systematic verification | EEAT-compliant fact checking with source validation |
| **fact-checker** | literature-search | Source verification | Verify academic claims against peer-reviewed sources |
| **engagement-analyst** | **seo-content-writer** | SEO optimization | Search engine optimization and keyword analysis |
| **engagement-analyst** | **content-repurposing-engine** | Format analysis | Analyze engagement potential across formats |

### Cross-Team Skills Available

The Writing Team can also leverage skills from other teams when needed:

| Skill | From Team | Use Case |
|-------|-----------|----------|
| mermaid-architect | General | Create flowcharts and diagrams for technical books |
| diagram | General | Visual content for complex topics |
| edge-tts | Translation/Course | Create audiobook versions |
| google-slides | Course | Presentation materials for book launches |
| latex | Academic | Technical document formatting |

### Quality Gates Enhanced with Skills

| Stage | Traditional Gate | **🆕 Skill-Enhanced Gate** |
|-------|------------------|---------------------------|
| **Research** | Sources identified | **deep-research-pro**: Multi-source synthesis with citations |
| **Draft** | Complete chapters | **content-quality-auditor**: 80-item quality assessment |
| **Fact Check** | Claims verified | **content-quality-auditor**: EEAT-compliant verification |
| **Copy Edit** | Error-free text | **copywriting**: Professional pattern optimization |
| **Engagement** | Reader-ready | **seo-content-writer**: SEO optimization + **content-repurposing-engine**: Multi-format potential |
| **Publishing** | Format ready | **markdown-converter**: Multi-format conversion + **wordpress-publishing-skill-for-claude**: Direct publishing |

### Workflow Enhancements with Skills

```
📚 ENHANCED BOOK PRODUCTION PIPELINE

1. book-architect + deep-research-pro
   ├── Multi-source research (web + academic)
   ├── Synthesized findings with citations
   └── Research quality baseline established

2. chapter-writer + writing + copywriting
   ├── Enhanced prose generation
   ├── Professional copywriting patterns
   └── Persuasive, engaging content

3. dev-editor + content-quality-auditor
   ├── 80-item CORE-EEAT quality assessment
   ├── Systematic quality scoring
   └── Priority improvement recommendations

4. fact-checker + content-quality-auditor + literature-search
   ├── EEAT-compliant fact verification
   ├── Academic source validation
   └── Source authority assessment

5. copy-editor + copywriting + academic-writing-refiner
   ├── Professional copywriting optimization
   ├── Style consistency enforcement
   └── Academic tone refinement (when needed)

6. engagement-analyst + seo-content-writer + content-repurposing-engine
   ├── SEO optimization analysis
   ├── Multi-format engagement potential
   └── Distribution strategy recommendations

7. publisher + content-repurposing-engine + markdown-converter + wordpress-publishing-skill-for-claude
   ├── Multi-format content generation
   ├── Seamless format conversion
   ├── Direct WordPress publishing
   └── Professional document assembly
```

### Enhanced Command Examples

```bash
# Research-enhanced book creation
"Arthur, create a book about [TOPIC] using deep-research-pro for comprehensive research"

# Quality-audited content
"Arthur, write a book and run content-quality-auditor on each chapter"

# Multi-format publishing
"Arthur, create a book and use content-repurposing-engine to generate blog series"

# Direct publishing workflow
"Arthur, write a book and publish directly to WordPress using wordpress-publishing-skill-for-claude"

# SEO-optimized content
"Arthur, create SEO-optimized book content using seo-content-writer"
```

---

## Modes

### Full Auto (Default)
- **book-architect** handles entire pipeline solo
- One spawn → complete book
- Background execution, ping when done

### Manual/Step-by-Step
- Spawn individual specialists for control
- Review at each stage
- Best for complex/sensitive projects

### 🔄 Human Collaboration Mode (NEW)

Iterative back-and-forth workflow until the book is complete.

```
┌─────────────────────────────────────────────────────────────┐
│              HUMAN COLLABORATION WORKFLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User provides: Topic + Requirements                      │
│         │                                                    │
│         ▼                                                    │
│  2. book-architect creates OUTLINE                          │
│         │                                                    │
│         ▼                                                    │
│  3. 👤 HUMAN REVIEW — Approve/Request Changes               │
│         │                                                    │
│         ├── Changes needed → Revise outline → Step 3        │
│         │                                                    │
│         ▼ Approved                                           │
│  4. research-lead conducts RESEARCH                         │
│         │                                                    │
│         ▼                                                    │
│  5. 👤 HUMAN REVIEW — Check sources, coverage               │
│         │                                                    │
│         ├── Gaps found → More research → Step 5             │
│         │                                                    │
│         ▼ Approved                                           │
│  6. chapter-writer drafts CHAPTERS (one at a time)          │
│         │                                                    │
│         ▼                                                    │
│  7. 👤 HUMAN REVIEW — Per-chapter feedback                  │
│         │                                                    │
│         ├── Revisions needed → Rewrite → Step 7             │
│         │                                                    │
│         ▼ Approved                                           │
│  8. [Repeat 6-7 for all chapters]                           │
│         │                                                    │
│         ▼                                                    │
│  9. dev-editor + copy-editor POLISH                         │
│         │                                                    │
│         ▼                                                    │
│  10. 👤 FINAL REVIEW                                        │
│         │                                                    │
│         ├── More polish needed → Step 9                     │
│         │                                                    │
│         ▼ Approved                                           │
│  11. publisher FORMATS & UPLOADS                            │
│         │                                                    │
│         ▼                                                    │
│  12. ✅ COMPLETE — Book delivered to Google Drive           │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**To start collaborative mode:**
```
Arthur, let's write a book together about [TOPIC].
I want to review and approve each stage.
```

**Feedback commands:**
- `"Approved, continue"` → Move to next stage
- `"Change X to Y"` → Specific revision
- `"More detail on Z"` → Expand section
- `"Cut this part"` → Remove content
- `"Start over on this chapter"` → Full rewrite

**Stage deliverables for review:**
| Stage | What you receive | What to check |
|-------|------------------|---------------|
| Outline | Chapter structure, key points | Flow, coverage, focus |
| Research | Sources, key findings | Quality, relevance |
| Chapter Draft | Full chapter text | Voice, accuracy, engagement |
| Edited Draft | Polished chapter | Clarity, style, errors |
| Final Book | Complete DOCX | Overall quality |

## Output Location

**All books uploaded to Google Drive:** `ArthurBotData/Books/[Book-Title]/`

### Google Drive Structure
```
Books/[Book-Title]/
├── 00-outline.md
├── 01-research/
│   ├── chapter-01-research.md
│   └── ...
├── 02-chapters/
│   ├── 00-front-matter.md
│   ├── 01-[chapter].md
│   └── 99-back-matter.md
├── 03-edits/
│   └── revision-notes.md
└── [Book-Title]-FINAL.docx    ⭐ Complete book
```

### Deliverables
- **Parts**: Individual .md files on Google Drive
- **Final**: Complete .docx on Google Drive
- **Links**: Folder link + direct DOCX link provided

## Target Specs

- **Length**: 120-200 pages
- **Chapters**: 7-10
- **Format**: Markdown → DOCX → Google Drive

## 🎯 Enhanced Quality Gates (Skill-Powered)

Each stage now includes skill-based verification for professional-grade output:

### Stage 1: Research Foundation
**Traditional Gate**: Real sources, properly cited, gaps noted
**🆕 Skill Enhancement**: **deep-research-pro** multi-source synthesis
- ✅ Minimum 15-30 verified sources across web + academic databases
- ✅ Synthesized findings with cross-reference validation
- ✅ Citation quality verified through literature-search
- ✅ Research gaps explicitly documented

### Stage 2: Content Quality Assessment
**Traditional Gate**: Complete chapters, research integrated
**🆕 Skill Enhancement**: **content-quality-auditor** 80-item assessment
- ✅ CORE-EEAT quality score ≥ 75/100 per chapter
- ✅ All 8 dimensions (C-O-R-E + E-E-A-T) evaluated
- ✅ Veto checks passed (T04, C01, R10)
- ✅ Priority improvements identified and addressed

### Stage 3: Fact Verification
**Traditional Gate**: All claims verified, sources valid
**🆕 Skill Enhancement**: **content-quality-auditor** EEAT compliance
- ✅ Level 1-5 fact checking protocol completed
- ✅ Source authority verification (A01-A10)
- ✅ Trust signals verified (T01-T10)
- ✅ Statistical claims double-checked

### Stage 4: Professional Copy Optimization
**Traditional Gate**: Error-free, consistent style
**🆕 Skill Enhancement**: **copywriting** pattern optimization
- ✅ Professional copywriting patterns applied
- ✅ Persuasive structure implemented
- ✅ Engagement hooks optimized
- ✅ Call-to-action effectiveness verified

### Stage 5: Multi-Format Preparation
**Traditional Gate**: Complete book, all parts present, formatted
**🆕 Skill Enhancement**: **content-repurposing-engine** + **markdown-converter**
- ✅ Content optimized for multiple formats
- ✅ Blog series potential identified
- ✅ Social media adaptations prepared
- ✅ Format-specific optimizations applied

### Stage 6: SEO & Distribution Optimization
**Traditional Gate**: Reader-ready content
**🆕 Skill Enhancement**: **seo-content-writer** optimization
- ✅ Keyword optimization completed
- ✅ Meta descriptions optimized
- ✅ Internal linking structure planned
- ✅ Search visibility maximized

### Stage 7: Publication-Ready Assembly
**Traditional Gate**: Formatted and ready
**🆕 Skill Enhancement**: **wordpress-publishing-skill-for-claude** + **ai-pdf-builder**
- ✅ WordPress-ready formatting
- ✅ Professional PDF generation
- ✅ Multi-format export capability
- ✅ Direct publishing workflow tested

### Mandatory Quality Checks

**Before ANY delivery, verify:**
```markdown
## Pre-Delivery Skill Verification Checklist

### Content Quality (content-quality-auditor)
- [ ] Overall CORE-EEAT score ≥ 75/100
- [ ] No veto triggers (T04, C01, R10)
- [ ] All priority improvements addressed

### Research Quality (deep-research-pro)
- [ ] Multi-source research completed
- [ ] Citations properly formatted
- [ ] Source authority verified

### Copy Quality (copywriting)
- [ ] Professional patterns applied
- [ ] Engagement optimized
- [ ] Persuasive structure implemented

### Format Quality (markdown-converter + content-repurposing-engine)
- [ ] Multi-format compatibility verified
- [ ] Conversion quality tested
- [ ] Distribution formats prepared

### SEO Quality (seo-content-writer)
- [ ] Keywords optimized
- [ ] Search visibility maximized
- [ ] Meta content prepared

### Publication Quality (wordpress-publishing-skill-for-claude)
- [ ] Publishing workflow tested
- [ ] Format integrity verified
- [ ] Distribution ready
```

---

## 🔍 Fact-Checker Agent (NEW)

**Purpose:** Verify all factual claims, statistics, quotes, and attributions.

### Verification Levels

| Level | Check | Required For |
|-------|-------|--------------|
| L1 | Source exists | All claims |
| L2 | Claim matches source | All claims |
| L3 | Source is authoritative | Statistics, quotes |
| L4 | Claim is current (not outdated) | Facts that may change |
| L5 | No contradictory evidence | Controversial claims |

### What Gets Checked
- ✅ Statistics and numbers
- ✅ Historical facts and dates
- ✅ Quotes and attributions
- ✅ Scientific claims
- ✅ Brand/product claims
- ✅ Legal/medical information

### Output: `FACT_CHECK_REPORT.md`
```markdown
## Fact Check Report

**Book:** [Title]
**Checked:** [Date]
**Status:** [PASS/FAIL/NEEDS_REVIEW]

### Claims Verified: X/Y

| Chapter | Claim | Source | Status | Notes |
|---------|-------|--------|--------|-------|
| Ch 1 | "50% of..." | [URL] | ✅ Verified | |
| Ch 3 | "In 1985..." | [URL] | ⚠️ Corrected | Was 1986 |
| Ch 5 | "According to..." | NOT FOUND | ❌ Remove | |

### Issues Found
1. [Issue 1]
2. [Issue 2]

### Recommendations
- [Action items]
```

---

## 📊 Engagement Analyst Agent (NEW)

**Purpose:** Analyze reader engagement potential and optimize pacing.

### Metrics Analyzed

| Metric | Target | Description |
|--------|--------|-------------|
| **Hook Score** | 8+/10 | Opening engagement per chapter |
| **Pacing Score** | 7+/10 | Flow and rhythm |
| **Cliffhanger Score** | 7+/10 | Chapter endings |
| **Fatigue Risk** | Low | Dense sections flagged |
| **Engagement Curve** | Rising | Overall arc |

### Analysis Per Chapter
```markdown
## Chapter Engagement Analysis

### Chapter 1: [Title]
- **Hook Score:** 9/10 — Strong opening
- **Pacing:** 8/10 — Good flow
- **Cliffhanger:** 7/10 — Decent ending
- **Fatigue Risk:** Low
- **Suggestions:** None

### Chapter 3: [Title]
- **Hook Score:** 5/10 ⚠️ — Weak opening
- **Pacing:** 6/10 ⚠️ — Dense middle section
- **Cliffhanger:** 8/10 — Strong ending
- **Fatigue Risk:** MEDIUM ⚠️
- **Suggestions:**
  - Add hook in first paragraph
  - Break up paragraphs on pages 45-48
```

### Overall Book Score
```
📊 ENGAGEMENT SCORECARD
├── Overall: 7.8/10
├── Best Chapter: Ch 7 (9.2/10)
├── Needs Work: Ch 3 (6.1/10)
├── Pacing: Good (minor fatigue in middle)
└── Recommendation: Strengthen Ch 3 opening
```

---

## 📝 Version Control System (NEW)

Track all drafts and changes throughout the writing process.

### Version Structure
```
📁 Book_Project/
├── v1/
│   ├── outline_v1.md
│   ├── chapters/
│   │   ├── ch01_v1.md
│   │   └── ...
│   └── VERSION_LOG.md
├── v2/
│   ├── chapters/
│   │   ├── ch01_v2.md (with changes)
│   │   └── ...
│   ├── CHANGES_v1_to_v2.md
│   └── VERSION_LOG.md
└── FINAL/
    └── [Book]-FINAL.docx
```

### Version Log Template
```markdown
# VERSION_LOG.md

## v2 (2026-02-12)
- Ch 3 rewritten for better pacing
- All fact-check issues resolved
- Added new scene in Ch 7

## v1 (2026-02-10)
- Initial complete draft
- 12 chapters, 45,000 words
```

### Rollback Capability
If a revision makes things worse:
1. Compare with previous version
2. Identify regression
3. Restore previous content
4. Document rollback reason

---

## 🎭 Genre Templates (NEW)

Quick-start templates for different book types:

### Fiction
- **Novel** — 80,000-100,000 words, 3-act structure
- **Novella** — 20,000-50,000 words
- **Short Story Collection** — 10-15 stories

### Non-Fiction
- **How-To Guide** — Step-by-step, exercises, examples
- **Biography/Memoir** — Chronological or thematic
- **Business Book** — Frameworks, case studies, action items

### Technical
- **Technical Manual** — Reference style, procedures
- **Textbook** — Learning objectives, assessments
- **Documentation** — API reference, tutorials

To use:
```
Arthur, write a [GENRE] book about [TOPIC].
Template: [fiction-novel/nonfiction-howto/technical-manual]
```

---

## 🖼️ Visual Assets Integration (NEW)

For books that need images, diagrams, or cover art:

### Chapter Headers
- Generate via image tools (DALL-E, Midjourney prompts)
- Consistent style across all chapters
- Store in `assets/chapter-headers/`

### Diagrams
- Mermaid diagrams for flowcharts
- SVG for technical illustrations
- Store in `assets/diagrams/`

### Cover Design
- Generate cover concept prompts
- Multiple variations
- Store in `assets/covers/`

### Asset Manifest
```markdown
# ASSET_MANIFEST.md

## Cover
- cover_v1.png — First concept
- cover_v2.png — Revised (chosen)

## Chapter Headers
- ch01_header.png — "The Beginning"
- ch02_header.png — "Rising Action"

## Diagrams
- fig_3_1.svg — System architecture
- fig_5_2.png — Timeline
```

## Cost Optimization

- Opus for thinking-heavy roles (architect, research, dev-edit, publish)
- Sonnet for execution-heavy roles (writing, copy-edit)
- Parallel chapter writing when possible
- Batch similar operations

---

## 📓 Obsidian Workspace

All Writing Team work is documented in Obsidian:

**Location:** `/home/clawdbot/obsidian-vault/Agents/Writing/`

```
Writing/
├── Projects/           # Active book projects
│   └── [Book-Title].md
├── Notes/              # Research & drafts
│   ├── Research/       # Source materials
│   └── Outlines/       # Chapter outlines
└── Templates/          # Book templates
    └── Book Project.md
```

**For each book, create:**
1. `Projects/[Book-Title].md` — Overview, outline, progress
2. `Notes/Research/[Book-Title]/` — Research materials
3. Link chapters with `[[wikilinks]]`

**Tags:** `#book`, `#writing`, `#research`, `#draft`, `#complete`
