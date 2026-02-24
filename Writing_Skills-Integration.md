---
title: "Writing Team Skills Integration"
tags: ["writing", "skills", "integration", "workflow"]
created: 2026-02-14
updated: 2026-02-24
---

# Writing Team Skills Integration

## Overview

The Writing Team leverages advanced skills to maximize content quality and output efficiency through comprehensive integration with newly installed capabilities.

## Skills Integration Matrix

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

## Enhanced Workflow Summary

### 🆕 Skill-Powered Book Production Pipeline

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

## Quick Reference Commands

### Research-Enhanced Book Creation
```bash
"Arthur, create a book about [TOPIC] using deep-research-pro for comprehensive research"
```

### Quality-Audited Content
```bash
"Arthur, write a book and run content-quality-auditor on each chapter"
```

### Multi-Format Publishing
```bash
"Arthur, create a book and use content-repurposing-engine to generate blog series"
```

### Direct Publishing Workflow
```bash
"Arthur, write a book and publish directly to WordPress using wordpress-publishing-skill-for-claude"
```

### SEO-Optimized Content
```bash
"Arthur, create SEO-optimized book content using seo-content-writer"
```

## Cross-Team Skills Available

| Skill | From Team | Use Case |
|-------|-----------|----------|
| mermaid-architect | General | Create flowcharts and diagrams for technical books |
| diagram | General | Visual content for complex topics |
| edge-tts | Translation/Course | Create audiobook versions |
| google-slides | Course | Presentation materials for book launches |
| latex | Academic | Technical document formatting |

## Quality Gates Enhanced with Skills

| Stage | Traditional Gate | **🆕 Skill-Enhanced Gate** |
|-------|------------------|---------------------------|
| **Research** | Sources identified | **deep-research-pro**: Multi-source synthesis with citations |
| **Draft** | Complete chapters | **content-quality-auditor**: 80-item quality assessment |
| **Fact Check** | Claims verified | **content-quality-auditor**: EEAT-compliant verification |
| **Copy Edit** | Error-free text | **copywriting**: Professional pattern optimization |
| **Engagement** | Reader-ready | **seo-content-writer**: SEO optimization + **content-repurposing-engine**: Multi-format potential |
| **Publishing** | Format ready | **markdown-converter**: Multi-format conversion + **wordpress-publishing-skill-for-claude**: Direct publishing |

## Links

- [[Writing_README|Writing Team README]]
- [[Shared_Available-Skills|Available Skills]]
- [[Shared_Cross-Team-Skills|Cross-Team Skills]]

---
*Last updated: February 14, 2026*
---
## 🆕 Updates (2026-02-24)
**New skills available:**
- `powerpoint-pptx` — Full-featured PPTX generation (replaces basic pptx)
- `pptx-pdf-font-fix` — Fix font issues in PPTX→PDF conversion
- `ai-ppt-generator` — AI-driven slide generation
- `deep-research-pro` — Already installed, confirmed active

**Removed:**
- ~~`pptx`~~ → use `powerpoint-pptx`
- ~~`openai-tts`~~ → use `edge-tts` (free, no API cost)
