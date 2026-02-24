# Translation Team — Professional Thai Translation Pipeline

## 🎯 Mission

Produce beautiful, natural Thai translations that read as if originally written in Thai. Every translation must be:
- **Natural** — Flows like native Thai prose, not translated text
- **Culturally Adapted** — Idioms, references, and tone fit Thai readers
- **Modern** — Contemporary language, avoiding outdated expressions
- **Professional** — Clean formatting, justified text, proper Thai typography
- **Faithful** — Preserves meaning, intent, and spirit of the original

## 🚀 Full Auto Mode

One spawn → Complete translated document.

```
[Source Document] ──► translation-architect ──► [Professional Thai Document]
                              │
                              ├── Analyze source structure & style
                              ├── Create translation glossary
                              ├── Primary translation (meaning + nuance)
                              ├── Cultural adaptation
                              ├── Thai style polishing
                              ├── Professional formatting
                              ├── Quality review
                              └── Upload to Google Drive
```

**To translate:**
```
Arthur, translate [DOCUMENT/TEXT] to Thai.
Style: [literary/business/academic/casual]
Output: [DOCX/PDF/Markdown]
```

---

## 🔄 Human Collaboration Mode (NEW)

Iterative back-and-forth workflow for translation projects with human review.

```
┌─────────────────────────────────────────────────────────────┐
│              HUMAN COLLABORATION WORKFLOW                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. User provides: Source document + Style preferences       │
│         │                                                    │
│         ▼                                                    │
│  2. translation-architect analyzes source, creates GLOSSARY │
│     ├── Key terms identified                                │
│     ├── Style recommendations                               │
│     └── Translation approach                                │
│         │                                                    │
│         ▼                                                    │
│  3. 👤 HUMAN REVIEW — Approve glossary & approach           │
│         │                                                    │
│         ├── Change terms → Update glossary → Step 3         │
│         │                                                    │
│         ▼ Approved                                           │
│  4. thai-linguist translates SECTION BY SECTION             │
│         │                                                    │
│         ▼                                                    │
│  5. 👤 HUMAN REVIEW — Per-section approval                  │
│         │                                                    │
│         ├── "This doesn't sound right" → Revise → Step 5    │
│         │                                                    │
│         ▼ Approved                                           │
│  6. [Repeat 4-5 for all sections]                           │
│         │                                                    │
│         ▼                                                    │
│  7. cultural-adapter reviews CULTURAL FIT                   │
│         │                                                    │
│         ▼                                                    │
│  8. 👤 HUMAN REVIEW — Cultural adaptations                  │
│         │                                                    │
│         ├── Adjust adaptations → Revise → Step 8            │
│         │                                                    │
│         ▼ Approved                                           │
│  9. style-polisher POLISHES prose                           │
│         │                                                    │
│         ▼                                                    │
│  10. 👤 HUMAN REVIEW — Style and flow                       │
│         │                                                    │
│         ├── More polish → Back to Step 9                    │
│         │                                                    │
│         ▼ Approved                                           │
│  11. back-translator performs QA CHECK                      │
│         │                                                    │
│         ▼                                                    │
│  12. 👤 HUMAN REVIEW — Back-translation report              │
│         │                                                    │
│         ├── Meaning drift found → Fix → Step 12             │
│         │                                                    │
│         ▼ Approved                                           │
│  13. document-formatter FORMATS final document              │
│         │                                                    │
│         ▼                                                    │
│  14. 👤 FINAL REVIEW                                        │
│         │                                                    │
│         ▼ Approved                                           │
│  15. ✅ COMPLETE — Translation delivered                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**To start collaborative mode:**
```
Arthur, let's translate this [document] together.
I want to review each section before finalizing.
Style: [literary/business/academic]
```

**Feedback commands:**
- `"Approved, continue"` → Move to next section
- `"This should be more formal"` → Adjust register
- `"Use X instead of Y"` → Specific term change
- `"This loses the nuance"` → Preserve meaning better
- `"Too literal"` → More natural Thai
- `"Keep the original here"` → Don't translate term

**Stage deliverables for review:**
| Stage | Deliverable | Review Focus |
|-------|-------------|--------------|
| Glossary | `GLOSSARY.md` | Term consistency |
| Section Draft | Per-section translation | Accuracy, flow |
| Cultural Review | Adaptation notes | Appropriateness |
| Polished Draft | Refined translation | Style, readability |
| Back-Translation | `BACK_TRANSLATION.md` | Meaning preservation |
| Final | `.docx` / `.pdf` | Overall quality |

**Parallel review (for long documents):**
```
Human reviews Section 1 → Agent works on Section 2
Human reviews Section 2 → Agent works on Section 3
...continues in parallel for efficiency
```

## 👥 Team Roster (Enhanced with Multimedia Skills)

| Agent | Role | Model | Expertise | **🆕 Key Skills** |
|-------|------|-------|-----------|-------------------|
| **translation-architect** | Project Lead | Opus | Coordination, glossary, consistency | **markdown-converter**, word-docx, writing |
| **thai-linguist** | Primary Translator | Opus | Thai language mastery, natural prose | **writing**, academic-writing-refiner |
| **cultural-adapter** | Localization | Opus | Cultural nuance, idioms, references | writing, style-guide-generator |
| **style-polisher** | Prose Quality | Opus | Literary Thai, modern expression | **writing**, copywriting, style-guide-generator |
| **document-formatter** | Layout & Design | Sonnet | Formatting, typography, professional look | **markdown-converter**, **word-docx**, latex, ai-pdf-builder |
| **quality-reviewer** | Final QA | Opus | Accuracy, consistency, polish | content-quality-auditor, writing |
| **translation-memory** | TM Management | Sonnet | Terminology DB, consistency enforcement | markdown-converter |
| **back-translator** | Accuracy Verify | Opus | Back-translation QA, meaning drift detection | writing, academic-writing-refiner |
| **🆕 audio-producer** | Audio Content | Sonnet | Text-to-speech, audio post-production | **edge-tts**, **video-subtitles** |
| **🆕 multimedia-specialist** | Video & Subtitles | Sonnet | Subtitle timing, video localization | **video-subtitles**, **edge-tts** |

---

## 🎬 Skills Integration for Multimedia Translation

The Translation Team now supports comprehensive multimedia translation with audio, video, and multi-format capabilities.

### Skills Integration Matrix

| Agent | Skill | When to Use | How It Helps |
|-------|-------|-------------|--------------|
| **translation-architect** | **markdown-converter** | Multi-format projects | Convert between markdown, HTML, DOCX seamlessly across languages |
| **translation-architect** | word-docx | Document projects | Professional DOCX generation with Thai typography |
| **translation-architect** | writing | Quality coordination | Enhanced writing coordination across translation stages |
| **thai-linguist** | **writing** | Core translation | Enhanced prose generation for natural Thai expression |
| **thai-linguist** | academic-writing-refiner | Academic translations | Scholarly Thai tone and academic structure |
| **cultural-adapter** | writing | Cultural adaptation | Enhanced cultural adaptation with writing patterns |
| **style-polisher** | **writing** | Final polish | Advanced Thai prose refinement and style optimization |
| **style-polisher** | copywriting | Marketing content | Professional copywriting patterns adapted for Thai |
| **document-formatter** | **markdown-converter** | Format conversion | Convert translated content across all major formats |
| **document-formatter** | **word-docx** | Document assembly | Professional Thai document formatting and layout |
| **document-formatter** | latex | Technical documents | LaTeX formatting for academic/technical Thai content |
| **🆕 audio-producer** | **edge-tts** | Audio content creation | Generate natural Thai speech from translated text |
| **🆕 multimedia-specialist** | **video-subtitles** | Video localization | Create Thai subtitles with proper timing and formatting |
| **🆕 multimedia-specialist** | **edge-tts** | Dubbing preparation | Generate Thai audio for dubbing scripts |

### New Multimedia Capabilities

| Content Type | Traditional Output | **🆕 Enhanced Output** |
|--------------|-------------------|------------------------|
| **Text Translation** | DOCX/PDF | + Markdown, HTML, LaTeX via **markdown-converter** |
| **Document Translation** | Formatted document | + Multiple formats + **word-docx** professional layout |
| **Book Translation** | Text only | + **edge-tts** audiobook version |
| **Video Content** | Subtitles only | + **video-subtitles** + **edge-tts** dubbing audio |
| **Academic Papers** | Thai translation | + **latex** academic formatting + audio abstract |
| **Marketing Content** | Translated copy | + **copywriting** optimization + multi-format output |

### Enhanced Translation Workflows

#### 1. Text-to-Audio Translation Pipeline
```
Source Text → thai-linguist (writing) → style-polisher (writing + copywriting)
    ↓
audio-producer (edge-tts) → Thai Audio Output
    ↓
Quality Check → Final Audio + Text Deliverables
```

#### 2. Video Localization Pipeline
```
Source Video + Script → thai-linguist → cultural-adapter
    ↓
multimedia-specialist (video-subtitles) → Thai Subtitles (SRT/VTT)
    ↓
audio-producer (edge-tts) → Thai Dubbing Audio
    ↓
Synchronized Output → Video + Subtitles + Dubbing Track
```

#### 3. Multi-Format Document Pipeline
```
Source Document → translation-architect (markdown-converter)
    ↓
thai-linguist (writing) → cultural-adapter → style-polisher
    ↓
document-formatter (markdown-converter + word-docx + latex)
    ↓
Multiple Format Output (DOCX, PDF, HTML, Markdown, LaTeX)
```

#### 4. Academic Translation Pipeline
```
Academic Paper → thai-linguist (academic-writing-refiner)
    ↓
document-formatter (latex) → Professional Academic Format
    ↓
audio-producer (edge-tts) → Thai Abstract Audio
    ↓
Complete Academic Package (PDF + Audio + Multiple Formats)
```

### Cross-Team Skills Integration

| Skill Source | Skill | Translation Use Case |
|--------------|-------|---------------------|
| Writing Team | content-quality-auditor | Quality assessment of translated content |
| Academic Team | literature-search | Verify translations of academic terminology |
| Course Team | google-slides | Translated presentation materials |
| Course Team | curriculum-generator | Educational content translation |
| General | mermaid-architect | Translate technical diagrams and flowcharts |

### Enhanced Command Examples

```bash
# Audio-enhanced translation
"Arthur, translate this document to Thai and create audio version using edge-tts"

# Video localization
"Arthur, translate this video using video-subtitles for Thai subtitles and edge-tts for dubbing"

# Multi-format translation
"Arthur, translate to Thai and output in multiple formats using markdown-converter"

# Academic translation
"Arthur, translate this paper to Thai using academic-writing-refiner and latex formatting"

# Marketing translation
"Arthur, translate this marketing content using copywriting optimization for Thai market"
```

### Audio Production Capabilities (edge-tts)

#### Supported Features
- **Natural Thai Speech**: High-quality text-to-speech in Thai
- **Multiple Voices**: Different speaker options for variety
- **Emotion Control**: Adjust tone and emotion for context
- **Speed Control**: Adjust reading speed for different audiences
- **Audio Formats**: MP3, WAV for different use cases

#### Use Cases
- **Audiobooks**: Convert translated books to Thai audio
- **Educational Content**: Audio versions of course materials
- **Marketing**: Voice-over for Thai advertisements
- **Accessibility**: Audio for visually impaired users

### Video Subtitle Capabilities (video-subtitles)

#### Subtitle Features
- **Timing Synchronization**: Precise timing alignment
- **Format Support**: SRT, VTT, ASS subtitle formats
- **Length Optimization**: Respect reading speed limits
- **Cultural Adaptation**: Adjust subtitle style for Thai viewers

#### Technical Specifications
- **Maximum Characters**: 42 per line, 2 lines max
- **Display Duration**: 1-7 seconds based on reading speed
- **Reading Speed**: 21 characters/second for Thai
- **Line Breaking**: Natural Thai phrase boundaries

---

## 📝 Translation Philosophy

### Core Principles

1. **Meaning Over Words**
   - Translate concepts, not just words
   - Preserve intent and emotional impact
   - "What would a Thai author write?" not "How do I translate this?"

2. **Natural Thai Flow**
   - Sentence structure should feel Thai, not English
   - Use Thai paragraph rhythm
   - Appropriate connectors (แต่, อย่างไรก็ตาม, ทั้งนี้)

3. **Register Awareness**
   - Match formality level to context
   - Academic → formal Thai (ภาษาเขียน)
   - Dialogue → natural speech patterns
   - Business → professional but accessible

4. **Cultural Bridge**
   - Adapt references Thai readers understand
   - Explain foreign concepts when needed
   - Preserve author's voice while being accessible

## 🎨 Thai Writing Style Guide

### Typography & Formatting

```
✓ Justified alignment (ชิดขอบทั้งสองด้าน)
✓ Proper Thai quotation marks: "..." or «...»
✓ Thai numerals when appropriate: ๑, ๒, ๓ (formal) vs 1, 2, 3 (modern)
✓ Proper spacing around Thai punctuation
✓ Elegant line breaks (avoid orphaned particles)
```

### Modern Professional Thai

```
✓ Use contemporary vocabulary
✓ Avoid archaic royal language (unless contextually appropriate)
✓ Balance formal and accessible
✓ Clear, direct sentences (avoid overly complex constructions)
✓ Active voice preferred
```

### Document Structure

```
✓ Thai-style headings (concise, descriptive)
✓ Appropriate section breaks
✓ Proper page margins for Thai text
✓ Font: modern Thai fonts (Sarabun, Prompt, IBM Plex Thai)
✓ Line height: 1.5-1.8 for readability
```

## 📋 Quality Standards

### Translation Accuracy
- ✅ All meaning preserved
- ✅ No omissions or additions
- ✅ Technical terms verified
- ✅ Names/titles handled consistently
- ✅ Numbers and dates correct

### Thai Language Quality
- ✅ Natural word order
- ✅ Correct particles (ครับ/ค่ะ, นะ, เถอะ)
- ✅ Appropriate pronouns (เขา/เธอ/ท่าน)
- ✅ Consistent terminology
- ✅ No Anglicisms where Thai words exist

### Formatting Standards
- ✅ Justified text alignment
- ✅ Consistent heading styles
- ✅ Proper indentation
- ✅ Clean page breaks
- ✅ Professional margins

## 🔄 Translation Workflow

### Phase 1: Analysis (translation-architect)
```
□ Understand source document fully
□ Identify style and register
□ Create terminology glossary
□ Note cultural adaptation needs
□ Set consistency guidelines
```

### Phase 2: Primary Translation (thai-linguist)
```
□ Translate for meaning, not words
□ Maintain author's voice
□ Flag uncertain passages
□ Use glossary consistently
```

### Phase 3: Cultural Adaptation (cultural-adapter)
```
□ Adapt idioms and expressions
□ Localize references
□ Ensure cultural appropriateness
□ Add context where needed
```

### Phase 4: Style Polish (style-polisher)
```
□ Enhance Thai prose quality
□ Improve flow and rhythm
□ Modernize language
□ Ensure readability
```

### Phase 5: Formatting (document-formatter)
```
□ Apply professional layout
□ Justify text
□ Set proper typography
□ Format headings/lists
□ Handle page breaks
```

### Phase 6: Quality Review (quality-reviewer)
```
□ Compare with source
□ Check consistency
□ Verify terminology
□ Final proofreading
□ Approve for delivery
```

## 📁 Output Formats

### DOCX (Default)
- Professional Thai document
- Justified alignment
- Modern fonts (Sarabun/TH Sarabun New)
- Clean formatting

### PDF
- Print-ready
- Embedded Thai fonts
- Professional layout

### Markdown
- For web/GitHub
- Preserves structure
- Easy to edit

## 📊 Glossary Management

Each translation project maintains a glossary:

```markdown
## Project Glossary

| English | Thai | Notes |
|---------|------|-------|
| Machine Learning | การเรียนรู้ของเครื่อง | Use consistently |
| Artificial Intelligence | ปัญญาประดิษฐ์ | Abbreviated: AI/เอไอ |
| Algorithm | อัลกอริทึม | Transliteration preferred |
```

## 📁 Output Location

**Google Drive:** `ArthurBotData/Translations/[Project-Name]/`

```
Translations/[Project-Name]/
├── 00-source/
│   └── original-document.md
├── 01-glossary.md
├── 02-drafts/
│   ├── v1-raw-translation.md
│   ├── v2-cultural-adapted.md
│   └── v3-polished.md
├── 03-qa/
│   ├── BACK_TRANSLATION.md        ⭐ NEW
│   └── QA_REPORT.md               ⭐ NEW
├── [Project-Name]-FINAL.docx    ⭐
└── [Project-Name]-FINAL.pdf
```

---

## 🧠 Translation Memory Agent (NEW)

**Purpose:** Maintain terminology databases and ensure consistency across projects.

### Translation Memory Database
```markdown
## Translation Memory — Project: [Name]

### Core Terms (Locked)
| English | Thai | Context | Notes |
|---------|------|---------|-------|
| Machine Learning | การเรียนรู้ของเครื่อง | Technical | Standard term |
| Neural Network | โครงข่ายประสาทเทียม | Technical | Preferred over alternatives |
| Algorithm | อัลกอริทึม | Technical | Transliteration |

### Character Names (Locked)
| Original | Thai | Pronunciation |
|----------|------|---------------|
| John | จอห์น | John |
| Sarah | ซาร่าห์ | Sa-ra |

### Phrases (Reusable)
| English | Thai | Usage Count |
|---------|------|-------------|
| "In conclusion" | "สรุปได้ว่า" | 5 |
| "According to" | "ตามที่" | 12 |
```

### Features

| Feature | Description |
|---------|-------------|
| **Auto-Suggest** | Suggests translations from memory |
| **Consistency Check** | Flags inconsistent translations |
| **Cross-Project** | Reuse terms from previous projects |
| **Domain-Specific** | Separate memories per domain |
| **Export/Import** | TMX format compatible |

### Memory Types
- **Global Memory** — Shared across all projects
- **Project Memory** — Project-specific terms
- **Domain Memory** — Technical, legal, medical, etc.

### Consistency Report
```markdown
## Consistency Report

### Inconsistencies Found: 3

| Term | Occurrences | Translations Used |
|------|-------------|-------------------|
| "framework" | 5 | "กรอบงาน" (3), "เฟรมเวิร์ค" (2) ⚠️ |
| "deploy" | 4 | "ติดตั้ง" (4) ✅ |

### Recommendation
Standardize "framework" → "เฟรมเวิร์ก" (modern, widely used)
```

---

## ↩️ Back-Translator Agent (NEW)

**Purpose:** Verify translation accuracy by translating back to source language.

### Back-Translation Process
```
Source (English) ──► Thai Translation ──► Back to English
        │                                        │
        └────────────── Compare ─────────────────┘
                            │
                      Identify drift
```

### Meaning Drift Detection

| Level | Description | Action |
|-------|-------------|--------|
| **None** | Meanings match perfectly | ✅ Approved |
| **Minor** | Stylistic differences only | ✅ Acceptable |
| **Moderate** | Some nuance lost | ⚠️ Review |
| **Significant** | Meaning changed | ❌ Revise |
| **Critical** | Opposite meaning | 🚨 Urgent fix |

### Back-Translation Report
```markdown
## Back-Translation QA Report

**Project:** Three Old Men
**Chapter:** 3
**Status:** ✅ PASS (Minor drift only)

### Sample Comparisons

#### Passage 1
- **Original:** "He walked slowly toward the door."
- **Thai:** "เขาเดินอย่างเชื่องช้าไปยังประตู"
- **Back:** "He walked leisurely toward the door."
- **Drift:** Minor ✅ (slowly → leisurely)

#### Passage 2
- **Original:** "She had never felt this way before."
- **Thai:** "เธอไม่เคยรู้สึกแบบนี้มาก่อน"
- **Back:** "She had never felt this way before."
- **Drift:** None ✅

#### Passage 3 ⚠️
- **Original:** "The irony was lost on him."
- **Thai:** "เขาไม่เข้าใจความย้อนแย้ง"
- **Back:** "He did not understand the contradiction."
- **Drift:** Moderate ⚠️ (irony → contradiction)
- **Recommendation:** Consider "ความประชดประชันนั้นเขาไม่เข้าใจ"
```

### When to Use
- Complex literary passages
- Nuanced emotional content
- Technical precision required
- Legal/medical accuracy critical

---

## 🌐 Multi-Language Support (NEW)

While Thai is the primary focus, the team can expand to other languages.

### Currently Supported
- **Thai** — Full support (8 agents)
- **English** — Source language

### Expansion Ready
To add a new language pair:

```
1. Clone thai-linguist → [lang]-linguist
2. Create language-specific style guide
3. Build initial translation memory
4. Configure cultural-adapter for target culture
```

### Language Pairs (Roadmap)
| Language | Status | Notes |
|----------|--------|-------|
| Thai ↔ English | ✅ Active | Full team |
| Chinese ↔ English | 📋 Planned | Simplified & Traditional |
| Japanese ↔ English | 📋 Planned | Kanji/Hiragana support |
| Korean ↔ English | 📋 Planned | Hangul support |

---

## 🎬 Audio/Video Transcription (NEW)

Support for multimedia localization.

### Capabilities
| Media Type | Support |
|------------|---------|
| **Subtitles** | SRT, VTT generation |
| **Dubbing Scripts** | Lip-sync friendly |
| **Transcription** | Audio → Text → Translation |
| **Timing** | Timestamp synchronization |

### Subtitle Workflow
```
Audio/Video ──► Transcription ──► Translation ──► Timing Sync ──► SRT/VTT
                                       │
                                 cultural-adapter
                                 (subtitle length limits)
```

### Subtitle Constraints
- Max 2 lines per subtitle
- Max 42 characters per line
- 1-7 seconds display time
- Reading speed: 21 chars/second

---

## 📚 Domain-Specific Expertise (NEW)

Specialized translation for different domains.

### Legal Translation
- Contract terminology
- Legal Thai precision
- Formal register
- Disclaimer accuracy

### Medical Translation
- Medical terminology
- Drug names (transliteration)
- Patient communication
- Clinical precision

### Technical Translation
- Software UI strings
- API documentation
- Technical manuals
- Industry jargon

### Literary Translation
- Voice preservation
- Rhythm and poetry
- Cultural adaptation
- Reader experience focus

---

## 📈 Quality Metrics (NEW)

Track translation quality over time.

### Metrics Dashboard
```markdown
## Translation Quality Metrics

### Project: [Name]

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Accuracy** | 95% | 95% | ✅ |
| **Fluency** | 92% | 90% | ✅ |
| **Consistency** | 88% | 90% | ⚠️ |
| **Cultural Fit** | 94% | 90% | ✅ |

### Improvement Trend
- v1: 85% overall
- v2: 91% overall
- v3: 94% overall ✅

### Issues by Category
- Terminology: 3 issues fixed
- Grammar: 2 issues fixed
- Cultural: 1 issue pending
```

### Continuous Improvement
- Track error patterns
- Update translation memory
- Refine style guide
- Document lessons learned

## 💡 Style-Specific Guidelines

### Literary Translation
- Preserve literary devices
- Maintain poetic rhythm
- Creative adaptation allowed
- Voice and tone paramount

### Business/Corporate
- Professional register
- Clear and direct
- Industry terminology
- Formal but modern

### Academic/Technical
- Precise terminology
- Formal language
- Citation preservation
- Technical accuracy

### Casual/Marketing
- Friendly tone
- Contemporary slang OK
- Punchy and engaging
- Cultural resonance

## 💰 Cost Optimization

- **Opus** for critical thinking (architect, linguist, cultural, style, QA)
- **Sonnet** for execution (formatting)
- Batch similar passages
- Reuse glossaries across related projects

---

*"การแปลที่ดีคือการทำให้ผู้อ่านลืมว่ากำลังอ่านงานแปล"*
*"A good translation makes readers forget they're reading a translation."*

---

## 📓 Obsidian Workspace

All Translation Team work is documented in Obsidian:

**Location:** `/home/clawdbot/obsidian-vault/Agents/Translation/`

```
Translation/
├── Projects/           # Active translations
│   └── [Project-Name].md
├── Notes/              # Glossaries & style
│   ├── Glossaries/     # Term glossaries
│   └── Style/          # Style guides
└── Templates/          # Translation templates
```

**For each translation, create:**
1. `Projects/[Project-Name].md` — Overview, progress, glossary link
2. `Notes/Glossaries/[Project-Name].md` — Project-specific terms
3. Link source/target with `[[wikilinks]]`

**Tags:** `#translation`, `#thai`, `#glossary`, `#style`, `#complete`
