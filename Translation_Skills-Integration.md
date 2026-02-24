---
title: "Translation Team Skills Integration"
tags: ["translation", "skills", "multimedia", "thai", "localization"]
created: 2026-02-14
updated: 2026-02-24
---

# Translation Team Skills Integration

## Overview

The Translation Team now supports comprehensive multimedia translation with audio, video, and multi-format capabilities through advanced skills integration.

## Skills Integration Matrix

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

## New Multimedia Capabilities

| Content Type | Traditional Output | **🆕 Enhanced Output** |
|--------------|-------------------|------------------------|
| **Text Translation** | DOCX/PDF | + Markdown, HTML, LaTeX via **markdown-converter** |
| **Document Translation** | Formatted document | + Multiple formats + **word-docx** professional layout |
| **Book Translation** | Text only | + **edge-tts** audiobook version |
| **Video Content** | Subtitles only | + **video-subtitles** + **edge-tts** dubbing audio |
| **Academic Papers** | Thai translation | + **latex** academic formatting + audio abstract |
| **Marketing Content** | Translated copy | + **copywriting** optimization + multi-format output |

## Enhanced Translation Workflows

### 1. Text-to-Audio Translation Pipeline
```
Source Text → thai-linguist (writing) → style-polisher (writing + copywriting)
    ↓
audio-producer (edge-tts) → Thai Audio Output
    ↓
Quality Check → Final Audio + Text Deliverables
```

### 2. Video Localization Pipeline
```
Source Video + Script → thai-linguist → cultural-adapter
    ↓
multimedia-specialist (video-subtitles) → Thai Subtitles (SRT/VTT)
    ↓
audio-producer (edge-tts) → Thai Dubbing Audio
    ↓
Synchronized Output → Video + Subtitles + Dubbing Track
```

### 3. Multi-Format Document Pipeline
```
Source Document → translation-architect (markdown-converter)
    ↓
thai-linguist (writing) → cultural-adapter → style-polisher
    ↓
document-formatter (markdown-converter + word-docx + latex)
    ↓
Multiple Format Output (DOCX, PDF, HTML, Markdown, LaTeX)
```

### 4. Academic Translation Pipeline
```
Academic Paper → thai-linguist (academic-writing-refiner)
    ↓
document-formatter (latex) → Professional Academic Format
    ↓
audio-producer (edge-tts) → Thai Abstract Audio
    ↓
Complete Academic Package (PDF + Audio + Multiple Formats)
```

## Enhanced Command Examples

### Audio-Enhanced Translation
```bash
"Arthur, translate this document to Thai and create audio version using edge-tts"
```

### Video Localization
```bash
"Arthur, translate this video using video-subtitles for Thai subtitles and edge-tts for dubbing"
```

### Multi-Format Translation
```bash
"Arthur, translate to Thai and output in multiple formats using markdown-converter"
```

### Academic Translation
```bash
"Arthur, translate this paper to Thai using academic-writing-refiner and latex formatting"
```

### Marketing Translation
```bash
"Arthur, translate this marketing content using copywriting optimization for Thai market"
```

## Audio Production Capabilities (edge-tts)

### Supported Features
- **Natural Thai Speech**: High-quality text-to-speech in Thai
- **Multiple Voices**: Different speaker options for variety
- **Emotion Control**: Adjust tone and emotion for context
- **Speed Control**: Adjust reading speed for different audiences
- **Audio Formats**: MP3, WAV for different use cases

### Use Cases
- **Audiobooks**: Convert translated books to Thai audio
- **Educational Content**: Audio versions of course materials
- **Marketing**: Voice-over for Thai advertisements
- **Accessibility**: Audio for visually impaired users

## Video Subtitle Capabilities (video-subtitles)

### Subtitle Features
- **Timing Synchronization**: Precise timing alignment
- **Format Support**: SRT, VTT, ASS subtitle formats
- **Length Optimization**: Respect reading speed limits
- **Cultural Adaptation**: Adjust subtitle style for Thai viewers

### Technical Specifications
- **Maximum Characters**: 42 per line, 2 lines max
- **Display Duration**: 1-7 seconds based on reading speed
- **Reading Speed**: 21 characters/second for Thai
- **Line Breaking**: Natural Thai phrase boundaries

## Cross-Team Skills Available

| Skill | Source | Translation Use Case |
|-------|--------|---------------------|
| content-quality-auditor | Writing | Quality assessment of translated content |
| literature-search | Academic | Verify translations of academic terminology |
| google-slides | Course | Translated presentation materials |
| curriculum-generator | Course | Educational content translation |
| mermaid-architect | General | Translate technical diagrams and flowcharts |

## Quick Reference Commands

### Traditional Translation
```bash
"Arthur, translate [DOCUMENT] to Thai. Style: [literary/business/academic/casual]"
```

### Enhanced Audio Translation
```bash
"Arthur, translate to Thai and create audiobook using edge-tts"
```

### Video Translation
```bash
"Arthur, localize this video with Thai subtitles and dubbing track"
```

### Multi-Format Academic
```bash
"Arthur, translate academic paper with latex formatting and audio abstract"
```

### Marketing Optimization
```bash
"Arthur, translate marketing content with copywriting enhancement for Thai market"
```

## Quality Standards Enhanced with Skills

### Translation Accuracy (Skill-Enhanced)
- ✅ **writing**: Enhanced prose generation for natural Thai expression
- ✅ **academic-writing-refiner**: Scholarly tone and structure refinement
- ✅ **copywriting**: Professional copywriting patterns adapted for Thai

### Format Quality (Skill-Enhanced)
- ✅ **markdown-converter**: Multi-format compatibility and conversion
- ✅ **word-docx**: Professional Thai document formatting and layout
- ✅ **latex**: Academic/technical Thai content formatting

### Multimedia Quality (Skill-Enhanced)
- ✅ **edge-tts**: Natural Thai speech generation with emotion control
- ✅ **video-subtitles**: Professional subtitle timing and formatting
- ✅ Synchronized audio and video output for complete localization

## Links

- [[Translation_README|Translation Team README]]
- [[Shared_Available-Skills|Available Skills]]
- [[Shared_Cross-Team-Skills|Cross-Team Skills]]

---
*Last updated: February 14, 2026*
---
## 🆕 Updates (2026-02-24)
**New skills available:**
- `pdf-to-structured` — Extract structured data from PDFs
- `telegram-history` — Access Telegram message history

**Removed:**
- ~~`openai-tts`~~ → use `edge-tts`
