# 📄 Document Templates

All reusable templates for document generation.

## Structure

```
templates/
├── README.md                          ← This file
├── reports/
│   └── General_Report_Template.docx   ← DEFAULT for all DOCX reports
└── academic/
    └── springer-lncs/                 ← AIiH 2026 & Springer conferences
        ├── latex/
        │   ├── llncs.cls              ← LNCS document class
        │   ├── splncs04.bst           ← Bibliography style
        │   ├── samplepaper.tex        ← Sample paper
        │   ├── fig1.eps               ← Sample figure
        │   └── llncsdoc.pdf           ← Full documentation
        └── word/
            ├── svlnproc1104.dot       ← Word template
            └── SPLNPROC_...pdf        ← Word formatting instructions
```

## Usage Guide

### General Reports (DOCX)
Use for: AI news digests, cost reports, research summaries, any non-academic document.

```python
from docx import Document
doc = Document('/home/clawdbot/clawd/templates/reports/General_Report_Template.docx')
# Clear sample content, keep styles
# Write using template styles: Heading 1, Heading 2, Heading 3, Normal
doc.save('output.docx')
```

**Features:** Cover page (title, subtitle, author, date, version), Table of Contents, 3 heading levels (H1: 16pt Navy, H2: 13pt Blue, H3: 12pt Gray), body text, tables, footnotes, 1-inch margins.

### Academic Papers — Springer LNCS (LaTeX)
Use for: AIiH 2026, any Springer LNCS conference.

```latex
\documentclass[runningheads]{llncs}
\bibliographystyle{splncs04}
```

Copy `llncs.cls` and `splncs04.bst` into your paper directory.

**Specs:** A4, 12+2 pages max, double-blind, Times Roman 10pt body.

### Academic Papers — Springer LNCS (Word)
Use for: Word submissions to Springer LNCS conferences.

Open `svlnproc1104.dot` as a new document template in Word.

## Adding New Templates

1. Create a folder under the appropriate category
2. Include a sample/example file
3. Update this README
4. Update `Quick-Reference.md` in Obsidian vault
