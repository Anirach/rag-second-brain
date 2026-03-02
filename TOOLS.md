# TOOLS.md - Local Notes

Skills define *how* tools work. This file is for *your* specifics — the stuff that's unique to your setup.

## Nano Banana Pro (Google AI Image Generation)

**API Key:** `AIzaSyC1BLzV7El8nNV5hqeCdo4R32Cd2HydyNk`
**Model:** `gemini-2.0-flash-exp-image-generation`

**Usage:**
```bash
python3 /home/clawdbot/clawd/nano_banana_google.py "your prompt here" /home/clawdbot/clawd/tmp/output.png
```

**⚠️ ALWAYS upload generated images to Google Drive `Generated-Images` folder:**
```bash
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py /home/clawdbot/clawd/tmp/output.png --folder "Generated-Images" "descriptive_name.png"
```

## DOCX Report Template (DEFAULT)

**When Anirach asks to generate a DOCX report, ALWAYS use this template as base:**
```
/home/clawdbot/clawd/templates/reports/General_Report_Template.docx
```

**Features:** Cover page (title, subtitle, author, date, version), Table of Contents, 3 heading levels (H1: 16pt Navy, H2: 13pt Blue, H3: 12pt Gray), body text, tables, footnotes, 1-inch margins.

**Usage with python-docx:**
```python
from docx import Document
doc = Document('/home/clawdbot/clawd/templates/reports/General_Report_Template.docx')
# Clear sample content, keep styles
# Write new content using the template's styles
doc.save('output.docx')
```

## Research Intelligence Reports - PREFERRED TOOL

**ALWAYS USE FOR ANIRACH'S RESEARCH REPORTS:**
```bash
python3 skills/professional-docx-generator/scripts/clean_table_docx_generator.py "Report Title" config.json output.docx
```

**Key Requirements:**
- Specific article URLs (not category pages)
- Clean "View Source" links in tables (no URLs shown)
- Professional formatting with proper page numbers and TOC
- Full references section with complete citations

## Image Generation Options

| Tool | API | Best For |
|------|-----|----------|
| **Nano Banana Pro** | Google AI | Fast, good quality, free tier |
| **DALL-E 3** | OpenAI | High quality, detailed prompts |
| **openai-image-gen skill** | OpenAI | Batch generation with gallery |

## Web Search

- **Provider:** Perplexity Sonar Pro via OpenRouter
- **API Key:** Configured in ClawdBot config

## Other Services

### Google Places
- **API Key:** `AIzaSyAXtg3wHGZzIzh6Q0nUcaUvn12GlWsukEs`

## Google Drive

**Folder:** `ArthurBotData`
**Credentials:** `/home/clawdbot/clawd/gdrive/credentials.json`
**Token:** `/home/clawdbot/clawd/gdrive/token.json`

**Upload a file:**
```bash
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py /path/to/file.docx
```

**List files:**
```bash
python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py --list
```

---

## HA Database (Company PostgreSQL)

**Alias:** HA (Healthcare Accreditation database)
**⚠️ Credentials stored ONLY in `tools/ha_query.py` — never in context files.**

**Access method (ONLY way):**
```bash
python3 tools/ha_query.py "SELECT ..."
python3 tools/ha_query.py "SELECT ..." --save-mapping   # for reports
python3 tools/ha_query.py "SELECT COUNT(*) ..." --raw-aggregate  # pure aggregates
```

**NEVER run raw psql.** No credentials are available outside the wrapper.

**Key tables (80 total):**
- `hai_employee_gofive_empeo` — HA staff (181)
- `View_WH_Surveyor` — Surveyors (591)
- `View_WH_Speaker` — Speakers (104)
- `View_WH_Manpower` — Manpower records (120k+)
- `accreditationHistory` — Hospital accreditation (113 cols)
- `bi_kpi*` — KPI data
- `pcu_*` — PCU data (buildings, equipment, vehicles)

---

## NEW: Document Conversion (LibreOffice)

**Version:** 24.2.7.2
**Best for:** DOCX → PDF with proper Thai fonts

```bash
# Convert to PDF
libreoffice --headless --convert-to pdf "document.docx"

# Convert to specific directory
libreoffice --headless --convert-to pdf "input.docx" --outdir /output/
```

---

## NEW: Browser Automation (Playwright)

**Browser:** Chromium Headless v145
**Best for:** Screenshots, web scraping, form automation

```javascript
// In Node.js
const { chromium } = require('playwright');
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage();
await page.goto('https://example.com');
await page.screenshot({ path: 'screenshot.png' });
await browser.close();
```

---

## NEW: Vector Database (ChromaDB)

**Location:** `~/.openclaw/chroma_db/`
**Helper:** `/home/clawdbot/clawd/tools/chroma_helper.py`

```bash
# Activate venv first!
source /home/clawdbot/clawd/.venv/bin/activate

# Index memory files
python3 tools/chroma_helper.py index

# Semantic search
python3 tools/chroma_helper.py search "your query"

# Check status
python3 tools/chroma_helper.py status
```

---

## NEW: Research Paper Search (Semantic Scholar)

**API:** Free tier, no key needed
**Helper:** `/home/clawdbot/clawd/tools/semantic_scholar.py`

```bash
# Search papers
python3 tools/semantic_scholar.py "longevity interventions"
```

```python
# In Python
from tools.semantic_scholar import search_papers
papers = search_papers("machine learning", limit=10)
```

---

## Virtual Environment

For ChromaDB and Python tools:
```bash
source /home/clawdbot/clawd/.venv/bin/activate
# ... run python scripts ...
deactivate
```

---

Add whatever helps you do your job. This is your cheat sheet.
