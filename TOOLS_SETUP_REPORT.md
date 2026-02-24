# System Enhancement Setup Report
**Date:** February 8, 2026
**Status:** ✅ Complete

---

## Summary

| Tool | Status | Notes |
|------|--------|-------|
| LibreOffice | ✅ Installed | v24.2.7.2 - Headless PDF generation |
| Playwright | ✅ Installed | Chromium v145 - Browser automation |
| ChromaDB | ✅ Installed | Vector database for semantic memory |
| Semantic Scholar | ✅ Working | API helper created - no key needed |
| Gmail API | ⚠️ Pending | Needs OAuth scope update |
| Ollama | 📋 Documented | Optional - install when needed |
| Tailscale | 📋 Documented | In config (mode: off) |

---

## 1. LibreOffice Headless ✅

**Version:** LibreOffice 24.2.7.2
**Location:** `/usr/bin/libreoffice`

### Usage
```bash
# Convert DOCX to PDF
libreoffice --headless --convert-to pdf "document.docx"

# Convert to specific format
libreoffice --headless --convert-to pdf:writer_pdf_Export "input.docx" --outdir /output/
```

### Test Result
- ✅ Converted Thai DOCX to PDF with proper font embedding
- ✅ 1MB PDF created with readable Thai text
- File: `/home/clawdbot/clawd/tmp/ชายชราสามคน-V2.pdf`

---

## 2. Playwright ✅

**Version:** Latest (installed Feb 8, 2026)
**Browser:** Chromium Headless 145.0.7632.6

### Installation
```bash
npm install playwright          # Local install
npx playwright install chromium # Download browser
npx playwright install-deps chromium  # System deps
```

### Usage
```javascript
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  await page.goto('https://example.com');
  await page.screenshot({ path: 'screenshot.png' });
  await browser.close();
})();
```

### Test Result
- ✅ Screenshot captured successfully
- File: `/home/clawdbot/clawd/tmp/playwright_test.png`

---

## 3. ChromaDB ✅

**Version:** 1.4.1
**Location:** Virtual environment at `/home/clawdbot/clawd/.venv/`
**Database:** `~/.openclaw/chroma_db/`

### Installation
```bash
# Activate venv first
source /home/clawdbot/clawd/.venv/bin/activate
```

### Helper Script
`/home/clawdbot/clawd/tools/chroma_helper.py`

### Usage
```bash
# Index memory files
python3 tools/chroma_helper.py index

# Search
python3 tools/chroma_helper.py search "query here"

# Check status
python3 tools/chroma_helper.py status
```

### Test Result
- ✅ Indexed 11 memory files (59 chunks)
- ✅ Semantic search working
- Uses MiniLM-L6-v2 embeddings (downloaded automatically)

---

## 4. Semantic Scholar API ✅

**API:** Free tier (no key required)
**Endpoint:** https://api.semanticscholar.org/graph/v1

### Helper Script
`/home/clawdbot/clawd/tools/semantic_scholar.py`

### Usage
```bash
# Search papers
python3 tools/semantic_scholar.py "longevity interventions"

# In Python
from tools.semantic_scholar import search_papers, get_paper
papers = search_papers("machine learning", limit=10)
```

### Test Result
- ✅ Retrieved 5 papers on "longevity interventions aging"
- ✅ Returns titles, authors, citations, abstracts, PDFs
- ✅ No rate limiting issues on basic queries

---

## 5. Gmail API ⚠️ Pending

**Status:** OAuth credentials exist but need Gmail scope

### Current State
- Google OAuth credentials at `/home/clawdbot/clawd/gdrive/`
- Currently only has Drive scope
- Need to add Gmail scope and re-authorize

### To Enable
1. Go to Google Cloud Console
2. Enable Gmail API
3. Update OAuth scopes to include:
   - `https://www.googleapis.com/auth/gmail.readonly`
   - `https://www.googleapis.com/auth/gmail.send`
4. Re-authorize the application

### Helper Script
Created placeholder at `/home/clawdbot/clawd/tools/gmail_helper.py`

---

## 6. Ollama 📋 Documented

**Status:** Not installed (optional - install when needed)

### To Install
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2:3b  # Small model
ollama pull mistral      # Good balance
```

### Usage
```bash
ollama run llama3.2 "Hello, how are you?"
```

### Notes
- Requires ~4GB RAM for small models
- GPU acceleration if available
- Good for: fast local queries, privacy, cost savings

---

## 7. Tailscale 📋 Documented

**Status:** Already in config (mode: off)

### To Enable
Edit `~/.openclaw/openclaw.json`:
```json
"tailscale": {
  "mode": "on"
}
```

Or use CLI:
```bash
openclaw config set gateway.tailscale.mode on
```

### Features
- Secure remote access to OpenClaw
- Access from phone/laptop anywhere
- Zero-config VPN

---

## New Tools Directory

Created `/home/clawdbot/clawd/tools/` with:
- `semantic_scholar.py` - Research paper search
- `chroma_helper.py` - Vector database operations
- `gmail_helper.py` - (placeholder for email)

---

## Virtual Environment

Created `/home/clawdbot/clawd/.venv/` with:
- Python 3.12
- chromadb
- requests
- numpy, onnxruntime (ChromaDB deps)

### Usage
```bash
source /home/clawdbot/clawd/.venv/bin/activate
python3 script.py
deactivate
```

---

## System Notes

⚠️ **Kernel Update Available**
- Current: 6.8.0-90-generic
- Available: 6.8.0-94-generic
- Recommend rebooting when convenient

---

## Next Steps

1. **Gmail API** - Enable in Google Cloud Console when email integration needed
2. **Ollama** - Install when local model inference needed
3. **Tailscale** - Enable when remote access needed
4. **Integrate ChromaDB** - Update memory_search to use ChromaDB for better semantic search

---

*Report generated: Feb 8, 2026*
