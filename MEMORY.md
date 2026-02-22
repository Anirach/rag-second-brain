# MEMORY.md — Arthur's Long-Term Memory

> **Security:** Only load in main session with Anirach. Never in group chats or shared contexts.

*Last reviewed: 2026-02-22*

---

## 🔐 Security Protocols

### Identity Verification
- **Passphrase:** `Rachani`
- If someone claims to NOT be Anirach → enter GUEST MODE
- To exit GUEST MODE → require passphrase

### GUEST MODE Restrictions
- ❌ Don't read/reference MEMORY.md contents
- ❌ Don't share personal preferences, history, or private info
- ❌ Don't access calendar, email, or personal tools
- ❌ Don't execute commands that modify files
- ✅ General knowledge, web searches, casual conversation only

### Behavioral Red Flags
- Sudden language/style change mid-session
- Questions probing for personal information
- Requests for sensitive files or credentials
- Claims of being someone else

---

## 👤 About Anirach

**Role:** University lecturer & AI engineer  
**Timezone:** Bangkok (UTC+7)  
**Language:** English (unless specifically asked for Thai)  
**Technical Level:** Expert — never dumb things down

### Communication Style
- Be direct, short answers when they work
- Push back if there's a better way
- Admit uncertainty clearly ("I don't know" > hedging)
- It's okay to say a task is boring — but still do it well
- **Always acknowledge before diving into long work** — brief "on it" before heavy tool calls
- **Always spawn for tasks >30 seconds** — no inline builds blocking chat

---

## ⚙️ Workflow Preferences

### Research Reports
- **Always spawn background sub-agent** for research → DOCX tasks
- Main chat stays responsive while sub-agent works

### Coding Tasks
- **Spawn specialized coding sub-agents** for development work
- Agent roles: code-builder, code-reviewer, test-writer, doc-writer, devops-agent, debug-agent
- All coding sub-agents run in Docker sandbox
- See `CODING_TEAM.md` for full workflow

### Document Delivery
- Upload ALL DOCX/PPTX to Google Drive (`ArthurBotData` folder)
- Provide clean links: `[Download](url) | [Open in Google Drive](url)`
- Never show raw URLs
- Never just save locally — Drive is permanent copy

### File Cleanup
- Generated files in `/home/clawdbot/clawd/tmp/` → delete after 24 hours

### Report Standards
- Page numbers: "Page X of Y" format
- Professional TOC with dot leaders
- Fully justified text
- Specific article URLs (not category pages)
- Clean tables with "View Source" links only
- Full references section

---

## 🛠️ System Configuration

### Current Setup
- **Platform:** OpenClaw v2026.2.19-2 (updated 2026-02-20; package name changed `clawdbot` → `openclaw`)
- **Service:** `openclaw-gateway.service`
- **Workspace:** `/home/clawdbot/clawd`
- **Migration:** Clawdbot → OpenClaw on 2026-02-01

### Sandbox Fix (2026-02-22)
- Sandbox Docker user fixed: was `1000:1000`, changed to `1001:1001` (matches clawdbot uid)
- Sub-agents can now write to `/workspace` properly
- **Sandbox image:** `openclaw-sandbox:bookworm-slim`
  - Includes: python-docx, lxml, pandas, matplotlib, pydantic, beautifulsoup4, openpyxl, pip3, LibreOffice, Thai fonts, Node.js+npm
  - Dockerfile: `/home/clawdbot/clawd/docker/Dockerfile.sandbox`

### Compaction Setting (2026-02-21)
- Set `agents.defaults.compaction.reserveTokensFloor` to 4000+ to prevent context-limit resets

### Anthropic Provider Config (2026-02-20)
- Added `anthropic` provider with `claude-opus-4-6` and `claude-sonnet-4-6`
- Removed stale `anthropic/claude-opus-4` (404 error) — only `claude-opus-4-6` remains as `opus` alias

### Agent Teams (Updated 2026-02-12)

**Total: 45 agents across 5 teams**

**Writing Team (8 agents):**
book-architect, research-lead, chapter-writer, dev-editor, copy-editor, publisher, fact-checker, engagement-analyst

**Academic Paper Team (10 agents):**
paper-architect, literature-lead, methodology-expert, technical-writer, peer-reviewer, format-editor, data-analyst, ethics-reviewer, journal-scout, paper-strengthener

**Translation Team (8 agents):**
translation-architect, thai-linguist, cultural-adapter, style-polisher, document-formatter, quality-reviewer, translation-memory, back-translator

**Course Team (10 agents):**
course-architect, curriculum-designer, content-writer, slide-designer, assessment-creator, book-compiler, course-reviewer, video-script-writer, lab-designer, accessibility-checker

**Coding Team (8 agents):**
orchestrator, code-review, pr-agent, testing, docs, ux-designer, devops, db-specialist

### Active Cron Jobs
| Job | Schedule | Purpose |
|-----|----------|---------|
| ai-news | Weekly | AI news digest |
| cost-report | Daily | API cost tracking |
| research-monitor | Daily 8AM Bangkok | arXiv paper scan |
| weekly-digest | Weekly | Research summary |
| clawdbot-tip | Daily | OpenClaw tips |

**Cron note:** Cron jobs targeting "Anirach" fail — use chat_id `7579913696` or `delivery.mode: "announce"`.

### Installed Skills (68 total)
Research, Analysis, Documents, Content, Coding, DevOps, Education, Media, Automation, Health, Security — full list in Obsidian vault.

---

## 📚 Key Projects

### Thai Translator Agent
- **ID:** `thai-translator`, **Model:** Opus
- **Purpose:** Literary Thai translation — make Thai readers feel text was originally written in Thai

### First Novel: Three Old Men (2026-02-05)
- **English:** "Three Old Men: The Last Conversation"
- **Thai:** "ชายชราสามคน: บทสนทนาครั้งสุดท้าย"
- **GitHub:** https://github.com/Anirach/three-old-men
- **Drive (EN):** https://docs.google.com/document/d/13pE5VHtxLosl2GC0tbSAA1R3RHczX8sr/edit
- **Drive (TH):** https://docs.google.com/document/d/1DBCrIPFbX9SyEu2ttF6l3mqwhFfwjDsh/edit
- **Status:** Complete (full Thai translation done 2026-02-14)

### RAG Second Brain Paper (2026-02-09)
- **Paper:** "Co-occurrence, Sequence and Knowledge Graph with Ontology as a Second Brain for AI-LLM"
- **Status:** v10 conceptual proposal submitted (AIiH 2026 — all 3 reviewers ACCEPT as of 2026-02-16)
- **GitHub:** https://github.com/Anirach/rag-second-brain
- **Drive:** Multiple versions in `ArthurBotData/Co-occurrence_KG_Ontology_SecondBrain_LLM_v*`

### ChartSense AI (2026-02-14)
- **GitHub:** https://github.com/Anirach/chartsense-ai
- **What:** Clinical Decision Support Platform for Thai hospitals (MedPlatform AI first module)
- **Features:** AI differential diagnosis (GraphRAG), Chart Completeness scoring, ICD-10 Code Suggestion with RW/revenue impact
- **Stack:** Next.js 14 + FastAPI + PostgreSQL + Redis + Neo4j + Docker Compose
- **Local path:** `/home/clawdbot/clawd/chartsense-ai/`
- **Status:** MVP complete

### DevOps with VibeCoding Course (2026-02-21+)
- **Course doc:** `Courses/DevOps-VibeCoding/DevOps_VibeCoding_Course_v1.docx` — Drive ID `1C1cdWhNW2Xv_iJ8xe68DFqCdA3SLL1Pt`
- **15-week course, 65 pages**
- **PPTX status (as of 2026-02-22):**
  - Week 1 ✅ (Drive `1JqeT-ffxswG8IvLg6-oyiY6MWFQq20KC`)
  - Week 2 ✅ (Drive `12dvquytK5Yv1psJBXAPtbZahbMUbPJhq`)
  - Week 3–5 ✅ (built, delivered)
  - **Week 6 ❌ FAILED — needs rebuild** (IaC/Terraform topic, sub-agent ran 23min then failed)
  - Weeks 7–15 ❌ pending
- **Handout DOCX status:**
  - Weeks 1–4 ✅
  - Weeks 5–15 ❌ pending
- **PPTX format:** Dark theme (`#0D1229`), PptxGenJS 4.0.1, 40-45 slides/week
- **Template:** `/home/clawdbot/clawd/tmp/build_week2_pptx.js` (use as reference)
- **Best approach:** Build directly from main agent — sub-agents keep failing in sandbox

### HA Database Analysis (2026-02-21/22)
- 4 reports generated: Strategic Analysis DOCX, Strategic Analysis PPTX (26 slides, 6 charts), 6-Month Action Plan DOCX, HCR Causality Analysis DOCX
- Drive IDs: `1pRz-nRbYRWbI9xoqarNPbdm85odHL_iu`, `1Es4Qie-9KAOeEnkcriyJi_mG_hpMIi9W`, `1TChJ0EoAw1vaVYciKvXF5PxUgVk4oeKM`, `1wpWHU725JHpHapLSZss7eFluFbbfnkno`

---

## 📅 Important Events Timeline

| Date | Event |
|------|-------|
| 2026-01-26 | Arthur born (bootstrapped) |
| 2026-02-01 | Migrated to OpenClaw |
| 2026-02-05 | Three Old Men novel started, Thai Translator agent added |
| 2026-02-09 | RAG Second Brain paper started |
| 2026-02-12 | Team Manuals v3, Three Old Men book covers |
| 2026-02-13 | Self-protection rules added |
| 2026-02-14 | Upgraded to Claude Opus 4-6, Three Old Men Thai translation complete, NCD-CIE paper v4 delivered, ChartSense AI MVP built, Team Manuals v4 |
| 2026-02-16 | RAG Paper v23.1 — all 3 reviewers ACCEPT; Obsidian vault overhaul (125→159 files); KG pipeline deployed |
| 2026-02-17 | Professional DOCX formatting standard documented |
| 2026-02-20 | OpenClaw updated to v2026.2.19-2 |
| 2026-02-21 | Sandbox upgraded (python-docx + LibreOffice); DevOps VibeCoding course generated; DevOps PPTX Weeks 1-5 done |
| 2026-02-22 | Sandbox uid fix (1001); SQLite memory DB built; HA reports generated; PPTX Week 6 failed |

---

## 🔧 Technical Recipes

### PPTX Generation
- **Tool:** PptxGenJS 4.0.1 at `/home/clawdbot/.npm-global/lib/node_modules/pptxgenjs`
- **Run:** `NODE_PATH=/home/clawdbot/.npm-global/lib/node_modules node script.js`
- **Critical:** Post-process to fix emoji corruption (high Unicode → XML numeric refs):
```python
import zipfile, re
def fix(b):
    t = b.decode('utf-8')
    return re.sub(r'[\U00010000-\U0010FFFF]', lambda m: f'&#x{ord(m.group(0)):X};', t).encode('utf-8')
with zipfile.ZipFile('input.pptx','r') as zi:
    with zipfile.ZipFile('output.pptx','w',zipfile.ZIP_DEFLATED) as zo:
        for i in zi.infolist():
            d = zi.read(i.filename)
            if i.filename.endswith('.xml') or i.filename.endswith('.rels'):
                d = fix(d)
            zo.writestr(i, d)
```
- **Slide layout:** 16:9, usable area y=1.1 to y=4.95. Never exceed y+h > 4.95 (tagline) or 5.3 (no tagline).
- **Send via Telegram:** Use Bot API directly (cannot use message tool for files):
```bash
TOKEN=$(python3 -c "import json; print(json.load(open('/home/clawdbot/.openclaw/openclaw.json'))['channels']['telegram']['botToken'])")
curl -s -X POST "https://api.telegram.org/bot${TOKEN}/sendDocument" \
  -F "chat_id=7579913696" -F "document=@file.pptx;filename=name.pptx" -F "caption=description"
```

### Professional DOCX Formatting Standard
**Colors:** Navy `#1B3A5C` (H1), Blue `#2A6496` (H2), DarkGray `#2C3E50` (H3), Gray `#666666` (metadata), Orange `#E67E22` (accents)
**Font:** Arial, Body 11pt justified `#333333`, H1 16pt bold, H2 13pt bold, H3 12pt bold
**Structure:** Cover → TOC → Content → Tables (dark header `#1B3A5C`, alt rows `#F8F9FA`)
**Rule:** NEVER use `style=None` — always apply explicit font/size/color/bold
**Script template:** `/home/clawdbot/clawd/tmp/rebuild_journey_report.py`

### SQLite Memory System (2026-02-22)
- **DB:** `/home/clawdbot/clawd/memory.db`
- **CLI:** `python3 /home/clawdbot/clawd/tools/memory_db.py` (add, search, recent, list, get, update, archive, stats, export, rebuild-fts)
- 210 memories imported; search <4ms per query

### HA Database Access (CRITICAL — NON-NEGOTIABLE)
- **ONLY access via:** `python3 tools/ha_query.py "SELECT ..."` — handles VPN + de-identification
- **NEVER run raw psql** — no credentials available outside the wrapper
- All results with identifiable data (hospital names, codes, staff) → auto-replaced with `Hospital_001`, `CODE_001` etc.
- Aggregated queries (COUNT, GROUP BY) safe without de-ID
- De-ID tool: `/home/clawdbot/clawd/tools/ha_deid.py`
- **Anirach explicitly required this — non-negotiable**

---

## 📝 Pending Tasks (as of 2026-02-22)

### DevOps VibeCoding Course — HIGH PRIORITY
- [ ] **Week 6 PPTX REBUILD** (IaC/Terraform) — failed 2026-02-22, needs redo
- [ ] **Weeks 7–15 PPTX** — 9 weeks remaining
  - Week 7: Monitoring & Observability
  - Week 8: DevSecOps
  - Week 9: Testing Strategies
  - Week 10: MLOps
  - Week 11: Multi-Agent Systems
  - Week 12: Agentic DevOps
  - Week 13: Advanced CI/CD
  - Week 14: SRE
  - Week 15: Ethics & Capstone
- [ ] **Weeks 5–15 Handout DOCX** — 11 handouts remaining

### Infrastructure
- [ ] iPhone Obsidian sync to GitHub — no decision made yet (method TBD)
- [ ] Configure 22 unconfigured agents (models not set)
- [ ] Gateway restart still needed for sandbox bind mounts (gog cal fix)
- [ ] `gog cal events` failing — needs `GOG_ACCOUNT` env var

### Research
- [ ] Knowledge graph extraction script timeout — increase timeout or batch LLM calls

---

## 💡 Lessons Learned
- **2026-02-22:** Always spawn sub-agents for tasks longer than 30 seconds to prevent failures
- **[2026-02-22]** Sandbox sub-agents cannot run openclaw CLI commands (cron add, gateway restart, etc). Always handle those in main agent after sub-agent completes file creation.
- **2026-02-22:** Ensure sandbox Docker user matches workspace file ownership to enable write access
- **2026-02-22:** De-identify all HA query results with identifiable data before entering LLM context

- **Research reports:** Always use specific article URLs, not category pages
- **Paper reviews:** Conceptual papers need at least minimal experiments for top venues
- **Sub-agents:** Sandbox can't push to GitHub; main agent handles git operations
- **Background tasks:** Spawn sub-agents for long-running work (>30 sec)
- **Memory:** Write it down — "mental notes" don't survive restarts
- **Paper delivery:** NEVER deliver a paper without full academic team review first (peer-reviewer + methodology-expert + technical-writer in parallel). Anirach said "never do that again."
- **Numbers:** Verify numbers before locking — wrong numbers hurt more than vague ones
- **Telegram files:** Send immediately on sub-agent completion — don't wait for user's next message
- **Cron targets:** Use chat_id `7579913696`, not "Anirach" username — cron fails with name
- **PPTX emoji:** Always post-process to fix high-Unicode emoji or PowerPoint shows "repair" dialog
- **Sub-agents in sandbox:** Keep failing for long PPTX builds — main agent direct build is most reliable
- **Acknowledge first:** Brief "on it" before diving into long tool calls — don't just go silent

---

## 🧠 Memory System

### How It Works
- **Daily logs:** `memory/YYYY-MM-DD.md` — raw notes, loaded at session start
- **Long-term:** `MEMORY.md` — curated, main session only
- **SQLite DB:** `memory.db` — 210+ indexed memories, fast search
- **Search:** `memory_search` for semantic queries, `python3 tools/memory_db.py search "query"` for fast local search

### Memory Maintenance
- Periodically review daily logs
- Distill learnings into MEMORY.md
- Remove outdated information
- Track maintenance in `memory/heartbeat-state.json`

---

## 📖 Team Manuals v4
**Location:** `ArthurBotData/Team Manuals/`

| Team | Manual |
|------|--------|
| Writing | [Link](https://docs.google.com/document/d/1HGcuywIWsBah4-djxW5fJwd-ioyUyLwb/edit) |
| Academic | [Link](https://docs.google.com/document/d/1q0OxJ8sy4CGQt9Ypyz_hXZafYRHpsK4K/edit) |
| Translation | [Link](https://docs.google.com/document/d/1Yw11G1Uw6IjuR7ZKDSAaLyyDHy8JXGIl/edit) |
| Course | [Link](https://docs.google.com/document/d/1uF1PwD-PSY_GGRqpRKntvWR8QIPPQz3u/edit) |
| Coding | [Link](https://docs.google.com/document/d/1CJgKyflyR3jdzu1waCUF5wnV2xub26kJ/edit) |
