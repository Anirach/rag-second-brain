# MEMORY.md — Arthur's Long-Term Memory

> **Security:** Only load in main session with Anirach. Never in group chats or shared contexts.

*Last reviewed: 2026-03-02*

---

## 🔐 Security Protocols

### Identity Verification
- **Passphrase:** `Rachani`
- If someone claims to NOT be Anirach → enter GUEST MODE
- To exit GUEST MODE → require passphrase

### GUEST MODE Restrictions
- ❌ Don't read/reference MEMORY.md contents, personal preferences, history, or private info
- ❌ Don't access calendar, email, personal tools, or modify files
- ✅ General knowledge, web searches, casual conversation only

---

## 👤 About Anirach

**Role:** University lecturer & AI engineer | **Timezone:** Bangkok (UTC+7) | **Language:** English | **Level:** Expert

### Communication Style
- Direct, short answers when they work. Push back if there's a better way.
- **Always acknowledge before long work** — brief "on it" before heavy tool calls
- **Always spawn for tasks >30 seconds** — no inline builds blocking chat

---

## ⚙️ Workflow Preferences

- **Research reports:** Spawn background sub-agent → DOCX. Use specific article URLs, not category pages.
- **Coding tasks:** Spawn specialized sub-agents (see `CODING_TEAM.md`)
- **Documents:** Upload ALL to Google Drive (`ArthurBotData`), provide clean links. Never just save locally.
- **Report format:** Page X of Y, TOC with dot leaders, justified text, clean tables with "View Source" links
- **Paper delivery:** NEVER deliver without full academic team review first (peer-reviewer + methodology-expert + technical-writer in parallel)
- **Telegram files:** Send immediately on sub-agent completion via `tg_send_file.sh`
- **Cron targets:** Use chat_id `7579913696`, not "Anirach" username

---

## 🛠️ System Configuration

- **Platform:** OpenClaw v2026.2.26 | **Workspace:** `/home/clawdbot/clawd`
- **Sandbox image:** `openclaw-sandbox:bookworm-slim` (uid 1001:1001, python-docx, LibreOffice, Thai fonts, Node.js)
- **Sandbox limits:** Can't write to `/tmp/` or `~/obsidian-vault/`, can't run `openclaw` CLI, can't push to GitHub
- **Compaction:** safeguard mode, reserveTokensFloor=8000
- **Embedding:** Gemini (`gemini-embedding-001`) — baseUrl must be `https://generativelanguage.googleapis.com/v1beta`

### Agent Teams (45 total across 5 teams)
- **Writing (8):** book-architect, research-lead, chapter-writer, dev-editor, copy-editor, publisher, fact-checker, engagement-analyst
- **Academic (10):** paper-architect, literature-lead, methodology-expert, technical-writer, peer-reviewer, format-editor, data-analyst, ethics-reviewer, journal-scout, paper-strengthener
- **Translation (8):** translation-architect, thai-linguist, cultural-adapter, style-polisher, document-formatter, quality-reviewer, translation-memory, back-translator
- **Course (10):** course-architect, curriculum-designer, content-writer, slide-designer, assessment-creator, book-compiler, course-reviewer, video-script-writer, lab-designer, accessibility-checker
- **Coding (8):** orchestrator (Opus), code-review (Opus), db-specialist (Opus), pr-agent (Sonnet), testing (Sonnet), docs (Sonnet), ux-designer (Sonnet), devops (Sonnet)

---

## 📚 Active Projects

### RAG Second Brain Paper (AIiH 2026)
- **Status:** v23.1 submitted — all 3 reviewers ACCEPT (2026-02-16)
- **Deadline:** April 10, 2026 (camera-ready)
- **Details:** See Quick-Reference.md

### DevOps with VibeCoding Course
- **Course doc:** Drive ID `1C1cdWhNW2Xv_iJ8xe68DFqCdA3SLL1Pt` (15 weeks, 65 pages)
- **PPTX:** Weeks 1–5 ✅, Week 6 ❌ FAILED (needs rebuild), Weeks 7–15 pending
- **Handouts:** Weeks 1–4 ✅, Weeks 5–15 pending
- **Format:** Dark theme `#0D1229`, PptxGenJS 4.0.1, 40-45 slides/week
- **Note:** Build PPTX from main agent — sub-agents keep failing in sandbox

### Completed Projects (details in Obsidian vault)
- Three Old Men novel (EN+TH complete) | ChartSense AI MVP | HA Database Analysis (4 reports) | NCD-CIE Paper v4

---

## 🔧 Technical Recipes

### PPTX Generation
- **Tool:** PptxGenJS 4.0.1 at `/home/clawdbot/.npm-global/lib/node_modules/pptxgenjs`
- **Run:** `NODE_PATH=/home/clawdbot/.npm-global/lib/node_modules node script.js`
- **Critical:** Post-process emoji corruption (high Unicode → XML numeric refs)
- **Layout:** 16:9, usable y=1.1 to y=4.95. Never exceed y+h > 5.3.
- **Send files:** `bash /home/clawdbot/clawd/tools/tg_send_file.sh /path/to/file "caption"`

### Professional DOCX Standard
- **Colors:** Navy `#1B3A5C` (H1), Blue `#2A6496` (H2), DarkGray `#2C3E50` (H3)
- **Font:** Arial, Body 11pt justified `#333333`, H1 16pt, H2 13pt, H3 12pt
- **Rule:** NEVER use `style=None` — always explicit formatting

### HA Database (NON-NEGOTIABLE)
- **ONLY access via:** `python3 tools/ha_query.py "SELECT ..."` — handles VPN + de-identification
- **NEVER run raw psql.** VPN check: ppp0 up + route via ppp0 + TCP 30503 reachable. Timeout 40s + 2s stabilization.

---

## 💡 Lessons Learned (Condensed)
- **2026-03-02:** - "https://unused"`, changed to `https://generativelanguage.googleapis.com/v1beta`
- **2026-03-02:** — 2026-03-01 23:00 BKK

- Heartbeat = HEARTBEAT_OK unless something needs attention. No status reports.
- Sandbox can't: write /tmp, access vault, run openclaw CLI, push git
- Verify numbers before locking — wrong numbers hurt more than vague ones
- PPTX emoji: always post-process or PowerPoint shows "repair" dialog
- Acknowledge first, then work — don't go silent on long tasks
- De-identify all HA query results before entering LLM context
- Coding team setup guide v4.0: `/home/clawdbot/clawd/tmp/CODING_TEAM_SETUP_v4.md`

---

## 📖 Team Manuals v4
| Team | Link |
|------|------|
| Writing | [Link](https://docs.google.com/document/d/1HGcuywIWsBah4-djxW5fJwd-ioyUyLwb/edit) |
| Academic | [Link](https://docs.google.com/document/d/1q0OxJ8sy4CGQt9Ypyz_hXZafYRHpsK4K/edit) |
| Translation | [Link](https://docs.google.com/document/d/1Yw11G1Uw6IjuR7ZKDSAaLyyDHy8JXGIl/edit) |
| Course | [Link](https://docs.google.com/document/d/1uF1PwD-PSY_GGRqpRKntvWR8QIPPQz3u/edit) |
| Coding | [Link](https://docs.google.com/document/d/1CJgKyflyR3jdzu1waCUF5wnV2xub26kJ/edit) |
