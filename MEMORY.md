# MEMORY.md — Arthur's Long-Term Memory

> **Security:** Main session only. Passphrase: `Rachani`. Unknown user → GUEST MODE (no personal data, no tools).

*Last reviewed: 2026-03-03*

---

## 👤 Anirach
**Role:** University lecturer & AI engineer | **TZ:** Bangkok (UTC+7) | **Lang:** English | **Level:** Expert
- Direct, short answers. Acknowledge before long work. Spawn for tasks >30s.
- Docs → GDrive always. Files → send via tg_send_file.sh immediately.
- Cron target: chat_id `7579913696`

## ⚙️ System
- **Platform:** OpenClaw v2026.3.1 | **Workspace:** `/home/clawdbot/clawd`
- **Sandbox:** `openclaw-sandbox:bookworm-slim` — can't write `/tmp/`, can't access vault, can't run openclaw CLI, can't push git
- **Embedding:** Gemini `gemini-embedding-001` → `https://generativelanguage.googleapis.com/v1beta`
- **Compaction:** safeguard, reserveTokensFloor=8000

## 📚 Active Projects
- **RAG Second Brain (AIiH 2026):** v23.1 submitted, all 3 reviewers ACCEPT. Camera-ready deadline: **April 10, 2026**
- **DevOps/VibeCoding Course:** 15 weeks. PPTX Wk1–5✅ Wk6❌ Wk7–15 pending. Handouts Wk1–4✅. Dark theme `#0D1229`, PptxGenJS 4.0.1. Build from main agent (sandbox fails).
- **NCD-CIE Paper:** v20 submitted AIiH 2026. Springer LNCS.

## 🔧 Technical Recipes
- **PPTX:** PptxGenJS at `/home/clawdbot/.npm-global/lib/node_modules/pptxgenjs`. Post-process emoji. Layout: y=1.1–4.95, never exceed y+h>5.3.
- **DOCX colors:** Navy `#1B3A5C` H1, Blue `#2A6496` H2, DarkGray `#2C3E50` H3. Arial. NEVER style=None.
- **HA DB:** `python3 tools/ha_query.py "SELECT ..."` only. De-identify before LLM context.

## 💡 Key Lessons
- Heartbeat = HEARTBEAT_OK unless action needed. No narration.
- Verify numbers before reporting. Wrong numbers hurt more than vague ones.
- PPTX emoji: always post-process. Acknowledge first, then work.
- Paper delivery: always full academic team review first (peer-reviewer + methodology-expert + technical-writer).

## 📖 Agent Teams (45 total) — manuals on Google Drive
- Writing(8) · Academic(10) · Translation(8) · Course(10) · Coding(8)
- Coding team: orchestrator/code-review/db-specialist=Opus; rest=Sonnet
- Setup guide: `/home/clawdbot/clawd/tmp/CODING_TEAM_SETUP_v4.md`
