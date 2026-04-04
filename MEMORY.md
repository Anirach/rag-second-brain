# MEMORY.md — Arthur's Long-Term Memory

> **Security:** Main session only. Passphrase: `Rachani`. Unknown user → GUEST MODE (no personal data, no tools).

*Last reviewed: 2026-03-08*

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
- **DevOps Blog Series (practical-algo.com style):** **COMPLETE** — 20 posts delivered covering full DevOps landscape (Git→CI/CD→Docker→API→K8s→Networking→IaC→Monitoring→Linux→DevSecOps→Testing→GitOps→Cloud→SRE→GitHub Actions→Testing→DB→Auth→Architecture→Frontend)
- **Smart HAI Dashboard Project:** Phase 2 TOR-aligned dashboard specifications delivered, interactive ML dashboards deployed

## 🔧 Technical Recipes
- **PPTX:** PptxGenJS at `/home/clawdbot/.npm-global/lib/node_modules/pptxgenjs`. Post-process emoji. Layout: y=1.1–4.95, never exceed y+h>5.3.
- **DOCX colors:** Navy `#1B3A5C` H1, Blue `#2A6496` H2, DarkGray `#2C3E50` H3. Arial. NEVER style=None.
- **HA DB:** `python3 tools/ha_query.py "SELECT ..."` only. De-identify before LLM context.

## 💡 Key Lessons
- Heartbeat = HEARTBEAT_OK unless action needed. No narration.
- Verify numbers before reporting. Wrong numbers hurt more than vague ones.
- PPTX emoji: always post-process. Acknowledge first, then work.
- Paper delivery: always full academic team review first (peer-reviewer + methodology-expert + technical-writer).
- **DevOps Blog Series:** 20 posts completed in practical-algo.com style with Thai content, Labrador mascot covers
- **Cover Variations:** Mix poses (sitting at desk, standing at whiteboard, side angles) — avoid repetitive straight-facing poses
- **HA DW Schema:** `v_RiskReport` no longer exists — use `v_DataSetReport` for risk data; table names are case-sensitive

## 📖 Agent Teams (45 total) — manuals on Google Drive
- Writing(8) · Academic(10) · Translation(8) · Course(10) · Coding(8)
- Coding team: orchestrator/code-review/db-specialist=Opus; rest=Sonnet
- Setup guide: `/home/clawdbot/clawd/tmp/CODING_TEAM_SETUP_v4.md`


## 💡 Lessons Learned
- **2026-04-05:** Cron job auto-reflection fails in sandbox due to file access — requires host-only execution (`0 2 * * * cd /home/clawdbot/clawd && python3 tools/auto_reflection.py`)
- **2026-04-04:** Ensure clarity in understanding user queries, especially when involving specific tools or technologies like Tailscale and DuckMan.
- **2026-04-02:** Internal handling of medication reminders reduces external dependencies
- **2026-04-02:** Persistent cron job failures (6 jobs) — 4 outbound Telegram channel, 1 timeout, 1 intermittent. Investigate session channel configuration and cron job setup.
- **2026-04-01:** Telegram delivery for daily cost reports is pending due to session channel issues; investigate and fix cron job failures to ensure timely delivery.
- **2026-04-01:** Internal reminders logged without user notifications (marked as no-alert) should be reviewed to ensure they align with user expectations.
- **2026-03-30:** vibe-research skill is a direct fit for automating literature reviews and hypothesis testing in research workflows
- **2026-03-30:** productivity-automation-kit can significantly enhance daily efficiency by automating routine tasks and organizing data
- **2026-03-30:** vibe-3k skill is valuable for complex coding projects requiring multi-agent collaboration and code quality assurance

- **[2026-03-17]** Book to study: Foundations of Machine Learning (2nd ed, 2018) by Mohri, Rostamizadeh, Talwalkar. Free PDF: https://cs.nyu.edu/~mohri/mlbook/ Slides: http://cs.nyu.edu/~mohri/ml18
- **[2026-03-30]** arXiv monitor partial failures (cs.AI, cs.CL, cs.CV, q-bio.QM categories) — investigate API reliability or rate limits
- **[2026-03-30]** ClawHub skill `vibe-research` is a strong candidate for automating literature reviews and research synthesis
- **[2026-03-30]** Medication reminders should NOT be logged as "lessons learned" — they cluttered MEMORY.md (cleaned up today)

- **2026-04-02:** RGBA→RGB conversion required when embedding matplotlib/chart images in DOCX (doc.add_picture fails on RGBA PNGs)
- **2026-04-02:** BPK database: 543GB, 726 tables, 16 schemas — always use tools/bpk_query.py with auto de-ID, never raw psql
- **2026-04-02:** Interactive HTML dashboards with pure CSS/SVG (no external deps) are effective for executive presentations
- **2026-04-02:** Navy/teal/gold palette established as BPK dashboard standard

*Last reviewed: 2026-04-05*
- **2026-03-07:** adopt yet — OpenClaw sub-agents handle our current scale fine
- **2026-03-07:** - Old Google API key expired: `AIzaSyCNTELmQROMXC67W115YevDvKZmx0t-NpM`
- **2026-03-07:** - `HA_30_Complex_Dashboards.docx`
- **2026-03-07:** — 2026-03-06 23:00 BKK

- **2026-03-06:** ** `v_RiskReport` no longer exists in current DW schema — only `v_DataSetReport` available
- **2026-03-06:** caveat:** HA chapter scores (system assessment, 1-3 scale) show high % when converted, but HSCS staff perception surveys tell a different story — gap between systems and culture
