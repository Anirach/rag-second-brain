# Running Log

Auto-updated daily by auto_reflection.py

---
## 2026-02-22 (15:55 BKK)

**Files reviewed:** 2026-02-22, 2026-02-21

**Lessons:**
- Always spawn sub-agents for tasks longer than 30 seconds to prevent failures
- Ensure sandbox Docker user matches workspace file ownership to enable write access
- De-identify all HA query results with identifiable data before entering LLM context

**Decisions:**
- Changed sandbox Docker user to 1001:1001 in openclaw.json
- Removed stale anthropic/claude-opus-4 model from config, keeping anthropic/claude-opus-4-6 as opus alias
- Implemented HA query wrapper (ha_query.py) to handle VPN and de-identification automatically


---
## 2026-02-23 (02:01 BKK)

**Files reviewed:** 2026-02-22

**Lessons:**
- Always spawn sub-agents for tasks longer than 30 seconds to prevent failures
- Ensure sandbox Docker user settings match workspace file ownership to enable proper write access
- De-identify all HA query results with identifiable data before entering LLM context to maintain privacy

**Decisions:**
- Changed `agents.defaults.sandbox.docker.user` to `1001:1001` in openclaw.json to fix sandbox write access
- Removed stale `anthropic/claude-opus-4` from config and kept only `anthropic/claude-opus-4-6` as `opus` alias
- Implemented a strict rule for de-identification of HA query results before LLM context inclusion


---
## 2026-02-24 (02:00 BKK)

**Files reviewed:** 2026-02-23

**Lessons:**
- Always check cron delivery settings when creating new jobs to avoid notification spam
- Arscontexta's /remember command can inspire session_remember.py for better session memory management

**Decisions:**
- Implement session_remember.py inspired by Arscontexta /remember command


---
## 2026-03-02 (02:01 BKK)

**Files reviewed:** 2026-03-01

**Lessons:**
- - "https://unused"`, changed to `https://generativelanguage.googleapis.com/v1beta`
- — 2026-03-01 23:00 BKK


---
## 2026-03-03 (02:00 BKK)

**Files reviewed:** 2026-03-02


---
## 2026-03-07 (02:00 BKK)

**Files reviewed:** 2026-03-06

**Lessons:**
- - `HA_30_Complex_Dashboards.docx`
- — 2026-03-06 23:00 BKK

**Decisions:**
- via information_schema query
- i_1_lrd, i_2_stg, i_3_pcm, i_4_kam, i_5_wkf, i_6_opt, ii_1_rsq...ii_9_com, iii_1_acn...iii_6_coc, iv_1_hcr, iv_2_cfr, iv


---
## 2026-03-08 (02:00 BKK)

**Files reviewed:** 2026-03-08, 2026-03-07

**Lessons:**
- — 2026-03-07 23:00 BKK
- adopt yet — OpenClaw sub-agents handle our current scale fine
- - Old Google API key expired: `AIzaSyCNTELmQROMXC67W115YevDvKZmx0t-NpM`

**Decisions:**
- - 8:00 AM and 10:00 PM daily (Bangkok time)


---
## 2026-04-05 (03:00 BKK)

**Weekly memory maintenance completed:**
- Read memory/ files from 2026-03-29 to 2026-04-04 (7 days)
- Updated MEMORY.md with new lessons:
  - Cron job auto-reflection fails in sandbox (needs host execution)
  - BPK dashboards: navy-teal-gold palette, pure CSS/SVG
- Archived no completed action items (all open items > 2 weeks remain pending)
- Ran quick_note.py for 2 new technical facts
- Committed and pushed obsidian vault: "Weekly memory consolidation: cron sandbox fix, BPK dashboards, vibe skills"

**Key learnings distilled:**
- Cron job failures persist (6 jobs: 4 outbound channel, 1 timeout, 1 intermittent)
- BPK database: 543GB, 726 tables, 16 schemas — always use tools/bpk_query.py
- Arxiv monitor needs resilience (4 categories failing)

