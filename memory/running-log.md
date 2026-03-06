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

