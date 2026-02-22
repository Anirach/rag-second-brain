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

