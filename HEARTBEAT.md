# HEARTBEAT.md - Proactive Checks

## ⚠️ STRICT Response Rules

**RULE 1: Silent when nothing is actionable**
- If all checks pass with no issues → reply EXACTLY: `HEARTBEAT_OK`
- That is your ENTIRE reply. No explanation. No summary. No "I checked X and Y."
- DO NOT write anything before or after HEARTBEAT_OK
- DO NOT say "Checking heartbeat status" or any other prefix

**RULE 2: Alert ONLY when something needs attention**
- Urgent email → alert
- Calendar event < 2h → alert
- Stuck/failed sub-agent → alert
- Critical error found → alert

**RULE 3: No status reports — EVER**
- Do NOT narrate what you checked
- Do NOT describe tool calls or intermediate steps
- Do NOT send any message if you would end it with HEARTBEAT_OK anyway
- Think of it like a smoke alarm: silent = everything fine, noise = real problem
- WRONG: "Checking status... all good. HEARTBEAT_OK"
- CORRECT: HEARTBEAT_OK

## Every Heartbeat (rotate 2-3 items)
- [ ] Check `memory/heartbeat-state.json` for last check times
- [ ] Email inbox - urgent unread?
- [ ] Calendar - events in next 24h?
- [ ] Weather in Bangkok - relevant for plans?
- [ ] Active sub-agents - any completed/stuck?

## Daily (once per day)
- [ ] Review yesterday's `memory/YYYY-MM-DD.md`
- [ ] Update MEMORY.md with learnings
- [ ] Check cron job health
- [ ] Git status on active projects

## Weekly
- [ ] Clean `/home/clawdbot/clawd/tmp/` (files >7 days)
- [ ] Review and prune old memory files
- [ ] Check for OpenClaw updates

## Rules
- Late night (23:00-08:00 Bangkok): Still report, but keep it brief
- Track checks in `memory/heartbeat-state.json`
- Reach out if something important found
- Do background work silently (git, organize, cleanup)
