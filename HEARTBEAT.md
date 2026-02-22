# HEARTBEAT.md - Proactive Checks

## Response Format (ALWAYS use this)
After completing checks, ALWAYS reply with a status report like this:

```
⚡ Heartbeat Report ({time}):
• Model: {your model name} ✅
• Status: {ok or alert details}
• Checks: {what you checked}
{any alerts or findings}
```

Do NOT reply with HEARTBEAT_OK ever. Do NOT include the word HEARTBEAT_OK anywhere in your response. Always give the full status report above — OpenClaw suppresses messages containing HEARTBEAT_OK.

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
