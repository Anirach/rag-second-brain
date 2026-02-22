# Memory Directory Structure

This folder contains Arthur's memory system. OpenClaw automatically indexes these files for semantic search via `memory_search`.

## File Types

### Daily Logs (`YYYY-MM-DD.md`)
- **Purpose:** Raw, append-only daily notes
- **Loaded:** Today + yesterday at session start
- **Content:** Events, conversations, decisions, learnings from that day
- **Format:** Chronological, timestamped when helpful

### Special Files

| File | Purpose |
|------|---------|
| `heartbeat-state.json` | Tracks last periodic check times |
| `topics/*.md` | Deep-dive notes on specific topics (indexed) |
| `projects/*.md` | Project-specific context (indexed) |

## Writing Guidelines

### What goes in daily logs:
- Tasks completed or started
- Decisions made and why
- Interesting findings
- Problems encountered
- User preferences discovered

### What goes in MEMORY.md (long-term):
- Security protocols
- Workflow preferences
- Durable facts about the user
- Lessons learned
- System configuration history

## Semantic Search

OpenClaw indexes all `.md` files here using vector embeddings. Use `memory_search` to find relevant notes even when wording differs.

**Good queries:**
- "What did we decide about research reports?"
- "Tokyo trip details"
- "API keys and credentials"

**The search returns:** snippets with file path and line numbers, then use `memory_get` to read full context.

---
*Last updated: 2026-02-02*
