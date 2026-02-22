---
name: session-snapshot
description: "Periodically snapshot session context to memory on message:received"
metadata: { "openclaw": { "emoji": "📸", "events": ["message:received"] } }
---

# Session Snapshot Hook

Triggers on every incoming message. Every 20 messages, saves a context
snapshot to memory/YYYY-MM-DD.md so continuity is preserved even without /new.

## What it does
- Counts inbound messages per session
- Every 20 messages: appends a timestamped summary to today's memory file
- Lightweight — only writes a brief snapshot, not full transcript
