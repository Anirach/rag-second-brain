#!/usr/bin/env python3
"""
Knowledge Graph Extractor
Fetches emails and calendar events via `gog` CLI, extracts entities using LLM,
and writes/updates structured Markdown notes in the Obsidian vault.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Optional

# --- Configuration ---
VAULT_BASE = Path(os.environ.get("KG_VAULT_PATH", "/home/clawdbot/obsidian-vault/KnowledgeGraph"))
def _get_openrouter_key():
    """Get OpenRouter API key from env or OpenClaw config."""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    try:
        import json as _json
        with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
            c = _json.load(f)
        return c.get("models", {}).get("providers", {}).get("openrouter", {}).get("apiKey", "")
    except Exception:
        return ""

OPENROUTER_API_KEY = _get_openrouter_key()
LLM_MODEL = os.environ.get("KG_LLM_MODEL", "deepseek/deepseek-chat")
GOG_CMD = os.environ.get("GOG_CMD", "gog")
TODAY = date.today().isoformat()
NOW = datetime.now().strftime("%Y-%m-%d %H:%M")

DIRS = ["People", "Projects", "Decisions", "Action-Items", "Documents"]


def run_cmd(args: list, timeout: int = 60) -> Optional[str]:
    """Run a command and return stdout, or None on failure."""
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return r.stdout.strip()
        print(f"  [warn] Command failed ({r.returncode}): {' '.join(args)}", file=sys.stderr)
        if r.stderr:
            print(f"  [stderr] {r.stderr[:200]}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"  [error] Command not found: {args[0]}", file=sys.stderr)
        return None
    except subprocess.TimeoutExpired:
        print(f"  [error] Command timed out after {timeout}s: {' '.join(args)}", file=sys.stderr)
        return None


def llm_extract(text: str) -> Optional[dict]:
    """Use OpenRouter API to extract structured entities from text."""
    if not OPENROUTER_API_KEY:
        return None

    prompt = f"""Extract entities from this email/calendar content. Return valid JSON only, no markdown fences.

{{
  "people": [
    {{"name": "Full Name", "role": "", "org": ""}}
  ],
  "projects": [
    {{"name": "Project Name", "status": "", "description": ""}}
  ],
  "decisions": [
    {{"summary": "What was decided", "owners": ["Name"], "date": "{TODAY}"}}
  ],
  "action_items": [
    {{"task": "What needs to be done", "owner": "Name", "deadline": "", "status": "open"}}
  ]
}}

If a category has no entities, use an empty list.

Content:
{text[:3000]}"""

    payload = json.dumps({
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 2000,
    })

    try:
        r = subprocess.run([
            "curl", "-s", "--max-time", "20", "-X", "POST", "https://openrouter.ai/api/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {OPENROUTER_API_KEY}",
            "-d", payload
        ], capture_output=True, text=True, timeout=25)

        if r.returncode != 0:
            return None

        resp = json.loads(r.stdout)
        content = resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        content = re.sub(r'^```(?:json)?\s*', '', content.strip())
        content = re.sub(r'\s*```$', '', content.strip())
        return json.loads(content)
    except Exception as e:
        print(f"  [warn] LLM extraction failed: {e}", file=sys.stderr)
        return None


def heuristic_extract(text: str, source: str) -> dict:
    """Fallback regex/heuristic extraction."""
    people = []
    for m in re.finditer(r'(?:From|To|Cc):\s*(?:"?([A-Z][a-z]+ [A-Z][a-z]+)"?\s*<?)', text):
        name = m.group(1).strip()
        if name and name not in [p["name"] for p in people]:
            people.append({"name": name, "role": "", "org": ""})

    for m in re.finditer(r'([A-Z][a-z]{1,15} [A-Z][a-z]{1,20})', text):
        name = m.group(1)
        skip = {"Action Items", "Next Steps", "Best Regards", "Kind Regards",
                "Thank You", "Hi There", "Dear Sir", "Dear Madam", "No Subject",
                "Project Alpha", "Quick Update", "Hi Team", "Hello Team",
                "Good Morning", "Good Afternoon", "Good Evening",
                "Sent From", "Warm Regards", "Many Thanks", "Looking Forward"}
        if (name not in [p["name"] for p in people] and name not in skip
                and not name.startswith(("Hi ", "Hello ", "Dear "))):
            people.append({"name": name, "role": "", "org": ""})

    action_items = []
    for m in re.finditer(r'(?:TODO|ACTION|TASK|→|\-\s*\[[ x]\])[:.]?\s*(.+)', text, re.IGNORECASE):
        action_items.append({
            "task": m.group(1).strip()[:200],
            "owner": "",
            "deadline": "",
            "status": "open"
        })

    return {
        "people": people[:10],
        "projects": [],
        "decisions": [],
        "action_items": action_items[:10],
    }


def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\s\-]', '', name).strip().replace(' ', '-')[:80]


def ensure_dirs():
    VAULT_BASE.mkdir(parents=True, exist_ok=True)
    for d in DIRS:
        (VAULT_BASE / d).mkdir(exist_ok=True)


def write_person_note(person: dict, source: str, interaction: str):
    name = person.get("name", "").strip()
    if not name or len(name) < 2:
        return
    fname = sanitize_filename(name) + ".md"
    fpath = VAULT_BASE / "People" / fname

    if fpath.exists():
        with open(fpath, "a") as f:
            f.write(f"\n\n## {NOW} — {source}\n")
            f.write(f"{interaction}\n")
        print(f"  Updated person: {name}")
    else:
        role = person.get("role", "")
        org = person.get("org", "")
        content = f"""---
name: "{name}"
type: person
role: "{role}"
org: "{org}"
tags: [person, knowledge-graph]
created: {TODAY}
---

# {name}

| Field | Value |
|-------|-------|
| Role  | {role or 'Unknown'} |
| Org   | {org or 'Unknown'} |

## Interactions

### {NOW} — {source}
{interaction}
"""
        with open(fpath, "w") as f:
            f.write(content)
        print(f"  Created person: {name}")


def write_project_note(project: dict, source: str):
    name = project.get("name", "").strip()
    if not name or len(name) < 2:
        return
    fname = sanitize_filename(name) + ".md"
    fpath = VAULT_BASE / "Projects" / fname

    if fpath.exists():
        with open(fpath, "a") as f:
            f.write(f"\n\n### {NOW} — {source}\n")
            f.write(f"Status: {project.get('status', 'N/A')}\n")
            f.write(f"{project.get('description', '')}\n")
        print(f"  Updated project: {name}")
    else:
        content = f"""---
name: "{name}"
type: project
status: "{project.get('status', 'active')}"
tags: [project, knowledge-graph]
created: {TODAY}
---

# {name}

{project.get('description', '')}

## Updates

### {NOW} — {source}
Status: {project.get('status', 'N/A')}
"""
        with open(fpath, "w") as f:
            f.write(content)
        print(f"  Created project: {name}")


def write_decision_note(decision: dict, source: str):
    summary = decision.get("summary", "").strip()
    if not summary or len(summary) < 3:
        return
    fname = f"{TODAY}-{sanitize_filename(summary)}.md"
    fpath = VAULT_BASE / "Decisions" / fname

    if fpath.exists():
        return

    owners = decision.get("owners", [])
    owner_links = ", ".join(f"[[{o}]]" for o in owners) if owners else "Unknown"

    content = f"""---
type: decision
date: {TODAY}
owners: {json.dumps(owners)}
tags: [decision, knowledge-graph]
source: "{source}"
---

# Decision: {summary}

- **Date:** {TODAY}
- **Owners:** {owner_links}
- **Source:** {source}

## Context
{decision.get('context', 'Extracted from: ' + source)}
"""
    with open(fpath, "w") as f:
        f.write(content)
    print(f"  Created decision: {summary[:60]}")


def write_action_item(item: dict, source: str):
    task = item.get("task", "").strip()
    if not task or len(task) < 3:
        return
    fname = f"{TODAY}-{sanitize_filename(task)}.md"
    fpath = VAULT_BASE / "Action-Items" / fname

    if fpath.exists():
        return

    owner = item.get("owner", "Unknown")
    owner_link = f"[[{owner}]]" if owner and owner != "Unknown" else "Unknown"

    content = f"""---
type: action-item
date: {TODAY}
owner: "{owner}"
deadline: "{item.get('deadline', '')}"
status: "{item.get('status', 'open')}"
tags: [action-item, knowledge-graph]
source: "{source}"
---

# {task}

- **Owner:** {owner_link}
- **Deadline:** {item.get('deadline', 'Not set')}
- **Status:** {item.get('status', 'open')}
- **Source:** {source}
"""
    with open(fpath, "w") as f:
        f.write(content)
    print(f"  Created action item: {task[:60]}")


def write_index():
    counts = {}
    for d in DIRS:
        p = VAULT_BASE / d
        counts[d] = len(list(p.glob("*.md"))) if p.exists() else 0

    recent_files = []
    for d in DIRS:
        p = VAULT_BASE / d
        if p.exists():
            for f in sorted(p.glob("*.md"), key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                recent_files.append((d, f.stem, f.stat().st_mtime))

    recent_files.sort(key=lambda x: x[2], reverse=True)

    content = f"""---
type: index
updated: {NOW}
tags: [knowledge-graph, index]
---

# Knowledge Graph Index

> Auto-generated on {NOW}

## Overview

| Category | Count |
|----------|-------|
| People | {counts.get('People', 0)} |
| Projects | {counts.get('Projects', 0)} |
| Decisions | {counts.get('Decisions', 0)} |
| Action Items | {counts.get('Action-Items', 0)} |

## Recent Activity

"""
    for category, name, mtime in recent_files[:10]:
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
        content += f"- `{ts}` — [[{name}]] ({category})\n"

    if not recent_files:
        content += "_No entries yet._\n"

    with open(VAULT_BASE / "_index.md", "w") as f:
        f.write(content)
    print("  Updated _index.md")


def fetch_emails() -> list:
    print("Fetching emails...")
    raw = run_cmd([GOG_CMD, "gmail", "search", "is:unread newer_than:1d", "-j"])
    if not raw:
        print("  No emails or gog not available.")
        return []

    emails = []
    current = {}
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            if current:
                emails.append(current)
                current = {}
            continue
        if line.startswith('Subject:'):
            current['subject'] = line[8:].strip()
        elif line.startswith('From:'):
            current['from'] = line[5:].strip()
        elif line.startswith('Date:'):
            current['date'] = line[5:].strip()
        elif line.startswith('To:'):
            current['to'] = line[3:].strip()
        elif line.startswith('Snippet:') or line.startswith('Body:'):
            current['body'] = line.split(':', 1)[1].strip()
        else:
            if 'body' in current:
                current['body'] += ' ' + line
            elif current:
                current.setdefault('body', '')
                current['body'] += line

    if current:
        emails.append(current)

    if not emails and raw:
        emails = [{"subject": "Batch", "body": raw, "from": "", "date": TODAY}]

    print(f"  Found {len(emails)} email(s)")
    return emails


def fetch_calendar() -> list:
    print("Fetching calendar events...")
    raw = run_cmd([GOG_CMD, "cal", "events", "-j"])
    if not raw:
        print("  No events or gog not available.")
        return []

    events = []
    current = {}
    for line in raw.split('\n'):
        line = line.strip()
        if not line:
            if current:
                events.append(current)
                current = {}
            continue
        if line.startswith('Title:') or line.startswith('Summary:'):
            current['title'] = line.split(':', 1)[1].strip()
        elif line.startswith('Start:') or line.startswith('When:'):
            current['start'] = line.split(':', 1)[1].strip()
        elif line.startswith('Attendees:'):
            current['attendees'] = line.split(':', 1)[1].strip()
        elif line.startswith('Description:') or line.startswith('Notes:'):
            current['description'] = line.split(':', 1)[1].strip()
        else:
            if current:
                current.setdefault('description', '')
                current['description'] += ' ' + line

    if current:
        events.append(current)

    if not events and raw:
        events = [{"title": "Batch", "description": raw, "start": TODAY}]

    print(f"  Found {len(events)} event(s)")
    return events


def process_content(text: str, source: str):
    if not text or len(text.strip()) < 10:
        return

    entities = llm_extract(text)
    if entities:
        print(f"  Using LLM extraction for: {source[:50]}")
    else:
        entities = heuristic_extract(text, source)
        print(f"  Using heuristic extraction for: {source[:50]}")

    interaction_summary = text[:300].replace('\n', ' ')

    for person in entities.get("people", []):
        write_person_note(person, source, interaction_summary)

    for project in entities.get("projects", []):
        write_project_note(project, source)

    for decision in entities.get("decisions", []):
        write_decision_note(decision, source)

    for item in entities.get("action_items", []):
        write_action_item(item, source)


def fetch_daily_memory() -> list:
    """Fetch today's and yesterday's memory files."""
    print("Fetching daily memory files...")
    memory_dir = Path("/home/clawdbot/clawd/memory")
    entries = []
    for days_ago in range(2):  # today + yesterday
        d = date.today() - __import__('datetime').timedelta(days=days_ago)
        f = memory_dir / f"{d.isoformat()}.md"
        if f.exists():
            content = f.read_text(encoding='utf-8')
            if content.strip():
                entries.append({"source": f"Memory: {d.isoformat()}", "text": content})
                print(f"  Found {f.name} ({len(content)} chars)")
    if not entries:
        print("  No recent memory files.")
    return entries


def fetch_obsidian_recent() -> list:
    """Fetch recently modified Obsidian vault notes (last 24h)."""
    print("Fetching recent Obsidian notes...")
    vault = Path("/home/clawdbot/obsidian-vault")
    entries = []
    cutoff = datetime.now().timestamp() - 86400  # 24h ago
    for md in vault.rglob("*.md"):
        # Skip Knowledge-Graph dir itself to avoid feedback loop
        if "Knowledge-Graph" in str(md):
            continue
        if md.stat().st_mtime > cutoff:
            content = md.read_text(encoding='utf-8')
            if content.strip() and len(content) > 50:
                rel = md.relative_to(vault)
                entries.append({"source": f"Obsidian: {rel}", "text": content[:3000]})
                print(f"  Found {rel} ({len(content)} chars)")
    if not entries:
        print("  No recently modified notes.")
    return entries


def fetch_subagent_results() -> list:
    """Fetch recent sub-agent session transcripts (last 24h)."""
    print("Fetching recent sub-agent results...")
    agents_dir = Path("/home/clawdbot/.openclaw/agents")
    entries = []
    cutoff = datetime.now().timestamp() - 86400
    if not agents_dir.exists():
        print("  No agents directory.")
        return entries
    for session_file in agents_dir.rglob("*.jsonl"):
        if session_file.stat().st_mtime > cutoff and "deleted" not in session_file.name:
            try:
                lines = session_file.read_text(encoding='utf-8').strip().split('\n')
                # Get last assistant message as summary
                for line in reversed(lines):
                    try:
                        msg = json.loads(line)
                        if msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                            if content and len(content) > 50:
                                agent_name = session_file.parent.parent.name
                                entries.append({
                                    "source": f"Agent: {agent_name}/{session_file.stem[:12]}",
                                    "text": content[:2000]
                                })
                                break
                    except json.JSONDecodeError:
                        continue
            except Exception:
                continue
    print(f"  Found {len(entries)} recent session(s)")
    return entries[:20]  # Cap at 20 to avoid overload


def fetch_memory_md() -> list:
    """Extract key sections from MEMORY.md."""
    print("Checking MEMORY.md for updates...")
    mem_file = Path("/home/clawdbot/clawd/MEMORY.md")
    if not mem_file.exists():
        return []
    content = mem_file.read_text(encoding='utf-8')
    # Only process if modified in last 24h
    if mem_file.stat().st_mtime < datetime.now().timestamp() - 86400:
        print("  MEMORY.md not recently modified.")
        return []
    print(f"  MEMORY.md found ({len(content)} chars)")
    return [{"source": "MEMORY.md", "text": content[:5000]}]


def main():
    print(f"=== Knowledge Graph Update: {NOW} ===")
    ensure_dirs()

    # Gather all activity sources (no email)
    all_content = []
    all_content.extend(fetch_daily_memory())
    all_content.extend(fetch_obsidian_recent())
    all_content.extend(fetch_subagent_results())
    all_content.extend(fetch_memory_md())

    # Calendar still useful
    events = fetch_calendar()
    for event in events:
        title = event.get("title", "Untitled Event")
        desc = event.get("description", "")
        attendees = event.get("attendees", "")
        full_text = f"Event: {title}\nAttendees: {attendees}\n{desc}"
        all_content.append({"source": f"Calendar: {title}", "text": full_text})

    if not all_content:
        print("No new data to process.")
        write_index()
        print("Done.")
        return

    # Batch all content into a single LLM call instead of one per source
    print(f"\nBatching {len(all_content)} source(s) into single LLM extraction...")
    combined_text = ""
    for item in all_content[:15]:  # Cap at 15 sources to stay within token limits
        combined_text += f"\n\n--- Source: {item['source']} ---\n{item['text'][:500]}"

    process_content(combined_text, "batch")

    write_index()
    print(f"\n=== Complete ===")


if __name__ == "__main__":
    main()
