#!/usr/bin/env python3
"""
session_remember.py — Arthur's Session Learning Miner
Inspired by Arscontexta's /remember command.

Mines the current session for:
1. New lessons learned
2. Decisions made
3. TODOs / pending tasks
4. Key facts to remember
5. New connections between existing knowledge

Writes to:
- memory/YYYY-MM-DD.md (daily log)
- MEMORY.md (Lessons Learned section)
- obsidian-vault/KnowledgeGraph/Action-Items/ (new TODOs)
- obsidian-vault/KnowledgeGraph/Decisions/ (new decisions)

Usage:
    python3 tools/session_remember.py "session summary or notes"
    python3 tools/session_remember.py --stdin   (read from stdin)
    python3 tools/session_remember.py --auto    (scan today's memory file)
"""

import os
import re
import sys
import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from argparse import ArgumentParser

BKK = timezone(timedelta(hours=7))
NOW = datetime.now(BKK)
TODAY = NOW.strftime("%Y-%m-%d")
TIME_STR = NOW.strftime("%H:%M")

WORKSPACE = Path("/home/clawdbot/clawd")
MEMORY_DIR = WORKSPACE / "memory"
MEMORY_MD = WORKSPACE / "MEMORY.md"
VAULT = Path("/home/clawdbot/obsidian-vault")
KG = VAULT / "KnowledgeGraph"


def load_api_key():
    try:
        with open(Path.home() / ".openclaw/openclaw.json") as f:
            cfg = json.load(f)
        return (cfg.get("models", {}).get("providers", {})
                .get("openrouter", {}).get("apiKey", ""))
    except Exception:
        return ""


def llm_extract(text, api_key):
    """Use LLM to extract structured learnings from session text."""
    prompt = f"""You are Arthur, an AI assistant. Analyze this session activity and extract key learnings.

Return ONLY valid JSON with this exact structure:
{{
  "lessons": [
    {{"text": "lesson description", "category": "workflow|technical|communication|research"}}
  ],
  "decisions": [
    {{"text": "decision made", "context": "brief context", "rationale": "why"}}
  ],
  "todos": [
    {{"text": "task description", "priority": "high|medium|low", "project": "project name or general"}}
  ],
  "facts": [
    {{"text": "fact to remember", "category": "system|person|project|config"}}
  ],
  "connections": [
    {{"text": "new insight connecting existing knowledge"}}
  ]
}}

Rules:
- Only extract things genuinely NEW and worth remembering
- Skip trivial observations
- Be specific and actionable
- Max 5 items per category
- If nothing worth capturing in a category, use empty array []

Session activity:
{text[:5000]}"""

    payload = json.dumps({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 1200,
    })

    try:
        r = subprocess.run([
            "curl", "-s", "-X", "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {api_key}",
            "-d", payload
        ], capture_output=True, text=True, timeout=120)
        resp = json.loads(r.stdout)
        raw = resp["choices"][0]["message"]["content"].strip()
        # Strip markdown code blocks if present
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw.strip())
        return json.loads(raw)
    except Exception as e:
        print(f"[remember] LLM extraction failed: {e}", file=sys.stderr)
        return None


def heuristic_extract(text):
    """Fallback: simple regex-based extraction."""
    lessons = []
    decisions = []
    todos = []
    facts = []

    patterns = {
        "lesson": re.compile(
            r"(?:lesson|learned?|remember|important|note:|tip:|warning:|don't|never|always|fixed?|issue)[:\s]+(.{20,250})",
            re.IGNORECASE
        ),
        "decision": re.compile(
            r"(?:decided?|confirmed?|agreed?|settled|chose|switching|using|set to)[:\s]+(.{20,200})",
            re.IGNORECASE
        ),
        "todo": re.compile(
            r"(?:TODO|FIXME|pending|need to|should|will|next step|remaining)[:\s]+(.{10,150})",
            re.IGNORECASE
        ),
        "fact": re.compile(
            r"(?:config|path|url|key|token|id|version|installed)[:\s]+(.{10,150})",
            re.IGNORECASE
        ),
    }

    for m in patterns["lesson"].finditer(text):
        lessons.append({"text": m.group(1).strip().split('\n')[0][:200], "category": "general"})
    for m in patterns["decision"].finditer(text):
        decisions.append({"text": m.group(1).strip().split('\n')[0][:200], "context": "", "rationale": ""})
    for m in patterns["todo"].finditer(text):
        todos.append({"text": m.group(1).strip().split('\n')[0][:150], "priority": "medium", "project": "general"})
    for m in patterns["fact"].finditer(text):
        facts.append({"text": m.group(1).strip().split('\n')[0][:150], "category": "config"})

    return {
        "lessons": list({l["text"]: l for l in lessons}.values())[:5],
        "decisions": list({d["text"]: d for d in decisions}.values())[:3],
        "todos": list({t["text"]: t for t in todos}.values())[:5],
        "facts": list({f["text"]: f for f in facts}.values())[:3],
        "connections": [],
    }


def append_to_daily_log(extractions):
    """Write extracted learnings to today's memory file."""
    daily_file = MEMORY_DIR / f"{TODAY}.md"
    MEMORY_DIR.mkdir(exist_ok=True)

    sections = []

    if extractions.get("lessons"):
        sections.append("### 🧠 Session Learnings")
        for l in extractions["lessons"]:
            cat = l.get("category", "general")
            sections.append(f"- [{cat}] {l['text']}")

    if extractions.get("decisions"):
        sections.append("\n### ✅ Decisions Made")
        for d in extractions["decisions"]:
            ctx = f" ({d['context']})" if d.get("context") else ""
            sections.append(f"- {d['text']}{ctx}")

    if extractions.get("todos"):
        sections.append("\n### 📋 New TODOs")
        for t in extractions["todos"]:
            pri = t.get("priority", "medium")
            sections.append(f"- [{pri.upper()}] {t['text']}")

    if extractions.get("facts"):
        sections.append("\n### 📌 Facts to Remember")
        for f in extractions["facts"]:
            sections.append(f"- {f['text']}")

    if extractions.get("connections"):
        sections.append("\n### 🔗 New Connections")
        for c in extractions["connections"]:
            sections.append(f"- {c['text']}")

    if not sections:
        print("[remember] Nothing worth capturing.")
        return

    entry = f"\n\n---\n## /remember — {TODAY} {TIME_STR} BKK\n\n"
    entry += "\n".join(sections) + "\n"

    with open(daily_file, "a", encoding="utf-8") as f:
        f.write(entry)

    print(f"[remember] ✅ Written to {daily_file.name}")
    return entry


def update_memory_md_lessons(extractions):
    """Add new lessons to MEMORY.md Lessons Learned section."""
    if not extractions.get("lessons"):
        return

    content = MEMORY_MD.read_text(encoding="utf-8") if MEMORY_MD.exists() else ""
    new_entries = []

    for l in extractions["lessons"]:
        text = l["text"].strip()
        if len(text) > 20 and text[:50] not in content:
            new_entries.append(f"- **{TODAY}:** {text[:250]}")

    if not new_entries:
        print("[remember] Lessons already in MEMORY.md")
        return

    marker = "## 💡 Lessons Learned"
    if marker in content:
        insert_pos = content.index(marker) + len(marker)
        insert_pos = content.index("\n", insert_pos) + 1
        block = "\n".join(new_entries) + "\n"
        content = content[:insert_pos] + block + content[insert_pos:]
    else:
        content += f"\n\n{marker}\n\n" + "\n".join(new_entries) + "\n"

    MEMORY_MD.write_text(content, encoding="utf-8")
    print(f"[remember] ✅ Added {len(new_entries)} lesson(s) to MEMORY.md")


def write_decision_notes(extractions):
    """Create decision notes in Obsidian KnowledgeGraph/Decisions/."""
    if not extractions.get("decisions"):
        return

    decisions_dir = KG / "Decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)

    for d in extractions["decisions"]:
        text = d["text"].strip()
        if len(text) < 10:
            continue
        # Sanitize filename
        slug = re.sub(r'[^\w\s-]', '', text[:50]).strip().replace(' ', '-').lower()
        fname = f"{TODAY}-{slug}.md"
        fpath = decisions_dir / fname

        if fpath.exists():
            continue  # Don't overwrite

        note = f"""# Decision: {text[:100]}

**Date:** {TODAY}
**Context:** {d.get('context', 'Session activity')}
**Rationale:** {d.get('rationale', 'See session log')}

## Details
{text}

## Tags
#decision #{TODAY}

## Links
- [[Daily/{TODAY}]]
"""
        fpath.write_text(note, encoding="utf-8")
        print(f"[remember] ✅ Decision note: {fname}")


def write_todo_notes(extractions):
    """Append new TODOs to KnowledgeGraph/Action-Items/Pending.md."""
    if not extractions.get("todos"):
        return

    ai_dir = KG / "Action-Items"
    ai_dir.mkdir(parents=True, exist_ok=True)
    pending_file = ai_dir / "Pending.md"

    # Read existing to avoid duplicates
    existing = pending_file.read_text(encoding="utf-8") if pending_file.exists() else ""

    new_todos = []
    for t in extractions["todos"]:
        text = t["text"].strip()
        if len(text) < 5 or text[:30] in existing:
            continue
        pri = t.get("priority", "medium").upper()
        proj = t.get("project", "general")
        new_todos.append(f"- [ ] [{pri}] {text} — *added {TODAY}* #todo #{proj.replace(' ', '-').lower()}")

    if not new_todos:
        return

    # Ensure header exists
    if not pending_file.exists():
        pending_file.write_text("# Pending Action Items\n\nAuto-maintained by session_remember.py\n\n", encoding="utf-8")

    with open(pending_file, "a", encoding="utf-8") as f:
        f.write(f"\n<!-- {TODAY} {TIME_STR} -->\n")
        f.write("\n".join(new_todos) + "\n")

    print(f"[remember] ✅ Added {len(new_todos)} TODO(s) to Action-Items/Pending.md")


def git_commit_vault():
    """Commit vault changes."""
    try:
        subprocess.run(["git", "-C", str(VAULT), "add", "-A"],
                       capture_output=True, timeout=30)
        result = subprocess.run(
            ["git", "-C", str(VAULT), "commit", "-m",
             f"session_remember {TODAY} {TIME_STR}: mined session learnings"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            subprocess.run(["git", "-C", str(VAULT), "push"],
                           capture_output=True, timeout=60)
            print("[remember] ✅ Vault committed and pushed")
        else:
            print("[remember] Nothing new to commit in vault")
    except Exception as e:
        print(f"[remember] Git error: {e}", file=sys.stderr)


def summarize_results(extractions):
    """Print a clean summary of what was captured."""
    total = sum(len(v) for v in extractions.values() if isinstance(v, list))
    if total == 0:
        print("\n[remember] Nothing new to capture from this session.")
        return

    print("\n=== 📚 Session Learnings Captured ===")
    cats = [
        ("lessons", "🧠 Lessons"),
        ("decisions", "✅ Decisions"),
        ("todos", "📋 TODOs"),
        ("facts", "📌 Facts"),
        ("connections", "🔗 Connections"),
    ]
    for key, label in cats:
        items = extractions.get(key, [])
        if items:
            print(f"\n{label} ({len(items)}):")
            for item in items:
                text = item.get("text", str(item))[:100]
                print(f"  • {text}")
    print()


def main():
    parser = ArgumentParser(description="Arthur's session learning miner")
    parser.add_argument("text", nargs="?", help="Session text to analyze")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-scan today's memory file")
    parser.add_argument("--no-git", action="store_true",
                        help="Skip git commit")
    args = parser.parse_args()

    # Get input text
    if args.stdin:
        text = sys.stdin.read()
    elif args.auto:
        daily_file = MEMORY_DIR / f"{TODAY}.md"
        if not daily_file.exists():
            print(f"[remember] No memory file for {TODAY}")
            return
        text = daily_file.read_text(encoding="utf-8")
        # Skip if already ran /remember today
        if "/remember —" in text:
            lines = text.split("\n")
            # Find last /remember block and only process content after it
            last_idx = 0
            for i, line in enumerate(lines):
                if "/remember —" in line:
                    last_idx = i
            text = "\n".join(lines[last_idx + 1:])
            if len(text.strip()) < 100:
                print("[remember] Not enough new content since last /remember")
                return
    elif args.text:
        text = args.text
    else:
        print("Usage: session_remember.py <text> | --stdin | --auto")
        sys.exit(1)

    if len(text.strip()) < 50:
        print("[remember] Input too short to extract learnings.")
        return

    print(f"=== /remember — {TODAY} {TIME_STR} BKK ===")
    print(f"[remember] Analyzing {len(text)} chars of session activity...")

    # Extract learnings
    api_key = load_api_key()
    extractions = None

    if api_key:
        print("[remember] Using LLM extraction (DeepSeek)...")
        extractions = llm_extract(text, api_key)

    if not extractions:
        print("[remember] Using heuristic extraction...")
        extractions = heuristic_extract(text)

    # Show summary
    summarize_results(extractions)

    # Write to all destinations
    append_to_daily_log(extractions)
    update_memory_md_lessons(extractions)
    write_decision_notes(extractions)
    write_todo_notes(extractions)

    # Git commit
    if not args.no_git:
        git_commit_vault()

    print("=== /remember complete ===")


if __name__ == "__main__":
    main()
