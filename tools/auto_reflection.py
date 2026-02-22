#!/usr/bin/env python3
"""
Auto-Reflection Script — Arthur's Daily Self-Learning Cycle
Inspired by TIGER Agent's auto-reflection concept.

Runs daily to:
1. Review recent daily memory files (last 24-48h)
2. Extract key learnings, decisions, lessons
3. Update MEMORY.md with distilled insights
4. Prune stale/outdated entries
5. Update running-log.md with activity summary
6. Git commit the vault

Schedule: Daily at 02:00 Bangkok time (via cron)
"""

import os
import re
import json
import subprocess
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

BKK = timezone(timedelta(hours=7))
NOW = datetime.now(BKK)
TODAY = NOW.strftime("%Y-%m-%d")
YESTERDAY = (NOW - timedelta(days=1)).strftime("%Y-%m-%d")

WORKSPACE = Path("/home/clawdbot/clawd")
MEMORY_DIR = WORKSPACE / "memory"
MEMORY_MD = WORKSPACE / "MEMORY.md"
RUNNING_LOG = MEMORY_DIR / "running-log.md"
VAULT = Path("/home/clawdbot/obsidian-vault")

OPENROUTER_KEY = ""
try:
    with open(Path.home() / ".openclaw/openclaw.json") as f:
        cfg = json.load(f)
    OPENROUTER_KEY = (cfg.get("models", {}).get("providers", {})
                      .get("openrouter", {}).get("apiKey", ""))
except Exception:
    pass


def read_file(path):
    try:
        return Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""


def write_file(path, content):
    Path(path).write_text(content, encoding="utf-8")


def llm_call(prompt, max_tokens=2000):
    """Call LLM via OpenRouter for reflection."""
    if not OPENROUTER_KEY:
        return None
    payload = json.dumps({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    })
    try:
        r = subprocess.run([
            "curl", "-s", "-X", "POST",
            "https://openrouter.ai/api/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-H", f"Authorization: Bearer {OPENROUTER_KEY}",
            "-d", payload
        ], capture_output=True, text=True, timeout=120)
        resp = json.loads(r.stdout)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"[reflection] LLM call failed: {e}")
        return None


def gather_recent_activity():
    """Read last 2 days of memory files."""
    texts = {}
    for days_ago in range(2):
        d = (NOW - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        f = MEMORY_DIR / f"{d}.md"
        if f.exists():
            content = f.read_text(encoding="utf-8")
            if content.strip():
                texts[d] = content
    return texts


def extract_lessons_heuristic(texts):
    """Extract lessons/decisions without LLM as fallback."""
    lessons = []
    decisions = []
    todos = []

    patterns = {
        "lesson": re.compile(r"(?:lesson|learned?|remember|don't|never|always|fix(?:ed)?)[:\s]+(.{20,200})", re.IGNORECASE),
        "decision": re.compile(r"(?:decided?|confirmed?|agreed?|settled)[:\s]+(.{20,200})", re.IGNORECASE),
        "todo": re.compile(r"(?:TODO|pending|needs?|still|remaining)[:\s]+(.{20,150})", re.IGNORECASE),
    }

    for date_str, text in texts.items():
        for m in patterns["lesson"].finditer(text):
            lessons.append(m.group(1).strip().split('\n')[0])
        for m in patterns["decision"].finditer(text):
            decisions.append(m.group(1).strip().split('\n')[0])
        for m in patterns["todo"].finditer(text):
            todos.append(m.group(1).strip().split('\n')[0])

    return {
        "lessons": list(set(lessons))[:5],
        "decisions": list(set(decisions))[:5],
        "todos": list(set(todos))[:5],
    }


def update_running_log(activity_texts, insights):
    """Append to running-log.md — raw activity record."""
    log_entry = f"\n---\n## {TODAY} ({NOW.strftime('%H:%M')} BKK)\n\n"

    # Activity summary
    if activity_texts:
        total_chars = sum(len(v) for v in activity_texts.items())
        log_entry += f"**Files reviewed:** {', '.join(activity_texts.keys())}\n\n"

    # Key insights
    if insights.get("lessons"):
        log_entry += "**Lessons:**\n"
        for l in insights["lessons"][:3]:
            log_entry += f"- {l[:120]}\n"
        log_entry += "\n"

    if insights.get("decisions"):
        log_entry += "**Decisions:**\n"
        for d in insights["decisions"][:3]:
            log_entry += f"- {d[:120]}\n"
        log_entry += "\n"

    # Ensure running log exists with header
    if not RUNNING_LOG.exists():
        RUNNING_LOG.write_text("# Running Log\n\nAuto-updated daily by auto_reflection.py\n", encoding="utf-8")

    with open(RUNNING_LOG, "a", encoding="utf-8") as f:
        f.write(log_entry)

    print(f"[reflection] Updated running-log.md")


def update_memory_md(insights):
    """Append new lessons/learnings to MEMORY.md's Lessons Learned section."""
    if not insights.get("lessons") and not insights.get("decisions"):
        print("[reflection] No new insights to add to MEMORY.md")
        return

    content = MEMORY_MD.read_text(encoding="utf-8") if MEMORY_MD.exists() else ""

    # Build new entries
    new_entries = []
    for lesson in insights.get("lessons", [])[:3]:
        lesson = lesson.strip()
        if lesson and len(lesson) > 20 and lesson not in content:
            new_entries.append(f"- **{TODAY}:** {lesson[:200]}")

    if not new_entries:
        print("[reflection] All insights already in MEMORY.md")
        return

    # Find Lessons Learned section and append
    lessons_marker = "## 💡 Lessons Learned"
    if lessons_marker in content:
        insert_pos = content.index(lessons_marker) + len(lessons_marker)
        # Find end of first line
        insert_pos = content.index("\n", insert_pos) + 1
        new_block = "\n".join(new_entries) + "\n"
        content = content[:insert_pos] + new_block + content[insert_pos:]
    else:
        content += f"\n\n{lessons_marker}\n\n" + "\n".join(new_entries) + "\n"

    MEMORY_MD.write_text(content, encoding="utf-8")
    print(f"[reflection] Added {len(new_entries)} new lesson(s) to MEMORY.md")


def update_last_reviewed():
    """Update the 'Last reviewed' date in MEMORY.md."""
    content = MEMORY_MD.read_text(encoding="utf-8") if MEMORY_MD.exists() else ""
    content = re.sub(
        r"\*Last reviewed: \d{4}-\d{2}-\d{2}\*",
        f"*Last reviewed: {TODAY}*",
        content
    )
    MEMORY_MD.write_text(content, encoding="utf-8")


def git_commit():
    """Commit changes to obsidian vault and workspace."""
    try:
        subprocess.run(
            ["git", "-C", str(VAULT), "add", "-A"],
            capture_output=True, timeout=30
        )
        result = subprocess.run(
            ["git", "-C", str(VAULT), "commit", "-m",
             f"Auto-reflection {TODAY}: daily memory distillation"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            subprocess.run(
                ["git", "-C", str(VAULT), "push"],
                capture_output=True, timeout=60
            )
            print("[reflection] Git committed and pushed")
        else:
            print("[reflection] Nothing new to commit")
    except Exception as e:
        print(f"[reflection] Git error: {e}")


def log_heartbeat_state():
    """Update heartbeat-state.json with last auto-reflection time."""
    state_file = MEMORY_DIR / "heartbeat-state.json"
    try:
        state = json.loads(state_file.read_text()) if state_file.exists() else {}
        state.setdefault("lastChecks", {})
        state["lastChecks"]["auto_reflection"] = int(NOW.timestamp())
        state["lastChecks"]["memory_maintenance"] = int(NOW.timestamp())
        state_file.write_text(json.dumps(state, indent=2))
        print("[reflection] Updated heartbeat-state.json")
    except Exception as e:
        print(f"[reflection] Heartbeat state error: {e}")


def main():
    print(f"=== Auto-Reflection: {NOW.strftime('%Y-%m-%d %H:%M')} BKK ===")

    # 1. Gather recent activity
    print("[reflection] Gathering recent memory files...")
    activity = gather_recent_activity()
    if not activity:
        print("[reflection] No recent activity to reflect on. Exiting.")
        return

    print(f"[reflection] Found {len(activity)} file(s): {', '.join(activity.keys())}")

    # 2. Extract insights (LLM if available, else heuristic)
    combined_text = "\n\n---\n\n".join(
        f"# {date}\n{text[:3000]}" for date, text in activity.items()
    )

    insights = None
    if OPENROUTER_KEY:
        print("[reflection] Using LLM for insight extraction...")
        prompt = f"""You are Arthur, an AI assistant reviewing your own activity logs for self-improvement.

Review these recent activity logs and extract:
1. Key lessons learned (things that went wrong and how to fix, or important discoveries)
2. Important decisions made
3. Patterns to remember for future sessions

Return JSON only:
{{
  "lessons": ["lesson 1", "lesson 2"],
  "decisions": ["decision 1"],
  "todos": ["pending task 1"]
}}

Activity logs:
{combined_text[:4000]}"""

        result = llm_call(prompt, max_tokens=800)
        if result:
            try:
                result = re.sub(r'^```(?:json)?\s*', '', result.strip())
                result = re.sub(r'\s*```$', '', result.strip())
                insights = json.loads(result)
                print(f"[reflection] LLM extracted {len(insights.get('lessons', []))} lessons")
            except Exception:
                pass

    if not insights:
        print("[reflection] Falling back to heuristic extraction...")
        insights = extract_lessons_heuristic(activity)

    # 3. Update running log
    update_running_log(activity, insights)

    # 4. Update MEMORY.md
    update_memory_md(insights)

    # 5. Update last reviewed date
    update_last_reviewed()

    # 6. Update heartbeat state
    log_heartbeat_state()

    # 7. Git commit
    git_commit()

    print(f"=== Auto-Reflection Complete ===")
    print(f"  Lessons: {len(insights.get('lessons', []))}")
    print(f"  Decisions: {len(insights.get('decisions', []))}")


if __name__ == "__main__":
    main()
