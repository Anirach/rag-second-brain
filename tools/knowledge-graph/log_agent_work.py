#!/usr/bin/env python3
"""
Agent Work Logger — Scans recent agent sessions and writes work records
to each team's Obsidian folder.
"""
import json
import os
import re
from datetime import datetime, date, timedelta
from pathlib import Path

VAULT = Path("/home/clawdbot/obsidian-vault")
AGENTS_DIR = Path("/home/clawdbot/.openclaw/agents")
TODAY = date.today().isoformat()
NOW = datetime.now().strftime("%Y-%m-%d %H:%M")
CUTOFF = datetime.now().timestamp() - 86400  # Last 24h

# Map agent IDs to teams and Obsidian folders
TEAM_MAP = {
    # Writing Team
    "book-architect": ("Writing", "Book Architect"),
    "book-compiler": ("Writing", "Book Compiler"),
    "book-orchestrator": ("Writing", "Book Orchestrator"),
    "research-lead": ("Writing", "Research Lead"),
    "writer": ("Writing", "Chapter Writer"),
    "editor": ("Writing", "Dev Editor"),
    "publisher": ("Writing", "Publisher"),
    "fact-checker": ("Writing", "Fact Checker"),
    # Academic Paper Team
    "paper-architect": ("Academic", "Paper Architect"),
    "methodology-expert": ("Academic", "Methodology Expert"),
    "technical-writer": ("Academic", "Technical Writer"),
    "peer-reviewer": ("Academic", "Peer Reviewer"),
    "format-editor": ("Academic", "Format Editor"),
    "data-analyst": ("Academic", "Data Analyst"),
    "researcher": ("Academic", "Researcher"),
    # Translation Team
    "thai-translator": ("Translation", "Thai Translator"),
    "translation-architect": ("Translation", "Translation Architect"),
    "document-formatter": ("Translation", "Document Formatter"),
    # Course Team
    "course-architect": ("Course", "Course Architect"),
    # Coding Team
    "orchestrator": ("Coding", "Orchestrator"),
    "code-review": ("Coding", "Code Review"),
    "pr-agent": ("Coding", "PR Agent"),
    "testing": ("Coding", "Testing"),
    "docs": ("Coding", "Docs"),
    "ux-designer": ("Coding", "UX Designer"),
}

def get_openrouter_key():
    try:
        with open(os.path.expanduser("~/.openclaw/openclaw.json")) as f:
            c = json.load(f)
        return c.get("models", {}).get("providers", {}).get("openrouter", {}).get("apiKey", "")
    except Exception:
        return ""

def llm_summarize(text, agent_name):
    """Use LLM to create a brief work summary."""
    import subprocess
    key = get_openrouter_key()
    if not key:
        # Fallback: first 200 chars
        return text[:200].replace('\n', ' ').strip()
    
    prompt = f"Summarize this agent work session in 1-2 sentences. Agent: {agent_name}. Output ONLY the summary, no preamble.\n\n{text[:3000]}"
    
    payload = json.dumps({
        "model": "deepseek/deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.3
    })
    
    try:
        result = subprocess.run(
            ["curl", "-s", "https://openrouter.ai/api/v1/chat/completions",
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-d", payload],
            capture_output=True, text=True, timeout=30
        )
        resp = json.loads(result.stdout)
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return text[:200].replace('\n', ' ').strip()


def scan_sessions():
    """Scan recent agent sessions and extract work records."""
    records = []
    
    for agent_dir in AGENTS_DIR.iterdir():
        if not agent_dir.is_dir():
            continue
        agent_id = agent_dir.name
        if agent_id == "main":
            continue  # Skip main agent
        
        sessions_dir = agent_dir / "sessions"
        if not sessions_dir.exists():
            continue
        
        for session_file in sessions_dir.glob("*.jsonl"):
            if "deleted" in session_file.name:
                continue
            if session_file.stat().st_mtime < CUTOFF:
                continue
            
            try:
                lines = session_file.read_text(encoding='utf-8').strip().split('\n')
                
                # Get the task (first user message) and result (last assistant message)
                task = ""
                result = ""
                for line in lines:
                    try:
                        msg = json.loads(line)
                        if msg.get("role") == "user" and not task:
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                            if isinstance(content, str) and len(content) > 20:
                                task = content[:500]
                        elif msg.get("role") == "assistant":
                            content = msg.get("content", "")
                            if isinstance(content, list):
                                content = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                            if isinstance(content, str) and len(content) > 20:
                                result = content[:2000]
                    except json.JSONDecodeError:
                        continue
                
                if task or result:
                    mtime = datetime.fromtimestamp(session_file.stat().st_mtime)
                    records.append({
                        "agent_id": agent_id,
                        "session": session_file.stem[:12],
                        "timestamp": mtime.strftime("%Y-%m-%d %H:%M"),
                        "date": mtime.strftime("%Y-%m-%d"),
                        "task": task,
                        "result": result,
                    })
            except Exception:
                continue
    
    return records


def write_team_logs(records):
    """Write work records to each team's Obsidian folder."""
    # Group by team
    team_records = {}
    for record in records:
        agent_id = record["agent_id"]
        if agent_id in TEAM_MAP:
            team, role = TEAM_MAP[agent_id]
        else:
            team, role = "Other", agent_id
        
        if team not in team_records:
            team_records[team] = []
        record["role"] = role
        team_records[team].append(record)
    
    for team, recs in team_records.items():
        team_dir = VAULT / "Agents" / team
        if not team_dir.exists():
            team_dir.mkdir(parents=True, exist_ok=True)
        
        # Work log file — one per date
        log_file = team_dir / f"Work-Log-{TODAY}.md"
        
        # Build content
        entries = []
        for r in sorted(recs, key=lambda x: x["timestamp"]):
            summary = llm_summarize(r["result"] or r["task"], r["role"])
            entries.append(f"""### {r['timestamp']} — {r['role']}
**Task:** {r['task'][:200].replace(chr(10), ' ')}
**Result:** {summary}
**Session:** `{r['session']}`
""")
        
        content = f"""---
type: work-log
team: {team}
date: {TODAY}
updated: {NOW}
tags: [work-log, {team.lower().replace(' ', '-')}]
---

# {team} Team — Work Log {TODAY}

{chr(10).join(entries)}

## Related
- [[Agents/{team} Team|{team} Team Overview]]
- [[Home|← Home]]
"""
        
        # Append if exists, create if not
        if log_file.exists():
            existing = log_file.read_text()
            # Append new entries only
            for entry in entries:
                session_id = entry.split("`")[1] if "`" in entry else ""
                if session_id and session_id not in existing:
                    existing = existing.rstrip() + "\n\n" + entry
            log_file.write_text(existing)
        else:
            log_file.write_text(content)
        
        print(f"  📝 {team}: {len(recs)} entries → {log_file.name}")
    
    return team_records


def update_team_indexes(team_records):
    """Update each team's index with link to latest work log."""
    for team in team_records:
        team_dir = VAULT / "Agents" / team
        index_file = team_dir / "_work-logs.md"
        
        # Find all work log files
        logs = sorted(team_dir.glob("Work-Log-*.md"), reverse=True)
        
        log_links = "\n".join(
            f"- [[Agents/{team}/{f.stem}|{f.stem}]]" for f in logs[:30]
        )
        
        index_file.write_text(f"""---
type: index
team: {team}
updated: {NOW}
tags: [index, work-log, {team.lower().replace(' ', '-')}]
---

# {team} Team — Work Logs

{log_links}

## Related
- [[Agents/{team} Team|{team} Team Overview]]
- [[Home|← Home]]
""")


def git_push():
    """Commit and push changes."""
    import subprocess
    os.chdir(str(VAULT))
    subprocess.run(["git", "add", "-A"], capture_output=True)
    result = subprocess.run(
        ["git", "commit", "-m", f"Agent work logs: {TODAY}"],
        capture_output=True, text=True
    )
    if "nothing to commit" in result.stdout:
        print("  No changes to commit.")
        return
    subprocess.run(["git", "push"], capture_output=True)
    print("  ✅ Pushed to GitHub")


def main():
    print(f"=== Agent Work Logger: {NOW} ===")
    
    records = scan_sessions()
    print(f"Found {len(records)} recent session(s)")
    
    if not records:
        print("No new work to log.")
        return
    
    team_records = write_team_logs(records)
    update_team_indexes(team_records)
    git_push()
    
    print(f"\n=== Complete: logged work for {len(team_records)} team(s) ===")


if __name__ == "__main__":
    main()
