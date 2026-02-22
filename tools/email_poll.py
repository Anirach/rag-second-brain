#!/usr/bin/env python3
"""
email_poll.py - Near-realtime email alert system
Polls Gmail every 5 minutes for urgent/important emails.
Sends Telegram alerts for new matches.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
# Resolve workspace: use the path that contains our memory/ directory
def _find_workspace() -> Path:
    for candidate in [Path("/home/clawdbot/clawd"), Path("/workspace")]:
        if (candidate / "memory").exists():
            return candidate
    return Path("/workspace")

WORKSPACE = _find_workspace()
SEEN_FILE = WORKSPACE / "memory" / "email-seen.json"
HEARTBEAT_FILE = WORKSPACE / "memory" / "heartbeat-state.json"
TELEGRAM_CHAT_ID = "7579913696"

URGENT_KEYWORDS = [
    "urgent", "important", "deadline", "asap", "critical",
    "ด่วน", "สำคัญ", "conference", "review", "submission",
    "meeting", "invite",
]

# Sender domains/patterns that always trigger alert
PRIORITY_DOMAINS = [
    "@kmutnb.ac.th",
    "@fitm.kmutnb.ac.th",
]

# Known academic journals/conferences (partial match)
ACADEMIC_SENDERS = [
    "elsevier", "springer", "ieee", "acm.org", "wiley",
    "mdpi", "frontiersin", "nature.com", "science.org",
    "tandfonline", "sagepub", "acs.org", "rsc.org",
    "hindawi", "plos", "bmc", "scopus", "clarivate",
    "easychair", "cmt3", "openreview", "edas.",
    "conferencealert", "iaria", "aaai", "neurips", "icml",
    "cvpr", "iccv", "acl", "emnlp", "naacl",
]


def get_telegram_token() -> str:
    for config_path in [
        Path("/home/clawdbot/.openclaw/openclaw.json"),
        Path.home() / ".openclaw" / "openclaw.json",
    ]:
        if config_path.exists():
            with open(config_path) as f:
                config = json.load(f)
            return config["channels"]["telegram"]["botToken"]
    raise FileNotFoundError("openclaw.json not found")


def send_telegram_alert(sender: str, subject: str, token: str) -> bool:
    """Send a Telegram notification for an urgent email."""
    # Escape HTML special chars
    def esc(s):
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    text = f"📧 <b>New email alert</b>\n👤 <b>From:</b> {esc(sender)}\n📌 <b>Subject:</b> {esc(subject)}"
    result = subprocess.run(
        [
            "curl", "-s", "-X", "POST",
            f"https://api.telegram.org/bot{token}/sendMessage",
            "-d", f"chat_id={TELEGRAM_CHAT_ID}",
            "-d", f"text={text}",
            "-d", "parse_mode=HTML",
        ],
        capture_output=True, text=True, timeout=15,
    )
    return result.returncode == 0


def load_seen_ids() -> set:
    if SEEN_FILE.exists():
        try:
            data = json.loads(SEEN_FILE.read_text())
            return set(data.get("seen", []))
        except Exception:
            pass
    return set()


def save_seen_ids(seen: set) -> None:
    # Keep only last 1000 IDs to avoid unbounded growth
    seen_list = list(seen)[-1000:]
    SEEN_FILE.write_text(json.dumps({"seen": seen_list, "updated": int(time.time())}, indent=2))


def update_heartbeat_email_timestamp() -> None:
    try:
        if HEARTBEAT_FILE.exists():
            data = json.loads(HEARTBEAT_FILE.read_text())
        else:
            data = {}
        data.setdefault("lastChecks", {})
        data["lastChecks"]["email"] = int(time.time())
        HEARTBEAT_FILE.write_text(json.dumps(data, indent=2))
    except Exception:
        pass


def fetch_recent_emails() -> list[dict]:
    """Run gog gmail search and parse results into list of email dicts."""
    cmd = (
        "GOG_KEYRING_PASSWORD=openclaw "
        "GOG_ACCOUNT=anirach.m@fitm.kmutnb.ac.th "
        "gog gmail search 'is:unread newer_than:10m' --max 10 --plain 2>/dev/null"
    )
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=30
        )
        output = result.stdout.strip()
    except subprocess.TimeoutExpired:
        return []
    except Exception:
        return []

    if not output:
        return []

    emails = []
    current = {}

    for line in output.splitlines():
        line = line.strip()
        if not line:
            if current:
                emails.append(current)
                current = {}
            continue

        # Try to parse key: value pairs
        if line.lower().startswith("id:") or line.lower().startswith("message-id:"):
            current["id"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("from:"):
            current["from"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("subject:"):
            current["subject"] = line.split(":", 1)[1].strip()
        elif line.lower().startswith("date:"):
            current["date"] = line.split(":", 1)[1].strip()
        # Some gog versions output lines differently; try index-based fallback
        elif not current and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                current = {"id": parts[0], "from": parts[1], "subject": parts[2]}
                emails.append(current)
                current = {}

    if current:
        emails.append(current)

    return emails


def is_priority_sender(sender: str) -> bool:
    sender_lower = sender.lower()
    for domain in PRIORITY_DOMAINS:
        if domain in sender_lower:
            return True
    for pattern in ACADEMIC_SENDERS:
        if pattern in sender_lower:
            return True
    return False


def is_urgent(email: dict) -> bool:
    text = f"{email.get('from', '')} {email.get('subject', '')}".lower()
    if is_priority_sender(email.get("from", "")):
        return True
    for kw in URGENT_KEYWORDS:
        if kw in text:
            return True
    return False


def make_email_id(email: dict) -> str:
    """Create a stable ID for deduplication."""
    raw_id = email.get("id", "")
    if raw_id:
        return raw_id
    # Fallback: hash from+subject
    return f"{email.get('from','')}__{email.get('subject','')}".replace(" ", "_")[:120]


def main():
    seen = load_seen_ids()

    try:
        token = get_telegram_token()
    except Exception as e:
        # Can't alert without token — exit silently
        sys.exit(0)

    emails = fetch_recent_emails()
    update_heartbeat_email_timestamp()

    alerted = 0
    new_seen = set()

    for email in emails:
        eid = make_email_id(email)
        new_seen.add(eid)

        if eid in seen:
            continue  # Already alerted

        if is_urgent(email):
            sender = email.get("from", "Unknown")
            subject = email.get("subject", "(no subject)")
            send_telegram_alert(sender, subject, token)
            alerted += 1

    # Update seen with everything we just fetched (urgent or not)
    seen.update(new_seen)
    save_seen_ids(seen)

    if alerted > 0:
        print(f"[email_poll] Sent {alerted} alert(s).")
    # Exit silently if nothing urgent


if __name__ == "__main__":
    main()
