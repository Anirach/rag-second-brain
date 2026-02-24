#!/usr/bin/env python3
"""Gmail helper for reading emails."""

import json
import os
import base64
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.json")
CREDS_PATH = os.path.join(SCRIPT_DIR, "..", "gdrive", "credentials.json")


def get_service():
    """Get authenticated Gmail service."""
    with open(TOKEN_PATH) as f:
        token_data = json.load(f)
    
    with open(CREDS_PATH) as f:
        creds_data = json.load(f)["installed"]
    
    creds = Credentials(
        token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_PATH, "w") as f:
            json.dump({
                "access_token": creds.token,
                "refresh_token": creds.refresh_token,
                "scope": " ".join(creds.scopes)
            }, f, indent=2)
    
    return build("gmail", "v1", credentials=creds)


def list_messages(query="is:unread", max_results=10):
    """List messages matching query."""
    service = get_service()
    results = service.users().messages().list(
        userId="me",
        q=query,
        maxResults=max_results
    ).execute()
    
    messages = results.get("messages", [])
    return messages


def get_message(msg_id):
    """Get a specific message by ID."""
    service = get_service()
    msg = service.users().messages().get(
        userId="me",
        id=msg_id,
        format="full"
    ).execute()
    return msg


def get_message_summary(msg):
    """Extract summary from message."""
    headers = {h["name"]: h["value"] for h in msg["payload"]["headers"]}
    
    # Get body
    body = ""
    if "parts" in msg["payload"]:
        for part in msg["payload"]["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    body = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                    break
    elif "body" in msg["payload"] and msg["payload"]["body"].get("data"):
        body = base64.urlsafe_b64decode(msg["payload"]["body"]["data"]).decode("utf-8", errors="ignore")
    
    return {
        "id": msg["id"],
        "from": headers.get("From", "Unknown"),
        "subject": headers.get("Subject", "(No subject)"),
        "date": headers.get("Date", "Unknown"),
        "snippet": msg.get("snippet", ""),
        "body": body[:1000] if body else ""
    }


def check_inbox(max_results=5, unread_only=True):
    """Check inbox and return summaries."""
    query = "is:unread" if unread_only else "in:inbox"
    messages = list_messages(query=query, max_results=max_results)
    
    if not messages:
        return []
    
    summaries = []
    for msg_info in messages:
        msg = get_message(msg_info["id"])
        summaries.append(get_message_summary(msg))
    
    return summaries


def print_inbox(max_results=5, unread_only=True):
    """Print inbox summary."""
    summaries = check_inbox(max_results, unread_only)
    
    if not summaries:
        print("📭 No unread messages." if unread_only else "📭 Inbox empty.")
        return
    
    print(f"📬 {'Unread' if unread_only else 'Recent'} messages ({len(summaries)}):\n")
    for s in summaries:
        print(f"{'─' * 50}")
        print(f"From: {s['from']}")
        print(f"Subject: {s['subject']}")
        print(f"Date: {s['date']}")
        print(f"Preview: {s['snippet'][:100]}...")
        print()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "unread":
            print_inbox(unread_only=True)
        elif sys.argv[1] == "inbox":
            print_inbox(unread_only=False)
        elif sys.argv[1] == "search" and len(sys.argv) > 2:
            query = " ".join(sys.argv[2:])
            messages = list_messages(query=query)
            print(f"Found {len(messages)} messages matching: {query}")
            for msg_info in messages[:5]:
                msg = get_message(msg_info["id"])
                s = get_message_summary(msg)
                print(f"  - {s['subject']} ({s['from']})")
    else:
        print("Usage: python gmail.py [unread|inbox|search <query>]")
        print_inbox()
