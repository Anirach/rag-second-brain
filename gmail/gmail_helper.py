#!/usr/bin/env python3
"""Gmail Helper - Read and manage emails"""

import os
import base64
from email.mime.text import MIMEText
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

TOKEN_FILE = '/home/clawdbot/clawd/gmail/token.json'
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels'
]

def get_service():
    """Get Gmail API service."""
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def list_messages(max_results=10, query=''):
    """List messages from inbox."""
    service = get_service()
    
    results = service.users().messages().list(
        userId='me',
        maxResults=max_results,
        q=query if query else 'in:inbox'
    ).execute()
    
    messages = results.get('messages', [])
    return messages

def get_message(msg_id):
    """Get full message details."""
    service = get_service()
    
    msg = service.users().messages().get(
        userId='me',
        id=msg_id,
        format='full'
    ).execute()
    
    # Parse headers
    headers = msg.get('payload', {}).get('headers', [])
    header_dict = {h['name']: h['value'] for h in headers}
    
    # Get snippet
    snippet = msg.get('snippet', '')
    
    return {
        'id': msg_id,
        'from': header_dict.get('From', 'Unknown'),
        'to': header_dict.get('To', ''),
        'subject': header_dict.get('Subject', '(No Subject)'),
        'date': header_dict.get('Date', ''),
        'snippet': snippet,
        'labels': msg.get('labelIds', [])
    }

def get_unread_count():
    """Get unread message count."""
    service = get_service()
    
    results = service.users().messages().list(
        userId='me',
        q='is:unread in:inbox'
    ).execute()
    
    return results.get('resultSizeEstimate', 0)

def list_inbox(max_results=10, unread_only=False):
    """List inbox messages with details."""
    query = 'in:inbox'
    if unread_only:
        query += ' is:unread'
    
    messages = list_messages(max_results=max_results, query=query)
    
    detailed = []
    for msg in messages:
        try:
            details = get_message(msg['id'])
            detailed.append(details)
        except Exception as e:
            print(f"Error getting message {msg['id']}: {e}")
    
    return detailed

if __name__ == "__main__":
    import sys
    import json
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        
        if cmd == 'unread':
            count = get_unread_count()
            print(f"📬 Unread messages: {count}")
        
        elif cmd == 'inbox':
            n = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            messages = list_inbox(max_results=n)
            print(f"📥 Latest {len(messages)} emails:\n")
            for i, msg in enumerate(messages, 1):
                unread = '🔵' if 'UNREAD' in msg['labels'] else '⚪'
                print(f"{i}. {unread} {msg['subject']}")
                print(f"   From: {msg['from']}")
                print(f"   Date: {msg['date']}")
                print(f"   {msg['snippet'][:100]}...")
                print()
        
        elif cmd == 'search':
            query = ' '.join(sys.argv[2:])
            messages = list_messages(max_results=10, query=query)
            print(f"Found {len(messages)} messages for: {query}")
            for msg in messages:
                details = get_message(msg['id'])
                print(f"- {details['subject']} (from: {details['from']})")
    else:
        print("Usage:")
        print("  python gmail_helper.py unread     - Get unread count")
        print("  python gmail_helper.py inbox [n]  - List n inbox messages")
        print("  python gmail_helper.py search <query> - Search emails")
