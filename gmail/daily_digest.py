#!/usr/bin/env python3
"""
Daily Email Digest
- Summarizes unread emails grouped by sender
- Marks platform/company emails as read
- Leaves personal emails unread
"""

import os
import re
from collections import defaultdict
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

TOKEN_FILE = '/home/clawdbot/clawd/gmail/token.json'
SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.modify'
]

# Platform/company email patterns (mark as read after digest)
PLATFORM_PATTERNS = [
    r'@github\.com',
    r'@linkedin\.com',
    r'@pinterest\.com',
    r'@medium\.com',
    r'@skool\.com',
    r'@google\.com',
    r'@googlealerts',
    r'scholaralerts',
    r'@facebook\.com',
    r'@twitter\.com',
    r'@x\.com',
    r'noreply@',
    r'no-reply@',
    r'notifications@',
    r'alerts@',
    r'digest@',
    r'newsletter@',
    r'news@',
    r'marketing@',
    r'promo@',
    r'@mailchimp',
    r'@sendgrid',
    r'@amazonses',
    r'@slack\.com',
    r'@notion\.so',
    r'@figma\.com',
    r'@vercel\.com',
    r'@netlify\.com',
    r'@heroku\.com',
    r'@aws\.amazon',
    r'@cloud\.google',
    r'@azure\.microsoft',
    r'jobalerts',
    r'jobmail',
    r'recommendations@',
    r'updates@',
    r'info@',
    r'support@',
    r'@academic\.net',
    r'@conference',
    r'@ieee\.org',
    r'@acm\.org',
    r'@springer',
    r'@elsevier',
]

def get_service():
    """Get Gmail API service with modify permissions."""
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, 'w') as f:
            f.write(creds.to_json())
    
    return build('gmail', 'v1', credentials=creds)

def is_platform_email(from_addr):
    """Check if email is from a platform/company."""
    from_lower = from_addr.lower()
    for pattern in PLATFORM_PATTERNS:
        if re.search(pattern, from_lower):
            return True
    return False

def get_sender_category(from_addr):
    """Categorize sender."""
    from_lower = from_addr.lower()
    
    if 'github' in from_lower:
        return '🐙 GitHub'
    elif 'linkedin' in from_lower:
        return '💼 LinkedIn'
    elif 'google' in from_lower or 'scholar' in from_lower:
        return '🔍 Google'
    elif 'medium' in from_lower:
        return '📝 Medium'
    elif 'skool' in from_lower:
        return '🎓 Skool'
    elif 'pinterest' in from_lower:
        return '📌 Pinterest'
    elif 'conference' in from_lower or 'ieee' in from_lower or 'acm' in from_lower or 'academic' in from_lower:
        return '📚 Academic/Conference'
    elif is_platform_email(from_addr):
        return '🤖 Other Platforms'
    else:
        return '👤 Personal'

def extract_sender_name(from_addr):
    """Extract clean sender name."""
    # Format: "Name <email>" or just "email"
    match = re.match(r'"?([^"<]+)"?\s*<?', from_addr)
    if match:
        return match.group(1).strip()
    return from_addr.split('@')[0]

def get_unread_emails(max_results=100):
    """Get all unread emails."""
    service = get_service()
    
    results = service.users().messages().list(
        userId='me',
        q='is:unread in:inbox',
        maxResults=max_results
    ).execute()
    
    messages = results.get('messages', [])
    
    emails = []
    for msg in messages:
        try:
            full_msg = service.users().messages().get(
                userId='me',
                id=msg['id'],
                format='metadata',
                metadataHeaders=['From', 'Subject', 'Date']
            ).execute()
            
            headers = {h['name']: h['value'] for h in full_msg.get('payload', {}).get('headers', [])}
            
            emails.append({
                'id': msg['id'],
                'from': headers.get('From', 'Unknown'),
                'subject': headers.get('Subject', '(No Subject)'),
                'date': headers.get('Date', ''),
                'snippet': full_msg.get('snippet', '')[:100]
            })
        except Exception as e:
            print(f"Error getting message: {e}")
    
    return emails

def mark_as_read(service, message_ids):
    """Mark messages as read."""
    if not message_ids:
        return
    
    service.users().messages().batchModify(
        userId='me',
        body={
            'ids': message_ids,
            'removeLabelIds': ['UNREAD']
        }
    ).execute()

def generate_digest():
    """Generate email digest."""
    service = get_service()
    emails = get_unread_emails()
    
    if not emails:
        return "📭 No unread emails!", []
    
    # Group by category
    grouped = defaultdict(list)
    platform_ids = []
    personal_count = 0
    
    for email in emails:
        category = get_sender_category(email['from'])
        grouped[category].append(email)
        
        if category != '👤 Personal':
            platform_ids.append(email['id'])
        else:
            personal_count += 1
    
    # Build digest
    lines = [f"📬 **Email Digest** — {len(emails)} unread\n"]
    
    # Sort categories (Personal last)
    categories = sorted(grouped.keys(), key=lambda x: (x == '👤 Personal', x))
    
    for category in categories:
        cat_emails = grouped[category]
        lines.append(f"\n**{category}** ({len(cat_emails)})")
        
        # Group by sender within category
        by_sender = defaultdict(list)
        for e in cat_emails:
            sender = extract_sender_name(e['from'])
            by_sender[sender].append(e)
        
        for sender, sender_emails in by_sender.items():
            if len(sender_emails) == 1:
                lines.append(f"• {sender}: {sender_emails[0]['subject'][:50]}")
            else:
                lines.append(f"• {sender}: {len(sender_emails)} emails")
                for e in sender_emails[:3]:  # Show up to 3
                    lines.append(f"  - {e['subject'][:40]}")
                if len(sender_emails) > 3:
                    lines.append(f"  - ...and {len(sender_emails) - 3} more")
    
    # Mark platform emails as read
    if platform_ids:
        mark_as_read(service, platform_ids)
        lines.append(f"\n✅ Marked {len(platform_ids)} platform emails as read")
    
    if personal_count > 0:
        lines.append(f"📌 {personal_count} personal emails left unread")
    
    return '\n'.join(lines), platform_ids

if __name__ == "__main__":
    digest, marked = generate_digest()
    print(digest)
