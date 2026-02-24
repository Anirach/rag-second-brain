#!/usr/bin/env python3
"""
Combined Morning Briefing
- Email summary (grouped by category)
- Today's calendar events
- Weather forecast
- Generates one unified morning message
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# Paths
GMAIL_TOKEN = '/home/clawdbot/clawd/gmail/token.json'
CALENDAR_TOKEN = '/home/clawdbot/clawd/gcal/token.json'

GMAIL_SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']
CALENDAR_SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

# Weather API (wttr.in - no key needed)
WEATHER_LOCATION = "Bangkok"

def get_gmail_service():
    """Get Gmail API service."""
    creds = Credentials.from_authorized_user_file(GMAIL_TOKEN, GMAIL_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('gmail', 'v1', credentials=creds)

def get_calendar_service():
    """Get Calendar API service."""
    creds = Credentials.from_authorized_user_file(CALENDAR_TOKEN, CALENDAR_SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('calendar', 'v3', credentials=creds)

def get_unread_summary():
    """Get summary of unread emails."""
    try:
        service = get_gmail_service()
        results = service.users().messages().list(
            userId='me',
            q='is:unread in:inbox',
            maxResults=50
        ).execute()
        
        messages = results.get('messages', [])
        count = results.get('resultSizeEstimate', len(messages))
        
        # Categorize
        categories = defaultdict(int)
        important = []
        
        for msg in messages[:20]:  # Sample first 20
            try:
                full = service.users().messages().get(
                    userId='me', id=msg['id'],
                    format='metadata',
                    metadataHeaders=['From', 'Subject']
                ).execute()
                
                headers = {h['name']: h['value'] for h in full.get('payload', {}).get('headers', [])}
                from_addr = headers.get('From', '').lower()
                subject = headers.get('Subject', '')
                
                if 'github' in from_addr:
                    categories['🐙 GitHub'] += 1
                elif 'linkedin' in from_addr:
                    categories['💼 LinkedIn'] += 1
                elif 'google' in from_addr or 'scholar' in from_addr:
                    categories['🔍 Google'] += 1
                elif any(x in from_addr for x in ['noreply', 'notification', 'newsletter', 'marketing']):
                    categories['📢 Notifications'] += 1
                else:
                    categories['👤 Personal'] += 1
                    important.append(f"{headers.get('From', 'Unknown').split('<')[0].strip()}: {subject[:40]}")
            except:
                pass
        
        return {
            'count': count,
            'categories': dict(categories),
            'important': important[:5]
        }
    except Exception as e:
        return {'error': str(e)}

def get_today_events():
    """Get today's calendar events."""
    try:
        service = get_calendar_service()
        
        # Today's range in UTC
        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_of_day.isoformat() + 'Z',
            timeMax=end_of_day.isoformat() + 'Z',
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])
        
        formatted = []
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            # Parse and format time
            if 'T' in start:
                dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                time_str = dt.strftime('%H:%M')
            else:
                time_str = 'All day'
            
            formatted.append({
                'time': time_str,
                'summary': event.get('summary', 'No title'),
                'location': event.get('location', '')
            })
        
        return formatted
    except Exception as e:
        return {'error': str(e)}

def get_weather():
    """Get weather from wttr.in."""
    try:
        url = f"https://wttr.in/{WEATHER_LOCATION}?format=j1"
        resp = requests.get(url, timeout=10)
        data = resp.json()
        
        current = data['current_condition'][0]
        
        return {
            'temp': current['temp_C'],
            'feels_like': current['FeelsLikeC'],
            'condition': current['weatherDesc'][0]['value'],
            'humidity': current['humidity'],
            'location': WEATHER_LOCATION
        }
    except Exception as e:
        return {'error': str(e)}

def generate_briefing():
    """Generate the full morning briefing."""
    now = datetime.now()
    date_str = now.strftime('%A, %B %d, %Y')
    
    lines = [f"☀️ **Good Morning!** — {date_str}\n"]
    
    # Weather
    weather = get_weather()
    if 'error' not in weather:
        lines.append(f"🌡️ **Weather in {weather['location']}**")
        lines.append(f"   {weather['temp']}°C (feels like {weather['feels_like']}°C)")
        lines.append(f"   {weather['condition']}, {weather['humidity']}% humidity\n")
    
    # Calendar
    events = get_today_events()
    if isinstance(events, list):
        if events:
            lines.append(f"📅 **Today's Schedule** ({len(events)} events)")
            for e in events[:5]:
                loc = f" 📍{e['location']}" if e['location'] else ""
                lines.append(f"   • {e['time']} — {e['summary']}{loc}")
            if len(events) > 5:
                lines.append(f"   ...and {len(events) - 5} more")
            lines.append("")
        else:
            lines.append("📅 **Today's Schedule:** No events\n")
    
    # Email
    email = get_unread_summary()
    if 'error' not in email:
        lines.append(f"📬 **Unread Emails:** {email['count']}")
        for cat, count in email['categories'].items():
            lines.append(f"   • {cat}: {count}")
        
        if email['important']:
            lines.append("\n📌 **Personal emails:**")
            for e in email['important']:
                lines.append(f"   • {e}")
    
    lines.append("\n---")
    lines.append("*Run email digest to mark platform emails as read*")
    
    return '\n'.join(lines)

if __name__ == "__main__":
    print(generate_briefing())
