#!/usr/bin/env python3
"""Google Calendar CLI for ClawdBot"""
import os
import sys
import json
from datetime import datetime, timedelta
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/calendar']
TOKEN_FILE = '/home/clawdbot/clawd/gcal/token.json'

def get_service():
    """Get authenticated Calendar service"""
    if not os.path.exists(TOKEN_FILE):
        print("❌ Not authenticated. Run gcal_auth.py first.")
        sys.exit(1)
    
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return build('calendar', 'v3', credentials=creds)

def list_events(days=7, max_results=20):
    """List upcoming events"""
    service = get_service()
    now = datetime.utcnow().isoformat() + 'Z'
    end = (datetime.utcnow() + timedelta(days=days)).isoformat() + 'Z'
    
    events_result = service.events().list(
        calendarId='primary',
        timeMin=now,
        timeMax=end,
        maxResults=max_results,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    
    events = events_result.get('items', [])
    
    if not events:
        print('📅 No upcoming events found.')
        return []
    
    print(f'📅 Upcoming events (next {days} days):\n')
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event.get('summary', 'No title')
        print(f"  • {start[:16]} - {summary}")
    
    return events

def create_event(summary, start_time, end_time=None, description='', location=''):
    """Create a new event"""
    service = get_service()
    
    if end_time is None:
        # Default to 1 hour duration
        start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
        end_dt = start_dt + timedelta(hours=1)
        end_time = end_dt.isoformat()
    
    event = {
        'summary': summary,
        'location': location,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': 'Asia/Bangkok',
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'Asia/Bangkok',
        },
    }
    
    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f"✅ Event created: {event.get('htmlLink')}")
    return event

def delete_event(event_id):
    """Delete an event"""
    service = get_service()
    service.events().delete(calendarId='primary', eventId=event_id).execute()
    print(f"✅ Event deleted: {event_id}")

def list_calendars():
    """List all calendars"""
    service = get_service()
    calendars = service.calendarList().list().execute()
    
    print("📅 Your calendars:\n")
    for cal in calendars.get('items', []):
        primary = " (PRIMARY)" if cal.get('primary') else ""
        print(f"  • {cal['summary']}{primary}")
        print(f"    ID: {cal['id']}")
    
    return calendars

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 gcal.py <command>")
        print("Commands: list, calendars, create")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == 'list':
        days = int(sys.argv[2]) if len(sys.argv) > 2 else 7
        list_events(days)
    elif cmd == 'calendars':
        list_calendars()
    elif cmd == 'create':
        if len(sys.argv) < 4:
            print("Usage: python3 gcal.py create 'Event Name' '2026-01-27T10:00:00'")
            sys.exit(1)
        create_event(sys.argv[2], sys.argv[3])
    else:
        print(f"Unknown command: {cmd}")
