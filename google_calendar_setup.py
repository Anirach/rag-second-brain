#!/usr/bin/env python3
"""
Google Calendar Integration Setup for Anirach
Run this script on a computer with browser access to complete OAuth setup
"""

import os
import sys
from datetime import datetime, timedelta
import json

def install_requirements():
    """Install required Google Calendar libraries"""
    print("📦 Installing Google Calendar dependencies...")
    os.system("pip3 install --user google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    print("✅ Dependencies installed!")

def setup_calendar_integration():
    """Complete OAuth setup for Google Calendar"""
    try:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError
    except ImportError:
        install_requirements()
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow
        from googleapiclient.discovery import build
        from googleapiclient.errors import HttpError

    SCOPES = ['https://www.googleapis.com/auth/calendar']  # Full calendar access
    
    print("🔐 Starting Google Calendar OAuth Setup...")
    print("📋 This will open a browser for you to authorize calendar access")
    
    # Check if credentials file exists
    if not os.path.exists('google_calendar_credentials.json'):
        print("❌ Credentials file not found!")
        print("📁 Make sure 'google_calendar_credentials.json' is in the same directory")
        return False
    
    creds = None
    # Check for existing token
    if os.path.exists('calendar_token.json'):
        creds = Credentials.from_authorized_user_file('calendar_token.json', SCOPES)
    
    # If no valid credentials, run OAuth flow
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 Refreshing expired token...")
            creds.refresh(Request())
        else:
            print("🌐 Opening browser for Google authorization...")
            flow = InstalledAppFlow.from_client_secrets_file(
                'google_calendar_credentials.json', SCOPES)
            creds = flow.run_local_server(port=8080)
        
        # Save credentials for future use
        with open('calendar_token.json', 'w') as token:
            token.write(creds.to_json())
        print("✅ Calendar authorization completed!")
    
    # Test the connection
    try:
        service = build('calendar', 'v3', creds=creds)
        
        # Get calendar list to verify access
        calendar_list = service.calendarList().list().execute()
        print(f"✅ Successfully connected to Google Calendar!")
        print(f"📅 Found {len(calendar_list['items'])} calendars")
        
        return True
        
    except HttpError as error:
        print(f'❌ Error accessing calendar: {error}')
        return False

def add_pending_events(service):
    """Add any pending calendar events"""
    pending_events = [
        {
            'summary': 'XSD meeting',
            'start': {
                'dateTime': '2026-01-29T10:30:00+07:00',  # Bangkok timezone
                'timeZone': 'Asia/Bangkok',
            },
            'end': {
                'dateTime': '2026-01-29T11:30:00+07:00',  # Assuming 1 hour meeting
                'timeZone': 'Asia/Bangkok',
            },
            'description': 'XSD meeting scheduled via ClawdBot'
        }
    ]
    
    print("📅 Adding pending events...")
    
    for event_data in pending_events:
        try:
            event = service.events().insert(calendarId='primary', body=event_data).execute()
            print(f"✅ Added: {event_data['summary']} - {event_data['start']['dateTime']}")
        except Exception as e:
            print(f"❌ Failed to add {event_data['summary']}: {e}")

def create_calendar_manager():
    """Create the main calendar management script"""
    
    calendar_manager_code = '''#!/usr/bin/env python3
"""
Google Calendar Manager for ClawdBot
"""

import os
import sys
from datetime import datetime, timedelta
import json

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/calendar']

def get_calendar_service():
    """Get authenticated calendar service"""
    creds = Credentials.from_authorized_user_file('calendar_token.json', SCOPES)
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open('calendar_token.json', 'w') as token:
            token.write(creds.to_json())
    
    return build('calendar', 'v3', creds=creds)

def get_schedule(date_str=None):
    """Get schedule for specified date (default: tomorrow)"""
    service = get_calendar_service()
    
    if date_str:
        target_date = datetime.strptime(date_str, '%Y-%m-%d')
    else:
        target_date = datetime.now() + timedelta(days=1)
    
    start_time = target_date.replace(hour=0, minute=0, second=0).isoformat() + '+07:00'
    end_time = target_date.replace(hour=23, minute=59, second=59).isoformat() + '+07:00'
    
    events_result = service.events().list(
        calendarId='primary',
        timeMin=start_time,
        timeMax=end_time,
        singleEvents=True,
        orderBy='startTime'
    ).execute()
    
    events = events_result.get('items', [])
    
    if not events:
        print(f'No events found for {target_date.strftime("%A, %B %d, %Y")}')
        return
    
    print(f'\\n📅 Schedule for {target_date.strftime("%A, %B %d, %Y")}:')
    print('=' * 50)
    
    for event in events:
        start = event['start'].get('dateTime', event['start'].get('date'))
        summary = event['summary']
        
        if 'T' in start:
            start_dt = datetime.fromisoformat(start)
            print(f'🕐 {start_dt.strftime("%I:%M %p")} - {summary}')
        else:
            print(f'📅 All Day - {summary}')
        
        if 'location' in event:
            print(f'📍 {event["location"]}')
        print('-' * 30)

def add_event(summary, start_datetime, end_datetime=None, description="", location=""):
    """Add new calendar event"""
    service = get_calendar_service()
    
    if not end_datetime:
        # Default to 1 hour meeting
        start_dt = datetime.fromisoformat(start_datetime)
        end_datetime = (start_dt + timedelta(hours=1)).isoformat()
    
    event = {
        'summary': summary,
        'start': {
            'dateTime': start_datetime,
            'timeZone': 'Asia/Bangkok',
        },
        'end': {
            'dateTime': end_datetime,
            'timeZone': 'Asia/Bangkok',
        },
        'description': description,
        'location': location
    }
    
    event = service.events().insert(calendarId='primary', body=event).execute()
    print(f'✅ Event added: {summary}')
    return event

if __name__ == '__main__':
    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == 'schedule':
            date_arg = sys.argv[2] if len(sys.argv) > 2 else None
            get_schedule(date_arg)
        elif command == 'add':
            if len(sys.argv) < 4:
                print("Usage: python3 calendar_manager.py add 'Event Name' '2026-01-29T10:30:00+07:00'")
            else:
                summary = sys.argv[2]
                start = sys.argv[3]
                end = sys.argv[4] if len(sys.argv) > 4 else None
                add_event(summary, start, end)
    else:
        # Default: show tomorrow's schedule
        get_schedule()
'''
    
    with open('calendar_manager.py', 'w') as f:
        f.write(calendar_manager_code)
    
    print("✅ Created calendar_manager.py")

def main():
    """Main setup function"""
    print("🚀 Google Calendar Integration Setup")
    print("=" * 40)
    
    # Step 1: Setup OAuth
    if setup_calendar_integration():
        print("\\n✅ OAuth setup completed!")
        
        # Step 2: Create calendar manager
        create_calendar_manager()
        
        # Step 3: Add pending XSD meeting
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        
        creds = Credentials.from_authorized_user_file('calendar_token.json', ['https://www.googleapis.com/auth/calendar'])
        service = build('calendar', 'v3', creds=creds)
        
        print("\\n📋 Would you like to add the XSD meeting now?")
        response = input("Add XSD meeting for tomorrow 10:30 AM? (y/n): ")
        
        if response.lower() == 'y':
            add_pending_events(service)
        
        print("\\n🎉 Setup Complete!")
        print("\\n📋 Available commands:")
        print("• python3 calendar_manager.py schedule  - View tomorrow's schedule")
        print("• python3 calendar_manager.py schedule 2026-01-30  - View specific date")
        print("• python3 calendar_manager.py add 'Meeting' '2026-01-29T14:00:00+07:00'  - Add event")
        
        # Copy files to ClawdBot directory
        print("\\n📁 Copying files to ClawdBot workspace...")
        os.system("cp calendar_token.json /home/clawdbot/clawd/ 2>/dev/null || true")
        os.system("cp calendar_manager.py /home/clawdbot/clawd/ 2>/dev/null || true")
        
    else:
        print("❌ Setup failed. Please check your credentials and try again.")

if __name__ == '__main__':
    main()