#!/usr/bin/env python3

import os
import sys
from datetime import datetime, timedelta
import json

try:
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
except ImportError:
    print("Google Calendar libraries not installed. Installing...")
    os.system("pip3 install --user google-api-python-client google-auth-httplib2 google-auth-oauthlib")
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError

# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar.readonly']

def main():
    """Shows basic usage of the Google Calendar API."""
    creds = None
    # The file token.json stores the user's access and refresh tokens.
    if os.path.exists('/home/clawdbot/clawd/calendar_token.json'):
        creds = Credentials.from_authorized_user_file('/home/clawdbot/clawd/calendar_token.json', SCOPES)
    
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                '/home/clawdbot/clawd/google_calendar_credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('/home/clawdbot/clawd/calendar_token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('calendar', 'v3', creds=creds)

        # Get tomorrow's date
        tomorrow = datetime.now() + timedelta(days=1)
        start_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
        end_time = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'

        print(f"Getting schedule for {tomorrow.strftime('%A, %B %d, %Y')}...")
        
        # Call the Calendar API
        events_result = service.events().list(
            calendarId='primary',
            timeMin=start_time,
            timeMax=end_time,
            singleEvents=True,
            orderBy='startTime'
        ).execute()
        
        events = events_result.get('items', [])

        if not events:
            print('No events found for tomorrow.')
            return

        print(f"\n📅 Schedule for {tomorrow.strftime('%A, %B %d, %Y')}:")
        print("=" * 50)
        
        for event in events:
            start = event['start'].get('dateTime', event['start'].get('date'))
            end = event['end'].get('dateTime', event['end'].get('date'))
            
            # Parse datetime
            if 'T' in start:  # Full datetime
                start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                end_dt = datetime.fromisoformat(end.replace('Z', '+00:00'))
                time_str = f"{start_dt.strftime('%I:%M %p')} - {end_dt.strftime('%I:%M %p')}"
            else:  # All-day event
                time_str = "All Day"
            
            print(f"🕐 {time_str}")
            print(f"📋 {event['summary']}")
            
            if 'location' in event:
                print(f"📍 {event['location']}")
            
            if 'description' in event:
                print(f"📝 {event['description'][:100]}{'...' if len(event['description']) > 100 else ''}")
            
            print("-" * 30)

    except HttpError as error:
        print('An error occurred: %s' % error)

if __name__ == '__main__':
    main()