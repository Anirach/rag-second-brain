#!/usr/bin/env python3
"""
Fix calendar and re-add XSD meeting with better settings
"""

import json
import requests
from datetime import datetime, timedelta

def load_token():
    """Load the saved access token"""
    with open('/home/clawdbot/clawd/calendar_token.json', 'r') as f:
        return json.load(f)

def get_headers():
    """Get authorization headers"""
    token_data = load_token()
    return {
        'Authorization': f'Bearer {token_data["access_token"]}',
        'Content-Type': 'application/json'
    }

def list_calendars():
    """List all available calendars"""
    print("📋 Checking your calendars...")
    
    url = "https://www.googleapis.com/calendar/v3/users/me/calendarList"
    headers = get_headers()
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        calendars = response.json()
        print(f"Found {len(calendars['items'])} calendars:")
        
        for cal in calendars['items']:
            is_primary = cal.get('primary', False)
            summary = cal.get('summary', 'No name')
            calendar_id = cal['id']
            
            print(f"  {'[PRIMARY] ' if is_primary else ''}📅 {summary}")
            print(f"      ID: {calendar_id}")
        
        return calendars['items']
    else:
        print(f"❌ Failed to list calendars: {response.status_code}")
        return []

def delete_duplicate_xsd_meetings():
    """Delete any existing XSD meetings to avoid duplicates"""
    print("🗑️ Checking for existing XSD meetings...")
    
    # Tomorrow's date range
    tomorrow = datetime.now() + timedelta(days=1)
    start_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
    end_time = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
    
    url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
    headers = get_headers()
    
    params = {
        'timeMin': start_time,
        'timeMax': end_time,
        'singleEvents': 'true',
        'q': 'XSD meeting'
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        events_data = response.json()
        events = events_data.get('items', [])
        
        for event in events:
            if 'XSD' in event['summary']:
                event_id = event['id']
                print(f"   Found existing XSD meeting: {event['summary']}")
                
                # Delete it
                delete_url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events/{event_id}"
                delete_response = requests.delete(delete_url, headers=headers)
                
                if delete_response.status_code == 204:
                    print(f"   ✅ Deleted existing XSD meeting")
                else:
                    print(f"   ❌ Failed to delete: {delete_response.status_code}")

def add_xsd_meeting_properly():
    """Add XSD meeting with proper settings"""
    print("➕ Adding XSD meeting with enhanced settings...")
    
    # Tomorrow at 10:30 AM Bangkok time
    tomorrow = datetime.now() + timedelta(days=1)
    start_time = tomorrow.replace(hour=10, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)
    
    event_data = {
        'summary': 'XSD meeting',
        'start': {
            'dateTime': start_time.isoformat(),
            'timeZone': 'Asia/Bangkok'
        },
        'end': {
            'dateTime': end_time.isoformat(),
            'timeZone': 'Asia/Bangkok'
        },
        'description': 'XSD meeting scheduled via ClawdBot assistant',
        'status': 'confirmed',
        'transparency': 'opaque',  # Show as busy
        'visibility': 'default',
        'reminders': {
            'useDefault': True
        }
    }
    
    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    headers = get_headers()
    
    response = requests.post(url, headers=headers, json=event_data)
    
    if response.status_code == 200:
        event = response.json()
        print(f"✅ XSD meeting added successfully!")
        print(f"   📅 {tomorrow.strftime('%A, %B %d, %Y')}")
        print(f"   🕐 10:30 AM - 11:30 AM (Bangkok time)")
        print(f"   📋 {event['summary']}")
        print(f"   🔗 Event ID: {event['id']}")
        print(f"   🌐 HTML Link: {event.get('htmlLink', 'N/A')}")
        return True
    else:
        print(f"❌ Failed to add meeting: {response.status_code}")
        print(response.text)
        return False

def main():
    """Main function"""
    print("🔧 Fixing Google Calendar XSD Meeting")
    print("=" * 40)
    
    # Step 1: List calendars
    calendars = list_calendars()
    print()
    
    # Step 2: Delete any existing XSD meetings
    delete_duplicate_xsd_meetings()
    print()
    
    # Step 3: Add the meeting properly
    if add_xsd_meeting_properly():
        print()
        print("🎉 Meeting should now appear in your calendar!")
        print("💡 Try refreshing your calendar view or waiting 1-2 minutes for sync.")
    
    print()
    print("📱 If it still doesn't appear, try:")
    print("   • Refresh your calendar app/website")
    print("   • Check if you're viewing the correct calendar")
    print("   • Make sure you're looking at January 29, 2026")

if __name__ == "__main__":
    main()