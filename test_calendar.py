#!/usr/bin/env python3
"""
Test calendar access and add XSD meeting
"""

import json
import requests
from datetime import datetime, timedelta

def load_token():
    """Load the saved access token"""
    with open('calendar_token.json', 'r') as f:
        return json.load(f)

def get_headers():
    """Get authorization headers"""
    token_data = load_token()
    return {
        'Authorization': f'Bearer {token_data["access_token"]}',
        'Content-Type': 'application/json'
    }

def test_calendar_access():
    """Test if we can access the calendar"""
    print("🔍 Testing calendar access...")
    
    url = "https://www.googleapis.com/calendar/v3/calendars/primary"
    headers = get_headers()
    
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        calendar_info = response.json()
        print(f"✅ Connected to calendar: {calendar_info.get('summary', 'Primary Calendar')}")
        return True
    else:
        print(f"❌ Calendar access failed: {response.status_code}")
        print(response.text)
        return False

def get_tomorrow_schedule():
    """Get tomorrow's schedule"""
    print("📅 Checking tomorrow's schedule...")
    
    # Tomorrow's date
    tomorrow = datetime.now() + timedelta(days=1)
    start_time = tomorrow.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
    end_time = tomorrow.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
    
    url = f"https://www.googleapis.com/calendar/v3/calendars/primary/events"
    headers = get_headers()
    
    params = {
        'timeMin': start_time,
        'timeMax': end_time,
        'singleEvents': 'true',
        'orderBy': 'startTime'
    }
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code == 200:
        events_data = response.json()
        events = events_data.get('items', [])
        
        print(f"📋 Found {len(events)} events for tomorrow ({tomorrow.strftime('%A, %B %d, %Y')}):")
        
        if not events:
            print("   No events scheduled")
        else:
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                summary = event['summary']
                
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    print(f"   🕐 {start_dt.strftime('%I:%M %p')} - {summary}")
                else:
                    print(f"   📅 All Day - {summary}")
        
        return events
    else:
        print(f"❌ Failed to get schedule: {response.status_code}")
        return None

def add_xsd_meeting():
    """Add the XSD meeting for tomorrow 10:30 AM"""
    print("➕ Adding XSD meeting...")
    
    # Tomorrow at 10:30 AM (Bangkok time)
    tomorrow = datetime.now() + timedelta(days=1)
    start_time = tomorrow.replace(hour=10, minute=30, second=0, microsecond=0)
    end_time = start_time + timedelta(hours=1)  # 1 hour meeting
    
    # Format for Google Calendar API (Bangkok timezone)
    start_str = start_time.isoformat() + '+07:00'
    end_str = end_time.isoformat() + '+07:00'
    
    event_data = {
        'summary': 'XSD meeting',
        'start': {
            'dateTime': start_str,
            'timeZone': 'Asia/Bangkok'
        },
        'end': {
            'dateTime': end_str,
            'timeZone': 'Asia/Bangkok'
        },
        'description': 'XSD meeting added via ClawdBot'
    }
    
    url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
    headers = get_headers()
    
    response = requests.post(url, headers=headers, json=event_data)
    
    if response.status_code == 200:
        event = response.json()
        print(f"✅ XSD meeting added successfully!")
        print(f"   📅 {tomorrow.strftime('%A, %B %d, %Y')}")
        print(f"   🕐 10:30 AM - 11:30 AM")
        print(f"   📋 XSD meeting")
        return True
    else:
        print(f"❌ Failed to add meeting: {response.status_code}")
        print(response.text)
        return False

def main():
    """Main test function"""
    print("🚀 Google Calendar Setup Test")
    print("=" * 40)
    
    # Test 1: Calendar access
    if not test_calendar_access():
        return
    
    print()
    
    # Test 2: Check current schedule
    get_tomorrow_schedule()
    
    print()
    
    # Test 3: Add XSD meeting
    if add_xsd_meeting():
        print()
        print("📅 Updated schedule:")
        get_tomorrow_schedule()
    
    print()
    print("🎉 Calendar setup complete!")

if __name__ == "__main__":
    main()