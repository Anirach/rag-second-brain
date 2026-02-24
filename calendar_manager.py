#!/usr/bin/env python3
"""
Permanent Google Calendar Manager for Anirach
"""

import json
import requests
from datetime import datetime, timedelta
import sys

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

def refresh_token_if_needed():
    """Refresh token if expired"""
    # This would handle token refresh - for now using existing token
    pass

def get_schedule(days_ahead=1):
    """Get schedule for specified number of days ahead (default: tomorrow)"""
    refresh_token_if_needed()
    
    target_date = datetime.now() + timedelta(days=days_ahead)
    start_time = target_date.replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + 'Z'
    end_time = target_date.replace(hour=23, minute=59, second=59, microsecond=999999).isoformat() + 'Z'
    
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
        
        day_name = "today" if days_ahead == 0 else "tomorrow" if days_ahead == 1 else f"in {days_ahead} days"
        
        result = f"📅 **Schedule for {day_name} ({target_date.strftime('%A, %B %d, %Y')}):**\\n"
        
        if not events:
            result += "No events scheduled\\n"
        else:
            for event in events:
                start = event['start'].get('dateTime', event['start'].get('date'))
                summary = event['summary']
                location = event.get('location', '')
                
                if 'T' in start:
                    start_dt = datetime.fromisoformat(start.replace('Z', '+00:00'))
                    result += f"🕐 **{start_dt.strftime('%I:%M %p')}** - {summary}"
                else:
                    result += f"📅 **All Day** - {summary}"
                
                if location:
                    result += f" 📍 {location}"
                result += "\\n"
        
        return result, events
    else:
        return f"❌ Failed to get schedule: {response.status_code}", None

def add_event(summary, date_str, time_str, duration_hours=1, description="", location=""):
    """Add new calendar event"""
    refresh_token_if_needed()
    
    try:
        # Parse date and time
        event_date = datetime.strptime(date_str, '%Y-%m-%d')
        if time_str:
            time_parts = time_str.split(':')
            hour = int(time_parts[0])
            minute = int(time_parts[1]) if len(time_parts) > 1 else 0
            
            # Handle AM/PM
            if 'pm' in time_str.lower() and hour != 12:
                hour += 12
            elif 'am' in time_str.lower() and hour == 12:
                hour = 0
            
            start_time = event_date.replace(hour=hour, minute=minute)
        else:
            # All day event
            start_time = event_date
        
        end_time = start_time + timedelta(hours=duration_hours)
        
        # Format for API
        if time_str:  # Timed event
            start_str = start_time.isoformat() + '+07:00'  # Bangkok timezone
            end_str = end_time.isoformat() + '+07:00'
            
            event_data = {
                'summary': summary,
                'start': {
                    'dateTime': start_str,
                    'timeZone': 'Asia/Bangkok'
                },
                'end': {
                    'dateTime': end_str,
                    'timeZone': 'Asia/Bangkok'
                }
            }
        else:  # All day event
            event_data = {
                'summary': summary,
                'start': {
                    'date': start_time.strftime('%Y-%m-%d')
                },
                'end': {
                    'date': end_time.strftime('%Y-%m-%d')
                }
            }
        
        if description:
            event_data['description'] = description
        if location:
            event_data['location'] = location
        
        url = "https://www.googleapis.com/calendar/v3/calendars/primary/events"
        headers = get_headers()
        
        response = requests.post(url, headers=headers, json=event_data)
        
        if response.status_code == 200:
            event = response.json()
            return f"✅ **Event added successfully!**\\n📅 {summary} - {date_str} {time_str}", True
        else:
            return f"❌ Failed to add event: {response.status_code}", False
            
    except Exception as e:
        return f"❌ Error adding event: {str(e)}", False

def quick_add_meeting(text):
    """Quick add meeting from natural language"""
    # Simple parser for common patterns
    words = text.lower().split()
    
    # Extract time patterns
    time_pattern = None
    date_pattern = None
    
    # Look for time patterns like "2pm", "10:30", "2:30pm"
    for i, word in enumerate(words):
        if ':' in word or 'pm' in word or 'am' in word:
            time_pattern = word
        elif word in ['tomorrow', 'today']:
            date_pattern = word
        elif word in ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']:
            date_pattern = word
    
    # Extract meeting name (everything before time/date words)
    meeting_words = []
    for word in words:
        if word in ['at', 'on', 'tomorrow', 'today', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'] or ':' in word or 'pm' in word or 'am' in word:
            break
        meeting_words.append(word)
    
    meeting_name = ' '.join(meeting_words) if meeting_words else 'Meeting'
    
    # Default to tomorrow if no date specified
    target_date = datetime.now() + timedelta(days=1)
    if date_pattern == 'today':
        target_date = datetime.now()
    
    date_str = target_date.strftime('%Y-%m-%d')
    time_str = time_pattern or '10:00am'
    
    return add_event(meeting_name, date_str, time_str)

# Test the functions
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        schedule, events = get_schedule(1)  # Tomorrow
        print(schedule)
    else:
        print("Calendar manager ready!")
        print("Functions available:")
        print("- get_schedule(days_ahead=1)")
        print("- add_event(summary, date_str, time_str)")
        print("- quick_add_meeting(text)")