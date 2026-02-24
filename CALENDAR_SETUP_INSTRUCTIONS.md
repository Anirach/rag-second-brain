# 📅 Google Calendar Integration Setup Instructions

## Quick Setup (5 minutes)

### Step 1: Download Files
Download these 2 files to your computer:
1. `google_calendar_setup.py` 
2. `google_calendar_credentials.json`

### Step 2: Run Setup Script
Open Terminal/Command Prompt and run:
```bash
python3 google_calendar_setup.py
```

### Step 3: Authorize Access
- Script will open your browser
- Login to your Google account  
- Grant calendar access permissions
- Return to terminal when done

### Step 4: Add XSD Meeting
The script will ask if you want to add your XSD meeting:
- **Event:** XSD meeting
- **Date:** Tomorrow (January 29, 2026)
- **Time:** 10:30 AM (Bangkok time)
- **Duration:** 1 hour (default)

Type `y` to confirm adding the meeting.

## After Setup ✅

Once complete, you'll have:
- ✅ Full Google Calendar integration
- ✅ XSD meeting added automatically
- ✅ Calendar manager for future use

## Available Commands

Check your schedule:
```bash
python3 calendar_manager.py schedule
```

Add new events:
```bash
python3 calendar_manager.py add "Meeting Name" "2026-01-29T14:00:00+07:00"
```

## Troubleshooting

**If you get "browser not found" error:**
- Make sure you're running on a computer with a web browser
- Not in a server/headless environment

**If credentials don't work:**
- Make sure both files are in the same folder
- Check that `google_calendar_credentials.json` is valid

## Next Steps

After completing setup:
1. I'll be able to check your schedule automatically
2. Add meetings directly through chat
3. Send you daily/weekly schedule summaries
4. Set up meeting reminders

Let me know when you've completed the setup! 🐕