#!/usr/bin/env python3
"""Google Calendar OAuth Authentication"""
import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

SCOPES = ['https://www.googleapis.com/auth/calendar']
CREDS_FILE = '/home/clawdbot/clawd/gcal/credentials.json'
TOKEN_FILE = '/home/clawdbot/clawd/gcal/token.json'

def authenticate():
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            # Use console-based auth for headless server
            creds = flow.run_local_server(port=8085, open_browser=False)
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    print("✅ Authentication successful!")
    return creds

if __name__ == '__main__':
    print("Starting Google Calendar authentication...")
    print("Please visit the URL below to authorize:")
    authenticate()
