#!/usr/bin/env python3
"""Gmail OAuth Setup - generates auth URL for user to authorize"""

import os
import json
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = [
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.labels',
    'https://www.googleapis.com/auth/gmail.modify'
]

CREDENTIALS_FILE = '/home/clawdbot/clawd/gdrive/credentials.json'
TOKEN_FILE = '/home/clawdbot/clawd/gmail/token.json'

def generate_auth_url():
    """Generate OAuth URL for user to authorize."""
    flow = InstalledAppFlow.from_client_secrets_file(
        CREDENTIALS_FILE, 
        SCOPES,
        redirect_uri='urn:ietf:wg:oauth:2.0:oob'
    )
    auth_url, _ = flow.authorization_url(
        access_type='offline',
        prompt='consent',
        login_hint='anirach.m@fitm.kmutnb.ac.th'
    )
    return auth_url, flow

def exchange_code(flow, code):
    """Exchange authorization code for credentials."""
    flow.fetch_token(code=code)
    creds = flow.credentials
    
    os.makedirs(os.path.dirname(TOKEN_FILE), exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        f.write(creds.to_json())
    
    return creds

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Exchange code mode
        code = sys.argv[1]
        flow = InstalledAppFlow.from_client_secrets_file(
            CREDENTIALS_FILE,
            SCOPES,
            redirect_uri='urn:ietf:wg:oauth:2.0:oob'
        )
        creds = exchange_code(flow, code)
        print(f"✅ Token saved to {TOKEN_FILE}")
    else:
        # Generate URL mode
        auth_url, _ = generate_auth_url()
        print(f"🔐 Visit this URL to authorize Gmail:\n\n{auth_url}")
