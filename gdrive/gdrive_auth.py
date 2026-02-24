#!/usr/bin/env python3
"""
Google Drive Authentication Helper
Run this to authenticate and get tokens
"""

import os
import json
from google_auth_oauthlib.flow import Flow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

SCOPES = ['https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = '/home/clawdbot/clawd/gdrive/credentials.json'
TOKEN_FILE = '/home/clawdbot/clawd/gdrive/token.json'
REDIRECT_URI = 'http://localhost'

def get_auth_url():
    """Generate authorization URL"""
    flow = Flow.from_client_secrets_file(
        CREDENTIALS_FILE,
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return auth_url, flow

def exchange_code(flow, code):
    """Exchange authorization code for tokens"""
    flow.fetch_token(code=code)
    creds = flow.credentials
    
    with open(TOKEN_FILE, 'w') as token:
        token.write(creds.to_json())
    
    return creds

def load_credentials():
    """Load existing credentials"""
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
        if creds and creds.valid:
            return creds
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
            with open(TOKEN_FILE, 'w') as token:
                token.write(creds.to_json())
            return creds
    return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Exchange code mode
        code = sys.argv[1]
        flow = Flow.from_client_secrets_file(
            CREDENTIALS_FILE,
            scopes=SCOPES,
            redirect_uri=REDIRECT_URI
        )
        creds = exchange_code(flow, code)
        print("✅ Authentication successful! Token saved.")
    else:
        # Generate URL mode
        auth_url, _ = get_auth_url()
        print("Visit this URL to authorize:")
        print(auth_url)
