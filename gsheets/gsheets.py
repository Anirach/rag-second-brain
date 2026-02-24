#!/usr/bin/env python3
"""Google Sheets helper for expense tracking."""

import json
import os
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_PATH = os.path.join(SCRIPT_DIR, "token.json")
CREDS_PATH = os.path.join(SCRIPT_DIR, "..", "gdrive", "credentials.json")

def get_service():
    """Get authenticated Sheets service."""
    with open(TOKEN_PATH) as f:
        token_data = json.load(f)
    
    with open(CREDS_PATH) as f:
        creds_data = json.load(f)["installed"]
    
    creds = Credentials(
        token=token_data["access_token"],
        refresh_token=token_data.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token",
        client_id=creds_data["client_id"],
        client_secret=creds_data["client_secret"],
        scopes=["https://www.googleapis.com/auth/spreadsheets"]
    )
    
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        # Save refreshed token
        with open(TOKEN_PATH, "w") as f:
            json.dump({
                "access_token": creds.token,
                "refresh_token": creds.refresh_token,
                "scope": " ".join(creds.scopes)
            }, f, indent=2)
    
    return build("sheets", "v4", credentials=creds)

def read_sheet(spreadsheet_id, range_name):
    """Read data from a sheet."""
    service = get_service()
    result = service.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range=range_name
    ).execute()
    return result.get("values", [])

def append_row(spreadsheet_id, range_name, values):
    """Append a row to a sheet."""
    service = get_service()
    body = {"values": [values]}
    result = service.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        insertDataOption="INSERT_ROWS",
        body=body
    ).execute()
    return result

def update_cell(spreadsheet_id, range_name, value):
    """Update a specific cell or range."""
    service = get_service()
    body = {"values": [[value]] if not isinstance(value, list) else [value]}
    result = service.spreadsheets().values().update(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption="USER_ENTERED",
        body=body
    ).execute()
    return result

def list_sheets(spreadsheet_id):
    """List all sheets in a spreadsheet."""
    service = get_service()
    result = service.spreadsheets().get(spreadsheetId=spreadsheet_id).execute()
    return [s["properties"]["title"] for s in result.get("sheets", [])]

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python gsheets.py <spreadsheet_id> <range>")
        print("Example: python gsheets.py 1abc...xyz 'Sheet1!A1:D10'")
        sys.exit(1)
    
    data = read_sheet(sys.argv[1], sys.argv[2])
    for row in data:
        print("\t".join(str(c) for c in row))
