#!/usr/bin/env python3
"""
Expense Tracker
- Parse natural language expense entries
- Log to Google Sheets
- Auto-categorize
"""

import os
import re
from datetime import datetime
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

TOKEN_FILE = '/home/clawdbot/clawd/gdrive/token.json'
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']

# Expense categories and keywords
CATEGORIES = {
    'Food & Dining': ['lunch', 'dinner', 'breakfast', 'coffee', 'restaurant', 'food', 'meal', 'eat', 'cafe', 'ข้าว', 'อาหาร'],
    'Transportation': ['taxi', 'grab', 'uber', 'bts', 'mrt', 'gas', 'fuel', 'parking', 'toll', 'รถ', 'แท็กซี่'],
    'Shopping': ['shop', 'buy', 'purchase', 'amazon', 'lazada', 'shopee', 'ซื้อ'],
    'Bills & Utilities': ['bill', 'electric', 'water', 'internet', 'phone', 'ค่า'],
    'Entertainment': ['movie', 'netflix', 'spotify', 'game', 'concert', 'หนัง'],
    'Health': ['doctor', 'medicine', 'pharmacy', 'hospital', 'gym', 'ยา', 'หมอ'],
    'Groceries': ['grocery', 'supermarket', 'market', 'tops', 'big c', 'lotus', 'ตลาด'],
    'Subscriptions': ['subscription', 'monthly', 'annual', 'premium', 'pro'],
    'Work': ['office', 'supplies', 'equipment', 'software'],
    'Other': []
}

def get_sheets_service():
    """Get Google Sheets API service."""
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('sheets', 'v4', credentials=creds)

def get_drive_service():
    """Get Google Drive API service."""
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build('drive', 'v3', credentials=creds)

def find_or_create_spreadsheet():
    """Find existing expense sheet or create new one."""
    drive = get_drive_service()
    
    # Search for existing
    results = drive.files().list(
        q="name='Expense Tracker 2026' and mimeType='application/vnd.google-apps.spreadsheet'",
        spaces='drive',
        fields='files(id, name)'
    ).execute()
    
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    
    # Create new spreadsheet
    sheets = get_sheets_service()
    spreadsheet = {
        'properties': {'title': 'Expense Tracker 2026'},
        'sheets': [{
            'properties': {'title': 'Expenses'},
            'data': [{
                'startRow': 0,
                'startColumn': 0,
                'rowData': [{
                    'values': [
                        {'userEnteredValue': {'stringValue': 'Date'}},
                        {'userEnteredValue': {'stringValue': 'Amount'}},
                        {'userEnteredValue': {'stringValue': 'Category'}},
                        {'userEnteredValue': {'stringValue': 'Description'}},
                        {'userEnteredValue': {'stringValue': 'Notes'}}
                    ]
                }]
            }]
        }]
    }
    
    result = sheets.spreadsheets().create(body=spreadsheet).execute()
    spreadsheet_id = result['spreadsheetId']
    
    print(f"Created new spreadsheet: https://docs.google.com/spreadsheets/d/{spreadsheet_id}")
    return spreadsheet_id

def detect_category(text):
    """Auto-detect expense category from description."""
    text_lower = text.lower()
    
    for category, keywords in CATEGORIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                return category
    
    return 'Other'

def parse_expense(text):
    """
    Parse natural language expense entry.
    Examples:
    - "Paid 308 for lunch"
    - "Grab to office 150"
    - "Coffee 85 baht"
    - "Lunch with team 450"
    """
    # Extract amount (number)
    amount_match = re.search(r'(\d+(?:\.\d{2})?)', text)
    if not amount_match:
        return None
    
    amount = float(amount_match.group(1))
    
    # Remove amount from text for description
    description = re.sub(r'\d+(?:\.\d{2})?', '', text)
    description = re.sub(r'\s+', ' ', description).strip()
    description = re.sub(r'^(paid|spent|for|baht|บาท)\s*', '', description, flags=re.IGNORECASE)
    description = re.sub(r'\s*(baht|บาท)$', '', description, flags=re.IGNORECASE)
    description = description.strip()
    
    # Detect category
    category = detect_category(text)
    
    return {
        'amount': amount,
        'description': description.capitalize() if description else 'Expense',
        'category': category,
        'date': datetime.now().strftime('%Y-%m-%d %H:%M')
    }

def log_expense(text, notes=''):
    """Log an expense from natural language."""
    expense = parse_expense(text)
    if not expense:
        return {'error': 'Could not parse expense. Please include an amount.'}
    
    # Get or create spreadsheet
    spreadsheet_id = find_or_create_spreadsheet()
    sheets = get_sheets_service()
    
    # Append row
    values = [[
        expense['date'],
        expense['amount'],
        expense['category'],
        expense['description'],
        notes
    ]]
    
    sheets.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range='Expenses!A:E',
        valueInputOption='USER_ENTERED',
        body={'values': values}
    ).execute()
    
    return {
        'success': True,
        'expense': expense,
        'spreadsheet_url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
    }

def get_summary(period='today'):
    """Get expense summary."""
    spreadsheet_id = find_or_create_spreadsheet()
    sheets = get_sheets_service()
    
    result = sheets.spreadsheets().values().get(
        spreadsheetId=spreadsheet_id,
        range='Expenses!A:E'
    ).execute()
    
    rows = result.get('values', [])[1:]  # Skip header
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    total = 0
    by_category = {}
    
    for row in rows:
        if len(row) >= 3:
            date = row[0][:10] if row[0] else ''
            
            if period == 'today' and date != today:
                continue
            
            amount = float(row[1]) if row[1] else 0
            category = row[2] if len(row) > 2 else 'Other'
            
            total += amount
            by_category[category] = by_category.get(category, 0) + amount
    
    return {
        'period': period,
        'total': total,
        'by_category': by_category
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        text = ' '.join(sys.argv[1:])
        result = log_expense(text)
        
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            e = result['expense']
            print(f"✅ Logged: {e['amount']} THB")
            print(f"   Category: {e['category']}")
            print(f"   Description: {e['description']}")
            print(f"   📊 {result['spreadsheet_url']}")
    else:
        print("Usage: python expense_tracker.py <expense description>")
        print("Example: python expense_tracker.py Paid 308 for lunch")
