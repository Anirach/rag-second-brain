#!/usr/bin/env python3
"""Google Drive file upload for ClawdBot"""
import os
import sys
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/drive.file'
]
CREDS_FILE = '/home/clawdbot/clawd/gcal/credentials.json'
TOKEN_FILE = '/home/clawdbot/clawd/gcal/token_drive.json'

def get_drive_service():
    """Get authenticated Drive service"""
    creds = None
    
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDS_FILE, SCOPES)
            auth_url, _ = flow.authorization_url(prompt='consent')
            print(f"AUTH_URL: {auth_url}")
            return None
        
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    
    return build('drive', 'v3', credentials=creds)

def find_or_create_folder(service, folder_name, parent_id=None):
    """Find folder by name or create if not exists"""
    query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    if parent_id:
        query += f" and '{parent_id}' in parents"
    
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    folders = results.get('files', [])
    
    if folders:
        return folders[0]['id']
    
    # Create folder
    file_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        file_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=file_metadata, fields='id').execute()
    print(f"Created folder: {folder_name}")
    return folder.get('id')

def upload_file(service, file_path, folder_id=None):
    """Upload a file to Google Drive"""
    file_name = os.path.basename(file_path)
    
    # Determine MIME type
    mime_types = {
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
    }
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    file_metadata = {'name': file_name}
    if folder_id:
        file_metadata['parents'] = [folder_id]
    
    media = MediaFileUpload(file_path, mimetype=mime_type)
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, webViewLink'
    ).execute()
    
    print(f"✅ Uploaded: {file_name}")
    print(f"🔗 Link: {file.get('webViewLink')}")
    return file

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 gdrive.py <file_path> [folder_name]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    folder_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    service = get_drive_service()
    if not service:
        print("Need to authenticate first!")
        sys.exit(1)
    
    folder_id = None
    if folder_name:
        folder_id = find_or_create_folder(service, folder_name)
    
    upload_file(service, file_path, folder_id)
