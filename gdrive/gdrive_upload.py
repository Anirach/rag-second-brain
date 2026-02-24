#!/usr/bin/env python3
"""
Google Drive Upload Helper
Upload files to ArthurBotData folder with subfolder support
"""

import os
import sys
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

SCOPES = ['https://www.googleapis.com/auth/drive.file']
TOKEN_FILE = '/home/clawdbot/clawd/gdrive/token.json'
FOLDER_NAME = 'ArthurBotData'

def get_service():
    """Get authenticated Drive service"""
    creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(TOKEN_FILE, 'w') as token:
            token.write(creds.to_json())
    return build('drive', 'v3', credentials=creds)

def get_or_create_folder(service, folder_name, parent_id=None):
    """Get folder ID or create if doesn't exist"""
    # Build query
    if parent_id:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and '{parent_id}' in parents and trashed=false"
    else:
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])
    
    if files:
        return files[0]['id']
    
    # Create folder
    folder_metadata = {
        'name': folder_name,
        'mimeType': 'application/vnd.google-apps.folder'
    }
    if parent_id:
        folder_metadata['parents'] = [parent_id]
    
    folder = service.files().create(body=folder_metadata, fields='id').execute()
    print(f"📁 Created folder '{folder_name}'")
    return folder.get('id')

def get_nested_folder(service, path, root_id=None):
    """Get or create nested folder path like 'Papers/Project1/code'"""
    parts = path.strip('/').split('/')
    current_parent = root_id
    
    for part in parts:
        current_parent = get_or_create_folder(service, part, current_parent)
    
    return current_parent

def make_public(service, file_id):
    """Make file accessible to anyone with the link"""
    permission = {
        'type': 'anyone',
        'role': 'reader'
    }
    service.permissions().create(
        fileId=file_id,
        body=permission
    ).execute()

# Folders that should be PUBLIC (anyone with link can view)
PUBLIC_FOLDERS = {"AI-News", "Daily-Reports", "Generated-Images", "Team Manuals"}
# Everything else is PRIVATE by default

def move_file(service, file_id, new_folder_id):
    """Move a file to a new folder"""
    # Get current parents
    file = service.files().get(fileId=file_id, fields='parents').execute()
    previous_parents = ",".join(file.get('parents', []))
    
    # Move to new folder
    file = service.files().update(
        fileId=file_id,
        addParents=new_folder_id,
        removeParents=previous_parents,
        fields='id, name, parents'
    ).execute()
    return file

def get_versioned_name(service, folder_id, file_name):
    """Check for existing files and return versioned name if needed"""
    import re
    
    # Get base name and extension
    base, ext = os.path.splitext(file_name)
    
    # Remove existing version suffix if present (e.g., "_v2" or "_v10")
    version_pattern = r'_v(\d+)$'
    match = re.search(version_pattern, base)
    if match:
        base = re.sub(version_pattern, '', base)
    
    # Search for existing files with same base name
    query = f"name contains '{base}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, fields='files(id, name)').execute()
    existing_files = results.get('files', [])
    
    if not existing_files:
        return file_name  # No existing file, use original name
    
    # Check if exact name exists
    exact_match = any(f['name'] == file_name for f in existing_files)
    if not exact_match:
        return file_name  # No exact match, use original name
    
    # Find highest version number
    max_version = 0
    for f in existing_files:
        fname = f['name']
        fbase, fext = os.path.splitext(fname)
        
        # Check for version suffix
        match = re.search(version_pattern, fbase)
        if match:
            version = int(match.group(1))
            max_version = max(max_version, version)
        elif fname == file_name:
            # Original file exists (no version suffix = v1)
            max_version = max(max_version, 1)
    
    # Return next version
    next_version = max_version + 1
    versioned_name = f"{base}_v{next_version}{ext}"
    print(f"⚠️  File exists, creating version: {versioned_name}")
    return versioned_name

def upload_file(file_path, subfolder=None, custom_name=None):
    """Upload a file to Google Drive, optionally to a subfolder"""
    service = get_service()
    
    # Get root folder
    root_id = get_or_create_folder(service, FOLDER_NAME)
    
    # Get or create subfolder if specified
    if subfolder:
        folder_id = get_nested_folder(service, subfolder, root_id)
    else:
        folder_id = root_id
    
    file_name = custom_name or os.path.basename(file_path)
    
    # Check for existing files and get versioned name if needed
    file_name = get_versioned_name(service, folder_id, file_name)
    
    # Determine MIME type
    mime_types = {
        '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.json': 'application/json',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        '.py': 'text/x-python',
        '.tex': 'text/x-tex',
        '.md': 'text/markdown',
        '.csv': 'text/csv',
    }
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = mime_types.get(ext, 'application/octet-stream')
    
    file_metadata = {
        'name': file_name,
        'parents': [folder_id]
    }
    
    media = MediaFileUpload(file_path, mimetype=mime_type, resumable=True)
    
    file = service.files().create(
        body=file_metadata,
        media_body=media,
        fields='id, name, webViewLink'
    ).execute()
    
    # Make file publicly accessible only for public folders
    file_id = file.get('id')
    folder_name = subfolder if subfolder else ""
    top_folder = folder_name.split("/")[0] if folder_name else ""
    if top_folder in PUBLIC_FOLDERS:
        make_public(service, file_id)
    
    return {
        'id': file_id,
        'name': file.get('name'),
        'link': file.get('webViewLink')
    }

def list_files(folder_name=FOLDER_NAME, parent_id=None):
    """List files in the folder"""
    service = get_service()
    
    if parent_id:
        folder_id = parent_id
    else:
        folder_id = get_or_create_folder(service, folder_name)
    
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query,
        spaces='drive',
        fields='files(id, name, mimeType, createdTime, size)',
        orderBy='createdTime desc'
    ).execute()
    
    return results.get('files', [])

def organize_files_to_folder(file_names, target_folder):
    """Move existing files to a subfolder"""
    service = get_service()
    root_id = get_or_create_folder(service, FOLDER_NAME)
    target_id = get_nested_folder(service, target_folder, root_id)
    
    # Find and move each file
    for name in file_names:
        query = f"name='{name}' and '{root_id}' in parents and trashed=false"
        results = service.files().list(q=query, fields='files(id, name)').execute()
        files = results.get('files', [])
        
        for f in files:
            move_file(service, f['id'], target_id)
            print(f"  ✅ Moved: {f['name']}")
    
    # Make folder public
    make_public(service, target_id)
    
    # Get folder link
    folder_info = service.files().get(fileId=target_id, fields='webViewLink').execute()
    return {
        'id': target_id,
        'link': folder_info.get('webViewLink')
    }

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 gdrive_upload.py <file_path> [--folder <subfolder>]")
        print("  python3 gdrive_upload.py --list")
        print("  python3 gdrive_upload.py --organize <folder_name> <file1> <file2> ...")
        sys.exit(1)
    
    if sys.argv[1] == '--list':
        files = list_files()
        print(f"Files in {FOLDER_NAME}:")
        for f in files:
            print(f"  - {f['name']}")
    
    elif sys.argv[1] == '--organize':
        if len(sys.argv) < 4:
            print("Usage: --organize <folder_name> <file1> <file2> ...")
            sys.exit(1)
        folder_name = sys.argv[2]
        file_names = sys.argv[3:]
        print(f"📁 Organizing {len(file_names)} files into '{folder_name}'...")
        result = organize_files_to_folder(file_names, folder_name)
        print(f"\n✅ Done! Folder: {result['link']}")
    
    else:
        file_path = sys.argv[1]
        subfolder = None
        custom_name = None
        
        # Parse arguments
        i = 2
        while i < len(sys.argv):
            if sys.argv[i] == '--folder' and i + 1 < len(sys.argv):
                subfolder = sys.argv[i + 1]
                i += 2
            else:
                custom_name = sys.argv[i]
                i += 1
        
        if not os.path.exists(file_path):
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        result = upload_file(file_path, subfolder=subfolder, custom_name=custom_name)
        file_id = result['id']
        print(f"✅ Uploaded: {result['name']}")
        if subfolder:
            print(f"📁 Folder: {subfolder}")
        print(f"📎 View: {result['link']}")
        print(f"📥 Download: https://drive.google.com/uc?export=download&id={file_id}")
