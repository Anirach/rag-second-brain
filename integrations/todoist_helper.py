#!/usr/bin/env python3
"""
Todoist Integration
- List tasks
- Add tasks
- Complete tasks
- Get overdue items
"""

import os
import json
import requests
from datetime import datetime, date

# Config file for API key
CONFIG_FILE = '/home/clawdbot/clawd/integrations/todoist_config.json'

def get_api_key():
    """Get Todoist API key from config."""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            config = json.load(f)
            return config.get('api_key')
    return None

def set_api_key(api_key):
    """Save Todoist API key to config."""
    config = {'api_key': api_key}
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f)
    print(f"✅ API key saved to {CONFIG_FILE}")

def api_request(method, endpoint, data=None):
    """Make Todoist API request."""
    api_key = get_api_key()
    if not api_key:
        return {'error': 'Todoist API key not configured. Run: python todoist_helper.py setup <API_KEY>'}
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    url = f'https://api.todoist.com/rest/v2/{endpoint}'
    
    if method == 'GET':
        resp = requests.get(url, headers=headers, params=data)
    elif method == 'POST':
        resp = requests.post(url, headers=headers, json=data)
    elif method == 'DELETE':
        resp = requests.delete(url, headers=headers)
    else:
        return {'error': f'Unknown method: {method}'}
    
    if resp.status_code in [200, 204]:
        return resp.json() if resp.text else {'success': True}
    else:
        return {'error': f'API error {resp.status_code}: {resp.text}'}

def get_tasks(filter_str=None):
    """Get tasks, optionally filtered."""
    params = {}
    if filter_str:
        params['filter'] = filter_str
    
    return api_request('GET', 'tasks', params)

def get_overdue():
    """Get overdue tasks."""
    return get_tasks('overdue')

def get_today():
    """Get today's tasks."""
    return get_tasks('today')

def add_task(content, due_string=None, priority=None, project_id=None):
    """Add a new task."""
    data = {'content': content}
    
    if due_string:
        data['due_string'] = due_string
    if priority:
        data['priority'] = priority  # 1-4, 4 is highest
    if project_id:
        data['project_id'] = project_id
    
    return api_request('POST', 'tasks', data)

def complete_task(task_id):
    """Mark task as complete."""
    return api_request('POST', f'tasks/{task_id}/close')

def get_projects():
    """Get all projects."""
    return api_request('GET', 'projects')

def format_task(task):
    """Format task for display."""
    priority_icons = {1: '⚪', 2: '🔵', 3: '🟡', 4: '🔴'}
    icon = priority_icons.get(task.get('priority', 1), '⚪')
    
    due = task.get('due', {})
    due_str = due.get('string', '') if due else ''
    
    return f"{icon} {task['content']}" + (f" (📅 {due_str})" if due_str else "")

def generate_summary():
    """Generate task summary for briefing."""
    lines = []
    
    # Overdue
    overdue = get_overdue()
    if isinstance(overdue, list) and overdue:
        lines.append(f"⚠️ **Overdue** ({len(overdue)})")
        for task in overdue[:5]:
            lines.append(f"   • {format_task(task)}")
        if len(overdue) > 5:
            lines.append(f"   ...and {len(overdue) - 5} more")
    
    # Today
    today = get_today()
    if isinstance(today, list) and today:
        lines.append(f"\n📋 **Today** ({len(today)})")
        for task in today[:5]:
            lines.append(f"   • {format_task(task)}")
        if len(today) > 5:
            lines.append(f"   ...and {len(today) - 5} more")
    
    if not lines:
        lines.append("✅ No tasks for today!")
    
    return '\n'.join(lines)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Todoist Helper")
        print("Usage:")
        print("  python todoist_helper.py setup <API_KEY>  - Set API key")
        print("  python todoist_helper.py today            - Show today's tasks")
        print("  python todoist_helper.py overdue          - Show overdue tasks")
        print("  python todoist_helper.py add <task>       - Add a task")
        print("  python todoist_helper.py summary          - Get briefing summary")
        sys.exit(0)
    
    cmd = sys.argv[1]
    
    if cmd == 'setup' and len(sys.argv) > 2:
        set_api_key(sys.argv[2])
    
    elif cmd == 'today':
        tasks = get_today()
        if 'error' in tasks:
            print(f"❌ {tasks['error']}")
        else:
            print(f"📋 Today's Tasks ({len(tasks)}):\n")
            for task in tasks:
                print(f"  {format_task(task)}")
    
    elif cmd == 'overdue':
        tasks = get_overdue()
        if 'error' in tasks:
            print(f"❌ {tasks['error']}")
        else:
            print(f"⚠️ Overdue Tasks ({len(tasks)}):\n")
            for task in tasks:
                print(f"  {format_task(task)}")
    
    elif cmd == 'add' and len(sys.argv) > 2:
        content = ' '.join(sys.argv[2:])
        result = add_task(content)
        if 'error' in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ Added: {result.get('content', content)}")
    
    elif cmd == 'summary':
        print(generate_summary())
    
    else:
        print(f"Unknown command: {cmd}")
