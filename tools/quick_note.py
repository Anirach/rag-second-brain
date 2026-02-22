#!/usr/bin/env python3
"""
Quick Note System — Route notes to the right place in Obsidian vault.

Usage:
  python3 quick_note.py fact "CMT Paper ID is 18"
  python3 quick_note.py decision "Use LNCS format for AIiH 2026"
  python3 quick_note.py todo "Set up iPhone Obsidian sync"
  python3 quick_note.py remember "Anirach prefers English replies"
  python3 quick_note.py event "Paper v23.1 submitted to CMT"
  python3 quick_note.py person "DuckMan - ChartSense AI collaborator"
  python3 quick_note.py project "ChartSense AI - MVP complete"

Categories:
  fact      → KnowledgeGraph/Documents/Facts.md
  decision  → KnowledgeGraph/Decisions/YYYY-MM-DD-{slug}.md
  todo      → KnowledgeGraph/Action-Items/YYYY-MM-DD-{slug}.md
  remember  → MEMORY.md (appended to Lessons Learned)
  event     → KnowledgeGraph/Events/ or Daily note
  person    → KnowledgeGraph/People/{Name}.md
  project   → KnowledgeGraph/Projects/{Name}.md (update status)
"""

import sys
import os
from datetime import datetime

VAULT = '/home/clawdbot/obsidian-vault'
MEMORY = '/home/clawdbot/clawd/MEMORY.md'
DAILY_DIR = '/home/clawdbot/clawd/memory'

def slugify(text, max_len=40):
    """Create a filename-safe slug"""
    slug = text.lower().strip()
    slug = ''.join(c if c.isalnum() or c == ' ' else '' for c in slug)
    slug = '-'.join(slug.split())
    return slug[:max_len]

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def now_str():
    return datetime.now().strftime('%Y-%m-%d %H:%M')

def today_str():
    return datetime.now().strftime('%Y-%m-%d')

def note_fact(text):
    """Append to a running facts file"""
    path = f'{VAULT}/KnowledgeGraph/Documents/Facts.md'
    ensure_dir(path)
    if not os.path.exists(path):
        with open(path, 'w') as f:
            f.write('---\ntype: facts\nupdated: {}\ntags: [facts, reference]\n---\n\n# 📌 Quick Facts\n\n'.format(today_str()))
    with open(path, 'a') as f:
        f.write(f'- **[{now_str()}]** {text}\n')
    print(f'✅ Fact saved to KnowledgeGraph/Documents/Facts.md')
    return path

def note_decision(text):
    """Create a decision note"""
    slug = slugify(text)
    path = f'{VAULT}/KnowledgeGraph/Decisions/{today_str()}-{slug}.md'
    ensure_dir(path)
    content = f'''---
type: decision
date: {today_str()}
tags: [decision]
---

# Decision: {text}

**Date:** {now_str()}
**Context:** (added via quick note)
**Decision:** {text}
**Rationale:** (to be filled)

## Related
- [[Home|← Home]]
'''
    with open(path, 'w') as f:
        f.write(content)
    print(f'✅ Decision saved: {os.path.basename(path)}')
    return path

def note_todo(text):
    """Create an action item"""
    slug = slugify(text)
    path = f'{VAULT}/KnowledgeGraph/Action-Items/{today_str()}-{slug}.md'
    ensure_dir(path)
    content = f'''---
type: action-item
date: {today_str()}
status: open
tags: [todo, action-item]
---

# TODO: {text}

**Created:** {now_str()}
**Status:** 🟢 Open
**Assigned:** Arthur 🐕

## Details
{text}

## Related
- [[Home|← Home]]
'''
    with open(path, 'w') as f:
        f.write(content)
    print(f'✅ Action item saved: {os.path.basename(path)}')
    return path

def note_remember(text):
    """Append to MEMORY.md lessons learned section"""
    with open(MEMORY, 'r') as f:
        content = f.read()
    
    marker = '## 💡 Lessons Learned'
    if marker in content:
        # Find the marker and append after the next line
        idx = content.index(marker)
        # Find end of that line
        end_of_line = content.index('\n', idx)
        # Find the next line after the subtitle
        next_line = content.index('\n', end_of_line + 1)
        insert_point = next_line + 1
        new_entry = f'- **[{today_str()}]** {text}\n'
        content = content[:insert_point] + new_entry + content[insert_point:]
    else:
        content += f'\n## 💡 Lessons Learned\n\n- **[{today_str()}]** {text}\n'
    
    with open(MEMORY, 'w') as f:
        f.write(content)
    print(f'✅ Remembered in MEMORY.md')
    return MEMORY

def note_event(text):
    """Add to daily memory and KG Events"""
    # Add to daily memory
    daily_path = f'{DAILY_DIR}/{today_str()}.md'
    if os.path.exists(daily_path):
        with open(daily_path, 'a') as f:
            f.write(f'\n### Event: {text}\n- **Time:** {now_str()}\n')
    else:
        with open(daily_path, 'w') as f:
            f.write(f'# {today_str()} — Session Notes\n\n### Event: {text}\n- **Time:** {now_str()}\n')
    print(f'✅ Event logged to daily note and memory')
    return daily_path

def note_person(text):
    """Create or update a person note"""
    # Parse "Name - description" or just "Name"
    if ' - ' in text:
        name, desc = text.split(' - ', 1)
    else:
        name, desc = text, ''
    
    name = name.strip()
    slug = name.replace(' ', '-')
    path = f'{VAULT}/KnowledgeGraph/People/{slug}.md'
    ensure_dir(path)
    
    if os.path.exists(path):
        # Append to existing
        with open(path, 'a') as f:
            f.write(f'\n- **[{now_str()}]** {desc}\n')
        print(f'✅ Updated person: {name}')
    else:
        content = f'''---
type: person
name: {name}
updated: {today_str()}
tags: [person]
---

# {name}

{desc}

**Added:** {now_str()}

## Notes

## Related
- [[Home|← Home]]
'''
        with open(path, 'w') as f:
            f.write(content)
        print(f'✅ Person created: {name}')
    return path

def note_project(text):
    """Update project status"""
    if ' - ' in text:
        name, status = text.split(' - ', 1)
    else:
        name, status = text, 'Updated'
    
    name = name.strip()
    slug = name.replace(' ', '-')
    path = f'{VAULT}/KnowledgeGraph/Projects/{slug}.md'
    ensure_dir(path)
    
    if os.path.exists(path):
        with open(path, 'a') as f:
            f.write(f'\n- **[{now_str()}]** {status}\n')
        print(f'✅ Updated project: {name}')
    else:
        content = f'''---
type: project
name: {name}
status: active
updated: {today_str()}
tags: [project]
---

# {name}

**Status:** {status}
**Updated:** {now_str()}

## Timeline
- **[{now_str()}]** {status}

## Related
- [[Home|← Home]]
'''
        with open(path, 'w') as f:
            f.write(content)
        print(f'✅ Project created: {name}')
    return path

HANDLERS = {
    'fact': note_fact,
    'decision': note_decision,
    'todo': note_todo,
    'remember': note_remember,
    'event': note_event,
    'person': note_person,
    'project': note_project,
}

def main():
    if len(sys.argv) < 3:
        print('Usage: python3 quick_note.py <category> "<text>"')
        print(f'Categories: {", ".join(HANDLERS.keys())}')
        sys.exit(1)
    
    category = sys.argv[1].lower()
    text = ' '.join(sys.argv[2:])
    
    if category not in HANDLERS:
        print(f'❌ Unknown category: {category}')
        print(f'Valid: {", ".join(HANDLERS.keys())}')
        sys.exit(1)
    
    path = HANDLERS[category](text)
    print(f'📁 File: {path}')

if __name__ == '__main__':
    main()
