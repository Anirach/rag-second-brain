#!/usr/bin/env python3
"""
Daily API Cost Report Generator
Generates a DOCX report of yesterday's API usage costs
"""

import json
import os
from datetime import datetime, timedelta
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

SESSIONS_DIR = "/home/clawdbot/.clawdbot/agents/main/sessions"
OUTPUT_DIR = "/home/clawdbot/clawd/cost_reports"
MONTHLY_HOSTING_COST = 10.0

def get_costs_for_date(target_date):
    """Extract all API costs for a specific date from session logs"""
    date_str = target_date.strftime("%Y-%m-%d")
    
    costs_by_session = {}
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_read = 0
    total_cache_write = 0
    total_cost = 0.0
    api_calls = 0
    
    files = [f for f in os.listdir(SESSIONS_DIR) if f.endswith('.jsonl') or '.jsonl.' in f]
    
    for filename in files:
        filepath = os.path.join(SESSIONS_DIR, filename)
        session_cost = 0.0
        session_calls = 0
        
        try:
            with open(filepath, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        ts = data.get('timestamp', '')
                        
                        if date_str not in ts:
                            continue
                            
                        if 'message' in data and 'usage' in data.get('message', {}):
                            usage = data['message']['usage']
                            
                            if 'cost' in usage and 'total' in usage['cost']:
                                cost = usage['cost']['total']
                                if isinstance(cost, (int, float)) and cost < 100:
                                    session_cost += cost
                                    total_cost += cost
                                    session_calls += 1
                                    api_calls += 1
                            
                            total_input_tokens += usage.get('input', 0)
                            total_output_tokens += usage.get('output', 0)
                            total_cache_read += usage.get('cacheRead', 0)
                            total_cache_write += usage.get('cacheWrite', 0)
                    except:
                        pass
        except:
            pass
        
        if session_cost > 0:
            session_name = filename[:40] if len(filename) > 40 else filename
            costs_by_session[session_name] = {
                'cost': session_cost,
                'calls': session_calls
            }
    
    return {
        'date': target_date,
        'total_cost': total_cost,
        'api_calls': api_calls,
        'input_tokens': total_input_tokens,
        'output_tokens': total_output_tokens,
        'cache_read': total_cache_read,
        'cache_write': total_cache_write,
        'sessions': costs_by_session,
        'hosting_daily': MONTHLY_HOSTING_COST / 30
    }

def create_report(data):
    """Generate DOCX report"""
    doc = Document()
    
    # Title
    title = doc.add_heading('Daily API Cost Report', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Date
    date_para = doc.add_paragraph()
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = date_para.add_run(data['date'].strftime('%B %d, %Y'))
    run.font.size = Pt(14)
    run.font.color.rgb = RGBColor(100, 100, 100)
    
    doc.add_paragraph()
    
    # Summary Section
    doc.add_heading('Cost Summary', level=1)
    
    summary_table = doc.add_table(rows=4, cols=2)
    summary_table.style = 'Table Grid'
    
    rows_data = [
        ('API Cost (Claude Opus 4.5)', f"${data['total_cost']:.2f}"),
        ('Hosting Cost (daily)', f"${data['hosting_daily']:.2f}"),
        ('Total Daily Cost', f"${data['total_cost'] + data['hosting_daily']:.2f}"),
        ('API Calls', str(data['api_calls']))
    ]
    
    for i, (label, value) in enumerate(rows_data):
        summary_table.rows[i].cells[0].text = label
        summary_table.rows[i].cells[1].text = value
        if i == 2:  # Highlight total
            for cell in summary_table.rows[i].cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
    
    doc.add_paragraph()
    
    # Token Usage Section
    doc.add_heading('Token Usage', level=1)
    
    token_table = doc.add_table(rows=4, cols=2)
    token_table.style = 'Table Grid'
    
    token_data = [
        ('Input Tokens', f"{data['input_tokens']:,}"),
        ('Output Tokens', f"{data['output_tokens']:,}"),
        ('Cache Read', f"{data['cache_read']:,}"),
        ('Cache Write', f"{data['cache_write']:,}")
    ]
    
    for i, (label, value) in enumerate(token_data):
        token_table.rows[i].cells[0].text = label
        token_table.rows[i].cells[1].text = value
    
    doc.add_paragraph()
    
    # Session Breakdown
    if data['sessions']:
        doc.add_heading('Session Breakdown', level=1)
        
        # Sort by cost descending
        sorted_sessions = sorted(data['sessions'].items(), key=lambda x: x[1]['cost'], reverse=True)
        
        session_table = doc.add_table(rows=len(sorted_sessions) + 1, cols=3)
        session_table.style = 'Table Grid'
        
        # Header
        session_table.rows[0].cells[0].text = 'Session'
        session_table.rows[0].cells[1].text = 'API Calls'
        session_table.rows[0].cells[2].text = 'Cost'
        for cell in session_table.rows[0].cells:
            for paragraph in cell.paragraphs:
                for run in paragraph.runs:
                    run.font.bold = True
        
        for i, (session_name, session_data) in enumerate(sorted_sessions, 1):
            session_table.rows[i].cells[0].text = session_name[:35]
            session_table.rows[i].cells[1].text = str(session_data['calls'])
            session_table.rows[i].cells[2].text = f"${session_data['cost']:.2f}"
    
    doc.add_paragraph()
    
    # Monthly Projection
    doc.add_heading('Monthly Projection', level=1)
    
    daily_avg = data['total_cost'] + data['hosting_daily']
    monthly_projection = daily_avg * 30
    
    proj_para = doc.add_paragraph()
    proj_para.add_run(f"Based on today's usage, projected monthly cost: ").font.size = Pt(12)
    run = proj_para.add_run(f"${monthly_projection:.2f}")
    run.font.size = Pt(14)
    run.font.bold = True
    
    # Footer
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run(f"Generated by Arthur 🐕 at {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}")
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(150, 150, 150)
    
    return doc

def upload_to_gdrive(file_path):
    """Upload file to Google Drive — Cost Reports subfolder"""
    try:
        import sys
        sys.path.insert(0, '/home/clawdbot/clawd/gdrive')
        from gdrive_upload import upload_file
        result = upload_file(file_path, subfolder='Daily-Reports/Cost-Reports')
        print(f"Uploaded to Google Drive: {result['link']}")
        return result
    except Exception as e:
        print(f"Google Drive upload failed: {e}")
        return None

def main():
    # Create temp directory for report generation
    import tempfile
    
    # Get yesterday's date
    yesterday = datetime.now() - timedelta(days=1)
    
    print(f"Generating cost report for {yesterday.strftime('%Y-%m-%d')}...")
    
    # Get costs
    data = get_costs_for_date(yesterday)
    
    # Generate report
    doc = create_report(data)
    
    # Save to temp file
    filename = f"Cost_Report_{yesterday.strftime('%Y-%m-%d')}.docx"
    output_path = os.path.join(tempfile.gettempdir(), filename)
    doc.save(output_path)
    
    print(f"Total API Cost: ${data['total_cost']:.2f}")
    print(f"Total Daily Cost: ${data['total_cost'] + data['hosting_daily']:.2f}")
    
    # Upload to Google Drive
    result = upload_to_gdrive(output_path)
    
    # Delete local temp file
    if os.path.exists(output_path):
        os.remove(output_path)
        print("Local temp file removed")
    
    return result

if __name__ == "__main__":
    main()
