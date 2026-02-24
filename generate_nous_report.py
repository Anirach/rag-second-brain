#!/usr/bin/env python3
"""
Nous Feature Research Report Generator
Professional DOCX report analyzing the Nous knowledge graph system
and recommending features for research productivity.
"""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import datetime

# Colors
NAVY = RGBColor(0x1B, 0x3A, 0x5C)
BLUE = RGBColor(0x2A, 0x64, 0x96)
DARK_GRAY = RGBColor(0x44, 0x44, 0x44)
LIGHT_GRAY = RGBColor(0xF0, 0xF4, 0xF8)
MID_BLUE = RGBColor(0xE8, 0xF1, 0xF8)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
GREEN = RGBColor(0x1A, 0x7A, 0x4A)
ORANGE = RGBColor(0xD4, 0x6A, 0x00)
RED = RGBColor(0xC0, 0x28, 0x28)
GOLD = RGBColor(0xB8, 0x86, 0x00)

def set_cell_bg(cell, r, g, b):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), f'{r:02X}{g:02X}{b:02X}')
    tcPr.append(shd)

def set_cell_border(cell, top=None, bottom=None, left=None, right=None):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = OxmlElement('w:tcBorders')
    for side, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        if val:
            border = OxmlElement(f'w:{side}')
            border.set(qn('w:val'), 'single')
            border.set(qn('w:sz'), '4')
            border.set(qn('w:space'), '0')
            border.set(qn('w:color'), val)
            tcBorders.append(border)
    tcPr.append(tcBorders)

def add_page_break(doc):
    para = doc.add_paragraph()
    run = para.add_run()
    run.add_break(docx_break_type())

def docx_break_type():
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    return br

def add_page_break_proper(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(0)
    run = p.add_run()
    br = OxmlElement('w:br')
    br.set(qn('w:type'), 'page')
    run._r.append(br)

def set_para_font(para, name='Arial', size=11, bold=False, italic=False, color=None):
    for run in para.runs:
        run.font.name = name
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.italic = italic
        if color:
            run.font.color.rgb = color

def add_heading1(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(18)
    para.paragraph_format.space_after = Pt(8)
    run = para.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(16)
    run.font.bold = True
    run.font.color.rgb = NAVY
    # Add bottom border
    pPr = para._p.get_or_add_pPr()
    pBdr = OxmlElement('w:pBdr')
    bottom = OxmlElement('w:bottom')
    bottom.set(qn('w:val'), 'single')
    bottom.set(qn('w:sz'), '6')
    bottom.set(qn('w:space'), '1')
    bottom.set(qn('w:color'), '1B3A5C')
    pBdr.append(bottom)
    pPr.append(pBdr)
    return para

def add_heading2(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(14)
    para.paragraph_format.space_after = Pt(6)
    run = para.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(13)
    run.font.bold = True
    run.font.color.rgb = BLUE
    return para

def add_heading3(doc, text):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(10)
    para.paragraph_format.space_after = Pt(4)
    run = para.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(12)
    run.font.bold = True
    run.font.color.rgb = DARK_GRAY
    return para

def add_body(doc, text, bold=False, italic=False, indent=False):
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(2)
    para.paragraph_format.space_after = Pt(4)
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if indent:
        para.paragraph_format.left_indent = Inches(0.3)
    run = para.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return para

def add_bullet(doc, text, level=0, bold_prefix=None):
    para = doc.add_paragraph(style='List Bullet')
    para.paragraph_format.space_before = Pt(1)
    para.paragraph_format.space_after = Pt(2)
    para.paragraph_format.left_indent = Inches(0.3 + level * 0.25)
    if bold_prefix:
        run1 = para.add_run(bold_prefix + ': ')
        run1.font.name = 'Arial'
        run1.font.size = Pt(11)
        run1.font.bold = True
        run1.font.color.rgb = NAVY
    run = para.add_run(text)
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    return para

def add_footer_with_page_numbers(doc):
    for section in doc.sections:
        footer = section.footer
        para = footer.paragraphs[0]
        para.clear()
        para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        run1 = para.add_run('Nous Knowledge Graph — Feature Research Report  |  Page ')
        run1.font.name = 'Arial'
        run1.font.size = Pt(9)
        run1.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        
        # Add page number field
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        instrText = OxmlElement('w:instrText')
        instrText.text = 'PAGE'
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'separate')
        fldChar3 = OxmlElement('w:fldChar')
        fldChar3.set(qn('w:fldCharType'), 'end')
        
        r = OxmlElement('w:r')
        rPr = OxmlElement('w:rPr')
        rFonts = OxmlElement('w:rFonts')
        rFonts.set(qn('w:ascii'), 'Arial')
        rPr.append(rFonts)
        sz = OxmlElement('w:sz')
        sz.set(qn('w:val'), '18')
        rPr.append(sz)
        r.append(rPr)
        r.append(fldChar1)
        r.append(instrText)
        r.append(fldChar2)
        r.append(fldChar3)
        para._p.append(r)
        
        run2 = para.add_run(' of ')
        run2.font.name = 'Arial'
        run2.font.size = Pt(9)
        run2.font.color.rgb = RGBColor(0x80, 0x80, 0x80)
        
        # Total pages field
        fldChar1b = OxmlElement('w:fldChar')
        fldChar1b.set(qn('w:fldCharType'), 'begin')
        instrText2 = OxmlElement('w:instrText')
        instrText2.text = 'NUMPAGES'
        fldChar2b = OxmlElement('w:fldChar')
        fldChar2b.set(qn('w:fldCharType'), 'separate')
        fldChar3b = OxmlElement('w:fldChar')
        fldChar3b.set(qn('w:fldCharType'), 'end')
        
        r2 = OxmlElement('w:r')
        rPr2 = OxmlElement('w:rPr')
        rFonts2 = OxmlElement('w:rFonts')
        rFonts2.set(qn('w:ascii'), 'Arial')
        rPr2.append(rFonts2)
        sz2 = OxmlElement('w:sz')
        sz2.set(qn('w:val'), '18')
        rPr2.append(sz2)
        r2.append(rPr2)
        r2.append(fldChar1b)
        r2.append(instrText2)
        r2.append(fldChar2b)
        r2.append(fldChar3b)
        para._p.append(r2)
        
        run3 = para.add_run(f'  |  Confidential — {datetime.datetime.now().strftime("%B %Y")}')
        run3.font.name = 'Arial'
        run3.font.size = Pt(9)
        run3.font.color.rgb = RGBColor(0x80, 0x80, 0x80)

def create_feature_table(doc, headers, rows, header_bg=(0x1B, 0x3A, 0x5C), col_widths=None):
    table = doc.add_table(rows=1+len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    hdr = table.rows[0]
    for i, (cell, h) in enumerate(zip(hdr.cells, headers)):
        set_cell_bg(cell, *header_bg)
        p = cell.paragraphs[0]
        p.clear()
        run = p.add_run(h)
        run.font.name = 'Arial'
        run.font.size = Pt(10)
        run.font.bold = True
        run.font.color.rgb = WHITE
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    
    # Data rows
    for ri, row_data in enumerate(rows):
        row = table.rows[ri + 1]
        bg = (0xF0, 0xF4, 0xF8) if ri % 2 == 0 else (0xFF, 0xFF, 0xFF)
        for i, (cell, val) in enumerate(zip(row.cells, row_data)):
            set_cell_bg(cell, *bg)
            p = cell.paragraphs[0]
            p.clear()
            if isinstance(val, tuple):
                text, bold, color = val
            else:
                text, bold, color = str(val), False, RGBColor(0x1A, 0x1A, 0x1A)
            run = p.add_run(text)
            run.font.name = 'Arial'
            run.font.size = Pt(10)
            run.font.bold = bold
            run.font.color.rgb = color
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
    
    # Set column widths
    if col_widths:
        for row in table.rows:
            for i, cell in enumerate(row.cells):
                if i < len(col_widths):
                    cell.width = Inches(col_widths[i])
    return table

def add_info_box(doc, title, content, bg_color=(0xE8, 0xF1, 0xF8), border_color='2A6496'):
    table = doc.add_table(rows=1, cols=1)
    table.style = 'Table Grid'
    cell = table.rows[0].cells[0]
    set_cell_bg(cell, *bg_color)
    set_cell_border(cell, top=border_color, bottom=border_color, left=border_color, right=border_color)
    p = cell.paragraphs[0]
    p.clear()
    run = p.add_run(f"💡 {title}\n")
    run.font.name = 'Arial'
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = BLUE
    run2 = p.add_run(content)
    run2.font.name = 'Arial'
    run2.font.size = Pt(10)
    run2.font.color.rgb = RGBColor(0x1A, 0x1A, 0x1A)
    doc.add_paragraph()


# ============================================================
# MAIN DOCUMENT CREATION
# ============================================================
doc = Document()

# Page setup
section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.left_margin = Inches(1.0)
section.right_margin = Inches(1.0)
section.top_margin = Inches(1.0)
section.bottom_margin = Inches(1.0)
section.different_first_page_header_footer = False

# Add footer
add_footer_with_page_numbers(doc)

# ============================================================
# COVER PAGE
# ============================================================
# Title block
para = doc.add_paragraph()
para.alignment = WD_ALIGN_PARAGRAPH.CENTER
para.paragraph_format.space_before = Pt(60)
para.paragraph_format.space_after = Pt(4)
run = para.add_run('NOUS')
run.font.name = 'Arial'
run.font.size = Pt(48)
run.font.bold = True
run.font.color.rgb = NAVY

para2 = doc.add_paragraph()
para2.alignment = WD_ALIGN_PARAGRAPH.CENTER
para2.paragraph_format.space_before = Pt(0)
para2.paragraph_format.space_after = Pt(8)
run2 = para2.add_run('Knowledge Graph Research System')
run2.font.name = 'Arial'
run2.font.size = Pt(22)
run2.font.bold = False
run2.font.color.rgb = BLUE

# Divider
div_para = doc.add_paragraph()
div_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
div_para.paragraph_format.space_before = Pt(4)
div_para.paragraph_format.space_after = Pt(20)
run_div = div_para.add_run('━' * 40)
run_div.font.color.rgb = BLUE
run_div.font.name = 'Arial'

# Subtitle
para3 = doc.add_paragraph()
para3.alignment = WD_ALIGN_PARAGRAPH.CENTER
para3.paragraph_format.space_before = Pt(4)
para3.paragraph_format.space_after = Pt(4)
run3 = para3.add_run('Feature Research & Enhancement Report')
run3.font.name = 'Arial'
run3.font.size = Pt(20)
run3.font.bold = True
run3.font.color.rgb = NAVY

para4 = doc.add_paragraph()
para4.alignment = WD_ALIGN_PARAGRAPH.CENTER
para4.paragraph_format.space_before = Pt(4)
para4.paragraph_format.space_after = Pt(30)
run4 = para4.add_run('Recommendations for Research Productivity & Academic Paper Quality')
run4.font.name = 'Arial'
run4.font.size = Pt(14)
run4.font.italic = True
run4.font.color.rgb = DARK_GRAY

# Info table
info_table = doc.add_table(rows=5, cols=2)
info_table.alignment = WD_TABLE_ALIGNMENT.CENTER
info_data = [
    ('Prepared For', 'Anirach — University Lecturer & AI Engineer'),
    ('Research Focus', 'AI, ML, Education Technology, Longevity Healthcare, Biomedical AI'),
    ('System Version', 'Nous API v3.0.0 (2,688 papers | 2,497 entities | 1,489 vectors)'),
    ('Report Date', datetime.datetime.now().strftime('%B %d, %Y')),
    ('Classification', 'Confidential — Strategic Planning Document'),
]
for ri, (label, value) in enumerate(info_data):
    row = info_table.rows[ri]
    set_cell_bg(row.cells[0], 0x1B, 0x3A, 0x5C)
    set_cell_bg(row.cells[1], 0xE8, 0xF1, 0xF8)
    
    p0 = row.cells[0].paragraphs[0]
    p0.clear()
    r0 = p0.add_run(label)
    r0.font.name = 'Arial'
    r0.font.size = Pt(10)
    r0.font.bold = True
    r0.font.color.rgb = WHITE
    
    p1 = row.cells[1].paragraphs[0]
    p1.clear()
    r1 = p1.add_run(value)
    r1.font.name = 'Arial'
    r1.font.size = Pt(10)
    r1.font.color.rgb = NAVY

# Set column widths for info table
for row in info_table.rows:
    row.cells[0].width = Inches(1.8)
    row.cells[1].width = Inches(4.7)

doc.add_paragraph()

para_end = doc.add_paragraph()
para_end.alignment = WD_ALIGN_PARAGRAPH.CENTER
para_end.paragraph_format.space_before = Pt(30)
run_end = para_end.add_run('Nous Research Intelligence System  ·  2026')
run_end.font.name = 'Arial'
run_end.font.size = Pt(10)
run_end.font.color.rgb = RGBColor(0x88, 0x88, 0x88)

add_page_break_proper(doc)

# ============================================================
# TABLE OF CONTENTS
# ============================================================
add_heading1(doc, 'Table of Contents')

toc_items = [
    ('1.', 'Executive Summary', '3'),
    ('   1.1', 'Current System State', '3'),
    ('   1.2', 'Top 10 Recommended Features', '3'),
    ('2.', 'Current System Analysis', '4'),
    ('   2.1', 'System Architecture & Stack', '4'),
    ('   2.2', 'Existing Capabilities', '4'),
    ('   2.3', 'Known Issues & Gaps', '5'),
    ('3.', 'State-of-the-Art Competitive Analysis', '6'),
    ('   3.1', 'Leading Research Tools Overview', '6'),
    ('   3.2', 'Feature Comparison Matrix', '7'),
    ('4.', 'Feature Gap Analysis', '8'),
    ('5.', 'Recommended Features (Prioritized)', '9'),
    ('   5.1', 'Tier 1 — Critical (Implement First)', '9'),
    ('   5.2', 'Tier 2 — High Value', '12'),
    ('   5.3', 'Tier 3 — Nice to Have', '15'),
    ('6.', 'Quick Wins (< 1 Day Implementation)', '17'),
    ('7.', 'Implementation Roadmap', '18'),
    ('8.', 'Conclusion', '20'),
]

toc_table = doc.add_table(rows=len(toc_items), cols=3)
toc_table.style = 'Table Grid'
toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
for ri, (num, title, page) in enumerate(toc_items):
    row = toc_table.rows[ri]
    bg = (0xF8, 0xFA, 0xFC) if ri % 2 == 0 else (0xFF, 0xFF, 0xFF)
    for cell in row.cells:
        set_cell_bg(cell, *bg)
    
    is_main = not num.strip().startswith(' ')
    
    p0 = row.cells[0].paragraphs[0]
    p0.clear()
    r0 = p0.add_run(num)
    r0.font.name = 'Arial'
    r0.font.size = Pt(10)
    r0.font.bold = is_main
    r0.font.color.rgb = NAVY if is_main else BLUE
    row.cells[0].width = Inches(0.5)
    
    p1 = row.cells[1].paragraphs[0]
    p1.clear()
    r1 = p1.add_run(title)
    r1.font.name = 'Arial'
    r1.font.size = Pt(10)
    r1.font.bold = is_main
    r1.font.color.rgb = NAVY if is_main else RGBColor(0x1A, 0x1A, 0x1A)
    row.cells[1].width = Inches(5.5)
    
    p2 = row.cells[2].paragraphs[0]
    p2.clear()
    p2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    r2 = p2.add_run(page)
    r2.font.name = 'Arial'
    r2.font.size = Pt(10)
    r2.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    row.cells[2].width = Inches(0.5)

add_page_break_proper(doc)

# ============================================================
# SECTION 1: EXECUTIVE SUMMARY
# ============================================================
add_heading1(doc, '1. Executive Summary')

add_body(doc, 
    'Nous is a graph-aware Research Intelligence System built on a modern, production-grade technology stack '
    '(Next.js + FastAPI + Neo4j + Qdrant + PostgreSQL + Redis). With 2,688 indexed papers, 2,497 named entities, '
    'and 1,489 vector embeddings, it represents a substantial and growing knowledge repository — currently '
    'focused on consciousness research but architecturally capable of supporting any academic domain including '
    'AI, machine learning, education technology, longevity healthcare, and biomedical AI.')

add_body(doc,
    'This report provides a comprehensive analysis of the system\'s current state, benchmarks it against leading '
    'research productivity tools (Elicit, Connected Papers, ResearchRabbit, Semantic Scholar, Litmaps, Obsidian '
    'Research), and delivers a prioritized roadmap of features that would transform Nous from a capable knowledge '
    'store into a world-class research intelligence platform.')

add_heading2(doc, '1.1 Current System State')

status_headers = ['Component', 'Status', 'Notes']
status_rows = [
    [('API Backend (FastAPI)', True, NAVY), ('✅ Live & Healthy', False, GREEN), ('25 endpoints, v3.0.0', False, DARK_GRAY)],
    [('Papers Database', True, NAVY), ('✅ 2,688 papers indexed', False, GREEN), ('1,199 pending processing', False, ORANGE)],
    [('Vector Search (Qdrant)', True, NAVY), ('⚠️ Partial', False, ORANGE), ('Only 1,489 / 2,688 papers vectorized', False, ORANGE)],
    [('Knowledge Graph (Neo4j)', True, NAVY), ('✅ Active', False, GREEN), ('2,497 entities, 1,266 relations', False, DARK_GRAY)],
    [('RAG Q&A System', True, NAVY), ('✅ Working', False, GREEN), ('Graph-aware synthesis with citations', False, DARK_GRAY)],
    [('Literature Review Gen.', True, NAVY), ('✅ Working', False, GREEN), ('7-section structured output', False, DARK_GRAY)],
    [('6 Graph Visualizations', True, NAVY), ('✅ Working', False, GREEN), ('Co-occurrence, citation, causal, etc.', False, DARK_GRAY)],
    [('BibTeX Export', True, NAVY), ('⚠️ Requires params', False, ORANGE), ('Needs ids or topic parameter', False, ORANGE)],
    [('Frontend Proxy /api/proxy/ask', True, NAVY), ('❌ Broken (404)', False, RED), ('Next.js route not connected to backend', False, RED)],
    [('Gap Detection', True, NAVY), ('✅ Working', False, GREEN), ('Graph-based gap scoring', False, DARK_GRAY)],
    [('APA Reference Export', True, NAVY), ('✅ Available', False, GREEN), ('Requires ids or topic', False, DARK_GRAY)],
    [('Paper Ingestion Pipeline', True, NAVY), ('⚠️ Incomplete', False, ORANGE), ('44.7% papers not yet chunked', False, ORANGE)],
]
create_feature_table(doc, status_headers, status_rows, col_widths=[2.0, 1.8, 2.7])

add_heading2(doc, '1.2 Top 10 Recommended Features')

top10_headers = ['Rank', 'Feature', 'Impact', 'Effort']
top10_rows = [
    [('1', True, NAVY), ('Smart Literature Review Assistant with Citation Export', True, NAVY), ('🔴 Critical', False, RED), ('Medium', False, DARK_GRAY)],
    [('2', True, NAVY), ('Fix Frontend Proxy & Add Streaming Response', True, NAVY), ('🔴 Critical', False, RED), ('Low', False, GREEN)],
    [('3', True, NAVY), ('Multi-Domain Paper Ingestion (arXiv, PubMed, Semantic Scholar)', True, NAVY), ('🔴 Critical', False, RED), ('Medium', False, DARK_GRAY)],
    [('4', True, NAVY), ('Advanced Research Gap Detection with Novel Hypothesis Generation', True, NAVY), ('🟠 High', False, ORANGE), ('Medium', False, DARK_GRAY)],
    [('5', True, NAVY), ('Zotero/Reference Manager Integration (Export + Import)', True, NAVY), ('🟠 High', False, ORANGE), ('Low', False, GREEN)],
    [('6', True, NAVY), ('Personalized Research Feed & New Paper Alerts', True, NAVY), ('🟠 High', False, ORANGE), ('Medium', False, DARK_GRAY)],
    [('7', True, NAVY), ('Paper Annotation, Notes & Highlight System', True, NAVY), ('🟠 High', False, ORANGE), ('Medium', False, DARK_GRAY)],
    [('8', True, NAVY), ('Interactive Citation Network Visualization (Seed-based)', True, NAVY), ('🟡 Medium', False, GOLD), ('Medium', False, DARK_GRAY)],
    [('9', True, NAVY), ('Research Question Decomposition & Methodology Suggester', True, NAVY), ('🟡 Medium', False, GOLD), ('High', False, RED)],
    [('10', True, NAVY), ('Multi-Format Export (LaTeX, Obsidian, Word, Notion)', True, NAVY), ('🟡 Medium', False, GOLD), ('Low', False, GREEN)],
]
create_feature_table(doc, top10_headers, top10_rows, col_widths=[0.4, 3.5, 1.0, 0.8])

add_page_break_proper(doc)

# ============================================================
# SECTION 2: CURRENT SYSTEM ANALYSIS
# ============================================================
add_heading1(doc, '2. Current System Analysis')

add_heading2(doc, '2.1 System Architecture & Stack')

add_body(doc, 
    'Nous is built on a well-architected, modern microservices stack that provides excellent foundations '
    'for future feature development. The architecture separates concerns effectively:')

arch_data = [
    [('Layer', True, NAVY), ('Technology', True, NAVY), ('Purpose', True, NAVY), ('Strength', True, NAVY)],
]
arch_rows = [
    [('Frontend', False, DARK_GRAY), ('Next.js (React)', False, DARK_GRAY), ('Research UI, visualizations', False, DARK_GRAY), ('Modern, SSR-capable, component ecosystem', False, GREEN)],
    [('API Backend', False, DARK_GRAY), ('FastAPI (Python)', False, DARK_GRAY), ('REST API, RAG orchestration', False, DARK_GRAY), ('Fast, async, auto-docs, type-safe', False, GREEN)],
    [('Knowledge Graph', False, DARK_GRAY), ('Neo4j', False, DARK_GRAY), ('Entity relations, graph queries', False, DARK_GRAY), ('ACID-compliant, Cypher query language', False, GREEN)],
    [('Vector Store', False, DARK_GRAY), ('Qdrant', False, DARK_GRAY), ('Semantic similarity search', False, DARK_GRAY), ('Fast ANN, payload filtering, production-ready', False, GREEN)],
    [('Relational DB', False, DARK_GRAY), ('PostgreSQL', False, DARK_GRAY), ('Papers, metadata, chunks', False, DARK_GRAY), ('ACID, full-text search, JSON support', False, GREEN)],
    [('Cache/Queue', False, DARK_GRAY), ('Redis', False, DARK_GRAY), ('Session cache, async jobs', False, DARK_GRAY), ('Sub-ms latency, pub/sub, streams', False, GREEN)],
]
create_feature_table(doc, ['Layer', 'Technology', 'Purpose', 'Strength'], arch_rows, col_widths=[1.1, 1.3, 1.8, 2.3])

add_body(doc, 
    'This stack is exceptionally well-suited for scaling a research intelligence system. The combination of '
    'Neo4j (graph relationships), Qdrant (semantic similarity), and PostgreSQL (structured metadata) provides '
    'a hybrid search capability that most commercial research tools cannot match.', italic=True)

add_heading2(doc, '2.2 Existing Capabilities')

add_heading3(doc, 'Core Research Functions')
add_body(doc,
    'The API exposes 25 endpoints across 5 functional categories. The current knowledge base focuses on '
    'consciousness research (IIT, Global Workspace Theory, panpsychism, phenomenology) but the architecture '
    'is domain-agnostic and ready for AI/ML, biomedical, and education technology research domains.')

caps_list = [
    ('Graph-Aware RAG Q&A', 'Combines Qdrant semantic search + Neo4j graph traversal + LLM synthesis to answer research questions with citations and source tracking. Returns question, answer, sources, entities_used, graph_nodes, and graph_edges.'),
    ('Semantic Search', 'Vector-similarity search over paper chunks with metadata (title, year, venue, authors, citation count, DOI, URL). Uses cosine similarity with scores.'),
    ('Literature Review Generation', 'Automated 7-section structured literature review (Introduction, Theoretical Foundations, plus domain sections). Returns title, sections with content, and reference list.'),
    ('Entity Knowledge Graph', 'Extracts and categorizes 2,497 entities across 70+ types (concept, theory, researcher, methodology, technique, brain_region, etc.) with paper counts and descriptions.'),
    ('Research Gap Detection', 'Graph-based gap detection finding entity pairs with high co-occurrence but no direct RELATES edge in Neo4j. Scores gaps by co-occurrence frequency.'),
    ('6 Graph Visualizations', 'Co-occurrence, knowledge, citation, taxonomy, debate, and causal graphs for visual knowledge exploration.'),
    ('Theory Comparison', 'Structured comparison of research schools (materialism, functionalism, panpsychism) with key claims, top researchers, and representative papers.'),
    ('BibTeX & APA Export', 'Reference export in standard citation formats.'),
]
for cap, desc in caps_list:
    add_bullet(doc, desc, bold_prefix=cap)

add_heading3(doc, 'Data Coverage Statistics')
stats_rows = [
    [('Papers Total', False, DARK_GRAY), ('2,688', False, NAVY), ('Fully chunked', False, DARK_GRAY), ('937 (34.9%)', False, ORANGE)],
    [('Entities', False, DARK_GRAY), ('2,497', False, NAVY), ('Entities extracted', False, DARK_GRAY), ('552 papers (20.5%)', False, ORANGE)],
    [('Vector Embeddings', False, DARK_GRAY), ('1,489', False, NAVY), ('Pending processing', False, DARK_GRAY), ('1,199 papers (44.6%)', False, RED)],
    [('Graph Relations', False, DARK_GRAY), ('1,266', False, NAVY), ('Entity types', False, DARK_GRAY), ('70+ types', False, GREEN)],
]
create_feature_table(doc, ['Metric', 'Count', 'Status Metric', 'Value'], stats_rows, col_widths=[1.5, 1.0, 1.8, 2.2])

add_heading2(doc, '2.3 Known Issues & Gaps')

issues = [
    ('CRITICAL: Broken Frontend Proxy', 
     'The Next.js frontend route /api/proxy/ask returns 404. The frontend cannot communicate with the '
     'FastAPI backend through the proxy layer. This is likely a missing Next.js API route handler at '
     'web/app/api/proxy/ask/route.ts or a misconfigured rewrite rule. Users cannot use the Q&A '
     'feature through the web interface.'),
    ('CRITICAL: Incomplete Vectorization Pipeline',
     '44.6% of papers (1,199 of 2,688) remain in "pending" status — not yet chunked or vectorized. '
     'This severely limits the recall of semantic search and RAG responses. The ingestion pipeline '
     'needs monitoring, error recovery, and backfill processing.'),
    ('HIGH: No User Authentication / Multi-User Support',
     'The API has no authentication layer visible in the OpenAPI spec. Personal reading lists, '
     'annotations, and research profiles cannot be maintained per-user. Collaboration features '
     'are impossible without this foundation.'),
    ('HIGH: No Paper Ingestion UI / API',
     'There is no visible endpoint for adding new papers (POST /papers). Research domains beyond '
     'consciousness (AI, ML, biomedical) cannot be populated without a proper ingestion interface.'),
    ('MEDIUM: Limited Citation Export Functionality',
     'BibTeX/APA export requires explicit "ids" or "topic" parameters. There is no bulk export, '
     'no collection-based export, and no integration with external citation managers like Zotero.'),
    ('MEDIUM: No Streaming Response for Q&A',
     'The /ask endpoint returns a complete response synchronously. For complex queries, this creates '
     'noticeable latency with no progress feedback. Server-sent events or WebSocket streaming would '
     'dramatically improve perceived performance.'),
    ('LOW: Entity Type Inconsistency',
     'Stats show both "researchers" (20 entities) and "researcher" (83 entities) as separate types, '
     'indicating inconsistent NER extraction. Similar deduplication issues exist for theories and '
     'other entity types. This reduces graph quality and gap detection accuracy.'),
]

for issue_title, issue_desc in issues:
    add_heading3(doc, issue_title)
    add_body(doc, issue_desc, indent=True)

add_page_break_proper(doc)

# ============================================================
# SECTION 3: COMPETITIVE ANALYSIS
# ============================================================
add_heading1(doc, '3. State-of-the-Art Competitive Analysis')

add_heading2(doc, '3.1 Leading Research Tools Overview')

tools_overview = [
    ('Elicit', 'AI-powered research assistant', 
     'Automated structured extraction (RCT data, methods, outcomes), column-based paper comparison, '
     'research question decomposition, claim verification with citations, systematic review workflows, '
     'CSV/Zotero export. Excels at literature synthesis and data extraction.',
     'Data extraction, systematic reviews'),
    ('Connected Papers', 'Visual citation network explorer',
     'Seed-paper based visual graphs showing prior work and derivative papers, cluster detection, '
     'research community mapping, influence scoring. One-click exploration from any Semantic Scholar paper.',
     'Citation discovery, field mapping'),
    ('ResearchRabbit', 'AI paper discovery & tracking',
     'Zotero integration, recommendation engine based on reading history, author network exploration, '
     'email alerts for new papers, collaborative collections, note-taking, timeline view of field evolution.',
     'Paper discovery, monitoring'),
    ('Semantic Scholar', 'Academic search engine',
     'AI-generated TLDRs, citation context (why papers cite each other), author influence metrics, '
     'open access detection, citation velocity tracking, field classification, full API access.',
     'Paper search, citation analysis'),
    ('Litmaps', 'Temporal citation maps',
     'Interactive temporal citation networks, grow/discover papers from seed, map evolution over time, '
     'citation export. Strong at showing how a field evolved.',
     'Historical field evolution'),
    ('Obsidian + Zotero', 'Personal knowledge management',
     'Bidirectional wiki-links, graph view, PDF annotation, Zotero integration, custom templates, '
     'Dataview plugin for custom queries, community plugins ecosystem.',
     'Personal knowledge base'),
    ('Scite.ai', 'Citation intelligence',
     'Smart citation context (supporting, mentioning, contrasting), claim verification, '
     'journal-level citation analysis, author credibility scoring.',
     'Citation quality assessment'),
    ('Consensus.app', 'AI research consensus',
     'Direct yes/no answers with evidence synthesis, consensus meter across papers, '
     'study type filtering, evidence quality scoring.',
     'Quick evidence summaries'),
]

add_body(doc, 
    'The following tools represent the current state-of-the-art in research productivity. Each has been '
    'analyzed for features that Nous could adopt or surpass:')

tools_headers = ['Tool', 'Specialty', 'Key Differentiating Features', 'Best For']
tools_rows = []
for name, specialty, features, best_for in tools_overview:
    tools_rows.append([
        (name, True, NAVY),
        (specialty, False, DARK_GRAY),
        (features, False, RGBColor(0x1A, 0x1A, 0x1A)),
        (best_for, False, BLUE)
    ])

create_feature_table(doc, tools_headers, tools_rows, col_widths=[1.2, 1.4, 3.2, 1.5])

add_heading2(doc, '3.2 Feature Comparison Matrix — Nous vs. Competitors')

comp_headers = ['Feature', 'Nous', 'Elicit', 'ConnPapers', 'RsrchRabbit', 'Sem.Scholar']
comp_rows = [
    [('Semantic Search', False, DARK_GRAY), ('✅', False, GREEN), ('✅', False, GREEN), ('❌', False, RED), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('Knowledge Graph', False, DARK_GRAY), ('✅', False, GREEN), ('❌', False, RED), ('⚠️', False, ORANGE), ('❌', False, RED), ('❌', False, RED)],
    [('RAG Q&A Assistant', False, DARK_GRAY), ('✅', False, GREEN), ('✅', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('Literature Review Gen.', False, DARK_GRAY), ('✅', False, GREEN), ('⚠️ Partial', False, ORANGE), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('Gap Detection', False, DARK_GRAY), ('✅ Unique', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('Citation Networks', False, DARK_GRAY), ('⚠️ Basic', False, ORANGE), ('❌', False, RED), ('✅ Excellent', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('Paper Annotations', False, DARK_GRAY), ('❌', False, RED), ('⚠️', False, ORANGE), ('❌', False, RED), ('✅', False, GREEN), ('❌', False, RED)],
    [('Zotero Integration', False, DARK_GRAY), ('❌', False, RED), ('✅', False, GREEN), ('❌', False, RED), ('✅', False, GREEN), ('❌', False, RED)],
    [('New Paper Alerts', False, DARK_GRAY), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('AI TLDR Summaries', False, DARK_GRAY), ('❌', False, RED), ('✅', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('✅', False, GREEN)],
    [('Author Networks', False, DARK_GRAY), ('⚠️ Basic', False, ORANGE), ('❌', False, RED), ('❌', False, RED), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('Structured Data Extraction', False, DARK_GRAY), ('❌', False, RED), ('✅ Best-in-class', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('BibTeX / APA Export', False, DARK_GRAY), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('6 Graph Types', False, DARK_GRAY), ('✅ Unique', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('Theory Comparison', False, DARK_GRAY), ('✅ Unique', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('Streaming Q&A', False, DARK_GRAY), ('❌', False, RED), ('✅', False, GREEN), ('❌', False, RED), ('❌', False, RED), ('❌', False, RED)],
    [('User Accounts / Auth', False, DARK_GRAY), ('❌', False, RED), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN)],
    [('Multi-domain Support', False, DARK_GRAY), ('⚠️ Consciousness only', False, ORANGE), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN), ('✅', False, GREEN)],
]
create_feature_table(doc, comp_headers, comp_rows, col_widths=[2.0, 1.0, 1.0, 1.1, 1.1, 1.2])

add_info_box(doc, 'Competitive Positioning',
    'Nous has two significant, genuinely unique strengths: (1) Graph-aware RAG combining Neo4j + Qdrant '
    'for superior knowledge synthesis, and (2) Multi-type graph visualizations (6 types). However, '
    'it critically lacks table-stakes features like user auth, Zotero integration, paper alerts, '
    'streaming responses, and annotation support that researchers expect from modern tools.')

add_page_break_proper(doc)

# ============================================================
# SECTION 4: FEATURE GAP ANALYSIS
# ============================================================
add_heading1(doc, '4. Feature Gap Analysis')

add_body(doc,
    'Based on live API testing, competitive benchmarking, and analysis of the system\'s architecture, '
    'the following gap analysis identifies the most impactful missing features across seven research workflow dimensions:')

gap_headers = ['Workflow Dimension', 'Current Nous State', 'Gap vs. Best-in-Class', 'Business Impact']
gap_rows = [
    [('Research Discovery', False, DARK_GRAY),
     ('Semantic search over 1,489 vectorized chunks; entity browsing', False, DARK_GRAY),
     ('No paper alerts, no external source integration (arXiv, PubMed), no recommendation engine based on reading history', False, DARK_GRAY),
     ('HIGH — missing papers = incomplete literature review', False, RED)],
    
    [('Writing Assistance', False, DARK_GRAY),
     ('7-section lit review generation; basic RAG Q&A', False, DARK_GRAY),
     ('No in-text citation insertion, no LaTeX/Word export of generated content, no structured abstract writer', False, DARK_GRAY),
     ('HIGH — researchers manually re-type generated content', False, RED)],
    
    [('Gap Identification', False, DARK_GRAY),
     ('Graph-based co-occurrence gap detection with scoring', False, DARK_GRAY),
     ('No LLM-powered hypothesis generation from gaps, no trend analysis, no temporal gap tracking', False, DARK_GRAY),
     ('MEDIUM — gap detection exists but lacks actionability', False, ORANGE)],
    
    [('Knowledge Synthesis', False, DARK_GRAY),
     ('Theory comparison, causal/debate/taxonomy graphs', False, DARK_GRAY),
     ('No cross-paper claim extraction, no contradiction detection, no structured comparison tables (like Elicit)', False, DARK_GRAY),
     ('HIGH — cannot extract structured data from papers', False, RED)],
    
    [('Citation Management', False, DARK_GRAY),
     ('BibTeX/APA export with topic/id filters', False, DARK_GRAY),
     ('No Zotero integration, no collection management, no in-document citation insertion', False, DARK_GRAY),
     ('HIGH — citation workflow is entirely manual', False, RED)],
    
    [('Visualization', False, DARK_GRAY),
     ('6 graph types, entity network, timeline, author network', False, DARK_GRAY),
     ('No interactive seed-paper-based exploration (like Connected Papers), no temporal field evolution maps', False, DARK_GRAY),
     ('MEDIUM — graphs exist but lack interactivity', False, ORANGE)],
    
    [('Collaboration', False, DARK_GRAY),
     ('None — no authentication layer detected', False, RED),
     ('No shared collections, no annotation sharing, no team workspaces, no reading lists', False, DARK_GRAY),
     ('HIGH — unusable for collaborative research teams', False, RED)],
    
    [('Workflow Automation', False, DARK_GRAY),
     ('None detected', False, RED),
     ('No paper ingestion UI/API, no new paper alerts, no scheduled searches, no PDF auto-processing', False, DARK_GRAY),
     ('HIGH — manual data entry limits knowledge base growth', False, RED)],
]
create_feature_table(doc, gap_headers, gap_rows, col_widths=[1.5, 1.7, 2.2, 1.1])

add_page_break_proper(doc)

# ============================================================
# SECTION 5: RECOMMENDED FEATURES
# ============================================================
add_heading1(doc, '5. Recommended Features (Prioritized)')

add_body(doc,
    'Features are organized into three priority tiers based on impact on research productivity, '
    'implementation complexity, and alignment with Anirach\'s research domains (AI, ML, education '
    'technology, longevity healthcare, biomedical AI). Each feature includes detailed specification '
    'to enable immediate implementation planning.')

# TIER 1
add_heading2(doc, '5.1 Tier 1 — Critical (Implement First)')

tier1_color = (0xC0, 0x28, 0x28)

# Feature 1
add_heading3(doc, 'T1.1 — Fix Frontend Proxy & Enable Streaming RAG')

feat1_headers = ['Attribute', 'Details']
feat1_rows = [
    [('What It Does', True, NAVY), ('Restores the broken /api/proxy/ask route in Next.js and adds Server-Sent Events (SSE) streaming for real-time token-by-token response display', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('The Q&A feature is completely inaccessible through the web UI. This is the #1 usability blocker. Streaming eliminates the "frozen UI" feeling during complex queries.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Researchers can use the Q&A interface to get synthesized answers with citations, directly supporting literature review writing.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Create web/app/api/proxy/ask/route.ts as Next.js API handler that forwards to FastAPI. Add ReadableStream for SSE. FastAPI: replace synchronous return with StreamingResponse using async generator.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('LOW — 2-4 hours for a senior developer. No architectural changes required.', False, GREEN)],
    [('Reference Tools', True, NAVY), ('Perplexity, ChatGPT, Claude — all use streaming responses as table stakes', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat1_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 2
add_heading3(doc, 'T1.2 — Multi-Domain Paper Ingestion System')

feat2_rows = [
    [('What It Does', True, NAVY), ('A paper ingestion API and UI supporting: (1) DOI/arXiv ID auto-fetch, (2) PDF upload with auto-parsing, (3) Semantic Scholar API integration, (4) PubMed/bioRxiv/medRxiv connectors for Anirach\'s biomedical AI domain', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('Currently no way to add papers to the knowledge base through the API. The system is frozen at its initial corpus. Anirach needs papers on AI/ML, education tech, longevity, and biomedical AI — not just consciousness research.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Directly determines the quality of literature reviews. A system with only consciousness papers cannot produce comprehensive reviews for AI healthcare papers.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('POST /papers endpoint with DOI resolver, PDF extractor (pdfplumber/PyMuPDF), Semantic Scholar API client, async processing queue via Redis, webhook notification on completion.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('MEDIUM — 2-3 days. Redis queue already exists. Need PDF parser + metadata fetcher + background worker.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('Zotero (one-click browser save), ResearchRabbit (auto-discovery), Elicit (bulk import)', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat2_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 3
add_heading3(doc, 'T1.3 — Enhanced Literature Review with Export (LaTeX/Word/Zotero)')

feat3_rows = [
    [('What It Does', True, NAVY), ('Upgrades the existing /literature-review endpoint to produce: (1) Structured sections with in-text citations [Author, Year], (2) One-click export to LaTeX .tex, Microsoft Word .docx, and Zotero-compatible RIS format', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('The current endpoint generates text but researchers must manually copy, format, and add citations. This friction kills adoption. A researcher should be able to generate a 5-page lit review draft and immediately open it in Word or LaTeX.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('CRITICAL for paper writing. Lit review quality directly determines acceptance rates. With proper citations and export, Nous can produce submission-ready literature review sections.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Add citation injection to review generator. Create export endpoints: GET /literature-review/export?format=latex|docx|ris&topic=X. Use python-docx for Word, pylatex for LaTeX.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('MEDIUM — 2-3 days. Review generation already works; need citation injection + export formatters.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('Elicit (best-in-class lit review export), Overleaf (LaTeX workflow)', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat3_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 4
add_heading3(doc, 'T1.4 — Complete Vectorization Pipeline with Backfill & Monitoring')

feat4_rows = [
    [('What It Does', True, NAVY), ('Automated pipeline that: (1) Processes the 1,199 pending papers, (2) Dashboard showing processing queue status, (3) Error recovery for failed papers, (4) Incremental processing of newly added papers', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('44.6% of the knowledge base is invisible to semantic search and RAG. Every pending paper is a missing citation in someone\'s literature review.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Directly increases recall coverage of the RAG system. More vectors = better answers = more comprehensive literature reviews.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Background Celery/Redis task for backfill. GET /pipeline/status endpoint. POST /pipeline/reprocess for manual trigger. Admin dashboard panel in frontend.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('LOW-MEDIUM — Redis already exists. Need background worker, status endpoint, and minimal UI.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('Semantic Scholar (full-text processing), OpenAlex (batch processing APIs)', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat4_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 5
add_heading3(doc, 'T1.5 — Advanced Research Gap Detection with Hypothesis Generation')

feat5_rows = [
    [('What It Does', True, NAVY), ('Upgrades gap detection to: (1) LLM-generated research hypothesis for each identified gap (e.g., "Tononi\'s IIT has not been empirically connected to Global Workspace Theory — a potential bridge study could..."), (2) Related paper suggestions that partially address the gap, (3) Gap trend tracking over time', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('The current gap detection returns entity pairs but leaves interpretation to the researcher. Adding LLM synthesis transforms gaps into actionable research opportunities — the "so what" that drives paper ideas.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Directly generates novel research questions suitable for paper proposals. A well-identified gap with hypothesis = the foundation of a publishable paper.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Enhance /gaps endpoint: after gap detection, call LLM to generate hypothesis paragraph for top gaps. Add gap_hypothesis field to response. Add temporal tracking via paper years.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('MEDIUM — Gap detection works. Need LLM post-processing step. ~1-2 days.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('Unique to Nous — no mainstream tool offers this. Academic analogy: Elicit\'s "unsettled questions" feature.', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat5_rows, col_widths=[1.5, 5.0])

add_page_break_proper(doc)

# TIER 2
add_heading2(doc, '5.2 Tier 2 — High Value Features')

# Feature 6
add_heading3(doc, 'T2.1 — Paper Annotation & Personal Notes System')

feat6_rows = [
    [('What It Does', True, NAVY), ('Per-paper annotation system allowing: (1) Highlight + note on paper sections, (2) Personal tags (e.g., "use-in-thesis", "contradicts-hypothesis"), (3) Reading status (to-read, reading, read, cite), (4) Personal rating/relevance score', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('Researchers need to maintain a personal layer of context over the knowledge base. Currently, Nous is read-only — there is no way to mark papers as relevant, add personal notes, or track reading progress.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Personal annotations are the bridge between discovered knowledge and written output. Notes like "use methodology from Section 3" directly feed into paper writing.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Add annotations table in PostgreSQL: user_id (session or email), paper_id, note_text, tags[], status, rating. REST endpoints: POST/GET/DELETE /papers/{id}/notes. Requires basic auth (JWT session or API key).', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('MEDIUM — Requires basic auth implementation first. Then ~2 days for annotation CRUD.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('ResearchRabbit (notes), Zotero (annotations), Readwise (highlights)', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat6_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 7
add_heading3(doc, 'T2.2 — Personalized Research Feed & New Paper Alerts')

feat7_rows = [
    [('What It Does', True, NAVY), ('(1) Daily/weekly digest of new papers matching saved research topics, (2) Citation alerts when papers in the knowledge base receive new citations, (3) "Trending in your domain" highlights, (4) Integration with Semantic Scholar\'s API for live external paper discovery', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('Research is fast-moving. Missing a key paper published last week means an incomplete literature review. Automated monitoring removes the need for manual arXiv/Google Scholar checks.', False, DARK_GRAY)],
    [('Paper Quality Impact', True, NAVY), ('Ensures literature reviews are current. Reviewers frequently reject papers for "missing recent work." Automated alerts prevent this.', False, DARK_GRAY)],
    [('Implementation', True, NAVY), ('Cron job + Redis queue: daily fetch from Semantic Scholar API for saved topics → auto-ingest relevant papers → Telegram/email digest. ~3 days with existing Redis.', False, DARK_GRAY)],
    [('Complexity', True, NAVY), ('MEDIUM — Requires topic profile storage and external API integration.', False, ORANGE)],
    [('Reference Tools', True, NAVY), ('ResearchRabbit (best-in-class alerts), Google Scholar Alerts, Semantic Scholar email alerts', False, DARK_GRAY)],
]
create_feature_table(doc, ['Attribute', 'Details'], feat7_rows, col_widths=[1.5, 5.0])
doc.add_paragraph()

# Feature 8
add_heading3(doc, 'T2.3 — AI-Powered Paper TLDR & Structured Summary Cards')

feat8_rows = [
    [('What It Does', True, NAVY), ('Auto-generated structured summary card for each paper: (1) One-sentence TLDR, (2) Research question answered, (3) Methodology used, (4) Key finding (3 bullets), (5) Limitations, (6) "Cite when you need to..." sentence', False, DARK_GRAY)],
    [('Why It Matters', True, NAVY), ('Researchers scan 50-100 papers before selecting 20 to read deeply. Structured summaries enable faster triage. Semantic Scholar\'s TL