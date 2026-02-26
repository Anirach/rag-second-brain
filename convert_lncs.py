#!/usr/bin/env python3
"""Convert MGNA v3 DOCX to LNCS format."""

from docx import Document
from docx.shared import Pt, Cm, Inches, RGBColor, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import re
import copy

src = Document('Multi_Graph_LLM_Second_Brain_v3.docx')
doc = Document()

# Page setup
section = doc.sections[0]
section.page_height = Cm(29.7)
section.page_width = Cm(21.0)
section.top_margin = Cm(6.058)
section.bottom_margin = Cm(6.275)
section.left_margin = Cm(4.678)
section.right_margin = Cm(4.678)

# Default style
style = doc.styles['Normal']
font = style.font
font.name = 'Times New Roman'
font.size = Pt(10)
pf = style.paragraph_format
pf.space_before = Pt(0)
pf.space_after = Pt(6)
pf.line_spacing = Pt(12)
pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
pf.first_line_indent = Cm(0.5)

# Set East Asia font
rpr = style.element.get_or_add_rPr()
rFonts = rpr.find(qn('w:rFonts'))
if rFonts is None:
    rFonts = OxmlElement('w:rFonts')
    rpr.append(rFonts)
rFonts.set(qn('w:eastAsia'), 'Times New Roman')

def add_paragraph(text, font_size=10, bold=False, italic=False, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  space_before=0, space_after=6, first_indent=None, font_name='Times New Roman',
                  left_indent=None, right_indent=None):
    p = doc.add_paragraph()
    pf = p.paragraph_format
    pf.space_before = Pt(space_before)
    pf.space_after = Pt(space_after)
    pf.line_spacing = Pt(12)
    pf.alignment = alignment
    if first_indent is not None:
        pf.first_line_indent = Cm(first_indent) if first_indent > 0 else Pt(0)
    else:
        pf.first_line_indent = Pt(0)
    if left_indent is not None:
        pf.left_indent = Cm(left_indent)
    if right_indent is not None:
        pf.right_indent = Cm(right_indent)
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    return p

def add_run_to_para(p, text, font_size=10, bold=False, italic=False, font_name='Times New Roman'):
    run = p.add_run(text)
    run.font.name = font_name
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    return run

# Collect all paragraphs
paras = [(p.text, p.style.name if p.style else 'Normal') for p in src.paragraphs]

# State machine
i = 0
after_heading = True  # first para after heading: no indent
in_references = False
in_abstract = False
abstract_text = []
section_counter = [0, 0, 0]  # for renumbering if needed

# Helper: detect if text is an ASCII art diagram line
def is_diagram_line(t):
    t = t.strip()
    if not t:
        return False
    # Lines with box-drawing chars or arrows
    if any(c in t for c in '┌┐└┘│─├┤┬┴┼▼▶►▲◄─═║╔╗╚╝╠╣╦╩╬'):
        return True
    if t.startswith('│') or t.startswith('┌') or t.startswith('└') or t.startswith('├'):
        return True
    return False

def is_figure_label(t):
    t = t.strip()
    return t.startswith('Figure ') and ':' in t[:20]

# Collect diagram blocks
diagram_buf = []
diagram_mode = False

while i < len(paras):
    text, sty = paras[i]
    stripped = text.strip()
    
    if not stripped:
        i += 1
        continue
    
    # Title (first two lines)
    if i == 0:
        add_paragraph(text, font_size=14, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=0, space_after=0, first_indent=0)
        i += 1
        continue
    if i == 1 and 'Unified Framework' in text:
        add_paragraph(text, font_size=14, bold=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=0, space_after=12, first_indent=0)
        i += 1
        continue
    
    # Author
    if stripped == '[Author Names]':
        add_paragraph(text, font_size=10, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=12, space_after=0, first_indent=0)
        i += 1
        continue
    
    # Affiliation
    if stripped.startswith('[Institutional'):
        add_paragraph(text, font_size=9, italic=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=0, space_after=0, first_indent=0)
        i += 1
        continue
    
    # Corresponding author
    if stripped.startswith('[Corresponding'):
        add_paragraph(text, font_size=9, italic=True, alignment=WD_ALIGN_PARAGRAPH.CENTER,
                      space_before=0, space_after=6, first_indent=0)
        i += 1
        continue
    
    # Article metadata lines (skip)
    if stripped.startswith('Article Type:') or stripped.startswith('Word Count:') or stripped.startswith('Tables:'):
        i += 1
        continue
    
    # ABSTRACT marker
    if stripped == 'ABSTRACT':
        in_abstract = True
        i += 1
        continue
    
    # Keywords line (ends abstract)
    if stripped.startswith('Keywords:'):
        # Flush abstract
        if abstract_text:
            p = doc.add_paragraph()
            pf = p.paragraph_format
            pf.space_before = Pt(6)
            pf.space_after = Pt(6)
            pf.line_spacing = Pt(12)
            pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf.left_indent = Cm(1)
            pf.right_indent = Cm(1)
            pf.first_line_indent = Pt(0)
            add_run_to_para(p, 'Abstract. ', font_size=9, bold=True)
            add_run_to_para(p, ' '.join(abstract_text), font_size=9)
            in_abstract = False
            abstract_text = []
        
        # Keywords
        kw_text = stripped[len('Keywords:'):].strip()
        # Replace semicolons with middle dots
        kw_text = kw_text.replace(';', ' ·')
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(6)
        pf.space_after = Pt(12)
        pf.line_spacing = Pt(12)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Cm(1)
        pf.right_indent = Cm(1)
        pf.first_line_indent = Pt(0)
        add_run_to_para(p, 'Keywords: ', font_size=9, bold=True)
        add_run_to_para(p, kw_text, font_size=9)
        after_heading = True
        i += 1
        continue
    
    # Collecting abstract paragraphs
    if in_abstract:
        abstract_text.append(stripped)
        i += 1
        continue
    
    # REFERENCES header
    if stripped == 'REFERENCES':
        in_references = True
        add_paragraph('References', font_size=10, bold=True, space_before=12, space_after=6,
                      first_indent=0, alignment=WD_ALIGN_PARAGRAPH.LEFT)
        after_heading = True
        i += 1
        continue
    
    # Reference entries
    if in_references and re.match(r'^\[\d+\]', stripped):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(2)
        pf.line_spacing = Pt(12)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Cm(0.5)
        pf.first_line_indent = Cm(-0.5)
        run = p.add_run(stripped)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        i += 1
        continue
    
    # Section headings: "1. INTRODUCTION" etc.
    h1_match = re.match(r'^(\d+)\.\s+(.+)$', stripped)
    h2_match = re.match(r'^(\d+\.\d+)\s+(.+)$', stripped)
    h3_match = re.match(r'^(\d+\.\d+\.\d+)\s+(.+)$', stripped)
    
    if h3_match:
        num, title = h3_match.groups()
        add_paragraph(f'{num} {title}', font_size=10, italic=True,
                      space_before=6, space_after=3, first_indent=0,
                      alignment=WD_ALIGN_PARAGRAPH.LEFT)
        after_heading = True
        i += 1
        continue
    
    if h2_match and not h3_match:
        num, title = h2_match.groups()
        add_paragraph(f'{num} {title}', font_size=10, bold=True,
                      space_before=6, space_after=3, first_indent=0,
                      alignment=WD_ALIGN_PARAGRAPH.LEFT)
        after_heading = True
        i += 1
        continue
    
    if h1_match and not h2_match:
        num, title = h1_match.groups()
        add_paragraph(f'{num}  {title}', font_size=10, bold=True,
                      space_before=12, space_after=6, first_indent=0,
                      alignment=WD_ALIGN_PARAGRAPH.LEFT)
        after_heading = True
        i += 1
        continue
    
    # Diagram lines - collect into a single courier block
    if is_diagram_line(stripped) or is_figure_label(stripped):
        # Diagram/figure lines in courier
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(0)
        pf.line_spacing = Pt(10)
        pf.alignment = WD_ALIGN_PARAGRAPH.LEFT
        pf.first_line_indent = Pt(0)
        run = p.add_run(stripped)
        run.font.name = 'Courier New'
        run.font.size = Pt(7)
        after_heading = False
        i += 1
        continue
    
    # Author contributions, conflicts, funding, data availability
    if stripped.startswith('Author Contributions:') or stripped.startswith('Conflicts of Interest:') or \
       stripped.startswith('Funding:') or stripped.startswith('Data Availability:'):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(3)
        pf.space_after = Pt(3)
        pf.line_spacing = Pt(12)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.first_line_indent = Pt(0)
        # Bold the label
        colon_idx = stripped.index(':')
        add_run_to_para(p, stripped[:colon_idx+1], font_size=9, bold=True)
        add_run_to_para(p, stripped[colon_idx+1:], font_size=9)
        i += 1
        continue
    
    # Bullet points (• lines)
    if stripped.startswith('•'):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(0)
        pf.space_after = Pt(3)
        pf.line_spacing = Pt(12)
        pf.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        pf.left_indent = Cm(0.5)
        pf.first_line_indent = Cm(-0.3)
        run = p.add_run(stripped)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(10)
        after_heading = False
        i += 1
        continue
    
    # Algorithm blocks - detect "Algorithm X:" pattern
    algo_match = re.match(r'^Algorithm\s+\d+:', stripped)
    if algo_match:
        # Create a bordered table with 1 cell for algorithm
        table = doc.add_table(rows=1, cols=1)
        table.style = 'Table Grid'
        cell = table.cell(0, 0)
        # Algorithm title
        ap = cell.paragraphs[0]
        ap.alignment = WD_ALIGN_PARAGRAPH.LEFT
        apf = ap.paragraph_format
        apf.space_before = Pt(3)
        apf.space_after = Pt(3)
        apf.line_spacing = Pt(12)
        run = ap.add_run(stripped)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(9)
        run.font.bold = True
        
        # Collect algorithm body lines
        i += 1
        while i < len(paras):
            atext = paras[i][0].strip()
            if not atext:
                i += 1
                continue
            # Check if next line is still part of algorithm
            if re.match(r'^(Input:|Output:|\d+[\.\):]|Step |  |return |for |if |while |end |Initialize|Set |Compute|Update|Perform|Select|Rank|Generate|Aggregate|Apply|Extract|Build|Construct)', atext, re.IGNORECASE):
                ap2 = cell.add_paragraph()
                ap2.alignment = WD_ALIGN_PARAGRAPH.LEFT
                apf2 = ap2.paragraph_format
                apf2.space_before = Pt(0)
                apf2.space_after = Pt(1)
                apf2.line_spacing = Pt(11)
                
                if atext.startswith('Input:') or atext.startswith('Output:'):
                    colon = atext.index(':')
                    r1 = ap2.add_run(atext[:colon+1])
                    r1.font.name = 'Times New Roman'
                    r1.font.size = Pt(9)
                    r1.font.bold = True
                    r2 = ap2.add_run(atext[colon+1:])
                    r2.font.name = 'Times New Roman'
                    r2.font.size = Pt(9)
                else:
                    r = ap2.add_run(atext)
                    r.font.name = 'Courier New'
                    r.font.size = Pt(8)
                i += 1
            else:
                break
        after_heading = False
        continue
    
    # Equations - lines with math symbols, centered
    # Detect lines that look like display equations
    if re.match(r'^[A-Za-z_].*=.*[∑∏∫√αβγδεζηθλμσφψω∈∀∃∪∩⊂⊆≤≥≠≈]', stripped) or \
       re.match(r'^\s*[A-Za-z]\s*[=<>≤≥]', stripped) and len(stripped) < 200 and '(' in stripped:
        pass  # Let it fall through to body text; complex equation detection is fragile
    
    # Table-like content: "Table X" headers
    if re.match(r'^Table\s+\d+', stripped):
        p = doc.add_paragraph()
        pf = p.paragraph_format
        pf.space_before = Pt(6)
        pf.space_after = Pt(3)
        pf.line_spacing = Pt(12)
        pf.alignment = WD_ALIGN_PARAGRAPH.CENTER
        pf.first_line_indent = Pt(0)
        # Bold "Table X."
        dot_idx = stripped.find('.')
        if dot_idx > 0:
            add_run_to_para(p, stripped[:dot_idx+1], font_size=9, bold=True)
            add_run_to_para(p, stripped[dot_idx+1:], font_size=9)
        else:
            add_run_to_para(p, stripped, font_size=9, bold=True)
        after_heading = False
        i += 1
        continue
    
    # Default: body paragraph
    indent = 0 if after_heading else 0.5
    add_paragraph(stripped, font_size=10, alignment=WD_ALIGN_PARAGRAPH.JUSTIFY,
                  space_before=0, space_after=6, first_indent=indent)
    after_heading = False
    i += 1

# Copy tables from source
for table in src.tables:
    # Add a simple representation
    new_table = doc.add_table(rows=len(table.rows), cols=len(table.columns))
    new_table.style = 'Table Grid'
    for ri, row in enumerate(table.rows):
        for ci, cell in enumerate(row.cells):
            new_cell = new_table.cell(ri, ci)
            new_cell.text = cell.text
            for p in new_cell.paragraphs:
                for run in p.runs:
                    run.font.name = 'Times New Roman'
                    run.font.size = Pt(9)
                p.paragraph_format.space_before = Pt(1)
                p.paragraph_format.space_after = Pt(1)
                p.paragraph_format.line_spacing = Pt(10)

# Add page numbers
for sect in doc.sections:
    footer = sect.footer
    footer.is_linked_to_previous = False
    fp = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    fp.paragraph_format.space_before = Pt(0)
    fp.paragraph_format.space_after = Pt(0)
    
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    run1 = fp.add_run()
    run1._r.append(fldChar1)
    run1.font.name = 'Times New Roman'
    run1.font.size = Pt(9)
    
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = ' PAGE '
    run2 = fp.add_run()
    run2._r.append(instrText)
    
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run3 = fp.add_run()
    run3._r.append(fldChar2)

doc.save('Multi_Graph_LLM_Second_Brain_v3_LNCS.docx')
print("Done! Saved LNCS version.")
