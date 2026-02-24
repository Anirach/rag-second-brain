#!/usr/bin/env python3
"""Generate professional DOCX report for Biological Age Biomarkers"""

from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import json

def add_page_number(run):
    """Add page number field"""
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    
    instrText = OxmlElement('w:instrText')
    instrText.text = "PAGE"
    
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    
    run._r.append(fldChar1)
    run._r.append(instrText)
    run._r.append(fldChar2)

def set_cell_shading(cell, color):
    """Set cell background color"""
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), color)
    tcPr.append(shading)

def create_report():
    doc = Document()
    
    # Set margins
    for section in doc.sections:
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(2.5)
        section.top_margin = Cm(2.5)
        section.bottom_margin = Cm(2.5)
    
    # Load data
    with open('biomarkers_age_report.json', 'r') as f:
        data = json.load(f)
    
    # Title
    title = doc.add_heading(data['title'], 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in title.runs:
        run.font.size = Pt(24)
        run.font.color.rgb = RGBColor(0, 51, 102)
    
    # Subtitle
    subtitle = doc.add_paragraph(data['subtitle'])
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in subtitle.runs:
        run.font.size = Pt(12)
        run.font.italic = True
        run.font.color.rgb = RGBColor(80, 80, 80)
    
    # Date
    date_para = doc.add_paragraph(data['date'])
    date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for run in date_para.runs:
        run.font.size = Pt(11)
        run.font.color.rgb = RGBColor(100, 100, 100)
    
    doc.add_paragraph()  # Spacing
    
    # Add horizontal line
    p = doc.add_paragraph()
    p_format = p.paragraph_format
    p_format.space_after = Pt(12)
    
    # Sections
    for section in data['sections']:
        # Section heading
        heading = doc.add_heading(section['heading'], 1)
        for run in heading.runs:
            run.font.size = Pt(14)
            run.font.color.rgb = RGBColor(0, 51, 102)
        
        # Section content
        content = section['content']
        paragraphs = content.split('\n\n')
        
        for para_text in paragraphs:
            if para_text.strip():
                # Check for bold subsections
                if para_text.startswith('**') and '**' in para_text[2:]:
                    # Extract bold title
                    end_bold = para_text.index('**', 2)
                    bold_title = para_text[2:end_bold]
                    rest = para_text[end_bold+2:].strip()
                    
                    p = doc.add_paragraph()
                    p.paragraph_format.space_before = Pt(12)
                    p.paragraph_format.space_after = Pt(6)
                    
                    run = p.add_run(bold_title)
                    run.font.bold = True
                    run.font.size = Pt(11)
                    run.font.color.rgb = RGBColor(0, 51, 102)
                    
                    if rest:
                        run2 = p.add_run('\n' + rest)
                        run2.font.size = Pt(11)
                else:
                    p = doc.add_paragraph(para_text)
                    p.paragraph_format.space_after = Pt(8)
                    p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
                    p.paragraph_format.line_spacing = 1.15
                    for run in p.runs:
                        run.font.size = Pt(11)
    
    # Comparison Table
    doc.add_heading('Biomarker Comparison Table', 1)
    
    table = doc.add_table(rows=7, cols=5)
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Table headers
    headers = ['Biomarker', 'Primary Use', 'Predictive Strength', 'Intervention Sensitivity', 'Clinical Access']
    header_row = table.rows[0]
    for i, header in enumerate(headers):
        cell = header_row.cells[i]
        cell.text = header
        set_cell_shading(cell, '003366')
        for para in cell.paragraphs:
            for run in para.runs:
                run.font.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(255, 255, 255)
    
    # Table data
    data_rows = [
        ['GrimAge', 'Mortality/disease prediction', 'Excellent', 'Moderate', 'Specialized lab'],
        ['DunedinPACE', 'Intervention monitoring', 'Very Good (56% ↑ mortality/unit)', 'Excellent', 'Specialized lab'],
        ['GDF15', 'Stress/aging assessment', 'Good (13-43% mediation)', 'Good', 'Standard assay'],
        ['Proteomic Clocks', 'Organ-specific aging', 'Excellent', 'Under study', 'Emerging'],
        ['NMR/GlycA', 'Inflammatory aging', 'Good', 'Good', 'NMR platform'],
        ['Telomere Length', 'Research only', 'Weak-Moderate', 'Variable', 'Standard assay']
    ]
    
    for row_idx, row_data in enumerate(data_rows):
        row = table.rows[row_idx + 1]
        for col_idx, cell_text in enumerate(row_data):
            cell = row.cells[col_idx]
            cell.text = cell_text
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.size = Pt(9)
            if row_idx % 2 == 0:
                set_cell_shading(cell, 'F0F4F8')
    
    # Set column widths
    for row in table.rows:
        row.cells[0].width = Inches(1.2)
        row.cells[1].width = Inches(1.5)
        row.cells[2].width = Inches(1.5)
        row.cells[3].width = Inches(1.3)
        row.cells[4].width = Inches(1.0)
    
    doc.add_paragraph()  # Spacing
    
    # References
    doc.add_heading('References', 1)
    
    for ref in data['references']:
        p = doc.add_paragraph()
        p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.left_indent = Inches(0.5)
        p.paragraph_format.first_line_indent = Inches(-0.5)
        
        run = p.add_run(f"{ref['number']}. {ref['citation']} ")
        run.font.size = Pt(10)
        
        # Add hyperlink text
        link_run = p.add_run(f"[View Source]")
        link_run.font.size = Pt(10)
        link_run.font.color.rgb = RGBColor(0, 102, 204)
        link_run.font.underline = True
    
    # Add footer with page numbers
    section = doc.sections[0]
    footer = section.footer
    footer.is_linked_to_previous = False
    footer_para = footer.paragraphs[0]
    footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    run = footer_para.add_run("Page ")
    run.font.size = Pt(9)
    run.font.color.rgb = RGBColor(100, 100, 100)
    
    page_run = footer_para.add_run()
    add_page_number(page_run)
    page_run.font.size = Pt(9)
    page_run.font.color.rgb = RGBColor(100, 100, 100)
    
    # Save
    doc.save('biomarkers_biological_age_report.docx')
    print("DOCX created: biomarkers_biological_age_report.docx")

if __name__ == '__main__':
    create_report()
