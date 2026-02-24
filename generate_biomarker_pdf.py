#!/usr/bin/env python3
"""Generate professional PDF report for Biological Age Biomarkers"""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, black, white
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
import json
import re

def clean_text(text):
    """Remove markdown-style formatting for PDF"""
    # Remove ** bold markers
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
    return text

def create_pdf():
    # Load data
    with open('biomarkers_age_report.json', 'r') as f:
        data = json.load(f)
    
    doc = SimpleDocTemplate(
        'biomarkers_biological_age_report.pdf',
        pagesize=letter,
        leftMargin=0.75*inch,
        rightMargin=0.75*inch,
        topMargin=0.75*inch,
        bottomMargin=0.75*inch
    )
    
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=22,
        textColor=HexColor('#003366'),
        spaceAfter=12,
        alignment=TA_CENTER
    )
    
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#505050'),
        spaceAfter=6,
        alignment=TA_CENTER,
        fontName='Helvetica-Oblique'
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=14,
        textColor=HexColor('#003366'),
        spaceBefore=18,
        spaceAfter=10,
        fontName='Helvetica-Bold'
    )
    
    subheading_style = ParagraphStyle(
        'CustomSubheading',
        parent=styles['Heading2'],
        fontSize=11,
        textColor=HexColor('#003366'),
        spaceBefore=12,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=8,
        alignment=TA_JUSTIFY
    )
    
    ref_style = ParagraphStyle(
        'CustomRef',
        parent=styles['Normal'],
        fontSize=9,
        leading=12,
        spaceAfter=4,
        leftIndent=20,
        firstLineIndent=-20
    )
    
    # Build content
    content = []
    
    # Title
    content.append(Paragraph(data['title'], title_style))
    content.append(Paragraph(data['subtitle'], subtitle_style))
    content.append(Paragraph(data['date'], subtitle_style))
    content.append(Spacer(1, 0.3*inch))
    
    # Add separator line
    line_data = [['_' * 100]]
    line_table = Table(line_data, colWidths=[7*inch])
    line_table.setStyle(TableStyle([
        ('TEXTCOLOR', (0, 0), (-1, -1), HexColor('#003366')),
        ('FONTSIZE', (0, 0), (-1, -1), 6),
    ]))
    content.append(line_table)
    content.append(Spacer(1, 0.2*inch))
    
    # Sections
    for section in data['sections']:
        content.append(Paragraph(section['heading'], heading_style))
        
        # Process content paragraphs
        paragraphs = section['content'].split('\n\n')
        
        for para_text in paragraphs:
            if para_text.strip():
                # Check for bold subsections
                if para_text.startswith('**') and '**' in para_text[2:]:
                    end_bold = para_text.index('**', 2)
                    bold_title = para_text[2:end_bold]
                    rest = para_text[end_bold+2:].strip()
                    
                    content.append(Paragraph(bold_title, subheading_style))
                    if rest:
                        content.append(Paragraph(clean_text(rest), body_style))
                else:
                    content.append(Paragraph(clean_text(para_text), body_style))
    
    # Comparison Table
    content.append(Paragraph('Biomarker Comparison Table', heading_style))
    
    table_data = [
        ['Biomarker', 'Primary Use', 'Predictive\nStrength', 'Intervention\nSensitivity', 'Clinical\nAccess'],
        ['GrimAge', 'Mortality/disease\nprediction', 'Excellent', 'Moderate', 'Specialized\nlab'],
        ['DunedinPACE', 'Intervention\nmonitoring', 'Very Good\n(56% ↑ mort/unit)', 'Excellent', 'Specialized\nlab'],
        ['GDF15', 'Stress/aging\nassessment', 'Good\n(13-43% mediation)', 'Good', 'Standard\nassay'],
        ['Proteomic\nClocks', 'Organ-specific\naging', 'Excellent', 'Under study', 'Emerging'],
        ['NMR/GlycA', 'Inflammatory\naging', 'Good', 'Good', 'NMR\nplatform'],
        ['Telomere\nLength', 'Research only', 'Weak-Moderate', 'Variable', 'Standard\nassay']
    ]
    
    table = Table(table_data, colWidths=[1.2*inch, 1.4*inch, 1.3*inch, 1.2*inch, 0.9*inch])
    table.setStyle(TableStyle([
        # Header row
        ('BACKGROUND', (0, 0), (-1, 0), HexColor('#003366')),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        
        # Data rows
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 1), (-1, -1), 'CENTER'),
        
        # Alternating row colors
        ('BACKGROUND', (0, 1), (-1, 1), HexColor('#F0F4F8')),
        ('BACKGROUND', (0, 3), (-1, 3), HexColor('#F0F4F8')),
        ('BACKGROUND', (0, 5), (-1, 5), HexColor('#F0F4F8')),
        
        # Grid
        ('GRID', (0, 0), (-1, -1), 0.5, HexColor('#CCCCCC')),
        ('BOX', (0, 0), (-1, -1), 1, HexColor('#003366')),
        
        # Padding
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    
    content.append(Spacer(1, 0.1*inch))
    content.append(table)
    content.append(Spacer(1, 0.3*inch))
    
    # References
    content.append(Paragraph('References', heading_style))
    
    for ref in data['references']:
        ref_text = f"{ref['number']}. {ref['citation']} <font color='#0066CC'><u><a href='{ref['url']}'>[View Source]</a></u></font>"
        content.append(Paragraph(ref_text, ref_style))
    
    # Build PDF
    doc.build(content)
    print("PDF created: biomarkers_biological_age_report.pdf")

if __name__ == '__main__':
    create_pdf()
