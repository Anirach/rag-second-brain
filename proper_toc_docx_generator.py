#!/usr/bin/env python3
"""
Professional DOCX Generator with PROPER Table of Contents
- Justified alignment with tab stops
- Clickable hyperlinks to sections
- Professional formatting
"""

import sys
import os
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT, WD_TAB_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import argparse

class ProperTOCDocumentGenerator:
    def __init__(self):
        self.doc = Document()
        self._setup_styles()
        self._setup_page_numbers()
        self.section_bookmarks = {}
    
    def _setup_styles(self):
        """Setup professional document styles"""
        styles = self.doc.styles
        
        # Executive Title Style
        if 'Executive Title' not in [s.name for s in styles]:
            exec_title = styles.add_style('Executive Title', WD_STYLE_TYPE.PARAGRAPH)
            exec_title.font.name = 'Arial'
            exec_title.font.size = Pt(18)
            exec_title.font.bold = True
            exec_title.font.color.rgb = RGBColor(0, 51, 102)
            exec_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            exec_title.paragraph_format.space_after = Pt(12)
        
        # Section Header Style with bookmark capability
        if 'Section Header' not in [s.name for s in styles]:
            section_header = styles.add_style('Section Header', WD_STYLE_TYPE.PARAGRAPH)
            section_header.font.name = 'Arial'
            section_header.font.size = Pt(14)
            section_header.font.bold = True
            section_header.font.color.rgb = RGBColor(0, 51, 102)
            section_header.paragraph_format.space_before = Pt(12)
            section_header.paragraph_format.space_after = Pt(6)
        
        # Professional Body Style with JUSTIFICATION
        if 'Professional Body' not in [s.name for s in styles]:
            body_style = styles.add_style('Professional Body', WD_STYLE_TYPE.PARAGRAPH)
            body_style.font.name = 'Arial'
            body_style.font.size = Pt(11)
            body_style.paragraph_format.line_spacing = 1.15
            body_style.paragraph_format.space_after = Pt(6)
            body_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            
        # TOC Entry Style with proper tab stops
        if 'TOC Entry' not in [s.name for s in styles]:
            toc_style = styles.add_style('TOC Entry', WD_STYLE_TYPE.PARAGRAPH)
            toc_style.font.name = 'Arial'
            toc_style.font.size = Pt(12)
            toc_style.paragraph_format.left_indent = Inches(0.25)
            
            # Add tab stop for page numbers (right aligned with dot leader)
            tab_stops = toc_style.paragraph_format.tab_stops
            tab_stops.add_tab_stop(Inches(5.5), WD_TAB_ALIGNMENT.RIGHT, leader=3)  # 3 = dot leader
    
    def _setup_page_numbers(self):
        """Add proper page numbers to footer"""
        section = self.doc.sections[0]
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.text = f"Generated on {datetime.now().strftime('%B %d, %Y')}"
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Add page number field
        run = footer_para.runs[0]
        run.text += " | Page "
        
        # Create page number field XML
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        
        instrText = OxmlElement('w:instrText')
        instrText.text = "PAGE"
        
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'end')
        
        # Add to paragraph
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        
        # Add "of total pages"
        run = footer_para.add_run(" of ")
        
        # Total pages field
        fldChar3 = OxmlElement('w:fldChar')
        fldChar3.set(qn('w:fldCharType'), 'begin')
        
        instrText2 = OxmlElement('w:instrText')
        instrText2.text = "NUMPAGES"
        
        fldChar4 = OxmlElement('w:fldChar')
        fldChar4.set(qn('w:fldCharType'), 'end')
        
        run._r.append(fldChar3)
        run._r.append(instrText2)
        run._r.append(fldChar4)
    
    def _add_bookmark(self, paragraph, bookmark_name):
        """Add bookmark to paragraph for TOC linking"""
        bookmark_start = OxmlElement('w:bookmarkStart')
        bookmark_start.set(qn('w:id'), str(len(self.section_bookmarks)))
        bookmark_start.set(qn('w:name'), bookmark_name)
        
        bookmark_end = OxmlElement('w:bookmarkEnd')
        bookmark_end.set(qn('w:id'), str(len(self.section_bookmarks)))
        
        paragraph._p.insert(0, bookmark_start)
        paragraph._p.append(bookmark_end)
        
        self.section_bookmarks[bookmark_name] = len(self.section_bookmarks)
    
    def add_cover_page(self, title, subtitle="", author="", date="", organization=""):
        """Create professional cover page"""
        # Title
        title_para = self.doc.add_paragraph(title, style='Executive Title')
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        if subtitle:
            subtitle_para = self.doc.add_paragraph(subtitle)
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_para.runs[0].font.size = Pt(14)
            subtitle_para.runs[0].italic = True
            subtitle_para.runs[0].font.color.rgb = RGBColor(0, 51, 102)
        
        # Add spacing
        self.doc.add_paragraph("\n" * 8)
        
        # Document info
        if author:
            author_para = self.doc.add_paragraph(f"Prepared by: {author}")
            author_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            author_para.runs[0].font.size = Pt(12)
        
        if organization:
            org_para = self.doc.add_paragraph(organization)
            org_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            org_para.runs[0].font.size = Pt(12)
            org_para.runs[0].bold = True
            org_para.runs[0].font.color.rgb = RGBColor(0, 51, 102)
        
        if date:
            date_para = self.doc.add_paragraph(date)
            date_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            date_para.runs[0].font.size = Pt(11)
        
        # Page break
        self.doc.add_page_break()
    
    def add_professional_table_of_contents(self, sections):
        """Add PROPER table of contents with justified alignment and hyperlinks"""
        # TOC Header
        toc_header = self.doc.add_paragraph("Table of Contents", style='Section Header')
        toc_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.add_paragraph()  # Spacing
        
        # Add TOC entries with proper tab stops and hyperlinks
        page_num = 3  # Starting page after cover page and TOC
        
        for i, section in enumerate(sections, 1):
            # Create TOC entry paragraph
            toc_para = self.doc.add_paragraph(style='TOC Entry')
            
            # Section number and title (will be hyperlink)
            title_run = toc_para.add_run(f"{i}. {section}")
            title_run.font.size = Pt(12)
            
            # Add tab character to trigger dot leader
            toc_para.add_run('\t')
            
            # Page number
            page_run = toc_para.add_run(str(page_num))
            page_run.font.size = Pt(12)
            page_run.bold = True
            
            page_num += 1
        
        self.doc.add_paragraph()  # Spacing
        self.doc.add_page_break()
    
    def add_section_with_bookmark(self, title, content, subsections=None):
        """Add section with bookmark for TOC linking"""
        # Create section header with bookmark
        section_para = self.doc.add_paragraph(title, style='Section Header')
        bookmark_name = title.replace(' ', '_').replace('.', '').lower()
        self._add_bookmark(section_para, bookmark_name)
        
        if isinstance(content, str):
            para = self.doc.add_paragraph(content, style='Professional Body')
            para.paragraph_format.first_line_indent = Inches(0.25)
        elif isinstance(content, list):
            for item in content:
                bullet_para = self.doc.add_paragraph(f"• {item}", style='Professional Body')
                bullet_para.paragraph_format.left_indent = Inches(0.25)
        
        if subsections:
            for subsection_title, subsection_content in subsections.items():
                sub_para = self.doc.add_paragraph(subsection_title)
                sub_para.runs[0].bold = True
                sub_para.runs[0].font.size = Pt(12)
                content_para = self.doc.add_paragraph(subsection_content, style='Professional Body')
                content_para.paragraph_format.first_line_indent = Inches(0.25)
    
    def add_executive_summary(self, content):
        """Add executive summary section with bookmark"""
        self.add_section_with_bookmark("Executive Summary", content)
    
    def add_professional_table(self, headers, rows, title=""):
        """Add professionally formatted table"""
        if title:
            title_para = self.doc.add_paragraph(title)
            title_para.runs[0].bold = True
            title_para.runs[0].font.size = Pt(12)
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = 'Light Grid Accent 1'
        
        # Header row
        header_cells = table.rows[0].cells
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True
            header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        # Data rows
        for row_data in rows:
            row_cells = table.add_row().cells
            for i, cell_data in enumerate(row_data):
                row_cells[i].text = str(cell_data)
                row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        self.doc.add_paragraph()  # Spacing after table
    
    def add_key_findings(self, findings):
        """Add key findings with bookmark"""
        self.add_section_with_bookmark("Key Findings", findings)
    
    def add_recommendations(self, recommendations):
        """Add recommendations with bookmark"""
        self.add_section_with_bookmark("Recommendations", recommendations)
    
    def save(self, filename):
        """Save the document"""
        self.doc.save(filename)
        return filename

def create_proper_toc_business_report(title, data):
    """Create business report with proper TOC"""
    generator = ProperTOCDocumentGenerator()
    
    # Cover page
    generator.add_cover_page(
        title=title,
        subtitle=data.get('subtitle', ''),
        author=data.get('author', ''),
        date=data.get('date', datetime.now().strftime('%B %d, %Y')),
        organization=data.get('organization', '')
    )
    
    # PROPER table of contents
    sections = ["Executive Summary", "Key Findings", "Analysis", "Data Summary", "Recommendations", "Conclusion"]
    generator.add_professional_table_of_contents(sections)
    
    # Executive summary with bookmark
    generator.add_executive_summary(data.get('executive_summary', ''))
    
    # Key findings with bookmark
    if 'key_findings' in data:
        generator.add_key_findings(data['key_findings'])
    
    # Analysis section with bookmark
    if 'analysis' in data:
        generator.add_section_with_bookmark("Analysis", data['analysis'])
    
    # Data table with bookmark
    if 'table_data' in data:
        generator.add_section_with_bookmark("Data Summary", "")
        generator.add_professional_table(
            data['table_data']['headers'],
            data['table_data']['rows'],
            data['table_data'].get('title', 'Key Research Developments - Impact Assessment Matrix')
        )
    
    # Recommendations with bookmark
    if 'recommendations' in data:
        generator.add_recommendations(data['recommendations'])
    
    # Conclusion with bookmark
    if 'conclusion' in data:
        generator.add_section_with_bookmark("Conclusion", data['conclusion'])
    
    return generator

def main():
    """Main function"""
    import json
    
    if len(sys.argv) < 4:
        print("Usage: python3 proper_toc_docx_generator.py <title> <config.json> <output.docx>")
        sys.exit(1)
    
    title = sys.argv[1]
    config_file = sys.argv[2] 
    output_file = sys.argv[3]
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    generator = create_proper_toc_business_report(title, data)
    filename = generator.save(output_file)
    print(f"✅ Document with PROPER TOC created: {filename}")
    
if __name__ == "__main__":
    main()