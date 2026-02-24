#!/usr/bin/env python3
"""
Enhanced Professional DOCX Generator with proper page numbers, TOC, and justification
"""

import sys
import os
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_PARAGRAPH_ALIGNMENT
from docx.enum.style import WD_STYLE_TYPE
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import argparse

class EnhancedProfessionalDocumentGenerator:
    def __init__(self):
        self.doc = Document()
        self._setup_styles()
        self._setup_page_numbers()
    
    def _setup_styles(self):
        """Setup professional document styles with proper justification"""
        styles = self.doc.styles
        
        # Executive Title Style
        if 'Executive Title' not in [s.name for s in styles]:
            exec_title = styles.add_style('Executive Title', WD_STYLE_TYPE.PARAGRAPH)
            exec_title.font.name = 'Arial'
            exec_title.font.size = Pt(18)
            exec_title.font.bold = True
            exec_title.font.color.rgb = RGBColor(0, 51, 102)  # Professional blue
            exec_title.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
            exec_title.paragraph_format.space_after = Pt(12)
        
        # Section Header Style
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
            body_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY  # Key fix!
            
        # TOC Style
        if 'TOC Style' not in [s.name for s in styles]:
            toc_style = styles.add_style('TOC Style', WD_STYLE_TYPE.PARAGRAPH)
            toc_style.font.name = 'Arial'
            toc_style.font.size = Pt(12)
            toc_style.paragraph_format.left_indent = Inches(0.25)
    
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
    
    def add_proper_table_of_contents(self, sections):
        """Add proper table of contents with dots and page references"""
        # TOC Header
        toc_header = self.doc.add_paragraph("Table of Contents", style='Section Header')
        toc_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.add_paragraph()  # Spacing
        
        # Add TOC entries with dots
        for i, section in enumerate(sections, 1):
            toc_para = self.doc.add_paragraph(style='TOC Style')
            
            # Section number and title
            run1 = toc_para.add_run(f"{i}. {section}")
            run1.font.size = Pt(12)
            
            # Dots leader - simplified version
            dots_run = toc_para.add_run(" " + "." * (50 - len(f"{i}. {section}") - 5))
            dots_run.font.size = Pt(12)
            
            # Page number placeholder
            page_run = toc_para.add_run(f" {i + 1}")
            page_run.font.size = Pt(12)
            page_run.bold = True
        
        self.doc.add_paragraph()  # Spacing
        self.doc.add_page_break()
    
    def add_executive_summary(self, content):
        """Add executive summary section with justified text"""
        self.doc.add_paragraph("Executive Summary", style='Section Header')
        summary_para = self.doc.add_paragraph(content, style='Professional Body')
        summary_para.paragraph_format.first_line_indent = Inches(0.25)
        # Justification is already set in the style
    
    def add_section(self, title, content, subsections=None):
        """Add a main section with justified text"""
        self.doc.add_paragraph(title, style='Section Header')
        
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
                # Center align first column, justify others
                if i == 0:
                    row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
                else:
                    row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        
        self.doc.add_paragraph()  # Spacing after table
    
    def add_key_findings(self, findings):
        """Add key findings with justified text"""
        self.doc.add_paragraph("Key Findings", style='Section Header')
        
        for i, finding in enumerate(findings, 1):
            finding_para = self.doc.add_paragraph(f"{i}. {finding}", style='Professional Body')
            finding_para.paragraph_format.left_indent = Inches(0.25)
            finding_para.paragraph_format.hanging_indent = Inches(0.25)
    
    def add_recommendations(self, recommendations):
        """Add recommendations with justified text"""
        self.doc.add_paragraph("Recommendations", style='Section Header')
        
        for i, rec in enumerate(recommendations, 1):
            rec_para = self.doc.add_paragraph(f"{i}. {rec}", style='Professional Body')
            rec_para.paragraph_format.left_indent = Inches(0.25)
            rec_para.paragraph_format.hanging_indent = Inches(0.25)
    
    def save(self, filename):
        """Save the document"""
        self.doc.save(filename)
        return filename

def create_enhanced_business_report(title, data):
    """Create enhanced business report with proper formatting"""
    generator = EnhancedProfessionalDocumentGenerator()
    
    # Cover page
    generator.add_cover_page(
        title=title,
        subtitle=data.get('subtitle', ''),
        author=data.get('author', ''),
        date=data.get('date', datetime.now().strftime('%B %d, %Y')),
        organization=data.get('organization', '')
    )
    
    # Enhanced table of contents
    sections = ["Executive Summary", "Key Findings", "Analysis", "Data Summary", "Recommendations", "Conclusion"]
    generator.add_proper_table_of_contents(sections)
    
    # Executive summary
    generator.add_executive_summary(data.get('executive_summary', ''))
    
    # Key findings
    if 'key_findings' in data:
        generator.add_key_findings(data['key_findings'])
    
    # Analysis section
    if 'analysis' in data:
        generator.add_section("Analysis", data['analysis'])
    
    # Data table if provided
    if 'table_data' in data:
        generator.add_professional_table(
            data['table_data']['headers'],
            data['table_data']['rows'],
            data['table_data'].get('title', 'Data Summary')
        )
    
    # Recommendations
    if 'recommendations' in data:
        generator.add_recommendations(data['recommendations'])
    
    # Conclusion
    if 'conclusion' in data:
        generator.add_section("Conclusion", data['conclusion'])
    
    return generator

def main():
    """Main function"""
    import json
    
    if len(sys.argv) < 4:
        print("Usage: python3 enhanced_docx_generator.py <title> <config.json> <output.docx>")
        sys.exit(1)
    
    title = sys.argv[1]
    config_file = sys.argv[2] 
    output_file = sys.argv[3]
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    generator = create_enhanced_business_report(title, data)
    filename = generator.save(output_file)
    print(f"✅ Enhanced professional document created: {filename}")
    
if __name__ == "__main__":
    main()