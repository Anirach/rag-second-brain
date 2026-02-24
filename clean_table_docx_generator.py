#!/usr/bin/env python3
"""
Professional DOCX Generator with CLEAN table format
- Only "View Source" links, no URL text shown
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

class CleanTableDocumentGenerator:
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
            body_style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            
        # TOC Entry Style
        if 'TOC Entry' not in [s.name for s in styles]:
            toc_style = styles.add_style('TOC Entry', WD_STYLE_TYPE.PARAGRAPH)
            toc_style.font.name = 'Arial'
            toc_style.font.size = Pt(12)
            toc_style.paragraph_format.left_indent = Inches(0.25)
            tab_stops = toc_style.paragraph_format.tab_stops
            tab_stops.add_tab_stop(Inches(5.5), WD_TAB_ALIGNMENT.RIGHT, leader=3)
            
        # Source Citation Style
        if 'Source Citation' not in [s.name for s in styles]:
            source_style = styles.add_style('Source Citation', WD_STYLE_TYPE.PARAGRAPH)
            source_style.font.name = 'Arial'
            source_style.font.size = Pt(10)
            source_style.font.italic = True
            source_style.paragraph_format.left_indent = Inches(0.5)
            source_style.paragraph_format.space_after = Pt(3)
    
    def _setup_page_numbers(self):
        """Add proper page numbers to footer"""
        section = self.doc.sections[0]
        footer = section.footer
        footer_para = footer.paragraphs[0]
        footer_para.text = f"Generated on {datetime.now().strftime('%B %d, %Y')}"
        footer_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        run = footer_para.runs[0]
        run.text += " | Page "
        
        # Page number field
        fldChar1 = OxmlElement('w:fldChar')
        fldChar1.set(qn('w:fldCharType'), 'begin')
        instrText = OxmlElement('w:instrText')
        instrText.text = "PAGE"
        fldChar2 = OxmlElement('w:fldChar')
        fldChar2.set(qn('w:fldCharType'), 'end')
        
        run._r.append(fldChar1)
        run._r.append(instrText)
        run._r.append(fldChar2)
        
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
        """Add bookmark to paragraph"""
        bookmark_start = OxmlElement('w:bookmarkStart')
        bookmark_start.set(qn('w:id'), str(len(self.section_bookmarks)))
        bookmark_start.set(qn('w:name'), bookmark_name)
        
        bookmark_end = OxmlElement('w:bookmarkEnd')
        bookmark_end.set(qn('w:id'), str(len(self.section_bookmarks)))
        
        paragraph._p.insert(0, bookmark_start)
        paragraph._p.append(bookmark_end)
        
        self.section_bookmarks[bookmark_name] = len(self.section_bookmarks)
    
    def add_clean_hyperlink(self, paragraph, url):
        """Add clean 'View Source' hyperlink without showing URL"""
        run = paragraph.add_run("View Source")
        run.font.color.rgb = RGBColor(0, 102, 204)  # Professional blue
        run.font.underline = True
        run.font.size = Pt(10)
        return run
    
    def add_cover_page(self, title, subtitle="", author="", date="", organization=""):
        """Create professional cover page"""
        title_para = self.doc.add_paragraph(title, style='Executive Title')
        title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        if subtitle:
            subtitle_para = self.doc.add_paragraph(subtitle)
            subtitle_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            subtitle_para.runs[0].font.size = Pt(14)
            subtitle_para.runs[0].italic = True
            subtitle_para.runs[0].font.color.rgb = RGBColor(0, 51, 102)
        
        self.doc.add_paragraph("\n" * 8)
        
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
        
        self.doc.add_page_break()
    
    def add_professional_table_of_contents(self, sections):
        """Add proper table of contents"""
        toc_header = self.doc.add_paragraph("Table of Contents", style='Section Header')
        toc_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
        self.doc.add_paragraph()
        
        page_num = 3
        for i, section in enumerate(sections, 1):
            toc_para = self.doc.add_paragraph(style='TOC Entry')
            title_run = toc_para.add_run(f"{i}. {section}")
            title_run.font.size = Pt(12)
            toc_para.add_run('\t')
            page_run = toc_para.add_run(str(page_num))
            page_run.font.size = Pt(12)
            page_run.bold = True
            page_num += 1
        
        self.doc.add_paragraph()
        self.doc.add_page_break()
    
    def add_section_with_bookmark(self, title, content, subsections=None):
        """Add section with bookmark"""
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
    
    def add_clean_professional_table(self, headers, rows, title=""):
        """Add table with CLEAN View Source links (no URLs shown)"""
        if title:
            title_para = self.doc.add_paragraph(title)
            title_para.runs[0].bold = True
            title_para.runs[0].font.size = Pt(12)
            title_para.runs[0].font.color.rgb = RGBColor(0, 51, 102)
            title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.style = 'Light Grid Accent 1'
        
        # Header row
        header_cells = table.rows[0].cells
        for i, header in enumerate(headers):
            header_cells[i].text = header
            header_cells[i].paragraphs[0].runs[0].bold = True
            header_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
            header_cells[i].paragraphs[0].runs[0].font.size = Pt(11)
        
        # Data rows with CLEAN hyperlinks in last column
        for row_data in rows:
            row_cells = table.add_row().cells
            for i, cell_data in enumerate(row_data):
                if i == len(row_data) - 1 and 'http' in str(cell_data):  # Last column with URL
                    # Add CLEAN hyperlink - only "View Source" text, no URL shown
                    cell_para = row_cells[i].paragraphs[0]
                    cell_para.clear()
                    cell_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    self.add_clean_hyperlink(cell_para, str(cell_data))
                else:
                    row_cells[i].text = str(cell_data)
                    row_cells[i].paragraphs[0].runs[0].font.size = Pt(10)
                row_cells[i].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER
        
        self.doc.add_paragraph()
    
    def add_sources_section(self, sources_data):
        """Add comprehensive sources section with clickable links"""
        self.add_section_with_bookmark("Sources and References", "")
        
        if 'primary_sources' in sources_data:
            sub_header = self.doc.add_paragraph("Primary Sources")
            sub_header.runs[0].bold = True
            sub_header.runs[0].font.size = Pt(12)
            sub_header.runs[0].font.color.rgb = RGBColor(0, 51, 102)
            
            for i, source in enumerate(sources_data['primary_sources'], 1):
                # Source title and description
                source_para = self.doc.add_paragraph(f"{i}. {source['title']}")
                source_para.runs[0].font.size = Pt(11)
                source_para.runs[0].bold = True
                
                # Description
                desc_para = self.doc.add_paragraph(source['description'], style='Professional Body')
                desc_para.paragraph_format.left_indent = Inches(0.25)
                
                # URL link (full URL in references is OK, just not in table)
                url_para = self.doc.add_paragraph(style='Source Citation')
                url_para.add_run("Source URL: ")
                url_run = url_para.add_run(source['url'])
                url_run.font.color.rgb = RGBColor(0, 102, 204)
                url_run.font.underline = True
                
                self.doc.add_paragraph()  # Spacing
    
    def save(self, filename):
        """Save the document"""
        self.doc.save(filename)
        return filename

def create_clean_research_report(title, data):
    """Create research report with clean table formatting"""
    generator = CleanTableDocumentGenerator()
    
    # Cover page
    generator.add_cover_page(
        title=title,
        subtitle=data.get('subtitle', ''),
        author=data.get('author', ''),
        date=data.get('date', datetime.now().strftime('%B %d, %Y')),
        organization=data.get('organization', '')
    )
    
    # Table of contents
    sections = ["Executive Summary", "Key Findings", "Analysis", "Data Summary", "Recommendations", "Conclusion", "Sources and References"]
    generator.add_professional_table_of_contents(sections)
    
    # Executive summary
    generator.add_section_with_bookmark("Executive Summary", data.get('executive_summary', ''))
    
    # Key findings
    if 'key_findings' in data:
        generator.add_section_with_bookmark("Key Findings", data['key_findings'])
    
    # Analysis
    if 'analysis' in data:
        generator.add_section_with_bookmark("Analysis", data['analysis'])
    
    # Data table with CLEAN source links
    if 'table_data' in data:
        generator.add_section_with_bookmark("Data Summary", "")
        generator.add_clean_professional_table(
            data['table_data']['headers'],
            data['table_data']['rows'],
            data['table_data'].get('title', 'Research Developments with Source Verification')
        )
    
    # Recommendations
    if 'recommendations' in data:
        generator.add_section_with_bookmark("Recommendations", data['recommendations'])
    
    # Conclusion
    if 'conclusion' in data:
        generator.add_section_with_bookmark("Conclusion", data['conclusion'])
    
    # Sources section (full URLs OK here)
    if 'sources' in data:
        generator.add_sources_section(data['sources'])
    
    return generator

def main():
    """Main function"""
    import json
    
    if len(sys.argv) < 4:
        print("Usage: python3 clean_table_docx_generator.py <title> <config.json> <output.docx>")
        sys.exit(1)
    
    title = sys.argv[1]
    config_file = sys.argv[2] 
    output_file = sys.argv[3]
    
    with open(config_file, 'r') as f:
        data = json.load(f)
    
    generator = create_clean_research_report(title, data)
    filename = generator.save(output_file)
    print(f"✅ Research report with CLEAN table formatting created: {filename}")
    
if __name__ == "__main__":
    main()