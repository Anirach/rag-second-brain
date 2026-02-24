#!/usr/bin/env python3
"""
Create a DOCX file from markdown chapters using only built-in Python modules.
DOCX is essentially a ZIP file containing XML files.
"""

import os
import zipfile
import re
from xml.etree import ElementTree as ET

# DOCX namespace definitions
NAMESPACES = {
    'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
    'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
    'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture',
}

def register_namespaces():
    for prefix, uri in NAMESPACES.items():
        ET.register_namespace(prefix, uri)

def create_content_types():
    """Create [Content_Types].xml"""
    root = ET.Element('Types', xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    ET.SubElement(root, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(root, 'Default', Extension='xml', ContentType='application/xml')
    ET.SubElement(root, 'Override', PartName='/word/document.xml', ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml')
    ET.SubElement(root, 'Override', PartName='/word/styles.xml', ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_rels():
    """Create _rels/.rels"""
    root = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    ET.SubElement(root, 'Relationship', Id='rId1', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument', Target='word/document.xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_document_rels():
    """Create word/_rels/document.xml.rels"""
    root = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    ET.SubElement(root, 'Relationship', Id='rId1', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles', Target='styles.xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_styles():
    """Create word/styles.xml with basic styles"""
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    
    root = ET.Element(f'{W}styles')
    
    # Default paragraph style
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Normal', f'{W}default': '1'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Normal'})
    pPr = ET.SubElement(style, f'{W}pPr')
    spacing = ET.SubElement(pPr, f'{W}spacing', {f'{W}after': '200', f'{W}line': '276', f'{W}lineRule': 'auto'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}rFonts', {f'{W}ascii': 'Georgia', f'{W}hAnsi': 'Georgia'})
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '24'})
    
    # Heading 1 style
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Heading1'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Heading 1'})
    ET.SubElement(style, f'{W}basedOn', {f'{W}val': 'Normal'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}spacing', {f'{W}before': '480', f'{W}after': '240'})
    ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}b')
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '48'})
    
    # Heading 2 style
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Heading2'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Heading 2'})
    ET.SubElement(style, f'{W}basedOn', {f'{W}val': 'Normal'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}spacing', {f'{W}before': '360', f'{W}after': '200'})
    ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}b')
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '36'})
    
    # Italic style for subtitles
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Subtitle'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Subtitle'})
    ET.SubElement(style, f'{W}basedOn', {f'{W}val': 'Normal'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}i')
    
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def md_to_docx_body(markdown_text):
    """Convert markdown text to DOCX body XML elements"""
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    
    body = ET.Element(f'{W}body')
    
    lines = markdown_text.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # H1 - # Title
        if line.startswith('# ') and not line.startswith('## '):
            text = line[2:].strip()
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}pStyle', {f'{W}val': 'Heading1'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = text
            i += 1
            continue
        
        # H2 - ## Subtitle
        if line.startswith('## '):
            text = line[3:].strip()
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}pStyle', {f'{W}val': 'Heading2'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = text
            i += 1
            continue
        
        # Horizontal rule / section break
        if line.strip() == '---':
            # Add empty paragraph with centered asterisks
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = '* * *'
            i += 1
            continue
        
        # Italic line (starts and ends with *)
        if line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            text = line.strip('*').strip()
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}pStyle', {f'{W}val': 'Subtitle'})
            r = ET.SubElement(p, f'{W}r')
            rPr = ET.SubElement(r, f'{W}rPr')
            ET.SubElement(rPr, f'{W}i')
            t = ET.SubElement(r, f'{W}t')
            t.text = text
            i += 1
            continue
        
        # Regular paragraph - handle inline formatting
        p = ET.SubElement(body, f'{W}p')
        
        # Parse inline formatting (bold, italic)
        text = line
        # Remove markdown bold/italic and just use plain text for simplicity
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        
        r = ET.SubElement(p, f'{W}r')
        t = ET.SubElement(r, f'{W}t')
        t.attrib['{http://www.w3.org/XML/1998/namespace}space'] = 'preserve'
        t.text = text
        
        i += 1
    
    # Section properties
    sectPr = ET.SubElement(body, f'{W}sectPr')
    ET.SubElement(sectPr, f'{W}pgSz', {f'{W}w': '12240', f'{W}h': '15840'})  # Letter size
    ET.SubElement(sectPr, f'{W}pgMar', {
        f'{W}top': '1440', f'{W}right': '1440', 
        f'{W}bottom': '1440', f'{W}left': '1440',
        f'{W}header': '720', f'{W}footer': '720'
    })
    
    return body

def create_document(markdown_text):
    """Create word/document.xml"""
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    
    root = ET.Element(f'{W}document')
    body = md_to_docx_body(markdown_text)
    root.append(body)
    
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_docx(markdown_text, output_path):
    """Create a DOCX file from markdown text"""
    register_namespaces()
    
    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', create_content_types())
        zf.writestr('_rels/.rels', create_rels())
        zf.writestr('word/_rels/document.xml.rels', create_document_rels())
        zf.writestr('word/styles.xml', create_styles())
        zf.writestr('word/document.xml', create_document(markdown_text))
    
    print(f"Created: {output_path}")

def read_chapters(chapter_dir):
    """Read all chapter files in order"""
    files = [
        '00-front-matter.md',
        '01-the-nature-of-time.md',
        '02-love-and-attachment.md',
        '03-the-illusion-of-control.md',
        '04-joy-and-suffering.md',
        '05-regret-and-forgiveness.md',
        '06-the-meaning-of-death.md',
        '07-the-last-conversation.md',
        '99-back-matter.md'
    ]
    
    full_text = []
    for f in files:
        path = os.path.join(chapter_dir, f)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                full_text.append(file.read())
                full_text.append('\n\n')
            print(f"Read: {f}")
        else:
            print(f"Missing: {f}")
    
    return '\n'.join(full_text)

if __name__ == '__main__':
    chapter_dir = 'three-old-men/02-chapters'
    output_file = 'three-old-men/Three-Old-Men-FINAL.docx'
    
    print("Reading chapters...")
    full_markdown = read_chapters(chapter_dir)
    
    print(f"\nTotal characters: {len(full_markdown)}")
    print(f"Estimated words: {len(full_markdown.split())}")
    
    print("\nCreating DOCX...")
    create_docx(full_markdown, output_file)
    
    print(f"\nDone! File created: {output_file}")
    print(f"File size: {os.path.getsize(output_file)} bytes")
