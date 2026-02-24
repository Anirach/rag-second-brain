#!/usr/bin/env python3
"""
Create a DOCX file from markdown chapters and output as base64.
"""

import os
import zipfile
import io
import base64
import re
from xml.etree import ElementTree as ET

def register_namespaces():
    namespaces = {
        'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
        'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    }
    for prefix, uri in namespaces.items():
        ET.register_namespace(prefix, uri)

def create_content_types():
    root = ET.Element('Types', xmlns='http://schemas.openxmlformats.org/package/2006/content-types')
    ET.SubElement(root, 'Default', Extension='rels', ContentType='application/vnd.openxmlformats-package.relationships+xml')
    ET.SubElement(root, 'Default', Extension='xml', ContentType='application/xml')
    ET.SubElement(root, 'Override', PartName='/word/document.xml', ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml')
    ET.SubElement(root, 'Override', PartName='/word/styles.xml', ContentType='application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_rels():
    root = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    ET.SubElement(root, 'Relationship', Id='rId1', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument', Target='word/document.xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_document_rels():
    root = ET.Element('Relationships', xmlns='http://schemas.openxmlformats.org/package/2006/relationships')
    ET.SubElement(root, 'Relationship', Id='rId1', Type='http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles', Target='styles.xml')
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_styles():
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    root = ET.Element(f'{W}styles')
    
    # Normal style
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Normal', f'{W}default': '1'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Normal'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}spacing', {f'{W}after': '200', f'{W}line': '276', f'{W}lineRule': 'auto'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}rFonts', {f'{W}ascii': 'Georgia', f'{W}hAnsi': 'Georgia'})
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '24'})
    
    # Heading1
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Heading1'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Heading 1'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}spacing', {f'{W}before': '480', f'{W}after': '240'})
    ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}b')
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '48'})
    
    # Heading2
    style = ET.SubElement(root, f'{W}style', {f'{W}type': 'paragraph', f'{W}styleId': 'Heading2'})
    ET.SubElement(style, f'{W}name', {f'{W}val': 'Heading 2'})
    pPr = ET.SubElement(style, f'{W}pPr')
    ET.SubElement(pPr, f'{W}spacing', {f'{W}before': '360', f'{W}after': '200'})
    ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
    rPr = ET.SubElement(style, f'{W}rPr')
    ET.SubElement(rPr, f'{W}b')
    ET.SubElement(rPr, f'{W}sz', {f'{W}val': '36'})
    
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def md_to_docx_body(markdown_text):
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    body = ET.Element(f'{W}body')
    
    for line in markdown_text.split('\n'):
        line = line.rstrip()
        if not line:
            continue
        
        # H1
        if line.startswith('# ') and not line.startswith('## '):
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}pStyle', {f'{W}val': 'Heading1'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = line[2:].strip()
            continue
        
        # H2
        if line.startswith('## '):
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}pStyle', {f'{W}val': 'Heading2'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = line[3:].strip()
            continue
        
        # HR
        if line.strip() == '---':
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
            r = ET.SubElement(p, f'{W}r')
            t = ET.SubElement(r, f'{W}t')
            t.text = '* * *'
            continue
        
        # Italic
        if line.startswith('*') and line.endswith('*') and not line.startswith('**'):
            p = ET.SubElement(body, f'{W}p')
            pPr = ET.SubElement(p, f'{W}pPr')
            ET.SubElement(pPr, f'{W}jc', {f'{W}val': 'center'})
            r = ET.SubElement(p, f'{W}r')
            rPr = ET.SubElement(r, f'{W}rPr')
            ET.SubElement(rPr, f'{W}i')
            t = ET.SubElement(r, f'{W}t')
            t.text = line.strip('*').strip()
            continue
        
        # Normal paragraph
        p = ET.SubElement(body, f'{W}p')
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', line)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        r = ET.SubElement(p, f'{W}r')
        t = ET.SubElement(r, f'{W}t')
        t.attrib['{http://www.w3.org/XML/1998/namespace}space'] = 'preserve'
        t.text = text
    
    # Section properties
    sectPr = ET.SubElement(body, f'{W}sectPr')
    ET.SubElement(sectPr, f'{W}pgSz', {f'{W}w': '12240', f'{W}h': '15840'})
    ET.SubElement(sectPr, f'{W}pgMar', {
        f'{W}top': '1440', f'{W}right': '1440',
        f'{W}bottom': '1440', f'{W}left': '1440'
    })
    
    return body

def create_document(markdown_text):
    W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
    root = ET.Element(f'{W}document')
    body = md_to_docx_body(markdown_text)
    root.append(body)
    return ET.tostring(root, encoding='unicode', xml_declaration=True)

def create_docx_bytes(markdown_text):
    register_namespaces()
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', create_content_types())
        zf.writestr('_rels/.rels', create_rels())
        zf.writestr('word/_rels/document.xml.rels', create_document_rels())
        zf.writestr('word/styles.xml', create_styles())
        zf.writestr('word/document.xml', create_document(markdown_text))
    return buffer.getvalue()

def read_chapters(chapter_dir):
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
    
    return '\n\n'.join(full_text)

if __name__ == '__main__':
    import sys
    chapter_dir = 'three-old-men/02-chapters'
    
    full_markdown = read_chapters(chapter_dir)
    print(f"Words: {len(full_markdown.split())}", file=sys.stderr)
    
    docx_bytes = create_docx_bytes(full_markdown)
    print(f"DOCX size: {len(docx_bytes)} bytes", file=sys.stderr)
    
    # Output base64
    b64 = base64.b64encode(docx_bytes).decode('ascii')
    print(b64)
