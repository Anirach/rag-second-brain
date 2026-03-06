---
name: docx-generator
description: Generate professional Word documents (.docx) using docx-js library with styled headings, TOC, headers/footers, tables, and page breaks. PRIMARY skill for creating new .docx reports, guides, manuals, and proposals. Use when creating Word documents from scratch. NOT for: reading/parsing existing DOCX files (use word-docx), Excel or PowerPoint (use office-document-specialist-suite).
---

# DOCX Generator Skill

Generate professional Word documents with full formatting support using Node.js and the docx library.

## Quick Start

```bash
# Install the docx library (required)
npm install -g docx
```

## Core Pattern

```javascript
const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat, TableOfContents,
        HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, 
        PageNumber } = require('docx');
const fs = require('fs');

const doc = new Document({
  styles: { /* style definitions */ },
  numbering: { config: [ /* list configurations */ ] },
  sections: [{ 
    properties: { page: { size: {}, margin: {} } },
    headers: { default: new Header({}) },
    footers: { default: new Footer({}) },
    children: [ /* paragraphs, tables, etc. */ ]
  }]
});

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('output.docx', buffer);
});
```

## Critical Rules

1. **Page size**: Always set explicitly (US Letter: 12240 x 15840 DXA)
2. **Never use `\n`**: Use separate Paragraph elements
3. **Never use unicode bullets**: Use `LevelFormat.BULLET` with numbering config
4. **PageBreak must be in Paragraph**: `new Paragraph({ children: [new PageBreak()] })`
5. **Tables need dual widths**: Set both `columnWidths` on table AND `width` on each cell
6. **Use ShadingType.CLEAR**: Never SOLID for table backgrounds
7. **TOC requires HeadingLevel**: Use `heading: HeadingLevel.HEADING_1` (not custom styles)
8. **Include outlineLevel in styles**: Required for TOC (0 for H1, 1 for H2, etc.)

## Document Components

### Styles Definition

```javascript
styles: {
  default: { document: { run: { font: "Arial", size: 22, color: "2D3748" } } },
  paragraphStyles: [
    { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
      quickFormat: true,
      run: { size: 36, bold: true, font: "Arial", color: "1A365D" },
      paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
    { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
      quickFormat: true,
      run: { size: 28, bold: true, font: "Arial", color: "2B6CB0" },
      paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
  ]
}
```

### Bullet and Numbered Lists

```javascript
numbering: {
  config: [
    { reference: "bullets", levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
    }]},
    { reference: "numbers", levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
    }]},
  ]
}

// Usage
new Paragraph({
  numbering: { reference: "bullets", level: 0 },
  children: [new TextRun("Bullet item")]
})
```

### Tables

```javascript
const border = { style: BorderStyle.SINGLE, size: 8, color: "CBD5E0" };
const borders = { top: border, bottom: border, left: border, right: border };

new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [4680, 4680],  // Must set at table level
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 4680, type: WidthType.DXA },  // Must match columnWidth
          shading: { fill: "EBF8FF", type: ShadingType.CLEAR },
          margins: { top: 80, bottom: 80, left: 100, right: 100 },
          children: [new Paragraph({ 
            children: [new TextRun({ text: "Header", bold: true })] 
          })]
        })
      ]
    })
  ]
})
```

### Table of Contents

```javascript
// Add TOC (updates when document is opened in Word)
new TableOfContents("Table of Contents", {
  hyperlink: true,
  headingStyleRange: "1-3"
})

// Headings MUST use HeadingLevel for TOC to work
new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun("Section Title")]
})
```

### Headers and Footers

```javascript
sections: [{
  headers: {
    default: new Header({
      children: [new Paragraph({
        alignment: AlignmentType.RIGHT,
        children: [new TextRun({ text: "Document Title", italics: true })]
      })]
    })
  },
  footers: {
    default: new Footer({
      children: [new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [
          new TextRun("Page "),
          new TextRun({ children: [PageNumber.CURRENT] }),
          new TextRun(" of "),
          new TextRun({ children: [PageNumber.TOTAL_PAGES] })
        ]
      })]
    })
  },
  children: [/* content */]
}]
```

### Page Breaks

```javascript
// Standalone page break
new Paragraph({ children: [new PageBreak()] })

// Or before a paragraph
new Paragraph({
  pageBreakBefore: true,
  children: [new TextRun("New page content")]
})
```

## Helper Functions

See `scripts/docx-helpers.js` for reusable helper functions:
- `heading1(text)`, `heading2(text)`, `heading3(text)`
- `para(text, spacing)`
- `boldPara(label, text)`
- `codeBlock(lines)`
- `createHeaderCell(text, width)`
- `createDataCell(text, width, isAlt)`
- `tipBox(title, text)`
- `warningBox(title, text)`

## Templates

See `references/` folder for complete templates:
- `report-template.js` - Professional report with TOC
- `guide-template.js` - Step-by-step installation guide
- `proposal-template.js` - Business proposal format

## Common Page Sizes (DXA units)

| Paper | Width | Height |
|-------|-------|--------|
| US Letter | 12,240 | 15,840 |
| A4 | 11,906 | 16,838 |

Note: 1440 DXA = 1 inch

## Output Location

Always save generated documents to `/mnt/user-data/outputs/` for user access.
