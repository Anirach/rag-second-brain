# DOCX-JS Quick Reference

## Installation

```bash
npm install -g docx
```

## Import Statement

```javascript
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, 
  PageNumber, ImageRun
} = require('docx');
const fs = require('fs');
```

## Document Structure

```javascript
const doc = new Document({
  styles: { ... },
  numbering: { config: [...] },
  sections: [{
    properties: { page: { size: {...}, margin: {...} } },
    headers: { default: new Header({...}) },
    footers: { default: new Footer({...}) },
    children: [...]
  }]
});

Packer.toBuffer(doc).then(buffer => fs.writeFileSync('output.docx', buffer));
```

## Page Sizes (DXA units, 1440 = 1 inch)

| Size | Width | Height |
|------|-------|--------|
| US Letter | 12240 | 15840 |
| A4 | 11906 | 16838 |
| Legal | 12240 | 20160 |

## Common Spacing Values

| Spacing | DXA Value |
|---------|-----------|
| 1/4 inch | 360 |
| 1/2 inch | 720 |
| 1 inch | 1440 |

## Font Sizes

docx-js uses half-points. Multiply desired pt by 2:
- 11pt = size: 22
- 12pt = size: 24
- 14pt = size: 28
- 18pt = size: 36
- 24pt = size: 48
- 36pt = size: 72

## Styles Configuration

```javascript
styles: {
  default: {
    document: { run: { font: "Arial", size: 22 } }
  },
  paragraphStyles: [
    {
      id: "Heading1",
      name: "Heading 1",
      basedOn: "Normal",
      next: "Normal",
      quickFormat: true,
      run: { size: 36, bold: true, font: "Arial" },
      paragraph: { 
        spacing: { before: 400, after: 200 }, 
        outlineLevel: 0  // Required for TOC!
      }
    }
  ]
}
```

## Numbering (Lists)

```javascript
numbering: {
  config: [
    {
      reference: "bullets",
      levels: [{
        level: 0,
        format: LevelFormat.BULLET,
        text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    },
    {
      reference: "numbers",
      levels: [{
        level: 0,
        format: LevelFormat.DECIMAL,
        text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    }
  ]
}
```

## Table Configuration

```javascript
const border = { style: BorderStyle.SINGLE, size: 8, color: "CCCCCC" };
const borders = { top: border, bottom: border, left: border, right: border };

new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [4680, 4680],  // Must set!
  rows: [
    new TableRow({
      children: [
        new TableCell({
          borders,
          width: { size: 4680, type: WidthType.DXA },  // Must match columnWidth!
          shading: { fill: "E0E0E0", type: ShadingType.CLEAR },  // Use CLEAR!
          margins: { top: 80, bottom: 80, left: 100, right: 100 },
          children: [new Paragraph({ children: [new TextRun("Cell")] })]
        })
      ]
    })
  ]
})
```

## Common Patterns

### Heading with HeadingLevel (for TOC)
```javascript
new Paragraph({
  heading: HeadingLevel.HEADING_1,
  children: [new TextRun("Title")]
})
```

### Page Break
```javascript
new Paragraph({ children: [new PageBreak()] })
```

### Table of Contents
```javascript
new TableOfContents("Table of Contents", {
  hyperlink: true,
  headingStyleRange: "1-3"
})
```

### Header with Page Numbers
```javascript
new Footer({
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
```

## Critical Rules

1. ❌ Never use `\n` - use separate Paragraphs
2. ❌ Never use unicode bullets (`•`) directly - use LevelFormat.BULLET
3. ❌ Never use ShadingType.SOLID - use ShadingType.CLEAR
4. ✅ Always set page size explicitly
5. ✅ Always set both table columnWidths AND cell widths
6. ✅ Always include outlineLevel in heading styles for TOC
7. ✅ Always put PageBreak inside a Paragraph
