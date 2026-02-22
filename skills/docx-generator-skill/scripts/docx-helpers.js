/**
 * DOCX Helper Functions
 * Reusable functions for creating professional Word documents
 * 
 * Usage: Copy needed functions into your document generation script
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, PageNumber
} = require('docx');

// ============================================================
// COLOR SCHEMES
// ============================================================

const colors = {
  // Professional Blue Theme
  blue: {
    primary: "1A365D",
    secondary: "2B6CB0",
    accent: "3182CE",
    headerBg: "EBF8FF",
    tableBorder: "CBD5E0",
    text: "2D3748",
    lightText: "718096",
    success: "276749",
    warning: "C05621",
    danger: "C53030"
  },
  // Corporate Gray Theme
  gray: {
    primary: "1A202C",
    secondary: "4A5568",
    accent: "718096",
    headerBg: "F7FAFC",
    tableBorder: "E2E8F0",
    text: "2D3748",
    lightText: "A0AEC0"
  },
  // Green Theme
  green: {
    primary: "1C4532",
    secondary: "276749",
    accent: "38A169",
    headerBg: "F0FFF4",
    tableBorder: "9AE6B4",
    text: "2D3748",
    lightText: "718096"
  }
};

// ============================================================
// BORDER HELPERS
// ============================================================

function createBorders(color = "CBD5E0", size = 8) {
  const border = { style: BorderStyle.SINGLE, size, color };
  return { top: border, bottom: border, left: border, right: border };
}

// ============================================================
// HEADING HELPERS
// ============================================================

function heading1(text, color = "1A365D") {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 36, color })]
  });
}

function heading2(text, color = "2B6CB0") {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 28, color })]
  });
}

function heading3(text, color = "3182CE") {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color })]
  });
}

// ============================================================
// PARAGRAPH HELPERS
// ============================================================

function para(text, spacing = { after: 150 }, textColor = "2D3748") {
  return new Paragraph({
    spacing,
    children: [new TextRun({ text, font: "Arial", size: 22, color: textColor })]
  });
}

function boldPara(label, text, spacing = { after: 150 }) {
  return new Paragraph({
    spacing,
    children: [
      new TextRun({ text: label, font: "Arial", size: 22, color: "2D3748", bold: true }),
      new TextRun({ text, font: "Arial", size: 22, color: "2D3748" })
    ]
  });
}

function centeredPara(text, size = 22, bold = false, color = "2D3748") {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text, font: "Arial", size, bold, color })]
  });
}

// ============================================================
// CODE BLOCK HELPERS
// ============================================================

function codeBlock(lines, bgColor = "F7FAFC") {
  return lines.map((line, i) => new Paragraph({
    spacing: { before: i === 0 ? 100 : 0, after: i === lines.length - 1 ? 150 : 30 },
    shading: { fill: bgColor, type: ShadingType.CLEAR },
    indent: { left: 200 },
    children: [new TextRun({ text: line, font: "Courier New", size: 18, color: "2D3748" })]
  }));
}

function inlineCode(text) {
  return new TextRun({
    text,
    font: "Courier New",
    size: 20,
    shading: { fill: "F7FAFC", type: ShadingType.CLEAR }
  });
}

// ============================================================
// TABLE HELPERS
// ============================================================

function createHeaderCell(text, width, bgColor = "EBF8FF", textColor = "1A365D") {
  return new TableCell({
    borders: createBorders(),
    width: { size: width, type: WidthType.DXA },
    shading: { fill: bgColor, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: textColor })]
    })]
  });
}

function createDataCell(text, width, isAlt = false, altColor = "F7FAFC") {
  return new TableCell({
    borders: createBorders(),
    width: { size: width, type: WidthType.DXA },
    shading: isAlt ? { fill: altColor, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, color: "2D3748" })]
    })]
  });
}

function simpleTable(headers, rows, columnWidths) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths,
    rows: [
      new TableRow({
        children: headers.map((h, i) => createHeaderCell(h, columnWidths[i]))
      }),
      ...rows.map((row, rowIndex) => new TableRow({
        children: row.map((cell, i) => createDataCell(cell, columnWidths[i], rowIndex % 2 === 1))
      }))
    ]
  });
}

// ============================================================
// CALLOUT BOX HELPERS
// ============================================================

function tipBox(title, text) {
  return new Paragraph({
    spacing: { before: 150, after: 150 },
    shading: { fill: "E6FFFA", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: "276749" } },
    indent: { left: 200 },
    children: [
      new TextRun({ text: `💡 ${title}: `, font: "Arial", size: 20, bold: true, color: "276749" }),
      new TextRun({ text, font: "Arial", size: 20, color: "2D3748" })
    ]
  });
}

function warningBox(title, text) {
  return new Paragraph({
    spacing: { before: 150, after: 150 },
    shading: { fill: "FFFAF0", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: "C05621" } },
    indent: { left: 200 },
    children: [
      new TextRun({ text: `⚠️ ${title}: `, font: "Arial", size: 20, bold: true, color: "C05621" }),
      new TextRun({ text, font: "Arial", size: 20, color: "2D3748" })
    ]
  });
}

function importantBox(title, text) {
  return new Paragraph({
    spacing: { before: 150, after: 150 },
    shading: { fill: "FFF5F5", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: "C53030" } },
    indent: { left: 200 },
    children: [
      new TextRun({ text: `🔴 ${title}: `, font: "Arial", size: 20, bold: true, color: "C53030" }),
      new TextRun({ text, font: "Arial", size: 20, color: "2D3748" })
    ]
  });
}

function infoBox(title, text) {
  return new Paragraph({
    spacing: { before: 150, after: 150 },
    shading: { fill: "EBF8FF", type: ShadingType.CLEAR },
    border: { left: { style: BorderStyle.SINGLE, size: 24, color: "3182CE" } },
    indent: { left: 200 },
    children: [
      new TextRun({ text: `ℹ️ ${title}: `, font: "Arial", size: 20, bold: true, color: "3182CE" }),
      new TextRun({ text, font: "Arial", size: 20, color: "2D3748" })
    ]
  });
}

// ============================================================
// LIST HELPERS
// ============================================================

function bulletItem(text, reference = "bullets") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

function numberedItem(text, reference = "numbers") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

function checkItem(text, reference = "checks") {
  return new Paragraph({
    numbering: { reference, level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

// ============================================================
// NUMBERING CONFIGURATIONS
// ============================================================

const standardNumbering = {
  config: [
    {
      reference: "bullets",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    },
    {
      reference: "numbers",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    },
    {
      reference: "checks",
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: "✓",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]
    },
    {
      reference: "steps",
      levels: [{
        level: 0, format: LevelFormat.DECIMAL, text: "Step %1:",
        alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 720 } } }
      }]
    }
  ]
};

// ============================================================
// STYLE CONFIGURATIONS
// ============================================================

function createStyles(colorScheme = colors.blue) {
  return {
    default: {
      document: {
        run: { font: "Arial", size: 22, color: colorScheme.text }
      }
    },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: colorScheme.primary },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: colorScheme.secondary },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 }
      },
      {
        id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: colorScheme.accent },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 }
      }
    ]
  };
}

// ============================================================
// PAGE CONFIGURATION
// ============================================================

const pageSettings = {
  usLetter: {
    size: { width: 12240, height: 15840 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
  },
  a4: {
    size: { width: 11906, height: 16838 },
    margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
  }
};

// ============================================================
// HEADER/FOOTER HELPERS
// ============================================================

function createHeader(text, alignment = AlignmentType.RIGHT) {
  return new Header({
    children: [new Paragraph({
      alignment,
      children: [new TextRun({ text, font: "Arial", size: 18, color: "718096", italics: true })]
    })]
  });
}

function createFooterWithPageNumbers() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "Page ", font: "Arial", size: 18, color: "718096" }),
        new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: "718096" }),
        new TextRun({ text: " of ", font: "Arial", size: 18, color: "718096" }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 18, color: "718096" })
      ]
    })]
  });
}

// ============================================================
// DOCUMENT BUILDER
// ============================================================

function createDocument(options = {}) {
  const {
    colorScheme = colors.blue,
    pageSize = 'usLetter',
    headerText = '',
    includePageNumbers = true,
    children = []
  } = options;

  return new Document({
    styles: createStyles(colorScheme),
    numbering: standardNumbering,
    sections: [{
      properties: {
        page: pageSettings[pageSize]
      },
      headers: headerText ? { default: createHeader(headerText) } : undefined,
      footers: includePageNumbers ? { default: createFooterWithPageNumbers() } : undefined,
      children
    }]
  });
}

// ============================================================
// EXPORTS
// ============================================================

module.exports = {
  // Colors
  colors,
  
  // Borders
  createBorders,
  
  // Headings
  heading1, heading2, heading3,
  
  // Paragraphs
  para, boldPara, centeredPara,
  
  // Code
  codeBlock, inlineCode,
  
  // Tables
  createHeaderCell, createDataCell, simpleTable,
  
  // Callout boxes
  tipBox, warningBox, importantBox, infoBox,
  
  // Lists
  bulletItem, numberedItem, checkItem,
  
  // Configurations
  standardNumbering, createStyles, pageSettings,
  
  // Header/Footer
  createHeader, createFooterWithPageNumbers,
  
  // Document builder
  createDocument
};
