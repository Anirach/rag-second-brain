/**
 * Professional Report Template
 * 
 * Features:
 * - Title page with metadata
 * - Table of Contents
 * - Headers and footers with page numbers
 * - Styled headings, tables, and lists
 * - Color-coded callout boxes
 * 
 * Usage: Customize the content arrays and run with Node.js
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, PageNumber
} = require('docx');
const fs = require('fs');

// ============================================================
// CONFIGURATION - Customize these values
// ============================================================

const config = {
  title: "Report Title",
  subtitle: "Report Subtitle",
  author: "Author Name",
  date: new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }),
  version: "1.0",
  classification: "Internal - Confidential",
  outputPath: "./tmp/sample_basic_report.docx"
};

const colors = {
  primary: "1A365D",
  secondary: "2B6CB0",
  accent: "3182CE",
  headerBg: "EBF8FF",
  tableBorder: "CBD5E0",
  text: "2D3748",
  lightText: "718096"
};

// ============================================================
// HELPER FUNCTIONS
// ============================================================

const border = { style: BorderStyle.SINGLE, size: 8, color: colors.tableBorder };
const borders = { top: border, bottom: border, left: border, right: border };

function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 36, color: colors.primary })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 28, color: colors.secondary })]
  });
}

function heading3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: colors.accent })]
  });
}

function para(text, spacing = { after: 150 }) {
  return new Paragraph({
    spacing,
    children: [new TextRun({ text, font: "Arial", size: 22, color: colors.text })]
  });
}

function bulletItem(text) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    children: [new TextRun({ text, font: "Arial", size: 22 })]
  });
}

function createHeaderCell(text, width) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: colors.primary })]
    })]
  });
}

function createDataCell(text, width, isAlt = false) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: isAlt ? { fill: "F7FAFC", type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, color: colors.text })]
    })]
  });
}

// ============================================================
// DOCUMENT STRUCTURE
// ============================================================

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Arial", size: 22, color: colors.text } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: colors.primary },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: colors.secondary },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 24, bold: true, font: "Arial", color: colors.accent },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 2 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
        alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
    ]
  },
  sections: [
    // ==================== TITLE PAGE ====================
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      children: [
        new Paragraph({ spacing: { before: 2000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: config.title, font: "Arial", size: 72, bold: true, color: colors.primary })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 200 },
          children: [new TextRun({ text: config.subtitle, font: "Arial", size: 36, color: colors.secondary })]
        }),
        new Paragraph({ spacing: { before: 2000 } }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: `Author: ${config.author}`, font: "Arial", size: 22, color: colors.text })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 100 },
          children: [new TextRun({ text: `Date: ${config.date}`, font: "Arial", size: 22, color: colors.text })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 100 },
          children: [new TextRun({ text: `Version: ${config.version}`, font: "Arial", size: 22, color: colors.text })]
        }),
        new Paragraph({
          alignment: AlignmentType.CENTER,
          spacing: { before: 200 },
          children: [new TextRun({ text: config.classification, font: "Arial", size: 22, bold: true, color: colors.primary })]
        }),
      ]
    },
    // ==================== CONTENT PAGES ====================
    {
      properties: {
        page: { size: { width: 12240, height: 15840 }, margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } }
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            alignment: AlignmentType.RIGHT,
            children: [new TextRun({ text: config.title, font: "Arial", size: 18, color: colors.secondary, italics: true })]
          })]
        })
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            alignment: AlignmentType.CENTER,
            children: [
              new TextRun({ text: "Page ", font: "Arial", size: 18 }),
              new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18 }),
              new TextRun({ text: " of ", font: "Arial", size: 18 }),
              new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 18 })
            ]
          })]
        })
      },
      children: [
        // Table of Contents
        heading1("Table of Contents"),
        new TableOfContents("Table of Contents", { hyperlink: true, headingStyleRange: "1-3" }),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== SECTION 1 ====================
        heading1("Executive Summary"),
        para("This section provides a high-level overview of the report findings and recommendations."),
        bulletItem("Key finding 1"),
        bulletItem("Key finding 2"),
        bulletItem("Key finding 3"),
        new Paragraph({ spacing: { after: 200 } }),

        // ==================== SECTION 2 ====================
        heading1("Introduction"),
        heading2("Background"),
        para("Provide background context for the report here."),
        
        heading2("Objectives"),
        para("List the objectives of this report:"),
        bulletItem("Objective 1"),
        bulletItem("Objective 2"),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== SECTION 3 ====================
        heading1("Analysis"),
        heading2("Data Overview"),
        para("Present your analysis data here."),
        
        // Example table
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          columnWidths: [3000, 3000, 3360],
          rows: [
            new TableRow({ children: [
              createHeaderCell("Category", 3000),
              createHeaderCell("Value", 3000),
              createHeaderCell("Status", 3360)
            ]}),
            new TableRow({ children: [
              createDataCell("Item 1", 3000),
              createDataCell("$10,000", 3000),
              createDataCell("Complete", 3360)
            ]}),
            new TableRow({ children: [
              createDataCell("Item 2", 3000, true),
              createDataCell("$15,000", 3000, true),
              createDataCell("In Progress", 3360, true)
            ]}),
          ]
        }),
        new Paragraph({ spacing: { after: 200 } }),

        heading2("Key Findings"),
        para("Summarize your key findings here."),
        new Paragraph({ children: [new PageBreak()] }),

        // ==================== SECTION 4 ====================
        heading1("Recommendations"),
        para("Based on the analysis, we recommend the following actions:"),
        bulletItem("Recommendation 1: Description"),
        bulletItem("Recommendation 2: Description"),
        bulletItem("Recommendation 3: Description"),
        new Paragraph({ spacing: { after: 200 } }),

        // ==================== SECTION 5 ====================
        heading1("Conclusion"),
        para("Summarize the report conclusions and next steps here."),
        new Paragraph({ spacing: { after: 400 } }),

        // Footer metadata
        new Paragraph({
          alignment: AlignmentType.CENTER,
          shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
          spacing: { before: 200, after: 100 },
          children: [new TextRun({ text: `Document Version: ${config.version} | Last Updated: ${config.date}`, font: "Arial", size: 18, color: colors.lightText })]
        }),
      ]
    }
  ]
});

// ============================================================
// GENERATE DOCUMENT
// ============================================================

Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync(config.outputPath, buffer);
  console.log(`Report generated: ${config.outputPath}`);
}).catch(err => {
  console.error('Error generating report:', err);
});
