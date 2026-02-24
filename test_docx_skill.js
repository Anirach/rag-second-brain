/**
 * Test the new DOCX Generator Skill
 * Creates a sample professional document to demonstrate capabilities
 */

const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  Header, Footer, AlignmentType, LevelFormat, TableOfContents,
  HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, PageNumber
} = require('docx');
const fs = require('fs');

// Professional colors
const colors = {
  primary: "1A365D",
  secondary: "2B6CB0", 
  headerBg: "EBF8FF",
  tableBorder: "CBD5E0",
  text: "2D3748"
};

// Helper function for creating headings
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    children: [new TextRun({
      text: text,
      bold: true,
      size: 36,
      font: "Arial",
      color: colors.primary
    })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    children: [new TextRun({
      text: text,
      bold: true,
      size: 28,
      font: "Arial",
      color: colors.secondary
    })]
  });
}

function para(text) {
  return new Paragraph({
    children: [new TextRun({
      text: text,
      size: 22,
      font: "Arial",
      color: colors.text
    })],
    spacing: { after: 200 }
  });
}

// Create professional table
function createTable() {
  const border = { style: BorderStyle.SINGLE, size: 8, color: colors.tableBorder };
  const borders = { top: border, bottom: border, left: border, right: border };

  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    columnWidths: [3120, 3120, 3120], // 3 equal columns
    rows: [
      // Header row
      new TableRow({
        children: [
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ 
                text: "Feature", 
                bold: true,
                font: "Arial",
                size: 22 
              })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ 
                text: "Capability", 
                bold: true,
                font: "Arial",
                size: 22 
              })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            shading: { fill: colors.headerBg, type: ShadingType.CLEAR },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ 
                text: "Status", 
                bold: true,
                font: "Arial",
                size: 22 
              })]
            })]
          })
        ]
      }),
      // Data rows
      new TableRow({
        children: [
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "Professional Formatting", font: "Arial", size: 20 })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "Headers, footers, styles, TOC", font: "Arial", size: 20 })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "✅ Available", font: "Arial", size: 20, color: "276749" })]
            })]
          })
        ]
      }),
      new TableRow({
        children: [
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "Tables & Data", font: "Arial", size: 20 })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "Professional tables with styling", font: "Arial", size: 20 })]
            })]
          }),
          new TableCell({
            borders,
            width: { size: 3120, type: WidthType.DXA },
            margins: { top: 80, bottom: 80, left: 100, right: 100 },
            children: [new Paragraph({
              children: [new TextRun({ text: "✅ Available", font: "Arial", size: 20, color: "276749" })]
            })]
          })
        ]
      })
    ]
  });
}

// Create bullet list
function createBulletList() {
  return [
    new Paragraph({
      numbering: { reference: "bullets", level: 0 },
      children: [new TextRun({
        text: "Professional document generation with Node.js",
        font: "Arial",
        size: 22
      })]
    }),
    new Paragraph({
      numbering: { reference: "bullets", level: 0 },
      children: [new TextRun({
        text: "Advanced table formatting with borders and styling",
        font: "Arial", 
        size: 22
      })]
    }),
    new Paragraph({
      numbering: { reference: "bullets", level: 0 },
      children: [new TextRun({
        text: "Automatic table of contents generation",
        font: "Arial",
        size: 22
      })]
    }),
    new Paragraph({
      numbering: { reference: "bullets", level: 0 },
      children: [new TextRun({
        text: "Headers, footers, and page numbering",
        font: "Arial",
        size: 22
      })]
    })
  ];
}

// Create the document
const doc = new Document({
  styles: {
    default: {
      document: { 
        run: { font: "Arial", size: 22, color: colors.text } 
      }
    },
    paragraphStyles: [
      {
        id: "Heading1",
        name: "Heading 1",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: colors.primary },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 }
      },
      {
        id: "Heading2", 
        name: "Heading 2",
        basedOn: "Normal",
        next: "Normal",
        quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: colors.secondary },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 }
      }
    ]
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [{
          level: 0,
          format: LevelFormat.BULLET,
          text: "•",
          alignment: AlignmentType.LEFT,
          style: { 
            paragraph: { indent: { left: 720, hanging: 360 } } 
          }
        }]
      }
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } // 1 inch margins
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ 
            text: "DOCX Generator Skill - Demo Document", 
            italics: true,
            font: "Arial",
            size: 20,
            color: colors.secondary
          })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ 
              text: "Page ",
              font: "Arial",
              size: 18
            }),
            new TextRun({ 
              children: [PageNumber.CURRENT],
              font: "Arial", 
              size: 18
            }),
            new TextRun({ 
              text: " of ",
              font: "Arial",
              size: 18
            }),
            new TextRun({ 
              children: [PageNumber.TOTAL_PAGES],
              font: "Arial",
              size: 18
            })
          ]
        })]
      })
    },
    children: [
      // Title
      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "DOCX Generator Skill",
          bold: true,
          size: 48,
          font: "Arial",
          color: colors.primary
        })],
        spacing: { after: 400 }
      }),

      new Paragraph({
        alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "Professional Document Generation Demonstration",
          size: 28,
          font: "Arial",
          color: colors.secondary,
          italics: true
        })],
        spacing: { after: 600 }
      }),

      // Table of Contents
      new TableOfContents("Table of Contents", {
        hyperlink: true,
        headingStyleRange: "1-2"
      }),

      new Paragraph({ children: [new PageBreak()] }),

      // Content sections
      heading1("Overview"),
      para("This document demonstrates the capabilities of the new DOCX Generator Skill that Anirach has provided. The skill uses Node.js and the docx library to create professional Word documents with advanced formatting features."),

      para("The skill includes comprehensive helper functions, color schemes, and templates for creating various types of professional documents including reports, manuals, guides, and proposals."),

      heading2("Key Features"),
      ...createBulletList(),

      heading2("Technical Capabilities"),
      para("The DOCX generator provides enterprise-grade document creation with the following technical features:"),

      createTable(),

      new Paragraph({ children: [new PageBreak()] }),

      heading1("Implementation Details"),
      para("The skill is implemented using modern JavaScript and the docx library, providing superior formatting control compared to previous Python-based solutions."),

      heading2("Color Schemes"),
      para("The skill includes pre-defined professional color schemes including blue, gray, and green themes for consistent branding across documents."),

      heading2("Helper Functions"),
      para("Comprehensive helper functions are provided for common document elements including headings, paragraphs, tables, lists, and special formatting boxes."),

      heading1("Conclusion"),
      para("This DOCX Generator Skill provides Arthur with professional-grade document creation capabilities, enabling the generation of high-quality Word documents for any purpose including reports, manuals, proposals, and other business documents."),

      para("The skill is now integrated and ready for use in generating professional documents as requested by Anirach.")
    ]
  }]
});

// Generate the document
console.log("🚀 Generating DOCX demonstration document...");

Packer.toBuffer(doc).then(buffer => {
  // Create output directory if needed
  const outputDir = '/home/clawdbot/clawd/outputs';
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }
  
  const outputPath = `${outputDir}/docx_skill_demo.docx`;
  fs.writeFileSync(outputPath, buffer);
  
  console.log(`✅ Professional DOCX document created: ${outputPath}`);
  console.log("📄 Document includes:");
  console.log("  • Professional title page and headers/footers");
  console.log("  • Table of contents with hyperlinks"); 
  console.log("  • Styled headings and formatted text");
  console.log("  • Professional table with data");
  console.log("  • Bullet list formatting");
  console.log("  • Multiple pages with page breaks");
  console.log("  • Enterprise-grade styling and layout");
});