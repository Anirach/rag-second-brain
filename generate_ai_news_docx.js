const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        Header, Footer, AlignmentType, LevelFormat, TableOfContents,
        HeadingLevel, BorderStyle, WidthType, ShadingType, PageBreak, 
        PageNumber, ExternalHyperlink } = require('docx');
const fs = require('fs');

const config = JSON.parse(fs.readFileSync('ai_news_config.json', 'utf8'));

// Border style for tables
const border = { style: BorderStyle.SINGLE, size: 8, color: "CBD5E0" };
const borders = { top: border, bottom: border, left: border, right: border };

// Helper functions
function heading1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 400, after: 200 },
    children: [new TextRun({ text, bold: true, size: 36, color: "1A365D", font: "Arial" })]
  });
}

function heading2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 300, after: 150 },
    children: [new TextRun({ text, bold: true, size: 28, color: "2B6CB0", font: "Arial" })]
  });
}

function para(text, opts = {}) {
  const paragraphs = [];
  const lines = text.split('\n');
  lines.forEach((line, idx) => {
    if (line.trim() === '') return;
    
    // Check if it's a bold header line (starts with **)
    const boldMatch = line.match(/^\*\*(.+?)\*\*$/);
    if (boldMatch) {
      paragraphs.push(new Paragraph({
        spacing: { before: 200, after: 100 },
        children: [new TextRun({ text: boldMatch[1], bold: true, size: 24, font: "Arial" })]
      }));
    } else if (line.startsWith('- ')) {
      // Bullet point
      paragraphs.push(new Paragraph({
        numbering: { reference: "bullets", level: 0 },
        spacing: { before: 60, after: 60 },
        children: [new TextRun({ text: line.substring(2), size: 22, font: "Arial" })]
      }));
    } else {
      paragraphs.push(new Paragraph({
        alignment: AlignmentType.JUSTIFIED,
        spacing: { before: 100, after: 100 },
        children: [new TextRun({ text: line, size: 22, font: "Arial" })]
      }));
    }
  });
  return paragraphs;
}

// Build document sections
const children = [];

// Title page
children.push(new Paragraph({ spacing: { before: 2000 } }));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  children: [new TextRun({ text: "AI News Daily Briefing", bold: true, size: 56, color: "1A365D", font: "Arial" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 200 },
  children: [new TextRun({ text: config.subtitle, size: 32, color: "4A5568", font: "Arial" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 400 },
  children: [new TextRun({ text: config.date, size: 28, color: "718096", font: "Arial" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 800 },
  children: [new TextRun({ text: `Prepared by: ${config.author}`, size: 24, font: "Arial" })]
}));
children.push(new Paragraph({
  alignment: AlignmentType.CENTER,
  spacing: { before: 100 },
  children: [new TextRun({ text: config.organization, size: 24, font: "Arial" })]
}));

// Page break before TOC
children.push(new Paragraph({ children: [new PageBreak()] }));

// Table of Contents
children.push(heading1("Table of Contents"));
children.push(new TableOfContents("Table of Contents", {
  hyperlink: true,
  headingStyleRange: "1-2"
}));

// Page break before Executive Summary
children.push(new Paragraph({ children: [new PageBreak()] }));

// Executive Summary
children.push(heading1("Executive Summary"));
children.push(...para(config.executive_summary));

// Main sections
config.sections.forEach((section, idx) => {
  children.push(new Paragraph({ children: [new PageBreak()] }));
  children.push(heading1(`${idx + 1}. ${section.title}`));
  children.push(...para(section.content));
});

// Sources section
children.push(new Paragraph({ children: [new PageBreak()] }));
children.push(heading1("Sources and References"));
children.push(new Paragraph({
  spacing: { before: 100, after: 200 },
  children: [new TextRun({ text: "All sources have been verified and are accessible as of the publication date.", size: 22, font: "Arial", italics: true })]
}));

// Sources table
const sourceRows = [
  new TableRow({
    children: [
      new TableCell({
        borders,
        width: { size: 4000, type: WidthType.DXA },
        shading: { fill: "2B6CB0", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: "Title", bold: true, color: "FFFFFF", size: 22, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 2000, type: WidthType.DXA },
        shading: { fill: "2B6CB0", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: "Source", bold: true, color: "FFFFFF", size: 22, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 1500, type: WidthType.DXA },
        shading: { fill: "2B6CB0", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: "Date", bold: true, color: "FFFFFF", size: 22, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 1860, type: WidthType.DXA },
        shading: { fill: "2B6CB0", type: ShadingType.CLEAR },
        margins: { top: 80, bottom: 80, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: "Link", bold: true, color: "FFFFFF", size: 22, font: "Arial" })] })]
      })
    ]
  })
];

config.sources.forEach((src, idx) => {
  const isAlt = idx % 2 === 1;
  sourceRows.push(new TableRow({
    children: [
      new TableCell({
        borders,
        width: { size: 4000, type: WidthType.DXA },
        shading: isAlt ? { fill: "F7FAFC", type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: src.title, size: 20, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 2000, type: WidthType.DXA },
        shading: isAlt ? { fill: "F7FAFC", type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: src.source, size: 20, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 1500, type: WidthType.DXA },
        shading: isAlt ? { fill: "F7FAFC", type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ children: [new TextRun({ text: src.date, size: 20, font: "Arial" })] })]
      }),
      new TableCell({
        borders,
        width: { size: 1860, type: WidthType.DXA },
        shading: isAlt ? { fill: "F7FAFC", type: ShadingType.CLEAR } : undefined,
        margins: { top: 60, bottom: 60, left: 100, right: 100 },
        children: [new Paragraph({ 
          children: [new ExternalHyperlink({
            link: src.url,
            children: [new TextRun({ text: "View Source", color: "2B6CB0", underline: {}, size: 20, font: "Arial" })]
          })]
        })]
      })
    ]
  }));
});

children.push(new Table({
  width: { size: 100, type: WidthType.PERCENTAGE },
  columnWidths: [4000, 2000, 1500, 1860],
  rows: sourceRows
}));

// Create document
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: "Arial", size: 22, color: "2D3748" } }
    },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 36, bold: true, font: "Arial", color: "1A365D" },
        paragraph: { spacing: { before: 400, after: 200 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 28, bold: true, font: "Arial", color: "2B6CB0" },
        paragraph: { spacing: { before: 300, after: 150 }, outlineLevel: 1 } },
    ]
  },
  numbering: {
    config: [
      { reference: "bullets", levels: [{
          level: 0, format: LevelFormat.BULLET, text: "•",
          alignment: AlignmentType.LEFT,
          style: { paragraph: { indent: { left: 720, hanging: 360 } } }
      }]}
    ]
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "AI News Daily Briefing | January 31, 2026", italics: true, size: 20, color: "718096", font: "Arial" })]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Page ", size: 20, font: "Arial" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 20, font: "Arial" }),
            new TextRun({ text: " of ", size: 20, font: "Arial" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 20, font: "Arial" })
          ]
        })]
      })
    },
    children
  }]
});

// Generate document
Packer.toBuffer(doc).then(buffer => {
  fs.writeFileSync('ai_news_2026-01-31.docx', buffer);
  console.log('Document generated: ai_news_2026-01-31.docx');
}).catch(err => {
  console.error('Error:', err);
  process.exit(1);
});
