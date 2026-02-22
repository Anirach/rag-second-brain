#!/usr/bin/env node
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, 
        WidthType, BorderStyle, AlignmentType, PageBreak, Header, Footer, 
        ShadingType, convertInchesToTwip } = require('docx');
const fs = require('fs');
const path = require('path');

const colors = {
  primary: '1a365d',
  secondary: '2c5282',
  accent: '3182ce',
  text: '2d3748',
  lightText: '718096',
  highlight: 'ebf8ff'
};

async function generateWhitePaper({ title, subtitle, author, date, content, output }) {
  const children = [];
  
  // Parse markdown content
  const lines = content.split('\n');
  let inReferences = false;
  
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();
    
    // Skip empty lines at start
    if (!trimmed && children.length === 0) continue;
    
    // H1 Title
    if (trimmed.startsWith('# ') && !trimmed.startsWith('## ')) {
      const text = trimmed.replace(/^# /, '').replace(/[🔬📊🎯📚]/g, '').trim();
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true, size: 56, color: colors.primary })],
        alignment: AlignmentType.CENTER,
        spacing: { before: 600, after: 200 },
      }));
      continue;
    }
    
    // H2 Subtitle or Section
    if (trimmed.startsWith('## ')) {
      const text = trimmed.replace(/^## /, '').replace(/[🔬📊🎯📚]/g, '').trim();
      
      // Check if it's a major section (add page break except for first few)
      if (children.length > 10 && !text.includes('Summary')) {
        children.push(new Paragraph({ children: [new PageBreak()] }));
      }
      
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true, size: 32, color: colors.secondary })],
        spacing: { before: 400, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: colors.accent } },
      }));
      continue;
    }
    
    // H3 Subsection
    if (trimmed.startsWith('### ')) {
      const text = trimmed.replace(/^### /, '').trim();
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true, size: 26, color: colors.primary })],
        spacing: { before: 300, after: 150 },
      }));
      continue;
    }
    
    // Horizontal rule
    if (trimmed === '---') {
      children.push(new Paragraph({
        spacing: { before: 200, after: 200 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 1, color: 'e2e8f0' } },
      }));
      continue;
    }
    
    // Check for references section
    if (trimmed.toLowerCase().includes('references')) {
      inReferences = true;
    }
    
    // Reference line [1] ...
    const refMatch = trimmed.match(/^\[(\d+)\]\s*(.+)$/);
    if (refMatch) {
      children.push(new Paragraph({
        children: [
          new TextRun({ text: `[${refMatch[1]}] `, bold: true, size: 20, color: colors.accent }),
          new TextRun({ text: refMatch[2], size: 20, color: colors.text, italics: true }),
        ],
        spacing: { after: 100 },
        indent: { left: 400, hanging: 400 },
      }));
      continue;
    }
    
    // Bold book title with author: **Title** by Author
    const bookMatch = trimmed.match(/^\*\*(.+?)\*\*\s+by\s+(.+)$/);
    if (bookMatch) {
      children.push(new Paragraph({
        children: [
          new TextRun({ text: bookMatch[1], bold: true, size: 24, color: colors.primary }),
          new TextRun({ text: ' by ', size: 22, color: colors.text }),
          new TextRun({ text: bookMatch[2], italics: true, size: 22, color: colors.text }),
        ],
        spacing: { before: 250, after: 100 },
      }));
      continue;
    }
    
    // Bold text **text**
    if (trimmed.startsWith('**') && trimmed.endsWith('**') && !trimmed.includes('by')) {
      const text = trimmed.replace(/\*\*/g, '');
      children.push(new Paragraph({
        children: [new TextRun({ text, bold: true, size: 22, color: colors.secondary })],
        spacing: { before: 200, after: 100 },
      }));
      continue;
    }
    
    // Relevance line
    if (trimmed.startsWith('**Relevance:**')) {
      const text = trimmed.replace('**Relevance:**', '').trim();
      children.push(new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [new TableRow({
          children: [new TableCell({
            children: [new Paragraph({
              children: [
                new TextRun({ text: 'Relevance: ', bold: true, size: 20, color: colors.secondary }),
                new TextRun({ text, size: 20, color: colors.text }),
              ],
            })],
            shading: { fill: colors.highlight },
            margins: { top: 100, bottom: 100, left: 150, right: 150 },
            borders: { left: { style: BorderStyle.SINGLE, size: 18, color: colors.accent } },
          })],
        })],
      }));
      children.push(new Paragraph({ spacing: { after: 150 } }));
      continue;
    }
    
    // Key Finding box
    if (trimmed.startsWith('**Key Finding:**')) {
      const text = trimmed.replace('**Key Finding:**', '').trim();
      children.push(new Table({
        width: { size: 100, type: WidthType.PERCENTAGE },
        rows: [new TableRow({
          children: [new TableCell({
            children: [
              new Paragraph({
                children: [new TextRun({ text: 'KEY FINDING', bold: true, size: 20, color: colors.secondary })],
                spacing: { after: 80 },
              }),
              new Paragraph({
                children: [new TextRun({ text, size: 22, color: colors.text })],
              }),
            ],
            shading: { fill: colors.highlight },
            margins: { top: 150, bottom: 150, left: 200, right: 200 },
            borders: { left: { style: BorderStyle.SINGLE, size: 24, color: colors.accent } },
          })],
        })],
      }));
      children.push(new Paragraph({ spacing: { after: 200 } }));
      continue;
    }
    
    // Bullet points
    if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
      const text = trimmed.substring(2);
      // Parse inline bold
      const parts = parseInlineBold(text);
      children.push(new Paragraph({
        children: parts,
        bullet: { level: 0 },
        spacing: { after: 80 },
      }));
      continue;
    }
    
    // Numbered list
    const numMatch = trimmed.match(/^(\d+)\.\s+\*\*(.+?)\*\*(.*)$/);
    if (numMatch) {
      children.push(new Paragraph({
        children: [
          new TextRun({ text: `${numMatch[1]}. `, bold: true, size: 22, color: colors.accent }),
          new TextRun({ text: numMatch[2], bold: true, size: 22, color: colors.primary }),
          new TextRun({ text: numMatch[3], size: 22, color: colors.text }),
        ],
        spacing: { after: 80 },
        indent: { left: 200 },
      }));
      continue;
    }
    
    // Regular paragraph
    if (trimmed) {
      const parts = parseInlineBold(trimmed);
      children.push(new Paragraph({
        children: parts,
        spacing: { after: 150 },
        alignment: AlignmentType.JUSTIFIED,
      }));
    } else {
      // Empty line = small space
      children.push(new Paragraph({ spacing: { after: 100 } }));
    }
  }

  const doc = new Document({
    styles: {
      paragraphStyles: [{
        id: 'Normal',
        name: 'Normal',
        run: { font: 'Calibri', size: 22 },
      }],
    },
    sections: [{
      properties: {
        page: {
          margin: {
            top: convertInchesToTwip(1),
            right: convertInchesToTwip(1),
            bottom: convertInchesToTwip(1),
            left: convertInchesToTwip(1),
          },
        },
      },
      headers: {
        default: new Header({
          children: [new Paragraph({
            children: [new TextRun({ text: title.toUpperCase(), size: 18, color: colors.lightText })],
            alignment: AlignmentType.RIGHT,
          })],
        }),
      },
      footers: {
        default: new Footer({
          children: [new Paragraph({
            children: [new TextRun({ text: 'STRATEGIC DEVELOPMENT CLASS | CONFIDENTIAL', size: 16, color: colors.lightText })],
            alignment: AlignmentType.CENTER,
          })],
        }),
      },
      children,
    }],
  });

  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.resolve(output);
  fs.writeFileSync(outputPath, buffer);
  console.log(`White paper generated: ${outputPath}`);
  return outputPath;
}

function parseInlineBold(text) {
  const parts = [];
  const regex = /\*\*(.+?)\*\*/g;
  let lastIndex = 0;
  let match;
  
  while ((match = regex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(new TextRun({ text: text.slice(lastIndex, match.index), size: 22, color: '2d3748' }));
    }
    parts.push(new TextRun({ text: match[1], bold: true, size: 22, color: '2c5282' }));
    lastIndex = regex.lastIndex;
  }
  
  if (lastIndex < text.length) {
    parts.push(new TextRun({ text: text.slice(lastIndex), size: 22, color: '2d3748' }));
  }
  
  return parts.length > 0 ? parts : [new TextRun({ text, size: 22, color: '2d3748' })];
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  let mdFile = null;
  let output = 'whitepaper.docx';
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i+1]) mdFile = args[++i];
    else if (args[i] === '--output' && args[i+1]) output = args[++i];
  }

  if (!mdFile) {
    console.error('Usage: node whitepaper-docx.js --input file.md --output report.docx');
    process.exit(1);
  }

  const content = fs.readFileSync(mdFile, 'utf-8');
  
  generateWhitePaper({
    title: 'Should Executives Read Literary Fiction?',
    subtitle: 'A White Paper on Strategic Thinking Under Uncertainty',
    author: 'Arthur for Anirach',
    date: 'January 2026',
    content,
    output
  }).catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}

module.exports = { generateWhitePaper };
