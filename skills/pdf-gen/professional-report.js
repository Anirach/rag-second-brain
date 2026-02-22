#!/usr/bin/env node
const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

// Professional color scheme
const colors = {
  primary: '#1a365d',
  secondary: '#2c5282',
  accent: '#3182ce',
  text: '#2d3748',
  lightText: '#718096',
  background: '#f7fafc',
  white: '#ffffff',
  divider: '#e2e8f0'
};

// Font paths
const fontsDir = path.join(__dirname, '../../fonts');
const fonts = {
  regular: fs.existsSync(path.join(fontsDir, 'NotoSans-Regular.ttf')) 
    ? path.join(fontsDir, 'NotoSans-Regular.ttf') : 'Helvetica',
  bold: fs.existsSync(path.join(fontsDir, 'NotoSans-Bold.ttf'))
    ? path.join(fontsDir, 'NotoSans-Bold.ttf') : 'Helvetica-Bold',
};

// Replace emojis with text for PDF compatibility (fallback if font doesn't support)
function sanitizeForPDF(text) {
  if (!text) return '';
  return text
    .replace(/🔬/g, '')
    .replace(/📊/g, '')
    .replace(/🎯/g, '')
    .replace(/📚/g, '')
    .replace(/🐕/g, '')
    .replace(/[\u{1F300}-\u{1F9FF}]/gu, '')
    .replace(/[\u{2600}-\u{26FF}]/gu, '')
    .replace(/[\u{2700}-\u{27BF}]/gu, '')
    .replace(/\s+/g, ' ')
    .trim();
}

function generateProfessionalReport({ title, subtitle, author, date, sections, references, output }) {
  const doc = new PDFDocument({ 
    size: 'A4',
    margins: { top: 60, bottom: 60, left: 60, right: 60 },
    bufferPages: true
  });
  
  const outputPath = path.resolve(output);
  const stream = fs.createWriteStream(outputPath);
  doc.pipe(stream);

  // Register fonts if available
  if (fonts.regular !== 'Helvetica') {
    doc.registerFont('Regular', fonts.regular);
    doc.registerFont('Bold', fonts.bold);
  }
  const fontRegular = fonts.regular !== 'Helvetica' ? 'Regular' : 'Helvetica';
  const fontBold = fonts.bold !== 'Helvetica-Bold' ? 'Bold' : 'Helvetica-Bold';

  const pageWidth = doc.page.width;
  const pageHeight = doc.page.height;
  const contentWidth = pageWidth - 120;

  // ============ COVER PAGE ============
  doc.rect(0, 0, pageWidth, 180).fill(colors.primary);
  
  doc.fillColor(colors.white)
     .font(fontBold)
     .fontSize(32)
     .text(sanitizeForPDF(title), 60, 80, { width: contentWidth, align: 'center' });

  if (subtitle) {
    doc.fontSize(16)
       .font(fontRegular)
       .text(sanitizeForPDF(subtitle), 60, 130, { width: contentWidth, align: 'center' });
  }

  // Report info box
  doc.roundedRect(60, 220, contentWidth, 100, 5)
     .fillAndStroke(colors.background, colors.divider);

  doc.fillColor(colors.text).font(fontBold).fontSize(11).text('PREPARED BY', 80, 240);
  doc.font(fontRegular).fontSize(12).text(sanitizeForPDF(author) || 'Research Team', 80, 255);

  doc.font(fontBold).fontSize(11).text('DATE', 300, 240);
  doc.font(fontRegular).fontSize(12).text(date || new Date().toLocaleDateString('en-US', { year: 'numeric', month: 'long', day: 'numeric' }), 300, 255);

  doc.font(fontBold).fontSize(11).text('DOCUMENT TYPE', 80, 285);
  doc.font(fontRegular).fontSize(12).text('Research Summary Report', 80, 300);

  // Executive summary
  doc.fillColor(colors.secondary).font(fontBold).fontSize(14).text('EXECUTIVE BRIEFING', 60, 360);
  doc.fillColor(colors.text).font(fontRegular).fontSize(11)
     .text('This report provides a comprehensive overview of the latest research developments and key findings in the field. Each section highlights critical discoveries with practical implications for stakeholders.', 60, 385, { width: contentWidth, lineGap: 4 });

  // Table of contents
  doc.fillColor(colors.secondary).font(fontBold).fontSize(14).text('CONTENTS', 60, 480);
  let tocY = 505;
  sections.forEach((section, i) => {
    doc.fillColor(colors.text).font(fontRegular).fontSize(11);
    const num = String(i + 1).padStart(2, '0');
    doc.text(`${num}`, 60, tocY);
    doc.text(sanitizeForPDF(section.title).toUpperCase().substring(0, 50), 90, tocY);
    tocY += 22;
    if (tocY > 700) return; // Limit TOC entries on cover
  });

  doc.fillColor(colors.lightText).fontSize(9)
     .text('CONFIDENTIAL', 60, pageHeight - 40, { width: contentWidth, align: 'center' });

  // ============ CONTENT PAGES ============
  sections.forEach((section, sectionIndex) => {
    doc.addPage();
    
    // Page header
    doc.rect(0, 0, pageWidth, 50).fill(colors.primary);
    doc.fillColor(colors.white).font(fontBold).fontSize(10)
       .text(sanitizeForPDF(title).toUpperCase(), 60, 20);
    doc.fillColor('#a0aec0').font(fontRegular).fontSize(9)
       .text(date || new Date().toLocaleDateString(), pageWidth - 150, 20, { width: 90, align: 'right' });

    // Section number and title
    doc.fillColor(colors.accent).font(fontBold).fontSize(40)
       .text(String(sectionIndex + 1).padStart(2, '0'), 60, 70);
    
    const sectionTitle = sanitizeForPDF(section.title);
    doc.fillColor(colors.primary).font(fontBold).fontSize(20);
    const titleHeight = doc.heightOfString(sectionTitle, { width: contentWidth });
    doc.text(sectionTitle, 60, 120, { width: contentWidth });
    
    const titleEndY = 120 + titleHeight + 10;
    doc.moveTo(60, titleEndY).lineTo(200, titleEndY).strokeColor(colors.accent).lineWidth(3).stroke();

    let yPos = titleEndY + 20;
    
    // Key insight box
    if (section.highlight) {
      const highlightText = sanitizeForPDF(section.highlight);
      const boxHeight = Math.max(60, doc.heightOfString(highlightText, { width: contentWidth - 30 }) + 40);
      doc.roundedRect(60, yPos, contentWidth, boxHeight, 5).fillAndStroke('#ebf8ff', colors.accent);
      doc.fillColor(colors.secondary).font(fontBold).fontSize(11).text('KEY INSIGHT', 75, yPos + 12);
      doc.fillColor(colors.text).font(fontRegular).fontSize(10)
         .text(highlightText, 75, yPos + 28, { width: contentWidth - 30, lineGap: 2 });
      yPos += boxHeight + 20;
    }

    // Content
    const contentText = sanitizeForPDF(section.content);
    const contentLines = contentText.split('\n');
    
    contentLines.forEach(line => {
      if (yPos > pageHeight - 100) {
        doc.addPage();
        doc.rect(0, 0, pageWidth, 30).fill(colors.primary);
        doc.fillColor(colors.white).font(fontRegular).fontSize(9)
           .text(sectionTitle + ' (continued)', 60, 10);
        yPos = 50;
      }

      const trimmedLine = line.trim();
      if (!trimmedLine) {
        yPos += 8;
        return;
      }

      if (trimmedLine.startsWith('- ') || trimmedLine.startsWith('• ')) {
        const bulletText = trimmedLine.substring(2);
        doc.circle(70, yPos + 5, 2).fill(colors.accent);
        doc.fillColor(colors.text).font(fontRegular).fontSize(10)
           .text(bulletText, 82, yPos, { width: contentWidth - 22, lineGap: 3 });
        yPos += doc.heightOfString(bulletText, { width: contentWidth - 22 }) + 8;
      } else if (trimmedLine.startsWith('**') && trimmedLine.endsWith('**')) {
        const headerText = trimmedLine.replace(/\*\*/g, '');
        doc.fillColor(colors.secondary).font(fontBold).fontSize(12).text(headerText, 60, yPos);
        yPos += 22;
      } else {
        doc.fillColor(colors.text).font(fontRegular).fontSize(10)
           .text(trimmedLine, 60, yPos, { width: contentWidth, lineGap: 4, align: 'justify' });
        yPos += doc.heightOfString(trimmedLine, { width: contentWidth }) + 12;
      }
    });

    // Page footer
    doc.fillColor(colors.divider);
    doc.moveTo(60, pageHeight - 50).lineTo(pageWidth - 60, pageHeight - 50).stroke();
    doc.fillColor(colors.lightText).fontSize(9)
       .text(`Page ${sectionIndex + 2}`, 60, pageHeight - 40, { width: contentWidth, align: 'center' });
  });

  // ============ REFERENCES PAGE ============
  if (references && references.length > 0) {
    doc.addPage();
    
    doc.rect(0, 0, pageWidth, 50).fill(colors.primary);
    doc.fillColor(colors.white).font(fontBold).fontSize(10)
       .text(sanitizeForPDF(title).toUpperCase(), 60, 20);

    doc.fillColor(colors.primary).font(fontBold).fontSize(20).text('References', 60, 70);
    doc.moveTo(60, 100).lineTo(200, 100).strokeColor(colors.accent).lineWidth(3).stroke();

    let refY = 120;
    references.forEach((ref) => {
      if (refY > pageHeight - 100) {
        doc.addPage();
        doc.rect(0, 0, pageWidth, 30).fill(colors.primary);
        doc.fillColor(colors.white).font(fontRegular).fontSize(9).text('References (continued)', 60, 10);
        refY = 50;
      }

      doc.fillColor(colors.accent).font(fontBold).fontSize(10).text(`[${ref.num}]`, 60, refY);
      
      const refText = sanitizeForPDF(ref.text);
      doc.fillColor(colors.text).font(fontRegular).fontSize(9)
         .text(refText, 90, refY, { width: contentWidth - 40, lineGap: 2 });
      refY += doc.heightOfString(refText, { width: contentWidth - 40 }) + 15;
    });

    doc.fillColor(colors.divider);
    doc.moveTo(60, pageHeight - 50).lineTo(pageWidth - 60, pageHeight - 50).stroke();
    doc.fillColor(colors.lightText).fontSize(9)
       .text('References', 60, pageHeight - 40, { width: contentWidth, align: 'center' });
  }

  // ============ BACK COVER ============
  doc.addPage();
  doc.rect(0, pageHeight - 180, pageWidth, 180).fill(colors.primary);
  
  doc.fillColor(colors.text).font(fontBold).fontSize(16).text('About This Report', 60, 80);
  doc.fillColor(colors.text).font(fontRegular).fontSize(10)
     .text('This report was compiled using the latest research from peer-reviewed journals, scientific publications, and trusted industry sources. The information presented is intended for educational and strategic planning purposes.', 60, 110, { width: contentWidth, lineGap: 4 });

  doc.fillColor(colors.text).font(fontBold).fontSize(12).text('Methodology', 60, 180);
  doc.font(fontRegular).fontSize(9)
     .text('Research gathered from peer-reviewed journals (Nature, Nature Aging, Nature Communications, Nature Metabolism) and curated longevity research platforms. All findings are cited with DOI links for verification.', 60, 200, { width: contentWidth, lineGap: 3 });

  doc.fillColor(colors.white).font(fontBold).fontSize(14)
     .text(sanitizeForPDF(author) || 'Research Team', 60, pageHeight - 100, { width: contentWidth, align: 'center' });
  doc.fillColor('#a0aec0').font(fontRegular).fontSize(10)
     .text(date || new Date().toLocaleDateString(), 60, pageHeight - 80, { width: contentWidth, align: 'center' });

  doc.end();

  return new Promise((resolve, reject) => {
    stream.on('finish', () => {
      console.log(`Professional report generated: ${outputPath}`);
      resolve(outputPath);
    });
    stream.on('error', reject);
  });
}

// Parse markdown into sections
function parseMarkdownToSections(markdown) {
  const sections = [];
  const references = [];
  const lines = markdown.split('\n');
  let currentSection = null;
  let inReferences = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    // Check for references section
    if (line.match(/^##.*References/i)) {
      if (currentSection && currentSection.content.trim()) {
        sections.push(currentSection);
      }
      currentSection = null;
      inReferences = true;
      continue;
    }

    // Parse references
    if (inReferences) {
      const refMatch = line.match(/^\[(\d+)\]\s*(.+)$/);
      if (refMatch) {
        references.push({ num: refMatch[1], text: refMatch[2] });
      }
      continue;
    }

    // Match numbered section headers (### 1. Title)
    const numberedSection = line.match(/^###\s+\d+\.\s+(.+)$/);
    if (numberedSection) {
      if (currentSection && currentSection.content.trim()) {
        sections.push(currentSection);
      }
      currentSection = {
        title: numberedSection[1].trim(),
        content: '',
        highlight: ''
      };
      continue;
    }

    // Match H2 headers (## Title) - but skip first title and references
    const h2Match = line.match(/^##\s+(.+)$/);
    if (h2Match && !line.includes('Summary') && !line.includes('References')) {
      if (currentSection && currentSection.content.trim()) {
        sections.push(currentSection);
      }
      currentSection = {
        title: h2Match[1].trim(),
        content: '',
        highlight: ''
      };
      continue;
    }

    // Add content to current section
    if (currentSection) {
      if (line.includes('**Implication:**')) {
        currentSection.highlight = line.replace('**Implication:**', '').trim();
      } else if (!line.startsWith('#') && !line.startsWith('---') && !line.startsWith('Compiled by') && !line.startsWith('Date:')) {
        currentSection.content += line + '\n';
      }
    }
  }
  
  // Don't forget the last section
  if (currentSection && currentSection.content.trim()) {
    sections.push(currentSection);
  }

  return { sections, references };
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  let mdFile = null;
  let output = 'professional-report.pdf';
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i+1]) mdFile = args[++i];
    else if (args[i] === '--output' && args[i+1]) output = args[++i];
  }

  if (!mdFile) {
    console.error('Usage: node professional-report.js --input file.md --output report.pdf');
    process.exit(1);
  }

  const markdown = fs.readFileSync(mdFile, 'utf-8');
  const { sections, references } = parseMarkdownToSections(markdown);
  
  console.log(`Parsed ${sections.length} sections and ${references.length} references`);

  generateProfessionalReport({
    title: 'Longevity Research Update',
    subtitle: 'January 2026 Summary',
    author: 'Arthur for Anirach',
    date: 'January 26, 2026',
    sections,
    references,
    output
  }).catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}

module.exports = { generateProfessionalReport, parseMarkdownToSections };
