#!/usr/bin/env node
const { Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell, 
        WidthType, BorderStyle, AlignmentType, PageBreak, Header, Footer, 
        TableOfContents, StyleLevel, ShadingType, convertInchesToTwip } = require('docx');
const fs = require('fs');
const path = require('path');

// Color scheme
const colors = {
  primary: '1a365d',
  secondary: '2c5282',
  accent: '3182ce',
  text: '2d3748',
  lightText: '718096',
  highlight: 'ebf8ff'
};

async function generateProfessionalDocx({ title, subtitle, author, date, sections, references, output }) {
  const children = [];

  // ============ COVER PAGE ============
  // Title
  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: title,
          bold: true,
          size: 72,
          color: colors.primary,
        }),
      ],
      alignment: AlignmentType.CENTER,
      spacing: { before: 1200, after: 400 },
    })
  );

  // Subtitle
  if (subtitle) {
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: subtitle,
            size: 32,
            color: colors.secondary,
          }),
        ],
        alignment: AlignmentType.CENTER,
        spacing: { after: 800 },
      })
    );
  }

  // Info table
  children.push(
    new Table({
      width: { size: 100, type: WidthType.PERCENTAGE },
      rows: [
        new TableRow({
          children: [
            new TableCell({
              children: [
                new Paragraph({
                  children: [
                    new TextRun({ text: 'PREPARED BY', bold: true, size: 20, color: colors.lightText }),
                  ],
                }),
                new Paragraph({
                  children: [
                    new TextRun({ text: author || 'Research Team', size: 24, color: colors.text }),
                  ],
                }),
              ],
              width: { size: 50, type: WidthType.PERCENTAGE },
              shading: { fill: 'f7fafc' },
              margins: { top: 200, bottom: 200, left: 200, right: 200 },
            }),
            new TableCell({
              children: [
                new Paragraph({
                  children: [
                    new TextRun({ text: 'DATE', bold: true, size: 20, color: colors.lightText }),
                  ],
                }),
                new Paragraph({
                  children: [
                    new TextRun({ text: date || new Date().toLocaleDateString(), size: 24, color: colors.text }),
                  ],
                }),
              ],
              width: { size: 50, type: WidthType.PERCENTAGE },
              shading: { fill: 'f7fafc' },
              margins: { top: 200, bottom: 200, left: 200, right: 200 },
            }),
          ],
        }),
      ],
      borders: {
        top: { style: BorderStyle.SINGLE, size: 1, color: 'e2e8f0' },
        bottom: { style: BorderStyle.SINGLE, size: 1, color: 'e2e8f0' },
        left: { style: BorderStyle.SINGLE, size: 1, color: 'e2e8f0' },
        right: { style: BorderStyle.SINGLE, size: 1, color: 'e2e8f0' },
      },
    })
  );

  // Executive Summary
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'EXECUTIVE BRIEFING', bold: true, size: 28, color: colors.secondary }),
      ],
      spacing: { before: 600, after: 200 },
    })
  );

  children.push(
    new Paragraph({
      children: [
        new TextRun({
          text: 'This report provides a comprehensive overview of the latest research developments and key findings in the field. Each section highlights critical discoveries with practical implications for stakeholders.',
          size: 22,
          color: colors.text,
        }),
      ],
      spacing: { after: 400 },
    })
  );

  // Table of Contents header
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'CONTENTS', bold: true, size: 28, color: colors.secondary }),
      ],
      spacing: { before: 400, after: 200 },
    })
  );

  // TOC entries
  sections.forEach((section, i) => {
    children.push(
      new Paragraph({
        children: [
          new TextRun({ text: `${String(i + 1).padStart(2, '0')}  `, bold: true, size: 22, color: colors.accent }),
          new TextRun({ text: section.title, size: 22, color: colors.text }),
        ],
        spacing: { after: 100 },
      })
    );
  });

  // Page break after cover
  children.push(new Paragraph({ children: [new PageBreak()] }));

  // ============ CONTENT SECTIONS ============
  sections.forEach((section, sectionIndex) => {
    // Section number
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: String(sectionIndex + 1).padStart(2, '0'),
            bold: true,
            size: 72,
            color: colors.accent,
          }),
        ],
        spacing: { before: 400, after: 200 },
      })
    );

    // Section title
    children.push(
      new Paragraph({
        children: [
          new TextRun({
            text: section.title,
            bold: true,
            size: 36,
            color: colors.primary,
          }),
        ],
        spacing: { after: 100 },
        border: {
          bottom: { style: BorderStyle.SINGLE, size: 12, color: colors.accent },
        },
      })
    );

    // Key insight box
    if (section.highlight) {
      children.push(
        new Table({
          width: { size: 100, type: WidthType.PERCENTAGE },
          rows: [
            new TableRow({
              children: [
                new TableCell({
                  children: [
                    new Paragraph({
                      children: [
                        new TextRun({ text: 'KEY INSIGHT', bold: true, italics: true, size: 22, color: colors.secondary }),
                      ],
                      spacing: { after: 100 },
                    }),
                    new Paragraph({
                      children: [
                        new TextRun({ text: section.highlight, size: 22, color: colors.text }),
                      ],
                    }),
                  ],
                  shading: { fill: colors.highlight },
                  margins: { top: 200, bottom: 200, left: 200, right: 200 },
                  borders: {
                    left: { style: BorderStyle.SINGLE, size: 24, color: colors.accent },
                  },
                }),
              ],
            }),
          ],
        })
      );
      children.push(new Paragraph({ spacing: { after: 200 } }));
    }

    // Content
    const contentLines = section.content.split('\n');
    contentLines.forEach(line => {
      const trimmed = line.trim();
      if (!trimmed) return;

      if (trimmed.startsWith('- ') || trimmed.startsWith('• ')) {
        children.push(
          new Paragraph({
            children: [
              new TextRun({ text: trimmed.substring(2), size: 22, color: colors.text }),
            ],
            bullet: { level: 0 },
            spacing: { after: 80 },
          })
        );
      } else if (trimmed.startsWith('**Source:')) {
        children.push(
          new Paragraph({
            children: [
              new TextRun({ text: trimmed.replace(/\*\*/g, ''), bold: true, size: 22, color: colors.secondary }),
            ],
            spacing: { before: 200, after: 100 },
          })
        );
      } else if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
        children.push(
          new Paragraph({
            children: [
              new TextRun({ text: trimmed.replace(/\*\*/g, ''), bold: true, size: 24, color: colors.secondary }),
            ],
            spacing: { before: 200, after: 100 },
          })
        );
      } else {
        children.push(
          new Paragraph({
            children: [
              new TextRun({ text: trimmed, size: 22, color: colors.text }),
            ],
            spacing: { after: 120 },
          })
        );
      }
    });

    // Page break between sections (except last)
    if (sectionIndex < sections.length - 1) {
      children.push(new Paragraph({ children: [new PageBreak()] }));
    }
  });

  // ============ REFERENCES ============
  if (references && references.length > 0) {
    children.push(new Paragraph({ children: [new PageBreak()] }));
    
    children.push(
      new Paragraph({
        children: [
          new TextRun({ text: 'References', bold: true, size: 40, color: colors.primary }),
        ],
        spacing: { after: 100 },
        border: {
          bottom: { style: BorderStyle.SINGLE, size: 12, color: colors.accent },
        },
      })
    );

    references.forEach(ref => {
      children.push(
        new Paragraph({
          children: [
            new TextRun({ text: `[${ref.num}] `, bold: true, size: 20, color: colors.accent }),
            new TextRun({ text: ref.text, size: 20, color: colors.text }),
          ],
          spacing: { after: 150 },
        })
      );
    });
  }

  // Create document
  const doc = new Document({
    styles: {
      paragraphStyles: [
        {
          id: 'Normal',
          name: 'Normal',
          run: {
            font: 'Calibri',
            size: 22,
          },
        },
      ],
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
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: title.toUpperCase(), size: 18, color: colors.lightText }),
              ],
              alignment: AlignmentType.RIGHT,
            }),
          ],
        }),
      },
      footers: {
        default: new Footer({
          children: [
            new Paragraph({
              children: [
                new TextRun({ text: 'CONFIDENTIAL', size: 16, color: colors.lightText }),
              ],
              alignment: AlignmentType.CENTER,
            }),
          ],
        }),
      },
      children,
    }],
  });

  // Write file
  const buffer = await Packer.toBuffer(doc);
  const outputPath = path.resolve(output);
  fs.writeFileSync(outputPath, buffer);
  console.log(`DOCX report generated: ${outputPath}`);
  return outputPath;
}

// Parse markdown (same as PDF version)
function parseMarkdownToSections(markdown) {
  const sections = [];
  const references = [];
  const lines = markdown.split('\n');
  let currentSection = null;
  let inReferences = false;

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    
    if (line.match(/^##.*References/i)) {
      if (currentSection && currentSection.content.trim()) {
        sections.push(currentSection);
      }
      currentSection = null;
      inReferences = true;
      continue;
    }

    if (inReferences) {
      const refMatch = line.match(/^\[(\d+)\]\s*(.+)$/);
      if (refMatch) {
        references.push({ num: refMatch[1], text: refMatch[2] });
      }
      continue;
    }

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

    const h2Match = line.match(/^##\s+(.+)$/);
    if (h2Match && !line.includes('Summary') && !line.includes('References')) {
      if (currentSection && currentSection.content.trim()) {
        sections.push(currentSection);
      }
      currentSection = {
        title: h2Match[1].replace(/[🔬📊🎯📚]/g, '').trim(),
        content: '',
        highlight: ''
      };
      continue;
    }

    if (currentSection) {
      if (line.includes('**Implication:**')) {
        currentSection.highlight = line.replace('**Implication:**', '').trim();
      } else if (!line.startsWith('#') && !line.startsWith('---') && !line.startsWith('Compiled by') && !line.startsWith('Date:')) {
        currentSection.content += line + '\n';
      }
    }
  }
  
  if (currentSection && currentSection.content.trim()) {
    sections.push(currentSection);
  }

  return { sections, references };
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  let mdFile = null;
  let output = 'report.docx';
  
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--input' && args[i+1]) mdFile = args[++i];
    else if (args[i] === '--output' && args[i+1]) output = args[++i];
  }

  if (!mdFile) {
    console.error('Usage: node professional-docx.js --input file.md --output report.docx');
    process.exit(1);
  }

  const markdown = fs.readFileSync(mdFile, 'utf-8');
  const { sections, references } = parseMarkdownToSections(markdown);
  
  console.log(`Parsed ${sections.length} sections and ${references.length} references`);

  generateProfessionalDocx({
    title: 'Longevity Research Update',
    subtitle: 'January 2026 Summary',
    author: 'Arthur 🐕 for Anirach',
    date: 'January 26, 2026',
    sections,
    references,
    output
  }).catch(err => {
    console.error('Error:', err);
    process.exit(1);
  });
}

module.exports = { generateProfessionalDocx, parseMarkdownToSections };
