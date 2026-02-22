#!/usr/bin/env node
const PDFDocument = require('pdfkit');
const fs = require('fs');
const path = require('path');

function parseArgs(args) {
  const opts = { output: 'output.pdf' };
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--title' && args[i + 1]) opts.title = args[++i];
    else if (args[i] === '--content' && args[i + 1]) opts.content = args[++i];
    else if (args[i] === '--output' && args[i + 1]) opts.output = args[++i];
  }
  return opts;
}

async function generatePDF({ title, content, output }) {
  // If content is a file path, read it
  let text = content;
  if (content && fs.existsSync(content)) {
    text = fs.readFileSync(content, 'utf-8');
  }

  const doc = new PDFDocument({ margin: 50 });
  const outputPath = path.resolve(output);
  const stream = fs.createWriteStream(outputPath);
  
  doc.pipe(stream);

  // Title
  if (title) {
    doc.fontSize(24).font('Helvetica-Bold').text(title, { align: 'center' });
    doc.moveDown(1);
  }

  // Content
  if (text) {
    doc.fontSize(12).font('Helvetica').text(text, {
      align: 'left',
      lineGap: 4
    });
  }

  doc.end();

  return new Promise((resolve, reject) => {
    stream.on('finish', () => {
      console.log(`PDF generated: ${outputPath}`);
      resolve(outputPath);
    });
    stream.on('error', reject);
  });
}

// CLI mode
if (require.main === module) {
  const opts = parseArgs(process.argv.slice(2));
  if (!opts.content) {
    console.error('Usage: node generate.js --content "text" [--title "Title"] [--output file.pdf]');
    process.exit(1);
  }
  generatePDF(opts).catch(err => {
    console.error('Error:', err.message);
    process.exit(1);
  });
}

module.exports = { generatePDF };
