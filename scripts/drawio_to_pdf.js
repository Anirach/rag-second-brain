#!/usr/bin/env node
/**
 * Convert draw.io files to PDF using Puppeteer
 * Usage: node drawio_to_pdf.js input.drawio output.pdf
 */

const puppeteer = require('puppeteer');
const fs = require('fs');
const path = require('path');

async function convertDrawioToPdf(inputFile, outputFile) {
  // Read the drawio XML
  const xmlContent = fs.readFileSync(inputFile, 'utf8');
  
  // Extract the diagram content (mxGraphModel)
  const diagramMatch = xmlContent.match(/<diagram[^>]*>([\s\S]*?)<\/diagram>/);
  if (!diagramMatch) {
    throw new Error('No diagram found in file');
  }
  
  // Create HTML that renders the diagram using draw.io viewer
  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body { margin: 0; padding: 20px; font-family: Arial, sans-serif; }
    .mxgraph { max-width: 100%; }
  </style>
  <script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</head>
<body>
  <div class="mxgraph" data-mxgraph='${JSON.stringify({ highlight: "#0000ff", nav: false, resize: true, xml: xmlContent })}'>
  </div>
</body>
</html>`;

  // Launch browser
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  try {
    const page = await browser.newPage();
    
    // Set content and wait for render
    await page.setContent(html, { waitUntil: 'networkidle0', timeout: 30000 });
    
    // Wait for diagram to render
    await page.waitForTimeout(2000);
    
    // Generate PDF
    await page.pdf({
      path: outputFile,
      format: 'A4',
      printBackground: true,
      margin: { top: '20mm', right: '20mm', bottom: '20mm', left: '20mm' }
    });
    
    console.log(`✅ PDF generated: ${outputFile}`);
  } finally {
    await browser.close();
  }
}

// Alternative: Convert to SVG first using mxGraph
async function convertDrawioToSvgPdf(inputFile, outputFile) {
  const xmlContent = fs.readFileSync(inputFile, 'utf8');
  
  // Create simple HTML for PDF
  const html = `
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>
    body { margin: 0; padding: 40px; font-family: Arial, sans-serif; background: white; }
    h1 { color: #1a365d; margin-bottom: 30px; }
    .diagram-container { 
      border: 1px solid #e2e8f0; 
      padding: 20px; 
      background: #fafafa;
      border-radius: 8px;
    }
  </style>
</head>
<body>
  <div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph='${JSON.stringify({ highlight: "#0000ff", nav: false, resize: true, xml: xmlContent }).replace(/'/g, "&#39;")}'>
  </div>
  <script src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>
</body>
</html>`;

  const browser = await puppeteer.launch({
    headless: 'new',
    args: ['--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage']
  });
  
  try {
    const page = await browser.newPage();
    await page.setViewport({ width: 1200, height: 800 });
    
    await page.setContent(html, { waitUntil: 'networkidle0', timeout: 60000 });
    
    // Wait for scripts to load and render
    await new Promise(r => setTimeout(r, 3000));
    
    await page.pdf({
      path: outputFile,
      format: 'A4',
      printBackground: true,
      margin: { top: '15mm', right: '15mm', bottom: '15mm', left: '15mm' }
    });
    
    console.log(`✅ PDF generated: ${outputFile}`);
    return true;
  } catch (err) {
    console.error('Error:', err.message);
    return false;
  } finally {
    await browser.close();
  }
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('Usage: node drawio_to_pdf.js <input.drawio> <output.pdf>');
    process.exit(1);
  }
  
  const inputFile = args[0];
  const outputFile = args[1];
  
  if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file not found: ${inputFile}`);
    process.exit(1);
  }
  
  convertDrawioToSvgPdf(inputFile, outputFile)
    .then(success => process.exit(success ? 0 : 1))
    .catch(err => {
      console.error('Error:', err);
      process.exit(1);
    });
}

module.exports = { convertDrawioToPdf, convertDrawioToSvgPdf };
