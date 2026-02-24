#!/usr/bin/env node
/**
 * Convert draw.io files to PDF using PDFKit (native, no browser needed)
 * Usage: node drawio_to_pdf_native.js input.drawio output.pdf
 */

const PDFDocument = require('pdfkit');
const fs = require('fs');
const { DOMParser } = require('xmldom');

// Color mappings from hex
function hexToRgb(hex) {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : { r: 200, g: 200, b: 200 };
}

// Parse style string into object
function parseStyle(styleStr) {
  const style = {};
  if (!styleStr) return style;
  
  styleStr.split(';').forEach(item => {
    const [key, value] = item.split('=');
    if (key && value) {
      style[key.trim()] = value.trim();
    } else if (key) {
      style[key.trim()] = true;
    }
  });
  return style;
}

// Draw a node/cell
function drawCell(doc, cell, scale = 0.85) {
  const geometry = cell.getElementsByTagName('mxGeometry')[0];
  if (!geometry) return;
  
  const x = (parseFloat(geometry.getAttribute('x')) || 0) * scale + 50;
  const y = (parseFloat(geometry.getAttribute('y')) || 0) * scale + 80;
  const width = (parseFloat(geometry.getAttribute('width')) || 100) * scale;
  const height = (parseFloat(geometry.getAttribute('height')) || 60) * scale;
  const value = cell.getAttribute('value') || '';
  const styleStr = cell.getAttribute('style') || '';
  const style = parseStyle(styleStr);
  
  // Get colors
  const fillColor = style.fillColor || '#ffffff';
  const strokeColor = style.strokeColor || '#000000';
  const fill = hexToRgb(fillColor);
  const stroke = hexToRgb(strokeColor);
  
  // Determine shape type
  const isVertex = cell.getAttribute('vertex') === '1';
  const isEdge = cell.getAttribute('edge') === '1';
  
  if (isVertex) {
    doc.save();
    
    // Draw shape based on style
    if (style.ellipse || style.shape === 'ellipse') {
      // Ellipse
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5)
         .ellipse(x + width/2, y + height/2, width/2, height/2)
         .fillAndStroke();
    } else if (style.rhombus) {
      // Diamond
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5)
         .polygon([x + width/2, y], [x + width, y + height/2], [x + width/2, y + height], [x, y + height/2])
         .fillAndStroke();
    } else if (style.shape === 'cylinder3' || style.shape === 'cylinder') {
      // Cylinder
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5);
      // Draw cylinder body
      doc.rect(x, y + 10, width, height - 20).fillAndStroke();
      // Draw top ellipse
      doc.ellipse(x + width/2, y + 10, width/2, 10).fillAndStroke();
      // Draw bottom ellipse
      doc.ellipse(x + width/2, y + height - 10, width/2, 10).stroke();
    } else if (style.shape === 'document') {
      // Document shape
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5);
      doc.moveTo(x, y)
         .lineTo(x + width, y)
         .lineTo(x + width, y + height - 10)
         .quadraticCurveTo(x + width * 0.75, y + height, x + width/2, y + height - 10)
         .quadraticCurveTo(x + width * 0.25, y + height - 20, x, y + height - 10)
         .lineTo(x, y)
         .fillAndStroke();
    } else if (style.shape === 'umlActor') {
      // Actor (stick figure)
      doc.strokeColor([stroke.r, stroke.g, stroke.b]).lineWidth(2);
      const cx = x + width/2;
      // Head
      doc.circle(cx, y + 10, 8).stroke();
      // Body
      doc.moveTo(cx, y + 18).lineTo(cx, y + 40).stroke();
      // Arms
      doc.moveTo(cx - 15, y + 28).lineTo(cx + 15, y + 28).stroke();
      // Legs
      doc.moveTo(cx, y + 40).lineTo(cx - 12, y + 55).stroke();
      doc.moveTo(cx, y + 40).lineTo(cx + 12, y + 55).stroke();
    } else if (style.rounded) {
      // Rounded rectangle
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5)
         .roundedRect(x, y, width, height, 8)
         .fillAndStroke();
    } else if (style.text) {
      // Text only (title)
      // Don't draw shape
    } else {
      // Default rectangle
      doc.fillColor([fill.r, fill.g, fill.b])
         .strokeColor([stroke.r, stroke.g, stroke.b])
         .lineWidth(1.5)
         .rect(x, y, width, height)
         .fillAndStroke();
    }
    
    // Draw text
    if (value && !style.shape?.includes('umlActor')) {
      const text = value.replace(/&#xa;/g, '\n').replace(/\\n/g, '\n');
      const fontSize = style.text ? 16 : 11;
      doc.fillColor([50, 50, 50])
         .fontSize(fontSize)
         .font('Helvetica-Bold')
         .text(text, x, y + height/2 - fontSize/2, {
           width: width,
           align: 'center',
           lineGap: 2
         });
    }
    
    // Label for actor
    if (style.shape === 'umlActor' && value) {
      doc.fillColor([50, 50, 50])
         .fontSize(11)
         .font('Helvetica-Bold')
         .text(value, x, y + height + 5, { width: width, align: 'center' });
    }
    
    doc.restore();
  }
}

// Draw edges/connectors
function drawEdges(doc, cells, scale = 0.85) {
  const cellMap = {};
  
  // Build cell map
  cells.forEach(cell => {
    const id = cell.getAttribute('id');
    const geometry = cell.getElementsByTagName('mxGeometry')[0];
    if (id && geometry) {
      cellMap[id] = {
        x: (parseFloat(geometry.getAttribute('x')) || 0) * scale + 50,
        y: (parseFloat(geometry.getAttribute('y')) || 0) * scale + 80,
        width: (parseFloat(geometry.getAttribute('width')) || 100) * scale,
        height: (parseFloat(geometry.getAttribute('height')) || 60) * scale
      };
    }
  });
  
  // Draw edges
  cells.forEach(cell => {
    if (cell.getAttribute('edge') !== '1') return;
    
    const sourceId = cell.getAttribute('source');
    const targetId = cell.getAttribute('target');
    const value = cell.getAttribute('value') || '';
    const styleStr = cell.getAttribute('style') || '';
    const style = parseStyle(styleStr);
    
    const source = cellMap[sourceId];
    const target = cellMap[targetId];
    
    if (!source || !target) return;
    
    // Calculate edge points
    const sx = source.x + source.width / 2;
    const sy = source.y + source.height / 2;
    const tx = target.x + target.width / 2;
    const ty = target.y + target.height / 2;
    
    // Adjust start/end to edge of shapes
    let startX = sx, startY = sy, endX = tx, endY = ty;
    
    if (Math.abs(tx - sx) > Math.abs(ty - sy)) {
      // Horizontal-ish
      startX = tx > sx ? source.x + source.width : source.x;
      endX = tx > sx ? target.x : target.x + target.width;
    } else {
      // Vertical-ish
      startY = ty > sy ? source.y + source.height : source.y;
      endY = ty > sy ? target.y : target.y + target.height;
    }
    
    const strokeColor = style.strokeColor || '#666666';
    const stroke = hexToRgb(strokeColor);
    
    doc.save();
    doc.strokeColor([stroke.r, stroke.g, stroke.b])
       .lineWidth(1.5);
    
    // Draw line
    doc.moveTo(startX, startY).lineTo(endX, endY).stroke();
    
    // Draw arrow
    const angle = Math.atan2(endY - startY, endX - startX);
    const arrowLength = 8;
    doc.moveTo(endX, endY)
       .lineTo(endX - arrowLength * Math.cos(angle - Math.PI/6), 
               endY - arrowLength * Math.sin(angle - Math.PI/6))
       .moveTo(endX, endY)
       .lineTo(endX - arrowLength * Math.cos(angle + Math.PI/6),
               endY - arrowLength * Math.sin(angle + Math.PI/6))
       .stroke();
    
    // Draw label
    if (value) {
      const midX = (startX + endX) / 2;
      const midY = (startY + endY) / 2;
      doc.fillColor([60, 60, 60])
         .fontSize(10)
         .font('Helvetica')
         .text(value, midX - 50, midY - 12, { width: 100, align: 'center' });
    }
    
    doc.restore();
  });
}

function convertDrawioToPdf(inputFile, outputFile) {
  const xmlContent = fs.readFileSync(inputFile, 'utf8');
  
  // Parse XML
  const parser = new DOMParser();
  const xmlDoc = parser.parseFromString(xmlContent, 'text/xml');
  
  // Get diagram title
  const diagramName = xmlDoc.getElementsByTagName('diagram')[0]?.getAttribute('name') || 'Diagram';
  
  // Get all cells
  const cells = Array.from(xmlDoc.getElementsByTagName('mxCell'));
  
  // Create PDF
  const doc = new PDFDocument({ size: 'A4', margin: 40 });
  const stream = fs.createWriteStream(outputFile);
  doc.pipe(stream);
  
  // Title
  doc.fontSize(18)
     .font('Helvetica-Bold')
     .fillColor([26, 54, 93])
     .text(diagramName, 40, 40, { align: 'center' });
  
  doc.moveDown();
  
  // Draw edges first (behind nodes)
  drawEdges(doc, cells);
  
  // Draw nodes
  cells.forEach(cell => {
    if (cell.getAttribute('vertex') === '1') {
      drawCell(doc, cell);
    }
  });
  
  // Footer
  doc.fontSize(8)
     .fillColor([150, 150, 150])
     .text(`Generated by Clawdbot - ${new Date().toLocaleDateString()}`, 40, 780, { align: 'center' });
  
  doc.end();
  
  return new Promise((resolve, reject) => {
    stream.on('finish', () => {
      console.log(`✅ PDF generated: ${outputFile}`);
      resolve(true);
    });
    stream.on('error', reject);
  });
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('Usage: node drawio_to_pdf_native.js <input.drawio> <output.pdf>');
    process.exit(1);
  }
  
  const inputFile = args[0];
  const outputFile = args[1];
  
  if (!fs.existsSync(inputFile)) {
    console.error(`Error: Input file not found: ${inputFile}`);
    process.exit(1);
  }
  
  convertDrawioToPdf(inputFile, outputFile)
    .then(() => process.exit(0))
    .catch(err => {
      console.error('Error:', err);
      process.exit(1);
    });
}

module.exports = { convertDrawioToPdf };
