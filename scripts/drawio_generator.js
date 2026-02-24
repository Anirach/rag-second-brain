#!/usr/bin/env node
/**
 * Draw.io Diagram Generator
 * Generates .drawio XML files from structured input
 * 
 * Usage: node drawio_generator.js <output.drawio> '<json_config>'
 * 
 * Example config:
 * {
 *   "title": "My Diagram",
 *   "nodes": [
 *     {"id": "1", "label": "Start", "x": 100, "y": 100, "type": "rounded", "color": "blue"},
 *     {"id": "2", "label": "Process", "x": 300, "y": 100, "type": "rectangle", "color": "green"}
 *   ],
 *   "edges": [
 *     {"from": "1", "to": "2", "label": "next"}
 *   ]
 * }
 */

const fs = require('fs');

// Color presets
const colors = {
  blue: { fill: '#dae8fc', stroke: '#6c8ebf' },
  green: { fill: '#d5e8d4', stroke: '#82b366' },
  yellow: { fill: '#fff2cc', stroke: '#d6b656' },
  orange: { fill: '#ffe6cc', stroke: '#d79b00' },
  red: { fill: '#f8cecc', stroke: '#b85450' },
  purple: { fill: '#e1d5e7', stroke: '#9673a6' },
  gray: { fill: '#f5f5f5', stroke: '#666666' },
  white: { fill: '#ffffff', stroke: '#000000' }
};

// Shape styles
const shapes = {
  rounded: 'rounded=1;whiteSpace=wrap;html=1;',
  rectangle: 'whiteSpace=wrap;html=1;',
  diamond: 'rhombus;whiteSpace=wrap;html=1;',
  cylinder: 'shape=cylinder3;whiteSpace=wrap;html=1;boundedLbl=1;backgroundOutline=1;size=15;',
  ellipse: 'ellipse;whiteSpace=wrap;html=1;',
  cloud: 'ellipse;shape=cloud;whiteSpace=wrap;html=1;',
  hexagon: 'shape=hexagon;perimeter=hexagonPerimeter2;whiteSpace=wrap;html=1;',
  parallelogram: 'shape=parallelogram;perimeter=parallelogramPerimeter;whiteSpace=wrap;html=1;',
  document: 'shape=document;whiteSpace=wrap;html=1;boundedLbl=1;',
  actor: 'shape=umlActor;verticalLabelPosition=bottom;verticalAlign=top;html=1;'
};

function generateNode(node, cellId) {
  const color = colors[node.color] || colors.blue;
  const shape = shapes[node.type] || shapes.rounded;
  const width = node.width || 120;
  const height = node.height || 60;
  const style = `${shape}fillColor=${color.fill};strokeColor=${color.stroke};fontStyle=1;fontSize=12;`;
  
  return `        <mxCell id="${cellId}" value="${escapeXml(node.label)}" style="${style}" vertex="1" parent="1">
          <mxGeometry x="${node.x}" y="${node.y}" width="${width}" height="${height}" as="geometry" />
        </mxCell>`;
}

function generateEdge(edge, cellId, nodeIdMap) {
  const sourceId = nodeIdMap[edge.from];
  const targetId = nodeIdMap[edge.to];
  const label = edge.label || '';
  const color = colors[edge.color] || colors.gray;
  const style = `edgeStyle=orthogonalEdgeStyle;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeWidth=2;strokeColor=${color.stroke};`;
  
  return `        <mxCell id="${cellId}" value="${escapeXml(label)}" style="${style}" edge="1" parent="1" source="${sourceId}" target="${targetId}">
          <mxGeometry relative="1" as="geometry" />
        </mxCell>`;
}

function escapeXml(str) {
  return str.replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&apos;')
            .replace(/\n/g, '&#xa;');
}

function generateDiagram(config) {
  const title = config.title || 'Diagram';
  const nodes = config.nodes || [];
  const edges = config.edges || [];
  
  let cellId = 2;
  const nodeIdMap = {};
  
  // Generate node cells
  const nodeCells = nodes.map(node => {
    nodeIdMap[node.id] = cellId;
    return generateNode(node, cellId++);
  }).join('\n');
  
  // Generate edge cells
  const edgeCells = edges.map(edge => {
    return generateEdge(edge, cellId++, nodeIdMap);
  }).join('\n');
  
  // Title cell
  const titleCell = title ? `        <mxCell id="${cellId}" value="${escapeXml(title)}" style="text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;whiteSpace=wrap;rounded=0;fontSize=18;fontStyle=1;" vertex="1" parent="1">
          <mxGeometry x="200" y="20" width="400" height="40" as="geometry" />
        </mxCell>` : '';

  return `<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="Clawdbot">
  <diagram name="Page-1" id="generated-diagram">
    <mxGraphModel dx="1000" dy="800" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="850" pageHeight="1100" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
${nodeCells}
${edgeCells}
${titleCell}
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>`;
}

// Flowchart generator helper
function generateFlowchart(steps, title = 'Flowchart') {
  const nodes = [];
  const edges = [];
  let y = 100;
  
  steps.forEach((step, i) => {
    let type = 'rounded';
    let color = 'blue';
    
    if (i === 0) { type = 'ellipse'; color = 'green'; }
    else if (i === steps.length - 1) { type = 'ellipse'; color = 'red'; }
    else if (step.toLowerCase().includes('decision') || step.includes('?')) { type = 'diamond'; color = 'yellow'; }
    
    nodes.push({
      id: String(i + 1),
      label: step,
      x: 300,
      y: y,
      type: type,
      color: color,
      width: 150,
      height: type === 'diamond' ? 80 : 60
    });
    
    if (i > 0) {
      edges.push({ from: String(i), to: String(i + 1) });
    }
    
    y += type === 'diamond' ? 120 : 100;
  });
  
  return { title, nodes, edges };
}

// Architecture diagram helper
function generateArchitecture(components, title = 'Architecture') {
  const nodes = [];
  const edges = [];
  const layers = {};
  
  // Group by layer
  components.forEach(comp => {
    const layer = comp.layer || 'default';
    if (!layers[layer]) layers[layer] = [];
    layers[layer].push(comp);
  });
  
  // Position nodes by layer
  let y = 100;
  const layerNames = Object.keys(layers);
  
  layerNames.forEach(layerName => {
    const layerComps = layers[layerName];
    let x = 100;
    
    layerComps.forEach((comp, i) => {
      nodes.push({
        id: comp.id,
        label: comp.label,
        x: x,
        y: y,
        type: comp.type || 'rounded',
        color: comp.color || 'blue',
        width: comp.width || 120,
        height: comp.height || 60
      });
      x += 160;
    });
    y += 120;
  });
  
  // Add connections
  components.forEach(comp => {
    if (comp.connects) {
      comp.connects.forEach(target => {
        edges.push({ from: comp.id, to: target, label: comp.connectLabel || '' });
      });
    }
  });
  
  return { title, nodes, edges };
}

// CLI
if (require.main === module) {
  const args = process.argv.slice(2);
  
  if (args.length < 2) {
    console.log('Usage: node drawio_generator.js <output.drawio> \'<json_config>\'');
    console.log('');
    console.log('Or use helpers:');
    console.log('  node drawio_generator.js <output.drawio> --flowchart "Step1" "Step2" "Step3"');
    console.log('');
    console.log('Node types: rounded, rectangle, diamond, cylinder, ellipse, cloud, hexagon, parallelogram, document, actor');
    console.log('Colors: blue, green, yellow, orange, red, purple, gray, white');
    process.exit(1);
  }
  
  const outputFile = args[0];
  let config;
  
  if (args[1] === '--flowchart') {
    const steps = args.slice(2);
    config = generateFlowchart(steps, 'Flowchart');
  } else {
    try {
      config = JSON.parse(args[1]);
    } catch (e) {
      console.error('Invalid JSON config:', e.message);
      process.exit(1);
    }
  }
  
  const xml = generateDiagram(config);
  fs.writeFileSync(outputFile, xml);
  console.log(`✅ Diagram generated: ${outputFile}`);
}

module.exports = { generateDiagram, generateFlowchart, generateArchitecture, colors, shapes };
