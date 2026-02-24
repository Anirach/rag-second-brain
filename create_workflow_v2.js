#!/usr/bin/env node

const fs = require('fs');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  HeadingLevel, AlignmentType, BorderStyle, WidthType, ShadingType, PageBreak, PageNumber, Footer
} = require('docx');

// Custom color scheme matching requirements
const NAVY = "1B3A5C";
const BLUE = "2A6496";
const DARK_GRAY = "2C3E50";
const GRAY = "666666";
const ORANGE = "E67E22";
const BLACK = "333333";
const WHITE = "FFFFFF";
const TABLE_ALT_BG = "F8F9FA";

// Helper functions
function createBorders(color = "CCCCCC", size = 4) {
  const border = { style: BorderStyle.SINGLE, size, color };
  return { top: border, bottom: border, left: border, right: border };
}

function coverLine(text, size, bold = false, color = NAVY, spacing = { after: 0 }) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing,
    children: [new TextRun({ text, bold, font: "Arial", size, color })]
  });
}

function heading1(text) {
  return new Paragraph({
    spacing: { before: 360, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: NAVY } },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 32, color: NAVY })]
  });
}

function heading2(text) {
  return new Paragraph({
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 26, color: BLUE })]
  });
}

function heading3(text) {
  return new Paragraph({
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, bold: true, font: "Arial", size: 24, color: DARK_GRAY })]
  });
}

function bodyText(text, spacing = { after: 120 }) {
  return new Paragraph({
    spacing,
    alignment: AlignmentType.JUSTIFIED,
    children: [new TextRun({ text, font: "Arial", size: 22, color: BLACK })]
  });
}

function bulletPoint(text, spacing = { after: 60 }) {
  return new Paragraph({
    spacing,
    indent: { left: 1000, hanging: 500 },
    children: [new TextRun({ text: "• " + text, font: "Arial", size: 22, color: BLACK })]
  });
}

function numberedPoint(num, text, spacing = { after: 60 }) {
  return new Paragraph({
    spacing,
    indent: { left: 1000, hanging: 500 },
    children: [new TextRun({ text: `${num}. ${text}`, font: "Arial", size: 22, color: BLACK })]
  });
}

function createHeaderCell(text, width) {
  return new TableCell({
    borders: createBorders(),
    width: { size: width, type: WidthType.DXA },
    shading: { fill: NAVY, type: ShadingType.CLEAR },
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, font: "Arial", size: 20, color: WHITE })]
    })]
  });
}

function createDataCell(text, width, isAlt = false) {
  return new TableCell({
    borders: createBorders(),
    width: { size: width, type: WidthType.DXA },
    shading: isAlt ? { fill: TABLE_ALT_BG, type: ShadingType.CLEAR } : undefined,
    margins: { top: 60, bottom: 60, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text, font: "Arial", size: 20, color: BLACK })]
    })]
  });
}

function createTable(headers, rows, columnWidths) {
  return new Table({
    width: { size: 100, type: WidthType.PERCENTAGE },
    rows: [
      new TableRow({
        children: headers.map((h, i) => createHeaderCell(h, columnWidths[i]))
      }),
      ...rows.map((row, rowIndex) => new TableRow({
        children: row.map((cell, i) => createDataCell(cell, columnWidths[i], rowIndex % 2 === 1))
      }))
    ]
  });
}

// Create footer with page numbers
function createFooter() {
  return new Footer({
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [
        new TextRun({ text: "Page ", font: "Arial", size: 18, color: GRAY }),
        new TextRun({ children: [PageNumber.CURRENT], font: "Arial", size: 18, color: GRAY }),
        new TextRun({ text: " of ", font: "Arial", size: 18, color: GRAY }),
        new TextRun({ children: [PageNumber.TOTAL_PAGES], font: "Arial", size: 18, color: GRAY })
      ]
    })]
  });
}

// Document content
const sections = [];

// Cover page
sections.push(
  new Paragraph({}), // spacer
  new Paragraph({}), // spacer
  new Paragraph({}), // spacer
  coverLine("Academic Paper Production Workflow v2", 48, true, NAVY, { after: 160 }),
  coverLine("From Idea to Submission in 7 Days", 36, false, BLUE, { after: 160 }),
  coverLine("Lessons Learned from 23 Versions of MKG-RAG Second Brain Paper", 30, false, GRAY, { after: 480 }),
  
  // Divider
  new Paragraph({
    alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━", font: "Arial", size: 24, color: ORANGE })]
  }),
  
  coverLine("Prepared by: Arthur AI Agent System", 22, false, GRAY, { after: 80 }),
  coverLine("Date: February 17, 2026", 22, false, GRAY, { after: 80 }),
  coverLine("Version 2.0  |  Internal Guide", 20, false, GRAY),
  
  new PageBreak()
);

// Content sections
sections.push(
  // 1. Executive Summary
  heading1("1. Executive Summary"),
  bodyText("This workflow replaces the ad-hoc process that produced 23 versions over 3 weeks. Based on lessons from the MKG-RAG Second Brain paper journey, it targets 3 drafts in 7 days with higher novelty and fewer iterations."),
  bodyText("Key improvements: experiments-first approach, upfront novelty validation, venue-first planning, batched reviews, and quality gates from day 1."),

  // 2. Problems with the Old Workflow
  heading1("2. Problems with the Old Workflow"),

  heading2("2.1 Experiments Came Too Late"),
  bodyText("v1-v9 built on proxy/synthetic claims. Real experiments (DDXPlus) didn't arrive until v15-v17. Papers without real results get destroyed in review."),

  heading2("2.2 No Novelty Validation Upfront"),
  bodyText('"Second Brain" was marketing until v23 when grounded in Extended Mind theory. Reviewers see through ungrounded framing immediately.'),

  heading2("2.3 Quality System Was Reactive"),
  bodyText("QUALITY_LOCKS and 3-agent review gate created AFTER the v9 crisis (score dropped from 8.5 to 4.4). Should have been there from the start."),

  heading2("2.4 Too Many Small Iterations"),
  bodyText("23 versions, most were incremental patches. Each review-fix cycle costs time and introduces regression risk."),

  heading2("2.5 Venue Selected Late"),
  bodyText("Started targeting Information Fusion, pivoted to AIiH. Format conversion (IEEE to LNCS) wasted multiple versions."),

  heading2("2.6 Theoretical Grounding Added Last"),
  bodyText("Clark & Chalmers Extended Mind, Hutchins Distributed Cognition added in final versions. Should anchor novelty claims from the beginning."),

  // 3. The New Workflow
  heading1("3. The New Workflow: 3 Phases, 7 Days"),

  heading2("3.1 Phase 1: Foundation (Day 1-2)"),
  bodyText("Tasks:"),
  bulletPoint("Pick target venue (format, page limit, scope, deadline)"),
  bulletPoint("Literature gap analysis: define novel contribution in ONE sentence"),
  bulletPoint('Run "Does this exist?" check (Semantic Scholar, Google Scholar, arXiv)'),
  bulletPoint("Build baseline experiments with real data"),
  bulletPoint("Design ablation-ready architecture (each component removable)"),
  bulletPoint("Create comparison table: what existing methods do vs what yours adds"),
  bulletPoint("Anchor method in existing theory (not just a name — cite cognitive science, information theory, etc.)"),

  bodyText("Deliverables table:"),
  createTable(
    ["Deliverable", "Description", "Owner"],
    [
      ["Venue Brief", "Target venue, format, page limit, scope", "Journal Scout"],
      ["Novelty Statement", "One sentence, grounded in theory", "Paper Architect"],
      ["Literature Gap Table", "Existing methods vs yours", "Literature Lead"],
      ["Baseline Results", "Real experiments, real data", "Data Analyst"],
      ["Ablation Design", "Component removal plan", "Methodology Expert"]
    ],
    [2500, 4000, 2500]
  ),

  heading2("3.2 Phase 2: Full Draft (Day 3-5)"),
  bodyText("Tasks:"),
  bulletPoint("Write complete paper with real results (not placeholders)"),
  bulletPoint("All sections: intro, related work, method, experiments, discussion, conclusion"),
  bulletPoint("Include all tables, figures, equations"),
  bulletPoint("Apply venue format from the start (e.g., llncs.cls, IEEE template)"),
  bulletPoint("3-agent parallel review (peer-reviewer, methodology-expert, technical-writer)"),
  bulletPoint("Batch ALL fixes into single revision (Draft 2)"),

  bodyText("Review gate table:"),
  createTable(
    ["Reviewer", "Focus", "Pass Criteria"],
    [
      ["Peer Reviewer", "Baselines, novelty, claims vs evidence", "No unsupported claims"],
      ["Methodology Expert", "Statistical rigor, reproducibility", "Proper CIs, tests, caveats"],
      ["Technical Writer", "Clarity, flow, readability", "Score ≥ 4.0/5.0"]
    ],
    [2500, 3500, 3000]
  ),

  bodyText('Rule: ALL THREE must pass. No exceptions. No "ship it with one approval."'),

  heading2("3.3 Phase 3: Polish & Submit (Day 6-7)"),
  bodyText("Tasks:"),
  bulletPoint("Apply all Draft 2 review fixes"),
  bulletPoint("Final 3-agent review (should be minor issues only)"),
  bulletPoint("Format compliance check (page count, references, anonymization)"),
  bulletPoint("Generate supplementary materials if needed"),
  bulletPoint("Submit"),

  // 4. Quality Gates
  heading1("4. Quality Gates (Active from Day 1)"),

  heading3("4.1 QUALITY_LOCKS"),
  bodyText("Lock any element praised by reviewers. Never regress."),

  heading3("4.2 POSITIVE_FEEDBACK tracker"),
  bodyText("Cumulative praise log across review rounds."),

  heading3("4.3 Three-Agent Review Gate"),
  bodyText("Every draft reviewed by 3 agents in parallel. Unanimous approval required."),

  heading3("4.4 Novelty Checkpoint"),
  bodyText("Before Phase 2, verify the one-sentence novelty claim still holds after literature review."),

  heading3("4.5 Claims-Evidence Alignment"),
  bodyText("Every claim in the paper must map to a specific experiment, table, or citation. No orphan claims."),

  // 5. Novelty Boosters
  heading1("5. Novelty Boosters"),

  heading3("5.1 Ablation-First Design"),
  bodyText('Build experiments so each component can be independently removed. Creates "unique contribution per component" — reviewers always ask for this.'),

  heading3("5.2 Theoretical Grounding Early"),
  bodyText("Don't just name your method — anchor it in established theory from draft 1. Examples: Extended Mind (Clark & Chalmers), Distributed Cognition (Hutchins), Information Foraging Theory."),

  heading3("5.3 Comparison Table in Outline"),
  bodyText("Before writing, create a table of existing methods and what yours uniquely adds. If no clear gap, rethink the contribution."),

  heading3("5.4 Real-World Framing"),
  bodyText('Position contribution as solving a real problem, not just improving metrics. "Why does this matter to a doctor/engineer/user?"'),

  heading3("5.5 Honest Limitations"),
  bodyText("Proactively state what your method does NOT do. Builds credibility and prevents reviewer attacks."),

  // 6. Agent Team Roles
  heading1("6. Agent Team Roles"),

  bodyText("Table of all 9 agents:"),
  createTable(
    ["Agent", "Phase", "Role", "Model"],
    [
      ["Paper Architect", "All", "Principal investigator, strategy", "Opus"],
      ["Literature Lead", "Phase 1", "Gap analysis, related work", "Opus"],
      ["Journal Scout", "Phase 1", "Venue selection, format requirements", "Opus"],
      ["Data Analyst", "Phase 1-2", "Experiments, statistics, ablations", "Opus"],
      ["Methodology Expert", "Phase 2-3", "Statistical rigor review", "Opus"],
      ["Technical Writer", "Phase 2-3", "Writing quality, clarity", "Sonnet"],
      ["Peer Reviewer", "Phase 2-3", "Tough conference reviewer simulation", "Opus"],
      ["Ethics Reviewer", "Phase 2", "Dataset ethics, bias, double-blind", "Opus"],
      ["Format Editor", "Phase 3", "Venue format compliance", "Sonnet"]
    ],
    [2200, 1800, 3500, 1500]
  ),

  // 7. Checklist
  heading1("7. Checklist: Before Submitting"),

  bodyText("Numbered checklist:"),
  numberedPoint(1, "Novelty claim is one sentence, grounded in theory"),
  numberedPoint(2, "All experiments use real data (no proxies)"),
  numberedPoint(3, "Ablation study shows each component's unique contribution"),
  numberedPoint(4, "Statistical tests applied (Wilson CIs, McNemar's, etc.)"),
  numberedPoint(5, "External baselines included (not just your own variants)"),
  numberedPoint(6, "All 3 reviewers gave ACCEPT"),
  numberedPoint(7, "Page count within venue limit"),
  numberedPoint(8, "Double-blind compliance (no author names, no self-citations revealing identity)"),
  numberedPoint(9, "References complete and correctly formatted"),
  numberedPoint(10, "Supplementary materials prepared (code repo, appendix)"),

  // 8. Timeline Comparison
  heading1("8. Timeline Comparison"),

  bodyText("Table:"),
  createTable(
    ["Aspect", "Old (MKG-RAG)", "New (v2 Workflow)"],
    [
      ["Duration", "~21 days", "7 days target"],
      ["Versions", "23", "3 drafts"],
      ["Real experiments", "Added at v15", "From Day 1"],
      ["Novelty validation", "v23 (last minute)", "Day 1"],
      ["Venue format", "Converted mid-way", "Applied from start"],
      ["Quality gates", "Added at v9 crisis", "Active from Day 1"],
      ["Review rounds", "10+", "2 (Phase 2 + Phase 3)"]
    ],
    [3000, 3000, 3000]
  ),

  // 9. Lessons
  heading1("9. Lessons from the MKG-RAG Journey"),

  bodyText("Bullet points:"),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Honesty > Impressive Claims: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Conceptual papers can be publishable without fake results", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Quality Locks Prevent Regression: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "23 locks ensured improvements were cumulative", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Three Parallel Reviewers catch different issues: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "baselines, statistics, readability", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "ML Baselines Beating RAG Is Okay: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Frame as accuracy vs generalization trade-off", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Medical Datasets > General Benchmarks: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Domain-specific data makes contribution tangible", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Verify Numbers Before Locking: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Wrong numbers hurt more than vague ones", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Venue Selection Matters: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Focused venue (AIiH) better than broad (Information Fusion)", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: '"Second Brain" Grounding: ', font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "Marketing language became theoretical contribution when grounded in Extended Mind thesis", font: "Arial", size: 22, color: BLACK })
    ]
  }),
  new Paragraph({
    spacing: { after: 60 },
    indent: { left: 1000, hanging: 500 },
    children: [
      new TextRun({ text: "• ", font: "Arial", size: 22, color: BLACK }),
      new TextRun({ text: "Never deliver without full team review: ", font: "Arial", size: 22, color: BLACK, bold: true }),
      new TextRun({ text: "One-agent papers get rejected", font: "Arial", size: 22, color: BLACK })
    ]
  })
);

// Create the document
const doc = new Document({
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 } // 1 inch margins
      }
    },
    footers: {
      default: createFooter()
    },
    children: sections
  }]
});

// Save the document
async function generateDoc() {
  try {
    const buffer = await Packer.toBuffer(doc);
    const filename = 'Academic_Paper_Production_Workflow_v2.docx';
    fs.writeFileSync(filename, buffer);
    console.log(`Document saved as ${filename}`);
    return filename;
  } catch (error) {
    console.error('Error generating document:', error);
    throw error;
  }
}

generateDoc().catch(console.error);