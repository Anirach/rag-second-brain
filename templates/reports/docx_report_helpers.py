#!/usr/bin/env python3
"""Rebuild Paper Journey Report using the Brute Force template formatting."""

from docx import Document
from docx.shared import Pt, Inches, RGBColor, Cm, Emu
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsdecls
from docx.oxml import parse_xml
import copy

# Colors matching the template
NAVY = RGBColor(0x1B, 0x3A, 0x5C)
BLUE = RGBColor(0x2A, 0x64, 0x96)
DARK_GRAY = RGBColor(0x2C, 0x3E, 0x50)
GRAY = RGBColor(0x66, 0x66, 0x66)
ORANGE = RGBColor(0xE6, 0x7E, 0x22)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
TABLE_HEADER_BG = "1B3A5C"
TABLE_ALT_BG = "F8F9FA"
BLACK = RGBColor(0x33, 0x33, 0x33)

doc = Document()

# Set margins (1 inch = 914400 EMU)
for section in doc.sections:
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

def set_font(run, name='Arial', size=None, bold=None, color=None):
    run.font.name = name
    if size: run.font.size = size
    if bold is not None: run.font.bold = bold
    if color: run.font.color.rgb = color

def add_cover_line(text, size, bold=None, color=NAVY, spacing_after=Pt(0)):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = spacing_after
    p.paragraph_format.space_before = Pt(0)
    r = p.add_run(text)
    set_font(r, size=size, bold=bold, color=color)
    return p

def add_heading1(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(8)
    r = p.add_run(text)
    set_font(r, size=Pt(16), bold=True, color=NAVY)
    # Add bottom border
    pPr = p._element.get_or_add_pPr()
    pBdr = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="4" w:space="4" w:color="1B3A5C"/></w:pBdr>')
    pPr.append(pBdr)
    return p

def add_heading2(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(12)
    p.paragraph_format.space_after = Pt(6)
    r = p.add_run(text)
    set_font(r, size=Pt(13), bold=True, color=BLUE)
    return p

def add_heading3(text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(8)
    p.paragraph_format.space_after = Pt(4)
    r = p.add_run(text)
    set_font(r, size=Pt(12), bold=True, color=DARK_GRAY)
    return p

def add_body(text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if bold_prefix:
        r1 = p.add_run(bold_prefix)
        set_font(r1, size=Pt(11), bold=True, color=BLACK)
        r2 = p.add_run(text)
        set_font(r2, size=Pt(11), color=BLACK)
    else:
        r = p.add_run(text)
        set_font(r, size=Pt(11), color=BLACK)
    return p

def add_bullet(text, bold_prefix=None):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    if bold_prefix:
        r0 = p.add_run("• ")
        set_font(r0, size=Pt(11), color=BLACK)
        r1 = p.add_run(bold_prefix)
        set_font(r1, size=Pt(11), bold=True, color=BLACK)
        r2 = p.add_run(text)
        set_font(r2, size=Pt(11), color=BLACK)
    else:
        r = p.add_run("• " + text)
        set_font(r, size=Pt(11), color=BLACK)
    return p

def add_numbered(num, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.left_indent = Inches(0.5)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    r = p.add_run(f"{num}. {text}")
    set_font(r, size=Pt(11), color=BLACK)
    return p

def set_cell_shading(cell, color):
    shading = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{color}"/>')
    cell._tc.get_or_add_tcPr().append(shading)

def add_table(headers, rows):
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    for i, h in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(h)
        set_font(r, size=Pt(10), bold=True, color=WHITE)
        set_cell_shading(cell, TABLE_HEADER_BG)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Data rows
    for ri, row in enumerate(rows):
        for ci, val in enumerate(row):
            cell = table.rows[ri + 1].cells[ci]
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            set_font(r, size=Pt(10), color=BLACK)
            if ri % 2 == 1:
                set_cell_shading(cell, TABLE_ALT_BG)
    
    # Add borders
    tbl = table._tbl
    tblPr = tbl.tblPr if tbl.tblPr is not None else parse_xml(f'<w:tblPr {nsdecls("w")}/>')
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        '  <w:top w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '  <w:left w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '  <w:bottom w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '  <w:right w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '  <w:insideH w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '  <w:insideV w:val="single" w:sz="4" w:space="0" w:color="CCCCCC"/>'
        '</w:tblBorders>'
    )
    tblPr.append(borders)
    
    doc.add_paragraph()  # spacing
    return table

# ==================== COVER PAGE ====================
doc.add_paragraph()  # spacer
doc.add_paragraph()  # spacer
doc.add_paragraph()  # spacer

add_cover_line("The RAG Second Brain Paper Journey", Pt(24), bold=True, color=NAVY, spacing_after=Pt(8))
add_cover_line("From Concept to AIiH 2026 Submission", Pt(18), color=BLUE, spacing_after=Pt(8))
add_cover_line("Multi-Source Knowledge Graph RAG for Medical Diagnosis", Pt(15), color=GRAY, spacing_after=Pt(24))

# Divider line
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("━" * 40)
set_font(r, size=Pt(12), color=ORANGE)

add_cover_line("Prepared by: Academic Paper Team (Arthur AI Agent System)", Pt(11), color=GRAY, spacing_after=Pt(4))
add_cover_line("Date: February 17, 2026", Pt(11), color=GRAY, spacing_after=Pt(4))
add_cover_line("Version 1.0  |  Internal Report", Pt(10), color=GRAY)

# Page break
doc.add_page_break()

# ==================== TABLE OF CONTENTS ====================
add_heading1("Table of Contents")

toc_items = [
    "1. Executive Summary",
    "2. The Initial Idea (v1–v3)",
    "3. Early Iterations & Reviews (v3–v9)",
    "4. The Quality System",
    "5. Pivot to Real Experiments (v10–v16)",
    "6. The DDXPlus Breakthrough (v17)",
    "7. Venue Change: Information Fusion → AIiH 2026",
    "8. LNCS Conversion & Reviews (v18–v20.3)",
    "9. Conference Review & Response (v21–v22)",
    "10. The Final Push (v23–v23.1)",
    "11. MKG-RAG-MedicalDiag Application",
    "12. CMT Submission",
    "13. The Academic Team",
    "14. Key Lessons Learned",
    "15. Complete Version History",
    "16. Review Scorecard",
]
for item in toc_items:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.left_indent = Inches(0.5)
    r = p.add_run(item)
    set_font(r, size=Pt(11), color=BLUE)

doc.add_page_break()

# ==================== 1. EXECUTIVE SUMMARY ====================
add_heading1("1. Executive Summary")
add_body('This report documents the complete journey of the research paper "Multi-Source Knowledge Graph Retrieval-Augmented Generation as a Second Brain for LLM-Based Medical Diagnosis" — from initial concept to submission at AIiH 2026 (Imperial College London).')
add_body("Over approximately three weeks (late January to mid-February 2026), the paper evolved through 23 versions, underwent multiple rounds of internal AI-agent peer review (using 9 specialized agents), pivoted from synthetic to real experiments, changed target venues, and ultimately achieved unanimous ACCEPT from all three internal reviewers.")
add_body("Key metrics: 23 versions • 3 venue considerations • 2 datasets (DDXPlus, Symptom2Disease) • 9 academic team agents • 94.3% Top-1 accuracy • 13 final pages • Submitted February 17, 2026.")

# ==================== 2. THE INITIAL IDEA ====================
add_heading1("2. The Initial Idea (v1–v3)")
add_body('The paper began with a core insight: Large Language Models suffer from hallucination and knowledge gaps in medical domains. The proposed solution was a "Second Brain" — a multi-source knowledge graph combining ontological structure with data-driven co-occurrence patterns to ground LLM reasoning.')

add_heading2("Original Architecture")
add_body("The framework integrated three complementary retrieval layers:")
add_bullet("Dense vector retrieval over medical knowledge bases")
add_bullet("Sparse lexical retrieval enhanced with PPMI co-occurrence statistics")
add_bullet("Multi-source knowledge graph combining ontological structure with data-driven co-occurrence patterns")

add_heading2("Initial Reception (v3) — Score: 5.2/10")
add_body('Reviewers acknowledged the important problem and praised the "thoughtful ontology component" and "clean two-stage training decomposition." However, significant gaps existed in baselines, implementation details, and experimental rigor.')

# ==================== 3. EARLY ITERATIONS ====================
add_heading1("3. Early Iterations & Reviews (v3–v9)")

add_heading2("Version 6 — Score: 5.6/10")
add_body("Added causal LM loss definition (Equations 15–16), materialization depth ablation (+2.9 EM gain), and comparisons against RePlug (56.4 EM) and Atlas (64.5 EM). The paper showed improvement but still lacked convincing experimental validation.")

add_heading2("Version 8 — Estimated Score: 8.5/10")
add_body("Addressed all 9 reviewer concerns: clarified pipeline steps 1–8, documented multiplicative gating mechanism, added oracle labeling details (GPT-3.5, 96.2% agreement). However, the score was based on incomplete validation.")

add_heading2("Version 9 — The Crisis: Score Dropped to 4.4/10")
add_body("A devastating review exposed the core problem: BM25/entity-match/RRF were tested as proxies but the paper claimed to validate PPMI/OWL+KG/learned-gating. This was a fundamental disconnect between claims and evidence.")
p = add_body("")
r = p.runs[0]
r.text = "KEY DECISION: "
set_font(r, size=Pt(11), bold=True, color=ORANGE)
r2 = p.add_run("Rather than fabricate experimental results, the paper would be reframed as a conceptual proposal with honest proof-of-concept evaluation. This decision ultimately led to the paper's success.")
set_font(r2, size=Pt(11), color=BLACK)

# ==================== 4. THE QUALITY SYSTEM ====================
add_heading1("4. The Quality System")
add_body("After the v9 score drop, a comprehensive quality management system was established:")

add_heading2("QUALITY_LOCKS.md — 23 Quality Locks")
add_body("Five categories: Core Architecture (immutable), Methodology, Metrics & Evaluation, Compute & Reproducibility, Baselines & Comparisons. Once a reviewer praised an element, it was locked against regression.")

add_heading2("POSITIVE_FEEDBACK.md — Cumulative Praise Tracker")
add_body("Tracked reviewer praise across rounds, ensuring acknowledged strengths were never inadvertently weakened.")

add_heading2("Three-Agent Review Gate")
add_body("Every version underwent parallel review by: peer-reviewer (tough CS conference simulation), methodology-expert (statistical rigor), and technical-writer (clarity). No version shipped without unanimous approval.")

# ==================== 5. PIVOT TO REAL EXPERIMENTS ====================
add_heading1("5. Pivot to Real Experiments (v10–v16)")

add_heading2("v10: Honest Reframe")
add_body("Explicitly repositioned as a conceptual proposal — no fake results, clear acknowledgment of what was implemented vs. proposed.")

add_heading2("v12–v14: HotpotQA Experiments")
add_body("Real experiments added using HotpotQA. Fourteen recent related work papers from 2024–2026 incorporated (Beam Retrieval, DualRAG, EVO-RAG, FrugalRAG, PlanRAG, REALM, etc.).")

add_heading2("v15–v16: DDXPlus Medical Dataset")
add_body("Complete implementation pushed to GitHub. Critical pivot from general NLP benchmarks to medical diagnosis datasets — specifically DDXPlus (49 diseases, 1,000 test cases) and Symptom2Disease (24 diseases, 320 test cases).")

# ==================== 6. DDXPlus BREAKTHROUGH ====================
add_heading1("6. The DDXPlus Breakthrough (v17)")
add_body("Version 17 marked the turning point with compelling experimental results:")

add_table(
    ["Metric", "DDXPlus (49 diseases)", "Symptom2Disease (24 diseases)"],
    [
        ["Top-1 Accuracy", "94.3% (n=1,000)", "90.3% (n=320)"],
        ["Top-5 Accuracy", "98.8%", "98.8%"],
        ["Hallucination Rate", "≤0.3%", "0%"],
    ]
)

add_heading2("Honest Framing")
add_body("ML baselines (XGBoost 99.4%) outperform RAG on pure accuracy. Framed as accuracy vs. generalization trade-off: ML classifiers are fixed to training distribution, while RAG adapts to new diseases and knowledge without retraining.")

# ==================== 7. VENUE CHANGE ====================
add_heading1("7. Venue Change: Information Fusion → AIiH 2026")
add_body("Originally targeted Information Fusion (Elsevier). Pivoted to AIiH 2026 — 3rd International Conference on AI in Healthcare, Imperial College London, August 26–28, 2026.")
add_bullet("Better alignment: AIiH focuses on AI in healthcare")
add_bullet("Format: Springer LNCS proceedings (12+2 pages)")
add_bullet("Timeline: Deadline April 10, 2026 was achievable")
add_bullet("Required: llncs.cls, splncs04.bst, double-blind format")

# ==================== 8. LNCS CONVERSION ====================
add_heading1("8. LNCS Conversion & Reviews (v18–v20.3)")

add_heading2("v17→v18: Major → Minor Revision")
add_body("LNCS format conversion and structural revision improved the internal verdict from Major to Minor Revision.")

add_heading2("v18→v19: Minor → Accept-with-Minor")
add_body("Writing clarity, table formatting, and reference completeness refinements.")

add_heading2("v20–v20.3: Full LNCS Compliance — ACCEPT ×3")
add_body("Multiple sub-versions refined formatting. All three internal reviewers gave ACCEPT. Version 20.3 submitted to AIiH 2026 via CMT.")

# ==================== 9. CONFERENCE REVIEW ====================
add_heading1("9. Conference Review & Response (v21–v22)")

add_heading2("v21: All 9 Conference Concerns Addressed")
add_body("Systematically addressed every CMT reviewer concern. All three internal agents gave ACCEPT.")

add_heading2("v22: Expanded to 14 Pages — CMT Score: 5/10 (Major)")
add_body("Added TikZ architecture figure, expanded Related Work and Discussion. Received third CMT review with four specific concerns:")
add_numbered(1, "Data leakage: ontological edges may encode training patterns")
add_numbered(2, "Proof-of-concept scope: limited to DDXPlus synthetic data")
add_numbered(3, "Retrieval overlap: unclear unique contribution per modality")
add_numbered(4, '"Second Brain" framing needs stronger theoretical grounding')

# ==================== 10. FINAL PUSH ====================
add_heading1("10. The Final Push (v23–v23.1)")

add_heading2("v23: Addressing All 4 CMT Concerns")
add_body('1. Leakage-Free Variant: A "reduced-leakage" experiment excluded ontological edges derived from dataset generative rules. Result: 92.7% Top-1 — only 1.6pp drop, proving the framework works without leaked edges.')
add_body("2. Proof-of-Concept Reframing: Explicitly positioned as proof-of-concept, not production system. Added MIMIC-IV as next step.")
add_body("3. Overlap Analysis: New Table 5 showing each retrieval layer uniquely rescues cases missed by others — principled complementarity, not additive combination.")
add_body('4. "Second Brain" Cognition: Grounded in Clark & Chalmers\' Extended Mind thesis and Hutchins\' Distributed Cognition — transformed marketing into theoretical contribution.')

add_heading2("v23→v23.1: The 11 Final Fixes")

add_table(
    ["#", "Fix", "Source"],
    [
        ["1", '"Zero hallucinations" → "near-zero (≤0.3%)"', "Methodology"],
        ["2", "External baselines (supervised 75–92%)", "Peer Reviewer"],
        ["3", "Wilson confidence intervals", "Peer Reviewer"],
        ["4", "McNemar's statistical test", "Peer Reviewer"],
        ["5", '"Leakage-free" → "reduced-leakage"', "Methodology"],
        ["6", "GPT-4o reversal discussion", "Peer Reviewer"],
        ["7", "BM25 hallucination explanation", "Peer Reviewer"],
        ["8", "Macro F1 removed (class imbalance artifact)", "All"],
        ["9", "Single-run reproducibility caveat", "Methodology"],
        ["10", "Repetitive text consolidated", "Writer"],
        ["11", "Vocabulary caveat moved earlier", "Writer"],
    ]
)

add_heading2("Final Review: Unanimous ACCEPT")
add_body("Trimmed from 15→13 pages. All three reviewers accepted:")
add_bullet("Peer Reviewer: ACCEPT ✅ — All 8 concerns resolved")
add_bullet("Methodology Expert: ACCEPT ✅ — Statistical reporting appropriate")
add_bullet("Technical Writer: 4.5/5 ✅ — Minor bibliography issues only")

# ==================== 11. APPLICATION ====================
add_heading1("11. MKG-RAG-MedicalDiag Application")
add_body("A full-stack web application was built implementing the paper's algorithms, demonstrating the framework works as a real system:")
add_bullet("Repository: ", bold_prefix=None)
p = doc.paragraphs[-1]
p.runs[0].text = ""
r1 = p.add_run("Repository: ")
set_font(r1, size=Pt(11), bold=True, color=BLACK)
r2 = p.add_run("github.com/Anirach/MKG-RAG-MedicalDiag (private)")
set_font(r2, size=Pt(11), color=BLACK)

add_bullet("Features: Interactive symptom input, multi-source retrieval, differential diagnosis")
add_bullet("Provides tangible artifact beyond the paper")

# ==================== 12. CMT SUBMISSION ====================
add_heading1("12. CMT Submission")

add_table(
    ["Field", "Value"],
    [
        ["Platform", "Microsoft CMT (cmt3.research.microsoft.com)"],
        ["Paper ID", "18"],
        ["Track", "Full-length Paper (Main Sessions)"],
        ["File", "MKG-RAG-SecondBrain-v23.1.pdf"],
        ["Pages", "13 (within 12+2 limit)"],
        ["Format", "Springer LNCS (llncs.cls)"],
        ["Conference", "AIiH 2026 — Imperial College London"],
        ["Dates", "August 26–28, 2026"],
        ["Deadline", "April 10, 2026"],
        ["Submitted", "February 17, 2026"],
    ]
)

# ==================== 13. ACADEMIC TEAM ====================
add_heading1("13. The Academic Team")

add_table(
    ["Agent", "Role", "Model", "Key Contribution"],
    [
        ["Paper Architect", "Principal Investigator", "Opus", "All major revisions, strategy"],
        ["Literature Lead", "Systematic Review", "Opus", "14+ related work papers"],
        ["Methodology Expert", "Research Methods", "Opus", "Statistical rigor review"],
        ["Technical Writer", "Section Drafting", "Sonnet", "Writing quality (4.2→4.5)"],
        ["Peer Reviewer", "Internal Review", "Opus", "Tough reviewer simulation"],
        ["Format Editor", "Journal Compliance", "Sonnet", "LNCS formatting"],
        ["Data Analyst", "Statistics", "Opus", "Wilson CIs, McNemar's"],
        ["Ethics Reviewer", "Ethics", "Opus", "Dataset ethics, double-blind"],
        ["Journal Scout", "Venue Intelligence", "Opus", "AIiH 2026 selection"],
    ]
)

add_body('Critical Rule: After an early incident where a paper was delivered with only paper-architect review, a non-negotiable quality gate was established: "NEVER deliver a paper without full academic team review first."')

# ==================== 14. KEY LESSONS ====================
add_heading1("14. Key Lessons Learned")

lessons = [
    ("Honesty > Impressive Claims — ", "Conceptual papers can be publishable without fake results. The decision to be honest about proxy experiments led to real experiments."),
    ("Quality Locks Prevent Regression — ", "23 locks ensured improvements were cumulative. No reviewer-praised feature was ever accidentally removed."),
    ("Three Parallel Reviewers — ", "Each catches different issues: baselines, statistical gaps, and readability. One agent alone is insufficient."),
    ("ML Baselines Beating RAG Is Okay — ", "Frame as accuracy vs. generalization trade-off. Honesty builds credibility."),
    ("Medical Datasets > General Benchmarks — ", "DDXPlus made the healthcare contribution tangible vs. HotpotQA."),
    ("Verify Numbers Before Locking — ", "Wrong compute claims in v8 hurt more than vague ones."),
    ("Venue Selection Matters — ", "AIiH 2026 (focused) was better than Information Fusion (broad)."),
    ('"Second Brain" Grounding — ', "Marketing language became theoretical contribution when grounded in Extended Mind thesis."),
]
for bold, text in lessons:
    add_bullet(text, bold_prefix=bold)

# ==================== 15. VERSION HISTORY ====================
add_heading1("15. Complete Version History")

add_table(
    ["Version", "Date", "Key Changes", "Verdict"],
    [
        ["v1–v2", "Late Jan", "Initial concept, three-source architecture", "—"],
        ["v3", "Jan 29", "First experiments, ablations, HotpotQA", "5.2/10"],
        ["v6", "Feb 5", "Causal LM loss, materialization ablation", "5.6/10"],
        ["v8", "Feb 8", "All 9 concerns, FRAMES, GraphRAG", "~8.5/10"],
        ["v9", "Feb 9", "CRISIS: proxy vs real exposed", "4.4/10"],
        ["v10", "Feb 9", "Honest conceptual reframe", "Conceptual"],
        ["v12–14", "Feb 10", "14 related work, HotpotQA experiments", "Major Rev"],
        ["v15–16", "Feb 10", "DDXPlus dataset, full implementation", "Improving"],
        ["v17", "Feb 16", "DDXPlus: 94.3% Top-1, S2D: 90.3%", "Major Rev"],
        ["v18", "Feb 16", "Structural revision", "Minor Rev"],
        ["v19", "Feb 16", "Writing refinements", "Accept w/ Minor"],
        ["v20–20.3", "Feb 16", "LNCS conversion, formatting", "ACCEPT ×3"],
        ["v21", "Feb 16", "9 conference concerns addressed", "ACCEPT ×3"],
        ["v22", "Feb 16", "14pp expanded, TikZ figure", "CMT: 5/10"],
        ["v23", "Feb 16", "4 CMT concerns resolved", "ACCEPT ×3"],
        ["v23.1", "Feb 16", "11 fixes, trimmed to 13pp", "ACCEPT ×3 ✅"],
    ]
)

# ==================== 16. REVIEW SCORECARD ====================
add_heading1("16. Review Scorecard")
add_body("The paper's trajectory through review:")
add_body("v3 (5.2) → v6 (5.6) → v8 (~8.5) → v9 (4.4 crisis) → v10–16 (rebuilding) → v17 (Major) → v18 (Minor) → v19 (Accept-w-Minor) → v20.3 (ACCEPT ×3) → v21 (ACCEPT ×3) → v22 (CMT 5/10) → v23 (ACCEPT ×3) → v23.1 (ACCEPT ×3) ✅")

doc.add_paragraph()
add_body("Final Status: Submitted to CMT as Paper ID 18. Awaiting conference review. Deadline: April 10, 2026.")

doc.add_paragraph()
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run("— End of Report —")
set_font(r, size=Pt(11), color=GRAY)

# Add page numbers in footer
for section in doc.sections:
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    r = p.add_run("Page ")
    set_font(r, size=Pt(9), color=GRAY)
    
    fldChar1 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    r1 = p.add_run()
    r1._r.append(fldChar1)
    
    instrText = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> PAGE </w:instrText>')
    r2 = p.add_run()
    r2._r.append(instrText)
    
    fldChar2 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    r3 = p.add_run()
    r3._r.append(fldChar2)
    
    r4 = p.add_run(" of ")
    set_font(r4, size=Pt(9), color=GRAY)
    
    fldChar3 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="begin"/>')
    r5 = p.add_run()
    r5._r.append(fldChar3)
    
    instrText2 = parse_xml(f'<w:instrText {nsdecls("w")} xml:space="preserve"> NUMPAGES </w:instrText>')
    r6 = p.add_run()
    r6._r.append(instrText2)
    
    fldChar4 = parse_xml(f'<w:fldChar {nsdecls("w")} w:fldCharType="end"/>')
    r7 = p.add_run()
    r7._r.append(fldChar4)

outpath = "/home/clawdbot/clawd/tmp/RAG_SecondBrain_Paper_Journey_Report_v2.docx"
doc.save(outpath)
print(f"Saved to {outpath}")
