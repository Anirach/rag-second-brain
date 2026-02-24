#!/usr/bin/env python3
"""
Create a professional PowerPoint presentation for the NCD-CIE research paper.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml

# Create presentation with widescreen dimensions
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)

# Color scheme
DARK_BLUE = RGBColor(0, 51, 102)  # #003366
MEDIUM_BLUE = RGBColor(0, 102, 153)  # #006699
LIGHT_BLUE = RGBColor(0, 153, 204)  # #0099CC
WHITE = RGBColor(255, 255, 255)
GRAY = RGBColor(100, 100, 100)
LIGHT_GRAY = RGBColor(240, 240, 240)

def add_slide_number(slide, num):
    """Add slide number to bottom right"""
    txBox = slide.shapes.add_textbox(Inches(12.5), Inches(7.0), Inches(0.5), Inches(0.3))
    tf = txBox.text_frame
    p = tf.paragraphs[0]
    p.text = str(num)
    p.font.size = Pt(10)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.RIGHT

def add_header_bar(slide):
    """Add a dark blue header bar"""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.8))
    shape.fill.solid()
    shape.fill.fore_color.rgb = DARK_BLUE
    shape.line.fill.background()

def add_footer_line(slide):
    """Add a subtle footer line"""
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.5), Inches(6.9), Inches(12.333), Inches(0.02))
    shape.fill.solid()
    shape.fill.fore_color.rgb = LIGHT_BLUE
    shape.line.fill.background()

def set_title_style(title_shape, size=32, color=DARK_BLUE):
    """Style a title"""
    title_shape.text_frame.paragraphs[0].font.size = Pt(size)
    title_shape.text_frame.paragraphs[0].font.bold = True
    title_shape.text_frame.paragraphs[0].font.color.rgb = color

def add_content_slide(prs, title_text, bullets, slide_num, emoji=""):
    """Create a standard content slide with bullets"""
    blank_layout = prs.slide_layouts[6]  # Blank layout
    slide = prs.slides.add_slide(blank_layout)
    
    add_header_bar(slide)
    add_footer_line(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = f"{emoji} {title_text}" if emoji else title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Content area
    content_box = slide.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(12), Inches(5.5))
    tf = content_box.text_frame
    tf.word_wrap = True
    
    for i, bullet in enumerate(bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(20)
        p.font.color.rgb = DARK_BLUE
        p.space_after = Pt(12)
        p.level = 0
    
    add_slide_number(slide, slide_num)
    return slide

def add_two_column_slide(prs, title_text, left_title, left_bullets, right_title, right_bullets, slide_num, emoji=""):
    """Create a two-column content slide"""
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    
    add_header_bar(slide)
    add_footer_line(slide)
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
    tf = title_box.text_frame
    p = tf.paragraphs[0]
    p.text = f"{emoji} {title_text}" if emoji else title_text
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Left column title
    left_title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.1), Inches(5.8), Inches(0.5))
    tf = left_title_box.text_frame
    p = tf.paragraphs[0]
    p.text = left_title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = MEDIUM_BLUE
    
    # Left column content
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.6), Inches(5.8), Inches(5))
    tf = left_box.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(left_bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(18)
        p.font.color.rgb = DARK_BLUE
        p.space_after = Pt(8)
    
    # Right column title
    right_title_box = slide.shapes.add_textbox(Inches(6.8), Inches(1.1), Inches(5.8), Inches(0.5))
    tf = right_title_box.text_frame
    p = tf.paragraphs[0]
    p.text = right_title
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = MEDIUM_BLUE
    
    # Right column content
    right_box = slide.shapes.add_textbox(Inches(6.8), Inches(1.6), Inches(5.8), Inches(5))
    tf = right_box.text_frame
    tf.word_wrap = True
    for i, bullet in enumerate(right_bullets):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = f"• {bullet}"
        p.font.size = Pt(18)
        p.font.color.rgb = DARK_BLUE
        p.space_after = Pt(8)
    
    add_slide_number(slide, slide_num)
    return slide

# ==================== SLIDE 1: Title Slide ====================
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)

# Full blue background
bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg_shape.fill.solid()
bg_shape.fill.fore_color.rgb = DARK_BLUE
bg_shape.line.fill.background()

# Decorative accent bar
accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(2.8), Inches(13.333), Inches(0.1))
accent.fill.solid()
accent.fill.fore_color.rgb = LIGHT_BLUE
accent.line.fill.background()

# Main title
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.0), Inches(12.333), Inches(1.5))
tf = title_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "A Hybrid Symbolic-Statistical Framework for Integrated Non-Communicable Disease Risk Assessment and Prevention"
p.font.size = Pt(32)
p.font.bold = True
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER

# Subtitle
subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.2), Inches(12.333), Inches(0.8))
tf = subtitle_box.text_frame
p = tf.paragraphs[0]
p.text = "The NCD Causal Insight Engine (NCD-CIE)"
p.font.size = Pt(28)
p.font.color.rgb = LIGHT_BLUE
p.alignment = PP_ALIGN.CENTER

# Authors
author_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(12.333), Inches(0.6))
tf = author_box.text_frame
p = tf.paragraphs[0]
p.text = "Anirach Mingkhwan and Kongkiat Kespechara"
p.font.size = Pt(24)
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER

# Affiliation
affil_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(12.333), Inches(0.5))
tf = affil_box.text_frame
p = tf.paragraphs[0]
p.text = "King Mongkut's University of Technology North Bangkok"
p.font.size = Pt(18)
p.font.color.rgb = RGBColor(180, 200, 220)
p.alignment = PP_ALIGN.CENTER

# ==================== SLIDE 2: Outline ====================
slide_num = 2
add_content_slide(prs, "Presentation Outline", [
    "Background & Problem Statement",
    "Research Gaps and Objectives",
    "Methodology: Hybrid Symbolic-Statistical Architecture",
    "System Components: Knowledge Base, DAG Modeling, Risk Scoring",
    "Case Study: Metabolic Syndrome Patient",
    "Results & Validation",
    "Limitations and Future Directions",
    "Conclusions"
], slide_num, "📋")

# ==================== SLIDE 3: Background & Problem ====================
slide_num = 3
add_content_slide(prs, "Background: The NCD Crisis", [
    "Non-Communicable Diseases (NCDs) cause 75% of global deaths",
    "Major NCDs: Cardiovascular disease, Diabetes, Cancer, Chronic respiratory diseases",
    "NCDs share common modifiable risk factors (diet, activity, smoking, alcohol)",
    "Current risk assessment tools have significant limitations:",
    "   → Siloed single-disease focus (e.g., Framingham for CVD only)",
    "   → Missing interconnections between conditions",
    "   → Limited actionability for personalized interventions"
], slide_num, "🌍")

# ==================== SLIDE 4: Gap Statement ====================
slide_num = 4
add_two_column_slide(prs, "Research Gap Analysis", 
    "WHY Gap (Motivation)",
    [
        "Existing tools treat NCDs in isolation",
        "Patients often have multiple risk factors",
        "Co-morbidities multiply risk synergistically",
        "No unified framework exists",
        "Clinical decisions lack causal insight"
    ],
    "HOW Gap (Technical)",
    [
        "Statistical models lack explainability",
        "Clinical guidelines lack personalization",
        "No mechanism for intervention simulation",
        "Evidence grading is inconsistent",
        "Causal relationships not modeled explicitly"
    ],
    slide_num, "🔍")

# ==================== SLIDE 5: Objectives ====================
slide_num = 5
add_content_slide(prs, "Research Objectives", [
    "Develop NCD-CIE: A unified framework for multi-disease risk assessment",
    "Integrate symbolic reasoning (knowledge graphs) with statistical methods",
    "Enable causal inference through DAG-based biomarker modeling",
    "Provide personalized, actionable intervention recommendations",
    "Implement what-if simulation for intervention planning",
    "Create transparent, explainable risk predictions",
    "Support clinical decision-making with evidence-graded insights"
], slide_num, "🎯")

# ==================== SLIDE 6: Methodology Overview ====================
slide_num = 6
add_two_column_slide(prs, "Methodology: Hybrid Architecture",
    "Symbolic Component",
    [
        "Knowledge Graph (KG) representation",
        "70+ biomarkers as nodes",
        "100+ causal edges with evidence grades",
        "Rule-based pattern detection",
        "Explainable reasoning chains"
    ],
    "Statistical Component",
    [
        "Reference range normalization",
        "Weighted risk score aggregation",
        "Trend analysis with deviation metrics",
        "Correlation pattern detection",
        "Predictive cascade modeling"
    ],
    slide_num, "⚙️")

# ==================== SLIDE 7: System Architecture ====================
slide_num = 7
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
add_header_bar(slide)
add_footer_line(slide)

# Title
title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "🏗️ System Architecture"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE

# Architecture boxes
components = [
    ("Data Input Layer", "Patient biomarkers\nHealth records\nLifestyle data", 0.5, 1.3),
    ("Knowledge Graph", "70+ biomarkers\n8 health domains\n100+ causal edges", 3.8, 1.3),
    ("Causal DAG Engine", "Evidence grading\nRelationship modeling\nCascade analysis", 7.1, 1.3),
    ("Risk Calculator", "Weighted scoring\nPattern detection\nTrend analysis", 10.4, 1.3),
    ("What-If Simulator", "Intervention modeling\nOutcome prediction\nCascade effects", 3.8, 4.3),
    ("Output: Reports", "Risk scores\nRecommendations\nExplanations", 7.1, 4.3),
]

for name, desc, x, y in components:
    # Box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(2.8), Inches(1.8))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = MEDIUM_BLUE
    box.line.width = Pt(2)
    
    # Box title
    title_tb = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.1), Inches(2.6), Inches(0.4))
    tf = title_tb.text_frame
    p = tf.paragraphs[0]
    p.text = name
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Box content
    content_tb = slide.shapes.add_textbox(Inches(x + 0.1), Inches(y + 0.5), Inches(2.6), Inches(1.2))
    tf = content_tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = desc
    p.font.size = Pt(11)
    p.font.color.rgb = GRAY
    p.alignment = PP_ALIGN.CENTER

# Arrows (using shapes)
arrow_positions = [
    (Inches(3.3), Inches(2.2), Inches(0.5), Inches(0.1)),
    (Inches(6.6), Inches(2.2), Inches(0.5), Inches(0.1)),
    (Inches(9.9), Inches(2.2), Inches(0.5), Inches(0.1)),
    (Inches(5.2), Inches(3.1), Inches(0.1), Inches(1.2)),
    (Inches(6.6), Inches(5.2), Inches(0.5), Inches(0.1)),
]

for x, y, w, h in arrow_positions:
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW if w > h else MSO_SHAPE.DOWN_ARROW, x, y, w, h)
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = MEDIUM_BLUE
    arrow.line.fill.background()

add_slide_number(slide, slide_num)

# ==================== SLIDE 8: Knowledge Base ====================
slide_num = 8
add_content_slide(prs, "Knowledge Base Structure", [
    "70+ Biomarkers organized into 8 health domains:",
    "   → Metabolic: Glucose, HbA1c, Lipids, Insulin resistance",
    "   → Cardiovascular: Blood pressure, Heart rate, Arterial stiffness",
    "   → Inflammatory: CRP, IL-6, TNF-α, Fibrinogen",
    "   → Hepatic: ALT, AST, GGT, Bilirubin",
    "   → Renal: Creatinine, eGFR, Albumin, BUN",
    "   → Anthropometric: BMI, Waist circumference, Body fat %",
    "   → Lifestyle: Physical activity, Diet quality, Sleep, Stress",
    "   → Hormonal: Thyroid, Cortisol, Testosterone/Estrogen"
], slide_num, "📊")

# ==================== SLIDE 9: Evidence Grading ====================
slide_num = 9
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
add_header_bar(slide)
add_footer_line(slide)

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "📈 Three-Tier Evidence Grading System"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE

# Tier boxes
tiers = [
    ("Tier 1: Highest Evidence", "RCT + Mendelian Randomization", 
     "• Randomized controlled trial support\n• Causal inference from genetic variants\n• Weight: 1.0 (full)", 
     RGBColor(0, 128, 0), 1.3),
    ("Tier 2: Strong Evidence", "Mendelian Randomization Only",
     "• Genetic causal inference\n• No RCT confirmation required\n• Weight: 0.8",
     RGBColor(0, 102, 204), 3.3),
    ("Tier 3: Mechanistic", "Biological Plausibility",
     "• Pathway-based reasoning\n• Observational study support\n• Weight: 0.5",
     RGBColor(255, 140, 0), 5.3),
]

for title, subtitle, content, color, y in tiers:
    # Main box
    box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1), Inches(y), Inches(11.333), Inches(1.7))
    box.fill.solid()
    box.fill.fore_color.rgb = LIGHT_GRAY
    box.line.color.rgb = color
    box.line.width = Pt(3)
    
    # Color indicator
    indicator = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(1), Inches(y), Inches(0.3), Inches(1.7))
    indicator.fill.solid()
    indicator.fill.fore_color.rgb = color
    indicator.line.fill.background()
    
    # Title
    tb = slide.shapes.add_textbox(Inches(1.5), Inches(y + 0.1), Inches(5), Inches(0.4))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    # Subtitle
    tb = slide.shapes.add_textbox(Inches(1.5), Inches(y + 0.5), Inches(5), Inches(0.3))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = subtitle
    p.font.size = Pt(14)
    p.font.italic = True
    p.font.color.rgb = GRAY
    
    # Content
    tb = slide.shapes.add_textbox(Inches(6.5), Inches(y + 0.15), Inches(5.5), Inches(1.5))
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = content
    p.font.size = Pt(14)
    p.font.color.rgb = DARK_BLUE

add_slide_number(slide, slide_num)

# ==================== SLIDE 10: DAG-Based Causal Modeling ====================
slide_num = 10
add_content_slide(prs, "DAG-Based Causal Modeling", [
    "Directed Acyclic Graph (DAG) represents causal relationships",
    "Nodes: Biomarkers, conditions, lifestyle factors, disease outcomes",
    "Edges: Causal relationships with evidence-graded weights",
    "Key properties:",
    "   → Acyclicity ensures consistent causal direction",
    "   → Transitive closure for indirect effect propagation",
    "   → Confounding control through d-separation",
    "Enables 'what-if' reasoning: If biomarker X changes, what cascades?",
    "Supports both forward (prediction) and backward (explanation) reasoning"
], slide_num, "🔗")

# ==================== SLIDE 11: Risk Score Algorithm ====================
slide_num = 11
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
add_header_bar(slide)
add_footer_line(slide)

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "🧮 Risk Score Algorithm"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE

# Formula box
formula_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1), Inches(1.2), Inches(11.333), Inches(1.5))
formula_box.fill.solid()
formula_box.fill.fore_color.rgb = RGBColor(245, 248, 255)
formula_box.line.color.rgb = MEDIUM_BLUE
formula_box.line.width = Pt(2)

formula_tb = slide.shapes.add_textbox(Inches(1.2), Inches(1.5), Inches(11), Inches(1))
tf = formula_tb.text_frame
p = tf.paragraphs[0]
p.text = "Risk Score = Σ (wᵢ × dᵢ × eᵢ) / Σ wᵢ"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = DARK_BLUE
p.alignment = PP_ALIGN.CENTER

# Variables explanation
vars_content = [
    ("wᵢ", "Base weight of biomarker i (domain importance)"),
    ("dᵢ", "Deviation score: (value - optimal) / (threshold - optimal)"),
    ("eᵢ", "Evidence multiplier from 3-tier grading system"),
]

y_pos = 3.0
for var, desc in vars_content:
    tb = slide.shapes.add_textbox(Inches(1.5), Inches(y_pos), Inches(10), Inches(0.5))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = f"{var}  =  {desc}"
    p.font.size = Pt(18)
    p.font.color.rgb = DARK_BLUE
    y_pos += 0.6

# Additional notes
notes_tb = slide.shapes.add_textbox(Inches(1), Inches(5.0), Inches(11), Inches(1.5))
tf = notes_tb.text_frame
tf.word_wrap = True
notes = [
    "• Scores normalized to 0-100 scale for interpretability",
    "• Domain-specific weights reflect clinical importance",
    "• Trend adjustment: ±10% for improving/worsening trajectories"
]
for i, note in enumerate(notes):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = note
    p.font.size = Pt(16)
    p.font.color.rgb = GRAY
    p.space_after = Pt(8)

add_slide_number(slide, slide_num)

# ==================== SLIDE 12: Pattern Detection ====================
slide_num = 12
add_content_slide(prs, "Automated Pattern Detection", [
    "5 Core Analytics Capabilities:",
    "",
    "1️⃣  Multi-Biomarker Correlation Clusters",
    "       Identifies co-occurring abnormalities across domains",
    "",
    "2️⃣  Temporal Trend Analysis",
    "       Tracks biomarker trajectories over time",
    "",
    "3️⃣  Risk Cascade Identification",
    "       Maps how one abnormality triggers others",
    "",
    "4️⃣  Intervention Impact Estimation",
    "       Predicts downstream effects of changes",
    "",
    "5️⃣  Personalized Threshold Adjustment",
    "       Adapts reference ranges to patient context"
], slide_num, "🔎")

# ==================== SLIDE 13: What-If Simulator ====================
slide_num = 13
add_content_slide(prs, "What-If Intervention Simulator", [
    "Enables prospective intervention planning",
    "User inputs: Proposed lifestyle or medication changes",
    "System simulates cascade effects through causal DAG",
    "Outputs:",
    "   → Predicted biomarker changes (with confidence intervals)",
    "   → Updated risk scores across all NCD domains",
    "   → Time-to-effect estimates",
    "   → Potential side effects or trade-offs",
    "Example: 'What if I reduce carbohydrate intake by 30%?'",
    "   → Predicts glucose, HbA1c, triglyceride, weight changes"
], slide_num, "🎮")

# ==================== SLIDE 14: Case Study ====================
slide_num = 14
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
add_header_bar(slide)
add_footer_line(slide)

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "👤 Case Study: Metabolic Syndrome Patient"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE

# Patient profile box
profile_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.2), Inches(5.8), Inches(3))
profile_box.fill.solid()
profile_box.fill.fore_color.rgb = RGBColor(240, 248, 255)
profile_box.line.color.rgb = MEDIUM_BLUE

profile_title = slide.shapes.add_textbox(Inches(0.7), Inches(1.3), Inches(5.4), Inches(0.4))
tf = profile_title.text_frame
p = tf.paragraphs[0]
p.text = "Patient Profile"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = DARK_BLUE

profile_content = slide.shapes.add_textbox(Inches(0.7), Inches(1.8), Inches(5.4), Inches(2.2))
tf = profile_content.text_frame
tf.word_wrap = True
profile_items = [
    "• Age: 55 years, Male",
    "• BMI: 31.2 kg/m² (Obese Class I)",
    "• Waist: 104 cm",
    "• Fasting glucose: 118 mg/dL (pre-diabetic)",
    "• HbA1c: 6.2% (pre-diabetic)",
    "• BP: 142/92 mmHg (Stage 2 HTN)",
    "• Triglycerides: 185 mg/dL (elevated)",
    "• HDL: 38 mg/dL (low)"
]
for i, item in enumerate(profile_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    p.font.color.rgb = DARK_BLUE
    p.space_after = Pt(4)

# Intervention box
interv_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.2), Inches(5.8), Inches(3))
interv_box.fill.solid()
interv_box.fill.fore_color.rgb = RGBColor(255, 248, 240)
interv_box.line.color.rgb = RGBColor(255, 140, 0)

interv_title = slide.shapes.add_textbox(Inches(7.0), Inches(1.3), Inches(5.4), Inches(0.4))
tf = interv_title.text_frame
p = tf.paragraphs[0]
p.text = "Intervention Protocol"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = DARK_BLUE

interv_content = slide.shapes.add_textbox(Inches(7.0), Inches(1.8), Inches(5.4), Inches(2.2))
tf = interv_content.text_frame
tf.word_wrap = True
interv_items = [
    "• Mediterranean diet adoption",
    "• 150 min/week moderate exercise",
    "• Weight loss target: 7% (5.5 kg)",
    "• Stress management program",
    "• Sleep optimization (7-8 hrs)",
    "",
    "Duration: 6 months",
    "Monitoring: Monthly biomarker checks"
]
for i, item in enumerate(interv_items):
    if i == 0:
        p = tf.paragraphs[0]
    else:
        p = tf.add_paragraph()
    p.text = item
    p.font.size = Pt(14)
    p.font.color.rgb = DARK_BLUE
    p.space_after = Pt(4)

# NCD-CIE Analysis note
analysis_box = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(12), Inches(2))
tf = analysis_box.text_frame
tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "NCD-CIE Analysis: Identified interconnected metabolic syndrome pattern affecting 5 biomarker domains with cascading CVD and T2DM risk. What-if simulation predicted intervention outcomes."
p.font.size = Pt(16)
p.font.color.rgb = GRAY

add_slide_number(slide, slide_num)

# ==================== SLIDE 15: Results ====================
slide_num = 15
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)
add_header_bar(slide)
add_footer_line(slide)

title_box = slide.shapes.add_textbox(Inches(0.5), Inches(0.1), Inches(12), Inches(0.7))
tf = title_box.text_frame
p = tf.paragraphs[0]
p.text = "📊 Results: Biomarker Changes & Risk Reduction"
p.font.size = Pt(28)
p.font.bold = True
p.font.color.rgb = WHITE

# Results table header
headers = ["Biomarker", "Baseline", "6-Month", "Change"]
col_widths = [3.0, 2.2, 2.2, 2.2]
x_positions = [1.0, 4.0, 6.2, 8.4]

for i, (header, x) in enumerate(zip(headers, x_positions)):
    box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.2), Inches(col_widths[i]), Inches(0.45))
    box.fill.solid()
    box.fill.fore_color.rgb = DARK_BLUE
    box.line.fill.background()
    
    tb = slide.shapes.add_textbox(Inches(x), Inches(1.25), Inches(col_widths[i]), Inches(0.4))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = header
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

# Data rows
data = [
    ("Fasting Glucose", "118 mg/dL", "102 mg/dL", "↓ 13.6%"),
    ("HbA1c", "6.2%", "5.6%", "↓ 9.7%"),
    ("Blood Pressure", "142/92", "128/82", "↓ 10%"),
    ("Triglycerides", "185 mg/dL", "142 mg/dL", "↓ 23.2%"),
    ("HDL Cholesterol", "38 mg/dL", "45 mg/dL", "↑ 18.4%"),
    ("Weight/BMI", "31.2 kg/m²", "28.9 kg/m²", "↓ 7.4%"),
]

y_start = 1.7
for row_idx, (bio, base, post, change) in enumerate(data):
    y = y_start + (row_idx * 0.5)
    bg_color = LIGHT_GRAY if row_idx % 2 == 0 else WHITE
    
    for i, (val, x) in enumerate(zip([bio, base, post, change], x_positions)):
        box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(col_widths[i]), Inches(0.45))
        box.fill.solid()
        box.fill.fore_color.rgb = bg_color
        box.line.fill.background()
        
        tb = slide.shapes.add_textbox(Inches(x), Inches(y + 0.05), Inches(col_widths[i]), Inches(0.35))
        tf = tb.text_frame
        p = tf.paragraphs[0]
        p.text = val
        p.font.size = Pt(13)
        p.font.color.rgb = DARK_BLUE if i < 3 else RGBColor(0, 128, 0)
        p.alignment = PP_ALIGN.CENTER

# Risk reduction summary
risk_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1), Inches(5.0), Inches(11), Inches(1.7))
risk_box.fill.solid()
risk_box.fill.fore_color.rgb = RGBColor(240, 255, 240)
risk_box.line.color.rgb = RGBColor(0, 128, 0)
risk_box.line.width = Pt(2)

risk_title = slide.shapes.add_textbox(Inches(1.2), Inches(5.1), Inches(10.6), Inches(0.4))
tf = risk_title.text_frame
p = tf.paragraphs[0]
p.text = "Calculated Risk Reductions (NCD-CIE)"
p.font.size = Pt(18)
p.font.bold = True
p.font.color.rgb = RGBColor(0, 100, 0)

risk_content = slide.shapes.add_textbox(Inches(1.2), Inches(5.5), Inches(10.6), Inches(1))
tf = risk_content.text_frame
p = tf.paragraphs[0]
p.text = "CVD 10-Year Risk: -32%    |    Type 2 Diabetes Risk: -37.5%    |    Metabolic Syndrome Score: -45%"
p.font.size = Pt(20)
p.font.bold = True
p.font.color.rgb = DARK_BLUE
p.alignment = PP_ALIGN.CENTER

add_slide_number(slide, slide_num)

# ==================== SLIDE 16: Prediction vs Observed ====================
slide_num = 16
add_content_slide(prs, "Prediction Accuracy: NCD-CIE vs Observed", [
    "Validation: Compared NCD-CIE predictions to actual outcomes",
    "",
    "Mean Absolute Difference: 3.2% across all biomarkers",
    "",
    "Prediction accuracy by biomarker:",
    "   → Glucose: Predicted ↓14%, Observed ↓13.6% (Δ 0.4%)",
    "   → HbA1c: Predicted ↓10%, Observed ↓9.7% (Δ 0.3%)",
    "   → Triglycerides: Predicted ↓25%, Observed ↓23.2% (Δ 1.8%)",
    "   → Blood Pressure: Predicted ↓12%, Observed ↓10% (Δ 2.0%)",
    "",
    "Result: NCD-CIE predictions closely matched real-world outcomes"
], slide_num, "✅")

# ==================== SLIDE 17: Validation Roadmap ====================
slide_num = 17
add_content_slide(prs, "Validation Roadmap", [
    "Current: Single-case proof-of-concept demonstration",
    "",
    "Planned validation framework:",
    "",
    "Phase 1: Target Trial Emulation",
    "   → Use observational data to emulate RCT conditions",
    "   → Compare NCD-CIE predictions to historical outcomes",
    "",
    "Phase 2: Prospective Cohort Study",
    "   → Recruit metabolic syndrome patients (n=100+)",
    "   → Track NCD-CIE predictions vs actual outcomes over 12 months",
    "",
    "Phase 3: Multi-Center Validation",
    "   → External validation across diverse populations"
], slide_num, "🗺️")

# ==================== SLIDE 18: Limitations ====================
slide_num = 18
add_content_slide(prs, "Limitations", [
    "N=1 Case Study: Single patient proof-of-concept only",
    "   → Cannot generalize to broader populations yet",
    "",
    "Static Weights: Evidence grades fixed at system design",
    "   → No dynamic updating from new research",
    "",
    "DAG Constraints: Acyclicity may oversimplify biology",
    "   → Some feedback loops cannot be represented",
    "",
    "Reference Range Assumptions: Based on general populations",
    "   → May not reflect individual physiological variation",
    "",
    "No Medication Modeling: Current focus on lifestyle only"
], slide_num, "⚠️")

# ==================== SLIDE 19: Future Directions ====================
slide_num = 19
add_content_slide(prs, "Future Directions", [
    "Prospective Validation Studies",
    "   → Large-scale cohort studies across multiple sites",
    "",
    "Machine Learning Enhancement",
    "   → Hybrid neuro-symbolic architecture",
    "   → Automated weight learning from data",
    "",
    "Multi-Omics Integration",
    "   → Genomics, proteomics, metabolomics data",
    "   → Polygenic risk scores integration",
    "",
    "Clinical Decision Support System",
    "   → EHR integration for real-time recommendations",
    "",
    "Mobile Health Application",
    "   → Patient-facing risk monitoring tool"
], slide_num, "🚀")

# ==================== SLIDE 20: Conclusions ====================
slide_num = 20
add_content_slide(prs, "Conclusions", [
    "NCD-CIE provides a unified framework for multi-disease risk assessment",
    "",
    "Key Contributions:",
    "   ✓ Hybrid symbolic-statistical architecture",
    "   ✓ Evidence-graded causal knowledge graph (70+ biomarkers)",
    "   ✓ DAG-based causal modeling for intervention simulation",
    "   ✓ Explainable, personalized risk predictions",
    "",
    "Case study demonstrated:",
    "   → Accurate prediction of biomarker changes (3.2% MAD)",
    "   → Meaningful risk reductions (CVD -32%, Diabetes -37.5%)",
    "",
    "NCD-CIE bridges the gap between statistical power and clinical reasoning"
], slide_num, "💡")

# ==================== SLIDE 21: References ====================
slide_num = 21
add_content_slide(prs, "Key References", [
    "WHO. Global Status Report on NCDs. World Health Organization, 2023.",
    "",
    "D'Agostino et al. Framingham Heart Study CVD Risk Functions. Circulation, 2008.",
    "",
    "Hernán & Robins. Causal Inference: What If. Chapman & Hall, 2020.",
    "",
    "Pearl, J. Causality: Models, Reasoning, and Inference. Cambridge, 2009.",
    "",
    "Smith & Ebrahim. Mendelian Randomization. IJE, 2003.",
    "",
    "Kohane. AI in Medicine. NEJM, 2019."
], slide_num, "📚")

# ==================== SLIDE 22: Thank You ====================
slide_num = 22
blank_layout = prs.slide_layouts[6]
slide = prs.slides.add_slide(blank_layout)

# Full blue background
bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
bg_shape.fill.solid()
bg_shape.fill.fore_color.rgb = DARK_BLUE
bg_shape.line.fill.background()

# Thank you text
thank_box = slide.shapes.add_textbox(Inches(0.5), Inches(2.0), Inches(12.333), Inches(1.5))
tf = thank_box.text_frame
p = tf.paragraphs[0]
p.text = "Thank You"
p.font.size = Pt(60)
p.font.bold = True
p.font.color.rgb = WHITE
p.alignment = PP_ALIGN.CENTER

# Q&A
qa_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.5), Inches(12.333), Inches(1))
tf = qa_box.text_frame
p = tf.paragraphs[0]
p.text = "Questions & Discussion"
p.font.size = Pt(36)
p.font.color.rgb = LIGHT_BLUE
p.alignment = PP_ALIGN.CENTER

# Contact
contact_box = slide.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(12.333), Inches(1))
tf = contact_box.text_frame
p = tf.paragraphs[0]
p.text = "Anirach Mingkhwan & Kongkiat Kespechara"
p.font.size = Pt(20)
p.font.color.rgb = RGBColor(180, 200, 220)
p.alignment = PP_ALIGN.CENTER
p = tf.add_paragraph()
p.text = "King Mongkut's University of Technology North Bangkok"
p.font.size = Pt(16)
p.font.color.rgb = RGBColor(150, 170, 190)
p.alignment = PP_ALIGN.CENTER

add_slide_number(slide, slide_num)

# Save the presentation
output_path = "/home/clawdbot/clawd/NCD_CIE_Research_Presentation.pptx"
prs.save(output_path)
print(f"Presentation saved to: {output_path}")
print(f"Total slides: {len(prs.slides)}")
