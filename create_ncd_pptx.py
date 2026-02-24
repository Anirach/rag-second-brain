#!/usr/bin/env python3
"""Create NCD-CIE Research Presentation with AI-generated images"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
import os

# Paths
IMAGE_DIR = "/home/clawdbot/clawd/ncd_nano_images"
OUTPUT_PATH = "/home/clawdbot/clawd/NCD_CIE_Presentation_NanoBanana.pptx"

# Colors
DARK_BLUE = RGBColor(0x00, 0x2B, 0x5C)
TEAL = RGBColor(0x00, 0x82, 0x9B)
LIGHT_BLUE = RGBColor(0x00, 0xA3, 0xE0)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LIGHT_GRAY = RGBColor(0xF0, 0xF0, 0xF0)

def add_title_slide(prs):
    """Slide 1: Title"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank
    
    # Background shape
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BLUE
    bg.line.fill.background()
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1.5))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "NCD Causal Insight Engine"
    p.font.size = Pt(44)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle = slide.shapes.add_textbox(Inches(0.5), Inches(4), Inches(9), Inches(0.8))
    tf = subtitle.text_frame
    p = tf.paragraphs[0]
    p.text = "AI-Powered Causal Analysis for Non-Communicable Disease Prevention"
    p.font.size = Pt(24)
    p.font.color.rgb = LIGHT_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Authors
    authors = slide.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(9), Inches(1))
    tf = authors.text_frame
    p = tf.paragraphs[0]
    p.text = "Research Team: NCD-CIE Consortium\nJune 2025"
    p.font.size = Pt(18)
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

def add_problem_slide(prs):
    """Slide 2: Problem Statement"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "The NCD Crisis"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Stat boxes
    stats = [
        ("41M", "Deaths/Year", "NCDs cause 74% of\nglobal deaths"),
        ("80%", "Preventable", "With early intervention\nand lifestyle changes"),
        ("$47T", "Cost by 2030", "Global economic burden\nof NCDs")
    ]
    
    for i, (num, label, desc) in enumerate(stats):
        x = Inches(0.5 + i * 3.1)
        # Box
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.8), Inches(2.9), Inches(2.5))
        box.fill.solid()
        box.fill.fore_color.rgb = TEAL if i == 1 else LIGHT_GRAY
        
        # Number
        num_box = slide.shapes.add_textbox(x, Inches(2), Inches(2.9), Inches(0.8))
        tf = num_box.text_frame
        p = tf.paragraphs[0]
        p.text = num
        p.font.size = Pt(40)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 1 else DARK_BLUE
        p.alignment = PP_ALIGN.CENTER
        
        # Label
        lbl = slide.shapes.add_textbox(x, Inches(2.7), Inches(2.9), Inches(0.5))
        tf = lbl.text_frame
        p = tf.paragraphs[0]
        p.text = label
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = WHITE if i == 1 else DARK_BLUE
        p.alignment = PP_ALIGN.CENTER
        
        # Description
        desc_box = slide.shapes.add_textbox(x + Inches(0.1), Inches(3.3), Inches(2.7), Inches(0.9))
        tf = desc_box.text_frame
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE if i == 1 else DARK_BLUE
        p.alignment = PP_ALIGN.CENTER
    
    # Bottom message
    msg = slide.shapes.add_textbox(Inches(0.5), Inches(5), Inches(9), Inches(1.5))
    tf = msg.text_frame
    p = tf.paragraphs[0]
    p.text = "Current Challenge: Healthcare systems react to disease, not prevent it.\nWe need systems that identify causal pathways BEFORE symptoms appear."
    p.font.size = Pt(18)
    p.font.color.rgb = DARK_BLUE
    p.alignment = PP_ALIGN.CENTER

def add_research_gap_slide(prs):
    """Slide 3: Research Gap"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "The Research Gap: WHY vs HOW"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Two columns
    # WHY column
    why_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.6), Inches(4.3), Inches(4.5))
    why_box.fill.solid()
    why_box.fill.fore_color.rgb = RGBColor(0xE8, 0xF4, 0xFC)
    
    why_title = slide.shapes.add_textbox(Inches(0.7), Inches(1.8), Inches(4), Inches(0.6))
    tf = why_title.text_frame
    p = tf.paragraphs[0]
    p.text = "Traditional ML: WHAT"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    why_text = slide.shapes.add_textbox(Inches(0.7), Inches(2.4), Inches(4), Inches(3.5))
    tf = why_text.text_frame
    for item in ["✗ Correlation-based predictions", "✗ Black-box risk scores", "✗ No mechanistic insight", "✗ Poor transferability", "✗ Cannot guide intervention"]:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = item
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(0xC0, 0x39, 0x2B)
        p.space_after = Pt(12)
    
    # HOW column
    how_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5.2), Inches(1.6), Inches(4.3), Inches(4.5))
    how_box.fill.solid()
    how_box.fill.fore_color.rgb = TEAL
    
    how_title = slide.shapes.add_textbox(Inches(5.4), Inches(1.8), Inches(4), Inches(0.6))
    tf = how_title.text_frame
    p = tf.paragraphs[0]
    p.text = "NCD-CIE: WHY + HOW"
    p.font.size = Pt(22)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    how_text = slide.shapes.add_textbox(Inches(5.4), Inches(2.4), Inches(4), Inches(3.5))
    tf = how_text.text_frame
    for item in ["✓ Causal pathway discovery", "✓ Explainable mechanisms", "✓ Knowledge-graph integration", "✓ Intervention simulation", "✓ Actionable recommendations"]:
        p = tf.add_paragraph() if tf.paragraphs[0].text else tf.paragraphs[0]
        p.text = item
        p.font.size = Pt(18)
        p.font.color.rgb = WHITE
        p.space_after = Pt(12)
    
    # Arrow
    arrow = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(4.5), Inches(3.5), Inches(0.8), Inches(0.5))
    arrow.fill.solid()
    arrow.fill.fore_color.rgb = DARK_BLUE

def add_image_slide(prs, title_text, image_name, subtitle=""):
    """Generic slide with title and image"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = title_text
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Image
    img_path = os.path.join(IMAGE_DIR, image_name)
    if os.path.exists(img_path):
        slide.shapes.add_picture(img_path, Inches(1), Inches(1.4), width=Inches(8))
    
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.5), Inches(6.8), Inches(9), Inches(0.5))
        tf = sub.text_frame
        p = tf.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(14)
        p.font.color.rgb = DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

def add_algorithm_slide(prs):
    """Slide 7: Risk Algorithm"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Causal Risk Algorithm"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Formula box
    formula_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.6), Inches(9), Inches(1.8))
    formula_box.fill.solid()
    formula_box.fill.fore_color.rgb = LIGHT_GRAY
    
    formula = slide.shapes.add_textbox(Inches(0.7), Inches(2), Inches(8.6), Inches(1))
    tf = formula.text_frame
    p = tf.paragraphs[0]
    p.text = "Risk(Y|do(X)) = Σ P(Y|X,Pa(X)) × P(Pa(X))"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Components
    components = [
        ("do(X)", "Causal intervention operator"),
        ("Pa(X)", "Parent nodes in causal graph"),
        ("P(Y|X,Pa(X))", "Conditional probability given parents")
    ]
    
    y_pos = 3.8
    for comp, desc in components:
        comp_box = slide.shapes.add_textbox(Inches(1), Inches(y_pos), Inches(2.5), Inches(0.5))
        tf = comp_box.text_frame
        p = tf.paragraphs[0]
        p.text = comp
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = TEAL
        
        desc_box = slide.shapes.add_textbox(Inches(3.5), Inches(y_pos), Inches(5.5), Inches(0.5))
        tf = desc_box.text_frame
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(18)
        p.font.color.rgb = DARK_BLUE
        
        y_pos += 0.7
    
    # Key insight box
    insight = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6), Inches(9), Inches(1.2))
    insight.fill.solid()
    insight.fill.fore_color.rgb = TEAL
    
    insight_text = slide.shapes.add_textbox(Inches(0.7), Inches(6.2), Inches(8.6), Inches(0.8))
    tf = insight_text.text_frame
    p = tf.paragraphs[0]
    p.text = "Key Insight: Unlike correlation, causal inference tells us what happens if we intervene"
    p.font.size = Pt(18)
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

def add_pattern_detection_slide(prs):
    """Slide 8: Pattern Detection Capabilities"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Pattern Detection Capabilities"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # 5 capability boxes
    capabilities = [
        ("🔍", "Multi-Biomarker\nAnalysis", "Integrates 50+ biomarkers\nacross metabolic panels"),
        ("🔗", "Pathway\nDiscovery", "Identifies hidden causal\nconnections between markers"),
        ("⚡", "Early Warning\nSignals", "Detects pre-disease states\n5-10 years ahead"),
        ("🎯", "Personalized\nRisk Profiles", "Individual-level causal\nrisk assessment"),
        ("💊", "Intervention\nSimulation", "Models effect of lifestyle\nand medication changes")
    ]
    
    for i, (icon, name, desc) in enumerate(capabilities):
        x = Inches(0.3 + i * 1.95)
        
        # Box
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, Inches(1.8), Inches(1.85), Inches(4.5))
        box.fill.solid()
        box.fill.fore_color.rgb = TEAL if i % 2 == 0 else LIGHT_GRAY
        
        # Icon
        icon_box = slide.shapes.add_textbox(x, Inches(2), Inches(1.85), Inches(0.8))
        tf = icon_box.text_frame
        p = tf.paragraphs[0]
        p.text = icon
        p.font.size = Pt(36)
        p.alignment = PP_ALIGN.CENTER
        
        # Name
        name_box = slide.shapes.add_textbox(x + Inches(0.05), Inches(2.8), Inches(1.75), Inches(0.9))
        tf = name_box.text_frame
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = WHITE if i % 2 == 0 else DARK_BLUE
        p.alignment = PP_ALIGN.CENTER
        
        # Description
        desc_box = slide.shapes.add_textbox(x + Inches(0.05), Inches(3.8), Inches(1.75), Inches(2))
        tf = desc_box.text_frame
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(12)
        p.font.color.rgb = WHITE if i % 2 == 0 else DARK_BLUE
        p.alignment = PP_ALIGN.CENTER

def add_case_study_slide(prs):
    """Slide 10: Case Study Patient Profile"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Case Study: Patient Profile"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    # Patient info box
    patient_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.5), Inches(4), Inches(5.5))
    patient_box.fill.solid()
    patient_box.fill.fore_color.rgb = LIGHT_GRAY
    
    patient_title = slide.shapes.add_textbox(Inches(0.7), Inches(1.7), Inches(3.6), Inches(0.5))
    tf = patient_title.text_frame
    p = tf.paragraphs[0]
    p.text = "👤 Patient #4521"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = DARK_BLUE
    
    patient_details = [
        "Age: 52, Male",
        "BMI: 31.2 (Obese)",
        "HbA1c: 6.2% (Pre-diabetic)",
        "LDL: 142 mg/dL (Elevated)",
        "CRP: 3.8 mg/L (High inflammation)",
        "Family Hx: T2DM (father)",
        "Lifestyle: Sedentary, high stress"
    ]
    
    y = 2.4
    for detail in patient_details:
        d_box = slide.shapes.add_textbox(Inches(0.7), Inches(y), Inches(3.6), Inches(0.4))
        tf = d_box.text_frame
        p = tf.paragraphs[0]
        p.text = detail
        p.font.size = Pt(16)
        p.font.color.rgb = DARK_BLUE
        y += 0.5
    
    # NCD-CIE Analysis box
    analysis_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(5), Inches(1.5), Inches(4.5), Inches(5.5))
    analysis_box.fill.solid()
    analysis_box.fill.fore_color.rgb = TEAL
    
    analysis_title = slide.shapes.add_textbox(Inches(5.2), Inches(1.7), Inches(4.1), Inches(0.5))
    tf = analysis_title.text_frame
    p = tf.paragraphs[0]
    p.text = "🔬 NCD-CIE Analysis"
    p.font.size = Pt(24)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    findings = [
        "Causal Chain Detected:",
        "  Stress → Cortisol ↑ → Insulin ↓",
        "  → Glucose ↑ → Inflammation ↑",
        "",
        "Risk Assessment:",
        "  • T2DM in 3 years: 78%",
        "  • CVD in 5 years: 45%",
        "",
        "Top Intervention:",
        "  Stress reduction + Exercise",
        "  → Projected risk reduction: 52%"
    ]
    
    y = 2.4
    for line in findings:
        f_box = slide.shapes.add_textbox(Inches(5.2), Inches(y), Inches(4.1), Inches(0.35))
        tf = f_box.text_frame
        p = tf.paragraphs[0]
        p.text = line
        p.font.size = Pt(14)
        p.font.color.rgb = WHITE
        y += 0.35

def add_limitations_slide(prs):
    """Slide 13: Limitations"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Limitations & Challenges"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    limitations = [
        ("Data Quality", "Requires standardized, high-quality longitudinal data"),
        ("Causal Assumptions", "DAG structure depends on domain expertise validation"),
        ("Generalizability", "Models trained on specific populations may not transfer"),
        ("Computational Cost", "Full causal discovery is computationally intensive"),
        ("Regulatory Path", "Novel AI approach requires new validation frameworks")
    ]
    
    y = 1.6
    for i, (lim_title, lim_desc) in enumerate(limitations):
        # Box
        box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(y), Inches(9), Inches(1))
        box.fill.solid()
        box.fill.fore_color.rgb = RGBColor(0xFF, 0xF3, 0xCD) if i % 2 == 0 else LIGHT_GRAY
        
        # Title
        t_box = slide.shapes.add_textbox(Inches(0.7), Inches(y + 0.15), Inches(2.5), Inches(0.4))
        tf = t_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"⚠️ {lim_title}"
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        
        # Description
        d_box = slide.shapes.add_textbox(Inches(3.3), Inches(y + 0.15), Inches(6), Inches(0.7))
        tf = d_box.text_frame
        p = tf.paragraphs[0]
        p.text = lim_desc
        p.font.size = Pt(16)
        p.font.color.rgb = DARK_BLUE
        
        y += 1.1

def add_conclusions_slide(prs):
    """Slide 14: Conclusions"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(1.2))
    bar.fill.solid()
    bar.fill.fore_color.rgb = DARK_BLUE
    bar.line.fill.background()
    
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "Conclusions"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    conclusions = [
        ("1", "Causal AI > Correlation", "Moving from 'what' to 'why' enables precision prevention"),
        ("2", "Knowledge Graphs Scale Expertise", "Encoding medical knowledge enables consistent reasoning"),
        ("3", "Early Detection Saves Lives", "Identifying causal chains 5-10 years before symptoms"),
        ("4", "Actionable Interventions", "Simulating intervention effects guides personalized care")
    ]
    
    y = 1.5
    for num, title_text, desc in conclusions:
        # Number circle
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.5), Inches(y), Inches(0.6), Inches(0.6))
        circle.fill.solid()
        circle.fill.fore_color.rgb = TEAL
        
        num_box = slide.shapes.add_textbox(Inches(0.5), Inches(y + 0.1), Inches(0.6), Inches(0.4))
        tf = num_box.text_frame
        p = tf.paragraphs[0]
        p.text = num
        p.font.size = Pt(24)
        p.font.bold = True
        p.font.color.rgb = WHITE
        p.alignment = PP_ALIGN.CENTER
        
        # Title
        t_box = slide.shapes.add_textbox(Inches(1.3), Inches(y), Inches(8), Inches(0.5))
        tf = t_box.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = DARK_BLUE
        
        # Description
        d_box = slide.shapes.add_textbox(Inches(1.3), Inches(y + 0.5), Inches(8), Inches(0.7))
        tf = d_box.text_frame
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(18)
        p.font.color.rgb = RGBColor(0x50, 0x50, 0x50)
        
        y += 1.35
    
    # Impact statement
    impact = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(6.2), Inches(9), Inches(1))
    impact.fill.solid()
    impact.fill.fore_color.rgb = DARK_BLUE
    
    impact_text = slide.shapes.add_textbox(Inches(0.7), Inches(6.4), Inches(8.6), Inches(0.6))
    tf = impact_text.text_frame
    p = tf.paragraphs[0]
    p.text = "NCD-CIE: Transforming reactive healthcare into proactive prevention"
    p.font.size = Pt(20)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

def add_thank_you_slide(prs):
    """Slide 15: Thank You / Q&A"""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(10), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = DARK_BLUE
    bg.line.fill.background()
    
    # Thank you
    thank = slide.shapes.add_textbox(Inches(0.5), Inches(2.5), Inches(9), Inches(1))
    tf = thank.text_frame
    p = tf.paragraphs[0]
    p.text = "Thank You"
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER
    
    # Q&A
    qa = slide.shapes.add_textbox(Inches(0.5), Inches(3.8), Inches(9), Inches(0.8))
    tf = qa.text_frame
    p = tf.paragraphs[0]
    p.text = "Questions & Discussion"
    p.font.size = Pt(32)
    p.font.color.rgb = LIGHT_BLUE
    p.alignment = PP_ALIGN.CENTER
    
    # Contact
    contact = slide.shapes.add_textbox(Inches(0.5), Inches(5.5), Inches(9), Inches(1))
    tf = contact.text_frame
    p = tf.paragraphs[0]
    p.text = "📧 ncd-cie@research.org  |  🌐 ncd-cie.org"
    p.font.size = Pt(20)
    p.font.color.rgb = WHITE
    p.alignment = PP_ALIGN.CENTER

def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    print("Creating slides...")
    
    # Slide 1: Title
    add_title_slide(prs)
    print("  ✓ Slide 1: Title")
    
    # Slide 2: Problem Statement
    add_problem_slide(prs)
    print("  ✓ Slide 2: Problem Statement")
    
    # Slide 3: Research Gap
    add_research_gap_slide(prs)
    print("  ✓ Slide 3: Research Gap")
    
    # Slide 4: System Architecture (with image)
    add_image_slide(prs, "System Architecture", "system_architecture.png", 
                   "Three-layer architecture: Data Ingestion → Knowledge Processing → Risk Analysis")
    print("  ✓ Slide 4: System Architecture")
    
    # Slide 5: Knowledge Graph (with image)
    add_image_slide(prs, "Medical Knowledge Graph", "knowledge_graph.png",
                   "Interconnected biomarker relationships powering causal inference")
    print("  ✓ Slide 5: Knowledge Graph")
    
    # Slide 6: Causal DAG (with image)
    add_image_slide(prs, "Causal DAG Modeling", "causal_dag.png",
                   "Directed acyclic graphs capturing disease progression pathways")
    print("  ✓ Slide 6: Causal DAG")
    
    # Slide 7: Risk Algorithm
    add_algorithm_slide(prs)
    print("  ✓ Slide 7: Risk Algorithm")
    
    # Slide 8: Pattern Detection
    add_pattern_detection_slide(prs)
    print("  ✓ Slide 8: Pattern Detection")
    
    # Slide 9: Dashboard (with image)
    add_image_slide(prs, "NCD-CIE Platform Dashboard", "risk_dashboard.png",
                   "Real-time risk visualization and intervention recommendations")
    print("  ✓ Slide 9: Dashboard")
    
    # Slide 10: Case Study
    add_case_study_slide(prs)
    print("  ✓ Slide 10: Case Study")
    
    # Slide 11: Results (with image)
    add_image_slide(prs, "Case Study Results", "case_study_results.png",
                   "Measurable improvements through causal-guided interventions")
    print("  ✓ Slide 11: Results")
    
    # Slide 12: Validation Roadmap (with image)
    add_image_slide(prs, "Clinical Validation Roadmap", "validation_roadmap.png",
                   "Three-phase validation: Retrospective → Prospective → Multi-site RCT")
    print("  ✓ Slide 12: Validation Roadmap")
    
    # Slide 13: Limitations
    add_limitations_slide(prs)
    print("  ✓ Slide 13: Limitations")
    
    # Slide 14: Conclusions
    add_conclusions_slide(prs)
    print("  ✓ Slide 14: Conclusions")
    
    # Slide 15: Thank You
    add_thank_you_slide(prs)
    print("  ✓ Slide 15: Thank You")
    
    # Save
    prs.save(OUTPUT_PATH)
    print(f"\n✅ Presentation saved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
