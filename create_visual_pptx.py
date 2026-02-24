#!/usr/bin/env python3
"""
NCD-CIE Research Presentation - VISUAL VERSION
Creates an infographic-style presentation with shapes, diagrams, and charts
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.chart import XL_CHART_TYPE
from pptx.chart.data import CategoryChartData
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml
import colorsys

# Color palette - modern, professional
COLORS = {
    'primary': RGBColor(0x1E, 0x3A, 0x5F),      # Dark blue
    'secondary': RGBColor(0x3D, 0x5A, 0x80),    # Medium blue
    'accent1': RGBColor(0x98, 0xC1, 0xD9),      # Light blue
    'accent2': RGBColor(0xE0, 0xFB, 0xFC),      # Very light blue
    'highlight': RGBColor(0xEE, 0x6C, 0x4D),    # Orange/coral
    'success': RGBColor(0x2E, 0xCC, 0x71),      # Green
    'warning': RGBColor(0xF3, 0x9C, 0x12),      # Yellow/orange
    'tier1': RGBColor(0x27, 0xAE, 0x60),        # Strong green
    'tier2': RGBColor(0xF1, 0xC4, 0x0F),        # Yellow
    'tier3': RGBColor(0xE6, 0x7E, 0x22),        # Orange
    'white': RGBColor(0xFF, 0xFF, 0xFF),
    'dark': RGBColor(0x2C, 0x3E, 0x50),
    'gray': RGBColor(0x7F, 0x8C, 0x8D),
    'light_gray': RGBColor(0xEC, 0xF0, 0xF1),
}

# Domain colors for knowledge base
DOMAIN_COLORS = [
    RGBColor(0xE7, 0x4C, 0x3C),  # Red - Cardiovascular
    RGBColor(0x34, 0x98, 0xDB),  # Blue - Metabolic
    RGBColor(0x2E, 0xCC, 0x71),  # Green - Inflammation
    RGBColor(0x9B, 0x59, 0xB6),  # Purple - Hormonal
    RGBColor(0xF3, 0x9C, 0x12),  # Orange - Nutrition
    RGBColor(0x1A, 0xBC, 0x9C),  # Teal - Liver
    RGBColor(0xE9, 0x1E, 0x63),  # Pink - Kidney
    RGBColor(0x00, 0xBC, 0xD4),  # Cyan - Thyroid
]

def add_styled_shape(slide, shape_type, left, top, width, height, fill_color, text="", font_size=12, font_color=None, bold=False):
    """Add a shape with styling"""
    shape = slide.shapes.add_shape(shape_type, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = fill_color
    shape.line.color.rgb = COLORS['dark']
    shape.line.width = Pt(1)
    
    if text:
        tf = shape.text_frame
        tf.word_wrap = True
        tf.auto_size = None
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(font_size)
        p.font.color.rgb = font_color if font_color else COLORS['white']
        p.font.bold = bold
        p.alignment = PP_ALIGN.CENTER
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        shape.text_frame.paragraphs[0].alignment = PP_ALIGN.CENTER
    
    return shape

def add_arrow_connector(slide, start_x, start_y, end_x, end_y, color=None):
    """Add an arrow line between two points"""
    connector = slide.shapes.add_shape(
        MSO_SHAPE.RIGHT_ARROW,
        start_x, start_y,
        end_x - start_x, Inches(0.3)
    )
    connector.fill.solid()
    connector.fill.fore_color.rgb = color if color else COLORS['highlight']
    connector.line.fill.background()
    return connector

def add_line(slide, start_x, start_y, end_x, end_y, color=None, width=2):
    """Add a simple line"""
    line = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT,
        start_x, start_y,
        end_x, end_y
    )
    line.line.color.rgb = color if color else COLORS['dark']
    line.line.width = Pt(width)
    return line

def create_title_slide(prs):
    """Slide 1: Visual Title Slide"""
    slide_layout = prs.slide_layouts[6]  # Blank
    slide = prs.slides.add_slide(slide_layout)
    
    # Background gradient effect with shapes
    bg_shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg_shape.fill.solid()
    bg_shape.fill.fore_color.rgb = COLORS['primary']
    bg_shape.line.fill.background()
    
    # Decorative circles
    for i, (x, y, size) in enumerate([
        (Inches(8), Inches(0.5), Inches(2)),
        (Inches(9), Inches(5), Inches(1.5)),
        (Inches(0.5), Inches(4), Inches(1)),
        (Inches(1), Inches(6), Inches(0.8)),
    ]):
        circle = slide.shapes.add_shape(MSO_SHAPE.OVAL, x, y, size, size)
        circle.fill.solid()
        circle.fill.fore_color.rgb = COLORS['accent1']
        circle.fill.fore_color.brightness = 0.3
        circle.line.fill.background()
    
    # Main title box
    title_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.5), Inches(1.5), Inches(9), Inches(2)
    )
    title_box.fill.solid()
    title_box.fill.fore_color.rgb = COLORS['white']
    title_box.line.fill.background()
    
    tf = title_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "NCD-CIE"
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    p.alignment = PP_ALIGN.CENTER
    
    p2 = tf.add_paragraph()
    p2.text = "Non-Communicable Disease Clinical Inference Engine"
    p2.font.size = Pt(24)
    p2.font.color.rgb = COLORS['secondary']
    p2.alignment = PP_ALIGN.CENTER
    
    # Subtitle
    subtitle = slide.shapes.add_textbox(Inches(1), Inches(4), Inches(8), Inches(1))
    tf = subtitle.text_frame
    p = tf.paragraphs[0]
    p.text = "AI-Powered Biomarker Analysis & Health Risk Prediction"
    p.font.size = Pt(20)
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER
    
    # Visual icons row at bottom
    icons = ["🧬", "📊", "🔬", "💉", "❤️"]
    for i, icon in enumerate(icons):
        icon_box = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            Inches(2 + i * 1.2), Inches(5.5), Inches(0.8), Inches(0.8)
        )
        icon_box.fill.solid()
        icon_box.fill.fore_color.rgb = COLORS['highlight']
        icon_box.line.fill.background()
        
        tf = icon_box.text_frame
        tf.paragraphs[0].text = icon
        tf.paragraphs[0].font.size = Pt(24)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER

def create_problem_slide(prs):
    """Slide 2: The Problem - Visual Statistics"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "THE PROBLEM: NCDs Account for 74% of Global Deaths"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Create 4 large stat boxes
    stats = [
        ("41M", "Deaths/Year", COLORS['highlight']),
        ("74%", "Global Deaths", COLORS['tier1']),
        ("80%", "Preventable", COLORS['secondary']),
        ("$47T", "Cost by 2030", COLORS['tier3']),
    ]
    
    for i, (number, label, color) in enumerate(stats):
        x = Inches(0.5 + i * 2.4)
        
        # Main stat box
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x, Inches(1.3), Inches(2.2), Inches(1.8)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()
        
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = number
        p.font.size = Pt(44)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.size = Pt(14)
        p2.font.color.rgb = COLORS['white']
        p2.alignment = PP_ALIGN.CENTER
    
    # Challenge boxes at bottom
    challenges = [
        ("📋", "Complex Biomarker\nInterpretation"),
        ("🔗", "Causal Relationships\nNot Clear"),
        ("📉", "Static Risk\nAssessments"),
        ("🎯", "Generic\nRecommendations"),
    ]
    
    for i, (icon, text) in enumerate(challenges):
        x = Inches(0.5 + i * 2.4)
        
        # Icon circle
        icon_circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            x + Inches(0.7), Inches(3.5), Inches(0.8), Inches(0.8)
        )
        icon_circle.fill.solid()
        icon_circle.fill.fore_color.rgb = COLORS['primary']
        icon_circle.line.fill.background()
        tf = icon_circle.text_frame
        tf.paragraphs[0].text = icon
        tf.paragraphs[0].font.size = Pt(24)
        tf.paragraphs[0].alignment = PP_ALIGN.CENTER
        
        # Text box below
        text_box = slide.shapes.add_textbox(x, Inches(4.4), Inches(2.2), Inches(1))
        tf = text_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(12)
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER
    
    # Down arrows connecting stats to challenges
    for i in range(4):
        x = Inches(1.4 + i * 2.4)
        arrow = slide.shapes.add_shape(
            MSO_SHAPE.DOWN_ARROW,
            x, Inches(3.15), Inches(0.4), Inches(0.35)
        )
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = COLORS['gray']
        arrow.line.fill.background()

def create_architecture_slide(prs):
    """Slide 3: System Architecture Flowchart"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "SYSTEM ARCHITECTURE"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Main flow: Input → Knowledge Graph → Risk Engine → Recommendations
    components = [
        ("USER\nINPUT", Inches(0.3), COLORS['secondary']),
        ("KNOWLEDGE\nGRAPH", Inches(2.5), COLORS['tier1']),
        ("RISK\nENGINE", Inches(4.7), COLORS['highlight']),
        ("ACTION\nPLAN", Inches(6.9), COLORS['primary']),
    ]
    
    # Draw main flow boxes
    for text, x, color in components:
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x, Inches(1.2), Inches(2), Inches(1.2)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()
        
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
    
    # Arrows between main boxes
    for i in range(3):
        arrow = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW,
            Inches(2.35 + i * 2.2), Inches(1.6), Inches(0.5), Inches(0.3)
        )
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = COLORS['dark']
        arrow.line.fill.background()
    
    # Sub-components under each main component
    sub_components = [
        # Under User Input
        [("Lab Results", COLORS['accent1']), ("Demographics", COLORS['accent1']), ("History", COLORS['accent1'])],
        # Under Knowledge Graph
        [("180+\nBiomarkers", COLORS['accent1']), ("8 Health\nDomains", COLORS['accent1']), ("DAG\nRelationships", COLORS['accent1'])],
        # Under Risk Engine
        [("Bayesian\nAnalysis", COLORS['accent1']), ("Causal\nInference", COLORS['accent1']), ("Multi-factor\nScoring", COLORS['accent1'])],
        # Under Output
        [("Risk\nScores", COLORS['accent1']), ("Priority\nActions", COLORS['accent1']), ("Monitoring\nPlan", COLORS['accent1'])],
    ]
    
    base_x = [Inches(0.3), Inches(2.5), Inches(4.7), Inches(6.9)]
    
    for i, subs in enumerate(sub_components):
        for j, (text, color) in enumerate(subs):
            sub_box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                base_x[i] + Inches(j * 0.68), Inches(2.8 + j * 0.55), Inches(0.65), Inches(0.5)
            )
            sub_box.fill.solid()
            sub_box.fill.fore_color.rgb = color
            sub_box.line.color.rgb = COLORS['secondary']
            sub_box.line.width = Pt(1)
            
            tf = sub_box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = text
            p.font.size = Pt(8)
            p.font.color.rgb = COLORS['dark']
            p.alignment = PP_ALIGN.CENTER
        
        # Vertical connector line from main box
        line = slide.shapes.add_shape(
            MSO_SHAPE.DOWN_ARROW,
            base_x[i] + Inches(0.85), Inches(2.4), Inches(0.3), Inches(0.35)
        )
        line.fill.solid()
        line.fill.fore_color.rgb = COLORS['secondary']
        line.line.fill.background()
    
    # Bottom banner
    banner = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(4.8), Inches(8.6), Inches(0.6)
    )
    banner.fill.solid()
    banner.fill.fore_color.rgb = COLORS['light_gray']
    banner.line.fill.background()
    
    tf = banner.text_frame
    p = tf.paragraphs[0]
    p.text = "🔄 Continuous Learning  •  📊 Evidence-Based  •  🎯 Personalized"
    p.font.size = Pt(14)
    p.font.color.rgb = COLORS['dark']
    p.alignment = PP_ALIGN.CENTER

def create_knowledge_base_slide(prs):
    """Slide 4: Knowledge Base - 8 Domain Grid"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "KNOWLEDGE BASE: 8 Health Domains"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # 8 domains in 2 rows of 4
    domains = [
        ("❤️", "CARDIOVASCULAR", "32 biomarkers", DOMAIN_COLORS[0]),
        ("🔥", "METABOLIC", "28 biomarkers", DOMAIN_COLORS[1]),
        ("🛡️", "INFLAMMATION", "24 biomarkers", DOMAIN_COLORS[2]),
        ("⚡", "HORMONAL", "22 biomarkers", DOMAIN_COLORS[3]),
        ("🥗", "NUTRITION", "26 biomarkers", DOMAIN_COLORS[4]),
        ("🫀", "LIVER", "18 biomarkers", DOMAIN_COLORS[5]),
        ("💧", "KIDNEY", "16 biomarkers", DOMAIN_COLORS[6]),
        ("🦋", "THYROID", "14 biomarkers", DOMAIN_COLORS[7]),
    ]
    
    for i, (icon, name, count, color) in enumerate(domains):
        row = i // 4
        col = i % 4
        x = Inches(0.4 + col * 2.3)
        y = Inches(1.0 + row * 1.9)
        
        # Main domain box
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x, y, Inches(2.1), Inches(1.6)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.fill.background()
        
        # Icon at top
        icon_text = slide.shapes.add_textbox(x, y + Inches(0.1), Inches(2.1), Inches(0.5))
        tf = icon_text.text_frame
        p = tf.paragraphs[0]
        p.text = icon
        p.font.size = Pt(32)
        p.alignment = PP_ALIGN.CENTER
        
        # Domain name
        name_text = slide.shapes.add_textbox(x, y + Inches(0.6), Inches(2.1), Inches(0.4))
        tf = name_text.text_frame
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        # Count badge
        badge = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            x + Inches(0.55), y + Inches(1.05), Inches(1), Inches(0.45)
        )
        badge.fill.solid()
        badge.fill.fore_color.rgb = COLORS['white']
        badge.line.fill.background()
        
        tf = badge.text_frame
        p = tf.paragraphs[0]
        p.text = count
        p.font.size = Pt(10)
        p.font.color.rgb = color
        p.font.bold = True
        p.alignment = PP_ALIGN.CENTER
    
    # Total at bottom
    total_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(2.5), Inches(4.9), Inches(4.2), Inches(0.6)
    )
    total_box.fill.solid()
    total_box.fill.fore_color.rgb = COLORS['primary']
    total_box.line.fill.background()
    
    tf = total_box.text_frame
    p = tf.paragraphs[0]
    p.text = "TOTAL: 180+ Biomarkers  •  500+ Relationships"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER

def create_evidence_grading_slide(prs):
    """Slide 5: Evidence Grading - Pyramid/Stacked Bars"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "EVIDENCE GRADING SYSTEM"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Create pyramid shape with 3 tiers
    tiers = [
        ("TIER 1", "Meta-analyses, Systematic Reviews", "Weight: 1.0", COLORS['tier1'], Inches(3), Inches(3)),
        ("TIER 2", "RCTs, Prospective Studies", "Weight: 0.7", COLORS['tier2'], Inches(2.2), Inches(4)),
        ("TIER 3", "Observational, Expert Consensus", "Weight: 0.4", COLORS['tier3'], Inches(1.4), Inches(5)),
    ]
    
    for tier_name, description, weight, color, width, top in tiers:
        x = Inches(4.75) - width / 2  # Center align
        
        # Tier shape (trapezoid effect with rectangles)
        box = slide.shapes.add_shape(
            MSO_SHAPE.PENTAGON if tier_name == "TIER 1" else MSO_SHAPE.CHEVRON,
            x, top, width, Inches(0.9)
        )
        if tier_name == "TIER 1":
            # Use rectangle for top
            box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, top, width, Inches(0.85))
        
        box.fill.solid()
        box.fill.fore_color.rgb = color
        box.line.color.rgb = COLORS['white']
        box.line.width = Pt(2)
        
        # Tier label inside
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{tier_name}: {weight}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        # Description to the side
        desc_box = slide.shapes.add_textbox(Inches(7.5), top + Inches(0.15), Inches(2.3), Inches(0.6))
        tf = desc_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = description
        p.font.size = Pt(11)
        p.font.color.rgb = COLORS['dark']
    
    # Left side: What it means
    info_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(1), Inches(2.8), Inches(2.5)
    )
    info_box.fill.solid()
    info_box.fill.fore_color.rgb = COLORS['light_gray']
    info_box.line.fill.background()
    
    tf = info_box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "📚 How It Works"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    points = [
        "✓ Every relationship tagged",
        "✓ Weights affect scoring",
        "✓ Transparent confidence",
        "✓ Continuous updates"
    ]
    for point in points:
        p = tf.add_paragraph()
        p.text = point
        p.font.size = Pt(11)
        p.font.color.rgb = COLORS['dark']
        p.space_before = Pt(6)

def create_dag_slide(prs):
    """Slide 6: Causal DAG - Actual Graph with Nodes and Arrows"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "CAUSAL DAG: How Biomarkers Connect"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Create nodes for the DAG
    nodes = {
        'visceral_fat': (Inches(0.5), Inches(1.8), "VISCERAL\nFAT", COLORS['highlight']),
        'inflammation': (Inches(3), Inches(1.3), "INFLAMMATION\n(CRP, IL-6)", COLORS['tier3']),
        'insulin_res': (Inches(3), Inches(3.2), "INSULIN\nRESISTANCE", COLORS['tier2']),
        'dyslipidemia': (Inches(5.5), Inches(1.3), "DYSLIPIDEMIA\n(TG, HDL)", COLORS['secondary']),
        'hypertension': (Inches(5.5), Inches(3.2), "HYPERTENSION", COLORS['secondary']),
        'diabetes': (Inches(8), Inches(2.2), "TYPE 2\nDIABETES", COLORS['highlight']),
    }
    
    # Draw nodes
    node_shapes = {}
    for key, (x, y, text, color) in nodes.items():
        node = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            x, y, Inches(1.7), Inches(1)
        )
        node.fill.solid()
        node.fill.fore_color.rgb = color
        node.line.color.rgb = COLORS['dark']
        node.line.width = Pt(2)
        
        tf = node.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        node_shapes[key] = (x, y)
    
    # Draw arrows between nodes (simplified visual arrows)
    arrows = [
        ('visceral_fat', 'inflammation', Inches(2.2), Inches(1.95)),
        ('visceral_fat', 'insulin_res', Inches(2.2), Inches(2.7)),
        ('inflammation', 'dyslipidemia', Inches(4.7), Inches(1.55)),
        ('inflammation', 'insulin_res', Inches(3.7), Inches(2.2)),
        ('insulin_res', 'hypertension', Inches(4.7), Inches(3.5)),
        ('insulin_res', 'diabetes', Inches(7), Inches(2.75)),
        ('dyslipidemia', 'diabetes', Inches(7), Inches(2.0)),
        ('hypertension', 'diabetes', Inches(7), Inches(3.0)),
    ]
    
    # Draw arrow shapes
    for src, dst, x, y in arrows:
        arrow = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_ARROW,
            x, y, Inches(0.5), Inches(0.2)
        )
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = COLORS['gray']
        arrow.line.fill.background()
        
        # Rotate some arrows based on direction
        if 'inflammation' in src and 'insulin' in dst:
            arrow.rotation = 45
        elif 'dyslipidemia' in src or 'hypertension' in src:
            arrow.rotation = -30 if 'dyslipidemia' in src else 30
    
    # Legend box
    legend = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(4.3), Inches(4), Inches(1.2)
    )
    legend.fill.solid()
    legend.fill.fore_color.rgb = COLORS['light_gray']
    legend.line.fill.background()
    
    tf = legend.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "🔬 DAG enables:"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    for item in ["• Root cause identification", "• Intervention targeting", "• Cascade effect prediction"]:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(11)
        p.font.color.rgb = COLORS['dark']

def create_risk_algorithm_slide(prs):
    """Slide 7: Risk Algorithm - Visual Formula Flow"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "RISK SCORING ALGORITHM"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Visual formula: boxes showing flow
    # Base Score + Interactions + Evidence Weight = Final Score
    
    formula_elements = [
        ("BASE\nSCORE", "Individual\nbiomarker\ndeviations", COLORS['secondary']),
        ("+", "", COLORS['dark']),
        ("INTERACTION\nFACTOR", "Causal\nrelationship\nweights", COLORS['tier1']),
        ("+", "", COLORS['dark']),
        ("EVIDENCE\nWEIGHT", "Study\nquality\ntiers", COLORS['tier2']),
        ("=", "", COLORS['dark']),
        ("RISK\nSCORE", "0-100\nscale", COLORS['highlight']),
    ]
    
    x = Inches(0.2)
    for main_text, sub_text, color in formula_elements:
        if main_text in ["+", "="]:
            # Operator
            op = slide.shapes.add_textbox(x, Inches(1.5), Inches(0.4), Inches(0.8))
            tf = op.text_frame
            p = tf.paragraphs[0]
            p.text = main_text
            p.font.size = Pt(36)
            p.font.bold = True
            p.font.color.rgb = COLORS['dark']
            p.alignment = PP_ALIGN.CENTER
            x += Inches(0.5)
        else:
            # Box
            box = slide.shapes.add_shape(
                MSO_SHAPE.ROUNDED_RECTANGLE,
                x, Inches(1.2), Inches(1.8), Inches(1.4)
            )
            box.fill.solid()
            box.fill.fore_color.rgb = color
            box.line.fill.background()
            
            tf = box.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.text = main_text
            p.font.size = Pt(12)
            p.font.bold = True
            p.font.color.rgb = COLORS['white']
            p.alignment = PP_ALIGN.CENTER
            
            if sub_text:
                p2 = tf.add_paragraph()
                p2.text = sub_text
                p2.font.size = Pt(9)
                p2.font.color.rgb = COLORS['white']
                p2.alignment = PP_ALIGN.CENTER
            
            x += Inches(2)
    
    # Risk level interpretation
    levels = [
        ("LOW", "0-30", COLORS['tier1']),
        ("MODERATE", "31-60", COLORS['tier2']),
        ("HIGH", "61-100", COLORS['highlight']),
    ]
    
    for i, (level, range_text, color) in enumerate(levels):
        # Bar segment
        bar = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.5 + i * 3), Inches(3.2), Inches(2.8), Inches(0.6)
        )
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        
        tf = bar.text_frame
        p = tf.paragraphs[0]
        p.text = f"{level}: {range_text}"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
    
    # Features at bottom
    features = ["Bayesian Updates", "Temporal Tracking", "Confidence Intervals", "Personalized Baselines"]
    for i, feature in enumerate(features):
        feat_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(0.3 + i * 2.35), Inches(4.2), Inches(2.2), Inches(0.5)
        )
        feat_box.fill.solid()
        feat_box.fill.fore_color.rgb = COLORS['accent1']
        feat_box.line.fill.background()
        
        tf = feat_box.text_frame
        p = tf.paragraphs[0]
        p.text = f"✓ {feature}"
        p.font.size = Pt(11)
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER

def create_pattern_detection_slide(prs):
    """Slide 8: Pattern Detection - Icons in a Row"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "PATTERN DETECTION ENGINE"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # 5 pattern types as icons
    patterns = [
        ("🔥", "Metabolic\nSyndrome", "5 criteria\nscreening", COLORS['highlight']),
        ("💔", "Cardiovascular\nRisk", "Lipid +\nBP analysis", COLORS['tier1']),
        ("🔬", "Insulin\nResistance", "HOMA-IR\ncalculation", COLORS['tier2']),
        ("⚠️", "Inflammation\nCascade", "CRP + IL-6\npattern", COLORS['tier3']),
        ("🧬", "Nutrient\nDeficiency", "Multi-marker\nanalysis", COLORS['secondary']),
    ]
    
    for i, (icon, name, desc, color) in enumerate(patterns):
        x = Inches(0.3 + i * 1.9)
        
        # Large icon circle
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            x + Inches(0.35), Inches(1.2), Inches(1.2), Inches(1.2)
        )
        circle.fill.solid()
        circle.fill.fore_color.rgb = color
        circle.line.color.rgb = COLORS['white']
        circle.line.width = Pt(3)
        
        # Icon text
        icon_text = slide.shapes.add_textbox(x + Inches(0.35), Inches(1.4), Inches(1.2), Inches(0.8))
        tf = icon_text.text_frame
        p = tf.paragraphs[0]
        p.text = icon
        p.font.size = Pt(36)
        p.alignment = PP_ALIGN.CENTER
        
        # Name
        name_box = slide.shapes.add_textbox(x, Inches(2.5), Inches(1.9), Inches(0.8))
        tf = name_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = name
        p.font.size = Pt(12)
        p.font.bold = True
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER
        
        # Description
        desc_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x + Inches(0.1), Inches(3.35), Inches(1.7), Inches(0.6)
        )
        desc_box.fill.solid()
        desc_box.fill.fore_color.rgb = COLORS['light_gray']
        desc_box.line.fill.background()
        
        tf = desc_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(9)
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER
    
    # Bottom: How patterns work
    flow_box = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(4.2), Inches(9.2), Inches(1)
    )
    flow_box.fill.solid()
    flow_box.fill.fore_color.rgb = COLORS['primary']
    flow_box.line.fill.background()
    
    # Flow arrows inside
    flow_items = ["Input Data", "→", "Pattern Match", "→", "Risk Flag", "→", "Action Priority"]
    flow_text = "   ".join(flow_items)
    
    tf = flow_box.text_frame
    p = tf.paragraphs[0]
    p.text = "📊 Detection Flow:  " + flow_text
    p.font.size = Pt(14)
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER

def create_results_slide(prs):
    """Slide 9: Results - Bar Chart"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "PILOT RESULTS: Key Improvements"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Create chart
    chart_data = CategoryChartData()
    chart_data.categories = ['HbA1c', 'LDL-C', 'Triglycerides', 'CRP', 'Blood Pressure']
    chart_data.add_series('Baseline', (8.2, 145, 220, 4.5, 145))
    chart_data.add_series('3-Month Follow-up', (6.8, 98, 155, 1.8, 128))
    
    x, y, cx, cy = Inches(0.5), Inches(0.9), Inches(6), Inches(3.5)
    chart = slide.shapes.add_chart(
        XL_CHART_TYPE.COLUMN_CLUSTERED, x, y, cx, cy, chart_data
    ).chart
    
    # Style the chart
    plot = chart.plots[0]
    plot.has_data_labels = True
    
    # Right side: Key metrics
    metrics = [
        ("📉", "-17%", "HbA1c Reduction"),
        ("📉", "-32%", "LDL-C Reduction"),
        ("📉", "-60%", "CRP Reduction"),
        ("📈", "95%", "Compliance Rate"),
    ]
    
    for i, (icon, value, label) in enumerate(metrics):
        y_pos = Inches(1 + i * 1.1)
        
        # Metric box
        box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            Inches(6.8), y_pos, Inches(2.8), Inches(0.95)
        )
        box.fill.solid()
        box.fill.fore_color.rgb = COLORS['tier1'] if i < 3 else COLORS['highlight']
        box.line.fill.background()
        
        tf = box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = f"{icon} {value}"
        p.font.size = Pt(20)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        p2 = tf.add_paragraph()
        p2.text = label
        p2.font.size = Pt(11)
        p2.font.color.rgb = COLORS['white']
        p2.alignment = PP_ALIGN.CENTER
    
    # Sample size note
    note = slide.shapes.add_textbox(Inches(0.5), Inches(4.5), Inches(6), Inches(0.5))
    tf = note.text_frame
    p = tf.paragraphs[0]
    p.text = "n=47 participants | 3-month intervention period"
    p.font.size = Pt(11)
    p.font.italic = True
    p.font.color.rgb = COLORS['gray']

def create_case_study_slide(prs):
    """Slide 10: Case Study - Patient Journey Timeline"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "CASE STUDY: Patient Journey"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Patient profile box
    profile = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(0.85), Inches(2.5), Inches(1.2)
    )
    profile.fill.solid()
    profile.fill.fore_color.rgb = COLORS['secondary']
    profile.line.fill.background()
    
    tf = profile.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = "👤 Patient Profile"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    
    for item in ["Male, 52 years", "BMI: 31.2", "Pre-diabetic"]:
        p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(10)
        p.font.color.rgb = COLORS['white']
    
    # Timeline
    timeline_y = Inches(2.5)
    
    # Timeline base line
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.5), timeline_y + Inches(0.35), Inches(9), Inches(0.08)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = COLORS['gray']
    line.line.fill.background()
    
    # Timeline points
    timeline_points = [
        ("Week 0", "Initial\nAnalysis", "Risk: 78/100", COLORS['highlight']),
        ("Week 4", "Intervention\nStarted", "Diet + Exercise", COLORS['secondary']),
        ("Week 8", "Progress\nCheck", "Risk: 52/100", COLORS['tier2']),
        ("Week 12", "Final\nReview", "Risk: 34/100", COLORS['tier1']),
    ]
    
    for i, (time, event, detail, color) in enumerate(timeline_points):
        x = Inches(0.5 + i * 2.4)
        
        # Circle marker
        marker = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            x + Inches(0.25), timeline_y + Inches(0.15), Inches(0.5), Inches(0.5)
        )
        marker.fill.solid()
        marker.fill.fore_color.rgb = color
        marker.line.color.rgb = COLORS['white']
        marker.line.width = Pt(2)
        
        # Time label above
        time_box = slide.shapes.add_textbox(x, timeline_y - Inches(0.4), Inches(1), Inches(0.35))
        tf = time_box.text_frame
        p = tf.paragraphs[0]
        p.text = time
        p.font.size = Pt(11)
        p.font.bold = True
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER
        
        # Event box below
        event_box = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x - Inches(0.1), timeline_y + Inches(0.75), Inches(1.2), Inches(0.9)
        )
        event_box.fill.solid()
        event_box.fill.fore_color.rgb = COLORS['light_gray']
        event_box.line.color.rgb = color
        event_box.line.width = Pt(2)
        
        tf = event_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = event
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = COLORS['dark']
        p.alignment = PP_ALIGN.CENTER
        
        p2 = tf.add_paragraph()
        p2.text = detail
        p2.font.size = Pt(9)
        p2.font.color.rgb = color
        p2.alignment = PP_ALIGN.CENTER
    
    # Outcome box
    outcome = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(4.2), Inches(9.2), Inches(1)
    )
    outcome.fill.solid()
    outcome.fill.fore_color.rgb = COLORS['tier1']
    outcome.line.fill.background()
    
    tf = outcome.text_frame
    p = tf.paragraphs[0]
    p.text = "🎯 OUTCOME: Risk Score 78 → 34  |  HbA1c 7.2% → 5.9%  |  No progression to T2DM"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER

def create_validation_roadmap_slide(prs):
    """Slide 11: Validation Roadmap - 3-Phase Timeline"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.2), Inches(9), Inches(0.6))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "VALIDATION ROADMAP"
    p.font.size = Pt(28)
    p.font.bold = True
    p.font.color.rgb = COLORS['primary']
    
    # Three phases as connected chevrons
    phases = [
        ("PHASE 1", "2024", "Retrospective\nValidation", ["n=500 records", "Sensitivity/Specificity", "Calibration testing"], COLORS['secondary']),
        ("PHASE 2", "2025", "Prospective\nPilot", ["n=200 patients", "Real-world testing", "6-month follow-up"], COLORS['tier1']),
        ("PHASE 3", "2026", "Clinical\nTrial", ["Multi-center RCT", "Regulatory pathway", "Publication"], COLORS['highlight']),
    ]
    
    for i, (phase, year, title_text, items, color) in enumerate(phases):
        x = Inches(0.3 + i * 3.2)
        
        # Phase chevron/arrow
        chevron = slide.shapes.add_shape(
            MSO_SHAPE.CHEVRON,
            x, Inches(1), Inches(3), Inches(1)
        )
        chevron.fill.solid()
        chevron.fill.fore_color.rgb = color
        chevron.line.fill.background()
        
        tf = chevron.text_frame
        p = tf.paragraphs[0]
        p.text = f"{phase}: {year}"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = COLORS['white']
        p.alignment = PP_ALIGN.CENTER
        
        # Content box below
        content = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x + Inches(0.1), Inches(2.2), Inches(2.8), Inches(2.2)
        )
        content.fill.solid()
        content.fill.fore_color.rgb = COLORS['light_gray']
        content.line.color.rgb = color
        content.line.width = Pt(2)
        
        tf = content.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p.alignment = PP_ALIGN.CENTER
        
        for item in items:
            p = tf.add_paragraph()
            p.text = f"✓ {item}"
            p.font.size = Pt(10)
            p.font.color.rgb = COLORS['dark']
            p.space_before = Pt(4)
    
    # Current status indicator
    status = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.3), Inches(4.6), Inches(3), Inches(0.6)
    )
    status.fill.solid()
    status.fill.fore_color.rgb = COLORS['tier1']
    status.line.fill.background()
    
    tf = status.text_frame
    p = tf.paragraphs[0]
    p.text = "📍 Current: Phase 1 In Progress"
    p.font.size = Pt(12)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER

def create_conclusion_slide(prs):
    """Slide 12: Conclusion - Key Takeaways"""
    slide_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(slide_layout)
    
    # Background
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, prs.slide_width, prs.slide_height)
    bg.fill.solid()
    bg.fill.fore_color.rgb = COLORS['primary']
    bg.line.fill.background()
    
    # Title
    title = slide.shapes.add_textbox(Inches(0.5), Inches(0.3), Inches(9), Inches(0.7))
    tf = title.text_frame
    p = tf.paragraphs[0]
    p.text = "KEY TAKEAWAYS"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER
    
    # 4 key points as cards
    takeaways = [
        ("🧬", "Evidence-Based", "180+ biomarkers with\ngraded evidence levels"),
        ("🔗", "Causal Modeling", "DAG-based relationship\nmapping for root causes"),
        ("🎯", "Personalized", "Individual risk scoring\nwith actionable plans"),
        ("📊", "Validated", "Pilot data shows\nsignificant improvements"),
    ]
    
    for i, (icon, title_text, desc) in enumerate(takeaways):
        row = i // 2
        col = i % 2
        x = Inches(0.5 + col * 4.7)
        y = Inches(1.3 + row * 1.7)
        
        # Card
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE,
            x, y, Inches(4.3), Inches(1.5)
        )
        card.fill.solid()
        card.fill.fore_color.rgb = COLORS['white']
        card.line.fill.background()
        
        # Icon
        icon_box = slide.shapes.add_textbox(x + Inches(0.2), y + Inches(0.2), Inches(0.8), Inches(0.8))
        tf = icon_box.text_frame
        p = tf.paragraphs[0]
        p.text = icon
        p.font.size = Pt(36)
        
        # Title
        title_box = slide.shapes.add_textbox(x + Inches(1), y + Inches(0.15), Inches(3), Inches(0.5))
        tf = title_box.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = COLORS['primary']
        
        # Description
        desc_box = slide.shapes.add_textbox(x + Inches(1), y + Inches(0.6), Inches(3.1), Inches(0.8))
        tf = desc_box.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = desc
        p.font.size = Pt(11)
        p.font.color.rgb = COLORS['dark']
    
    # Call to action
    cta = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(1.5), Inches(4.8), Inches(6.8), Inches(0.7)
    )
    cta.fill.solid()
    cta.fill.fore_color.rgb = COLORS['highlight']
    cta.line.fill.background()
    
    tf = cta.text_frame
    p = tf.paragraphs[0]
    p.text = "🚀 Seeking Research Partners & Clinical Collaborators"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = COLORS['white']
    p.alignment = PP_ALIGN.CENTER

def main():
    """Create the presentation"""
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(5.625)  # 16:9 aspect ratio
    
    print("Creating NCD-CIE Visual Presentation...")
    
    create_title_slide(prs)
    print("✓ Title slide")
    
    create_problem_slide(prs)
    print("✓ Problem slide")
    
    create_architecture_slide(prs)
    print("✓ Architecture flowchart")
    
    create_knowledge_base_slide(prs)
    print("✓ Knowledge base grid")
    
    create_evidence_grading_slide(prs)
    print("✓ Evidence grading pyramid")
    
    create_dag_slide(prs)
    print("✓ Causal DAG diagram")
    
    create_risk_algorithm_slide(prs)
    print("✓ Risk algorithm formula")
    
    create_pattern_detection_slide(prs)
    print("✓ Pattern detection icons")
    
    create_results_slide(prs)
    print("✓ Results bar chart")
    
    create_case_study_slide(prs)
    print("✓ Case study timeline")
    
    create_validation_roadmap_slide(prs)
    print("✓ Validation roadmap")
    
    create_conclusion_slide(prs)
    print("✓ Conclusion slide")
    
    # Save
    output_path = '/home/clawdbot/clawd/NCD_CIE_Research_Presentation_Visual.pptx'
    prs.save(output_path)
    print(f"\n✅ Presentation saved to: {output_path}")
    print(f"📊 Total slides: {len(prs.slides)}")

if __name__ == '__main__':
    main()
