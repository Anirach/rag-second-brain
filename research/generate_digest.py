#!/usr/bin/env python3
"""
Research Digest Generator - Creates weekly summary of relevant papers
"""

import json
import glob
import os
from datetime import datetime, timedelta
import sys
sys.path.append('/home/clawdbot/clawd/professional-docx-generator')
from scripts.docx_generator import ProfessionalDocumentGenerator

def load_recent_papers(days_back=7):
    """Load papers from the last week"""
    all_papers = []
    cutoff_date = datetime.now() - timedelta(days=days_back)
    
    paper_files = glob.glob('/home/clawdbot/clawd/research/papers/papers_*.json')
    
    for file_path in paper_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check if file is within date range
            timestamp = datetime.strptime(data['timestamp'], '%Y%m%d_%H%M%S')
            if timestamp >= cutoff_date:
                all_papers.extend(data['relevant_papers'])
                
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return all_papers

def generate_citation(paper, format_style="APA"):
    """Generate formatted citation"""
    authors = paper["authors"]
    title = paper["title"]
    year = paper["published"][:4]
    arxiv_id = paper["arxiv_id"]
    
    if format_style == "APA":
        citation = f"{authors} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
    elif format_style == "IEEE":
        citation = f"{authors}, \"{title},\" arXiv:{arxiv_id}, {year}."
    else:  # Default to APA
        citation = f"{authors} ({year}). {title}. arXiv preprint arXiv:{arxiv_id}."
    
    return citation

def create_research_digest():
    """Generate comprehensive research digest"""
    papers = load_recent_papers(7)
    
    if not papers:
        print("📭 No relevant papers found in the last week")
        return None
    
    # Sort papers by relevance score
    papers.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
    
    # Take top papers for digest
    top_papers = papers[:10]
    
    # Create professional document
    generator = ProfessionalDocumentGenerator()
    
    # Add cover page
    week_start = (datetime.now() - timedelta(days=7)).strftime('%B %d')
    week_end = datetime.now().strftime('%B %d, %Y')
    
    generator.add_cover_page(
        title="Weekly Research Digest",
        subtitle=f"AI & Machine Learning Research Update: {week_start} - {week_end}",
        author="Anirach Mingkhwan",
        date=datetime.now().strftime('%B %d, %Y'),
        organization="FITM, KMUTNB"
    )
    
    # Executive summary
    summary_text = f"This weekly digest summarizes {len(top_papers)} highly relevant papers from arXiv across artificial intelligence, machine learning, and related fields. Papers are selected based on relevance to current research interests and ongoing projects, with a focus on practical applications and novel methodologies."
    generator.add_executive_summary(summary_text)
    
    # Key findings
    key_findings = []
    categories = set([paper.get('category', '') for paper in top_papers])
    for category in categories:
        cat_papers = [p for p in top_papers if p.get('category') == category]
        key_findings.append(f"{len(cat_papers)} papers in {category} covering emerging trends and methodologies")
    
    if key_findings:
        generator.add_key_findings(key_findings)
    
    # Paper summaries section
    generator.add_section("Featured Papers", "")
    
    for i, paper in enumerate(top_papers, 1):
        # Paper header
        paper_title = f"{i}. {paper['title']}"
        generator.doc.add_paragraph(paper_title).runs[0].bold = True
        
        # Authors and metadata
        metadata = f"Authors: {paper['authors']} | Category: {paper.get('category', 'N/A')} | Published: {paper['published']} | arXiv:{paper['arxiv_id']}"
        generator.doc.add_paragraph(metadata).runs[0].italic = True
        
        # Summary
        generator.doc.add_paragraph(f"Summary: {paper['summary']}")
        
        # Relevance
        if paper.get('relevance_reasons'):
            relevance_text = "Relevance: " + "; ".join(paper['relevance_reasons'])
            generator.doc.add_paragraph(relevance_text).runs[0].font.color.rgb = generator.doc.styles['Professional Body'].font.color.rgb
        
        # URL
        generator.doc.add_paragraph(f"URL: {paper['url']}")
        
        # Citation
        citation = generate_citation(paper)
        cite_para = generator.doc.add_paragraph(f"Citation: {citation}")
        cite_para.runs[0].font.size = generator.doc.styles['Professional Body'].font.size
        
        generator.doc.add_paragraph("")  # Spacing
    
    # Citations section
    generator.add_section("Complete Bibliography", "")
    
    for i, paper in enumerate(top_papers, 1):
        citation = generate_citation(paper)
        generator.doc.add_paragraph(f"{i}. {citation}")
    
    # Save digest
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"/home/clawdbot/clawd/research/digests/research_digest_{timestamp}.docx"
    
    generator.save(filename)
    print(f"📊 Research digest created: {filename}")
    
    return filename

def main():
    """Generate weekly research digest"""
    print("📚 Generating weekly research digest...")
    digest_file = create_research_digest()
    
    if digest_file:
        print(f"✅ Weekly digest generated: {digest_file}")
    else:
        print("❌ No digest generated - insufficient papers")

if __name__ == "__main__":
    main()
