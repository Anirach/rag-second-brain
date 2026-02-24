#!/usr/bin/env python3
"""
Academic Research Workflow Automation Setup
Creates automated system for monitoring arXiv feeds, summarizing papers, and organizing research
"""

import json
import os
from datetime import datetime, timedelta

def create_research_config():
    """Create research monitoring configuration"""
    research_config = {
        "user_profile": {
            "name": "Anirach Mingkhwan",
            "title": "University Lecturer & AI Engineer",
            "institution": "FITM, KMUTNB",
            "research_interests": [
                "artificial intelligence",
                "machine learning", 
                "deep learning",
                "transformer architectures",
                "natural language processing",
                "computer vision",
                "neural networks",
                "AI applications in education",
                "large language models",
                "AI engineering"
            ],
            "current_projects": [
                "ClawdBot enterprise AI deployment",
                "Local LLM implementation",
                "AI education tools",
                "Academic research automation"
            ]
        },
        "arxiv_categories": [
            "cs.AI",     # Artificial Intelligence
            "cs.LG",     # Machine Learning  
            "cs.CL",     # Computation and Language (NLP)
            "cs.CV",     # Computer Vision
            "cs.NE",     # Neural and Evolutionary Computing
            "cs.HC",     # Human-Computer Interaction
            "stat.ML"    # Statistics - Machine Learning
        ],
        "keywords": [
            "transformer", "attention mechanism", "large language model", "LLM",
            "GPT", "BERT", "neural network", "deep learning", "machine learning",
            "artificial intelligence", "natural language processing", "NLP",
            "computer vision", "AI education", "educational AI", "teaching AI",
            "prompt engineering", "fine-tuning", "transfer learning",
            "multimodal", "vision-language", "chatbot", "conversational AI"
        ],
        "monitoring_schedule": {
            "daily_check": "08:00",      # 8 AM Bangkok time
            "weekly_digest": "Monday 09:00",  # Monday 9 AM
            "monthly_report": "1st 10:00"     # 1st of month 10 AM
        },
        "output_settings": {
            "papers_per_digest": 10,
            "summary_length": "detailed",  # brief, detailed, comprehensive
            "citation_format": "APA",      # APA, MLA, IEEE
            "save_location": "/home/clawdbot/clawd/research",
            "gdrive_upload": True
        }
    }
    
    # Create research directory structure
    os.makedirs("/home/clawdbot/clawd/research/papers", exist_ok=True)
    os.makedirs("/home/clawdbot/clawd/research/summaries", exist_ok=True)
    os.makedirs("/home/clawdbot/clawd/research/digests", exist_ok=True)
    os.makedirs("/home/clawdbot/clawd/research/citations", exist_ok=True)
    
    # Save configuration
    with open('/home/clawdbot/clawd/research/config.json', 'w') as f:
        json.dump(research_config, f, indent=2)
    
    print("✅ Research automation configuration created")
    return research_config

def create_paper_monitor_script():
    """Create script to monitor arXiv for new papers"""
    monitor_script = '''#!/usr/bin/env python3
"""
ArXiv Paper Monitor - Daily research feed checker
"""

import json
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import os
import sys

def load_config():
    """Load research configuration"""
    with open('/home/clawdbot/clawd/research/config.json', 'r') as f:
        return json.load(f)

def search_arxiv_papers(categories, keywords, days_back=1):
    """Search arXiv for relevant papers"""
    papers = []
    
    # Calculate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    for category in categories:
        # ArXiv API query
        query = f"cat:{category}"
        base_url = "http://export.arxiv.org/api/query"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 50,
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(base_url, params=params)
            root = ET.fromstring(response.content)
            
            # Parse results
            for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
                paper = {}
                paper["id"] = entry.find("{http://www.w3.org/2005/Atom}id").text
                paper["title"] = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
                paper["summary"] = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
                paper["published"] = entry.find("{http://www.w3.org/2005/Atom}published").text
                paper["updated"] = entry.find("{http://www.w3.org/2005/Atom}updated").text
                paper["category"] = category
                
                # Extract authors
                authors = []
                for author in entry.findall("{http://www.w3.org/2005/Atom}author"):
                    name = author.find("{http://www.w3.org/2005/Atom}name").text
                    authors.append(name)
                paper["authors"] = authors
                
                # Check if paper matches keywords
                text_to_check = (paper["title"] + " " + paper["summary"]).lower()
                if any(keyword.lower() in text_to_check for keyword in keywords):
                    # Check if paper is within date range
                    pub_date = datetime.fromisoformat(paper["published"].replace("Z", "+00:00"))
                    if pub_date >= start_date:
                        papers.append(paper)
                        
        except Exception as e:
            print(f"Error fetching papers for category {category}: {e}")
    
    return papers

def analyze_paper_relevance(paper, user_interests, current_projects):
    """Analyze how relevant a paper is to user interests"""
    relevance_score = 0
    relevance_reasons = []
    
    title_lower = paper["title"].lower()
    summary_lower = paper["summary"].lower()
    text = title_lower + " " + summary_lower
    
    # Check research interests
    for interest in user_interests:
        if interest.lower() in text:
            relevance_score += 2
            relevance_reasons.append(f"Matches research interest: {interest}")
    
    # Check current projects  
    for project in current_projects:
        project_words = project.lower().split()
        if any(word in text for word in project_words if len(word) > 3):
            relevance_score += 3
            relevance_reasons.append(f"Relevant to project: {project}")
    
    return relevance_score, relevance_reasons

def generate_paper_summary(paper, relevance_reasons):
    """Generate structured summary of paper"""
    return {
        "arxiv_id": paper["id"].split("/")[-1],
        "title": paper["title"],
        "authors": ", ".join(paper["authors"][:3]) + ("..." if len(paper["authors"]) > 3 else ""),
        "category": paper["category"],
        "published": paper["published"][:10],
        "summary": paper["summary"][:500] + ("..." if len(paper["summary"]) > 500 else ""),
        "relevance_reasons": relevance_reasons,
        "url": paper["id"]
    }

def main():
    """Main monitoring function"""
    config = load_config()
    
    print(f"🔍 Searching arXiv for papers relevant to {config['user_profile']['name']}")
    
    # Search for papers
    papers = search_arxiv_papers(
        config["arxiv_categories"], 
        config["keywords"],
        days_back=1
    )
    
    print(f"📄 Found {len(papers)} potentially relevant papers")
    
    # Analyze relevance and create summaries
    relevant_papers = []
    for paper in papers:
        score, reasons = analyze_paper_relevance(
            paper,
            config["user_profile"]["research_interests"],
            config["user_profile"]["current_projects"]
        )
        
        if score >= 2:  # Minimum relevance threshold
            summary = generate_paper_summary(paper, reasons)
            summary["relevance_score"] = score
            relevant_papers.append(summary)
    
    # Sort by relevance score
    relevant_papers.sort(key=lambda x: x["relevance_score"], reverse=True)
    
    print(f"⭐ {len(relevant_papers)} highly relevant papers identified")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/home/clawdbot/clawd/research/papers/papers_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump({
            "timestamp": timestamp,
            "user_profile": config["user_profile"]["name"],
            "total_papers_found": len(papers),
            "relevant_papers_count": len(relevant_papers),
            "relevant_papers": relevant_papers
        }, f, indent=2)
    
    print(f"💾 Results saved to {filename}")
    
    # Return for further processing
    return relevant_papers

if __name__ == "__main__":
    main()
'''
    
    with open('/home/clawdbot/clawd/research/monitor_papers.py', 'w') as f:
        f.write(monitor_script)
    
    os.chmod('/home/clawdbot/clawd/research/monitor_papers.py', 0o755)
    print("✅ Paper monitoring script created")

def create_digest_generator():
    """Create weekly research digest generator"""
    digest_script = '''#!/usr/bin/env python3
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
        citation = f"{authors}, \\"{title},\\" arXiv:{arxiv_id}, {year}."
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
'''
    
    with open('/home/clawdbot/clawd/research/generate_digest.py', 'w') as f:
        f.write(digest_script)
    
    os.chmod('/home/clawdbot/clawd/research/generate_digest.py', 0o755)
    print("✅ Digest generator script created")

def main():
    """Set up complete research automation system"""
    print("🚀 Setting up Academic Research Workflow Automation")
    print("=" * 60)
    
    # Create configuration
    config = create_research_config()
    
    # Create monitoring script
    create_paper_monitor_script()
    
    # Create digest generator
    create_digest_generator()
    
    print("\n✅ Research automation system setup complete!")
    print("\n📋 System Components Created:")
    print("  • Research configuration (config.json)")
    print("  • Daily paper monitor (monitor_papers.py)")
    print("  • Weekly digest generator (generate_digest.py)")
    print("  • Organized folder structure")
    
    print(f"\n📁 Files saved to: /home/clawdbot/clawd/research/")
    
    return config

if __name__ == "__main__":
    main()