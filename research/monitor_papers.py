#!/usr/bin/env python3
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
