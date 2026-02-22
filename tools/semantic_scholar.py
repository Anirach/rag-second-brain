#!/usr/bin/env python3
"""
Semantic Scholar API Helper
Free tier - no API key needed for basic queries
https://api.semanticscholar.org/
"""

import requests
import json
from typing import Optional, List, Dict
import time

BASE_URL = "https://api.semanticscholar.org/graph/v1"

def search_papers(
    query: str,
    limit: int = 10,
    fields: str = "title,authors,year,abstract,citationCount,url,openAccessPdf"
) -> List[Dict]:
    """
    Search for papers by query string.
    
    Args:
        query: Search query
        limit: Max results (default 10, max 100)
        fields: Comma-separated fields to return
    
    Returns:
        List of paper dictionaries
    """
    url = f"{BASE_URL}/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": fields
    }
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return data.get("data", [])
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return []

def get_paper(paper_id: str, fields: str = "title,authors,year,abstract,citationCount,references,citations") -> Optional[Dict]:
    """
    Get details for a specific paper by ID.
    
    Args:
        paper_id: Semantic Scholar paper ID, DOI, or arXiv ID
        fields: Comma-separated fields
    
    Returns:
        Paper dictionary or None
    """
    url = f"{BASE_URL}/paper/{paper_id}"
    params = {"fields": fields}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_author(author_id: str, fields: str = "name,affiliations,paperCount,citationCount,hIndex") -> Optional[Dict]:
    """
    Get author details by ID.
    """
    url = f"{BASE_URL}/author/{author_id}"
    params = {"fields": fields}
    
    response = requests.get(url, params=params)
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

def search_by_topic(topic: str, year_from: int = None, limit: int = 20) -> List[Dict]:
    """
    Search papers by topic with optional year filter.
    """
    query = topic
    if year_from:
        query = f"{topic} year:{year_from}-"
    
    return search_papers(query, limit=limit)

def format_paper(paper: Dict) -> str:
    """Format a paper for display."""
    title = paper.get("title", "Unknown")
    year = paper.get("year", "?")
    citations = paper.get("citationCount", 0)
    authors = paper.get("authors", [])
    author_names = ", ".join([a.get("name", "?") for a in authors[:3]])
    if len(authors) > 3:
        author_names += " et al."
    
    url = paper.get("url", "")
    pdf = paper.get("openAccessPdf", {})
    pdf_url = pdf.get("url", "") if pdf else ""
    
    output = f"**{title}** ({year})\n"
    output += f"  Authors: {author_names}\n"
    output += f"  Citations: {citations}\n"
    if url:
        output += f"  URL: {url}\n"
    if pdf_url:
        output += f"  PDF: {pdf_url}\n"
    
    abstract = paper.get("abstract", "")
    if abstract:
        output += f"  Abstract: {abstract[:200]}...\n"
    
    return output

# CLI interface
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python semantic_scholar.py <search query>")
        print("Example: python semantic_scholar.py 'longevity interventions'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    print(f"Searching for: {query}\n")
    
    papers = search_papers(query, limit=5)
    
    if not papers:
        print("No papers found.")
    else:
        print(f"Found {len(papers)} papers:\n")
        for i, paper in enumerate(papers, 1):
            print(f"{i}. {format_paper(paper)}")
