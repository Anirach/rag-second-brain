#!/usr/bin/env python3
"""
Automated Related Work Positioning Map Generator

Generates a Mermaid diagram showing where a paper sits relative to existing work.
Takes a paper's key terms + related work references and creates a visual positioning map.

Usage:
    python3 tools/academic/positioning_map.py --title "Our Paper" --config positioning.json --output map.md

Config JSON format:
{
    "our_paper": {
        "title": "Our Paper Title",
        "contributions": ["KG + RAG", "Ontology mapping", "Clinical decision support"],
        "axes": {
            "x": {"label": "Approach", "position": "hybrid"},
            "y": {"label": "Domain", "position": "clinical"}
        }
    },
    "related_works": [
        {
            "id": "GraphRAG",
            "title": "GraphRAG (Microsoft, 2024)",
            "similarities": ["Graph-based retrieval"],
            "differences": ["No ontology", "General purpose"],
            "threat_level": "low"
        },
        ...
    ],
    "dimensions": ["methodology", "domain", "scale", "novelty"]
}
"""

import json
import sys
import argparse
from pathlib import Path
from datetime import datetime


def generate_mermaid_quadrant(config):
    """Generate a Mermaid quadrant chart positioning map."""
    our = config["our_paper"]
    related = config.get("related_works", [])
    
    # Build quadrant chart
    lines = []
    lines.append("```mermaid")
    lines.append("quadrantChart")
    lines.append(f'    title Related Work Positioning: {our["title"][:50]}')
    
    axes = our.get("axes", {})
    x_label = axes.get("x", {}).get("label", "Technical Novelty")
    y_label = axes.get("y", {}).get("label", "Domain Specificity")
    
    lines.append(f'    x-axis "Low {x_label}" --> "High {x_label}"')
    lines.append(f'    y-axis "Low {y_label}" --> "High {y_label}"')
    
    lines.append('    quadrant-1 "Our Target Zone"')
    lines.append('    quadrant-2 "Domain Leaders"')
    lines.append('    quadrant-3 "General Purpose"')
    lines.append('    quadrant-4 "Technical Leaders"')
    
    # Position our paper
    lines.append(f'    "⭐ {our["title"][:30]}": [0.85, 0.85]')
    
    # Position related works based on threat level
    for i, work in enumerate(related):
        threat = work.get("threat_level", "medium")
        # Assign positions based on characteristics
        x = work.get("x", 0.3 + (i * 0.1) % 0.5)
        y = work.get("y", 0.2 + (i * 0.15) % 0.6)
        lines.append(f'    "{work["id"]}": [{x:.2f}, {y:.2f}]')
    
    lines.append("```")
    return "\n".join(lines)


def generate_comparison_table(config):
    """Generate a detailed comparison table."""
    our = config["our_paper"]
    related = config.get("related_works", [])
    dimensions = config.get("dimensions", ["methodology", "domain", "scale", "evaluation"])
    
    lines = []
    lines.append("## Detailed Comparison Matrix")
    lines.append("")
    
    # Header
    header = "| Aspect |"
    separator = "|--------|"
    for work in related:
        header += f" {work['id']} |"
        separator += "---------|"
    header += f" **Ours** |"
    separator += "---------|"
    
    lines.append(header)
    lines.append(separator)
    
    # Rows for similarities/differences
    for dim in dimensions:
        row = f"| {dim.title()} |"
        for work in related:
            cell = work.get("details", {}).get(dim, "—")
            row += f" {cell} |"
        our_val = our.get("details", {}).get(dim, "✅ Novel")
        row += f" **{our_val}** |"
        lines.append(row)
    
    return "\n".join(lines)


def generate_threat_analysis(config):
    """Analyze which related works pose the biggest differentiation challenge."""
    related = config.get("related_works", [])
    
    lines = []
    lines.append("## Differentiation Threat Analysis")
    lines.append("")
    lines.append("| Work | Threat | Similarities | Key Differentiator |")
    lines.append("|------|--------|-------------|-------------------|")
    
    for work in related:
        threat = work.get("threat_level", "medium")
        icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(threat, "⚪")
        sims = ", ".join(work.get("similarities", []))
        diffs = work.get("key_differentiator", ", ".join(work.get("differences", [])))
        lines.append(f"| {work['id']} | {icon} {threat.upper()} | {sims} | {diffs} |")
    
    return "\n".join(lines)


def generate_novelty_map(config):
    """Generate a contribution-level novelty mapping."""
    our = config["our_paper"]
    contributions = our.get("contributions", [])
    related = config.get("related_works", [])
    
    lines = []
    lines.append("## Contribution Novelty Map")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    lines.append(f'    OUR["⭐ {our["title"][:40]}"]')
    
    for i, contrib in enumerate(contributions):
        node_id = f"C{i}"
        lines.append(f'    OUR --> {node_id}["{contrib}"]')
        lines.append(f'    style {node_id} fill:#2ecc71,color:#fff')
    
    for work in related:
        w_id = work["id"].replace(" ", "_").replace("-", "_")
        lines.append(f'    {w_id}["{work["id"]}"]')
        for sim in work.get("similarities", []):
            # Find matching contribution
            for i, contrib in enumerate(contributions):
                if any(word.lower() in contrib.lower() for word in sim.lower().split()):
                    lines.append(f'    {w_id} -.->|"partial overlap"| C{i}')
                    break
        
        threat = work.get("threat_level", "medium")
        color = {"high": "#e74c3c", "medium": "#f39c12", "low": "#3498db"}.get(threat, "#95a5a6")
        lines.append(f'    style {w_id} fill:{color},color:#fff')
    
    lines.append("```")
    return "\n".join(lines)


def generate_full_report(config):
    """Generate the complete positioning map report."""
    our = config["our_paper"]
    
    lines = []
    lines.append(f"# 🗺️ Related Work Positioning Map")
    lines.append(f"")
    lines.append(f"**Paper:** {our['title']}")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"**Contributions:** {', '.join(our.get('contributions', []))}")
    lines.append(f"")
    
    # Quadrant chart
    lines.append("## Positioning Quadrant")
    lines.append("")
    lines.append(generate_mermaid_quadrant(config))
    lines.append("")
    
    # Threat analysis
    lines.append(generate_threat_analysis(config))
    lines.append("")
    
    # Novelty map
    lines.append(generate_novelty_map(config))
    lines.append("")
    
    # Comparison table (if details provided)
    if any("details" in w for w in config.get("related_works", [])):
        lines.append(generate_comparison_table(config))
        lines.append("")
    
    # Recommendations
    high_threats = [w for w in config.get("related_works", []) if w.get("threat_level") == "high"]
    
    lines.append("## 💡 Positioning Recommendations")
    lines.append("")
    
    if high_threats:
        lines.append("### ⚠️ High-Threat Works (Must Differentiate)")
        for work in high_threats:
            lines.append(f"- **{work['id']}**: {', '.join(work.get('differences', ['Needs analysis']))}")
            lines.append(f"  - Recommended: Explicitly cite and differentiate in Related Work section")
    
    lines.append("")
    lines.append("### Suggested Related Work Framing")
    lines.append("")
    lines.append("Use this structure in your Related Work section:")
    lines.append("1. **General category** — Cite foundational works")
    lines.append("2. **Closest approaches** — Cite and differentiate from high-threat works")
    lines.append("3. **Our positioning** — Clear statement of what's new")
    lines.append("")
    lines.append("### \"How is this different from X?\" Preemptive Answers")
    lines.append("")
    
    for work in config.get("related_works", []):
        if work.get("threat_level") in ("high", "medium"):
            diffs = work.get("differences", ["To be analyzed"])
            lines.append(f"**Q: How is this different from {work['id']}?**")
            lines.append(f"A: Unlike {work['id']}, our approach {'; '.join(diffs[:2])}.")
            lines.append("")
    
    return "\n".join(lines)


def create_sample_config():
    """Create a sample configuration file."""
    sample = {
        "our_paper": {
            "title": "RAG Second Brain with KG and Ontology",
            "contributions": [
                "Co-occurrence KG construction",
                "Ontology-guided retrieval",
                "Clinical decision support via RAG"
            ],
            "axes": {
                "x": {"label": "Technical Novelty", "position": "high"},
                "y": {"label": "Domain Specificity", "position": "high"}
            },
            "details": {
                "methodology": "KG + Ontology + RAG fusion",
                "domain": "Clinical healthcare",
                "scale": "Production-ready",
                "evaluation": "Multi-benchmark"
            }
        },
        "related_works": [
            {
                "id": "GraphRAG",
                "title": "GraphRAG (Microsoft, 2024)",
                "similarities": ["Graph-based retrieval", "LLM integration"],
                "differences": ["No ontology mapping", "General purpose, not clinical"],
                "key_differentiator": "We add ontology-guided retrieval for clinical specificity",
                "threat_level": "high",
                "x": 0.75,
                "y": 0.30,
                "details": {
                    "methodology": "Community-based graph summarization",
                    "domain": "General",
                    "scale": "Large-scale",
                    "evaluation": "Single benchmark"
                }
            },
            {
                "id": "MedRAG",
                "title": "MedRAG (2024)",
                "similarities": ["Medical domain", "RAG pipeline"],
                "differences": ["No knowledge graph", "Retrieval only, no reasoning"],
                "key_differentiator": "We combine KG structure with retrieval for deeper reasoning",
                "threat_level": "medium",
                "x": 0.40,
                "y": 0.70,
                "details": {
                    "methodology": "Chunk-based medical RAG",
                    "domain": "Medical",
                    "scale": "Research prototype",
                    "evaluation": "Medical QA"
                }
            },
            {
                "id": "KGQA",
                "title": "KG-based QA Systems",
                "similarities": ["Knowledge graph", "Question answering"],
                "differences": ["No RAG component", "Structured queries only"],
                "key_differentiator": "We bridge KG and unstructured retrieval",
                "threat_level": "low",
                "x": 0.60,
                "y": 0.50,
                "details": {
                    "methodology": "SPARQL-based KG querying",
                    "domain": "Various",
                    "scale": "Research",
                    "evaluation": "KG benchmarks"
                }
            }
        ],
        "dimensions": ["methodology", "domain", "scale", "evaluation"]
    }
    return sample


def main():
    parser = argparse.ArgumentParser(description="Related Work Positioning Map Generator")
    parser.add_argument("--config", help="Path to positioning config JSON")
    parser.add_argument("--output", help="Output report path (default: stdout)")
    parser.add_argument("--sample", action="store_true", help="Generate sample config")
    parser.add_argument("--title", help="Paper title (for inline use)")
    args = parser.parse_args()
    
    if args.sample:
        sample = create_sample_config()
        output_path = args.output or "positioning_config_sample.json"
        Path(output_path).write_text(json.dumps(sample, indent=2))
        print(f"Sample config written to {output_path}")
        return
    
    if not args.config:
        print("Error: --config required (or use --sample to generate template)")
        sys.exit(1)
    
    config = json.loads(Path(args.config).read_text())
    report = generate_full_report(config)
    
    if args.output:
        Path(args.output).write_text(report)
        print(f"Positioning map written to {args.output}")
    else:
        print(report)


if __name__ == "__main__":
    main()
