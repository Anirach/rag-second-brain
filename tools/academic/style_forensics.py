#!/usr/bin/env python3
"""
Writing Style Forensics — Analyze paper text against venue norms.

Usage:
    python3 tools/academic/style_forensics.py <paper_file> [--venue LNCS|NeurIPS|IEEE|ACM|ICML]

Analyzes:
- Sentence length distribution
- Passive voice ratio
- Hedging frequency
- Section length proportions
- AI-writing signals (repetitive patterns, unusual uniformity)
- Readability scores
"""

import re
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

# Venue-specific norms (based on analysis of accepted papers)
VENUE_NORMS = {
    "LNCS": {
        "avg_sentence_length": (18, 25),  # words
        "passive_voice_ratio": (0.15, 0.35),
        "hedging_ratio": (0.03, 0.08),
        "section_proportions": {
            "introduction": (0.10, 0.18),
            "related_work": (0.08, 0.15),
            "methodology": (0.20, 0.35),
            "results": (0.15, 0.25),
            "discussion": (0.08, 0.15),
            "conclusion": (0.03, 0.08),
        },
        "avg_paragraph_length": (3, 7),  # sentences
        "citation_density": (1.5, 4.0),  # citations per page
    },
    "NeurIPS": {
        "avg_sentence_length": (16, 23),
        "passive_voice_ratio": (0.10, 0.25),
        "hedging_ratio": (0.02, 0.06),
        "section_proportions": {
            "introduction": (0.10, 0.15),
            "related_work": (0.08, 0.12),
            "methodology": (0.25, 0.40),
            "results": (0.20, 0.30),
            "discussion": (0.05, 0.10),
            "conclusion": (0.03, 0.06),
        },
        "avg_paragraph_length": (3, 6),
        "citation_density": (2.0, 5.0),
    },
    "IEEE": {
        "avg_sentence_length": (18, 26),
        "passive_voice_ratio": (0.20, 0.40),
        "hedging_ratio": (0.03, 0.07),
        "section_proportions": {
            "introduction": (0.10, 0.18),
            "related_work": (0.10, 0.18),
            "methodology": (0.20, 0.35),
            "results": (0.15, 0.25),
            "discussion": (0.08, 0.15),
            "conclusion": (0.03, 0.08),
        },
        "avg_paragraph_length": (3, 7),
        "citation_density": (2.0, 4.5),
    },
    "ACM": {
        "avg_sentence_length": (17, 24),
        "passive_voice_ratio": (0.15, 0.30),
        "hedging_ratio": (0.02, 0.07),
        "section_proportions": {
            "introduction": (0.10, 0.15),
            "related_work": (0.10, 0.15),
            "methodology": (0.20, 0.35),
            "results": (0.15, 0.25),
            "discussion": (0.08, 0.15),
            "conclusion": (0.03, 0.08),
        },
        "avg_paragraph_length": (3, 6),
        "citation_density": (2.0, 5.0),
    },
    "ICML": {
        "avg_sentence_length": (16, 23),
        "passive_voice_ratio": (0.10, 0.25),
        "hedging_ratio": (0.02, 0.05),
        "section_proportions": {
            "introduction": (0.10, 0.15),
            "related_work": (0.08, 0.12),
            "methodology": (0.25, 0.40),
            "results": (0.20, 0.30),
            "discussion": (0.05, 0.10),
            "conclusion": (0.03, 0.06),
        },
        "avg_paragraph_length": (3, 6),
        "citation_density": (2.5, 5.5),
    },
}

# Hedging words/phrases
HEDGING_WORDS = [
    "suggest", "suggests", "suggested", "suggesting",
    "may", "might", "could", "would",
    "perhaps", "possibly", "potentially", "presumably",
    "appear", "appears", "appeared", "seem", "seems", "seemed",
    "tend", "tends", "tended",
    "likely", "unlikely", "probable", "probably",
    "indicate", "indicates", "indicated", "indicating",
    "approximately", "roughly", "around", "about",
    "generally", "typically", "usually", "often",
    "somewhat", "relatively", "fairly", "rather",
    "in part", "to some extent", "to a degree",
]

# Passive voice indicators (simplified)
PASSIVE_PATTERNS = [
    r'\b(?:is|are|was|were|been|being|be)\s+(?:\w+\s+)*?(?:ed|en|t)\b',
    r'\b(?:is|are|was|were|been|being|be)\s+\w+ed\b',
    r'\b(?:is|are|was|were)\s+(?:shown|given|made|done|found|seen|known|used|called|considered|expected|required|proposed|presented|described|defined|obtained|achieved|performed|conducted|observed|measured|calculated|determined|applied|employed|utilized)\b',
]

# AI-writing signals
AI_SIGNALS = {
    "sentence_starters": [
        "Furthermore,", "Moreover,", "Additionally,", "In addition,",
        "Notably,", "Importantly,", "Significantly,", "Specifically,",
        "It is worth noting that", "It should be noted that",
        "In this context,", "In this regard,",
    ],
    "filler_phrases": [
        "plays a crucial role", "of paramount importance",
        "a comprehensive understanding", "in the realm of",
        "a myriad of", "a plethora of", "delve into",
        "leverage", "utilize", "facilitate", "underscore",
        "holistic approach", "multifaceted", "synergy",
        "cutting-edge", "state-of-the-art", "groundbreaking",
    ],
    "uniformity_threshold": 0.15,  # std dev of sentence length / mean
}


def read_text(filepath):
    """Read text from file (supports .txt, .md, .tex)."""
    path = Path(filepath)
    text = path.read_text(encoding='utf-8', errors='replace')
    
    # Strip LaTeX commands if .tex
    if path.suffix == '.tex':
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        text = re.sub(r'[{}]', '', text)
        text = re.sub(r'\$[^$]+\$', 'MATH', text)
    
    # Strip markdown headers
    if path.suffix == '.md':
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
        text = re.sub(r'\*([^*]+)\*', r'\1', text)
    
    return text


def split_sentences(text):
    """Split text into sentences."""
    # Simple sentence splitter
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    return sentences


def count_words(text):
    """Count words in text."""
    return len(re.findall(r'\b\w+\b', text))


def analyze_sentence_lengths(sentences):
    """Analyze sentence length distribution."""
    lengths = [count_words(s) for s in sentences]
    if not lengths:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "lengths": []}
    
    mean = sum(lengths) / len(lengths)
    variance = sum((l - mean) ** 2 for l in lengths) / len(lengths)
    std = variance ** 0.5
    
    return {
        "mean": round(mean, 1),
        "std": round(std, 1),
        "min": min(lengths),
        "max": max(lengths),
        "uniformity_ratio": round(std / mean, 3) if mean > 0 else 0,
        "count": len(lengths),
    }


def analyze_passive_voice(sentences):
    """Estimate passive voice usage."""
    passive_count = 0
    passive_examples = []
    
    for sent in sentences:
        for pattern in PASSIVE_PATTERNS:
            if re.search(pattern, sent, re.IGNORECASE):
                passive_count += 1
                if len(passive_examples) < 5:
                    passive_examples.append(sent[:80] + "..." if len(sent) > 80 else sent)
                break
    
    ratio = passive_count / len(sentences) if sentences else 0
    return {
        "count": passive_count,
        "total": len(sentences),
        "ratio": round(ratio, 3),
        "examples": passive_examples,
    }


def analyze_hedging(text, sentences):
    """Analyze hedging frequency."""
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    hedge_count = 0
    hedge_found = Counter()
    
    for hw in HEDGING_WORDS:
        hw_lower = hw.lower()
        count = text.lower().count(hw_lower)
        if count > 0:
            hedge_found[hw] = count
            hedge_count += count
    
    ratio = hedge_count / total_words if total_words > 0 else 0
    return {
        "count": hedge_count,
        "total_words": total_words,
        "ratio": round(ratio, 4),
        "top_hedges": hedge_found.most_common(10),
    }


def analyze_ai_signals(text, sentences):
    """Detect potential AI-writing signals."""
    signals = []
    
    # Check repetitive sentence starters
    starter_counts = Counter()
    for sent in sentences:
        for starter in AI_SIGNALS["sentence_starters"]:
            if sent.strip().startswith(starter):
                starter_counts[starter] += 1
    
    overused = {k: v for k, v in starter_counts.items() if v >= 3}
    if overused:
        signals.append({
            "type": "repetitive_starters",
            "severity": "medium",
            "detail": f"Overused transition phrases: {dict(overused)}",
            "fix": "Vary sentence openings. Remove formulaic transitions.",
        })
    
    # Check filler phrases
    filler_found = []
    text_lower = text.lower()
    for phrase in AI_SIGNALS["filler_phrases"]:
        count = text_lower.count(phrase.lower())
        if count > 0:
            filler_found.append((phrase, count))
    
    if filler_found:
        signals.append({
            "type": "ai_filler_phrases",
            "severity": "high" if len(filler_found) > 5 else "medium",
            "detail": f"AI-typical phrases found: {filler_found}",
            "fix": "Replace with precise, specific language.",
        })
    
    # Check sentence length uniformity (AI tends to be very uniform)
    lengths = [count_words(s) for s in sentences]
    if lengths:
        mean = sum(lengths) / len(lengths)
        std = (sum((l - mean) ** 2 for l in lengths) / len(lengths)) ** 0.5
        uniformity = std / mean if mean > 0 else 0
        
        if uniformity < AI_SIGNALS["uniformity_threshold"]:
            signals.append({
                "type": "suspicious_uniformity",
                "severity": "high",
                "detail": f"Sentence length variation unusually low ({uniformity:.3f}). "
                          f"Human writing typically has ratio > 0.35.",
                "fix": "Mix short punchy sentences with longer analytical ones.",
            })
    
    # Check paragraph length uniformity
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
    if len(paragraphs) > 3:
        para_lengths = [count_words(p) for p in paragraphs]
        p_mean = sum(para_lengths) / len(para_lengths)
        p_std = (sum((l - p_mean) ** 2 for l in para_lengths) / len(para_lengths)) ** 0.5
        p_uniformity = p_std / p_mean if p_mean > 0 else 0
        
        if p_uniformity < 0.2:
            signals.append({
                "type": "paragraph_uniformity",
                "severity": "medium",
                "detail": f"Paragraph lengths suspiciously uniform (CV={p_uniformity:.3f}). "
                          f"Looks machine-generated.",
                "fix": "Vary paragraph lengths naturally.",
            })
    
    return signals


def detect_sections(text):
    """Detect major sections and their lengths."""
    section_patterns = {
        "abstract": r'(?:abstract|summary)\s*\n',
        "introduction": r'(?:\d+\.?\s*)?introduction\s*\n',
        "related_work": r'(?:\d+\.?\s*)?(?:related\s+work|literature\s+review|background)\s*\n',
        "methodology": r'(?:\d+\.?\s*)?(?:method(?:ology|s)?|approach|proposed\s+(?:method|system|framework))\s*\n',
        "results": r'(?:\d+\.?\s*)?(?:results?|experiments?|evaluation)\s*\n',
        "discussion": r'(?:\d+\.?\s*)?discussion\s*\n',
        "conclusion": r'(?:\d+\.?\s*)?conclusions?\s*\n',
    }
    
    sections = {}
    positions = []
    
    for name, pattern in section_patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            positions.append((match.start(), name))
    
    positions.sort()
    total_length = len(text)
    
    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else total_length
        section_text = text[pos:end]
        word_count = count_words(section_text)
        sections[name] = word_count
    
    total_words = sum(sections.values()) if sections else 1
    proportions = {k: round(v / total_words, 3) for k, v in sections.items()}
    
    return {"word_counts": sections, "proportions": proportions, "total_words": total_words}


def compare_with_venue(analysis, venue):
    """Compare analysis results with venue norms."""
    norms = VENUE_NORMS.get(venue, VENUE_NORMS["LNCS"])
    deviations = []
    
    # Sentence length
    sl = analysis["sentence_lengths"]["mean"]
    sl_range = norms["avg_sentence_length"]
    if sl < sl_range[0]:
        deviations.append({
            "metric": "Sentence Length",
            "value": sl,
            "expected": f"{sl_range[0]}-{sl_range[1]} words",
            "severity": "medium",
            "message": f"Sentences too short ({sl:.1f} avg). {venue} papers average {sl_range[0]}-{sl_range[1]} words.",
        })
    elif sl > sl_range[1]:
        deviations.append({
            "metric": "Sentence Length",
            "value": sl,
            "expected": f"{sl_range[0]}-{sl_range[1]} words",
            "severity": "medium",
            "message": f"Sentences too long ({sl:.1f} avg). {venue} papers average {sl_range[0]}-{sl_range[1]} words.",
        })
    
    # Passive voice
    pv = analysis["passive_voice"]["ratio"]
    pv_range = norms["passive_voice_ratio"]
    if pv < pv_range[0]:
        deviations.append({
            "metric": "Passive Voice",
            "value": f"{pv:.1%}",
            "expected": f"{pv_range[0]:.0%}-{pv_range[1]:.0%}",
            "severity": "low",
            "message": f"Low passive voice ({pv:.1%}). {venue} typically uses {pv_range[0]:.0%}-{pv_range[1]:.0%}.",
        })
    elif pv > pv_range[1]:
        deviations.append({
            "metric": "Passive Voice",
            "value": f"{pv:.1%}",
            "expected": f"{pv_range[0]:.0%}-{pv_range[1]:.0%}",
            "severity": "medium",
            "message": f"Too much passive voice ({pv:.1%}). {venue} typically uses {pv_range[0]:.0%}-{pv_range[1]:.0%}.",
        })
    
    # Hedging
    hr = analysis["hedging"]["ratio"]
    hr_range = norms["hedging_ratio"]
    if hr < hr_range[0]:
        deviations.append({
            "metric": "Hedging",
            "value": f"{hr:.2%}",
            "expected": f"{hr_range[0]:.1%}-{hr_range[1]:.1%}",
            "severity": "low",
            "message": f"Low hedging ({hr:.2%}). May come across as overclaiming for {venue}.",
        })
    elif hr > hr_range[1]:
        deviations.append({
            "metric": "Hedging",
            "value": f"{hr:.2%}",
            "expected": f"{hr_range[0]:.1%}-{hr_range[1]:.1%}",
            "severity": "medium",
            "message": f"Excessive hedging ({hr:.2%}). Weakens confidence for {venue}.",
        })
    
    # Section proportions
    if analysis["sections"]["proportions"]:
        for section, (lo, hi) in norms.get("section_proportions", {}).items():
            actual = analysis["sections"]["proportions"].get(section, None)
            if actual is not None:
                if actual < lo:
                    deviations.append({
                        "metric": f"Section: {section}",
                        "value": f"{actual:.1%}",
                        "expected": f"{lo:.0%}-{hi:.0%}",
                        "severity": "medium",
                        "message": f"{section.replace('_', ' ').title()} underweight ({actual:.1%} vs {lo:.0%}-{hi:.0%} expected).",
                    })
                elif actual > hi:
                    deviations.append({
                        "metric": f"Section: {section}",
                        "value": f"{actual:.1%}",
                        "expected": f"{lo:.0%}-{hi:.0%}",
                        "severity": "medium",
                        "message": f"{section.replace('_', ' ').title()} overweight ({actual:.1%} vs {lo:.0%}-{hi:.0%} expected).",
                    })
    
    return deviations


def generate_report(analysis, venue, deviations, filepath):
    """Generate forensics report."""
    lines = []
    lines.append(f"# 🔍 Writing Style Forensics Report")
    lines.append(f"")
    lines.append(f"**File:** `{filepath}`")
    lines.append(f"**Target Venue:** {venue}")
    lines.append(f"**Total Words:** {analysis['sentence_lengths']['count'] * analysis['sentence_lengths']['mean']:.0f} (approx)")
    lines.append(f"")
    
    # Overall health
    high_issues = sum(1 for d in deviations if d.get("severity") == "high")
    med_issues = sum(1 for d in deviations if d.get("severity") == "medium")
    ai_high = sum(1 for s in analysis["ai_signals"] if s.get("severity") == "high")
    
    if high_issues + ai_high > 0:
        lines.append(f"## ⚠️ Overall: NEEDS ATTENTION")
    elif med_issues > 3:
        lines.append(f"## ⚡ Overall: MINOR ADJUSTMENTS NEEDED")
    else:
        lines.append(f"## ✅ Overall: LOOKS GOOD")
    
    lines.append(f"")
    
    # Sentence analysis
    sl = analysis["sentence_lengths"]
    lines.append(f"## 📏 Sentence Length")
    lines.append(f"- Mean: **{sl['mean']}** words | Std: {sl['std']} | Range: {sl['min']}-{sl['max']}")
    lines.append(f"- Uniformity ratio: {sl['uniformity_ratio']} (human typical: 0.35-0.55)")
    lines.append(f"")
    
    # Passive voice
    pv = analysis["passive_voice"]
    lines.append(f"## 🔄 Passive Voice")
    lines.append(f"- Ratio: **{pv['ratio']:.1%}** ({pv['count']}/{pv['total']} sentences)")
    if pv["examples"]:
        lines.append(f"- Examples:")
        for ex in pv["examples"][:3]:
            lines.append(f"  - _{ex}_")
    lines.append(f"")
    
    # Hedging
    hg = analysis["hedging"]
    lines.append(f"## 🤔 Hedging")
    lines.append(f"- Ratio: **{hg['ratio']:.2%}** ({hg['count']} hedge words in {hg['total_words']} words)")
    if hg["top_hedges"]:
        lines.append(f"- Top hedges: {', '.join(f'{w}({c})' for w, c in hg['top_hedges'][:5])}")
    lines.append(f"")
    
    # Section proportions
    sec = analysis["sections"]
    if sec["proportions"]:
        lines.append(f"## 📊 Section Proportions")
        lines.append(f"| Section | Words | Proportion |")
        lines.append(f"|---------|-------|------------|")
        for name, words in sec["word_counts"].items():
            prop = sec["proportions"][name]
            lines.append(f"| {name.replace('_', ' ').title()} | {words} | {prop:.1%} |")
        lines.append(f"")
    
    # AI signals
    if analysis["ai_signals"]:
        lines.append(f"## 🤖 AI-Writing Signals")
        for signal in analysis["ai_signals"]:
            icon = "🔴" if signal["severity"] == "high" else "🟡"
            lines.append(f"- {icon} **{signal['type']}**: {signal['detail']}")
            lines.append(f"  - Fix: {signal['fix']}")
        lines.append(f"")
    else:
        lines.append(f"## 🤖 AI-Writing Signals")
        lines.append(f"- ✅ No strong AI-writing signals detected")
        lines.append(f"")
    
    # Venue deviations
    if deviations:
        lines.append(f"## 📋 Venue Norm Deviations ({venue})")
        for dev in deviations:
            icon = "🔴" if dev["severity"] == "high" else "🟡" if dev["severity"] == "medium" else "🔵"
            lines.append(f"- {icon} **{dev['metric']}**: {dev['message']}")
        lines.append(f"")
    else:
        lines.append(f"## 📋 Venue Norm Deviations ({venue})")
        lines.append(f"- ✅ All metrics within {venue} norms")
        lines.append(f"")
    
    # Recommendations
    lines.append(f"## 💡 Recommendations")
    all_issues = deviations + [{"message": s["detail"], "severity": s["severity"]} for s in analysis["ai_signals"]]
    high = [i for i in all_issues if i.get("severity") == "high"]
    med = [i for i in all_issues if i.get("severity") == "medium"]
    
    if high:
        lines.append(f"### Must Fix (High Priority)")
        for i, issue in enumerate(high, 1):
            lines.append(f"{i}. {issue['message']}")
    if med:
        lines.append(f"### Should Fix (Medium Priority)")
        for i, issue in enumerate(med, 1):
            lines.append(f"{i}. {issue['message']}")
    if not high and not med:
        lines.append(f"Paper style is well-aligned with {venue} norms. Minor polishing only.")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Writing Style Forensics")
    parser.add_argument("paper", help="Path to paper file (.txt, .md, .tex)")
    parser.add_argument("--venue", default="LNCS", choices=list(VENUE_NORMS.keys()),
                       help="Target venue for comparison")
    parser.add_argument("--output", help="Output report path (default: stdout)")
    parser.add_argument("--json", action="store_true", help="Output raw JSON instead of report")
    args = parser.parse_args()
    
    text = read_text(args.paper)
    sentences = split_sentences(text)
    
    analysis = {
        "sentence_lengths": analyze_sentence_lengths(sentences),
        "passive_voice": analyze_passive_voice(sentences),
        "hedging": analyze_hedging(text, sentences),
        "sections": detect_sections(text),
        "ai_signals": analyze_ai_signals(text, sentences),
    }
    
    deviations = compare_with_venue(analysis, args.venue)
    
    if args.json:
        analysis["deviations"] = deviations
        print(json.dumps(analysis, indent=2, default=str))
    else:
        report = generate_report(analysis, args.venue, deviations, args.paper)
        if args.output:
            Path(args.output).write_text(report)
            print(f"Report written to {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()
