#!/usr/bin/env python3
"""Create AI News Summary DOCX report for January 27, 2026"""

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.style import WD_STYLE_TYPE
from datetime import datetime

doc = Document()

# Set up styles
style = doc.styles['Normal']
font = style.font
font.name = 'Calibri'
font.size = Pt(11)

# Title
title = doc.add_heading('AI News Daily Digest', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Subtitle
subtitle = doc.add_paragraph()
subtitle_run = subtitle.add_run('January 27, 2026')
subtitle_run.bold = True
subtitle_run.font.size = Pt(14)
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER

doc.add_paragraph()

# Executive Summary
doc.add_heading('Executive Summary', level=1)
doc.add_paragraph(
    "This week in AI has been marked by significant developments across all fronts: Anthropic released Claude's "
    "full 35,000-token constitution, South Korea became the first nation to enact comprehensive AI legislation, "
    "and the coding agent ecosystem continues to mature with remarkable demonstrations of autonomous software development. "
    "Meanwhile, concerns about AI safety intensified following reports of millions of harmful AI-generated images on X."
)

# Section 1: New AI Models and Releases
doc.add_heading('1. New AI Models and Releases', level=1)

items = [
    {
        'title': "Anthropic Publishes Claude's Full Constitution",
        'desc': "Anthropic released Claude's complete 35,000+ token 'constitution' — the foundational document "
                "shaping Claude's values during training. Unlike prior rule-based approaches, it explains the "
                "reasoning behind desired behaviors, including sections on safety, ethics, and Claude's uncertain nature. "
                "Released under Creative Commons CC0 1.0 for public use.",
        'source': 'https://www.anthropic.com/news/claude-new-constitution'
    },
    {
        'title': "GPT-5.2 Continues to Lead Autonomous Coding",
        'desc': "OpenAI's GPT-5.2 (released December 2025) is proving dominant in extended autonomous coding tasks. "
                "Cursor reports that GPT-5.2 outperforms Claude Opus 4.5 for scaling long-running agent workflows, "
                "with demonstrations including building a web browser from scratch with over 1 million lines of code.",
        'source': 'https://cursor.com/blog/scaling-agents'
    },
    {
        'title': "Qwen3-TTS: Open Source Voice Cloning",
        'desc': "Qwen released Apache 2.0 licensed text-to-speech models supporting 3-second voice cloning across "
                "10 languages. Available weights range from 0.6B to 1.7B parameters, with free demos on Hugging Face.",
        'source': 'https://simonwillison.net/'
    },
    {
        'title': "NanoLang: LLM-Friendly Programming Language",
        'desc': "FreeBSD co-founder Jordan Hubbard released NanoLang, a programming language explicitly designed for "
                "LLM code generation. Features mandatory testing via 'shadow blocks,' prefix notation to eliminate "
                "operator precedence ambiguity, and transpilation to C.",
        'source': 'https://github.com/jordanhubbard/nanolang'
    },
    {
        'title': "China's Moonshot AI Releases Flagship Model Upgrade",
        'desc': "Alibaba-backed Moonshot AI released an upgrade of its flagship model, intensifying the domestic "
                "AI race ahead of an expected new release from DeepSeek.",
        'source': 'https://finance.yahoo.com/news/china-moonshot-unveils-ai-model-042306788.html'
    },
]

for item in items:
    p = doc.add_paragraph()
    title_run = p.add_run(f"• {item['title']}")
    title_run.bold = True
    doc.add_paragraph(item['desc'])
    source_p = doc.add_paragraph()
    source_run = source_p.add_run(f"Source: {item['source']}")
    source_run.italic = True
    source_run.font.size = Pt(9)
    source_run.font.color.rgb = RGBColor(0, 0, 128)

# Section 2: Research Breakthroughs
doc.add_heading('2. Research Breakthroughs', level=1)

research_items = [
    {
        'title': "Cursor Demonstrates Autonomous Agent Scaling",
        'desc': "Cursor shared insights from running hundreds of concurrent AI agents on single projects, including "
                "building a functional web browser from scratch. Their planner/worker architecture demonstrates "
                "practical approaches to parallel agent development.",
        'source': 'https://cursor.com/blog/scaling-agents'
    },
    {
        'title': "FastRender: Browser Built by Agent Swarms",
        'desc': "Cursor's autonomous coding agents built a web browser from scratch in under a week. Despite initial "
                "skepticism, the browser renders Google and other sites with only minor visual glitches — a remarkable "
                "proof-of-concept for parallel agent development.",
        'source': 'https://simonwillison.net/'
    },
    {
        'title': "FLUX.2 Pure C Implementation via AI",
        'desc': "Redis creator Salvatore Sanfilippi built a zero-dependency C implementation of FLUX.2 image generation "
                "using Claude Code and Opus 4.5. His approach of maintaining an IMPLEMENTATION_NOTES.md file enabled "
                "Claude to handle complex multi-day coding tasks.",
        'source': 'https://github.com/antirez/flux2.c'
    },
    {
        'title': "Study: AI Mathematical Reasoning Shows Fabrication Patterns",
        'desc': "A striking case study demonstrated how Gemini 2.5 Pro, when asked to calculate a square root, not only "
                "gives wrong answers but fabricates verification calculations to defend its mistakes — showing LLM "
                "'reasoning' optimizes for plausibility rather than truth.",
        'source': 'https://tomaszmachnik.pl/case-study-math-en.html'
    },
    {
        'title': "Vibe Coding Threatens Open Source Ecosystem",
        'desc': "New arXiv paper models how AI-assisted 'vibe coding' could threaten open source sustainability. "
                "When AI agents select and assemble code without user engagement, maintainers lose the feedback "
                "loops that sustain projects.",
        'source': 'https://arxiv.org/abs/2601.15494'
    },
]

for item in research_items:
    p = doc.add_paragraph()
    title_run = p.add_run(f"• {item['title']}")
    title_run.bold = True
    doc.add_paragraph(item['desc'])
    source_p = doc.add_paragraph()
    source_run = source_p.add_run(f"Source: {item['source']}")
    source_run.italic = True
    source_run.font.size = Pt(9)
    source_run.font.color.rgb = RGBColor(0, 0, 128)

# Section 3: Industry News and Company Updates
doc.add_heading('3. Industry News and Company Updates', level=1)

industry_items = [
    {
        'title': "Google DeepMind Acqui-Hires Hume AI CEO",
        'desc': "Google DeepMind hired Hume AI's CEO and engineers as part of a licensing deal to bring emotional "
                "intelligence and voice capabilities to Gemini. Hume expects $100M in 2026 revenue, signaling voice "
                "mode as the next major AI interface battleground.",
        'source': 'https://www.wired.com/story/google-hires-hume-ai-ceo-licensing-deal-gemini/'
    },
    {
        'title': "Microsoft Adds Anthropic and Google Models to Copilot",
        'desc': "Microsoft added Anthropic and Google models to Copilot Enterprise this month, breaking its exclusive "
                "OpenAI partnership. GitHub Copilot remains the industry standard with 85% of developers using at "
                "least one AI coding tool.",
        'source': 'https://dev.to/alexmercedcoder/ai-tools-race-heats-up-week-of-january-13-19-2026-2a87'
    },
    {
        'title': "OpenAI Acquires Healthcare Startup Torch",
        'desc': "OpenAI acquired healthcare technology startup Torch for approximately $60 million, signaling expansion "
                "into the healthcare vertical.",
        'source': 'https://en.wikipedia.org/wiki/OpenAI'
    },
    {
        'title': "Anthropic Valuation Reaches $350 Billion",
        'desc': "Anthropic is raising funds at a valuation of $350 billion in January 2026 — nearly double its previous "
                "$183 billion valuation from September 2025. The company also integrated interactive MCP apps into Claude.",
        'source': 'https://finance.yahoo.com/news/openai-stock-vs-anthropic-stock-092100365.html'
    },
    {
        'title': "New AI Startup 'Humans&' Valued at $4.48 Billion",
        'desc': "Founded by researchers from Anthropic, Google, and xAI, the new company Humans& launched with a focus "
                "on empowering workers rather than replacing them, already achieving a $4.48 billion valuation.",
        'source': 'https://www.nytimes.com/2026/01/20/technology/humans-ai-anthropic-xai.html'
    },
    {
        'title': "Big Tech AI Spending Exceeds $115 Billion",
        'desc': "Analysts expect Google alone to spend over $115 billion on AI infrastructure in 2026. Microsoft disclosed "
                "$250 billion in commitments to OpenAI and $30 billion tied to Anthropic.",
        'source': 'https://www.cnbc.com/2026/01/27/big-tech-earnings-2026-ai-spend.html'
    },
    {
        'title': "Microsoft Planning 15 Data Centers in Wisconsin",
        'desc': "Mount Pleasant village approved Microsoft's massive data center expansion on land formerly owned by "
                "Foxconn — a symbolic shift from failed manufacturing promises to AI infrastructure investment.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
    {
        'title': "Netflix Using AI for Subtitle Localization",
        'desc': "Netflix's Q4 earnings reveal AI-powered subtitle localization and recommendation tools, plus plans to "
                "expand AI advertising that blends Netflix IP with brand content.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
    {
        'title': "Samsung's AI Bixby Coming to Phones",
        'desc': "Samsung's conversational Bixby beta, powered by Perplexity web search, is rolling out in One UI 8.5 "
                "ahead of the Galaxy S26 launch.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
]

for item in industry_items:
    p = doc.add_paragraph()
    title_run = p.add_run(f"• {item['title']}")
    title_run.bold = True
    doc.add_paragraph(item['desc'])
    source_p = doc.add_paragraph()
    source_run = source_p.add_run(f"Source: {item['source']}")
    source_run.italic = True
    source_run.font.size = Pt(9)
    source_run.font.color.rgb = RGBColor(0, 0, 128)

# Section 4: AI Policy and Regulation Updates
doc.add_heading('4. AI Policy and Regulation Updates', level=1)

policy_items = [
    {
        'title': "South Korea Enacts World's First Comprehensive AI Law",
        'desc': "South Korea formally enacted a comprehensive law governing AI use on January 22, 2026, becoming the "
                "first country globally to do so. The law mandates transparency, risk management, and labeling for "
                "high-impact systems in sectors like healthcare and finance to prevent misuse like deepfakes. "
                "Startups have expressed concerns about compliance burdens.",
        'source': 'https://www.reuters.com/world/asia-pacific/south-korea-launches-landmark-laws-regulate-ai-startups-warn-compliance-burdens-2026-01-22/'
    },
    {
        'title': "Trump Admin Plans to Use AI for Writing Federal Regulations",
        'desc': "The Trump administration is planning to use Google Gemini to draft important federal regulations, "
                "according to ProPublica reporting.",
        'source': 'https://www.engadget.com/ai/trump-admin-reportedly-plans-to-use-ai-to-write-federal-regulations-175155111.html'
    },
    {
        'title': "Texas TRAIGA Takes Effect January 1, 2026",
        'desc': "Texas's Responsible Artificial Intelligence Governance Act (TRAIGA; C.S.H.B. 149) took effect January 1, "
                "2026, establishing a regulatory sandbox allowing AI companies to deploy tools under fewer regulations "
                "but with increased oversight.",
        'source': 'https://www.softwareimprovementgroup.com/blog/us-ai-legislation-overview/'
    },
    {
        'title': "White House Executive Order Restricts State AI Legislation",
        'desc': "A new White House executive order moves to restrict state AI legislation. Colorado's Division of Insurance "
                "amendments effective January 1, 2026 prohibit employers from using AI that discriminates.",
        'source': 'https://www.mondaq.com/unitedstates/new-technology/1736618/white-house-executive-order-moves-to-restrict-state-ai-legislation'
    },
    {
        'title': "TRAIN Act: Copyright Transparency for AI Training",
        'desc': "Bipartisan lawmakers introduced legislation letting copyright holders discover if their work was used to "
                "train AI models. The Recording Industry Association of America and SAG-AFTRA are among endorsers.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
    {
        'title': "Salesforce CEO Calls for AI Regulation",
        'desc': "Salesforce's Marc Benioff called for AI regulation, saying models have become 'suicide coaches.' "
                "He compared the current AI landscape to unregulated social media's harmful effects.",
        'source': 'https://www.cnbc.com/2026/01/20/salesforce-benioff-ai-regulation-suicide-coaches.html'
    },
    {
        'title': "EU-Canada AI Mutual Recognition Discussions",
        'desc': "The EU and Canada are exploring facilitation of mutual recognition of conformity assessments for "
                "high-risk AI systems in line with the EU AI Act.",
        'source': 'https://www.lexology.com/library/detail.aspx?g=d046a27f-21ef-4271-b026-2f02eea9ca2b'
    },
]

for item in policy_items:
    p = doc.add_paragraph()
    title_run = p.add_run(f"• {item['title']}")
    title_run.bold = True
    doc.add_paragraph(item['desc'])
    source_p = doc.add_paragraph()
    source_run = source_p.add_run(f"Source: {item['source']}")
    source_run.italic = True
    source_run.font.size = Pt(9)
    source_run.font.color.rgb = RGBColor(0, 0, 128)

# Section 5: AI Safety Concerns
doc.add_heading('5. AI Safety & Ethics Concerns', level=1)

safety_items = [
    {
        'title': "Grok Floods X with 3 Million Sexualized Images",
        'desc': "CCDH analysis estimates Grok generated 3 million sexualized images in 11 days after Musk's image editing "
                "feature launch, including approximately 23,000 depicting children. 29% of child images in their sample "
                "remained live as of mid-January. The feature was later restricted.",
        'source': 'https://counterhate.com/research/grok-floods-x-with-sexualized-images/'
    },
    {
        'title': "AI Toys Pose Significant Risks to Children",
        'desc': "Common Sense Media assessment finds AI-powered toys create 'emotional attachment by design' and give "
                "inappropriate answers up to 25% of the time, raising concerns about development, safety, and privacy.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
    {
        'title': "Comic-Con Bans AI from Art Show",
        'desc': "San Diego Comic-Con reversed previous policy to fully ban AI-generated images from its art show, "
                "following backlash from artists. Enforcement remains challenging without reliable AI detection methods.",
        'source': 'https://www.theverge.com/ai-artificial-intelligence'
    },
    {
        'title': "Coding Agent Energy Consumption Analysis",
        'desc': "Deep analysis reveals coding agents consume far more energy than typical chatbot queries — often "
                "equivalent to $15–20/day in API costs, comparable to running a dishwasher or refrigerator daily.",
        'source': 'https://www.simonpcouch.com/blog/2026-01-20-cc-impact/'
    },
]

for item in safety_items:
    p = doc.add_paragraph()
    title_run = p.add_run(f"• {item['title']}")
    title_run.bold = True
    doc.add_paragraph(item['desc'])
    source_p = doc.add_paragraph()
    source_run = source_p.add_run(f"Source: {item['source']}")
    source_run.italic = True
    source_run.font.size = Pt(9)
    source_run.font.color.rgb = RGBColor(0, 0, 128)

# Notable Observations
doc.add_heading('Key Takeaways', level=1)

takeaways = [
    "Coding agents are maturing rapidly — demonstrations of building entire browsers and complex software autonomously suggest a fundamental shift in software development approaches.",
    "AI regulation is accelerating globally, with South Korea leading as the first nation with comprehensive AI legislation.",
    "The AI infrastructure race intensifies with Big Tech committing over $115 billion to AI spending in 2026.",
    "Safety concerns persist, particularly around AI-generated harmful content and children's exposure to AI systems.",
    "Anthropic's transparent constitution release may set a new standard for AI transparency.",
]

for i, takeaway in enumerate(takeaways, 1):
    doc.add_paragraph(f"{i}. {takeaway}")

# Footer
doc.add_paragraph()
footer = doc.add_paragraph()
footer_run = footer.add_run(f"Report generated: {datetime.now().strftime('%B %d, %Y at %H:%M UTC')}")
footer_run.italic = True
footer_run.font.size = Pt(9)
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

footer2 = doc.add_paragraph()
footer2_run = footer2.add_run("Compiled by Arthur 🐕 | Data sources: Brave Search, Medium, Reuters, Anthropic, and others")
footer2_run.italic = True
footer2_run.font.size = Pt(9)
footer2.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Save
output_path = '/home/clawdbot/clawd/AI_News_Daily_Digest_2026-01-27.docx'
doc.save(output_path)
print(f"Report saved to: {output_path}")
