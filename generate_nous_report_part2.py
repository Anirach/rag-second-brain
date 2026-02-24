#!/usr/bin/env python3
"""
Nous Feature Research Report - Part 2 (continuation)
This script continues from where part 1 left off, adding more sections.
"""

# This is called by the main script after part 1 finishes building the document.
# Import from part 1 is handled by exec chaining.

def add_tier2_continued(doc, add_heading2, add_heading3, add_body, create_feature_table, add_bullet, add_page_break_proper, add_info_box, NAVY, BLUE, DARK_GRAY, GREEN, ORANGE, RED, WHITE, GOLD):
    
    # Feature 8 continued
    feat8_rows = [
        [('What It Does', True, NAVY), ('Auto-generated structured summary card for each paper: (1) One-sentence TLDR, (2) Research question answered, (3) Methodology used, (4) Key finding (3 bullets), (5) Limitations, (6) "Cite when you need to..." guidance', False, DARK_GRAY)],
        [('Why It Matters', True, NAVY), ('Researchers scan 50-100 papers before selecting 20 to read deeply. Structured summaries enable faster triage. Semantic Scholar TLDRs increased paper engagement by 40%.', False, DARK_GRAY)],
        [('Paper Quality Impact', True, NAVY), ('The "cite when" field directly tells a researcher under what conditions to cite this paper, improving citation accuracy and relevance in papers.', False, DARK_GRAY)],
        [('Implementation', True, NAVY), ('Add background LLM call on paper ingestion: generate structured JSON summary, store in PostgreSQL as paper.summary_json. Expose via GET /paper/{id}/summary.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('LOW-MEDIUM — Template-based LLM prompt + storage. ~1-2 days.', False, ORANGE)],
        [('Reference Tools', True, NAVY), ('Semantic Scholar (TLDR), Elicit (structured extraction), Scite.ai (citation context)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], feat8_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature 9
    add_heading3(doc, 'T2.4 — Interactive Seed-Paper Citation Network (Connected Papers-style)')

    feat9_rows = [
        [('What It Does', True, NAVY), ('Starting from any paper in the knowledge base, generate an interactive D3.js/Cytoscape graph showing: (1) Prior foundational papers, (2) Derivative papers that cite it, (3) Temporal layout (x-axis = year), (4) Node size = citation count, (5) Color = research school/theme', False, DARK_GRAY)],
        [('Why It Matters', True, NAVY), ('Discovering the "genealogy" of a research idea helps researchers identify: foundational papers to cite, recent work to differentiate from, and research communities to engage with.', False, DARK_GRAY)],
        [('Paper Quality Impact', True, NAVY), ('Prevents the common reviewer criticism "authors missed seminal work X." Visual citation navigation ensures comprehensive coverage.', False, DARK_GRAY)],
        [('Implementation', True, NAVY), ('Extend /graph/six/citation endpoint: add seed_paper_id parameter. Compute ego network in Neo4j using 2-hop CITES relationships. Return D3 force graph format.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('MEDIUM — Citation graph infrastructure exists. Need ego-network query and enhanced frontend component.', False, ORANGE)],
        [('Reference Tools', True, NAVY), ('Connected Papers (best-in-class), Litmaps, Citation Gecko', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], feat9_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature 10
    add_heading3(doc, 'T2.5 — Zotero / Citation Manager Integration')

    feat10_rows = [
        [('What It Does', True, NAVY), ('(1) One-click export of any paper collection to Zotero via Zotero Web API, (2) Import papers from Zotero collections into Nous, (3) Sync reading status between systems, (4) Generate formatted bibliography in 12+ citation styles (APA, MLA, IEEE, Vancouver, Harvard)', False, DARK_GRAY)],
        [('Why It Matters', True, NAVY), ('Zotero is the #1 citation manager for academics. Without integration, researchers maintain two separate workflows. Integration removes manual copy-paste of citations.', False, DARK_GRAY)],
        [('Paper Quality Impact', True, NAVY), ('Correct citations are the most mechanical part of academic writing. Integration eliminates citation errors, saves ~2-3 hours per paper in citation formatting.', False, DARK_GRAY)],
        [('Implementation', True, NAVY), ('Integrate pyzotero library. POST /export/zotero with Zotero API key. Add citation style rendering via citeproc-py. ~2 days.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('LOW-MEDIUM — Well-defined Zotero API + Python library available.', False, ORANGE)],
        [('Reference Tools', True, NAVY), ('Zotero, Mendeley, ResearchRabbit (Zotero sync)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], feat10_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature 11
    add_heading3(doc, 'T2.6 — Research Question Decomposition & Study Design Assistant')

    feat11_rows = [
        [('What It Does', True, NAVY), ('Given a broad research topic (e.g., "AI for early Alzheimer\'s detection"), automatically: (1) Decompose into 5-7 sub-questions, (2) Identify relevant existing papers per sub-question, (3) Suggest appropriate research methodology, (4) Identify required data sources, (5) Flag ethical considerations for biomedical AI research', False, DARK_GRAY)],
        [('Why It Matters', True, NAVY), ('Writing a research proposal requires structuring vague ideas into concrete questions. This mirrors Elicit\'s most valued feature. For Anirach\'s biomedical AI domain, methodology selection is particularly critical.', False, DARK_GRAY)],
        [('Paper Quality Impact', True, NAVY), ('Well-decomposed research questions = clearer paper structure = stronger contribution statements. Helps write Section 1 (Introduction/Motivation) of papers.', False, DARK_GRAY)],
        [('Implementation', True, NAVY), ('New endpoint: POST /research/decompose. LLM prompt with domain context. For each sub-question, run semantic search to find supporting papers. Return structured JSON with sub-questions, papers, methodology, and data suggestions.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('MEDIUM-HIGH — Requires careful prompt engineering + graph-augmented retrieval. ~3-5 days.', False, RED)],
        [('Reference Tools', True, NAVY), ('Elicit (question decomposition), Consensus.app (evidence synthesis)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], feat11_rows, col_widths=[1.5, 5.0])

    add_page_break_proper(doc)

def add_tier3(doc, add_heading2, add_heading3, add_body, create_feature_table, add_bullet, add_page_break_proper, add_info_box, NAVY, BLUE, DARK_GRAY, GREEN, ORANGE, RED, WHITE, GOLD):

    add_heading2(doc, '5.3 Tier 3 — Nice to Have Features')

    add_body(doc,
        'These features add significant long-term value but are not blockers for core research productivity. '
        'They represent the vision for Nous as a comprehensive research intelligence platform.')

    # Feature T3.1
    add_heading3(doc, 'T3.1 — Multi-Format Export (LaTeX, Obsidian, Notion, Markdown)')

    t3_1_rows = [
        [('What It Does', True, NAVY), ('Export any knowledge artifact (literature review, entity graph, gap analysis) to: LaTeX (.tex with \\cite{}), Obsidian Markdown ([[wikilinks]]), Notion API import, Standard Markdown with backlinks', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('Obsidian is Anirach\'s personal knowledge system. Bidirectional linking between Nous entities and Obsidian notes creates a powerful personal research knowledge base.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('LOW — Format converters are straightforward. Obsidian vault sync ~2 days.', False, GREEN)],
        [('Reference Tools', True, NAVY), ('Obsidian Zotero Plugin, Logseq, Roam Research', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_1_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature T3.2
    add_heading3(doc, 'T3.2 — Structured Data Extraction (Elicit-style Tables)')

    t3_2_rows = [
        [('What It Does', True, NAVY), ('Given a set of papers and researcher-defined columns (e.g., "Dataset used", "Model architecture", "Accuracy reported", "Sample size", "Year"), extract structured data into a comparison table via LLM.', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('This is Elicit\'s signature feature and the single most time-saving function for systematic reviews. Extracting data from 20 papers into a table takes hours manually, minutes automatically.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('HIGH — Requires reliable structured extraction from PDF text with high accuracy. 5-7 days.', False, RED)],
        [('Reference Tools', True, NAVY), ('Elicit (best-in-class), SciSpace (table extraction)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_2_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature T3.3
    add_heading3(doc, 'T3.3 — Claim Verification & Contradiction Detection')

    t3_3_rows = [
        [('What It Does', True, NAVY), ('(1) Given a research claim (e.g., "Transformer attention is equivalent to consciousness"), retrieve papers that support, contradict, or are neutral toward it. (2) Highlight contradictions between papers on the same topic.', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('Academic papers require balanced literature reviews that acknowledge debates. Automatic contradiction detection prevents the "cherry-picking" criticism from reviewers.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('HIGH — Requires NLI (natural language inference) fine-tuning or powerful LLM with structured prompting. 7-10 days.', False, RED)],
        [('Reference Tools', True, NAVY), ('Scite.ai (supporting/contrasting citations), Semantic Scholar (citation context)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_3_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature T3.4
    add_heading3(doc, 'T3.4 — Collaborative Workspace with Shared Collections')

    t3_4_rows = [
        [('What It Does', True, NAVY), ('Multi-user support with: team-shared paper collections, annotation sharing, role-based access (Owner/Editor/Viewer), activity feed, comment threads on papers and entities.', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('University research teams need shared knowledge bases. A lecturer and students working on the same project should share a Nous workspace. This unlocks use in courses and research labs.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('HIGH — Requires full auth system, RBAC, real-time updates. 2-3 weeks for MVP.', False, RED)],
        [('Reference Tools', True, NAVY), ('ResearchRabbit (shared collections), Zotero Groups, Notion (team wikis)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_4_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature T3.5
    add_heading3(doc, 'T3.5 — AI Abstract & Section Writer (Research Writing Assistant)')

    t3_5_rows = [
        [('What It Does', True, NAVY), ('Given: research question + key papers + methodology + results, generate: (1) structured abstract (Background/Methods/Results/Conclusion), (2) Related work section draft with proper in-text citations, (3) Introduction with motivation paragraph, (4) Future work suggestions', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('Writing assistance directly accelerates paper production. For non-native English speakers (common in Thai universities), having a fluent first draft to edit is transformative.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('MEDIUM-HIGH — Build on existing lit review generator. RAG-augmented generation with templates. 5-7 days.', False, RED)],
        [('Reference Tools', True, NAVY), ('Jenni.ai, SciSpace, Elicit (section writing), Semantic Scholar (context-aware writing)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_5_rows, col_widths=[1.5, 5.0])
    doc.add_paragraph()

    # Feature T3.6
    add_heading3(doc, 'T3.6 — Research Trend Analysis & Field Trajectory Forecasting')

    t3_6_rows = [
        [('What It Does', True, NAVY), ('Temporal analysis of the knowledge base: (1) Topic popularity over time (paper count by year per entity/concept), (2) Emerging topics (high recent growth rate), (3) Declining topics (fewer recent papers), (4) Citation velocity curves, (5) Prediction of 2025-2026 hot topics using trend extrapolation', False, DARK_GRAY)],
        [('Research Impact', True, NAVY), ('Choosing research topics with good timing maximizes impact and publication chances. Papers on trending topics get more citations. Declining fields have fewer high-impact venues.', False, DARK_GRAY)],
        [('Complexity', True, NAVY), ('MEDIUM — Year data exists in PostgreSQL. Need time-series aggregation + visualization. ~3 days.', False, ORANGE)],
        [('Reference Tools', True, NAVY), ('Semantic Scholar (citation trends), Dimensions.ai, Scimago (field rankings)', False, DARK_GRAY)],
    ]
    create_feature_table(doc, ['Attribute', 'Details'], t3_6_rows, col_widths=[1.5, 5.0])


def add_quick_wins(doc, add_heading1, add_heading2, add_heading3, add_body, create_feature_table, add_bullet, add_page_break_proper, add_info_box, NAVY, BLUE, DARK_GRAY, GREEN, ORANGE, RED, WHITE, GOLD):

    add_page_break_proper(doc)
    add_heading1(doc, '6. Quick Wins (< 1 Day Implementation)')

    add_body(doc,
        'These are high-impact fixes that can be implemented within a single working day. They remove '
        'immediate blockers and deliver visible value to researchers without requiring architectural changes.')

    qw_headers = ['#', 'Quick Win', 'Problem Fixed', 'Implementation', 'Estimated Time']
    qw_rows = [
        [
            ('QW1', True, NAVY),
            ('Fix /api/proxy/ask Route', True, NAVY),
            ('Frontend Q&A completely broken — 404 error', False, RED),
            ('Create web/app/api/proxy/ask/route.ts → forward to FastAPI /ask endpoint with proper headers and body forwarding', False, DARK_GRAY),
            ('2-3 hours', False, GREEN)
        ],
        [
            ('QW2', True, NAVY),
            ('Add TLDR to Paper Detail View', True, NAVY),
            ('Researchers must read abstract to understand relevance', False, ORANGE),
            ('Add LLM call in paper ingestion pipeline: generate 1-2 sentence TLDR, store in papers.tldr column. Display on paper cards.', False, DARK_GRAY),
            ('3-4 hours', False, GREEN)
        ],
        [
            ('QW3', True, NAVY),
            ('Fix Entity Type Deduplication', True, NAVY),
            ('"researcher" vs "researchers" creates split entity types, degrading graph quality', False, ORANGE),
            ('SQL normalization query: UPDATE entities SET entity_type = \'researcher\' WHERE entity_type = \'researchers\'. Repeat for other duplicates.', False, DARK_GRAY),
            ('1-2 hours', False, GREEN)
        ],
        [
            ('QW4', True, NAVY),
            ('Add Processing Status Dashboard', True, NAVY),
            ('No visibility into the 1,199 pending papers — pipeline appears stalled', False, ORANGE),
            ('Create /admin/pipeline-status page showing queue depth, processing rate, error count. Use existing /stats data.', False, DARK_GRAY),
            ('3-4 hours', False, GREEN)
        ],
        [
            ('QW5', True, NAVY),
            ('Add BibTeX/APA Export Button to Search Results', True, NAVY),
            ('Citation export exists but is hidden — no UI access', False, ORANGE),
            ('Add "Export Citations" button to paper list/search results UI. Call /export/bibtex?ids=paper1,paper2. Copy to clipboard or download.', False, DARK_GRAY),
            ('2-3 hours', False, GREEN)
        ],
        [
            ('QW6', True, NAVY),
            ('Add Gap Hypothesis Text to Gap Detection', True, NAVY),
            ('Gap detection returns entity pairs with no actionable guidance', False, ORANGE),
            ('Add LLM call to /gaps endpoint: for each gap, generate 2-sentence hypothesis statement. Add gap_hypothesis field to response.', False, DARK_GRAY),
            ('3-4 hours', False, GREEN)
        ],
    ]
    create_feature_table(doc, qw_headers, qw_rows, col_widths=[0.5, 1.5, 1.5, 2.2, 0.9])

    add_info_box(doc, 'Quick Win Priority Order',
        '1. QW1 (Fix proxy) → CRITICAL BLOCKER must go first\n'
        '2. QW3 (Deduplicate entities) → Improves all downstream graph quality\n'
        '3. QW6 (Gap hypotheses) → Immediately makes gap detection actionable for research papers\n'
        '4. QW5 (Export UI) → Removes friction from citation workflow\n'
        '5. QW2 (TLDR), QW4 (Pipeline dashboard) → Nice usability improvements')


def add_roadmap(doc, add_heading1, add_heading2, add_heading3, add_body, create_feature_table, add_bullet, add_page_break_proper, add_info_box, NAVY, BLUE, DARK_GRAY, GREEN, ORANGE, RED, WHITE, GOLD, RGBColor, Pt):

    add_page_break_proper(doc)
    add_heading1(doc, '7. Implementation Roadmap')

    add_body(doc,
        'The following phased roadmap prioritizes features that unlock maximum research productivity for '
        'Anirach\'s specific research domains: AI/ML, education technology, longevity healthcare, and '
        'biomedical AI. Each phase builds on the previous, with no phase requiring future phases to function.')

    add_heading2(doc, 'Phase 1 — Foundation Fixes (Week 1-2)')

    add_body(doc, 'Goal: Fix critical blockers, complete vectorization pipeline, establish reliable core workflow.')

    p1_headers = ['Task', 'Priority', 'Effort', 'Owner', 'Success Metric']
    p1_rows = [
        [('Fix frontend proxy /api/proxy/ask', False, DARK_GRAY), ('P0 — Blocker', True, RED), ('0.5 days', False, DARK_GRAY), ('Frontend Dev', False, DARK_GRAY), ('Q&A accessible via web UI', False, GREEN)],
        [('Deduplicate entity types in Neo4j', False, DARK_GRAY), ('P0', True, RED), ('0.25 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Single entity_type per concept', False, GREEN)],
        [('Add streaming to /ask (SSE)', False, DARK_GRAY), ('P1', True, ORANGE), ('1 day', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('First token within 500ms', False, GREEN)],
        [('Backfill 1,199 pending papers', False, DARK_GRAY), ('P1', True, ORANGE), ('0.5 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('< 50 pending papers', False, GREEN)],
        [('Add pipeline status endpoint + UI', False, DARK_GRAY), ('P2', True, GOLD), ('0.5 days', False, DARK_GRAY), ('Full-Stack', False, DARK_GRAY), ('Admin can see processing status', False, GREEN)],
        [('Add gap hypothesis LLM call', False, DARK_GRAY), ('P1', True, ORANGE), ('0.5 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Each gap has hypothesis text', False, GREEN)],
        [('Citation export button in UI', False, DARK_GRAY), ('P2', True, GOLD), ('0.5 days', False, DARK_GRAY), ('Frontend Dev', False, DARK_GRAY), ('One-click BibTeX copy/download', False, GREEN)],
    ]
    create_feature_table(doc, p1_headers, p1_rows, col_widths=[2.2, 1.0, 0.7, 0.8, 1.8])

    add_heading2(doc, 'Phase 2 — Research Productivity Core (Week 3-6)')

    add_body(doc, 'Goal: Transform Nous into a tool researchers actively use for paper writing and literature review.')

    p2_rows = [
        [('Multi-domain paper ingestion API', False, DARK_GRAY), ('P1', True, ORANGE), ('3 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Add paper via DOI/arXiv ID', False, GREEN)],
        [('PDF upload + auto-processing', False, DARK_GRAY), ('P1', True, ORANGE), ('2 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('PDF → entities + vectors in <5 min', False, GREEN)],
        [('Lit review LaTeX/Word export', False, DARK_GRAY), ('P1', True, ORANGE), ('2 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Download .tex and .docx from UI', False, GREEN)],
        [('Per-paper TLDR generation', False, DARK_GRAY), ('P2', True, GOLD), ('1.5 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('All papers have TLDR in UI', False, GREEN)],
        [('Zotero integration (export)', False, DARK_GRAY), ('P1', True, ORANGE), ('2 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Papers sync to Zotero library', False, GREEN)],
        [('Basic auth system (JWT)', False, DARK_GRAY), ('P1', True, ORANGE), ('3 days', False, DARK_GRAY), ('Full-Stack', False, DARK_GRAY), ('User login, personal collections', False, GREEN)],
        [('Paper annotation system', False, DARK_GRAY), ('P2', True, GOLD), ('2 days', False, DARK_GRAY), ('Full-Stack', False, DARK_GRAY), ('Notes and tags per paper per user', False, GREEN)],
        [('Interactive citation network', False, DARK_GRAY), ('P2', True, GOLD), ('3 days', False, DARK_GRAY), ('Full-Stack', False, DARK_GRAY), ('Click paper → see citation graph', False, GREEN)],
    ]
    create_feature_table(doc, p1_headers, p2_rows, col_widths=[2.2, 1.0, 0.7, 0.8, 1.8])

    add_heading2(doc, 'Phase 3 — Research Intelligence (Week 7-12)')

    add_body(doc, 'Goal: Add AI-powered features that actively assist research discovery, hypothesis generation, and paper writing.')

    p3_rows = [
        [('New paper alerts (Sem. Scholar feed)', False, DARK_GRAY), ('P1', True, ORANGE), ('3 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Daily digest via email/Telegram', False, GREEN)],
        [('Research question decomposition', False, DARK_GRAY), ('P2', True, GOLD), ('4 days', False, DARK_GRAY), ('AI Eng.', False, DARK_GRAY), ('Topic → sub-questions + papers', False, GREEN)],
        [('Research trend analysis', False, DARK_GRAY), ('P2', True, GOLD), ('3 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Timeline viz of topic popularity', False, GREEN)],
        [('AI abstract/section writer', False, DARK_GRAY), ('P2', True, GOLD), ('5 days', False, DARK_GRAY), ('AI Eng.', False, DARK_GRAY), ('Generate structured abstract draft', False, GREEN)],
        [('Structured data extraction tables', False, DARK_GRAY), ('P2', True, GOLD), ('7 days', False, DARK_GRAY), ('AI Eng.', False, DARK_GRAY), ('N papers → comparison table', False, GREEN)],
        [('Obsidian vault export', False, DARK_GRAY), ('P3', True, RGBColor(0x44, 0x44, 0x44)), ('2 days', False, DARK_GRAY), ('Backend Dev', False, DARK_GRAY), ('Export entities as [[wikilinks]]', False, GREEN)],
        [('Contradiction/debate detection', False, DARK_GRAY), ('P3', True, RGBColor(0x44, 0x44, 0x44)), ('7 days', False, DARK_GRAY), ('AI Eng.', False, DARK_GRAY), ('Papers flagged as contradicting', False, GREEN)],
        [('Team collaboration workspace', False, DARK_GRAY), ('P3', True, RGBColor(0x44, 0x44, 0x44)), ('14 days', False, DARK_GRAY), ('Full-Stack', False, DARK_GRAY), ('Shared collections + annotations', False, GREEN)],
    ]
    create_feature_table(doc, p1_headers, p3_rows, col_widths=[2.2, 1.0, 0.7, 0.8, 1.8])

    add_heading2(doc, 'Cumulative Impact by Phase')

    impact_headers = ['Phase', 'Timeline', 'Research Productivity Impact', 'Paper Quality Impact']
    impact_rows = [
        [('Phase 1', True, NAVY), ('Weeks 1-2', False, DARK_GRAY), 
         ('Q&A functional; gap detection actionable; faster search; all 2,688 papers searchable', False, DARK_GRAY),
         ('Better literature coverage; hypothesis-driven gap analysis', False, DARK_GRAY)],
        [('Phase 2', True, NAVY), ('Weeks 3-6', False, DARK_GRAY),
         ('Can add any paper; generate and export lit reviews; Zotero sync; personal annotations; visual citation maps', False, DARK_GRAY),
         ('Complete citation management; submission-ready lit review drafts; no missed seminal papers', False, DARK_GRAY)],
        [('Phase 3', True, NAVY), ('Weeks 7-12', False, DARK_GRAY),
         ('Proactive paper alerts; AI-assisted writing; trend awareness; structured data extraction; team collaboration', False, DARK_GRAY),
         ('Current literature always up-to-date; AI-drafted paper sections; quantitative comparisons; reviewer-ready balance', False, DARK_GRAY)],
    ]
    create_feature_table(doc, impact_headers, impact_rows, col_widths=[0.8, 1.0, 2.7, 2.0])


def add_conclusion(doc, add_heading1, add_heading2, add_heading3, add_body, add_bullet, add_page_break_proper, add_info_box, NAVY, BLUE, DARK_GRAY, GREEN, ORANGE, RED, Pt, WD_ALIGN_PARAGRAPH, RGBColor):
    from docx.shared import Pt
    from docx.enum.text import WD_ALIGN_PARAGRAPH

    add_page_break_proper(doc)
    add_heading1(doc, '8. Conclusion')

    add_body(doc,
        'Nous represents a technically sophisticated and architecturally sound research intelligence system '
        'with genuine competitive advantages. Its graph-aware RAG combining Neo4j knowledge graphs with '
        'Qdrant vector search is more powerful than what any single commercial tool offers. Its 6-type '
        'graph visualization suite and automatic theory comparison are genuinely unique capabilities.')

    add_body(doc,
        'However, the system currently fails at the research workflow integration level. The broken frontend '
        'proxy, incomplete vectorization, absent citation manager integration, and missing paper ingestion '
        'interface collectively prevent researchers from using it as their primary tool — despite the '
        'underlying technology being superior.')

    add_body(doc,
        'The path forward is clear and executable. The Quick Wins (Section 6) can be completed within '
        'one working day and will immediately unblock the core Q&A workflow, improve graph quality, '
        'and make gap detection actionable. Phase 1 (2 weeks) brings the system to reliable parity '
        'with basic research productivity expectations. Phase 2 (6 weeks) makes Nous the researcher\'s '
        'primary tool for literature review and citation management. Phase 3 (12 weeks) establishes '
        'Nous as an AI research partner that proactively discovers knowledge, identifies opportunities, '
        'and assists in writing.')

    add_heading2(doc, 'Strategic Recommendations')

    strategic_points = [
        ('Immediate Priority: Fix the Proxy', 
         'The 404 on /api/proxy/ask is a trust-destroying bug. Fix it today. Every hour it stays broken is a researcher who gives up and opens Google Scholar instead.'),
        ('Domain Expansion is Critical for Anirach', 
         'The current corpus covers consciousness research. Anirach\'s actual work spans AI/ML, education technology, longevity, and biomedical AI. Multi-domain ingestion (arXiv, PubMed, bioRxiv) must be implemented in Phase 2 to make Nous relevant to ongoing research.'),
        ('Leverage the Unique Knowledge Graph Advantage', 
         'No commercial tool has Neo4j + Qdrant + hypothesis generation. This is Nous\'s competitive moat. The gap detection → hypothesis generation → research question decomposition pipeline should be the flagship differentiator — explicitly marketed as "AI-powered research opportunity discovery."'),
        ('Zotero Integration Opens the Academic Market', 
         'Zotero is used by virtually every academic researcher. Integration (even just export) immediately makes Nous compatible with existing workflows and removes the most common "I\'d use this but..." objection.'),
        ('Streaming Responses are Non-Negotiable', 
         'In 2025-2026, researchers expect real-time AI responses. Synchronous wait for RAG answers feels archaic and makes the system feel slow even when it is performing within spec. SSE streaming should be implemented before any user-facing launch.'),
    ]

    for title, content in strategic_points:
        add_bullet(doc, content, bold_prefix=title)

    add_heading2(doc, 'Expected Outcomes After Full Implementation')

    outcomes_para = doc.add_paragraph()
    outcomes_para.paragraph_format.space_before = Pt(8)
    outcomes_para.paragraph_format.space_after = Pt(8)
    outcomes_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY

    outcome_items = [
        ('Literature Review Time', '4-6 hours → 30-45 minutes', 'via automated generation + Zotero export'),
        ('Citation Coverage', '70-80% typical recall → 95%+ recall', 'via complete vectorization + multi-domain ingestion'),
        ('Research Opportunity Discovery', 'Manual literature scan (days) → automated gap + hypothesis (minutes)', 'via enhanced gap detection'),
        ('Paper Writing Speed', 'Baseline → 40-60% faster first draft', 'via section writer + annotation integration'),
        ('Citation Accuracy', 'Manual copy-paste errors → near-zero errors', 'via Zotero integration'),
    ]

    # Summary table
    from docx.shared import Inches
    outcome_table = doc.add_table(rows=1 + len(outcome_items), cols=3)
    outcome_table.style = 'Table Grid'
    
    from docx.oxml.ns import qn as oxml_qn
    from docx.oxml import OxmlElement as OXmlEl
    
    hdr = outcome_table.rows[0]
    for cell, text in zip(hdr.cells, ['Research Activity', 'Time/Quality Change', 'Mechanism']):
        p = cell.paragraphs[0]
        p.clear()
        r = p.add_run(text)
        r.font.name = 'Arial'
        r.font.size = Pt(10)
        r.font.bold = True
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OXmlEl('w:shd')
        shd.set(oxml_qn('w:val'), 'clear')
        shd.set(oxml_qn('w:color'), 'auto')
        shd.set(oxml_qn('w:fill'), '1B3A5C')
        tcPr.append(shd)

    for ri, (activity, change, mechanism) in enumerate(outcome_items):
        row = outcome_table.rows[ri + 1]
        bg = 'F0F4F8' if ri % 2 == 0 else 'FFFFFF'
        for cell in row.cells:
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            shd = OXmlEl('w:shd')
            shd.set(oxml_qn('w:val'), 'clear')
            shd.set(oxml_qn('w:color'), 'auto')
            shd.set(oxml_qn('w:fill'), bg)
            tcPr.append(shd)
        
        for cell, text, clr in zip(row.cells, [activity, change, mechanism], 
                                    [NAVY, GREEN, DARK_GRAY]):
            p = cell.paragraphs[0]
            p.clear()
            r = p.add_run(text)
            r.font.name = 'Arial'
            r.font.size = Pt(10)
            r.font.bold = (clr == NAVY)
            r.font.color.rgb = clr

    for row in outcome_table.rows:
        row.cells[0].width = Inches(1.8)
        row.cells[1].width = Inches(2.5)
        row.cells[2].width = Inches(2.2)

    doc.add_paragraph()

    # Final statement
    final_para = doc.add_paragraph()
    final_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    final_para.paragraph_format.space_before = Pt(20)
    final_para.paragraph_format.space_after = Pt(8)
    r_final = final_para.add_run(
        '"Nous has the architecture of a world-class research intelligence system.\n'
        'The features described in this report will give it the soul of one."')
    r_final.font.name = 'Arial'
    r_final.font.size = Pt(12)
    r_final.font.italic = True
    r_final.font.color.rgb = NAVY

    sign_para = doc.add_paragraph()
    sign_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sign = sign_para.add_run(
        f'— Nous Feature Research Report | {__import__("datetime").datetime.now().strftime("%B %Y")}')
    r_sign.font.name = 'Arial'
    r_sign.font.size = Pt(10)
    r_sign.font.color.rgb = RGBColor(0x88, 0x88, 0x88)
