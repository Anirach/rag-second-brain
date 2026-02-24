#!/usr/bin/env python3
"""Part 2: Sections 11-16 + footer + save"""

# This is appended to build_memory_report.py logic
# Import and continue from the doc object

def build_sections_11_to_16(doc):
    from docx.shared import Pt, RGBColor, Inches, Cm
    from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
    from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
    from docx.oxml.ns import qn
    from docx.oxml import OxmlElement

    NAVY      = RGBColor(0x1B, 0x3A, 0x5C)
    BLUE      = RGBColor(0x2A, 0x64, 0x96)
    DARKGRAY  = RGBColor(0x2C, 0x3E, 0x50)
    BODY_COLOR= RGBColor(0x33, 0x33, 0x33)
    WHITE     = RGBColor(0xFF, 0xFF, 0xFF)

    def set_cell_bg(cell, hex_color):
        tc = cell._tc
        tcPr = tc.get_or_add_tcPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), hex_color)
        tcPr.append(shd)

    def add_h1(doc, text):
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = 'Arial'; run.font.size = Pt(16)
        run.font.bold = True; run.font.color.rgb = NAVY
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = para.paragraph_format
        fmt.space_before = Pt(18); fmt.space_after = Pt(8)
        fmt.keep_with_next = True
        pPr = para._p.get_or_add_pPr()
        pBdr = OxmlElement('w:pBdr')
        bottom = OxmlElement('w:bottom')
        bottom.set(qn('w:val'), 'single'); bottom.set(qn('w:sz'), '8')
        bottom.set(qn('w:space'), '1'); bottom.set(qn('w:color'), '1B3A5C')
        pBdr.append(bottom); pPr.append(pBdr)
        return para

    def add_h2(doc, text):
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = 'Arial'; run.font.size = Pt(13)
        run.font.bold = True; run.font.color.rgb = BLUE
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = para.paragraph_format
        fmt.space_before = Pt(14); fmt.space_after = Pt(5)
        fmt.keep_with_next = True
        return para

    def add_h3(doc, text):
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = 'Arial'; run.font.size = Pt(12)
        run.font.bold = True; run.font.color.rgb = DARKGRAY
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = para.paragraph_format
        fmt.space_before = Pt(10); fmt.space_after = Pt(4)
        fmt.keep_with_next = True
        return para

    def add_body(doc, text, italic=False, bold=False, color=None, size=11):
        para = doc.add_paragraph()
        run = para.add_run(text)
        run.font.name = 'Arial'; run.font.size = Pt(size)
        run.font.color.rgb = color or BODY_COLOR
        run.font.italic = italic; run.font.bold = bold
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt = para.paragraph_format
        fmt.space_before = Pt(4); fmt.space_after = Pt(4)
        fmt.line_spacing_rule = WD_LINE_SPACING.MULTIPLE
        fmt.line_spacing = 1.15
        return para

    def add_bullet(doc, text, bold_prefix=None):
        para = doc.add_paragraph(style='List Bullet')
        if bold_prefix:
            r1 = para.add_run(bold_prefix)
            r1.font.name = 'Arial'; r1.font.size = Pt(11)
            r1.font.bold = True; r1.font.color.rgb = BODY_COLOR
            r2 = para.add_run(text)
            r2.font.name = 'Arial'; r2.font.size = Pt(11)
            r2.font.color.rgb = BODY_COLOR
        else:
            run = para.add_run(text)
            run.font.name = 'Arial'; run.font.size = Pt(11)
            run.font.color.rgb = BODY_COLOR
        para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        fmt = para.paragraph_format
        fmt.space_before = Pt(2); fmt.space_after = Pt(2)
        return para

    def add_code_block(doc, code_text):
        para = doc.add_paragraph()
        run = para.add_run(code_text)
        run.font.name = 'Courier New'; run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(0x1B, 0x3A, 0x5C)
        para.alignment = WD_ALIGN_PARAGRAPH.LEFT
        fmt = para.paragraph_format
        fmt.space_before = Pt(6); fmt.space_after = Pt(6)
        fmt.left_indent = Inches(0.3)
        pPr = para._p.get_or_add_pPr()
        shd = OxmlElement('w:shd')
        shd.set(qn('w:val'), 'clear')
        shd.set(qn('w:color'), 'auto')
        shd.set(qn('w:fill'), 'EEF4FA')
        pPr.append(shd)
        return para

    def add_table(doc, headers, rows, col_widths=None):
        table = doc.add_table(rows=1 + len(rows), cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.style = 'Table Grid'
        hdr_row = table.rows[0]
        for i, h in enumerate(headers):
            cell = hdr_row.cells[i]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            set_cell_bg(cell, '1B3A5C')
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = para.add_run(h)
            run.font.name = 'Arial'; run.font.size = Pt(10)
            run.font.bold = True; run.font.color.rgb = WHITE
        for ri, row_data in enumerate(rows):
            row = table.rows[ri + 1]
            bg = 'F2F6FA' if ri % 2 == 0 else 'FFFFFF'
            for ci, cell_text in enumerate(row_data):
                cell = row.cells[ci]
                cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
                set_cell_bg(cell, bg)
                para = cell.paragraphs[0]
                para.alignment = WD_ALIGN_PARAGRAPH.LEFT
                run = para.add_run(str(cell_text))
                run.font.name = 'Arial'; run.font.size = Pt(10)
                run.font.color.rgb = BODY_COLOR
        if col_widths:
            for row in table.rows:
                for ci, width in enumerate(col_widths):
                    row.cells[ci].width = Inches(width)
        return table

    # ══════════════════════════════════════════════════════════════
    # SECTION 11 — HEARTBEAT SYSTEM
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "11. Proactive Memory (Heartbeat System)")
    add_body(doc,
        "The heartbeat system is Arthur's mechanism for proactive, context-aware monitoring "
        "between user-initiated sessions. Rather than waiting passively for messages, Arthur "
        "performs lightweight checks every ~30 minutes and reaches out only when something "
        "genuinely actionable is found.")

    add_h2(doc, "11.1 Trigger Mechanism")
    add_body(doc,
        "OpenClaw sends a heartbeat poll message approximately every 30 minutes. Arthur "
        "reads HEARTBEAT.md at the start of each poll to determine what to check and under "
        "what conditions to send a notification. State is tracked in heartbeat-state.json "
        "to prevent repeat alerts.")

    add_h2(doc, "11.2 Check Sequence")
    headers_hb = ["Check", "Condition to Alert", "Check Interval"]
    rows_hb = [
        ("Email inbox", "Urgent/unread messages from known senders", "Every ~60 min"),
        ("Calendar", "Events starting within 2 hours", "Every ~30 min"),
        ("Weather", "Relevant if outdoor activity planned", "2x daily"),
        ("Active sub-agents", "Stalled jobs or unexpected failures", "Every ~30 min"),
    ]
    add_table(doc, headers_hb, rows_hb, col_widths=[1.8, 3.0, 1.7])

    add_h2(doc, "11.3 Silence Rules")
    add_body(doc,
        "The golden rule of the heartbeat system is: when in doubt, stay silent. Excessive "
        "notifications are worse than missing one. Arthur applies these silence rules:")
    add_bullet(doc, "Late night (23:00–08:00 BKK): run checks but send only genuinely urgent alerts")
    add_bullet(doc, "Checked <30 min ago: skip the check entirely, return HEARTBEAT_OK")
    add_bullet(doc, "Nothing new since last check: HEARTBEAT_OK immediately")
    add_bullet(doc, "Already alerted on this event: do not re-alert until event changes")

    add_h2(doc, "11.4 Background Work During Heartbeats")
    add_body(doc,
        "If no alerts are needed, Arthur uses heartbeat time productively for background work:")
    add_bullet(doc, "Review recent memory files and update MEMORY.md with new insights")
    add_bullet(doc, "Check git status of workspace and vault")
    add_bullet(doc, "Monitor cron job health (did nightly scripts run successfully?)")
    add_bullet(doc, "Organise and tidy workspace files")
    add_bullet(doc, "Commit any pending vault changes")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════
    # SECTION 12 — CRON JOBS
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "12. Cron Jobs — Memory-Related Automation")
    add_body(doc,
        "Arthur's knowledge and memory systems are supported by a suite of automated cron "
        "jobs that run on Bangkok timezone schedules. Together they ensure memories are "
        "captured, distilled, and backed up without manual intervention.")

    add_h2(doc, "12.1 Complete Cron Schedule")
    headers_c = ["Job Name", "Schedule (BKK)", "Script", "Purpose"]
    rows_c = [
        ("session-remember", "Daily 23:00", "tools/session_remember.py --auto",
         "Mine today's session log for learnings; write to MEMORY.md and KnowledgeGraph"),
        ("auto-reflection", "Daily 02:00", "tools/auto_reflection.py",
         "Reflect on last 2 days; update running-log.md and MEMORY.md; git push vault"),
        ("memory-maintenance", "Weekly Sun 03:00", "(maintenance script)",
         "Review and prune MEMORY.md; remove outdated entries; reorganise sections"),
        ("knowledge-graph-update", "3x daily\n01:00, 06:00, 11:00", "tools/kg_builder.py",
         "Rebuild knowledge_graph.json from current vault state; run kg_auto_link.py"),
        ("vault-git-sync", "4x daily\n02:00, 08:00, 14:00, 20:00", "git add -A && git commit && git push",
         "Push vault to GitHub for off-site backup and version history"),
    ]
    add_table(doc, headers_c, rows_c, col_widths=[1.6, 1.3, 1.9, 2.7])

    add_h2(doc, "12.2 Failure Handling")
    add_body(doc,
        "All cron jobs are designed with graceful failure modes:")
    add_bullet(doc, "LLM extraction jobs fall back to heuristic parsing if API is unavailable")
    add_bullet(doc, "Git push failures are logged but do not block other operations")
    add_bullet(doc, "Jobs write to session log on completion/failure for heartbeat monitoring")
    add_body(doc,
        "Arthur monitors cron health during heartbeats — if a critical job (session-remember, "
        "auto-reflection) hasn't run in >25 hours, it is flagged as an anomaly.")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════
    # SECTION 13 — SECURITY & PRIVACY
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "13. Security & Privacy")
    add_body(doc,
        "Arthur handles sensitive personal and institutional data. The security model "
        "prioritises data isolation, identity verification, and protection of critical files.")

    add_h2(doc, "13.1 Session Isolation")
    add_h3(doc, "MEMORY.md Access Control")
    add_body(doc,
        "MEMORY.md is the most sensitive file — it contains personal context, preferences, "
        "and potentially sensitive identifiers. It is loaded ONLY during main sessions "
        "(Anirach's direct Telegram conversation). It is never loaded in:")
    add_bullet(doc, "Group chats (Discord, Telegram groups, WhatsApp groups)")
    add_bullet(doc, "Sessions triggered by unknown senders")
    add_bullet(doc, "Sub-agent contexts (sub-agents receive AGENTS.md but not MEMORY.md)")
    add_body(doc,
        "This prevents personal context from leaking to third-party participants in shared chats.")

    add_h2(doc, "13.2 Identity Verification")
    add_body(doc,
        "Arthur operates in 'guest mode' when the sender's identity cannot be verified. "
        "In guest mode, Arthur responds helpfully but does not access MEMORY.md, "
        "personal vault data, or execute sensitive operations. A passphrase mechanism "
        "exists for identity confirmation (details redacted from this report — note "
        "that such a mechanism exists and is configured).")

    add_h2(doc, "13.3 Healthcare Database (HA) De-Identification")
    add_body(doc,
        "All queries to the HA (Healthcare Accreditation) PostgreSQL database MUST use "
        "the de-identification wrapper at tools/ha_query.py. This wrapper:")
    add_bullet(doc, "Routes queries through VPN to the private database")
    add_bullet(doc, "Replaces all PII (names, IDs, emails) with pseudonymous tokens")
    add_bullet(doc, "Returns de-identified results to the LLM context")
    add_bullet(doc, "Maintains a local mapping file (/tmp/ha_deid_mapping.json) for report generation")
    add_body(doc,
        "Raw psql access is prohibited. No database credentials are stored in context files "
        "or accessible outside the wrapper script. This is non-negotiable and applies to "
        "all agents and sub-agents.")

    add_h2(doc, "13.4 Core File Protection")
    add_body(doc,
        "Six core files are protected from accidental deletion or overwrite. Arthur always "
        "requests explicit confirmation before modifying them:")
    headers_prot = ["Protected File", "Why Critical"]
    rows_prot = [
        ("SOUL.md", "Defines Arthur's entire identity and persona"),
        ("AGENTS.md", "Operational rulebook for all agents"),
        ("MEMORY.md", "Long-term curated memory — irreplaceable if deleted"),
        ("USER.md", "Anirach's profile — took effort to build"),
        ("HEARTBEAT.md", "Proactive check configuration"),
        ("OpenClaw config", "Gateway configuration — breakage = service outage"),
    ]
    add_table(doc, headers_prot, rows_prot, col_widths=[2.0, 4.5])

    add_h2(doc, "13.5 Destructive Command Policy")
    add_body(doc,
        "Before running any destructive command (rm, overwrite, restart, config change), "
        "Arthur describes the impact and requests explicit confirmation: "
        "\"This would [describe impact]. Confirm? (yes/no)\". "
        "The policy uses trash (recoverable) over rm (permanent) wherever possible.")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════
    # SECTION 14 — AGENT TEAMS & MEMORY
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "14. Agent Teams & Memory")
    add_body(doc,
        "Arthur orchestrates five specialised agent teams, each operating in its own "
        "Obsidian workspace with team-specific templates, work logs, and project files. "
        "Memory integration ensures continuity across agent spawns and team collaboration.")

    add_h2(doc, "14.1 Team Overview")
    headers_teams = ["Team", "Workspace", "Focus Area", "Agent Count"]
    rows_teams = [
        ("Writing", "Agents/Writing/", "Academic papers, articles, content drafting", "~9"),
        ("Academic", "Agents/Academic/", "Literature review, research analysis, citations", "~9"),
        ("Translation", "Agents/Translation/", "Thai/English translation, localisation", "~9"),
        ("Course", "Agents/Course/", "Lecture materials, slides, handouts, quizzes", "~9"),
        ("Coding", "Agents/Coding/", "Software development, DevOps, debugging", "~9"),
    ]
    add_table(doc, headers_teams, rows_teams, col_widths=[1.2, 2.0, 2.5, 1.3])
    add_body(doc, "Total: approximately 45 agents across all five teams.")

    add_h2(doc, "14.2 Team Memory Architecture")
    add_h3(doc, "Work Logs")
    add_body(doc,
        "Each team maintains dated work logs in its workspace:")
    add_code_block(doc,
        "Agents/{Team}/Work-Log-YYYY-MM-DD.md")
    add_body(doc,
        "Work logs record: tasks completed, decisions made, files produced, and deliverables "
        "with Google Drive links. The main agent updates the relevant team work log after "
        "each sub-agent spawn completes.")

    add_h3(doc, "Templates")
    add_body(doc,
        "Each team has standardised templates in Agents/{Team}/Templates/ for its most "
        "common deliverables. Using templates ensures consistent formatting and reduces "
        "per-task setup overhead.")

    add_h3(doc, "Cross-Team Linking")
    add_body(doc,
        "Agents use wiki-links to reference work from other teams. For example, a Writing "
        "team paper note might link [[Academic/Literature-Review-RAG]] to the Academic "
        "team's research notes, creating a connected knowledge graph across teams.")

    add_h2(doc, "14.3 Spawn Protocol")
    add_body(doc,
        "When Arthur spawns a sub-agent, it always:")
    add_bullet(doc, "Provides the full task description including specific requirements, project path, tech stack, and expected deliverables")
    add_bullet(doc, "Includes the instruction to send deliverable files immediately via Telegram (tg_send_file.sh)")
    add_bullet(doc, "Updates the team work log after the sub-agent completes")
    add_bullet(doc, "Logs the Google Drive link in the Deliverables section of the work log")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════
    # SECTION 15 — ARSCONTEXTA INSPIRATION
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "15. Arscontexta Inspiration")
    add_body(doc,
        "Arscontexta is a methodology for AI memory and context management that inspired "
        "several components of Arthur's knowledge system. Understanding the inspiration "
        "helps explain the design philosophy behind the pipelines.")

    add_h2(doc, "15.1 What We Borrowed")
    headers_a = ["Arscontexta Concept", "Our Implementation"]
    rows_a = [
        ("/remember command",
         "session_remember.py — mines session logs and stores learnings in structured format"),
        ("6 Rs Processing Pipeline\n(Record→Reduce→Reflect→Reweave→Verify→Rethink)",
         "session_remember.py + auto_reflection.py together implement this pipeline across the day"),
        ("Fresh context per processing phase",
         "Each pipeline step uses a fresh LLM call (or sub-agent) to avoid context contamination"),
        ("Three-space architecture\n(self / notes / ops)",
         "Maps directly to: SOUL.md + USER.md (self) / Obsidian vault (notes) / AGENTS.md + cron (ops)"),
    ]
    add_table(doc, headers_a, rows_a, col_widths=[2.5, 4.0])

    add_h2(doc, "15.2 What We Already Had")
    add_body(doc,
        "Interestingly, Arthur's system already implemented several patterns that Arscontexta "
        "recommends — confirming the architectural soundness of these approaches:")
    headers_already = ["Pattern", "Our Pre-existing Implementation"]
    rows_already = [
        ("Vault with wiki-links", "Obsidian vault with full KnowledgeGraph/ structure"),
        ("Maps of Content (MOCs)", "MOC-Research.md, MOC-Infrastructure.md, MOC-Teaching.md"),
        ("Session capture", "memory/YYYY-MM-DD.md daily logging"),
        ("Auto git commits", "vault-git-sync cron (4x daily) + on-write commits"),
        ("Structured memory types", "SQLite memory.db with typed records"),
    ]
    add_table(doc, headers_already, rows_already, col_widths=[2.5, 4.0])

    add_h2(doc, "15.3 The 6 Rs in Arthur's System")
    add_body(doc,
        "The 6 Rs processing pipeline maps to Arthur's automation as follows:")
    add_bullet(doc, "Record — During session: Arthur writes to memory/YYYY-MM-DD.md")
    add_bullet(doc, "Reduce — 23:00 cron: session_remember.py extracts structured learnings")
    add_bullet(doc, "Reflect — 02:00 cron: auto_reflection.py synthesises patterns across days")
    add_bullet(doc, "Reweave — kg_builder.py + kg_auto_link.py reconnects new knowledge to existing graph")
    add_bullet(doc, "Verify — Heartbeat monitoring + memory-maintenance weekly review")
    add_bullet(doc, "Rethink — Future: planned contradiction-detection and belief-update pipeline")

    doc.add_page_break()

    # ══════════════════════════════════════════════════════════════
    # SECTION 16 — QUICK REFERENCE COMMANDS
    # ══════════════════════════════════════════════════════════════
    add_h1(doc, "16. Quick Reference — Commands")
    add_body(doc,
        "The following tables and code blocks provide a complete operational reference "
        "for all memory and knowledge management commands.")

    add_h2(doc, "16.1 Search Commands")
    add_code_block(doc,
        "# Semantic search (OpenClaw built-in)\n"
        "memory_search \"what did we decide about course structure?\"\n\n"
        "# SQLite FTS search\n"
        "python3 tools/memory_db.py search \"RAG system\"\n"
        "python3 tools/memory_db.py search \"hospital\" --type decision\n\n"
        "# Full vault search\n"
        "python3 tools/obsidian_search.py \"Arscontexta\"\n"
        "python3 tools/obsidian_search.py \"#project\" --tag")

    add_h2(doc, "16.2 Add Memory Commands")
    add_code_block(doc,
        "# Route notes to correct store via quick_note.py\n"
        "python3 tools/quick_note.py fact \"Google Drive ArthurBotData folder ID: 1abc...\"\n"
        "python3 tools/quick_note.py decision \"Use DeepSeek for LLM extraction tasks\"\n"
        "python3 tools/quick_note.py todo \"Review MOC-Research.md for outdated links\"\n"
        "python3 tools/quick_note.py remember \"User prefers bullet points over paragraphs\"\n"
        "python3 tools/quick_note.py event \"Paper submitted to AIiH2026 conference\"\n"
        "python3 tools/quick_note.py person \"Dr Smith — HA project collaborator\"\n"
        "python3 tools/quick_note.py project \"RAG-SecondBrain: Phase 2 complete\"")

    add_h2(doc, "16.3 Session Learning Commands")
    add_code_block(doc,
        "# Process specific text immediately\n"
        "python3 tools/session_remember.py \"We decided to use chunked context for RAG\"\n\n"
        "# Auto-mine today's session log\n"
        "python3 tools/session_remember.py --auto\n\n"
        "# Pipe content from file\n"
        "cat memory/2026-02-23.md | python3 tools/session_remember.py --stdin")

    add_h2(doc, "16.4 SQLite Memory Database Commands")
    add_code_block(doc,
        "python3 tools/memory_db.py add --type fact \"text\"\n"
        "python3 tools/memory_db.py search \"query\"\n"
        "python3 tools/memory_db.py recent --days 7\n"
        "python3 tools/memory_db.py list --type decision\n"
        "python3 tools/memory_db.py get 42\n"
        "python3 tools/memory_db.py update 42 \"updated text\"\n"
        "python3 tools/memory_db.py archive 42\n"
        "python3 tools/memory_db.py stats\n"
        "python3 tools/memory_db.py export memories.json\n"
        "python3 tools/memory_db.py rebuild-fts")

    add_h2(doc, "16.5 Vault Operations")
    add_code_block(doc,
        "# Commit and push vault to GitHub\n"
        "cd /home/clawdbot/obsidian-vault\n"
        "git add -A && git commit -m \"Update: notes from 2026-02-23\" && git push\n\n"
        "# Rebuild knowledge graph\n"
        "python3 /home/clawdbot/clawd/tools/kg_builder.py\n\n"
        "# Auto-link new notes\n"
        "python3 /home/clawdbot/clawd/tools/kg_auto_link.py\n\n"
        "# Run auto-reflection manually\n"
        "python3 /home/clawdbot/clawd/tools/auto_reflection.py")

    add_h2(doc, "16.6 File Delivery Commands")
    add_code_block(doc,
        "# Send file via Telegram (sub-agents MUST use this)\n"
        "bash /home/clawdbot/clawd/tools/tg_send_file.sh /path/to/file.docx \"Caption here\"\n\n"
        "# Upload to Google Drive\n"
        "python3 /home/clawdbot/clawd/gdrive/gdrive_upload.py /path/to/file.docx \\\n"
        "  --folder \"Projects/Project-Name\" \"File_Name_v1.docx\"")

    add_h2(doc, "16.7 All Memory Commands — Summary Table")
    headers_all = ["Command", "Purpose", "Output Location"]
    rows_all = [
        ("memory_search \"q\"", "Semantic search", "MEMORY.md + memory/*.md"),
        ("memory_db.py search \"q\"", "SQLite FTS", "memory.db"),
        ("obsidian_search.py \"q\"", "Vault full-text search", "obsidian-vault/"),
        ("quick_note.py fact \"t\"", "Add a fact", "KnowledgeGraph/Documents/Facts.md"),
        ("quick_note.py decision \"t\"", "Add a decision", "KnowledgeGraph/Decisions/"),
        ("quick_note.py todo \"t\"", "Add a task", "KnowledgeGraph/Action-Items/"),
        ("quick_note.py remember \"t\"", "Add a lesson", "MEMORY.md Lessons Learned"),
        ("quick_note.py event \"t\"", "Log an event", "Daily memory note"),
        ("session_remember.py --auto", "Mine session learnings", "MEMORY.md + KnowledgeGraph"),
        ("auto_reflection.py", "Nightly reflection", "running-log.md + MEMORY.md"),
        ("kg_builder.py", "Rebuild knowledge graph", "knowledge_graph.json"),
        ("memory_db.py stats", "DB statistics", "Console output"),
    ]
    add_table(doc, headers_all, rows_all, col_widths=[2.2, 2.1, 2.2])

    doc.add_paragraph()

    # ── Closing note ──
    add_body(doc,
        "This document covers the complete architecture of Arthur's knowledge and memory "
        "system as of February 2026. As the system evolves, this reference should be "
        "updated accordingly. For the latest operational state, consult MEMORY.md, "
        "Quick-Reference.md, and the running-log.md.",
        italic=True, color=RGBColor(0x66, 0x66, 0x66))
