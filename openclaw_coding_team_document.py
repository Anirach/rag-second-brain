#!/usr/bin/env python3
"""
Generate a professional DOCX document on "Using OpenClaw to Set Up a Coding Team"
"""

from docx import Document
from docx.shared import Inches, Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn, nsmap
from docx.oxml import OxmlElement
import datetime

def add_page_number(section):
    """Add page numbers to footer"""
    footer = section.footer
    footer.is_linked_to_previous = False
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # Add "Page X of Y" format
    run = p.add_run()
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    run._r.append(fldChar1)
    
    run2 = p.add_run()
    instrText = OxmlElement('w:instrText')
    instrText.text = "PAGE"
    run2._r.append(instrText)
    
    run3 = p.add_run()
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'end')
    run3._r.append(fldChar2)
    
    run4 = p.add_run(" of ")
    
    run5 = p.add_run()
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'begin')
    run5._r.append(fldChar3)
    
    run6 = p.add_run()
    instrText2 = OxmlElement('w:instrText')
    instrText2.text = "NUMPAGES"
    run6._r.append(instrText2)
    
    run7 = p.add_run()
    fldChar4 = OxmlElement('w:fldChar')
    fldChar4.set(qn('w:fldCharType'), 'end')
    run7._r.append(fldChar4)

def add_code_block(doc, code, language="yaml"):
    """Add a formatted code block"""
    p = doc.add_paragraph()
    p.paragraph_format.left_indent = Inches(0.25)
    p.paragraph_format.space_before = Pt(6)
    p.paragraph_format.space_after = Pt(6)
    
    run = p.add_run(code)
    run.font.name = 'Courier New'
    run.font.size = Pt(9)
    
    # Add shading
    shading = OxmlElement('w:shd')
    shading.set(qn('w:fill'), 'F5F5F5')
    p._p.get_or_add_pPr().append(shading)
    
    return p

def create_table(doc, headers, rows):
    """Create a formatted table"""
    table = doc.add_table(rows=len(rows)+1, cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    # Header row
    header_cells = table.rows[0].cells
    for i, header in enumerate(headers):
        header_cells[i].text = header
        for paragraph in header_cells[i].paragraphs:
            for run in paragraph.runs:
                run.bold = True
        shading = OxmlElement('w:shd')
        shading.set(qn('w:fill'), '4472C4')
        header_cells[i]._tc.get_or_add_tcPr().append(shading)
        for paragraph in header_cells[i].paragraphs:
            for run in paragraph.runs:
                run.font.color.rgb = None
    
    # Data rows
    for i, row in enumerate(rows):
        row_cells = table.rows[i+1].cells
        for j, cell in enumerate(row):
            row_cells[j].text = str(cell)
    
    return table

def main():
    doc = Document()
    
    # Set up styles
    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)
    
    # Title Page
    title = doc.add_heading('Using OpenClaw to Set Up a Coding Team', 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    subtitle = doc.add_paragraph('A Comprehensive Guide to Multi-Agent Architecture for Development Teams')
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    subtitle.runs[0].font.size = Pt(14)
    subtitle.runs[0].italic = True
    
    doc.add_paragraph()
    doc.add_paragraph()
    
    version_info = doc.add_paragraph()
    version_info.alignment = WD_ALIGN_PARAGRAPH.CENTER
    version_info.add_run(f'Version 1.0\n{datetime.datetime.now().strftime("%B %d, %Y")}\n\nOpenClaw Documentation Series')
    
    doc.add_page_break()
    
    # Table of Contents placeholder
    toc_heading = doc.add_heading('Table of Contents', 1)
    doc.add_paragraph('1. Executive Summary')
    doc.add_paragraph('2. Introduction to OpenClaw Multi-Agent Setup')
    doc.add_paragraph('3. Architecture Overview')
    doc.add_paragraph('4. Step-by-Step Setup Guide')
    doc.add_paragraph('5. Example Configurations')
    doc.add_paragraph('6. Agent Role Use Cases')
    doc.add_paragraph('7. Best Practices & Security Considerations')
    doc.add_paragraph('8. Troubleshooting Common Issues')
    doc.add_paragraph('9. Conclusion')
    doc.add_paragraph('10. References')
    
    doc.add_page_break()
    
    # Section 1: Executive Summary
    doc.add_heading('1. Executive Summary', 1)
    
    doc.add_paragraph(
        'OpenClaw is an open-source, locally-running AI assistant framework that enables sophisticated '
        'multi-agent architectures for software development teams. This document provides a comprehensive '
        'guide to setting up a coding team using OpenClaw\'s multi-agent capabilities, including agent '
        'routing, session isolation, workspace configuration, and specialized coding skills.'
    )
    
    doc.add_paragraph(
        'Key benefits of using OpenClaw for coding teams include:'
    )
    
    benefits = [
        'Complete isolation between agents with separate workspaces, sessions, and authentication profiles',
        'Flexible routing of messages from different channels to specialized agents',
        'Hierarchical orchestration patterns with Opus-level orchestrators and specialized workers',
        'Integration with GitHub CLI for automated code reviews and PR management',
        'Sandboxed execution environments for secure code execution',
        'Persistent memory and context across sessions for continuous development workflows'
    ]
    
    for benefit in benefits:
        p = doc.add_paragraph(benefit, style='List Bullet')
    
    doc.add_paragraph(
        'This guide covers everything from initial setup to advanced multi-agent orchestration patterns, '
        'with practical examples and configuration snippets that can be directly applied to your development workflow.'
    )
    
    doc.add_page_break()
    
    # Section 2: Introduction
    doc.add_heading('2. Introduction to OpenClaw Multi-Agent Setup', 1)
    
    doc.add_heading('2.1 What is OpenClaw?', 2)
    doc.add_paragraph(
        'OpenClaw (formerly known as Moltbot and Clawdbot) is a locally-running AI assistant that operates '
        'directly on your machine. It integrates with multiple AI model providers (Anthropic Claude, OpenAI, '
        'Google Gemini, and others) and can be accessed through various messaging platforms including Telegram, '
        'Discord, Slack, WhatsApp, and more.'
    )
    
    doc.add_heading('2.2 What Constitutes One Agent?', 2)
    doc.add_paragraph(
        'An agent in OpenClaw is not just a model or a prompt—it\'s a complete isolated environment with three core components:'
    )
    
    doc.add_paragraph('Workspace and Agent Directory', style='List Bullet')
    doc.add_paragraph(
        'Every agent operates from its own workspace, typically defined by an agentDir path. This directory '
        'contains the agent\'s configuration files (AGENTS.md, SOUL.md, USER.md, TOOLS.md), memory storage, '
        'and skill definitions. Two agents cannot accidentally share or overwrite each other\'s files unless '
        'explicitly configured.'
    )
    
    doc.add_paragraph('Session Store', style='List Bullet')
    doc.add_paragraph(
        'Each agent maintains its own conversation history and session state under ~/.openclaw/agents/<agentId>/sessions. '
        'This isolation means Agent A\'s conversation with a user on Discord remains completely separate from '
        'Agent B\'s conversation with the same user on Telegram.'
    )
    
    doc.add_paragraph('Authentication Context', style='List Bullet')
    doc.add_paragraph(
        'Authentication profiles are strictly per-agent. When you configure API keys, OAuth tokens, or service '
        'credentials for one agent, these credentials belong exclusively to that agent. This design prevents '
        'credential leakage between agents.'
    )
    
    doc.add_heading('2.3 When Multi-Agent Architecture Makes Sense', 2)
    doc.add_paragraph(
        'Multi-agent configurations become valuable when you have genuinely distinct operational domains '
        'that benefit from isolation:'
    )
    
    use_cases = [
        ('Different Security Contexts', 'When some conversations require access to sensitive systems while others should remain sandboxed'),
        ('Specialized Expertise', 'Running a research agent alongside a coding agent allows each to optimize for its specific domain'),
        ('Channel-Specific Behaviors', 'Different personalities or capabilities on different channels'),
        ('Resource Management', 'Different agents can use different models (Opus for complex reasoning, Sonnet for routine tasks)')
    ]
    
    for title, desc in use_cases:
        p = doc.add_paragraph()
        p.add_run(f'{title}: ').bold = True
        p.add_run(desc)
    
    doc.add_page_break()
    
    # Section 3: Architecture Overview
    doc.add_heading('3. Architecture Overview', 1)
    
    doc.add_heading('3.1 System Architecture Diagram', 2)
    doc.add_paragraph(
        'The following describes the OpenClaw multi-agent architecture for a coding team:'
    )
    
    arch_desc = '''
┌─────────────────────────────────────────────────────────────────────┐
│                        OpenClaw Gateway                              │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                     Message Router                            │   │
│  │  (Bindings: channel → accountId → peer → agentId)            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│     ┌────────────────────────┼────────────────────────┐             │
│     ▼                        ▼                        ▼             │
│  ┌─────────┐          ┌─────────────┐          ┌──────────┐        │
│  │ Orchestrator │     │ Code Review │          │ PR Agent │        │
│  │   Agent      │     │    Agent    │          │          │        │
│  │ (Opus 4.5)   │     │ (Sonnet 4.5)│          │(Sonnet)  │        │
│  └─────────┘          └─────────────┘          └──────────┘        │
│       │                     │                        │              │
│  ┌────┴────┐          ┌─────┴─────┐          ┌──────┴──────┐       │
│  │Workspace│          │ Workspace │          │  Workspace  │       │
│  │ -orch   │          │ -review   │          │    -pr      │       │
│  └─────────┘          └───────────┘          └─────────────┘       │
└─────────────────────────────────────────────────────────────────────┘
'''
    add_code_block(doc, arch_desc, "text")
    
    doc.add_heading('3.2 Core Components', 2)
    
    components = [
        ('Gateway', 'The WebSocket server that handles channels, nodes, sessions, and hooks. Runs on port 18789 by default.'),
        ('Agents', 'Isolated AI personas with their own workspace, sessions, and authentication.'),
        ('Bindings', 'Routing rules that map incoming messages to specific agents based on channel, account, and peer.'),
        ('Workspaces', 'The agent\'s home directory containing configuration files, memory, and skills.'),
        ('Skills', 'Extensions that teach the agent how to use tools (AgentSkills-compatible).'),
        ('Sessions', 'Conversation history and state, stored per-agent under ~/.openclaw/agents/<agentId>/sessions.')
    ]
    
    table = doc.add_table(rows=len(components)+1, cols=2)
    table.style = 'Table Grid'
    
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = 'Component'
    hdr_cells[1].text = 'Description'
    for cell in hdr_cells:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    
    for i, (comp, desc) in enumerate(components):
        row_cells = table.rows[i+1].cells
        row_cells[0].text = comp
        row_cells[1].text = desc
    
    doc.add_paragraph()
    
    doc.add_heading('3.3 Path Structure', 2)
    doc.add_paragraph('Understanding the file structure is essential for multi-agent setups:')
    
    paths = '''
~/.openclaw/
├── openclaw.json              # Main configuration file
├── credentials/               # OAuth tokens, API keys
├── skills/                    # Managed/shared skills
├── workspace/                 # Default agent workspace
├── workspace-<agentId>/       # Per-agent workspaces
└── agents/
    └── <agentId>/
        ├── agent/             # Per-agent auth profiles
        └── sessions/          # Session transcripts
'''
    add_code_block(doc, paths, "text")
    
    doc.add_page_break()
    
    # Section 4: Step-by-Step Setup Guide
    doc.add_heading('4. Step-by-Step Setup Guide', 1)
    
    doc.add_heading('4.1 Prerequisites', 2)
    doc.add_paragraph('Before setting up a multi-agent coding team, ensure you have:')
    
    prereqs = [
        'Node.js v18+ installed',
        'Docker installed (for sandboxing)',
        'OpenClaw installed: npm install -g openclaw',
        'API keys for your preferred AI providers (Anthropic, OpenAI, etc.)',
        'GitHub CLI (gh) installed and authenticated for code integration'
    ]
    for p in prereqs:
        doc.add_paragraph(p, style='List Bullet')
    
    doc.add_heading('4.2 Initial Configuration', 2)
    doc.add_paragraph('Run the initial setup:')
    
    setup_code = '''# Initialize OpenClaw
openclaw onboard --install-daemon

# Configure API keys
openclaw configure --section keys

# Verify installation
openclaw doctor'''
    add_code_block(doc, setup_code, "bash")
    
    doc.add_heading('4.3 Creating Agent Workspaces', 2)
    doc.add_paragraph('Create separate workspaces for each coding team agent:')
    
    workspace_code = '''# Create orchestrator agent
openclaw agents add orchestrator --workspace ~/.openclaw/workspace-orchestrator

# Create code review agent
openclaw agents add code-review --workspace ~/.openclaw/workspace-code-review

# Create PR agent
openclaw agents add pr-agent --workspace ~/.openclaw/workspace-pr-agent

# Create testing agent
openclaw agents add testing --workspace ~/.openclaw/workspace-testing

# Create documentation agent
openclaw agents add docs --workspace ~/.openclaw/workspace-docs

# Verify agents
openclaw agents list --bindings'''
    add_code_block(doc, workspace_code, "bash")
    
    doc.add_heading('4.4 Workspace File Structure', 2)
    doc.add_paragraph('Each workspace should contain the following files:')
    
    files = [
        ('AGENTS.md', 'Operating instructions for the agent and how it should use memory'),
        ('SOUL.md', 'Persona, tone, and behavioral boundaries'),
        ('USER.md', 'Information about who the user is and how to address them'),
        ('TOOLS.md', 'Notes about local tools and conventions (guidance only)'),
        ('IDENTITY.md', 'The agent\'s name, theme, and emoji'),
        ('skills/', 'Workspace-specific skills that override shared skills')
    ]
    
    table = doc.add_table(rows=len(files)+1, cols=2)
    table.style = 'Table Grid'
    hdr = table.rows[0].cells
    hdr[0].text = 'File'
    hdr[1].text = 'Purpose'
    for cell in hdr:
        for p in cell.paragraphs:
            for r in p.runs:
                r.bold = True
    
    for i, (f, p) in enumerate(files):
        row = table.rows[i+1].cells
        row[0].text = f
        row[1].text = p
    
    doc.add_page_break()
    
    # Section 5: Example Configurations
    doc.add_heading('5. Example Configurations', 1)
    
    doc.add_heading('5.1 Complete gateway.yaml for Multi-Agent Coding Team', 2)
    doc.add_paragraph('The following configuration sets up a complete coding team with specialized agents:')
    
    config_yaml = '''{
  // Multi-Agent Coding Team Configuration
  agents: {
    list: [
      {
        id: "orchestrator",
        default: true,
        name: "Orchestrator",
        workspace: "~/.openclaw/workspace-orchestrator",
        agentDir: "~/.openclaw/agents/orchestrator/agent",
        model: "anthropic/claude-opus-4-5",
        identity: {
          name: "Team Lead",
          theme: "Technical architect and coordinator",
          emoji: "🎯"
        },
        subagents: {
          allowAgents: ["code-review", "pr-agent", "testing", "docs"]
        }
      },
      {
        id: "code-review",
        name: "Code Reviewer",
        workspace: "~/.openclaw/workspace-code-review",
        agentDir: "~/.openclaw/agents/code-review/agent",
        model: "anthropic/claude-sonnet-4-5",
        identity: {
          name: "Code Reviewer",
          theme: "Senior engineer focused on code quality",
          emoji: "🔍"
        },
        sandbox: {
          mode: "all",
          scope: "agent"
        },
        tools: {
          allow: ["group:fs", "group:runtime", "group:web", "group:sessions"]
        }
      },
      {
        id: "pr-agent",
        name: "PR Manager",
        workspace: "~/.openclaw/workspace-pr-agent",
        agentDir: "~/.openclaw/agents/pr-agent/agent",
        model: "anthropic/claude-sonnet-4-5",
        identity: {
          name: "PR Bot",
          theme: "GitHub PR and issue management specialist",
          emoji: "🔀"
        }
      },
      {
        id: "testing",
        name: "Test Engineer",
        workspace: "~/.openclaw/workspace-testing",
        agentDir: "~/.openclaw/agents/testing/agent",
        model: "anthropic/claude-sonnet-4-5",
        identity: {
          name: "Test Bot",
          theme: "Quality assurance and test automation",
          emoji: "🧪"
        },
        sandbox: {
          mode: "all",
          scope: "agent",
          docker: {
            setupCommand: "apt-get update && apt-get install -y nodejs npm"
          }
        }
      },
      {
        id: "docs",
        name: "Documentation Writer",
        workspace: "~/.openclaw/workspace-docs",
        agentDir: "~/.openclaw/agents/docs/agent",
        model: "anthropic/claude-sonnet-4-5",
        identity: {
          name: "Docs Bot",
          theme: "Technical documentation specialist",
          emoji: "📚"
        }
      }
    ]
  },

  // Route messages to appropriate agents
  bindings: [
    // Slack: #code-review channel → code-review agent
    {
      agentId: "code-review",
      match: {
        channel: "slack",
        peer: { kind: "channel", id: "C0123REVIEW" }
      }
    },
    // Slack: #pull-requests channel → pr-agent
    {
      agentId: "pr-agent",
      match: {
        channel: "slack",
        peer: { kind: "channel", id: "C0123PR" }
      }
    },
    // Telegram: Testing group → testing agent
    {
      agentId: "testing",
      match: {
        channel: "telegram",
        peer: { kind: "group", id: "-1001234567890" }
      }
    },
    // Default: All other messages → orchestrator
    {
      agentId: "orchestrator",
      match: { channel: "telegram" }
    },
    {
      agentId: "orchestrator", 
      match: { channel: "slack" }
    }
  ],

  // Tool configuration
  tools: {
    profile: "coding",
    web: {
      search: { enabled: true },
      fetch: { enabled: true }
    },
    exec: {
      pathPrepend: ["~/.local/bin", "/usr/local/bin"]
    }
  },

  // Sub-agent configuration
  agents: {
    defaults: {
      subagents: {
        maxConcurrent: 4,
        archiveAfterMinutes: 120,
        model: "anthropic/claude-sonnet-4-5"
      }
    }
  }
}'''
    add_code_block(doc, config_yaml, "json5")
    
    doc.add_page_break()
    
    doc.add_heading('5.2 SOUL.md Example for Code Review Agent', 2)
    doc.add_paragraph('The SOUL.md file defines the agent\'s persona and behavioral guidelines:')
    
    soul_md = '''# SOUL.md - Code Review Agent

## Identity
You are the Code Review Agent, a meticulous senior engineer dedicated to 
maintaining code quality across the team's repositories.

## Personality
- Thorough but not pedantic
- Constructive in feedback
- Focus on patterns, not just syntax
- Praise good practices as well as identify issues

## Core Responsibilities
1. Review pull request diffs for code quality issues
2. Check for security vulnerabilities and anti-patterns
3. Verify adherence to project coding standards
4. Suggest improvements and optimizations
5. Report findings to the team via notifications

## Communication Style
- Be specific with line numbers and file references
- Provide rationale for suggested changes
- Include code examples when proposing alternatives
- Prioritize issues by severity (Critical > Major > Minor)

## Boundaries
- Do NOT merge PRs or push code directly
- Do NOT access production systems
- Always recommend human review for critical changes
- Escalate security concerns to the orchestrator agent

## Tools You Use
- GitHub CLI (gh) for fetching PR details and diffs
- Web search for researching best practices
- File tools for reading local documentation
- Message tool for sending review summaries'''
    add_code_block(doc, soul_md, "markdown")
    
    doc.add_heading('5.3 AGENTS.md Example for Orchestrator', 2)
    
    agents_md = '''# AGENTS.md - Orchestrator Agent

## Role
You are the Team Lead orchestrator. You coordinate work between specialized 
agents and handle complex tasks that require multiple perspectives.

## Sub-Agent Coordination
Use `sessions_spawn` to delegate tasks to specialized agents:

- **code-review**: For PR review tasks
- **pr-agent**: For GitHub operations
- **testing**: For test creation and execution
- **docs**: For documentation updates

## Workflow Patterns

### New Feature Review
1. Spawn code-review agent to analyze the diff
2. Wait for review completion
3. Spawn testing agent to verify test coverage
4. Consolidate findings and report to the team

### Bug Triage
1. Analyze the bug report
2. Spawn pr-agent to find related issues/PRs
3. If fix exists, coordinate with code-review
4. If new, create an issue via pr-agent

## Memory Management
- Log significant decisions in memory/YYYY-MM-DD.md
- Update MEMORY.md with recurring patterns and team preferences
- Reference past decisions when making similar choices'''
    add_code_block(doc, agents_md, "markdown")
    
    doc.add_page_break()
    
    # Section 6: Use Cases
    doc.add_heading('6. Agent Role Use Cases', 1)
    
    doc.add_heading('6.1 Code Review Agent', 2)
    doc.add_paragraph(
        'The Code Review Agent specializes in analyzing pull request diffs and providing '
        'constructive feedback without requiring write access to repositories.'
    )
    
    doc.add_paragraph('Workflow:', style='Intense Quote')
    
    review_workflow = '''# Automated Code Review Workflow

## 1. Fetch PR details
gh pr view 123 --json title,body,files,additions,deletions

## 2. Get the diff
gh pr diff 123

## 3. Analyze and generate review summary
# Agent processes the diff and creates structured feedback

## 4. Send notification to team
# Message sent via Slack/Telegram with:
# - Summary of changes
# - Issues found (categorized by severity)
# - Suggestions for improvement
# - Direct link to PR'''
    add_code_block(doc, review_workflow, "bash")
    
    doc.add_heading('6.2 PR Agent', 2)
    doc.add_paragraph(
        'The PR Agent handles GitHub operations including issue creation, PR monitoring, '
        'and workflow status checks.'
    )
    
    pr_workflow = '''# PR Agent Capabilities

## List open PRs needing review
gh pr list --state open --json number,title,author,reviewDecision

## Check CI/CD status
gh run list --limit 5 --json status,conclusion,name

## Create issues from bug reports
gh issue create --title "Bug: ..." --body "..." --label bug

## Monitor for failed workflows
gh run list --status failure --json databaseId,name,conclusion | \\
  jq '.[] | select(.conclusion == "failure")'

## Notify team of stale PRs (>3 days old)
gh pr list --json number,title,createdAt | \\
  jq '[.[] | select((.createdAt | fromdateiso8601) < (now - 259200))]' '''
    add_code_block(doc, pr_workflow, "bash")
    
    doc.add_heading('6.3 Testing Agent', 2)
    doc.add_paragraph(
        'The Testing Agent creates and runs tests, analyzes coverage, and identifies '
        'gaps in test suites.'
    )
    
    test_workflow = '''# Testing Agent Responsibilities

## Run test suite
npm test -- --coverage

## Analyze test results
# Parse coverage reports and identify:
# - Files with low coverage (<80%)
# - Uncovered branches
# - Missing edge case tests

## Generate test suggestions
# For each uncovered function, suggest test cases:
# - Happy path scenarios
# - Error handling
# - Boundary conditions

## Create test files
# Write new test files following project conventions'''
    add_code_block(doc, test_workflow, "bash")
    
    doc.add_heading('6.4 Documentation Agent', 2)
    doc.add_paragraph(
        'The Documentation Agent maintains README files, API documentation, and '
        'inline code comments.'
    )
    
    doc.add_page_break()
    
    # Section 7: Best Practices
    doc.add_heading('7. Best Practices & Security Considerations', 1)
    
    doc.add_heading('7.1 Security Best Practices', 2)
    
    security_items = [
        ('Use Read-Only GitHub Access', 'Give agents read-only access to repositories. Route all write operations through human approval.'),
        ('Enable Sandboxing', 'Run coding agents in Docker containers to isolate code execution.'),
        ('Separate Auth Profiles', 'Never share authentication credentials between agents. Each agent should have its own API keys.'),
        ('Branch Protection', 'Enable GitHub branch protection rules as a safety net regardless of agent permissions.'),
        ('Audit Logging', 'Enable the command-logger hook to track all agent actions.')
    ]
    
    for title, desc in security_items:
        p = doc.add_paragraph()
        p.add_run(f'{title}: ').bold = True
        p.add_run(desc)
    
    doc.add_heading('7.2 Performance Optimization', 2)
    
    perf_items = [
        'Use Claude Opus for orchestration and complex reasoning tasks',
        'Use Claude Sonnet for routine operations like code review',
        'Set appropriate maxConcurrent limits for sub-agents',
        'Archive sessions after completion to manage storage',
        'Use tool profiles (coding, messaging, minimal) to reduce token overhead'
    ]
    for item in perf_items:
        doc.add_paragraph(item, style='List Bullet')
    
    doc.add_heading('7.3 Workspace Organization', 2)
    
    workspace_tips = [
        'Keep each agent\'s workspace in version control (private repo)',
        'Use consistent naming conventions across workspaces',
        'Document agent capabilities in IDENTITY.md',
        'Regularly update MEMORY.md with lessons learned',
        'Create shared skills in ~/.openclaw/skills/ for common functionality'
    ]
    for tip in workspace_tips:
        doc.add_paragraph(tip, style='List Bullet')
    
    doc.add_page_break()
    
    # Section 8: Troubleshooting
    doc.add_heading('8. Troubleshooting Common Issues', 1)
    
    doc.add_heading('8.1 Agent Routing Issues', 2)
    doc.add_paragraph('If messages are going to the wrong agent:')
    
    routing_fixes = '''# Check current bindings
openclaw agents list --bindings

# Verify binding order (most-specific first)
# Peer bindings > channel+accountId > channel-only

# Test routing with debug mode
openclaw gateway --verbose'''
    add_code_block(doc, routing_fixes, "bash")
    
    doc.add_heading('8.2 Session Isolation Problems', 2)
    doc.add_paragraph('If agents are sharing context unexpectedly:')
    
    session_fixes = '''# Verify agent directories are separate
ls -la ~/.openclaw/agents/

# Check session keys
openclaw sessions --json | jq '.[] | .sessionKey'

# Ensure agentDir is unique per agent in config'''
    add_code_block(doc, session_fixes, "bash")
    
    doc.add_heading('8.3 Sandbox Container Issues', 2)
    doc.add_paragraph('If sandboxed execution fails:')
    
    sandbox_fixes = '''# List sandbox containers
openclaw sandbox list

# Recreate containers after config changes
openclaw sandbox recreate --all

# Check sandbox policy
openclaw sandbox explain --agent code-review'''
    add_code_block(doc, sandbox_fixes, "bash")
    
    doc.add_heading('8.4 Sub-Agent Communication Failures', 2)
    doc.add_paragraph('If sub-agents aren\'t announcing results:')
    
    subagent_fixes = '''# Check sub-agent status
/subagents list

# View sub-agent logs
/subagents log <id> 50 tools

# Verify allowAgents configuration
# Ensure target agent is in the allowlist'''
    add_code_block(doc, subagent_fixes, "bash")
    
    doc.add_page_break()
    
    # Section 9: Conclusion
    doc.add_heading('9. Conclusion', 1)
    
    doc.add_paragraph(
        'OpenClaw provides a robust foundation for building sophisticated multi-agent coding teams. '
        'By leveraging its workspace isolation, flexible routing, and specialized skills, development '
        'teams can create AI assistants that handle code reviews, PR management, testing, and '
        'documentation with minimal human oversight.'
    )
    
    doc.add_paragraph(
        'Key takeaways from this guide:'
    )
    
    takeaways = [
        'Start with a single well-configured agent before scaling to multiple agents',
        'Use the hierarchical pattern: Opus orchestrator with Sonnet workers',
        'Maintain strict isolation between agents for security and clarity',
        'Route write operations through human approval workflows',
        'Enable sandboxing for code execution tasks',
        'Use read-only GitHub integration with notification-based feedback loops'
    ]
    for t in takeaways:
        doc.add_paragraph(t, style='List Bullet')
    
    doc.add_paragraph(
        'For ongoing support and updates, refer to the official OpenClaw documentation at '
        'docs.openclaw.ai and the ClawHub skill registry at clawhub.com.'
    )
    
    doc.add_page_break()
    
    # Section 10: References
    doc.add_heading('10. References', 1)
    
    refs = [
        ('OpenClaw Official Documentation', 'https://docs.openclaw.ai'),
        ('Multi-Agent Routing Guide', 'https://docs.openclaw.ai/concepts/multi-agent'),
        ('Agent Workspace Documentation', 'https://docs.openclaw.ai/concepts/agent-workspace'),
        ('Skills System Documentation', 'https://docs.openclaw.ai/tools/skills'),
        ('ClawHub Skill Registry', 'https://clawhub.com'),
        ('GitHub CLI Documentation', 'https://cli.github.com/manual/'),
        ('OpenClaw GitHub Repository', 'https://github.com/openclaw/openclaw'),
        ('Awesome OpenClaw Skills Collection', 'https://github.com/VoltAgent/awesome-openclaw-skills')
    ]
    
    for title, url in refs:
        p = doc.add_paragraph()
        p.add_run(f'{title}: ').bold = True
        p.add_run(url)
    
    # Add page numbers
    for section in doc.sections:
        add_page_number(section)
    
    # Save document
    output_path = '/home/clawdbot/clawd/Using_OpenClaw_to_Set_Up_a_Coding_Team.docx'
    doc.save(output_path)
    print(f"Document saved to: {output_path}")
    return output_path

if __name__ == "__main__":
    main()
