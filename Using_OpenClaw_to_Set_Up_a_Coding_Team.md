*A Comprehensive Guide to Multi-Agent Architecture for Development
Teams*

Version 1.0\
February 04, 2026\
\
OpenClaw Documentation Series

# Table of Contents

1\. Executive Summary

2\. Introduction to OpenClaw Multi-Agent Setup

3\. Architecture Overview

4\. Step-by-Step Setup Guide

5\. Example Configurations

6\. Agent Role Use Cases

7\. Best Practices & Security Considerations

8\. Troubleshooting Common Issues

9\. Conclusion

10\. References

# 1. Executive Summary

OpenClaw is an open-source, locally-running AI assistant framework that
enables sophisticated multi-agent architectures for software development
teams. This document provides a comprehensive guide to setting up a
coding team using OpenClaw\'s multi-agent capabilities, including agent
routing, session isolation, workspace configuration, and specialized
coding skills.

Key benefits of using OpenClaw for coding teams include:

-   Complete isolation between agents with separate workspaces,
    sessions, and authentication profiles

-   Flexible routing of messages from different channels to specialized
    agents

-   Hierarchical orchestration patterns with Opus-level orchestrators
    and specialized workers

-   Integration with GitHub CLI for automated code reviews and PR
    management

-   Sandboxed execution environments for secure code execution

-   Persistent memory and context across sessions for continuous
    development workflows

This guide covers everything from initial setup to advanced multi-agent
orchestration patterns, with practical examples and configuration
snippets that can be directly applied to your development workflow.

# 2. Introduction to OpenClaw Multi-Agent Setup

## 2.1 What is OpenClaw?

OpenClaw (formerly known as Moltbot and Clawdbot) is a locally-running
AI assistant that operates directly on your machine. It integrates with
multiple AI model providers (Anthropic Claude, OpenAI, Google Gemini,
and others) and can be accessed through various messaging platforms
including Telegram, Discord, Slack, WhatsApp, and more.

## 2.2 What Constitutes One Agent?

An agent in OpenClaw is not just a model or a prompt---it\'s a complete
isolated environment with three core components:

-   Workspace and Agent Directory

Every agent operates from its own workspace, typically defined by an
agentDir path. This directory contains the agent\'s configuration files
(AGENTS.md, SOUL.md, USER.md, TOOLS.md), memory storage, and skill
definitions. Two agents cannot accidentally share or overwrite each
other\'s files unless explicitly configured.

-   Session Store

Each agent maintains its own conversation history and session state
under \~/.openclaw/agents/\<agentId\>/sessions. This isolation means
Agent A\'s conversation with a user on Discord remains completely
separate from Agent B\'s conversation with the same user on Telegram.

-   Authentication Context

Authentication profiles are strictly per-agent. When you configure API
keys, OAuth tokens, or service credentials for one agent, these
credentials belong exclusively to that agent. This design prevents
credential leakage between agents.

## 2.3 When Multi-Agent Architecture Makes Sense

Multi-agent configurations become valuable when you have genuinely
distinct operational domains that benefit from isolation:

**Different Security Contexts:** When some conversations require access
to sensitive systems while others should remain sandboxed

**Specialized Expertise:** Running a research agent alongside a coding
agent allows each to optimize for its specific domain

**Channel-Specific Behaviors:** Different personalities or capabilities
on different channels

**Resource Management:** Different agents can use different models (Opus
for complex reasoning, Sonnet for routine tasks)

# 3. Architecture Overview

## 3.1 System Architecture Diagram

The following describes the OpenClaw multi-agent architecture for a
coding team:

> ┌─────────────────────────────────────────────────────────────────────┐\
> │ OpenClaw Gateway │\
> │ ┌─────────────────────────────────────────────────────────────┐ │\
> │ │ Message Router │ │\
> │ │ (Bindings: channel → accountId → peer → agentId) │ │\
> │ └─────────────────────────────────────────────────────────────┘ │\
> │ │ │\
> │ ┌────────────────────────┼────────────────────────┐ │\
> │ ▼ ▼ ▼ │\
> │ ┌─────────┐ ┌─────────────┐ ┌──────────┐ │\
> │ │ Orchestrator │ │ Code Review │ │ PR Agent │ │\
> │ │ Agent │ │ Agent │ │ │ │\
> │ │ (Opus 4.5) │ │ (Sonnet 4.5)│ │(Sonnet) │ │\
> │ └─────────┘ └─────────────┘ └──────────┘ │\
> │ │ │ │ │\
> │ ┌────┴────┐ ┌─────┴─────┐ ┌──────┴──────┐ │\
> │ │Workspace│ │ Workspace │ │ Workspace │ │\
> │ │ -orch │ │ -review │ │ -pr │ │\
> │ └─────────┘ └───────────┘ └─────────────┘ │\
> └─────────────────────────────────────────────────────────────────────┘

## 3.2 Core Components

  -------------------------------------------------------------------------------
  **Component**                       **Description**
  ----------------------------------- -------------------------------------------
  Gateway                             The WebSocket server that handles channels,
                                      nodes, sessions, and hooks. Runs on port
                                      18789 by default.

  Agents                              Isolated AI personas with their own
                                      workspace, sessions, and authentication.

  Bindings                            Routing rules that map incoming messages to
                                      specific agents based on channel, account,
                                      and peer.

  Workspaces                          The agent\'s home directory containing
                                      configuration files, memory, and skills.

  Skills                              Extensions that teach the agent how to use
                                      tools (AgentSkills-compatible).

  Sessions                            Conversation history and state, stored
                                      per-agent under
                                      \~/.openclaw/agents/\<agentId\>/sessions.
  -------------------------------------------------------------------------------

## 3.3 Path Structure

Understanding the file structure is essential for multi-agent setups:

> \~/.openclaw/\
> ├── openclaw.json \# Main configuration file\
> ├── credentials/ \# OAuth tokens, API keys\
> ├── skills/ \# Managed/shared skills\
> ├── workspace/ \# Default agent workspace\
> ├── workspace-\<agentId\>/ \# Per-agent workspaces\
> └── agents/\
> └── \<agentId\>/\
> ├── agent/ \# Per-agent auth profiles\
> └── sessions/ \# Session transcripts

# 4. Step-by-Step Setup Guide

## 4.1 Prerequisites

Before setting up a multi-agent coding team, ensure you have:

-   Node.js v18+ installed

-   Docker installed (for sandboxing)

-   OpenClaw installed: npm install -g openclaw

-   API keys for your preferred AI providers (Anthropic, OpenAI, etc.)

-   GitHub CLI (gh) installed and authenticated for code integration

## 4.2 Initial Configuration

Run the initial setup:

> \# Initialize OpenClaw\
> openclaw onboard \--install-daemon\
> \
> \# Configure API keys\
> openclaw configure \--section keys\
> \
> \# Verify installation\
> openclaw doctor

## 4.3 Creating Agent Workspaces

Create separate workspaces for each coding team agent:

> \# Create orchestrator agent\
> openclaw agents add orchestrator \--workspace
> \~/.openclaw/workspace-orchestrator\
> \
> \# Create code review agent\
> openclaw agents add code-review \--workspace
> \~/.openclaw/workspace-code-review\
> \
> \# Create PR agent\
> openclaw agents add pr-agent \--workspace
> \~/.openclaw/workspace-pr-agent\
> \
> \# Create testing agent\
> openclaw agents add testing \--workspace
> \~/.openclaw/workspace-testing\
> \
> \# Create documentation agent\
> openclaw agents add docs \--workspace \~/.openclaw/workspace-docs\
> \
> \# Verify agents\
> openclaw agents list \--bindings

## 4.4 Workspace File Structure

Each workspace should contain the following files:

  -----------------------------------------------------------------------
  **File**                            **Purpose**
  ----------------------------------- -----------------------------------
  AGENTS.md                           Operating instructions for the
                                      agent and how it should use memory

  SOUL.md                             Persona, tone, and behavioral
                                      boundaries

  USER.md                             Information about who the user is
                                      and how to address them

  TOOLS.md                            Notes about local tools and
                                      conventions (guidance only)

  IDENTITY.md                         The agent\'s name, theme, and emoji

  skills/                             Workspace-specific skills that
                                      override shared skills
  -----------------------------------------------------------------------

# 5. Example Configurations

## 5.1 Complete gateway.yaml for Multi-Agent Coding Team

The following configuration sets up a complete coding team with
specialized agents:

> {\
> // Multi-Agent Coding Team Configuration\
> agents: {\
> list: \[\
> {\
> id: \"orchestrator\",\
> default: true,\
> name: \"Orchestrator\",\
> workspace: \"\~/.openclaw/workspace-orchestrator\",\
> agentDir: \"\~/.openclaw/agents/orchestrator/agent\",\
> model: \"anthropic/claude-opus-4-5\",\
> identity: {\
> name: \"Team Lead\",\
> theme: \"Technical architect and coordinator\",\
> emoji: \"🎯\"\
> },\
> subagents: {\
> allowAgents: \[\"code-review\", \"pr-agent\", \"testing\", \"docs\"\]\
> }\
> },\
> {\
> id: \"code-review\",\
> name: \"Code Reviewer\",\
> workspace: \"\~/.openclaw/workspace-code-review\",\
> agentDir: \"\~/.openclaw/agents/code-review/agent\",\
> model: \"anthropic/claude-sonnet-4-5\",\
> identity: {\
> name: \"Code Reviewer\",\
> theme: \"Senior engineer focused on code quality\",\
> emoji: \"🔍\"\
> },\
> sandbox: {\
> mode: \"all\",\
> scope: \"agent\"\
> },\
> tools: {\
> allow: \[\"group:fs\", \"group:runtime\", \"group:web\",
> \"group:sessions\"\]\
> }\
> },\
> {\
> id: \"pr-agent\",\
> name: \"PR Manager\",\
> workspace: \"\~/.openclaw/workspace-pr-agent\",\
> agentDir: \"\~/.openclaw/agents/pr-agent/agent\",\
> model: \"anthropic/claude-sonnet-4-5\",\
> identity: {\
> name: \"PR Bot\",\
> theme: \"GitHub PR and issue management specialist\",\
> emoji: \"🔀\"\
> }\
> },\
> {\
> id: \"testing\",\
> name: \"Test Engineer\",\
> workspace: \"\~/.openclaw/workspace-testing\",\
> agentDir: \"\~/.openclaw/agents/testing/agent\",\
> model: \"anthropic/claude-sonnet-4-5\",\
> identity: {\
> name: \"Test Bot\",\
> theme: \"Quality assurance and test automation\",\
> emoji: \"🧪\"\
> },\
> sandbox: {\
> mode: \"all\",\
> scope: \"agent\",\
> docker: {\
> setupCommand: \"apt-get update && apt-get install -y nodejs npm\"\
> }\
> }\
> },\
> {\
> id: \"docs\",\
> name: \"Documentation Writer\",\
> workspace: \"\~/.openclaw/workspace-docs\",\
> agentDir: \"\~/.openclaw/agents/docs/agent\",\
> model: \"anthropic/claude-sonnet-4-5\",\
> identity: {\
> name: \"Docs Bot\",\
> theme: \"Technical documentation specialist\",\
> emoji: \"📚\"\
> }\
> }\
> \]\
> },\
> \
> // Route messages to appropriate agents\
> bindings: \[\
> // Slack: #code-review channel → code-review agent\
> {\
> agentId: \"code-review\",\
> match: {\
> channel: \"slack\",\
> peer: { kind: \"channel\", id: \"C0123REVIEW\" }\
> }\
> },\
> // Slack: #pull-requests channel → pr-agent\
> {\
> agentId: \"pr-agent\",\
> match: {\
> channel: \"slack\",\
> peer: { kind: \"channel\", id: \"C0123PR\" }\
> }\
> },\
> // Telegram: Testing group → testing agent\
> {\
> agentId: \"testing\",\
> match: {\
> channel: \"telegram\",\
> peer: { kind: \"group\", id: \"-1001234567890\" }\
> }\
> },\
> // Default: All other messages → orchestrator\
> {\
> agentId: \"orchestrator\",\
> match: { channel: \"telegram\" }\
> },\
> {\
> agentId: \"orchestrator\",\
> match: { channel: \"slack\" }\
> }\
> \],\
> \
> // Tool configuration\
> tools: {\
> profile: \"coding\",\
> web: {\
> search: { enabled: true },\
> fetch: { enabled: true }\
> },\
> exec: {\
> pathPrepend: \[\"\~/.local/bin\", \"/usr/local/bin\"\]\
> }\
> },\
> \
> // Sub-agent configuration\
> agents: {\
> defaults: {\
> subagents: {\
> maxConcurrent: 4,\
> archiveAfterMinutes: 120,\
> model: \"anthropic/claude-sonnet-4-5\"\
> }\
> }\
> }\
> }

## 5.2 SOUL.md Example for Code Review Agent

The SOUL.md file defines the agent\'s persona and behavioral guidelines:

> \# SOUL.md - Code Review Agent\
> \
> \## Identity\
> You are the Code Review Agent, a meticulous senior engineer dedicated
> to\
> maintaining code quality across the team\'s repositories.\
> \
> \## Personality\
> - Thorough but not pedantic\
> - Constructive in feedback\
> - Focus on patterns, not just syntax\
> - Praise good practices as well as identify issues\
> \
> \## Core Responsibilities\
> 1. Review pull request diffs for code quality issues\
> 2. Check for security vulnerabilities and anti-patterns\
> 3. Verify adherence to project coding standards\
> 4. Suggest improvements and optimizations\
> 5. Report findings to the team via notifications\
> \
> \## Communication Style\
> - Be specific with line numbers and file references\
> - Provide rationale for suggested changes\
> - Include code examples when proposing alternatives\
> - Prioritize issues by severity (Critical \> Major \> Minor)\
> \
> \## Boundaries\
> - Do NOT merge PRs or push code directly\
> - Do NOT access production systems\
> - Always recommend human review for critical changes\
> - Escalate security concerns to the orchestrator agent\
> \
> \## Tools You Use\
> - GitHub CLI (gh) for fetching PR details and diffs\
> - Web search for researching best practices\
> - File tools for reading local documentation\
> - Message tool for sending review summaries

## 5.3 AGENTS.md Example for Orchestrator

> \# AGENTS.md - Orchestrator Agent\
> \
> \## Role\
> You are the Team Lead orchestrator. You coordinate work between
> specialized\
> agents and handle complex tasks that require multiple perspectives.\
> \
> \## Sub-Agent Coordination\
> Use \`sessions_spawn\` to delegate tasks to specialized agents:\
> \
> - \*\*code-review\*\*: For PR review tasks\
> - \*\*pr-agent\*\*: For GitHub operations\
> - \*\*testing\*\*: For test creation and execution\
> - \*\*docs\*\*: For documentation updates\
> \
> \## Workflow Patterns\
> \
> \### New Feature Review\
> 1. Spawn code-review agent to analyze the diff\
> 2. Wait for review completion\
> 3. Spawn testing agent to verify test coverage\
> 4. Consolidate findings and report to the team\
> \
> \### Bug Triage\
> 1. Analyze the bug report\
> 2. Spawn pr-agent to find related issues/PRs\
> 3. If fix exists, coordinate with code-review\
> 4. If new, create an issue via pr-agent\
> \
> \## Memory Management\
> - Log significant decisions in memory/YYYY-MM-DD.md\
> - Update MEMORY.md with recurring patterns and team preferences\
> - Reference past decisions when making similar choices

# 6. Agent Role Use Cases

## 6.1 Code Review Agent

The Code Review Agent specializes in analyzing pull request diffs and
providing constructive feedback without requiring write access to
repositories.

> Workflow:
>
> \# Automated Code Review Workflow\
> \
> \## 1. Fetch PR details\
> gh pr view 123 \--json title,body,files,additions,deletions\
> \
> \## 2. Get the diff\
> gh pr diff 123\
> \
> \## 3. Analyze and generate review summary\
> \# Agent processes the diff and creates structured feedback\
> \
> \## 4. Send notification to team\
> \# Message sent via Slack/Telegram with:\
> \# - Summary of changes\
> \# - Issues found (categorized by severity)\
> \# - Suggestions for improvement\
> \# - Direct link to PR

## 6.2 PR Agent

The PR Agent handles GitHub operations including issue creation, PR
monitoring, and workflow status checks.

> \# PR Agent Capabilities\
> \
> \## List open PRs needing review\
> gh pr list \--state open \--json number,title,author,reviewDecision\
> \
> \## Check CI/CD status\
> gh run list \--limit 5 \--json status,conclusion,name\
> \
> \## Create issues from bug reports\
> gh issue create \--title \"Bug: \...\" \--body \"\...\" \--label bug\
> \
> \## Monitor for failed workflows\
> gh run list \--status failure \--json databaseId,name,conclusion \|
> \\\
> jq \'.\[\] \| select(.conclusion == \"failure\")\'\
> \
> \## Notify team of stale PRs (\>3 days old)\
> gh pr list \--json number,title,createdAt \| \\\
> jq \'\[.\[\] \| select((.createdAt \| fromdateiso8601) \< (now -
> 259200))\]\'

## 6.3 Testing Agent

The Testing Agent creates and runs tests, analyzes coverage, and
identifies gaps in test suites.

> \# Testing Agent Responsibilities\
> \
> \## Run test suite\
> npm test \-- \--coverage\
> \
> \## Analyze test results\
> \# Parse coverage reports and identify:\
> \# - Files with low coverage (\<80%)\
> \# - Uncovered branches\
> \# - Missing edge case tests\
> \
> \## Generate test suggestions\
> \# For each uncovered function, suggest test cases:\
> \# - Happy path scenarios\
> \# - Error handling\
> \# - Boundary conditions\
> \
> \## Create test files\
> \# Write new test files following project conventions

## 6.4 Documentation Agent

The Documentation Agent maintains README files, API documentation, and
inline code comments.

# 7. Best Practices & Security Considerations

## 7.1 Security Best Practices

**Use Read-Only GitHub Access:** Give agents read-only access to
repositories. Route all write operations through human approval.

**Enable Sandboxing:** Run coding agents in Docker containers to isolate
code execution.

**Separate Auth Profiles:** Never share authentication credentials
between agents. Each agent should have its own API keys.

**Branch Protection:** Enable GitHub branch protection rules as a safety
net regardless of agent permissions.

**Audit Logging:** Enable the command-logger hook to track all agent
actions.

## 7.2 Performance Optimization

-   Use Claude Opus for orchestration and complex reasoning tasks

-   Use Claude Sonnet for routine operations like code review

-   Set appropriate maxConcurrent limits for sub-agents

-   Archive sessions after completion to manage storage

-   Use tool profiles (coding, messaging, minimal) to reduce token
    overhead

## 7.3 Workspace Organization

-   Keep each agent\'s workspace in version control (private repo)

-   Use consistent naming conventions across workspaces

-   Document agent capabilities in IDENTITY.md

-   Regularly update MEMORY.md with lessons learned

-   Create shared skills in \~/.openclaw/skills/ for common
    functionality

# 8. Troubleshooting Common Issues

## 8.1 Agent Routing Issues

If messages are going to the wrong agent:

> \# Check current bindings\
> openclaw agents list \--bindings\
> \
> \# Verify binding order (most-specific first)\
> \# Peer bindings \> channel+accountId \> channel-only\
> \
> \# Test routing with debug mode\
> openclaw gateway \--verbose

## 8.2 Session Isolation Problems

If agents are sharing context unexpectedly:

> \# Verify agent directories are separate\
> ls -la \~/.openclaw/agents/\
> \
> \# Check session keys\
> openclaw sessions \--json \| jq \'.\[\] \| .sessionKey\'\
> \
> \# Ensure agentDir is unique per agent in config

## 8.3 Sandbox Container Issues

If sandboxed execution fails:

> \# List sandbox containers\
> openclaw sandbox list\
> \
> \# Recreate containers after config changes\
> openclaw sandbox recreate \--all\
> \
> \# Check sandbox policy\
> openclaw sandbox explain \--agent code-review

## 8.4 Sub-Agent Communication Failures

If sub-agents aren\'t announcing results:

> \# Check sub-agent status\
> /subagents list\
> \
> \# View sub-agent logs\
> /subagents log \<id\> 50 tools\
> \
> \# Verify allowAgents configuration\
> \# Ensure target agent is in the allowlist

# 9. Conclusion

OpenClaw provides a robust foundation for building sophisticated
multi-agent coding teams. By leveraging its workspace isolation,
flexible routing, and specialized skills, development teams can create
AI assistants that handle code reviews, PR management, testing, and
documentation with minimal human oversight.

Key takeaways from this guide:

-   Start with a single well-configured agent before scaling to multiple
    agents

-   Use the hierarchical pattern: Opus orchestrator with Sonnet workers

-   Maintain strict isolation between agents for security and clarity

-   Route write operations through human approval workflows

-   Enable sandboxing for code execution tasks

-   Use read-only GitHub integration with notification-based feedback
    loops

For ongoing support and updates, refer to the official OpenClaw
documentation at docs.openclaw.ai and the ClawHub skill registry at
clawhub.com.

# 10. References

**OpenClaw Official Documentation:** https://docs.openclaw.ai

**Multi-Agent Routing Guide:**
https://docs.openclaw.ai/concepts/multi-agent

**Agent Workspace Documentation:**
https://docs.openclaw.ai/concepts/agent-workspace

**Skills System Documentation:** https://docs.openclaw.ai/tools/skills

**ClawHub Skill Registry:** https://clawhub.com

**GitHub CLI Documentation:** https://cli.github.com/manual/

**OpenClaw GitHub Repository:** https://github.com/openclaw/openclaw

**Awesome OpenClaw Skills Collection:**
https://github.com/VoltAgent/awesome-openclaw-skills
