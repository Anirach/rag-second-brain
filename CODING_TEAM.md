# CODING_TEAM.md — Multi-Agent Software Development Workflow

When Anirach asks for coding work, coordinate specialized sub-agents for professional, thoughtful software implementation.

## Philosophy

> "First, solve the problem. Then, write the code." — John Johnson

Before spawning agents, I (Arthur) act as the **initial requirements analyst**:
1. Understand what's really being asked
2. Break down into actionable tasks
3. Identify dependencies and order
4. Delegate to the right specialists

## 📋 Project Assessment Mode

The Coding Team can perform **comprehensive project assessments** - analyzing and reporting without implementing changes. Perfect for understanding a codebase before making decisions.

### Assessment Report Structure

```markdown
# Project Assessment Report
## [Project Name]
Generated: [Date]

---

## 📊 Executive Summary
- Overall Health Score: [X/10]
- Critical Issues: [N]
- Recommended Priority Actions: [List]

---

## 🐛 Bug Analysis
### Critical Bugs
- [Bug 1]: Location, Impact, Suggested Fix
- [Bug 2]: ...

### Potential Issues
- [Issue 1]: Risk level, Location
- ...

### Bug Prevention Recommendations
- [Recommendation 1]
- ...

---

## ⚡ Performance Assessment
### Current Metrics
- Bundle Size: [X KB]
- Load Time: [X ms] (estimated)
- Memory Usage: [Assessment]

### Bottlenecks Identified
| Location | Issue | Impact | Priority |
|----------|-------|--------|----------|
| ... | ... | ... | ... |

### Optimization Opportunities
- [Opportunity 1]: Expected improvement
- ...

---

## 🧹 Code Quality Review
### Quality Score: [X/10]

### Code Smells Found
| Type | Count | Locations |
|------|-------|-----------|
| Long functions | X | file1.ts, file2.ts |
| Deep nesting | X | ... |
| Duplicated code | X | ... |
| ... | ... | ... |

### Technical Debt
- [Debt item 1]: Effort to fix, Priority
- ...

### Refactoring Recommendations
- [Recommendation 1]: Files affected, Complexity
- ...

---

## 🔒 Security Audit
### Security Score: [X/10]

### Vulnerabilities Found
| Severity | Issue | Location | OWASP Category |
|----------|-------|----------|----------------|
| Critical | ... | ... | ... |
| High | ... | ... | ... |
| Medium | ... | ... | ... |

### Security Recommendations
- [Recommendation 1]
- ...

### Dependency Vulnerabilities
- [Package]: Version, CVE, Recommended version
- ...

---

## 🎨 UI/UX Review
### UX Score: [X/10]

### Usability Issues
| Page/Component | Issue | Severity | Suggestion |
|----------------|-------|----------|------------|
| ... | ... | ... | ... |

### Accessibility (a11y) Issues
- [Issue 1]: WCAG guideline, Location
- ...

### Design Consistency
- Inconsistencies found: [List]
- Missing patterns: [List]

### UI Improvement Recommendations
- [Recommendation 1]
- ...

---

## ✨ Feature Gap Analysis
### Missing Common Features
- [Feature 1]: Importance, Effort estimate
- ...

### Enhancement Opportunities
- [Enhancement 1]: User value, Complexity
- ...

### Integration Opportunities
- [Integration 1]: Benefit
- ...

---

## 📈 Test Coverage Analysis
### Current Coverage: [X%]

### Coverage by Area
| Area | Coverage | Critical Gaps |
|------|----------|---------------|
| API Routes | X% | ... |
| Components | X% | ... |
| Utils | X% | ... |

### Untested Critical Paths
- [Path 1]: Risk if broken
- ...

### Testing Recommendations
- [Recommendation 1]
- ...

---

## 📝 Documentation Review
### Documentation Score: [X/10]

### Existing Documentation
| Document | Status | Quality |
|----------|--------|---------|
| README | ✅/❌ | Good/Fair/Poor |
| API Docs | ✅/❌ | ... |
| Setup Guide | ✅/❌ | ... |
| Architecture | ✅/❌ | ... |

### Missing Documentation
- [Doc 1]: Importance, Suggested content
- ...

### Documentation Structure Recommendation
```
docs/
├── README.md              # Project overview, quick start
├── CONTRIBUTING.md        # How to contribute
├── ARCHITECTURE.md        # System design, data flow
├── API.md                 # API reference
├── DEPLOYMENT.md          # Deploy instructions
├── TROUBLESHOOTING.md     # Common issues
└── guides/
    ├── setup.md           # Detailed setup
    ├── development.md     # Dev workflow
    └── testing.md         # Testing guide
```

---

## 🎯 Prioritized Action Plan

### Immediate (This Week)
1. [Action 1] - Reason
2. [Action 2] - Reason

### Short-term (This Month)
1. [Action 1] - Reason
2. [Action 2] - Reason

### Long-term (This Quarter)
1. [Action 1] - Reason
2. [Action 2] - Reason

---

## 📊 Summary Scores

| Category | Score | Status |
|----------|-------|--------|
| Bugs | X/10 | 🔴/🟡/🟢 |
| Performance | X/10 | 🔴/🟡/🟢 |
| Code Quality | X/10 | 🔴/🟡/🟢 |
| Security | X/10 | 🔴/🟡/🟢 |
| UI/UX | X/10 | 🔴/🟡/🟢 |
| Test Coverage | X/10 | 🔴/🟡/🟢 |
| Documentation | X/10 | 🔴/🟡/🟢 |
| **Overall** | **X/10** | 🔴/🟡/🟢 |
```

### Assessment Trigger Commands

```
"Assess project /projects/my-app"
"Full audit of /projects/dashboard"
"Review codebase at /home/user/project"
"Evaluate code quality of [repo]"
```

### Assessment Agent Assignment

| Assessment Area | Primary Agent | Expertise |
|-----------------|---------------|-----------|
| Full Assessment | `code-review` | Comprehensive analysis |
| Bug Analysis | `code-review` | Issue detection |
| Performance | `orchestrator` | Profiling & optimization |
| Code Quality | `code-review` | Clean code, patterns |
| Security | `code-review` | Vulnerabilities, OWASP |
| UI/UX | `ux-designer` | Usability, design |
| Test Coverage | `testing` | Test analysis |
| Documentation | `docs` | Doc structure |

### Output Options

| Format | Use Case |
|--------|----------|
| **Markdown Report** | Full detailed assessment |
| **DOCX Document** | Formal deliverable |
| **Summary Only** | Quick overview |
| **Single Area** | Focused audit |

---

## 🔧 Working with Existing Codebases

The Coding Team excels at both **new development** and **existing codebase work**.

### Working Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| **🔨 Solo Mode** | One agent handles everything | Small fixes, single feature, solo work |
| **👥 Team Mode** | Multiple agents coordinate | Large projects, parallel work, team collaboration |

---

## 🔨 Solo Mode (Fix All)

One agent owns the entire task from start to finish.

```
Arthur → spawn orchestrator → Analyze → Fix All → Test → Deliver
```

**Best for:**
- Single bug fixes
- One feature addition
- Focused refactoring
- Security patches

**Trigger:**
```
"Fix all the performance issues in /projects/app"
"Add user authentication to /projects/api"
"Clean up the entire codebase"
```

---

## 👥 Team Mode (Collaborative)

Multiple agents work on the same project **without conflicts**.

### Coordination Protocol

```
┌─────────────────────────────────────────────────────────────┐
│                    TEAM MODE WORKFLOW                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Arthur receives multi-part task                          │
│         │                                                    │
│         ▼                                                    │
│  2. Create PROJECT_LOCK.md in project root                   │
│     ├── Lists all active agents                              │
│     ├── Assigns file/folder ownership                        │
│     └── Tracks work status                                   │
│         │                                                    │
│         ▼                                                    │
│  3. Spawn agents with SCOPED assignments                     │
│     ├── Agent A: owns /src/api/* (backend)                   │
│     ├── Agent B: owns /src/components/* (frontend)           │
│     ├── Agent C: owns /src/utils/* (shared)                  │
│     └── Each agent knows their boundaries                    │
│         │                                                    │
│         ▼                                                    │
│  4. Agents work in FEATURE BRANCHES                          │
│     ├── feature/agent-a-api-improvements                     │
│     ├── feature/agent-b-ui-polish                            │
│     └── feature/agent-c-utils-refactor                       │
│         │                                                    │
│         ▼                                                    │
│  5. Integration Phase                                        │
│     ├── pr-agent merges branches sequentially                │
│     ├── Resolves any conflicts                               │
│     └── Runs full test suite                                 │
│         │                                                    │
│         ▼                                                    │
│  6. Remove PROJECT_LOCK.md                                   │
│         │                                                    │
│         ▼                                                    │
│  7. Deliver unified result                                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### PROJECT_LOCK.md Template

```markdown
# PROJECT_LOCK.md
**Project:** /projects/my-app
**Created:** 2026-02-11 22:30
**Coordinator:** Arthur (main)

## Active Agents

| Agent | Session | Owned Paths | Status | Branch |
|-------|---------|-------------|--------|--------|
| orchestrator | abc123 | /src/api/*, /src/services/* | 🔄 Working | feature/api-perf |
| ux-designer | def456 | /src/components/*, /src/styles/* | 🔄 Working | feature/ui-polish |
| testing | ghi789 | /tests/*, /__tests__/* | ⏳ Waiting | feature/test-coverage |

## Shared Files (Coordinate Before Editing)
- package.json → Notify all agents
- tsconfig.json → Notify all agents
- .env.example → Notify all agents

## Communication Log
- [22:30] orchestrator started API performance work
- [22:31] ux-designer started UI polish
- [22:35] orchestrator needs to update package.json - notified all
- [22:36] ux-designer acknowledged

## Integration Order
1. orchestrator (backend first)
2. ux-designer (frontend second)
3. testing (tests last)

## Status: 🔄 IN PROGRESS
```

### Conflict Prevention Rules

```
✅ DO:
- Work only in assigned paths
- Use feature branches
- Communicate before touching shared files
- Pull latest before starting
- Run tests before committing

❌ DON'T:
- Edit files outside your scope
- Push directly to main
- Modify shared configs without notice
- Ignore other agents' work
```

### Scoped Task Brief Template

```markdown
## Task: [Specific Area]

### Your Scope
**Owned Paths:**
- /src/api/*
- /src/services/*

**Off Limits:**
- /src/components/* (owned by ux-designer)
- /tests/* (owned by testing)

### Coordination
**Shared Files:** Check PROJECT_LOCK.md before editing
**Branch:** feature/your-branch-name
**Integration Order:** You are #1

### Your Deliverables
- [ ] Complete work in owned paths
- [ ] All tests pass for your code
- [ ] No changes outside scope
- [ ] Ready for integration

### Communication
If you need to modify shared files:
1. Note it in PROJECT_LOCK.md
2. Wait for acknowledgment
3. Then proceed
```

### Team Mode Triggers

```
"Improve /projects/app with multiple agents:
 - orchestrator: fix backend performance
 - ux-designer: polish the UI
 - testing: add test coverage"

"Split work on /projects/dashboard:
 - API improvements
 - Frontend redesign
 - Documentation update"

"Parallel improvement of /projects/system:
 - Security fixes (code-review)
 - Performance (orchestrator)
 - UI (ux-designer)"
```

### Integration Workflow

```
1. All agents complete their branches
         │
         ▼
2. pr-agent takes over
         │
         ▼
3. Merge in specified order:
   main ← feature/backend
        ← feature/frontend  
        ← feature/tests
         │
         ▼
4. Resolve conflicts (if any)
         │
         ▼
5. Run full test suite
         │
         ▼
6. Final verification
         │
         ▼
7. Clean up branches & lock file
```

---

## 🤝 Human Collaboration Mode

Work alongside human developers on the same project. Agents handle their assigned parts, humans handle theirs, everything integrates via Git/PR workflow.

### Collaboration Workflow

```
┌─────────────────────────────────────────────────────────────┐
│              HUMAN + AGENT COLLABORATION                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Human assigns task to Coding Team                        │
│     "Add payment integration while I work on user profiles"  │
│         │                                                    │
│         ▼                                                    │
│  2. Agent creates feature branch                             │
│     git checkout -b feature/agent-payment-integration        │
│         │                                                    │
│         ▼                                                    │
│  3. PARALLEL WORK                                            │
│     ┌─────────────────┬─────────────────┐                   │
│     │   🤖 AGENT      │   👤 HUMAN      │                   │
│     ├─────────────────┼─────────────────┤                   │
│     │ Payment API     │ User profiles   │                   │
│     │ Stripe setup    │ Avatar upload   │                   │
│     │ Webhook handler │ Settings page   │                   │
│     │ Tests           │ Their tests     │                   │
│     └─────────────────┴─────────────────┘                   │
│         │                                                    │
│         ▼                                                    │
│  4. Agent completes work                                     │
│     ├── Commits with clear messages                          │
│     ├── Pushes branch to remote                              │
│     └── Creates Pull Request                                 │
│         │                                                    │
│         ▼                                                    │
│  5. HUMAN REVIEW                                             │
│     ├── Reviews PR on GitHub/GitLab                          │
│     ├── Requests changes (if needed)                         │
│     ├── Agent addresses feedback                             │
│     └── Human approves & merges                              │
│         │                                                    │
│         ▼                                                    │
│  6. Integration complete                                     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Human Collaboration Triggers

```
"Work on /projects/app:
 - I'll handle the frontend
 - You add the API endpoints
 - Push for my review when done"

"Create feature branch and implement [FEATURE]
 - Push to GitHub when ready
 - I'll review and merge"

"Fix the backend bugs while I work on mobile app
 - Branch: feature/backend-fixes
 - Create PR when done"
```

### Branch Naming Convention

```
feature/agent-[description]     # New features
fix/agent-[description]         # Bug fixes
refactor/agent-[description]    # Refactoring
perf/agent-[description]        # Performance
docs/agent-[description]        # Documentation

Examples:
- feature/agent-payment-integration
- fix/agent-login-validation
- refactor/agent-api-cleanup
- perf/agent-query-optimization
```

### Commit Message Standards

```
type(scope): description

Types:
- feat: New feature
- fix: Bug fix
- refactor: Code refactoring
- perf: Performance improvement
- test: Adding tests
- docs: Documentation
- chore: Maintenance

Examples:
- feat(payment): add Stripe checkout integration
- fix(auth): resolve token refresh race condition
- refactor(api): simplify error handling middleware
- test(payment): add webhook handler tests
```

### Pull Request Template

```markdown
## 🤖 Agent Work Summary

### What was done
- [x] Implemented [feature/fix]
- [x] Added tests for [component]
- [x] Updated documentation

### Files Changed
| File | Change Type | Description |
|------|-------------|-------------|
| src/api/payment.ts | Added | Stripe integration |
| src/hooks/usePayment.ts | Added | Payment hook |
| tests/payment.test.ts | Added | Unit tests |

### Testing
- [ ] All existing tests pass
- [ ] New tests added: X tests
- [ ] Manual testing completed

### Screenshots (if UI changes)
[Include if applicable]

### How to Test
1. [Step 1]
2. [Step 2]
3. [Expected result]

### Notes for Reviewer
- [Any important notes]
- [Decisions made and why]
- [Areas that need extra attention]

### Checklist
- [ ] Code follows project conventions
- [ ] Tests pass locally
- [ ] No new warnings/errors
- [ ] Documentation updated
- [ ] Ready for review

---
*Generated by Coding Team Agent*
```

### Handling Review Feedback

When human requests changes:

```
┌─────────────────────────────────────────────────────────────┐
│              REVIEW FEEDBACK LOOP                            │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Human leaves review comments on PR                       │
│         │                                                    │
│         ▼                                                    │
│  2. Arthur receives notification                             │
│     "Review feedback on PR #42: [comments]"                  │
│         │                                                    │
│         ▼                                                    │
│  3. Agent addresses each comment                             │
│     ├── Makes requested changes                              │
│     ├── Responds to questions                                │
│     └── Pushes new commits                                   │
│         │                                                    │
│         ▼                                                    │
│  4. Notify human                                             │
│     "Addressed feedback, ready for re-review"                │
│         │                                                    │
│         ▼                                                    │
│  5. Repeat until approved                                    │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Communication Protocol

**Agent → Human:**
- "Started work on [branch], will notify when PR is ready"
- "PR created: [link] - ready for your review"
- "Addressed your feedback, please re-review"
- "Question: [clarification needed about requirement]"
- "Blocked: [issue that needs human decision]"

**Human → Agent:**
- "Work on [feature] in /projects/app"
- "Review comments on PR #42"
- "Change approach to [alternative]"
- "Approved, you can merge" or "I'll merge it"

### Git Operations Summary

| Action | Command | When |
|--------|---------|------|
| Create branch | `git checkout -b feature/agent-xxx` | Start work |
| Commit | `git commit -m "type(scope): msg"` | Save progress |
| Push | `git push origin feature/agent-xxx` | Share work |
| Create PR | `gh pr create --title "..." --body "..."` | Ready for review |
| Update PR | `git push` (after new commits) | Address feedback |
| Rebase | `git rebase main` | Stay up to date |

### Example Collaboration Scenarios

**Scenario 1: Feature Split**
```
Human: "I'm building the dashboard UI. You add the analytics API.
        Branch: feature/agent-analytics-api
        Push for review when done."

Agent: Creates branch → Implements API → Tests → PR → Waits for review
```

**Scenario 2: Bug Fix Support**
```
Human: "Critical bug in production. Fix the payment webhook
        while I handle customer communication.
        Branch: fix/agent-webhook-timeout
        Need it ASAP."

Agent: Hotfix branch → Quick fix → Tests → PR → Notifies human
```

**Scenario 3: Parallel Development**
```
Human: "Sprint planning:
        - You: User authentication (feature/agent-auth)
        - Me: Product catalog
        - Review each other's PRs"

Agent: Works independently → PR → Reviews human's PR too (if requested)
```

---

### Supported Operations

| Operation | Description | Primary Agent |
|-----------|-------------|---------------|
| **Bug Fix** | Diagnose and fix issues | `orchestrator` or `code-review` |
| **Performance** | Optimize speed, memory, bundle size | `orchestrator` |
| **Quality** | Refactor, clean code, reduce tech debt | `code-review` |
| **New Feature** | Add functionality to existing app | `orchestrator` |
| **UI Polish** | Improve UX/UI of existing screens | `ux-designer` |
| **Security Audit** | Find and fix vulnerabilities | `code-review` |
| **Test Coverage** | Add tests to existing code | `testing` |
| **Documentation** | Document existing codebase | `docs` |
| **Code Wiki** | Generate full codebase wiki (Google CodeWiki style) | `docs` |

### Codebase Analysis Workflow

When working with existing code, agents follow this process:

```
1. UNDERSTAND
   ├── Read README, package.json, project structure
   ├── Identify tech stack and patterns
   ├── Map key files and their purposes
   └── Note existing conventions

2. ANALYZE
   ├── Run existing tests (if any)
   ├── Check for lint errors
   ├── Identify problem areas
   └── Understand data flow

3. PLAN
   ├── Define scope of changes
   ├── Identify files to modify
   ├── Consider side effects
   └── Plan rollback strategy

4. IMPLEMENT
   ├── Make minimal, focused changes
   ├── Follow existing code style
   ├── Add/update tests
   └── Update documentation

5. VERIFY
   ├── Run full test suite
   ├── Check build passes
   ├── Manual smoke test
   └── Performance check (if relevant)
```

### Task Brief for Existing Codebase

```markdown
## Task: [Bug Fix / Performance / Feature / etc.]

### Project Location
- **Path:** /path/to/project
- **Repo:** https://github.com/user/repo (if applicable)
- **Branch:** main (or specify)

### Current State
[Describe what's happening now - the bug, the slow performance, etc.]

### Expected State
[Describe what should happen after the fix]

### Reproduction Steps (for bugs)
1. [Step 1]
2. [Step 2]
3. [See error/issue]

### Constraints
- [ ] Don't break existing functionality
- [ ] Follow existing code patterns
- [ ] Maintain backward compatibility
- [ ] Keep changes minimal and focused

### Acceptance Criteria
- [ ] Issue is resolved
- [ ] Tests pass
- [ ] No new lint errors
- [ ] Code reviewed
```

## Execution Environment

**All coding agents use Claude Code** as the underlying execution engine:

```bash
claude -p "task description" --dangerously-skip-permissions
```

Why Claude Code:
- ✅ Native file operations (read/write/edit)
- ✅ Shell command execution
- ✅ Git operations built-in
- ✅ Multi-file awareness
- ✅ Iterative problem solving

The sub-agent spawns Claude Code in the project directory and monitors output.

---

## 💻 Agent Roles & Skills Integration

### 🎭 Orchestrator
**For:** Complex multi-part features, architectural decisions, coordinating multiple specialists
**Personality:** Technical Lead with 10+ years experience. Plans before coding.
**Workspace:** `~/.openclaw/workspace-orchestrator`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **perf-profiler**, **debug-pro**, **backend-patterns**, **api-dev**, mermaid-architect

**Enhanced Capabilities with Skills:**
- 🔍 **perf-profiler**: Advanced performance analysis and bottleneck detection
- 🐛 **debug-pro**: Systematic debugging with advanced diagnostic tools
- 🏗️ **backend-patterns**: Apply proven architectural patterns
- 🚀 **api-dev**: Professional API design and development
- 📊 **mermaid-architect**: System architecture visualization

### 🔍 Code Review
**For:** Reviewing existing code, security audits, quality checks
**Personality:** Senior Engineer focused on bugs, security, maintainability
**Workspace:** `~/.openclaw/workspace-code-review`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **clean-code-review**, **security-auditor**, **perf-profiler**, debug-pro

**Enhanced Capabilities with Skills:**
- 🛡️ **security-auditor**: Comprehensive security vulnerability detection and remediation
- 📊 **clean-code-review**: Systematic code quality assessment with clean code principles
- ⚡ **perf-profiler**: Performance analysis and optimization recommendations
- 🐛 **debug-pro**: Advanced debugging and issue root cause analysis
- 🔧 Clean code pattern enforcement and refactoring guidance

### 🔀 PR Agent  
**For:** Git operations, creating PRs, merging, resolving conflicts
**Personality:** DevOps Engineer who values clean Git history
**Workspace:** `~/.openclaw/workspace-pr-agent`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **git-essentials**, clean-code-review

**Enhanced Capabilities with Skills:**
- 📚 **git-essentials**: Advanced Git workflow management and best practices
- 🔍 **clean-code-review**: Pre-merge code quality validation
- 🔄 Automated conflict resolution and merge optimization

### 🧪 Testing
**For:** Writing tests, running test suites, improving coverage
**Personality:** QA Engineer who tests edge cases obsessively
**Workspace:** `~/.openclaw/workspace-testing`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **e2e-testing-patterns**, clean-code-review, debug-pro

**Enhanced Capabilities with Skills:**
- 📈 **e2e-testing-patterns**: Comprehensive end-to-end testing strategies
- 🧪 **clean-code-review**: Test code quality and maintainability
- 🐛 **debug-pro**: Advanced test debugging and failure analysis
- ⚡ Performance test implementation and analysis

### 📝 Docs
**For:** README updates, API documentation, code comments
**Personality:** Technical Writer who makes code understandable
**Workspace:** `~/.openclaw/workspace-docs`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** mermaid-architect, diagram, markdown-converter, api-dev

**Enhanced Capabilities with Skills:**
- 📊 **mermaid-architect**: Technical documentation with flowcharts and diagrams
- 🎨 **diagram**: Custom documentation visuals
- 📄 **markdown-converter**: Multi-format documentation generation
- 🚀 **api-dev**: Professional API documentation standards
- 📖 **Code Wiki Generation**: Full codebase wiki (see [CODEWIKI_GUIDE.md](CODEWIKI_GUIDE.md))

**Code Wiki Task:** When spawned with "generate code wiki", produces a complete `docs/wiki/` directory with architecture diagrams, module docs, API reference, data flow, setup guide, and Mermaid diagrams — all cross-linked with file:line references. Run automatically after every project build.

### 🎨 UX/UI Designer
**For:** Interface design, user experience, visual polish, component styling
**Personality:** Senior Product Designer obsessed with usability and aesthetics. Anti-AI-slop crusader.
**Workspace:** `~/.openclaw/workspace-ux-designer`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **frontend-design-ultimate**, **shadcn-ui**, **ui-ux-pro-max**, mermaid-architect, diagram

**Enhanced Design Capabilities with Skills:**
- 🎨 **frontend-design-ultimate**: Advanced frontend design patterns and modern UI frameworks
- 🧩 **shadcn-ui**: Professional component library implementation
- 👤 **ui-ux-pro-max**: User experience optimization and design system creation
- 📊 **mermaid-architect**: User flow and information architecture diagrams
- 🔧 **diagram**: Custom UI mockups and design documentation

**Skill-Enhanced Design Principles:**
- ✅ **shadcn-ui**: Professional component consistency
- ✅ **ui-ux-pro-max**: Evidence-based UX decisions
- ✅ **frontend-design-ultimate**: Modern design pattern implementation
- ❌ NO generic frameworks without skill-enhanced customization
- ❌ NO UX decisions without ui-ux-pro-max validation

### 🔧 DevOps Engineer
**For:** CI/CD pipelines, deployment, Docker, infrastructure, environment management
**Personality:** Site Reliability Engineer who automates everything. "If you do it twice, automate it."
**Workspace:** `~/.openclaw/workspace-devops`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **docker-essentials**, **vercel**, **git-essentials**, security-auditor

**Enhanced DevOps Capabilities with Skills:**
- 🐳 **docker-essentials**: Professional containerization and orchestration
- 🚀 **vercel**: Optimized Vercel deployment and configuration
- 📚 **git-essentials**: Advanced Git workflow automation
- 🛡️ **security-auditor**: Infrastructure security assessment and hardening
- ⚙️ CI/CD pipeline optimization with security integration

### 🗄️ Database Specialist
**For:** Schema design, migrations, query optimization, data modeling
**Personality:** Data architect who thinks in relations and indexes. Performance obsessed.
**Workspace:** `~/.openclaw/workspace-db-specialist`
**Execution:** `claude --dangerously-skip-permissions`
**🆕 Key Skills:** **sql-toolkit**, **postgres**, perf-profiler, backend-patterns

**Enhanced Database Capabilities with Skills:**
- 📊 **sql-toolkit**: Advanced SQL query optimization and database utilities
- 🐘 **postgres**: PostgreSQL-specific optimization and feature utilization
- ⚡ **perf-profiler**: Database performance analysis and monitoring
- 🏗️ **backend-patterns**: Database design pattern implementation
- 🔒 Advanced security and performance optimization

---

## 🔧 Skills Integration Matrix

| Agent | Skill | When to Use | How It Helps |
|-------|-------|-------------|--------------|
| **orchestrator** | **perf-profiler** | Performance bottlenecks | Advanced profiling and optimization analysis |
| **orchestrator** | **debug-pro** | Complex debugging | Systematic debugging with diagnostic tools |
| **orchestrator** | **backend-patterns** | Architecture decisions | Apply proven backend architectural patterns |
| **orchestrator** | **api-dev** | API development | Professional API design and implementation |
| **code-review** | **clean-code-review** | Code quality assessment | Systematic clean code pattern enforcement |
| **code-review** | **security-auditor** | Security audits | Comprehensive vulnerability detection |
| **testing** | **e2e-testing-patterns** | End-to-end testing | Professional testing strategy implementation |
| **ux-designer** | **frontend-design-ultimate** | Modern UI development | Advanced frontend patterns and frameworks |
| **ux-designer** | **shadcn-ui** | Component libraries | Professional UI component implementation |
| **ux-designer** | **ui-ux-pro-max** | UX optimization | User experience validation and improvement |
| **devops** | **docker-essentials** | Containerization | Professional Docker workflows |
| **devops** | **vercel** | Deployment | Optimized Vercel configuration |
| **devops** | **git-essentials** | Git operations | Advanced Git workflow management |
| **db-specialist** | **sql-toolkit** | Database optimization | Advanced SQL utilities and optimization |
| **db-specialist** | **postgres** | PostgreSQL projects | PostgreSQL-specific features and optimization |
| **pr-agent** | **git-essentials** | Git operations | Professional Git workflow and merge management |
| **docs** | **mermaid-architect** | Technical documentation | Flowcharts and architecture diagrams |
| **docs** | **api-dev** | API documentation | Professional API documentation standards |

### Cross-Team Skills Available

| Skill | Source | Coding Team Use Case |
|-------|--------|---------------------|
| **markdown-converter** | Writing/Translation | Documentation format conversion |
| **content-quality-auditor** | Writing | Code documentation quality assessment |
| **latex** | Academic | Technical documentation formatting |
| **edge-tts** | Translation/Course | Audio documentation and tutorials |

---

## 🚀 Enhanced Development Workflows with Skills

### 1. Full-Stack Development with Skill Enhancement
```
"Arthur, build a complete user management system with skill-enhanced quality"

orchestrator + api-dev + backend-patterns → Professional API Design
    ↓
ux-designer + frontend-design-ultimate + shadcn-ui → Modern UI Implementation
    ↓
db-specialist + postgres + sql-toolkit → Optimized Database Layer
    ↓
testing + e2e-testing-patterns → Comprehensive Test Coverage
    ↓
code-review + clean-code-review + security-auditor → Quality Assurance
    ↓
devops + docker-essentials + vercel → Production Deployment
```

### 2. Performance Optimization Workflow
```
"Arthur, optimize application performance using advanced profiling"

orchestrator + perf-profiler → Performance Analysis
    ├── Frontend performance audit
    ├── Backend bottleneck detection
    ├── Database query optimization
    └── Bundle size analysis

code-review + perf-profiler → Performance Review
    ├── Code-level optimizations
    ├── Memory usage improvements
    └── Caching strategy implementation
```

### 3. Security-First Development
```
"Arthur, implement secure authentication with comprehensive security audit"

code-review + security-auditor → Security Assessment
    ├── Vulnerability scanning
    ├── OWASP compliance check
    └── Threat model analysis

orchestrator + api-dev + backend-patterns → Secure API Implementation
    ├── Authentication patterns
    ├── Authorization middleware
    └── Input validation

testing + e2e-testing-patterns → Security Testing
    ├── Authentication tests
    ├── Authorization tests
    └── Penetration testing scenarios
```

### 4. Modern UI/UX Development
```
"Arthur, create a modern, accessible user interface"

ux-designer + ui-ux-pro-max → UX Research & Design
    ├── User journey analysis
    ├── Accessibility requirements
    └── Design system planning

ux-designer + frontend-design-ultimate + shadcn-ui → Implementation
    ├── Modern component library
    ├── Responsive design patterns
    └── Performance-optimized frontend

testing + e2e-testing-patterns → UI/UX Testing
    ├── User flow testing
    ├── Accessibility testing
    └── Cross-browser compatibility
```

### 5. Professional Deployment Pipeline
```
"Arthur, set up production-ready deployment with monitoring"

devops + docker-essentials → Containerization
    ├── Multi-stage Docker builds
    ├── Container optimization
    └── Security scanning

devops + vercel + git-essentials → CI/CD Pipeline
    ├── Automated testing
    ├── Deployment automation
    └── Rollback procedures

orchestrator + perf-profiler → Production Monitoring
    ├── Performance monitoring
    ├── Error tracking
    └── Health checks
```

### Enhanced Command Examples

```bash
# Security-enhanced development
"Arthur, build a secure API using security-auditor and clean-code-review"

# Performance-optimized application
"Arthur, create an app using perf-profiler for optimization and postgres for database performance"

# Modern UI with professional patterns
"Arthur, build a frontend using frontend-design-ultimate, shadcn-ui, and ui-ux-pro-max"

# Full testing coverage
"Arthur, implement comprehensive testing using e2e-testing-patterns"

# Professional deployment
"Arthur, deploy using docker-essentials and vercel with git-essentials workflows"

# Database-optimized application
"Arthur, create a data-heavy app using sql-toolkit and postgres optimization"

# API-first development
"Arthur, design and implement APIs using api-dev and backend-patterns"

# Code quality focus
"Arthur, refactor this codebase using clean-code-review and debug-pro"
```

### Skill Combination Strategies

#### High-Performance Stack
- **orchestrator** (perf-profiler, debug-pro, backend-patterns)
- **db-specialist** (sql-toolkit, postgres)
- **ux-designer** (frontend-design-ultimate)
- **devops** (docker-essentials, vercel)

#### Security-Focused Stack  
- **code-review** (security-auditor, clean-code-review)
- **testing** (e2e-testing-patterns)
- **devops** (docker-essentials, git-essentials)
- **orchestrator** (api-dev, backend-patterns)

#### Modern UI/UX Stack
- **ux-designer** (frontend-design-ultimate, shadcn-ui, ui-ux-pro-max)
- **testing** (e2e-testing-patterns)
- **docs** (mermaid-architect, diagram)
- **devops** (vercel)

## Spawn Decision Matrix

| Task Type | Spawn? | Agent | Rationale |
|-----------|--------|-------|-----------|
| **📋 Project Assessment** |
| Full assessment | ✅ | code-review | Comprehensive audit |
| Bug analysis | ✅ | code-review | Issue detection |
| Performance review | ✅ | orchestrator | Profiling analysis |
| Security audit | ✅ | code-review | Vulnerability scan |
| Code quality review | ✅ | code-review | Quality metrics |
| UI/UX review | ✅ | ux-designer | Usability audit |
| Test coverage review | ✅ | testing | Coverage analysis |
| Documentation review | ✅ | docs | Doc assessment |
| **🆕 New Development** |
| Quick bug fix | ❌ | Do inline | < 5 min work |
| New feature | ✅ | orchestrator | Needs planning |
| New project | ✅ | orchestrator | Full setup needed |
| Simple PR | ❌ | Do inline | Straightforward |
| **🔧 Existing Codebase** |
| Complex bug fix | ✅ | orchestrator | Needs diagnosis |
| Performance issue | ✅ | orchestrator | Profiling + fix |
| Security fix | ✅ | code-review | Deep analysis |
| Code quality fix | ✅ | code-review | Refactoring |
| Tech debt | ✅ | code-review | Cleanup |
| Add feature | ✅ | orchestrator | Integration work |
| Complex refactor | ✅ | orchestrator | Coordination needed |
| **🧪 Testing & Docs** |
| Write tests | ✅ | testing | Specialized skill |
| Add test coverage | ✅ | testing | Existing code |
| Update docs | ✅ | docs | Focused output |
| API documentation | ✅ | docs | Technical writing |
| **🎨 UI/UX** |
| UI polish | ✅ | ux-designer | Design expertise |
| New UI screens | ✅ | ux-designer | User-first design |
| Design system | ✅ | ux-designer | Consistency |
| Fix UI bugs | ✅ | ux-designer | Visual issues |
| **🔀 Git Operations** |
| Complex merge | ✅ | pr-agent | Conflict expertise |
| PR preparation | ✅ | pr-agent | Clean history |
| Branch cleanup | ✅ | pr-agent | Git hygiene |
| **🔧 DevOps** |
| CI/CD setup | ✅ | devops | Pipeline creation |
| Fix build failures | ✅ | devops | CI debugging |
| Docker setup | ✅ | devops | Containerization |
| Deployment | ✅ | devops | Ship to production |
| Environment config | ✅ | devops | Dev/staging/prod |
| Monitoring setup | ✅ | devops | Sentry, logging |
| **🗄️ Database** |
| Schema design | ✅ | db-specialist | Data modeling |
| Migrations | ✅ | db-specialist | Safe schema changes |
| Query optimization | ✅ | db-specialist | Performance |
| Index strategy | ✅ | db-specialist | Speed up queries |

## Task Brief Template

When spawning sub-agents, provide complete context:

```markdown
## Task: [Clear, actionable title]

### Background
[Why this work is needed, what problem it solves]

### Requirements
1. [Specific requirement with acceptance criteria]
2. [Another requirement]
3. [Edge cases to handle]

### Technical Context
- **Project:** [path/to/project]
- **Tech Stack:** [relevant technologies]
- **Key Files:** 
  - `path/to/important/file.ts` — [what it does]
  - `path/to/schema.prisma` — [current schema]

### Out of Scope
- [What NOT to change]
- [Related work that's separate]

### Deliverables
- [ ] Working code that passes build
- [ ] Tests for new functionality
- [ ] Updated documentation
- [ ] Pushed to GitHub

### Notes
[Any additional context, constraints, or preferences]
```

## 🎯 Skill-Enhanced Quality Standards

### Every Code Delivery Must Pass Skill-Based Gates:

#### Code Quality Gate (clean-code-review)
- ✅ **clean-code-review**: All clean code principles enforced
- ✅ Single Responsibility Principle validation
- ✅ DRY principle compliance check
- ✅ Function length ≤ 20 lines verified
- ✅ Nesting depth ≤ 2 levels confirmed
- ✅ Meaningful naming conventions enforced

#### Security Gate (security-auditor)
- ✅ **security-auditor**: Comprehensive vulnerability scan
- ✅ Input validation and sanitization verified
- ✅ Authentication/authorization checks confirmed
- ✅ Secrets management validated
- ✅ OWASP compliance verified
- ✅ Dependency vulnerability scan passed

#### Performance Gate (perf-profiler)
- ✅ **perf-profiler**: Performance analysis completed
- ✅ Database query optimization verified
- ✅ Bundle size within limits
- ✅ Memory usage optimized
- ✅ API response time < 500ms
- ✅ Frontend loading performance validated

#### Testing Gate (e2e-testing-patterns)
- ✅ **e2e-testing-patterns**: Comprehensive test coverage
- ✅ Unit tests for all business logic
- ✅ Integration tests for API endpoints
- ✅ End-to-end user flow tests
- ✅ Error scenario testing
- ✅ Performance testing included

#### UI/UX Gate (frontend-design-ultimate + ui-ux-pro-max + shadcn-ui)
- ✅ **frontend-design-ultimate**: Modern design patterns applied
- ✅ **ui-ux-pro-max**: User experience validated
- ✅ **shadcn-ui**: Professional component library standards
- ✅ Mobile-responsive design verified
- ✅ Accessibility (WCAG 2.1 AA) compliance
- ✅ Dark mode support implemented

#### DevOps Gate (docker-essentials + vercel + git-essentials)
- ✅ **docker-essentials**: Containerization best practices
- ✅ **vercel**: Deployment optimization
- ✅ **git-essentials**: Professional Git workflow
- ✅ CI/CD pipeline validation
- ✅ Environment configuration verified

#### Database Gate (sql-toolkit + postgres)
- ✅ **sql-toolkit**: Query optimization validated
- ✅ **postgres**: PostgreSQL best practices applied
- ✅ Index strategy optimized
- ✅ Migration safety verified
- ✅ Data integrity constraints enforced

### Skill-Enhanced Development Standards

#### API Development (api-dev + backend-patterns)
- ✅ **api-dev**: Professional API design patterns
- ✅ **backend-patterns**: Proven architectural patterns
- ✅ RESTful API conventions followed
- ✅ Error handling standardized
- ✅ Rate limiting implemented
- ✅ API documentation generated

#### Debugging Standards (debug-pro)
- ✅ **debug-pro**: Systematic debugging approach
- ✅ Comprehensive logging implemented
- ✅ Error tracking configured
- ✅ Debug information available
- ✅ Issue reproduction documented

## 🏗️ Code Quality Requirements

### Clean Code Principles
```
✓ Single Responsibility — One function, one job
✓ DRY — Don't repeat code, extract utilities
✓ KISS — Simple > clever
✓ YAGNI — Don't build for hypothetical futures
✓ Meaningful names — getAvailableSlots() not getData()
```

### Code Smells to Reject
```
❌ Functions > 50 lines
❌ Files > 300 lines  
❌ Nesting > 3 levels
❌ Magic numbers/strings
❌ Copy-pasted code
❌ console.log left in
❌ any types in TypeScript
❌ Commented-out code
```

## ⚡ Performance Requirements

### Response Time Targets
| Operation | Target | Maximum |
|-----------|--------|---------|
| Page load | < 1s | 2s |
| API response | < 200ms | 500ms |
| Database query | < 50ms | 100ms |
| Bulk operation | < 2s | 5s |

### Performance Checklist
```
□ Database queries use indexes
□ No N+1 query patterns
□ Pagination for large lists
□ Images optimized (WebP, lazy load)
□ Components don't re-render unnecessarily
□ Bundle size < 200KB initial JS
```

### Optimization Patterns
```typescript
// ✅ Eager loading (avoid N+1)
prisma.appointment.findMany({
  include: { service: true, staff: true }
})

// ✅ Memoization
const sorted = useMemo(() => items.sort(...), [items])

// ✅ Debounce user input
const search = useDebouncedCallback(term => fetch(...), 300)

// ✅ Pagination
prisma.appointment.findMany({ take: 50, skip: page * 50 })
```

## 🔒 Security Requirements

### Authentication & Authorization
```
□ All protected routes verify authentication
□ Admin endpoints check admin role
□ Users can only access own data (no IDOR)
□ Rate limiting on auth endpoints
□ Strong password requirements
```

### Input Validation (Defense in Depth)
```
Layer 1: Client-side (UX only, not security)
Layer 2: API validation (zod schemas)
Layer 3: Database constraints
```

### Security Checklist
```
□ All inputs validated and sanitized
□ No SQL injection (Prisma handles this)
□ No XSS (React escapes by default)
□ Secrets not in code or client bundle
□ Error messages don't leak internals
□ HTTPS enforced in production
□ npm audit passes
```

### Security Red Flags 🚩
```
❌ eval() with user input
❌ dangerouslySetInnerHTML without sanitization
❌ User ID from client instead of session
❌ Missing role checks on destructive operations
❌ Secrets logged or in error messages
❌ CORS set to "*"
```

## Pre-Delivery Verification

Before marking ANY task complete, verify:

```bash
# 1. Build passes
npm run build

# 2. Lint passes
npm run lint

# 3. No security vulnerabilities
npm audit

# 4. Tests pass (if applicable)
npm test

# 5. No secrets in code
git diff --staged | grep -iE "(password|secret|api_key|token)"
```

## Agent Communication

### When Receiving Results:
1. Review the deliverables
2. Verify against requirements
3. Test critical paths
4. Report to Anirach with summary

### When Blocked:
- Identify what's blocking
- Propose alternatives
- Ask for clarification if needed
- Don't guess on ambiguous requirements

## 🔄 Common Existing Codebase Workflows

### Bug Fix Workflow
```
1. Arthur receives bug report
2. Spawn orchestrator with:
   - Project path
   - Bug description
   - Reproduction steps
3. orchestrator:
   - Analyzes codebase
   - Identifies root cause
   - Implements fix
   - Adds regression test
   - Verifies fix works
4. Deliver: Fixed code + test + explanation
```

### Performance Optimization Workflow
```
1. Arthur receives "app is slow" complaint
2. Spawn orchestrator with:
   - Project path
   - Which part is slow
   - Current metrics (if known)
3. orchestrator:
   - Profiles the application
   - Identifies bottlenecks
   - Implements optimizations
   - Benchmarks improvements
4. Deliver: Optimized code + before/after metrics
```

### Code Quality Improvement Workflow
```
1. Arthur receives "clean up codebase" request
2. Spawn code-review with:
   - Project path
   - Areas of concern
3. code-review:
   - Analyzes code quality
   - Identifies issues (smells, debt, violations)
   - Prioritizes fixes
   - Implements improvements
   - Documents changes
4. Deliver: Cleaned code + quality report
```

### UI/UX Polish Workflow
```
1. Arthur receives "make it look better" request
2. Spawn ux-designer with:
   - Project path
   - Current screenshots (if possible)
   - Target aesthetic
3. ux-designer:
   - Audits existing UI
   - Creates improvement plan
   - Implements changes
   - Ensures consistency
4. Deliver: Polished UI + design notes
```

### Add Feature to Existing App Workflow
```
1. Arthur receives feature request
2. Spawn orchestrator with:
   - Project path
   - Feature requirements
   - Integration points
3. orchestrator:
   - Understands existing architecture
   - Plans integration approach
   - Implements feature
   - Adds tests
   - Updates docs
4. Deliver: New feature integrated + tests + docs
```

## Memory & Learning

### Daily Log
After completing coding tasks, note in `memory/YYYY-MM-DD.md`:
- What was built
- Key decisions made
- Lessons learned
- Technical debt incurred

### Obsidian Workspace
All Coding Team work is documented in Obsidian:

**Location:** `/home/clawdbot/obsidian-vault/Agents/Coding/`

```
Coding/
├── Projects/           # Active code projects
│   └── [Project-Name].md
├── Notes/              # Tech decisions, patterns, learnings
│   ├── Decisions/      # Architecture Decision Records
│   └── Patterns/       # Reusable patterns
└── Templates/          # Project templates
    └── Project Memory.md
```

**For each project, create:**
1. `Projects/[Project-Name].md` — Overview, progress, links
2. `Projects/[Project-Name]-Memory.md` — Decisions, patterns, tech debt

**Tags:** `#coding`, `#project`, `#decision`, `#pattern`, `#techdebt`

**Benefits:**
- 🔗 Link between projects
- 🔍 Search across all projects
- 📊 Track patterns & decisions
- 🔄 Reuse solutions

---

## ✅ Quality Gates (MANDATORY - NO EXCEPTIONS)

⚠️ **CRITICAL: No work is delivered until ALL gates pass locally.**

**The agent MUST:**
1. Run `npm run build` (or equivalent) and verify it succeeds
2. Run `npm run lint` and fix any errors
3. Run tests if available
4. Only THEN commit, push, and report completion

**If build fails → FIX IT before reporting done.**
**Never tell the user "done" if build hasn't been verified.**

**Every PR must pass these gates before review:**

```
┌─────────────────────────────────────────────────────────────┐
│                    QUALITY GATE CHECKLIST                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Gate 1: BUILD                                               │
│  □ npm run build (or equivalent) passes                      │
│  □ No TypeScript errors                                      │
│  □ No compilation warnings                                   │
│                                                              │
│  Gate 2: LINT                                                │
│  □ npm run lint passes                                       │
│  □ No ESLint errors                                          │
│  □ Prettier formatting correct                               │
│                                                              │
│  Gate 3: TEST                                                │
│  □ npm test passes                                           │
│  □ No test regressions                                       │
│  □ New code has tests (if applicable)                        │
│                                                              │
│  Gate 4: SECURITY                                            │
│  □ npm audit --audit-level=high passes                       │
│  □ No secrets in code                                        │
│  □ No known vulnerabilities                                  │
│                                                              │
│  Gate 5: BUNDLE (for frontend)                               │
│  □ Bundle size within limits                                 │
│  □ No unexpected size increases                              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Pre-PR Command Sequence

```bash
# Run all gates before creating PR
npm run build          # Gate 1 - MUST PASS
npm run lint           # Gate 2 - MUST PASS  
npm test               # Gate 3 - MUST PASS (if tests exist)
npm audit              # Gate 4 - Check for vulnerabilities
npm run analyze        # Gate 5 (if available)
```

### 🚨 Delivery Verification Protocol

**Before reporting ANY task as complete:**

```
┌─────────────────────────────────────────────────────────────┐
│              DELIVERY CHECKLIST (MANDATORY)                  │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  □ 1. Code changes are complete                              │
│  □ 2. `npm run build` executed and PASSED ✅                 │
│  □ 3. `npm run lint` executed and PASSED ✅                  │
│  □ 4. Tests run (if applicable) and PASSED ✅                │
│  □ 5. Changes committed with descriptive message             │
│  □ 6. Changes pushed to remote                               │
│  □ 7. Deployment triggered (if auto-deploy)                  │
│  □ 8. ONLY NOW report "Task complete" to user                │
│                                                              │
│  ❌ If ANY gate fails:                                       │
│     - DO NOT report completion                               │
│     - FIX the issue first                                    │
│     - Re-run gates                                           │
│     - Only report done when ALL pass                         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**Example completion report:**
```
✅ Task complete!

Verification:
- Build: ✅ passed
- Lint: ✅ passed  
- Tests: ✅ passed (or N/A)
- Pushed: ✅ commit abc123
- Deploy: ✅ triggered

Changes: [summary]
```

### Gate Failure Protocol

```
If any gate fails:
1. DO NOT create PR
2. Fix the issue
3. Re-run failed gate
4. Continue only when all pass
```

---

## 🧠 Project Memory System

Each project maintains a memory file for smarter decisions across sessions.

### PROJECT_MEMORY.md Template

```markdown
# Project Memory: [Project Name]
**Location:** /path/to/project
**Created:** [Date]
**Last Updated:** [Date]

---

## 🏗️ Architecture Decisions

| Date | Decision | Rationale | Alternatives Considered |
|------|----------|-----------|------------------------|
| 2026-02-11 | Use Prisma ORM | Type safety, migrations | Drizzle, raw SQL |
| 2026-02-10 | Next.js App Router | Server components | Pages router |

---

## 📐 Patterns Used

### State Management
- Zustand for client state
- React Query for server state
- URL state for filters/pagination

### API Design
- REST with versioning (/api/v1/)
- Zod validation on all endpoints
- Consistent error format

### Component Patterns
- Compound components for complex UI
- Render props for flexibility
- Custom hooks for logic reuse

---

## 🐛 Known Issues

| Issue | Severity | Workaround | Ticket |
|-------|----------|------------|--------|
| Auth race condition | Medium | Retry logic added | #142 |
| Slow dashboard query | Low | Pagination added | #156 |

---

## 💳 Tech Debt Log

| Item | Impact | Effort | Priority |
|------|--------|--------|----------|
| Migrate to App Router | High | High | P2 |
| Add E2E tests | Medium | Medium | P3 |
| Refactor auth module | Low | Low | P4 |

---

## 🔧 Environment Notes

### Development
- Node 20.x required
- PostgreSQL 15 local
- Redis optional (uses memory fallback)

### Secrets Required
- DATABASE_URL
- NEXTAUTH_SECRET
- STRIPE_SECRET_KEY

---

## 📝 Conventions

### File Naming
- Components: PascalCase.tsx
- Utilities: camelCase.ts
- Constants: SCREAMING_SNAKE_CASE

### Code Style
- Prefer named exports
- Max file length: 300 lines
- Max function length: 50 lines

---

## 📊 Performance Baselines

| Metric | Baseline | Target | Current |
|--------|----------|--------|---------|
| LCP | 2.5s | <2.0s | 1.8s |
| Bundle Size | 250KB | <200KB | 185KB |
| API p95 | 500ms | <200ms | 180ms |

---

## 🔄 Recent Changes

### 2026-02-11
- Added payment integration
- Optimized dashboard queries
- Fixed auth race condition

### 2026-02-10
- Initial project setup
- Basic CRUD operations
- Auth implementation
```

### Memory Usage

**When starting work on a project:**
1. Read PROJECT_MEMORY.md if it exists
2. Follow established patterns
3. Respect architecture decisions
4. Update memory after significant changes

**When to update:**
- New architecture decisions
- New patterns introduced
- Issues discovered
- Tech debt added/resolved
- Performance changes

---

## 🚀 Deployment Capabilities

### Supported Platforms

| Platform | Command | Config File |
|----------|---------|-------------|
| **Vercel** | `vercel deploy` | vercel.json |
| **Railway** | `railway up` | railway.toml |
| **Fly.io** | `fly deploy` | fly.toml |
| **Docker** | `docker build & push` | Dockerfile |
| **AWS** | `aws deploy` | serverless.yml |

### Deployment Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT WORKFLOW                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. All quality gates pass                                   │
│         │                                                    │
│         ▼                                                    │
│  2. Create deployment branch                                 │
│     git checkout -b deploy/v1.2.3                           │
│         │                                                    │
│         ▼                                                    │
│  3. Build production bundle                                  │
│     npm run build                                           │
│         │                                                    │
│         ▼                                                    │
│  4. Deploy to staging first                                  │
│     vercel deploy --env=staging                             │
│         │                                                    │
│         ▼                                                    │
│  5. Run smoke tests on staging                               │
│         │                                                    │
│         ▼                                                    │
│  6. Deploy to production                                     │
│     vercel deploy --prod                                    │
│         │                                                    │
│         ▼                                                    │
│  7. Verify production                                        │
│         │                                                    │
│         ▼                                                    │
│  8. Tag release                                              │
│     git tag v1.2.3                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Environment Configuration

```markdown
## Environment Setup

### Development (.env.local)
DATABASE_URL=postgresql://localhost:5432/myapp_dev
NEXTAUTH_URL=http://localhost:3000

### Staging (.env.staging)
DATABASE_URL=postgresql://staging-db/myapp_staging
NEXTAUTH_URL=https://staging.myapp.com

### Production (.env.production)
DATABASE_URL=postgresql://prod-db/myapp_prod
NEXTAUTH_URL=https://myapp.com
```

---

## 📊 Performance Benchmarks

### Required Metrics

Every performance-related change must include before/after metrics:

```markdown
## Performance Change Report

### Summary
Optimized dashboard query performance

### Metrics

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Query Time (p50) | 450ms | 120ms | -73% |
| Query Time (p95) | 890ms | 250ms | -72% |
| Memory Usage | 150MB | 145MB | -3% |

### Test Conditions
- Dataset: 100,000 records
- Concurrent users: 50
- Test duration: 5 minutes

### How to Reproduce
```bash
npm run benchmark:dashboard
```
```

### Benchmark Commands

```bash
# Bundle analysis
npm run analyze

# Lighthouse audit
npx lighthouse https://myapp.com --output=json

# Load testing
npx autocannon -c 100 -d 30 https://api.myapp.com/endpoint

# Database query analysis
EXPLAIN ANALYZE SELECT * FROM ...
```

### Performance Budgets

```javascript
// next.config.js or similar
{
  performance: {
    budgets: [
      { type: 'initial-js', maxSize: '200kb' },
      { type: 'initial-css', maxSize: '50kb' },
      { type: 'image', maxSize: '500kb' },
      { type: 'total', maxSize: '1mb' }
    ]
  }
}
```

---

## 📋 Project Templates

Quick-start templates for common stacks.

### Next.js + Prisma + Auth

```bash
# Initialize
npx create-next-app@latest my-app --typescript --tailwind --app

# Add dependencies
npm install prisma @prisma/client next-auth @auth/prisma-adapter
npm install -D @types/node

# Initialize Prisma
npx prisma init

# Recommended structure
my-app/
├── src/
│   ├── app/
│   │   ├── (auth)/
│   │   │   ├── login/
│   │   │   └── register/
│   │   ├── (dashboard)/
│   │   │   └── dashboard/
│   │   ├── api/
│   │   │   ├── auth/[...nextauth]/
│   │   │   └── v1/
│   │   ├── layout.tsx
│   │   └── page.tsx
│   ├── components/
│   │   ├── ui/           # shadcn components
│   │   └── shared/       # app components
│   ├── lib/
│   │   ├── auth.ts
│   │   ├── db.ts
│   │   └── utils.ts
│   └── types/
├── prisma/
│   └── schema.prisma
├── public/
└── tests/
```

### Express API + PostgreSQL

```bash
# Initialize
mkdir my-api && cd my-api
npm init -y
npm install express cors helmet prisma @prisma/client zod
npm install -D typescript @types/express @types/node ts-node nodemon

# Recommended structure
my-api/
├── src/
│   ├── routes/
│   │   ├── index.ts
│   │   ├── users.ts
│   │   └── products.ts
│   ├── middleware/
│   │   ├── auth.ts
│   │   ├── error.ts
│   │   └── validate.ts
│   ├── services/
│   ├── utils/
│   ├── types/
│   └── index.ts
├── prisma/
├── tests/
└── Dockerfile
```

### React + Vite + Tailwind

```bash
# Initialize
npm create vite@latest my-app -- --template react-ts
cd my-app
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p

# Add shadcn/ui
npx shadcn-ui@latest init

# Recommended structure
my-app/
├── src/
│   ├── components/
│   │   ├── ui/
│   │   └── features/
│   ├── hooks/
│   ├── lib/
│   ├── pages/
│   ├── stores/
│   ├── types/
│   └── main.tsx
├── public/
└── tests/
```

---

## ⏪ Rollback Procedures

### Git Rollback

```bash
# Revert last commit (keeps in history)
git revert HEAD

# Revert specific commit
git revert <commit-hash>

# Hard reset (dangerous - loses commits)
git reset --hard <commit-hash>
git push --force-with-lease
```

### Deployment Rollback

```bash
# Vercel - instant rollback to previous
vercel rollback

# Railway - redeploy previous
railway redeploy <deployment-id>

# Fly.io - rollback
fly releases rollback <version>

# Docker - use previous tag
docker pull myapp:v1.2.2
docker-compose up -d
```

### Rollback Decision Tree

```
┌─────────────────────────────────────────────────────────────┐
│                 ROLLBACK DECISION TREE                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Production issue detected                                   │
│         │                                                    │
│         ▼                                                    │
│  Is it critical? (data loss, security, total outage)         │
│         │                                                    │
│    YES ─┴─ NO                                                │
│     │      │                                                 │
│     ▼      ▼                                                 │
│  IMMEDIATE   Can it wait for hotfix?                         │
│  ROLLBACK         │                                          │
│     │        YES ─┴─ NO                                      │
│     │         │      │                                       │
│     │         ▼      ▼                                       │
│     │     HOTFIX   ROLLBACK                                  │
│     │     DEPLOY   + HOTFIX                                  │
│     │                                                        │
│     ▼                                                        │
│  1. Rollback deployment                                      │
│  2. Notify team                                              │
│  3. Document incident                                        │
│  4. Root cause analysis                                      │
│  5. Fix & re-deploy                                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Rollback Checklist

```markdown
## Rollback Executed

- [ ] Previous version deployed
- [ ] Verified functionality restored
- [ ] Database state checked (migrations?)
- [ ] Cache cleared if needed
- [ ] Team notified
- [ ] Incident documented
- [ ] Root cause identified
- [ ] Fix PR created
```

---

## 📈 Monitoring Integration

### Supported Tools

| Tool | Purpose | Setup |
|------|---------|-------|
| **Sentry** | Error tracking | `npm install @sentry/nextjs` |
| **LogRocket** | Session replay | `npm install logrocket` |
| **Datadog** | APM & logging | `npm install dd-trace` |
| **Vercel Analytics** | Web vitals | Built-in |
| **PostHog** | Product analytics | `npm install posthog-js` |

### Sentry Setup

```typescript
// sentry.client.config.ts
import * as Sentry from "@sentry/nextjs";

Sentry.init({
  dsn: process.env.SENTRY_DSN,
  environment: process.env.NODE_ENV,
  tracesSampleRate: 1.0,
  replaysSessionSampleRate: 0.1,
  replaysOnErrorSampleRate: 1.0,
});
```

### Error Tracking Pattern

```typescript
// lib/errors.ts
import * as Sentry from "@sentry/nextjs";

export function captureError(error: Error, context?: Record<string, any>) {
  console.error(error);
  
  Sentry.captureException(error, {
    extra: context,
  });
}

// Usage
try {
  await riskyOperation();
} catch (error) {
  captureError(error, { userId, operation: 'payment' });
  throw error;
}
```

### Logging Standards

```typescript
// lib/logger.ts
type LogLevel = 'debug' | 'info' | 'warn' | 'error';

interface LogEntry {
  level: LogLevel;
  message: string;
  timestamp: string;
  context?: Record<string, any>;
}

export const logger = {
  info: (message: string, context?: Record<string, any>) => {
    log('info', message, context);
  },
  warn: (message: string, context?: Record<string, any>) => {
    log('warn', message, context);
  },
  error: (message: string, context?: Record<string, any>) => {
    log('error', message, context);
  },
};

// What to log:
// ✅ API requests (method, path, duration, status)
// ✅ Authentication events (login, logout, failures)
// ✅ Business events (orders, payments, signups)
// ✅ Errors with context
// ❌ Sensitive data (passwords, tokens, PII)
// ❌ Every function call (too noisy)
```

### Health Check Endpoint

```typescript
// app/api/health/route.ts
export async function GET() {
  const health = {
    status: 'ok',
    timestamp: new Date().toISOString(),
    checks: {
      database: await checkDatabase(),
      redis: await checkRedis(),
      external: await checkExternalServices(),
    },
  };

  const allHealthy = Object.values(health.checks).every(c => c.status === 'ok');
  
  return Response.json(health, { 
    status: allHealthy ? 200 : 503 
  });
}
```

### Alerting Rules

```yaml
# Example alerting config
alerts:
  - name: High Error Rate
    condition: error_rate > 5%
    duration: 5m
    severity: critical
    notify: [slack, pagerduty]

  - name: Slow Response Time
    condition: p95_latency > 2000ms
    duration: 10m
    severity: warning
    notify: [slack]

  - name: Database Connection Failures
    condition: db_connection_errors > 0
    duration: 1m
    severity: critical
    notify: [slack, pagerduty, sms]
```

---

*"Programs must be written for people to read, and only incidentally for machines to execute."* — Harold Abelson
