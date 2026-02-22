# Research Workflow — Parallel Sub-Agent Strategy

## 🎯 When to Use Parallel Research

Use parallel sub-agents when:
- Topic naturally splits into 2-4 sub-topics
- Research would take >10 minutes sequential
- Different aspects need deep focus
- User needs comprehensive coverage

## 🚀 Standard Parallel Research Flow

```
User Request
     │
     ▼
┌─────────────────────────────────────┐
│  Arthur (Main) - Research Planner   │
│  • Analyze request                  │
│  • Split into 2-4 research streams  │
│  • Spawn parallel agents            │
│  • Monitor progress                 │
│  • Synthesize final report          │
└─────────────────────────────────────┘
     │
     ├──────────────┬──────────────┬──────────────┐
     ▼              ▼              ▼              ▼
┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐
│ Agent 1 │   │ Agent 2 │   │ Agent 3 │   │ Agent 4 │
│ Stream A│   │ Stream B│   │ Stream C│   │ Stream D│
│ (Opus)  │   │ (Sonnet)│   │ (Sonnet)│   │ (Opus)  │
└─────────┘   └─────────┘   └─────────┘   └─────────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │  Synthesized Report │
              │  • Combined findings│
              │  • Cross-references │
              │  • Recommendations  │
              └─────────────────────┘
```

## 📋 Research Task Template

When spawning research sub-agents, use this template:

```markdown
## Research Task: [Specific Topic]

### Scope
[What to research, boundaries]

### Key Questions
1. [Question 1]
2. [Question 2]
3. [Question 3]

### Sources to Check
- Academic: Semantic Scholar, PubMed, arXiv
- News: Recent articles (last 6 months)
- Industry: Company announcements, reports

### Output Format
- Summary (2-3 paragraphs)
- Key Findings (bullet points)
- Notable Sources (with URLs)
- Gaps/Uncertainties

### Deliverable
Save findings to: [workspace path]
```

## 🔧 Implementation

### Spawning Parallel Agents

```javascript
// Arthur spawns multiple agents simultaneously
sessions_spawn({ agentId: "research-lead", task: "Research Stream A..." })
sessions_spawn({ agentId: "research-lead", task: "Research Stream B..." })
sessions_spawn({ agentId: "research-lead", task: "Research Stream C..." })
// All run in parallel, report back when done
```

### Model Strategy

| Research Type | Model | Rationale |
|---------------|-------|-----------|
| Literature synthesis | Opus | Complex reasoning |
| Data gathering | Sonnet | Fast, cost-effective |
| Trend analysis | Opus | Pattern recognition |
| Fact compilation | Sonnet | Straightforward task |

### Timeout Settings

- Quick research: 10 minutes (600s)
- Standard research: 30 minutes (1800s)
- Deep research: 60 minutes (3600s)

## 📊 Example: Longevity Research

**User Request:** "Research latest longevity interventions"

**Parallel Streams:**

| Stream | Agent | Focus | Model |
|--------|-------|-------|-------|
| A | research-lead | Caloric restriction & fasting | Opus |
| B | research-lead | Senolytics & senomorphics | Sonnet |
| C | research-lead | Epigenetic reprogramming | Opus |
| D | research-lead | Exercise & lifestyle | Sonnet |

**Synthesis:** Main agent combines all findings into unified report

## 📁 Output Structure

```
research/[topic]/
├── stream-a-findings.md
├── stream-b-findings.md
├── stream-c-findings.md
├── stream-d-findings.md
├── synthesis.md
└── [Topic]-Report-FINAL.docx  → Google Drive
```

## ⚡ Quick Commands

For Anirach to trigger parallel research:

```
"Deep research on [topic]"
"Comprehensive analysis of [topic]"
"Research [topic] - cover [aspect1], [aspect2], [aspect3]"
```

Arthur will automatically:
1. Split into appropriate streams
2. Spawn parallel agents
3. Monitor completion
4. Synthesize and deliver report

## 🎯 Quality Gates

Each research stream must:
- ✅ Cite specific sources (URLs)
- ✅ Include publication dates
- ✅ Note confidence levels
- ✅ Flag contradictions
- ✅ Identify gaps

Final synthesis must:
- ✅ Cross-reference streams
- ✅ Resolve contradictions
- ✅ Provide actionable insights
- ✅ Include full source list

---

*Implemented: February 8, 2026*
